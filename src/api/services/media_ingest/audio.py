"""Audio ingest policy: probe parsing, container cleanup and waveform peaks (S2a).

Everything here is pure (no I/O, no subprocess): ``MediaIngestService.prepare_audio`` runs
ffprobe / ffmpeg and feeds their output through these functions, so the policy is testable
without either binary.

What is stored is the upload's audio in its original container: **stream copy for MP3 (frames
only), M4A and WAV; lossless re-encode for FLAC (bit-identical samples)**. Tags, chapters,
attached pictures and bytes outside MP3 frames are removed (Terms 7.5: embedded metadata is
removed). Nothing in this module (or its callers) may log a tag value or a file name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from src.api.services.media_ingest.errors import (
    AudioCode,
    InvalidAudioError,
    UnsupportedAudioError,
)
from src.core.enums import MediaFormat

__all__ = [
    "AUDIO_MAX_SECONDS",
    "AUDIO_MIN_SECONDS",
    "MAX_PEAKS_SAMPLES",
    "PEAKS_BUCKETS",
    "PEAKS_BYTES",
    "PEAKS_SAMPLE_RATE_HZ",
    "PEAKS_VERSION",
    "AudioProbe",
    "compute_peaks",
    "parse_audio_probe",
    "remux_arguments",
]

AUDIO_MIN_SECONDS: Final = 5
AUDIO_MAX_SECONDS: Final = 900
"""Duration limits: module constants, not settings - nothing needs to tune them per deployment."""

PEAKS_VERSION: Final = 1
PEAKS_BUCKETS: Final = 1024
PEAKS_BYTES: Final = PEAKS_BUCKETS * 2
"""Interleaved ``int8`` ``[min0, max0, min1, max1, ...]``."""
PEAKS_SAMPLE_RATE_HZ: Final = 11025
"""The mono rate the peaks are decoded at (``-ac 1 -ar 11025 -f s16le``)."""
MAX_PEAKS_SAMPLES: Final = (AUDIO_MAX_SECONDS + 1) * PEAKS_SAMPLE_RATE_HZ
"""Hard upper bound for the peaks decoder, including its one-second duration slack."""

_MAX_CHANNELS: Final = 8
"""The ``library_audio_profiles.channels`` CHECK is ``BETWEEN 1 AND 8``."""

# ffprobe ``format_name`` -> the stored format. The ISO-BMFF family name is one comma list.
_FORMAT_BY_PROBE_NAME: Final[dict[str, MediaFormat]] = {
    "mp3": MediaFormat.MP3,
    "mov,mp4,m4a,3gp,3g2,mj2": MediaFormat.M4A,
    "wav": MediaFormat.WAV,
    "flac": MediaFormat.FLAC,
}

# The one codec each container may carry. Anything else is "valid media outside the policy":
# the browser would not play it (or, for a lossy codec in an unexpected box, we do not vouch).
_CODECS_BY_FORMAT: Final[dict[MediaFormat, frozenset[str]]] = {
    MediaFormat.MP3: frozenset({"mp3"}),
    MediaFormat.M4A: frozenset({"aac"}),
    MediaFormat.WAV: frozenset({"pcm_s16le", "pcm_s24le"}),
    MediaFormat.FLAC: frozenset({"flac"}),
}
_FLAC_SAMPLE_FORMATS: Final = frozenset({"s16", "s32"})
"""Sample formats the bundled FLAC encoder can preserve without an implicit conversion."""

# ffmpeg muxer names. ``ipod`` is the M4A muxer (``M4A `` brand); ``mp4`` would write ``isom``.
_MUXER_BY_FORMAT: Final[dict[MediaFormat, str]] = {
    MediaFormat.MP3: "mp3",
    MediaFormat.M4A: "ipod",
    MediaFormat.WAV: "wav",
    MediaFormat.FLAC: "flac",
}

# Per-container muxer options of the remux (everything common is in ``remux_arguments``).
_MUXER_OPTIONS: Final[dict[MediaFormat, tuple[str, ...]]] = {
    # No ID3v1, no ID3v2 (tags and cover art live there); a Xing header keeps duration and
    # seeking exact in browsers.
    MediaFormat.MP3: ("-write_id3v1", "0", "-id3v2_version", "0", "-write_xing", "1"),
    MediaFormat.M4A: ("-movflags", "+faststart"),
    MediaFormat.WAV: ("-rf64", "auto"),
    MediaFormat.FLAC: (),
}

PROBE_SHOW_ENTRIES: Final = (
    "stream=index,codec_type,codec_name,sample_fmt,sample_rate,channels,bit_rate,duration"
    ":stream_disposition=attached_pic"
    ":format=format_name,duration,bit_rate"
)
"""The ``-show_entries`` value of the probe: only what ``parse_audio_probe`` reads (no tags).

``disposition`` is a section of its own: listing it inside ``stream=`` prints nothing, which would
make an attached cover picture look like a video stream."""


@dataclass(frozen=True, slots=True)
class AudioProbe:
    """The probed facts of an accepted audio file (the first audio stream's)."""

    format: MediaFormat
    codec: str
    sample_fmt: str | None
    sample_rate_hz: int
    channels: int
    bit_rate: int | None
    duration_ms: int


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _positive_int(value: object) -> int | None:
    number = _positive_float(value)
    return None if number is None else int(number)


def _is_attached_picture(stream: dict[str, object]) -> bool:
    disposition = stream.get("disposition")
    return isinstance(disposition, dict) and bool(disposition.get("attached_pic"))


def parse_audio_probe(decoded: object) -> AudioProbe:
    """Apply the audio policy to a decoded ``ffprobe -of json`` document.

    The checks run in this order, so the code a client sees is the first that applies: format,
    video stream, audio stream, codec, duration.

    Raises:
        UnsupportedAudioError: ``audio_format_not_supported``, ``audio_has_video`` or
            ``audio_codec_not_supported``.
        InvalidAudioError: ``invalid_audio`` (no usable audio stream, unreadable probe) or
            ``audio_duration_out_of_range`` (``detail`` carries the limits in seconds).
    """
    if not isinstance(decoded, dict):
        raise InvalidAudioError(AudioCode.INVALID, "audio is not decodable")
    container = decoded.get("format")
    streams = decoded.get("streams")
    if not isinstance(container, dict) or not isinstance(streams, list):
        raise InvalidAudioError(AudioCode.INVALID, "audio is not decodable")

    media_format = _FORMAT_BY_PROBE_NAME.get(str(container.get("format_name", "")))
    if media_format is None:
        raise UnsupportedAudioError(AudioCode.FORMAT_NOT_SUPPORTED, "audio format is not supported")

    typed = [stream for stream in streams if isinstance(stream, dict)]
    if any(s.get("codec_type") == "video" and not _is_attached_picture(s) for s in typed):
        raise UnsupportedAudioError(AudioCode.HAS_VIDEO, "the file has a video stream")

    audio = next((s for s in typed if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise InvalidAudioError(AudioCode.INVALID, "the file has no audio stream")

    codec = audio.get("codec_name")
    sample_fmt = audio.get("sample_fmt")
    channels = _positive_int(audio.get("channels"))
    sample_rate = _positive_int(audio.get("sample_rate"))
    if (
        not isinstance(codec, str)
        or codec not in _CODECS_BY_FORMAT[media_format]
        or channels is None
        or channels > _MAX_CHANNELS
    ):
        raise UnsupportedAudioError(
            AudioCode.CODEC_NOT_SUPPORTED, "the audio codec or channel layout is not supported"
        )
    if sample_rate is None:
        raise InvalidAudioError(AudioCode.INVALID, "the audio stream has no sample rate")
    if media_format is MediaFormat.FLAC and sample_fmt not in _FLAC_SAMPLE_FORMATS:
        raise UnsupportedAudioError(
            AudioCode.CODEC_NOT_SUPPORTED,
            "the FLAC sample format cannot be preserved by the lossless encoder",
        )

    # The audio stream's own duration first, then the container's. Valid only if one is positive.
    seconds = _positive_float(audio.get("duration")) or _positive_float(container.get("duration"))
    if seconds is None:
        raise InvalidAudioError(AudioCode.INVALID, "the audio has no usable duration")
    if not AUDIO_MIN_SECONDS <= seconds <= AUDIO_MAX_SECONDS:
        raise InvalidAudioError(
            AudioCode.DURATION_OUT_OF_RANGE,
            "the audio duration is outside the accepted range",
            {"min_seconds": AUDIO_MIN_SECONDS, "max_seconds": AUDIO_MAX_SECONDS},
        )

    return AudioProbe(
        format=media_format,
        codec=codec,
        sample_fmt=sample_fmt if isinstance(sample_fmt, str) else None,
        sample_rate_hz=sample_rate,
        channels=channels,
        bit_rate=_positive_int(audio.get("bit_rate")) or _positive_int(container.get("bit_rate")),
        duration_ms=max(1, round(seconds * 1000)),
    )


def remux_arguments(media_format: MediaFormat, *, sample_fmt: str | None = None) -> list[str]:
    """The ffmpeg options (between ``-i <in>`` and the output path) for audio cleanup.

    Maps the first audio stream only (so attached pictures, extra streams and data tracks are
    dropped). MP3, M4A and WAV are stream-copied. FLAC is losslessly encoded at the same decoder
    sample format; ``parse_audio_probe`` refuses formats the encoder cannot preserve. Every
    metadata scope and chapters are cleared. ``bitexact`` prevents a new encoder tag.
    """
    if media_format is MediaFormat.FLAC:
        if sample_fmt not in _FLAC_SAMPLE_FORMATS:
            raise UnsupportedAudioError(
                AudioCode.CODEC_NOT_SUPPORTED,
                "the FLAC sample format cannot be preserved by the lossless encoder",
            )
        return [
            "-map",
            "0:a:0",
            "-c:a",
            "flac",
            "-sample_fmt",
            sample_fmt,
            "-compression_level",
            "5",
            "-map_metadata",
            "-1",
            "-map_metadata:s:a",
            "-1",
            "-map_chapters",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags:a",
            "+bitexact",
            "-f",
            "flac",
        ]
    return [
        "-map",
        "0:a:0",
        "-c:a",
        "copy",
        "-map_metadata",
        "-1",
        "-map_metadata:s:a",
        "-1",
        "-map_chapters",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:a",
        "+bitexact",
        *_MUXER_OPTIONS[media_format],
        "-f",
        _MUXER_BY_FORMAT[media_format],
    ]


def compute_peaks(pcm: bytes) -> bytes:
    """Reduce mono signed 16-bit little-endian PCM to ``PEAKS_BUCKETS`` min/max pairs.

    The samples are split into exactly ``PEAKS_BUCKETS`` near-equal buckets; per bucket the min
    and max are shifted right by 8 and clipped to ``[-128, 127]``. The result is ``int8``
    interleaved ``[min0, max0, min1, max1, ...]`` (``PEAKS_BYTES`` bytes). Deterministic.

    Raises:
        InvalidAudioError: Fewer samples than buckets (an empty or near-empty decode).
    """
    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2")
    if samples.size > MAX_PEAKS_SAMPLES:
        raise InvalidAudioError(AudioCode.INVALID, "the audio decodes longer than allowed")
    if samples.size < PEAKS_BUCKETS:
        raise InvalidAudioError(AudioCode.INVALID, "the audio decoded to too few samples")
    peaks = np.empty(PEAKS_BUCKETS * 2, dtype=np.int8)
    for index, bucket in enumerate(np.array_split(samples, PEAKS_BUCKETS)):
        low = int(bucket.min()) >> 8
        high = int(bucket.max()) >> 8
        peaks[2 * index] = max(-128, min(127, low))
        peaks[2 * index + 1] = max(-128, min(127, high))
    return peaks.tobytes()
