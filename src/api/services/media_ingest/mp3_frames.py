"""Keep MPEG Layer III frames while dropping bytes between or around them.

MP3 demuxers may attach arbitrary data between frames to an adjacent packet. A stream-copy
remux therefore cannot guarantee that only audio frames remain. This small scanner recognizes
MPEG-1, MPEG-2 and MPEG-2.5 Layer III frame boundaries without decoding the audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from src.api.services.media_ingest.errors import AudioCode, InvalidAudioError

__all__ = ["FrameScan", "audible_frames", "frames_only", "has_xing_header"]

_BITRATES_MPEG1: Final = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0)
_BITRATES_MPEG2: Final = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0)
_SAMPLE_RATES: Final = (44100, 48000, 32000)


@dataclass(frozen=True, slots=True)
class FrameScan:
    """The concatenated MPEG frames and number of bytes discarded by the scan."""

    frames: bytes
    dropped_bytes: int


def _frame_length(data: bytes, offset: int) -> int | None:
    """Return a valid Layer III frame length at ``offset``, or ``None``."""
    if offset < 0 or offset + 4 > len(data):
        return None
    header = int.from_bytes(data[offset : offset + 4], "big")
    if header >> 21 != 0x7FF:  # 11 sync bits
        return None

    version = (header >> 19) & 0b11
    layer = (header >> 17) & 0b11
    bitrate_index = (header >> 12) & 0b1111
    sample_rate_index = (header >> 10) & 0b11
    padding = (header >> 9) & 1
    if version == 0b01 or layer != 0b01 or bitrate_index in (0, 0b1111):
        return None
    if sample_rate_index == 0b11:
        return None

    if version == 0b11:
        bitrate_kbps = _BITRATES_MPEG1[bitrate_index]
        sample_rate = _SAMPLE_RATES[sample_rate_index]
        coefficient = 144
    else:
        bitrate_kbps = _BITRATES_MPEG2[bitrate_index]
        sample_rate = _SAMPLE_RATES[sample_rate_index] // (2 if version == 0b10 else 4)
        coefficient = 72
    if not bitrate_kbps:
        return None
    length = coefficient * bitrate_kbps * 1000 // sample_rate + padding
    return length if offset + length <= len(data) else None


def has_xing_header(data: bytes) -> bool:
    """Return whether the first MPEG frame contains a Xing/Info seek header.

    MP3 muxers rewrite this metadata-only header frame during a stream-copy remux. It
    carries gapless/seek information rather than audible samples, so decoded PCM checks
    should omit the first frame from both sides if either side has one.
    """
    length = _frame_length(data, 0)
    if length is None:
        return False
    header = int.from_bytes(data[:4], "big")
    version = (header >> 19) & 0b11
    mono = ((header >> 6) & 0b11) == 0b11
    crc_bytes = 0 if (header >> 16) & 1 else 2
    side_info_bytes = (17 if mono else 32) if version == 0b11 else (9 if mono else 17)
    marker_offset = 4 + crc_bytes + side_info_bytes
    return length >= marker_offset + 4 and data[marker_offset : marker_offset + 4] in (
        b"Xing",
        b"Info",
    )


def _without_first_frame(data: bytes) -> bytes:
    """Return the byte stream after its first complete MPEG Layer III frame."""
    length = _frame_length(data, 0)
    if length is None or length >= len(data):
        raise InvalidAudioError(AudioCode.INVALID, "audio needs at least two MPEG audio frames")
    return data[length:]


def audible_frames(frames: bytes) -> bytes:
    """Return the stream's audio frames, dropping only its own Xing/Info frame.

    A leading Xing/Info frame carries no samples; the MP3 demuxer uses its LAME gapless
    values to trim encoder delay and padding. A remux writes a new header whose trim values
    can differ from the source's. Removing each stream's own header lets the decoded PCM
    comparison cover the same frames without applying either stream's gapless trim.
    """
    if has_xing_header(frames):
        return _without_first_frame(frames)
    return frames


def _leading_id3_end(data: bytes) -> int:
    """Skip one well-formed leading ID3v2 tag so the first frame has an anchored boundary."""
    if len(data) < 10 or data[:3] != b"ID3":
        return 0
    size_bytes = data[6:10]
    if any(value & 0x80 for value in size_bytes):
        return 0
    size = (size_bytes[0] << 21) | (size_bytes[1] << 14) | (size_bytes[2] << 7) | size_bytes[3]
    end = 10 + size + (10 if data[5] & 0x10 else 0)
    return end if end <= len(data) else 0


def frames_only(data: bytes) -> FrameScan:
    """Return only confirmed MPEG Layer III frames from ``data``.

    A candidate is confirmed by a header exactly one frame later, or by the end of the input.
    The first frame after a recognized leading ID3v2 tag (or at byte zero) is anchored by the
    stream boundary; once it is confirmed, that adjacency retains the final frame before a junk
    gap too. False syncs in arbitrary bytes must pass the double-header check before they can
    start a chain.

    Raises:
        InvalidAudioError: No complete MPEG Layer III frame was found.
    """
    if not data:
        raise InvalidAudioError(AudioCode.INVALID, "audio has no MPEG audio frames")

    output = bytearray()
    stream_start = _leading_id3_end(data)
    offset = 0
    linked_next = False
    while offset < len(data):
        length = _frame_length(data, offset)
        if length is not None:
            end = offset + length
            next_header = end == len(data) or _frame_length(data, end) is not None
            anchored_first = not output and offset == stream_start
            if linked_next or next_header or anchored_first:
                output.extend(data[offset:end])
                offset = end
                linked_next = True
                continue
        offset += 1
        linked_next = False

    if not output:
        raise InvalidAudioError(AudioCode.INVALID, "audio has no MPEG audio frames")
    frames = bytes(output)
    return FrameScan(frames=frames, dropped_bytes=len(data) - len(frames))
