"""Error categories for media preparation.

Only deterministic malformed/unsupported inputs use ``InvalidMediaError``.
Capacity, executable, timeout and native-library failures are operational and
must not be turned into a client-side "not decodable" response.
"""

from __future__ import annotations

from enum import StrEnum


class MediaIngestError(Exception):
    """Base class for preparation failures."""


class InvalidMediaError(MediaIngestError):
    """The source bytes are malformed or exceed a configured media limit."""


class UnsupportedMediaError(InvalidMediaError):
    """The source is valid media but outside the supported ingest policy."""


class MediaProcessingError(MediaIngestError):
    """A local operational dependency failed while preparing media."""


class AudioCode(StrEnum):
    """The typed ``422`` codes of an audio upload (the wire contract)."""

    FORMAT_NOT_SUPPORTED = "audio_format_not_supported"
    HAS_VIDEO = "audio_has_video"
    INVALID = "invalid_audio"
    CODEC_NOT_SUPPORTED = "audio_codec_not_supported"
    DURATION_OUT_OF_RANGE = "audio_duration_out_of_range"


class InvalidAudioError(InvalidMediaError):
    """No usable audio stream, an undecodable file, or a duration out of range.

    ``code`` is the typed ``422`` code and ``detail`` carries numbers only (never a file name, a
    tag value or probe output), so a refusal can be logged and returned as is.
    """

    def __init__(self, code: AudioCode, message: str, detail: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


class UnsupportedAudioError(UnsupportedMediaError):
    """Valid media outside the audio policy: its format, its codec, or a video stream."""

    def __init__(self, code: AudioCode, message: str, detail: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


AudioRejection = InvalidAudioError | UnsupportedAudioError
"""Every typed audio refusal (for ``except`` clauses and annotations)."""
