"""Public contracts for the media preparation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from src.api.services.media_ingest.audio import PEAKS_BYTES
from src.core.enums import MediaFormat  # noqa: TC001 - used for runtime validation
from src.core.media_hash import HashSet  # noqa: TC001 - carried by runtime values


class ImageIngestPolicy(StrEnum):
    """Image acceptance policy for uploads and provider-produced media."""

    UPLOAD = "upload"
    PROVIDER = "provider"


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """Sanitized image bytes and metadata used by every original-image writer."""

    data: bytes
    format: MediaFormat
    width: int
    height: int
    hash_set: HashSet
    orientation_baked: bool
    converted: bool

    def __post_init__(self) -> None:
        if not self.format.is_image:
            raise ValueError("PreparedImage requires an image format")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("prepared image dimensions must be positive")

    @property
    def content_type(self) -> str:
        return self.format.content_type


@dataclass(frozen=True, slots=True)
class VideoStreamProfile:
    """Probed facts about the prepared visual stream. Informational only — never a gate."""

    container: MediaFormat
    codec: str
    """ffprobe ``codec_name``: h264, hevc, vp9, av1, ..."""
    codec_profile: str | None
    """e.g. ``High``, ``Main 10``."""
    pix_fmt: str | None
    """e.g. ``yuv420p``, ``yuv420p10le``."""
    color_transfer: str | None
    """e.g. ``bt709``, ``smpte2084``, ``arib-std-b67``."""
    color_primaries: str | None
    """e.g. ``bt709``, ``bt2020``."""
    rotation_degrees: int
    """Display-matrix rotation in degrees normalized into ``[0, 360)``; 0 when absent or unreadable."""
    sample_aspect_ratio: str | None
    has_audio: bool

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in _HDR_TRANSFERS


_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})


@dataclass(frozen=True, slots=True)
class PreparedVideo:
    """Sanitized/remuxed video bytes and PDQ samples from decoded output frames."""

    data: bytes
    format: MediaFormat
    width: int
    height: int
    duration_ms: int
    hash_set: HashSet
    stream_profile: VideoStreamProfile

    def __post_init__(self) -> None:
        if not self.format.is_video:
            raise ValueError("PreparedVideo requires a video format")
        if self.width <= 0 or self.height <= 0 or self.duration_ms <= 0:
            raise ValueError("prepared video dimensions and duration must be positive")
        if any(sample.frame_timestamp_ms is None for sample in self.hash_set.samples):
            raise ValueError("video hash samples require timestamps")

    @property
    def content_type(self) -> str:
        return self.format.content_type


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    """Remuxed audio bytes (the upload's own stream, no tags) and the facts probed from them.

    The facts describe the **stored** bytes: they come from a second probe of the remuxed file.
    """

    data: bytes
    format: MediaFormat
    codec: str
    sample_rate_hz: int
    channels: int
    bit_rate: int | None
    duration_ms: int
    peaks: bytes
    peaks_version: int

    def __post_init__(self) -> None:
        if not self.format.is_audio:
            raise ValueError("PreparedAudio requires an audio format")
        if len(self.peaks) != PEAKS_BYTES:
            raise ValueError(f"prepared audio peaks must be {PEAKS_BYTES} bytes")
        if not self.codec:
            raise ValueError("prepared audio requires a codec")
        if (
            self.sample_rate_hz <= 0
            or self.channels <= 0
            or self.duration_ms <= 0
            or self.peaks_version <= 0
            or not self.data
        ):
            raise ValueError("prepared audio numbers must be positive")
        if self.bit_rate is not None and self.bit_rate <= 0:
            raise ValueError("prepared audio bit rate must be positive when present")

    @property
    def content_type(self) -> str:
        return self.format.content_type


class MediaIngestor(Protocol):
    """Small dependency used by writers; it owns no storage or database work."""

    async def prepare_image(self, data: bytes, *, policy: ImageIngestPolicy) -> PreparedImage: ...

    async def prepare_video(
        self, data: bytes, *, max_duration_seconds: float | None = None
    ) -> PreparedVideo: ...

    async def prepare_audio(self, data: bytes) -> PreparedAudio: ...
