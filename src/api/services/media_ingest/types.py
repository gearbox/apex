"""Public contracts for the media preparation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

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
        if self.format.is_video:
            raise ValueError("PreparedImage requires an image format")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("prepared image dimensions must be positive")

    @property
    def content_type(self) -> str:
        return self.format.content_type


@dataclass(frozen=True, slots=True)
class PreparedVideo:
    """Sanitized/remuxed video bytes and PDQ samples from decoded output frames."""

    data: bytes
    format: MediaFormat
    width: int
    height: int
    duration_ms: int
    hash_set: HashSet

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


class MediaIngestor(Protocol):
    """Small dependency used by writers; it owns no storage or database work."""

    async def prepare_image(self, data: bytes, *, policy: ImageIngestPolicy) -> PreparedImage: ...

    async def prepare_video(
        self, data: bytes, *, max_duration_seconds: float | None = None
    ) -> PreparedVideo: ...
