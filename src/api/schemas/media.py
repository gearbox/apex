"""Unified media object schema — images and videos with srcset-ready variants."""

from __future__ import annotations

import msgspec

from src.core.enums import OutputMediaType


class ImageVariant(msgspec.Struct, kw_only=True):
    """A downscaled preview variant (always WEBP)."""

    label: str
    """Bucket label: 'sm' (150px longest edge) or 'md' (512px longest edge)."""

    width: int | None = None
    """Actual pixel width of this variant."""

    height: int | None = None
    """Actual pixel height of this variant."""

    url: str
    """Stable content-proxy path: '/v1/content/outputs/{id}' or '/v1/content/uploads/{id}'."""


class MediaOriginal(msgspec.Struct, kw_only=True):
    """Full-resolution original asset."""

    url: str
    """Stable content-proxy path. For video, this is the MP4 source."""

    width: int | None = None
    height: int | None = None

    content_type: str
    """MIME type: 'image/png', 'image/jpeg', 'video/mp4', etc."""

    size_bytes: int


class MediaObject(msgspec.Struct, kw_only=True):
    """Unified media envelope for images and videos.

    For video: original.url is the MP4, variants are poster-frame rasters.
    For images: original.url is the full-res image, variants are preview thumbnails.
    """

    media_type: OutputMediaType
    """'image' or 'video'. Derived from original.content_type."""

    original: MediaOriginal

    variants: list[ImageVariant] = msgspec.field(default_factory=list)
    """Preview rasters, ascending by width. May be empty when no thumbnails exist."""
