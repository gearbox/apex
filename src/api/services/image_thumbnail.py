"""Server-side image thumbnail generation (Pillow)."""

from __future__ import annotations

import asyncio
import io
from collections.abc import Sequence
from dataclasses import dataclass

import structlog
from PIL import Image

from src.core.thumbnails import THUMBNAIL_SPECS, ThumbnailSpec

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ImageDimensions:
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ThumbnailResult:
    data: bytes
    width: int
    height: int
    content_type: str = "image/webp"
    format: str = "webp"


@dataclass(frozen=True, slots=True)
class GeneratedThumbnail:
    spec: ThumbnailSpec
    result: ThumbnailResult


async def read_dimensions(image_bytes: bytes) -> ImageDimensions | None:
    """Read pixel dimensions from image bytes. Returns None on failure."""
    return await asyncio.to_thread(_read_dimensions_sync, image_bytes)


async def make_image_thumbnail(
    image_bytes: bytes, *, max_edge: int = 512, quality: int = 80
) -> ThumbnailResult | None:
    """Downscale to fit within max_edge (longest side), WEBP. None on failure."""
    try:
        return await asyncio.to_thread(_make_sync, image_bytes, max_edge, quality)
    except Exception:
        logger.warning("thumbnail.image.failed")
        return None


async def make_image_thumbnails(
    image_bytes: bytes,
    specs: Sequence[ThumbnailSpec] = THUMBNAIL_SPECS,
    *,
    quality: int = 80,
) -> list[GeneratedThumbnail]:
    """Decode source once and produce all requested size variants (WEBP).

    Per-spec failures are skipped and logged — never raised. Returns an empty
    list if the source image cannot be decoded at all.
    """
    return await asyncio.to_thread(_make_all_sync, image_bytes, list(specs), quality)


def _read_dimensions_sync(image_bytes: bytes) -> ImageDimensions | None:
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            return ImageDimensions(width=im.width, height=im.height)
    except Exception:
        return None


def _make_sync(image_bytes: bytes, max_edge: int, quality: int) -> ThumbnailResult | None:
    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        # Preserve alpha; WEBP supports it. Convert palette/CMYK to a safe mode.
        img: Image.Image
        if source.mode not in ("RGB", "RGBA"):
            img = source.convert("RGBA" if "A" in source.mode else "RGB")
        else:
            img = source
        img.thumbnail((max_edge, max_edge))  # in place, preserves aspect ratio
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=quality, method=4)
        return ThumbnailResult(data=buf.getvalue(), width=img.width, height=img.height)


def _make_all_sync(
    image_bytes: bytes,
    specs: list[ThumbnailSpec],
    quality: int,
) -> list[GeneratedThumbnail]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            base_mode = source.mode
            base_pixels = source.tobytes()
            base_size = source.size
    except Exception:
        logger.warning("thumbnail.image.decode_failed")
        return []

    results: list[GeneratedThumbnail] = []
    for spec in specs:
        try:
            # Reconstruct a fresh copy from the decoded pixels each time so
            # Pillow's in-place .thumbnail() doesn't corrupt subsequent sizes.
            img = Image.frombytes(base_mode, base_size, base_pixels)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA" if "A" in img.mode else "RGB")
            img.thumbnail((spec.max_edge, spec.max_edge))
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=quality, method=4)
            thumb = ThumbnailResult(data=buf.getvalue(), width=img.width, height=img.height)
            results.append(GeneratedThumbnail(spec=spec, result=thumb))
        except Exception:
            logger.warning("thumbnail.image.failed", max_edge=spec.max_edge)

    return results
