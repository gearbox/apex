"""Server-side image thumbnail generation (Pillow)."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass

import structlog
from PIL import Image

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
