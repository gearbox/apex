"""Server-side image thumbnail generation (Pillow)."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from PIL import Image

from src.api.services.image_normalization import coerce_mode_for_encode
from src.core.thumbnails import THUMBNAIL_SPECS, ThumbnailSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

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
        img = coerce_mode_for_encode(source)
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
            # Convert to a safe mode while the source is still open (palette intact).
            # frombytes() on a "P" image strips the palette, so convert("RGB") on the
            # reconstructed image maps through an empty palette → solid black output.
            if source.mode in ("RGB", "RGBA"):
                base = source.copy()
            else:
                base = coerce_mode_for_encode(source)
    except Exception:
        logger.warning("thumbnail.image.decode_failed")
        return []

    results: list[GeneratedThumbnail] = []
    for spec in specs:
        try:
            img = base.copy()
            img.thumbnail((spec.max_edge, spec.max_edge))
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=quality, method=4)
            results.append(
                GeneratedThumbnail(
                    spec=spec,
                    result=ThumbnailResult(data=buf.getvalue(), width=img.width, height=img.height),
                )
            )
        except Exception:
            logger.warning("thumbnail.image.failed", max_edge=spec.max_edge)

    return results
