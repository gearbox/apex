"""Byte-level image format sniffing and normalization (Pillow).

The sniffed bytes are the single source of truth for image format.
Declared content types and filename extensions are provenance metadata only.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from enum import StrEnum

import structlog
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from src.api.services.storage import MediaFormat

register_heif_opener()

logger = structlog.get_logger(__name__)


class ImageNormalizationError(Exception):
    """Raised when bytes cannot be decoded as an image."""


class SniffedFormat(StrEnum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"  # static
    WEBP_ANIMATED = "webp_animated"
    HEIF = "heif"  # heic/heif brands
    AVIF = "avif"
    UNKNOWN = "unknown"


_HEIF_BRANDS = {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1"}
_AVIF_BRANDS = {b"avif", b"avis"}


def sniff_format(data: bytes) -> SniffedFormat:
    """Inspect header bytes to determine image format without decoding."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return SniffedFormat.PNG
    if data[:3] == b"\xff\xd8\xff":
        return SniffedFormat.JPEG
    if len(data) >= 16 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk_fourcc = data[12:16]
        if chunk_fourcc in (b"VP8 ", b"VP8L"):
            return SniffedFormat.WEBP
        if chunk_fourcc == b"VP8X":
            if len(data) >= 21 and data[20] & 0x02:
                return SniffedFormat.WEBP_ANIMATED
            return SniffedFormat.WEBP
        return SniffedFormat.UNKNOWN
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in _AVIF_BRANDS:
            return SniffedFormat.AVIF
        return SniffedFormat.HEIF if brand in _HEIF_BRANDS else SniffedFormat.UNKNOWN
    return SniffedFormat.UNKNOWN


@dataclass(frozen=True, slots=True)
class NormalizedImage:
    data: bytes
    format: MediaFormat  # PNG | JPEG | WEBP only
    content_type: str  # derived: format.content_type
    converted: bool  # True when bytes were re-encoded
    sniffed: SniffedFormat  # what the input actually was


_PASSTHROUGH_FORMATS = {
    SniffedFormat.PNG: MediaFormat.PNG,
    SniffedFormat.JPEG: MediaFormat.JPEG,
    SniffedFormat.WEBP: MediaFormat.WEBP,
}


def _convert_to_png_sync(data: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.seek(0)
            img: Image.Image = ImageOps.exif_transpose(source) or source
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA" if "A" in img.mode else "RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG", icc_profile=img.info.get("icc_profile"))
            return buf.getvalue()
    except Exception as e:
        raise ImageNormalizationError("File is not a decodable image") from e


def _normalize_sync(data: bytes) -> NormalizedImage:
    sniffed = sniff_format(data)
    passthrough_format = _PASSTHROUGH_FORMATS.get(sniffed)
    if passthrough_format is not None:
        return NormalizedImage(
            data=data,
            format=passthrough_format,
            content_type=passthrough_format.content_type,
            converted=False,
            sniffed=sniffed,
        )

    converted_data = _convert_to_png_sync(data)
    return NormalizedImage(
        data=converted_data,
        format=MediaFormat.PNG,
        content_type=MediaFormat.PNG.content_type,
        converted=True,
        sniffed=sniffed,
    )


def _ensure_comfyui_input_sync(data: bytes) -> NormalizedImage:
    sniffed = sniff_format(data)
    if sniffed == SniffedFormat.WEBP:
        converted_data = _convert_to_png_sync(data)
        return NormalizedImage(
            data=converted_data,
            format=MediaFormat.PNG,
            content_type=MediaFormat.PNG.content_type,
            converted=True,
            sniffed=sniffed,
        )
    return _normalize_sync(data)


async def normalize_image(data: bytes) -> NormalizedImage:
    """Normalize arbitrary image bytes to a storable format.

    PNG, JPEG, and static WebP pass through unchanged. Everything else
    (animated WebP, HEIF, AVIF, unrecognized headers) is decoded and
    re-encoded as PNG. Raises ``ImageNormalizationError`` if the bytes
    cannot be decoded by Pillow.
    """
    return await asyncio.to_thread(_normalize_sync, data)


async def ensure_comfyui_input(data: bytes) -> NormalizedImage:
    """Like ``normalize_image``, but also converts static WebP to PNG.

    ComfyUI-side WebP handling is unreliable; this bridge guarantees
    ComfyUI only ever receives PNG or JPEG bytes.
    """
    return await asyncio.to_thread(_ensure_comfyui_input_sync, data)
