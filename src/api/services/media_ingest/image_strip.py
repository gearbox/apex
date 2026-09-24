"""Conservative byte-level metadata stripping for PNG, JPEG and WebP.

The transformations retain rendering payloads (including ICC, color and pixel
aspect data) but remove descriptive/application chunks.  ICC is deliberately a
functional-data exception: it can itself carry descriptive text, so this module
does not promise removal of every possible identifying byte.
"""

from __future__ import annotations

import struct
import zlib

from src.api.services.media_ingest.errors import InvalidMediaError
from src.core.enums import MediaFormat

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_ALLOWED = {
    b"IHDR",
    b"PLTE",
    b"IDAT",
    b"IEND",
    b"tRNS",
    b"cHRM",
    b"gAMA",
    b"iCCP",
    b"sBIT",
    b"sRGB",
    b"cICP",
    b"mDCV",
    b"cLLI",
    b"pHYs",
    b"acTL",
    b"fcTL",
    b"fdAT",
}
_WEBP_TOP_LEVEL = {b"VP8X", b"ICCP", b"ANIM", b"ANMF", b"ALPH", b"VP8 ", b"VP8L"}
_WEBP_ANMF = {b"ALPH", b"VP8 ", b"VP8L"}
_PNG_MAX_ICC_DECOMPRESSED = 4 * 1024 * 1024


def strip_image_metadata(data: bytes, image_format: MediaFormat) -> bytes:
    """Return validated bytes with unapproved container metadata removed."""
    if image_format is MediaFormat.PNG:
        return strip_png(data)
    if image_format is MediaFormat.JPEG:
        return strip_jpeg(data)
    if image_format is MediaFormat.WEBP:
        return strip_webp(data)
    raise InvalidMediaError(f"unsupported image format {image_format.value!r}")


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _is_png_critical(kind: bytes) -> bool:
    return bool(kind) and 65 <= kind[0] <= 90


def _validate_png_chunk(kind: bytes, payload: bytes) -> None:
    """Validate fixed-size ancillary chunks and bound retained ICC expansion."""
    expected_lengths = {
        b"cHRM": 32,
        b"gAMA": 4,
        b"sRGB": 1,
        b"cICP": 4,
        b"mDCV": 24,
        b"cLLI": 4,
        b"pHYs": 9,
        b"acTL": 8,
        b"fcTL": 26,
    }
    if kind in expected_lengths and len(payload) != expected_lengths[kind]:
        raise InvalidMediaError(f"invalid PNG {kind.decode('ascii')} length")
    if kind != b"iCCP":
        return
    separator = payload.find(b"\x00")
    if not 1 <= separator <= 79 or separator + 2 > len(payload) or payload[separator + 1] != 0:
        raise InvalidMediaError("invalid PNG ICC profile")
    try:
        decompressor = zlib.decompressobj()
        expanded = decompressor.decompress(payload[separator + 2 :], _PNG_MAX_ICC_DECOMPRESSED + 1)
    except zlib.error as exc:
        raise InvalidMediaError("invalid PNG ICC profile") from exc
    if (
        len(expanded) > _PNG_MAX_ICC_DECOMPRESSED
        or decompressor.unconsumed_tail
        or not decompressor.eof
    ):
        raise InvalidMediaError("PNG ICC profile exceeds the allowed size")


def strip_png(data: bytes) -> bytes:
    """Allowlist PNG/APNG chunks while checking CRCs and basic ordering."""
    if not data.startswith(_PNG_SIGNATURE):
        raise InvalidMediaError("invalid PNG signature")
    offset = len(_PNG_SIGNATURE)
    chunks: list[tuple[bytes, bytes]] = []
    seen_ihdr = False
    seen_iend = False
    seen_idat = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise InvalidMediaError("truncated PNG chunk")
        size = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + size
        if end > len(data):
            raise InvalidMediaError("PNG chunk exceeds input")
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + size]
        crc = struct.unpack_from(">I", data, offset + 8 + size)[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != crc:
            raise InvalidMediaError("PNG CRC mismatch")
        if not seen_ihdr:
            if kind != b"IHDR" or size != 13:
                raise InvalidMediaError("PNG must start with a 13-byte IHDR")
            seen_ihdr = True
        if kind == b"IHDR" and chunks:
            raise InvalidMediaError("duplicate PNG IHDR")
        if kind == b"IEND":
            if size != 0 or not seen_idat:
                raise InvalidMediaError("invalid PNG IEND")
            seen_iend = True
            chunks.append((kind, payload))
            if end != len(data):
                # Bytes after IEND are not part of the image and are dropped.
                offset = end
            break
        if kind == b"IDAT":
            seen_idat = True
        elif seen_idat and kind == b"PLTE":
            raise InvalidMediaError("PNG PLTE appears after image data")
        if kind not in _PNG_ALLOWED and _is_png_critical(kind):
            raise InvalidMediaError(f"unknown critical PNG chunk {kind!r}")
        if kind in _PNG_ALLOWED:
            _validate_png_chunk(kind, payload)
            chunks.append((kind, payload))
        offset = end
    if not seen_ihdr or not seen_iend:
        raise InvalidMediaError("PNG is missing required chunks")
    return _PNG_SIGNATURE + b"".join(_png_chunk(kind, payload) for kind, payload in chunks)


def _jpeg_segment(marker: int, payload: bytes) -> bytes:
    return b"\xff" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload


def _valid_jfif(payload: bytes) -> bytes | None:
    if not payload.startswith(b"JFIF\x00") or len(payload) < 14:
        return None
    x_thumb, y_thumb = payload[12], payload[13]
    expected = 14 + 3 * x_thumb * y_thumb
    return None if len(payload) != expected else payload[:12] + b"\x00\x00"


def _valid_icc(payload: bytes) -> tuple[int, int] | None:
    prefix = b"ICC_PROFILE\x00"
    if not payload.startswith(prefix) or len(payload) <= len(prefix) + 1:
        return None
    sequence = payload[len(prefix)]
    count = payload[len(prefix) + 1]
    if sequence == 0 or count == 0 or sequence > count:
        return None
    return sequence, count


def _valid_adobe(payload: bytes) -> bool:
    # APP14 is exactly "Adobe" + version + flags0 + flags1 + transform.
    return len(payload) == 12 and payload.startswith(b"Adobe")


def _validate_icc_parts(parts: list[tuple[int, int]]) -> None:
    if not parts:
        return
    count = parts[0][1]
    if any(part_count != count for _, part_count in parts) or {
        sequence for sequence, _ in parts
    } != set(range(1, count + 1)):
        raise InvalidMediaError("incomplete JPEG ICC profile sequence")


def strip_jpeg(data: bytes) -> bytes:
    """Strip JPEG APP/COM metadata while preserving scans and coding tables."""
    if not data.startswith(b"\xff\xd8"):
        raise InvalidMediaError("invalid JPEG SOI")
    out = bytearray(b"\xff\xd8")
    offset = 2
    icc_parts: list[tuple[int, int]] = []
    saw_scan = False
    while offset < len(data):
        if data[offset] != 0xFF:
            raise InvalidMediaError("JPEG marker expected")
        marker_start = offset
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            raise InvalidMediaError("truncated JPEG marker")
        marker = data[offset]
        offset += 1
        if marker == 0xD9:
            if not saw_scan:
                raise InvalidMediaError("JPEG has no scan")
            _validate_icc_parts(icc_parts)
            out.extend(b"\xff\xd9")
            return bytes(out)
        if marker in {0xD8, 0x01} or 0xD0 <= marker <= 0xD7:
            out.extend(data[marker_start:offset])
            continue
        if offset + 2 > len(data):
            raise InvalidMediaError("truncated JPEG segment length")
        length = struct.unpack_from(">H", data, offset)[0]
        if length < 2 or offset + length > len(data):
            raise InvalidMediaError("invalid JPEG segment length")
        payload = data[offset + 2 : offset + length]
        offset += length
        keep = not (marker == 0xFE or 0xE0 <= marker <= 0xEF)
        replacement: bytes | None = None
        if marker == 0xE0:
            replacement = _valid_jfif(payload)
            keep = replacement is not None
        elif marker == 0xE2:
            icc_part = _valid_icc(payload)
            keep = icc_part is not None
            if icc_part is not None:
                icc_parts.append(icc_part)
        elif marker == 0xEE:
            keep = _valid_adobe(payload)
        if keep:
            out.extend(_jpeg_segment(marker, replacement if replacement is not None else payload))
        if marker != 0xDA:
            continue
        saw_scan = True
        # Entropy-coded data continues until a non-stuffed, non-restart marker.
        scan_start = offset
        while offset < len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            run_start = offset
            while offset < len(data) and data[offset] == 0xFF:
                offset += 1
            if offset >= len(data):
                raise InvalidMediaError("truncated JPEG scan")
            next_marker = data[offset]
            if next_marker == 0x00 or 0xD0 <= next_marker <= 0xD7:
                offset += 1
                continue
            out.extend(data[scan_start:run_start])
            offset = run_start
            break
        else:
            raise InvalidMediaError("JPEG scan has no EOI")
    if not saw_scan:
        raise InvalidMediaError("JPEG has no scan")
    raise InvalidMediaError("JPEG has no EOI")


def _webp_chunk(kind: bytes, payload: bytes) -> bytes:
    padding = b"\x00" if len(payload) % 2 else b""
    return kind + struct.pack("<I", len(payload)) + payload + padding


def _parse_webp_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise InvalidMediaError("invalid WebP RIFF header")
    declared_end = 8 + struct.unpack_from("<I", data, 4)[0]
    if declared_end < 12 or declared_end > len(data):
        raise InvalidMediaError("WebP RIFF extent is invalid")
    offset = 12
    chunks: list[tuple[bytes, bytes]] = []
    while offset < declared_end:
        if offset + 8 > declared_end:
            raise InvalidMediaError("truncated WebP chunk")
        kind = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        payload_end = offset + 8 + size
        padded_end = payload_end + size % 2
        if padded_end > declared_end:
            raise InvalidMediaError("WebP chunk exceeds RIFF extent")
        chunks.append((kind, data[offset + 8 : payload_end]))
        offset = padded_end
    if offset != declared_end:
        raise InvalidMediaError("invalid WebP chunk alignment")
    return chunks


def _strip_anmf(payload: bytes) -> bytes:
    if len(payload) < 16:
        raise InvalidMediaError("truncated WebP ANMF header")
    offset = 16
    result = bytearray(payload[:16])
    image_chunks = 0
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise InvalidMediaError("truncated WebP ANMF subchunk")
        kind = payload[offset : offset + 4]
        size = struct.unpack_from("<I", payload, offset + 4)[0]
        end = offset + 8 + size
        padded_end = end + size % 2
        if padded_end > len(payload):
            raise InvalidMediaError("WebP ANMF subchunk exceeds frame")
        if kind in _WEBP_ANMF:
            if kind in {b"VP8 ", b"VP8L"}:
                image_chunks += 1
            result.extend(_webp_chunk(kind, payload[offset + 8 : end]))
        offset = padded_end
    if image_chunks != 1:
        raise InvalidMediaError("WebP ANMF frame must contain exactly one image payload")
    return bytes(result)


def strip_webp(data: bytes) -> bytes:
    """Allowlist WebP chunks, including frame-local chunks in animated files."""
    chunks = _parse_webp_chunks(data)
    clean: list[tuple[bytes, bytes]] = []
    vp8x_index: int | None = None
    saw_anim = False
    saw_frame = False
    for kind, chunk_payload in chunks:
        if kind not in _WEBP_TOP_LEVEL:
            continue
        if kind == b"VP8X":
            if len(chunk_payload) != 10 or vp8x_index is not None:
                raise InvalidMediaError("invalid WebP VP8X chunk")
            vp8x_index = len(clean)
        elif kind == b"ANMF":
            saw_frame = True
            clean.append((kind, _strip_anmf(chunk_payload)))
            continue
        elif kind == b"ANIM":
            if len(chunk_payload) != 6 or saw_anim:
                raise InvalidMediaError("invalid WebP ANIM chunk")
            saw_anim = True
        elif kind == b"ALPH" and not chunk_payload:
            raise InvalidMediaError("invalid empty WebP ALPH chunk")
        clean.append((kind, chunk_payload))
    if not clean or all(kind not in {b"VP8 ", b"VP8L", b"ANMF"} for kind, _ in clean):
        raise InvalidMediaError("WebP has no image payload")
    if vp8x_index is not None:
        flags = bytearray(clean[vp8x_index][1])
        flags[0] &= ~0x0C  # EXIF and XMP are deliberately removed.
        clean[vp8x_index] = (b"VP8X", bytes(flags))
        if saw_anim and not (clean[vp8x_index][1][0] & 0x02):
            raise InvalidMediaError(
                "WebP animation flag is missing or animation chunks are inconsistent"
            )
    if saw_anim != saw_frame or (saw_anim and vp8x_index is None):
        raise InvalidMediaError("inconsistent WebP animation chunks")
    body = b"".join(_webp_chunk(kind, payload) for kind, payload in clean)
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body
