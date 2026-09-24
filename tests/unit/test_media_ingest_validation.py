"""Malformed carrier and value contracts at the media ingest boundary."""

from __future__ import annotations

import io
import struct
import zlib
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from src.api.services.media_ingest import InvalidMediaError
from src.api.services.media_ingest.image_strip import strip_image_metadata
from src.api.services.media_ingest.types import PreparedImage, PreparedVideo
from src.core.enums import MediaFormat
from src.core.media_hash import HashSample, HashSet, PdqHash

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit


def _image(image_format: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 6), "red").save(output, format=image_format)
    return output.getvalue()


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _webp_chunk(kind: bytes, payload: bytes) -> bytes:
    return kind + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) & 1 else b"")


def _riff(body: bytes) -> bytes:
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: b"bad" + data[3:],
        lambda data: data[:12],
        lambda data: data[:20],
        lambda data: data[:29] + bytes([data[29] ^ 1]) + data[30:],
        lambda data: data[:8] + _png_chunk(b"tEXt", b"x") + data[8:],
        lambda data: data[:33] + data[8:33] + data[33:],
        lambda data: data[:-12] + _png_chunk(b"IEND", b"x"),
        lambda data: data[:33] + _png_chunk(b"ABCD", b"") + data[33:],
        lambda data: data[:-12],
        lambda data: data[:33] + _png_chunk(b"pHYs", b"x") + data[33:],
        lambda data: data[:33] + _png_chunk(b"iCCP", b"x") + data[33:],
        lambda data: data[:33] + _png_chunk(b"iCCP", b"n\x00\x00garbage") + data[33:],
        lambda data: (
            data[:33] + _png_chunk(b"iCCP", b"n\x00\x00" + zlib.compress(b"icc")[:-2]) + data[33:]
        ),
        lambda data: data[:-12] + _png_chunk(b"PLTE", b"\x00\x00\x00") + data[-12:],
    ],
)
def test_malformed_png_is_rejected(mutation: Callable[[bytes], bytes]) -> None:
    source = _image("PNG")
    with pytest.raises(InvalidMediaError):
        strip_image_metadata(mutation(source), MediaFormat.PNG)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: b"bad" + data[3:],
        lambda _data: b"\xff\xd8\x00",
        lambda _data: b"\xff\xd8\xff",
        lambda _data: b"\xff\xd8\xff\xd9",
        lambda _data: b"\xff\xd8\xff\xe1",
        lambda _data: b"\xff\xd8\xff\xe1\x00\x01",
        lambda data: data[:-2] + b"\xff",
        lambda data: data[:-2],
        lambda data: data[:2] + b"\xff\xe2\x00\x17ICC_PROFILE\x00\x01\x02private" + data[2:],
    ],
)
def test_malformed_jpeg_is_rejected(mutation: Callable[[bytes], bytes]) -> None:
    source = _image("JPEG")
    with pytest.raises(InvalidMediaError):
        strip_image_metadata(mutation(source), MediaFormat.JPEG)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: b"bad" + data[3:],
        lambda data: data[:4] + struct.pack("<I", len(data) + 10) + data[8:],
        lambda _data: b"RIFF\x08\x00\x00\x00WEBPxxxx",
        lambda data: data[:16] + struct.pack("<I", len(data)) + data[20:],
        lambda _data: _riff(_webp_chunk(b"XMP ", b"private")),
        lambda data: _riff(_webp_chunk(b"VP8X", b"short") + data[12:]),
        lambda data: _riff(_webp_chunk(b"ANIM", b"short") + data[12:]),
        lambda data: _riff(_webp_chunk(b"ANIM", b"\x00" * 6) + data[12:]),
        lambda data: _riff(_webp_chunk(b"ANMF", b"short") + data[12:]),
    ],
)
def test_malformed_webp_is_rejected(mutation: Callable[[bytes], bytes]) -> None:
    source = _image("WEBP")
    with pytest.raises(InvalidMediaError):
        strip_image_metadata(mutation(source), MediaFormat.WEBP)


def test_hash_value_validation() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        PdqHash(bits=b"short", quality=50)
    with pytest.raises(ValueError, match="quality"):
        PdqHash(bits=b"\x00" * 32, quality=101)

    pdq = PdqHash(bits=b"\x00" * 32, quality=50)
    with pytest.raises(ValueError, match="sample_index"):
        HashSample(pdq=pdq, sample_index=-1)
    with pytest.raises(ValueError, match="frame_timestamp_ms"):
        HashSample(pdq=pdq, sample_index=0, frame_timestamp_ms=-1)

    still = HashSample(pdq=pdq, sample_index=0)
    with pytest.raises(ValueError, match="identifiers"):
        HashSet(profile_id="", sampling_profile="still-v1", samples=(still,))
    with pytest.raises(ValueError, match="at least one"):
        HashSet(profile_id="p", sampling_profile="still-v1", samples=())
    with pytest.raises(ValueError, match="contiguous"):
        HashSet(profile_id="p", sampling_profile="still-v1", samples=(HashSample(pdq, 1),))
    with pytest.raises(ValueError, match="exactly one"):
        HashSet(profile_id="p", sampling_profile="still-v1", samples=(still, HashSample(pdq, 1)))
    with pytest.raises(ValueError, match="all require timestamps"):
        HashSet(
            profile_id="p",
            sampling_profile="video-v1",
            samples=(HashSample(pdq, 0, 0), HashSample(pdq, 1)),
        )
    with pytest.raises(ValueError, match="nondecreasing"):
        HashSet(
            profile_id="p",
            sampling_profile="video-v1",
            samples=(HashSample(pdq, 0, 100), HashSample(pdq, 1, 0)),
        )


def test_prepared_media_type_validation() -> None:
    pdq = PdqHash(bits=b"\x00" * 32, quality=50)
    still = HashSet("p", "still-v1", (HashSample(pdq, 0),))
    video = HashSet("p", "video-v1", (HashSample(pdq, 0, 0),))
    with pytest.raises(ValueError, match="image format"):
        PreparedImage(b"x", MediaFormat.MP4, 1, 1, still, False, False)
    with pytest.raises(ValueError, match="dimensions"):
        PreparedImage(b"x", MediaFormat.PNG, 0, 1, still, False, False)
    with pytest.raises(ValueError, match="video format"):
        PreparedVideo(b"x", MediaFormat.PNG, 1, 1, 1000, video)
    with pytest.raises(ValueError, match="dimensions and duration"):
        PreparedVideo(b"x", MediaFormat.MP4, 1, 1, 0, video)
    with pytest.raises(ValueError, match="timestamps"):
        PreparedVideo(b"x", MediaFormat.MP4, 1, 1, 1000, still)
