"""The sole native ``pdqhash`` import and the v1 hash-input profile.

V1 decodes the prepared image with Pillow, composites alpha on opaque white,
uses Pillow's non-colour-managed RGB conversion, and packs bits big-endian.
It is intentionally not a claim of interchangeability with other PDQ pipelines.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from PIL import Image

from src.api.services.media_ingest.errors import MediaProcessingError
from src.core.media_hash import PdqHash

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def _rgb_v1(image: Image.Image) -> NDArray[np.uint8]:
    """Return a contiguous uint8 RGB matrix using the documented v1 treatment."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - installation failure path
        raise MediaProcessingError("NumPy is unavailable for PDQ preparation") from exc

    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        rgb = background.convert("RGB")
    else:
        rgb = image.convert("RGB")
    return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))


def pdq_from_pillow(image: Image.Image) -> PdqHash:
    """Compute a canonical 256-bit PDQ value from an already-selected image frame."""
    try:
        import numpy as np
        import pdqhash
    except ImportError as exc:  # pragma: no cover - exercised by deployment smoke check
        raise MediaProcessingError("pdqhash is unavailable for media preparation") from exc
    try:
        raw_bits, quality = pdqhash.compute(_rgb_v1(image))
        bits = np.asarray(raw_bits, dtype=np.uint8).reshape(-1)
        _validate_pdq_bits(bits)
        packed = np.packbits(bits, bitorder="big").tobytes()
        return PdqHash(bits=packed, quality=int(quality))
    except MediaProcessingError:
        raise
    except Exception as exc:
        raise MediaProcessingError("PDQ computation failed") from exc


def _validate_pdq_bits(bits: NDArray[np.uint8]) -> None:
    """Check the native wrapper's vector contract outside the compute try block."""
    import numpy as np

    if bits.size != 256 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("pdqhash returned an invalid bit vector")


def pdq_from_image_bytes(data: bytes) -> PdqHash:
    """Decode the first displayed frame in prepared image data and hash it."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.seek(0)
            image.load()
            return pdq_from_pillow(image)
    except MediaProcessingError:
        raise
    except Exception as exc:
        raise MediaProcessingError("prepared image could not be decoded for PDQ") from exc
