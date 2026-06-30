"""Pure builder for MediaObject — no DB or IO calls.

Build a MediaObject from a full row (GenerationOutput or UserImage) and its
derivative rows. Callers are responsible for loading the data; this module
only assembles the DTO.
"""

from __future__ import annotations

from collections.abc import Sequence

import structlog

from src.api.schemas.media import ImageVariant, MediaObject, MediaOriginal
from src.core.enums import OutputMediaType
from src.core.thumbnails import label_for_max_edge
from src.db.models.storage import GenerationOutput, UserImage

logger = structlog.get_logger(__name__)

OUTPUT_PREFIX = "/v1/content/outputs"
UPLOAD_PREFIX = "/v1/content/uploads"

_MediaRow = GenerationOutput | UserImage


def _build_media(
    *,
    full: _MediaRow,
    derivatives: Sequence[_MediaRow],
    url_prefix: str,
) -> MediaObject:
    media_type = (
        OutputMediaType.VIDEO if full.content_type.startswith("video/") else OutputMediaType.IMAGE
    )

    original = MediaOriginal(
        url=f"{url_prefix}/{full.id}",
        width=full.width,
        height=full.height,
        content_type=full.content_type,
        size_bytes=full.size_bytes,
    )

    variants: list[ImageVariant] = []
    for d in derivatives:
        label = label_for_max_edge(d.thumbnail_max_edge)
        if label is None:
            continue
        w, h = d.width, d.height
        if w is None or h is None:
            logger.warning(
                "media.variant.missing_dims",
                content_id=str(d.id),
                label=label,
                thumbnail_max_edge=d.thumbnail_max_edge,
            )
            continue
        variants.append(ImageVariant(label=label, width=w, height=h, url=f"{url_prefix}/{d.id}"))

    variants.sort(key=lambda v: v.width)

    return MediaObject(
        media_type=media_type,
        original=original,
        variants=variants,
    )


def build_output_media(
    full: GenerationOutput,
    derivatives: Sequence[GenerationOutput],
) -> MediaObject:
    """Build a MediaObject for a generation output."""
    return _build_media(full=full, derivatives=derivatives, url_prefix=OUTPUT_PREFIX)


def build_upload_media(
    full: UserImage,
    derivatives: Sequence[UserImage],
) -> MediaObject:
    """Build a MediaObject for a user upload."""
    return _build_media(full=full, derivatives=derivatives, url_prefix=UPLOAD_PREFIX)
