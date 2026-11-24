"""Library media-object construction, shared by the library and the studio.

``build_media_object`` is the one place a library asset's wire ``MediaObject`` (original +
thumbnail variants, content-proxy URLs) is built from normalized fields. It lives in its own small
module so a bounded context that shows library assets (the studio cast) imports it from here
instead of re-implementing URL building - and so the library never has to import that context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.schemas.media import ImageVariant, MediaObject, MediaOriginal
from src.api.services.media import OUTPUT_PREFIX, UPLOAD_PREFIX
from src.core.enums import media_kind_from_content_type
from src.core.library_ref import LibraryAssetSource, format_asset_ref
from src.core.thumbnails import label_for_max_edge

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from src.db.models.storage import GenerationOutput, UserImage

__all__ = ["build_media_object"]


def build_media_object(
    *,
    source: LibraryAssetSource,
    asset_id: UUID,
    width: int | None,
    height: int | None,
    content_type: str,
    size_bytes: int,
    duration_ms: int | None,
    derivatives: Sequence[GenerationOutput | UserImage],
) -> MediaObject:
    """Build a MediaObject from normalized fields shared by both source tables.

    Distinct from build_output_media/build_upload_media in services/media.py:
    those take a concrete GenerationOutput/UserImage ORM instance, but the
    library list path works from LibraryAssetRow — a cross-table projection
    that isn't either concrete type. Same construction logic, different input
    shape; the URL prefix constants are still reused from services/media.py.
    """
    prefix = OUTPUT_PREFIX if source == LibraryAssetSource.OUTPUT else UPLOAD_PREFIX
    media_type = media_kind_from_content_type(content_type)

    original = MediaOriginal(
        url=f"{prefix}/{asset_id}",
        width=width,
        height=height,
        content_type=content_type,
        size_bytes=size_bytes,
        duration_ms=duration_ms,
    )

    variants: list[ImageVariant] = []
    for d in derivatives:
        label = label_for_max_edge(d.thumbnail_max_edge)
        if label is None or d.width is None or d.height is None:
            continue
        variants.append(
            ImageVariant(label=label, width=d.width, height=d.height, url=f"{prefix}/{d.id}")
        )
    variants.sort(key=lambda v: v.width)

    return MediaObject(
        media_type=media_type,
        original=original,
        variants=variants,
        asset_ref=format_asset_ref(source, asset_id),
    )
