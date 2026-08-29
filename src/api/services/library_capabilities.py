"""Pure library action-capability resolver — no I/O, no DB session."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from src.core.enums import MediaKind

if TYPE_CHECKING:
    from src.core.library_ref import LibraryAssetSource
    from src.core.product import ProductConfig


class LibraryAction(StrEnum):
    """Actions a client may offer for a single library asset."""

    REMIX = "remix"
    CREATE_VARIATION = "create_variation"
    ANIMATE = "animate"
    EXTEND = "extend"
    EXTRACT_FRAME = "extract_frame"
    USE_AS_REFERENCE = "use_as_reference"
    USE_AS_FIRST_FRAME = "use_as_first_frame"
    USE_AS_LAST_FRAME = "use_as_last_frame"
    VIEW_SETTINGS = "view_settings"
    REPRODUCE = "reproduce"
    FAVORITE = "favorite"
    RENAME = "rename"
    DOWNLOAD = "download"
    DELETE = "delete"


# Deterministic, ordered action groups. Order here is the order returned by
# resolve_library_actions — never sorted or set-deduplicated.
_ALWAYS_ACTIONS: tuple[LibraryAction, ...] = (
    LibraryAction.FAVORITE,
    LibraryAction.RENAME,
    LibraryAction.DOWNLOAD,
    LibraryAction.DELETE,
)

_IMAGE_ACTIONS: tuple[LibraryAction, ...] = (
    LibraryAction.REMIX,
    LibraryAction.CREATE_VARIATION,
    LibraryAction.ANIMATE,
    LibraryAction.USE_AS_REFERENCE,
    LibraryAction.USE_AS_FIRST_FRAME,
    LibraryAction.USE_AS_LAST_FRAME,
)

_VIDEO_ACTIONS: tuple[LibraryAction, ...] = (
    LibraryAction.REMIX,
    LibraryAction.EXTEND,
    LibraryAction.EXTRACT_FRAME,
)

_GENERATION_METADATA_ACTIONS: tuple[LibraryAction, ...] = (
    LibraryAction.VIEW_SETTINGS,
    LibraryAction.REPRODUCE,
)


def resolve_library_actions(
    *,
    media_type: MediaKind,
    source: LibraryAssetSource,
    has_generation_metadata: bool,
    product_config: ProductConfig,
) -> tuple[LibraryAction, ...]:
    """Resolve the ordered, deterministic set of actions available for an asset.

    Pure function: no I/O, no DB session, safe to unit-test exhaustively.

    Args:
        media_type: IMAGE or VIDEO — selects the media-specific action group.
        source: Which table the asset lives in. Accepted for API symmetry
            with the rest of the library module; the MVP table derives
            everything it needs from ``has_generation_metadata`` instead
            (callers set that from ``source == OUTPUT``).
        has_generation_metadata: True when the asset has an associated
            generation job (i.e. is an output) — unlocks VIEW_SETTINGS/REPRODUCE.
        product_config: Accepted as the extension point for gating a future
            action on a product feature flag (``product_config.has_feature(...)``).
            No current action is feature-flagged, so it is unused today —
            do NOT invent a flag to use it.

    Returns:
        Ordered tuple of available LibraryAction values.
    """
    del source, product_config  # see docstring — accepted for future/API-symmetry use only

    actions: list[LibraryAction] = list(_ALWAYS_ACTIONS)
    actions.extend(_VIDEO_ACTIONS if media_type == MediaKind.VIDEO else _IMAGE_ACTIONS)
    if has_generation_metadata:
        actions.extend(_GENERATION_METADATA_ACTIONS)

    return tuple(actions)
