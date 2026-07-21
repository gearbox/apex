"""Shared numeric limits for library assets.

Pure module: no DB, no Litestar imports. Kept separate from
``src.core.library_ref`` (identity/parsing primitives) so that both the
schemas layer (``src.api.schemas.library``) and the services layer
(``src.api.services.library``) can import a single source of truth without
schemas depending on services (layering) or services depending on schemas
for a bare constant.
"""

from __future__ import annotations

MAX_TAGS_PER_ASSET = 20
"""Maximum number of tags a single library asset may carry, enforced by
every mutation path: ``patch_asset`` (replace-set), bulk ``add_tags``
(union of existing + new), and the ``LibraryAssetPatch.tag_ids`` wire
bound."""
