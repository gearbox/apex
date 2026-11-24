"""Shared limits and sentinels for library assets.

Pure module: no DB, no Litestar imports. Kept separate from
``src.core.library_ref`` (identity/parsing primitives) so that both the
schemas layer (``src.api.schemas.library``) and the services layer
(``src.api.services.library``) can import a single source of truth without
schemas depending on services (layering) or services depending on schemas
for a bare constant.
"""

from __future__ import annotations

from datetime import UTC, datetime

MAX_TAGS_PER_ASSET = 20
"""Maximum number of tags a single library asset may carry, enforced by
every mutation path: ``patch_asset`` (replace-set), bulk ``add_tags``
(union of existing + new), and the ``LibraryAssetPatch.tag_ids`` wire
bound."""

PINNED_EXPIRY = datetime(9999, 12, 31, tzinfo=UTC)
"""The *effective expiry* of a held asset: a held asset never expires while it is held.

``library_asset_holds`` exempts an asset from the retention sweep, so its stored ``expires_at``
routinely lies in the past. The ``expiring`` filter and the ``expiring_soon`` sort (and the cursor of
that sort) therefore key off ``CASE WHEN EXISTS (hold of this asset) THEN PINNED_EXPIRY ELSE
expires_at END`` - computed once, in SQL, as the ``expiry_key`` column of every library row. The
wire's ``expires_at`` keeps the stored value; this constant never reaches a client except inside an
opaque ``expiring_soon`` cursor."""
