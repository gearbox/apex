"""Application-wide constants."""

import re
from collections.abc import Mapping
from uuid import UUID

from src.core.enums import ScriptVariant

# Sentinel user representing the system itself.
# Used as `created_by` for rows seeded during migrations (e.g. default pricing rules).
# Inserted into `users` by migration 002. Cannot authenticate — has no password and is inactive.
SYSTEM_USER_ID = UUID("00000000-0000-0000-0000-000000000001")

# Video frame extraction request caps. Shared by src/api/schemas/frames.py (msgspec.Meta
# bounds) and src/core/config.py (frame_extract_stale_running_seconds validator) — the
# sweep threshold must exceed worst-case job runtime, which is derived from these caps.
MAX_PREVIEW_FRAME_COUNT = 60
MAX_EXTRACT_TIMESTAMPS = 50

# GET /v1/provisioning/scripts/{variant}/{ref} (D3): variant -> (owner/repo, path in that repo).
# This mapping — not the request — is the only source of which private repo/path is ever
# fetched. A generic {owner}/{repo}/{path} proxy would be an arbitrary read oracle into
# every private repo the server-side GitHub token can see, so this table is load-bearing,
# not stylistic. `base` (src/core/enums.py ScriptVariant) is a reserved slot with no source
# yet — requesting it resolves to "not found" until a second variant ships.
SCRIPT_VARIANT_SOURCES: Mapping[ScriptVariant, tuple[str, str]] = {
    ScriptVariant.comfyui: ("gearbox/aisha", "scripts/aisha-provision-comfyui.sh"),
}

# Accepted `ref` values for the provisioning-script endpoint and for
# Settings.provisioning_script_ref: a release tag (vX.Y.Z) or a full 40-hex commit SHA.
# Both are immutable, so a resolved script for one can be cached indefinitely. A single
# additional dev branch may be allowed outside production — see
# Settings.provisioning_script_dev_ref — but it is checked separately, never folded into
# this pattern, so it never gets the long "immutable" cache TTL.
PROVISIONING_REF_PATTERN = re.compile(r"^(v\d+\.\d+\.\d+|[0-9a-f]{40})$")
