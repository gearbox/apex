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

# T3 (round-3 remediation): characters that are unsafe in Settings.provisioning_script_dev_ref
# because it is interpolated raw as the single `{ref:str}` path segment of
# GET /v1/provisioning/scripts/{variant}/{ref} (see
# src/api/services/gpu_session/_env_builder.py's build_provisioning_callback_urls). A
# '/' splits it across two path segments — the route 404s at the router, after a GPU
# instance has already been rented for the session; '?'/'#' splice into the query
# string/fragment ahead of the real session/token params; '%' would be reinterpreted as
# percent-encoding on decode.
_DEV_REF_UNSAFE_CHARS = frozenset("/?#%")


def validate_dev_ref_is_route_safe(ref: str) -> None:
    """Raise ValueError if `ref` cannot safely be one URL path segment.

    None of this is an attack surface — the value is operator-set, not
    request-supplied — but every one of these shapes is a foot-gun that costs a
    rented GPU node to discover: the node's bootstrap script URL is built with
    this value already substituted in, so a bad ref only 404s once the node
    tries to fetch it at boot.
    """
    for ch in _DEV_REF_UNSAFE_CHARS:
        if ch in ref:
            raise ValueError(
                f"provisioning_script_dev_ref must not contain {ch!r} — it is interpolated as "
                "a single URL path segment in the bootstrap-script URL; use a slash-free "
                "branch name (e.g. 'feature-bootstrap', not 'feature/bootstrap')"
            )
    if any(ch.isspace() for ch in ref):
        raise ValueError(
            "provisioning_script_dev_ref must not contain whitespace — it is interpolated as "
            "a single URL path segment in the bootstrap-script URL"
        )
    if ref in {".", ".."}:
        raise ValueError(
            f"provisioning_script_dev_ref must not be {ref!r} — it is interpolated as a single "
            "URL path segment in the bootstrap-script URL and this is a path-traversal shape"
        )
