"""Builder for the env dict passed to vastai_client.create_instance.

Centralizes the env-var contract with the Aisha CLI in one place so
service.py (initial start_session) and provisioning_worker.py (retry
path) cannot drift apart.

SECURITY: the returned dict contains tokens — never log it. Since D3, that
includes PROVISIONING_SCRIPT and PROVISIONER_WEBHOOK_URL, which carry the
per-session callback token in their query string — treat them exactly like
tunnel_token/callback_token themselves; never log a full value, only redact
or log lengths/prefixes if logging is ever needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlencode

from src.core.config import normalize_apex_callback_url
from src.core.constants import PROVISIONING_REF_PATTERN, validate_dev_ref_is_route_safe
from src.core.enums import ScriptVariant

if TYPE_CHECKING:
    from uuid import UUID

    from src.core.config import Settings


def build_provisioning_callback_urls(
    *,
    apex_callback_url: str,
    session_id: UUID,
    callback_token: str,
    provision_script_ref: str,
) -> tuple[str, str]:
    """Build the script and failure-webhook URLs from one normalized origin.

    Raises:
        ValueError: `provision_script_ref` is not route-safe as a single URL path
            segment (T3, round-3 remediation) — it is interpolated raw into the
            script URL below, and the GET /v1/provisioning/scripts/{variant}/{ref}
            route accepts exactly one path segment for `{ref}`. Checked here too,
            not only in Settings' validator, so this function itself can never
            emit an unroutable URL regardless of who validated the ref upstream.
            A pinned immutable ref (tag/SHA) always passes — only a dev ref can
            be unsafe, and Settings.validate_provisioning_bootstrap_settings
            already rejects an unsafe one at boot; this is defense in depth.
    """
    if not PROVISIONING_REF_PATTERN.fullmatch(provision_script_ref):
        validate_dev_ref_is_route_safe(provision_script_ref)
    query = urlencode({"session": str(session_id), "token": callback_token})
    script_url = (
        f"{apex_callback_url}/v1/provisioning/scripts/"
        f"{ScriptVariant.comfyui.value}/{provision_script_ref}?{query}"
    )
    webhook_query = urlencode({"token": callback_token})
    webhook_url = f"{apex_callback_url}/v1/provisioning/webhook/{session_id}?{webhook_query}"
    return script_url, webhook_url


def build_acs_env(
    *,
    settings: Settings,
    session_id: UUID,
    operation_id: UUID,
    bundle_name: str,
    bundle_version: str | None,
    comfyui_port: int,
    tunnel_token: str,
    callback_token: str,
    provision_script_sha256: str,
) -> dict[str, str]:
    """Build the env dict for vastai_client.create_instance.

    Args:
        settings: Apex Settings (provides ai_bundles_*, aisha_*, hf_token, etc.)
        session_id: GpuSession UUID — surfaced to the CLI for log enrichment
        operation_id: Apex-owned bootstrap operation UUID for telemetry correlation
        bundle_name: Resolved bundle name (e.g. "wan_2.2_i2v")
        bundle_version: Pinned version or None to mean "current"
        comfyui_port: Per-bundle port from bundle.hardware.comfyui_port
        tunnel_token: Cloudflared tunnel token (ephemeral, per-session)
        callback_token: Phase-2 callback auth token (ephemeral, per-session)
        provision_script_sha256: sha256 of the bootstrap script apex is about to
            serve for this ref (from ProvisioningScriptService.resolve()) — the
            aisha script echoes this back in acs.provision.ready so apex can
            detect drift between what it served and what actually ran (D3).

    Returns:
        Env dict ready to pass to vastai_client.create_instance(env=...).
        Contains the ACS_* contract keys plus the docker -p port mapping.

    Notes:
        - The "-p {port}:{port}" entry is a docker run flag, not an env var.
          Vast.ai's API treats both interchangeably in the env block.
        - ACS_COMFYUI_PORT and the -p mapping use the SAME port. They must.
        - PROVISIONING_SCRIPT is Vast's own onstart-script-URL env var name;
          PROVISIONER_WEBHOOK_URL/PROVISIONER_FAILURE_ACTION are provisioner
          *settings* overrides — neither family gets the ACS_ prefix (D3).
          Apex owns the ref pin (settings.provisioning_script_ref); the Vast
          template itself no longer sets PROVISIONING_SCRIPT (D2).
    """
    callback_origin = normalize_apex_callback_url(settings.apex_callback_url)
    if callback_origin is None:
        raise ValueError("settings.apex_callback_url must be a valid absolute http(s) origin")
    script_url, webhook_url = build_provisioning_callback_urls(
        apex_callback_url=callback_origin,
        session_id=session_id,
        callback_token=callback_token,
        provision_script_ref=settings.provisioning_script_ref,
    )
    return {
        # --- Bundle selection ---
        "ACS_BUNDLE": bundle_name,
        "ACS_BUNDLE_VERSION": bundle_version or "current",
        # --- GitHub auth + repos ---
        "ACS_GITHUB_TOKEN": settings.github_content_token,
        "ACS_BUNDLES_REPO": settings.ai_bundles_repo_url,
        "ACS_BUNDLES_BRANCH": settings.ai_bundles_branch,
        "ACS_AISHA_REPO": settings.aisha_repo_url,
        "ACS_AISHA_BRANCH": settings.aisha_branch,
        # --- Tunnel + apex callback ---
        "ACS_CF_TUNNEL_TOKEN": tunnel_token,  # Aisha script / prefixed convention
        # Vast.ai Instance Portal reads this exact (unprefixed) name at boot to run a NAMED
        # tunnel instead of account-less quick tunnels. See Vast.ai Instance Portal docs →
        # "Named Tunnels". This is a third-party-mandated env-var name, NOT an apex CF-API
        # field, so it is intentionally unprefixed. Do not rename.
        "CF_TUNNEL_TOKEN": tunnel_token,
        "ACS_APEX_SESSION_ID": str(session_id),
        "ACS_APEX_OPERATION_ID": str(operation_id),
        "ACS_APEX_CALLBACK_URL": callback_origin,
        "ACS_APEX_CALLBACK_TOKEN": callback_token,
        # --- Bootstrap script delivery (D3) — Vast's own onstart-fetch + provisioner
        # settings overrides. See the module/function docstrings for the naming rule.
        "PROVISIONING_SCRIPT": script_url,
        "PROVISIONER_WEBHOOK_URL": webhook_url,
        "PROVISIONER_FAILURE_ACTION": "destroy",
        "ACS_PROVISION_SCRIPT_SHA256": provision_script_sha256,
        # --- Model download tokens ---
        "ACS_HF_TOKEN": settings.hf_token,
        "ACS_CIVITAI_API_TOKEN": settings.civitai_api_token,
        # --- ComfyUI runtime (supervisord launches ComfyUI with these) ---
        "ACS_COMFYUI_PORT": str(comfyui_port),
        "ACS_COMFYUI_HOST": settings.aisha_comfyui_host,
        "ACS_COMFYUI_EXTRA_ARGS": settings.aisha_comfyui_extra_args,
        # --- Docker port mapping (same port as ACS_COMFYUI_PORT) ---
        f"-p {comfyui_port}:{comfyui_port}": "1",
    }
