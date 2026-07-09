"""Builder for the env dict passed to vastai_client.create_instance.

Centralizes the env-var contract with the Aisha CLI in one place so
service.py (initial start_session) and provisioning_worker.py (retry
path) cannot drift apart.

SECURITY: the returned dict contains tokens — never log it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from src.core.config import Settings


def build_acs_env(
    *,
    settings: Settings,
    session_id: UUID,
    bundle_name: str,
    bundle_version: str | None,
    comfyui_port: int,
    tunnel_token: str,
    callback_token: str,
) -> dict[str, str]:
    """Build the env dict for vastai_client.create_instance.

    Args:
        settings: Apex Settings (provides ai_bundles_*, aisha_*, hf_token, etc.)
        session_id: GpuSession UUID — surfaced to the CLI for log enrichment
        bundle_name: Resolved bundle name (e.g. "wan_2.2_i2v")
        bundle_version: Pinned version or None to mean "current"
        comfyui_port: Per-bundle port from bundle.hardware.comfyui_port
        tunnel_token: Cloudflared tunnel token (ephemeral, per-session)
        callback_token: Phase-2 callback auth token (ephemeral, per-session)

    Returns:
        Env dict ready to pass to vastai_client.create_instance(env=...).
        Contains the ACS_* contract keys plus the docker -p port mapping.

    Notes:
        - The "-p {port}:{port}" entry is a docker run flag, not an env var.
          Vast.ai's API treats both interchangeably in the env block.
        - ACS_COMFYUI_PORT and the -p mapping use the SAME port. They must.
    """
    return {
        # --- Bundle selection ---
        "ACS_BUNDLE": bundle_name,
        "ACS_BUNDLE_VERSION": bundle_version or "current",
        # --- GitHub auth + repos ---
        "ACS_GITHUB_TOKEN": settings.ai_bundles_github_token,
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
        "ACS_APEX_CALLBACK_URL": settings.apex_callback_url,
        "ACS_APEX_CALLBACK_TOKEN": callback_token,
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
