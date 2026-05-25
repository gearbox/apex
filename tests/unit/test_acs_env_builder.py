"""Unit tests for build_acs_env.

The builder is a pure function — no mocks needed. Tests cover:
- all contract keys present
- each key maps to the correct settings field
- defaults and type coercions
- port consistency invariant (ACS_COMFYUI_PORT == -p mapping port)
- closed contract (no unexpected keys)
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID, uuid4

from src.api.services.gpu_session._env_builder import build_acs_env
from src.core.bundle_config import DEFAULT_COMFYUI_PORT

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

_EXPECTED_ACS_KEYS = {
    "ACS_BUNDLE",
    "ACS_BUNDLE_VERSION",
    "ACS_GITHUB_TOKEN",
    "ACS_BUNDLES_REPO",
    "ACS_BUNDLES_BRANCH",
    "ACS_AISHA_REPO",
    "ACS_AISHA_BRANCH",
    "ACS_CF_TUNNEL_TOKEN",
    "CF_TUNNEL_TOKEN",
    "ACS_APEX_SESSION_ID",
    "ACS_APEX_CALLBACK_URL",
    "ACS_APEX_CALLBACK_TOKEN",
    "ACS_HF_TOKEN",
    "ACS_CIVITAI_API_TOKEN",
    "ACS_COMFYUI_PORT",
    "ACS_COMFYUI_HOST",
    "ACS_COMFYUI_EXTRA_ARGS",
}


def _make_settings(**overrides: object) -> MagicMock:
    s = MagicMock()
    s.ai_bundles_github_token = "ghp_test_token"
    s.ai_bundles_repo_url = "https://github.com/gearbox/ai-bundles.git"
    s.ai_bundles_branch = "master"
    s.aisha_repo_url = "https://github.com/gearbox/aisha.git"
    s.aisha_branch = "master"
    s.apex_callback_url = "https://apex.example.com/callback"
    s.hf_token = "hf-test-token"
    s.civitai_api_token = "civitai-test-token"
    s.aisha_comfyui_host = "0.0.0.0"  # noqa: S104
    s.aisha_comfyui_extra_args = ""
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _build(**overrides: object) -> dict[str, str]:
    """Helper: build env with sensible defaults, overridable per-test."""
    session_id: UUID = overrides.pop("session_id", uuid4())  # type: ignore[assignment]
    settings = overrides.pop("settings", _make_settings())  # type: ignore[assignment]
    return build_acs_env(
        settings=settings,
        session_id=session_id,
        bundle_name=overrides.pop("bundle_name", "wan_2.2_i2v"),  # type: ignore[arg-type]
        bundle_version=overrides.pop("bundle_version", "260105-01"),  # type: ignore[arg-type]
        comfyui_port=overrides.pop("comfyui_port", DEFAULT_COMFYUI_PORT),  # type: ignore[arg-type]
        tunnel_token=overrides.pop("tunnel_token", "tunnel-secret"),  # type: ignore[arg-type]
        callback_token=overrides.pop("callback_token", "cb-secret"),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_acs_keys_present() -> None:
    env = _build()
    for key in _EXPECTED_ACS_KEYS:
        assert key in env, f"Missing required key: {key}"


def test_acs_github_token_from_settings() -> None:
    s = _make_settings(ai_bundles_github_token="ghp_my_pat")
    env = _build(settings=s)
    assert env["ACS_GITHUB_TOKEN"] == "ghp_my_pat"


def test_acs_bundles_repo_from_settings() -> None:
    s = _make_settings(ai_bundles_repo_url="https://github.com/gearbox/ai-bundles.git")
    env = _build(settings=s)
    assert env["ACS_BUNDLES_REPO"] == "https://github.com/gearbox/ai-bundles.git"


def test_acs_branches_default_to_master() -> None:
    s = _make_settings(ai_bundles_branch="master", aisha_branch="master")
    env = _build(settings=s)
    assert env["ACS_BUNDLES_BRANCH"] == "master"
    assert env["ACS_AISHA_BRANCH"] == "master"


def test_acs_comfyui_port_matches_port_mapping() -> None:
    env = _build(comfyui_port=12345)
    assert env["ACS_COMFYUI_PORT"] == "12345"
    assert env["-p 12345:12345"] == "1"


def test_acs_comfyui_host_default_is_0_0_0_0() -> None:
    s = _make_settings(aisha_comfyui_host="0.0.0.0")  # noqa: S104
    env = _build(settings=s)
    assert env["ACS_COMFYUI_HOST"] == "0.0.0.0"  # noqa: S104


def test_acs_comfyui_extra_args_default_is_empty() -> None:
    s = _make_settings(aisha_comfyui_extra_args="")
    env = _build(settings=s)
    assert env["ACS_COMFYUI_EXTRA_ARGS"] == ""


def test_bundle_version_none_becomes_current() -> None:
    env = _build(bundle_version=None)
    assert env["ACS_BUNDLE_VERSION"] == "current"


def test_session_id_is_stringified() -> None:
    sid = uuid4()
    env = _build(session_id=sid)
    assert env["ACS_APEX_SESSION_ID"] == str(sid)
    assert isinstance(env["ACS_APEX_SESSION_ID"], str)


def test_dict_has_no_unexpected_keys() -> None:
    port = DEFAULT_COMFYUI_PORT
    env = _build(comfyui_port=port)
    port_key = f"-p {port}:{port}"
    actual_keys = set(env.keys()) - {port_key}
    assert actual_keys == _EXPECTED_ACS_KEYS


def test_tunnel_token_and_callback_token_set_correctly() -> None:
    env = _build(tunnel_token="my-tunnel-token", callback_token="my-callback-token")
    assert env["ACS_CF_TUNNEL_TOKEN"] == "my-tunnel-token"
    assert env["ACS_APEX_CALLBACK_TOKEN"] == "my-callback-token"


def test_bundle_name_passed_through() -> None:
    env = _build(bundle_name="flux_dev_t2i", bundle_version="260201-01")
    assert env["ACS_BUNDLE"] == "flux_dev_t2i"
    assert env["ACS_BUNDLE_VERSION"] == "260201-01"


def test_default_port_8188_mapping_present() -> None:
    env = _build(comfyui_port=DEFAULT_COMFYUI_PORT)
    assert env["ACS_COMFYUI_PORT"] == str(DEFAULT_COMFYUI_PORT)
    assert f"-p {DEFAULT_COMFYUI_PORT}:{DEFAULT_COMFYUI_PORT}" in env
    assert env[f"-p {DEFAULT_COMFYUI_PORT}:{DEFAULT_COMFYUI_PORT}"] == "1"


def test_cf_tunnel_token_present_and_unprefixed() -> None:
    """The portal-mandated CF_TUNNEL_TOKEN must be present, unprefixed, and equal
    to ACS_CF_TUNNEL_TOKEN."""
    env = _build(tunnel_token="test-tunnel-secret")
    assert env["CF_TUNNEL_TOKEN"] == env["ACS_CF_TUNNEL_TOKEN"] == "test-tunnel-secret"
