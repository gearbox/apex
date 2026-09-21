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
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import pytest
from litestar import Litestar

from src.api.routes.provisioning import ProvisioningController
from src.api.services.gpu_session._env_builder import (
    build_acs_env,
    build_provisioning_callback_urls,
)
from src.core.config import Settings

# Single source of truth — stays in sync with Settings automatically.
_DEFAULT_COMFYUI_PORT: int = Settings.model_fields["comfyui_port"].default

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
    "ACS_APEX_OPERATION_ID",
    "ACS_APEX_CALLBACK_URL",
    "ACS_APEX_CALLBACK_TOKEN",
    "PROVISIONING_SCRIPT",
    "PROVISIONER_WEBHOOK_URL",
    "PROVISIONER_FAILURE_ACTION",
    "ACS_PROVISION_SCRIPT_SHA256",
    "ACS_HF_TOKEN",
    "ACS_CIVITAI_API_TOKEN",
    "ACS_COMFYUI_PORT",
    "ACS_COMFYUI_HOST",
    "ACS_COMFYUI_EXTRA_ARGS",
}


def _make_settings(**overrides: object) -> MagicMock:
    s = MagicMock()
    s.github_content_token = "ghp_test_token"
    s.ai_bundles_repo_url = "https://github.com/gearbox/ai-bundles.git"
    s.ai_bundles_branch = "master"
    s.aisha_repo_url = "https://github.com/gearbox/aisha.git"
    s.aisha_branch = "master"
    s.apex_callback_url = "https://apex.example.com"
    s.provisioning_script_ref = "v1.2.3"
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
    operation_id: UUID = overrides.pop("operation_id", uuid4())  # type: ignore[assignment]
    settings: Settings = overrides.pop("settings", _make_settings())  # type: ignore[assignment]
    return build_acs_env(
        settings=settings,
        session_id=session_id,
        operation_id=operation_id,
        bundle_name=overrides.pop("bundle_name", "wan_2.2_i2v"),  # type: ignore[arg-type]
        bundle_version=overrides.pop("bundle_version", "260105-01"),  # type: ignore[arg-type]
        comfyui_port=overrides.pop("comfyui_port", _DEFAULT_COMFYUI_PORT),  # type: ignore[arg-type]
        tunnel_token=overrides.pop("tunnel_token", "tunnel-secret"),  # type: ignore[arg-type]
        callback_token=overrides.pop("callback_token", "cb-secret"),  # type: ignore[arg-type]
        provision_script_sha256=overrides.pop(  # type: ignore[arg-type]
            "provision_script_sha256", "a" * 64
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_acs_keys_present() -> None:
    env = _build()
    for key in _EXPECTED_ACS_KEYS:
        assert key in env, f"Missing required key: {key}"


def test_acs_github_token_from_settings() -> None:
    s = _make_settings(github_content_token="ghp_my_pat")
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
    port = _DEFAULT_COMFYUI_PORT
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


def test_default_port_mapping_present() -> None:
    port = _DEFAULT_COMFYUI_PORT
    env = _build(comfyui_port=port)
    assert env["ACS_COMFYUI_PORT"] == str(port)
    assert f"-p {port}:{port}" in env
    assert env[f"-p {port}:{port}"] == "1"


def test_cf_tunnel_token_present_and_unprefixed() -> None:
    """The portal-mandated CF_TUNNEL_TOKEN must be present, unprefixed, and equal
    to ACS_CF_TUNNEL_TOKEN."""
    env = _build(tunnel_token="test-tunnel-secret")
    assert env["CF_TUNNEL_TOKEN"] == env["ACS_CF_TUNNEL_TOKEN"] == "test-tunnel-secret"


def test_provisioning_script_url_contains_ref_session_and_token() -> None:
    sid = uuid4()
    s = _make_settings(
        apex_callback_url="https://apex.example.com", provisioning_script_ref="v9.9.9"
    )
    env = _build(settings=s, session_id=sid, callback_token="tok-123")
    expected = (
        f"https://apex.example.com/v1/provisioning/scripts/comfyui/v9.9.9"
        f"?session={sid}&token=tok-123"
    )
    assert env["PROVISIONING_SCRIPT"] == expected


def test_provisioner_webhook_url_contains_session_and_token() -> None:
    sid = uuid4()
    s = _make_settings(apex_callback_url="https://apex.example.com")
    env = _build(settings=s, session_id=sid, callback_token="tok-456")
    assert env["PROVISIONER_WEBHOOK_URL"] == (
        f"https://apex.example.com/v1/provisioning/webhook/{sid}?token=tok-456"
    )


def test_provisioner_failure_action_is_always_destroy() -> None:
    env = _build()
    assert env["PROVISIONER_FAILURE_ACTION"] == "destroy"


def test_provision_script_sha256_passed_through() -> None:
    env = _build(provision_script_sha256="b" * 64)
    assert env["ACS_PROVISION_SCRIPT_SHA256"] == "b" * 64


# ---------------------------------------------------------------------------
# T3 (round-3 remediation): build_provisioning_callback_urls must never emit
# an unroutable script URL, regardless of who validated the ref upstream.
# ---------------------------------------------------------------------------


class TestBuildProvisioningCallbackUrlsRouteSafety:
    @pytest.mark.parametrize(
        "bad_ref", ["feature/bootstrap", "a?b", "a#b", "a%b", "a b", ".", ".."]
    )
    def test_unsafe_ref_raises(self, bad_ref: str) -> None:
        with pytest.raises(ValueError, match=r"provision_script_dev_ref|route-safe|path segment"):
            build_provisioning_callback_urls(
                apex_callback_url="https://apex.example.com",
                session_id=uuid4(),
                callback_token="tok",
                provision_script_ref=bad_ref,
            )

    def test_pinned_immutable_ref_is_never_checked_for_route_safety(self) -> None:
        """A release tag/SHA always passes PROVISIONING_REF_PATTERN, so the
        route-safety check is skipped for it entirely — it can never contain
        a '/' by construction of the pattern anyway."""
        script_url, _ = build_provisioning_callback_urls(
            apex_callback_url="https://apex.example.com",
            session_id=uuid4(),
            callback_token="tok",
            provision_script_ref="v1.2.3",
        )
        assert "/v1.2.3?" in script_url

    def test_slash_free_dev_ref_round_trips_through_the_registered_route(self) -> None:
        app = Litestar(route_handlers=[ProvisioningController])
        route = next(r for r in app.routes if "/scripts/" in r.path)
        expected_path = route.path.replace("{variant:str}", "comfyui").replace(
            "{ref:str}", "bootstrap-test"
        )

        script_url, _ = build_provisioning_callback_urls(
            apex_callback_url="https://apex.example.com",
            session_id=uuid4(),
            callback_token="tok",
            provision_script_ref="bootstrap-test",
        )

        assert urlsplit(script_url).path == expected_path
