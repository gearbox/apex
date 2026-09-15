"""Tests for provisioning-related Settings fields introduced in the 2026-05-12 fix."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from src.core.config import Settings
from tests.unit.helpers import hermetic_settings

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_COMFYUI_PORT: int = Settings.model_fields["comfyui_port"].default


def _base_settings(**overrides: object) -> Settings:
    """Hermetic against a local .env file (S4) — see tests.unit.helpers.hermetic_settings."""
    return hermetic_settings(
        comfyui_host="127.0.0.1",
        comfyui_port=_DEFAULT_COMFYUI_PORT,
        **overrides,
    )


class TestProvisioningOfferWalkDepth:
    def test_default_is_10(self) -> None:
        s = _base_settings()
        assert s.provisioning_offer_walk_depth == 10

    def test_custom_value(self) -> None:
        s = _base_settings(provisioning_offer_walk_depth=5)
        assert s.provisioning_offer_walk_depth == 5

    def test_zero_is_invalid(self) -> None:
        with pytest.raises(ValidationError):
            _base_settings(provisioning_offer_walk_depth=0)

    def test_above_max_is_invalid(self) -> None:
        with pytest.raises(ValidationError):
            _base_settings(provisioning_offer_walk_depth=21)

    def test_boundary_min_1(self) -> None:
        s = _base_settings(provisioning_offer_walk_depth=1)
        assert s.provisioning_offer_walk_depth == 1

    def test_boundary_max_20(self) -> None:
        s = _base_settings(provisioning_offer_walk_depth=20)
        assert s.provisioning_offer_walk_depth == 20


class TestProvisioningRecreationAttempts:
    def test_default_is_1_no_autonomous_retry(self) -> None:
        """Policy: no autonomous retry by default."""
        s = _base_settings()
        assert s.provisioning_recreation_attempts == 1

    def test_custom_value_3(self) -> None:
        s = _base_settings(provisioning_recreation_attempts=3)
        assert s.provisioning_recreation_attempts == 3

    def test_zero_is_invalid(self) -> None:
        with pytest.raises(ValidationError):
            _base_settings(provisioning_recreation_attempts=0)

    def test_above_max_is_invalid(self) -> None:
        with pytest.raises(ValidationError):
            _base_settings(provisioning_recreation_attempts=4)

    def test_boundary_min_1(self) -> None:
        s = _base_settings(provisioning_recreation_attempts=1)
        assert s.provisioning_recreation_attempts == 1

    def test_boundary_max_3(self) -> None:
        s = _base_settings(provisioning_recreation_attempts=3)
        assert s.provisioning_recreation_attempts == 3


class TestOldSettingRemoved:
    def test_max_node_provisioning_retries_no_longer_exists(self) -> None:
        """max_node_provisioning_retries was removed; setting it must have no effect."""
        s = _base_settings()
        assert not hasattr(s, "max_node_provisioning_retries")


class TestGpuProvisionTimeoutSeconds:
    def test_default_is_2000(self) -> None:
        assert Settings.model_fields["gpu_provision_timeout_seconds"].default == 2000

    def test_custom_value(self) -> None:
        s = _base_settings(gpu_provision_timeout_seconds=500)
        assert s.gpu_provision_timeout_seconds == 500

    def test_below_min_is_invalid(self) -> None:
        with pytest.raises(ValidationError):
            _base_settings(gpu_provision_timeout_seconds=299)

    def test_boundary_min_300(self) -> None:
        s = _base_settings(gpu_provision_timeout_seconds=300)
        assert s.gpu_provision_timeout_seconds == 300

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GPU_PROVISION_TIMEOUT_SECONDS", "500")
        s = _base_settings()
        assert s.gpu_provision_timeout_seconds == 500


class TestGpuProvisionTerminalGraceProbes:
    def test_default_is_3(self) -> None:
        assert Settings.model_fields["gpu_provision_terminal_grace_probes"].default == 3

    def test_custom_value(self) -> None:
        s = _base_settings(gpu_provision_terminal_grace_probes=5)
        assert s.gpu_provision_terminal_grace_probes == 5

    def test_zero_is_invalid(self) -> None:
        with pytest.raises(ValidationError):
            _base_settings(gpu_provision_terminal_grace_probes=0)


class TestProvisioningScriptSettings:
    def test_ref_defaults_empty(self) -> None:
        s = _base_settings()
        assert s.provisioning_script_ref == ""

    def test_ref_custom_value(self) -> None:
        s = _base_settings(
            provisioning_script_ref="v1.2.3",
            apex_callback_url="https://apex.example.test",
        )
        assert s.provisioning_script_ref == "v1.2.3"

    def test_dev_ref_defaults_none(self) -> None:
        s = _base_settings()
        assert s.provisioning_script_dev_ref is None

    def test_dev_ref_empty_or_whitespace_is_none(self) -> None:
        assert _base_settings(provisioning_script_dev_ref="").provisioning_script_dev_ref is None
        assert _base_settings(provisioning_script_dev_ref="  ").provisioning_script_dev_ref is None

    def test_empty_callback_url_is_rejected_when_script_delivery_is_configured(self) -> None:
        with pytest.raises(ValidationError, match="apex_callback_url"):
            _base_settings(provisioning_script_ref="v1.2.3", apex_callback_url="")

    @pytest.mark.parametrize("value", ["apex.example.test", "https://apex.example.test/callback"])
    def test_callback_url_must_be_an_absolute_origin(self, value: str) -> None:
        with pytest.raises(ValidationError, match="apex_callback_url"):
            _base_settings(provisioning_script_ref="v1.2.3", apex_callback_url=value)

    def test_callback_url_is_normalized_to_an_origin(self) -> None:
        with_slash = _base_settings(
            provisioning_script_ref="v1.2.3", apex_callback_url="https://apex.example.test/"
        )
        without_slash = _base_settings(
            provisioning_script_ref="v1.2.3", apex_callback_url="https://apex.example.test"
        )
        assert with_slash.apex_callback_url == without_slash.apex_callback_url

    def test_malformed_callback_url_rejected_even_with_no_script_ref(self) -> None:
        """S6: the URL check must not be gated on the ref being set — apex_callback_url
        is used by every node callback, not just bootstrap-script delivery. Previously
        this was accepted (and left unnormalized) at startup because the whole
        validator returned early when provisioning_script_ref was empty."""
        with pytest.raises(ValidationError, match="apex_callback_url"):
            _base_settings(provisioning_script_ref="", apex_callback_url="not-a-url")

    def test_callback_url_alone_is_still_normalized_with_no_script_ref(self) -> None:
        s = _base_settings(
            provisioning_script_ref="", apex_callback_url="https://apex.example.test/"
        )
        assert s.apex_callback_url == "https://apex.example.test"

    def test_both_empty_is_still_accepted_for_local_dev(self) -> None:
        s = _base_settings(provisioning_script_ref="", apex_callback_url="")
        assert s.provisioning_script_ref == ""
        assert s.apex_callback_url == ""

    def test_ref_validity_does_not_depend_on_callback_url(self) -> None:
        """The ref check must fire on its own, independent of apex_callback_url."""
        with pytest.raises(ValidationError, match="provisioning_script_ref"):
            _base_settings(provisioning_script_ref="not-a-valid-ref", apex_callback_url="")

    def test_production_rejects_mutable_script_ref(self) -> None:
        with pytest.raises(ValidationError, match="provisioning_script_ref"):
            _base_settings(
                environment="production",
                provisioning_script_ref="main",
                provisioning_script_dev_ref="main",
                apex_callback_url="https://apex.example.test",
            )

    def test_staging_accepts_its_explicit_dev_ref(self) -> None:
        settings = _base_settings(
            environment="staging",
            provisioning_script_ref="main",
            provisioning_script_dev_ref="main",
            apex_callback_url="https://apex.example.test",
        )
        assert settings.provisioning_script_ref == "main"

    def test_cache_ttl_defaults(self) -> None:
        s = _base_settings()
        assert s.provisioning_script_cache_ttl_seconds == 86400
        assert s.provisioning_script_dev_cache_ttl_seconds == 60

    def test_rate_limit_default(self) -> None:
        s = _base_settings()
        assert s.rate_limit_provisioning_script == "120/minute"

    def test_defaults_are_hermetic_against_a_local_env_file(self, tmp_path: Path) -> None:
        """S4 regression: `.env.example` leaves these blank, so this only passed by
        accident before — a developer's real `.env` setting PROVISIONING_SCRIPT_REF
        would silently change what "default" means for every test in this class."""
        dotenv = tmp_path / ".env"
        dotenv.write_text(
            "PROVISIONING_SCRIPT_REF=v1.2.3\nAPEX_CALLBACK_URL=https://from-dotenv.test\n"
        )

        # A bare Settings() honoring that file would resolve non-empty here...
        loaded = Settings(
            _env_file=str(dotenv),  # pyright: ignore[reportCallIssue]
            comfyui_host="127.0.0.1",
            comfyui_port=_DEFAULT_COMFYUI_PORT,
        )
        assert loaded.provisioning_script_ref == "v1.2.3"

        # ...but _base_settings (hermetic_settings under the hood) never sees it.
        s = _base_settings()
        assert s.provisioning_script_ref == ""
