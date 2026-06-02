"""Tests for provisioning-related Settings fields introduced in the 2026-05-12 fix."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.config import Settings

_DEFAULT_COMFYUI_PORT: int = Settings.model_fields["comfyui_port"].default


def _base_settings(**overrides: object) -> Settings:
    return Settings(
        comfyui_host="127.0.0.1",
        comfyui_port=_DEFAULT_COMFYUI_PORT,
        **overrides,  # type: ignore[arg-type]
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
        s = _base_settings()
        assert s.gpu_provision_timeout_seconds == 2000

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
        s = Settings(comfyui_host="127.0.0.1", comfyui_port=_DEFAULT_COMFYUI_PORT)
        assert s.gpu_provision_timeout_seconds == 500
