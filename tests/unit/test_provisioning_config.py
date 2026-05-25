"""Tests for provisioning-related Settings fields introduced in the 2026-05-12 fix."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.config import Settings


def _base_settings(**overrides: object) -> Settings:
    return Settings(
        comfyui_host="127.0.0.1",
        comfyui_port=8188,
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
