"""Tests for BundleIndexService.get_generation_config()."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from src.api.services.bundle_index import BundleIndexService
from src.core.enums import Resolution, Sampler, Scheduler
from src.core.generation_config import BundleConfigError

_TEST_CAPS: dict[str, Any] = {
    "default_comfyui_port": 18188,
    "max_download_bytes": 50 * 1024 * 1024,
    "max_member_count": 5000,
    "max_member_size_bytes": 10 * 1024 * 1024,
    "max_uncompressed_bytes": 100 * 1024 * 1024,
}

_FULL_GENERATION_BLOCK: dict[str, Any] = {
    "defaults": {
        "resolution": "standard",
        "steps": 12,
        "cfg": 1.1,
        "sampler": "euler",
        "scheduler": "beta",
        "denoise": 1.0,
    },
    "constraints": {
        "max_megapixels": 1.05,
        "latent_multiple": 16,
        "max_edge": 1536,
        "min_steps": 1,
        "max_steps": 30,
        "min_cfg": 0.0,
        "max_cfg": 15.0,
        "allowed_samplers": [],
        "allowed_schedulers": [],
    },
}


def _make_bundle_dir(
    tmp_path: Path,
    bundle_name: str = "test_bundle",
    version: str = "260101-01",
    *,
    generation_block: dict[str, Any] | None | str = None,
    extra_hardware: dict[str, Any] | None = None,
) -> tuple[Path, Path, BundleIndexService]:
    """Create a temp bundle tree and a BundleIndexService pointing at it.

    generation_block=None → omit 'generation' key (backward-compat test)
    generation_block=<dict> → include it
    generation_block="MISSING_FILE" → don't create bundle.yaml at all
    """
    cache_dir = tmp_path / "cache"
    bundle_path = cache_dir / "bundles" / bundle_name
    version_dir = bundle_path / version
    version_dir.mkdir(parents=True)
    (bundle_path / "current").symlink_to(version)

    if generation_block != "MISSING_FILE":
        hw: dict[str, Any] = {
            "gpu_whitelist": [],
            "min_disk_gb": 100,
            "min_network_upload_mbps": 100,
            "min_network_download_mbps": 100,
            "cuda_min_version": "12.1",
            "num_gpus": 1,
            **(extra_hardware or {}),
        }
        bundle_data: dict[str, Any] = {"hardware": hw}
        if generation_block is not None:
            bundle_data["generation"] = generation_block
        (version_dir / "bundle.yaml").write_text(yaml.dump(bundle_data))

    # Build a minimal index yaml so _parse_index can find the bundle
    index_yaml = cache_dir / "bundle-index.yaml"
    index_yaml.write_text(
        yaml.dump(
            {
                "bundles": [
                    {
                        "name": bundle_name,
                        "path": f"bundles/{bundle_name}",
                        "model_type": "aisha-image",
                        "default_bundle": True,
                    }
                ]
            }
        )
    )

    # Build a mock Settings with sensible fallbacks
    mock_settings = MagicMock()
    mock_settings.default_steps = 12
    mock_settings.max_steps = 20
    mock_settings.default_cfg = 1.1
    mock_settings.default_sampler = "euler"
    mock_settings.default_scheduler = "beta"
    mock_settings.default_denoise = 1.0
    mock_settings.default_resolution = "standard"

    svc = BundleIndexService(
        repo_url="https://github.com/gearbox/ai-bundles.git",
        github_token="",
        branch="master",
        sync_interval_minutes=15,
        cache_dir=cache_dir,
        settings=mock_settings,
        **_TEST_CAPS,
    )
    # Manually trigger index parse without fetching
    svc._parse_index()
    return bundle_path, version_dir, svc


class TestGetGenerationConfigFullBlock:
    def test_full_block_parses_correctly(self, tmp_path: Path) -> None:
        _, _, svc = _make_bundle_dir(tmp_path, generation_block=_FULL_GENERATION_BLOCK)
        cfg = svc.get_generation_config("test_bundle", "260101-01")

        assert cfg.defaults.resolution == Resolution.STANDARD
        assert cfg.defaults.steps == 12
        assert cfg.defaults.cfg == pytest.approx(1.1)
        assert cfg.defaults.sampler == Sampler.EULER
        assert cfg.defaults.scheduler == Scheduler.BETA
        assert cfg.defaults.denoise == pytest.approx(1.0)
        assert cfg.constraints.max_megapixels == pytest.approx(1.05)
        assert cfg.constraints.latent_multiple == 16
        assert cfg.constraints.max_edge == 1536
        assert cfg.constraints.min_steps == 1
        assert cfg.constraints.max_steps == 30
        assert cfg.constraints.min_cfg == pytest.approx(0.0)
        assert cfg.constraints.max_cfg == pytest.approx(15.0)
        assert cfg.constraints.allowed_samplers == frozenset()
        assert cfg.constraints.allowed_schedulers == frozenset()

    def test_allowed_samplers_parsed(self, tmp_path: Path) -> None:
        block = {
            **_FULL_GENERATION_BLOCK,
            "constraints": {
                **_FULL_GENERATION_BLOCK["constraints"],
                "allowed_samplers": ["euler", "dpmpp_2m"],
                "allowed_schedulers": ["beta", "karras"],
            },
        }
        _, _, svc = _make_bundle_dir(tmp_path, generation_block=block)
        cfg = svc.get_generation_config("test_bundle", "260101-01")
        assert cfg.constraints.allowed_samplers == frozenset({Sampler.EULER, Sampler.DPMPP_2M})
        assert cfg.constraints.allowed_schedulers == frozenset({Scheduler.BETA, Scheduler.KARRAS})

    def test_result_is_cached(self, tmp_path: Path) -> None:
        _, _, svc = _make_bundle_dir(tmp_path, generation_block=_FULL_GENERATION_BLOCK)
        cfg1 = svc.get_generation_config("test_bundle", "260101-01")
        cfg2 = svc.get_generation_config("test_bundle", "260101-01")
        assert cfg1 is cfg2


class TestGetGenerationConfigMissingBlock:
    def test_missing_generation_block_falls_back_to_settings(self, tmp_path: Path) -> None:
        # bundle.yaml with no 'generation' key
        _, _, svc = _make_bundle_dir(tmp_path, generation_block=None)
        cfg = svc.get_generation_config("test_bundle", "260101-01")

        # Should use Settings fallback values
        assert cfg.defaults.resolution == Resolution.STANDARD
        assert cfg.defaults.steps == 12
        assert cfg.constraints.max_megapixels == pytest.approx(1.0)
        assert cfg.constraints.allowed_samplers == frozenset()

    def test_missing_bundle_yaml_falls_back(self, tmp_path: Path) -> None:
        _, _, svc = _make_bundle_dir(tmp_path, generation_block="MISSING_FILE")
        cfg = svc.get_generation_config("test_bundle", "260101-01")
        assert cfg.defaults.steps == 12  # Settings fallback

    def test_unknown_bundle_falls_back(self, tmp_path: Path) -> None:
        _, _, svc = _make_bundle_dir(tmp_path, generation_block=None)
        cfg = svc.get_generation_config("nonexistent_bundle", None)
        assert cfg.defaults.steps == 12


class TestGetGenerationConfigPartialBlock:
    def test_partial_defaults_fall_back(self, tmp_path: Path) -> None:
        """Only steps in defaults; all other keys should use Settings fallback."""
        block: dict[str, Any] = {"defaults": {"steps": 20}, "constraints": {}}
        _, _, svc = _make_bundle_dir(tmp_path, generation_block=block)
        cfg = svc.get_generation_config("test_bundle", "260101-01")

        assert cfg.defaults.steps == 20
        assert cfg.defaults.cfg == pytest.approx(1.1)  # Settings fallback
        assert cfg.defaults.sampler == Sampler.EULER  # Settings fallback
        assert cfg.defaults.resolution == Resolution.STANDARD  # Settings fallback


class TestGetGenerationConfigInvalidEnum:
    def test_invalid_sampler_raises_bundle_config_error(self, tmp_path: Path) -> None:
        block: dict[str, Any] = {
            "defaults": {"sampler": "not_a_real_sampler"},
            "constraints": {},
        }
        _, _, svc = _make_bundle_dir(tmp_path, generation_block=block)
        with pytest.raises(BundleConfigError, match="not_a_real_sampler"):
            svc.get_generation_config("test_bundle", "260101-01")

    def test_invalid_scheduler_in_allowed_list_raises(self, tmp_path: Path) -> None:
        block: dict[str, Any] = {
            "defaults": {},
            "constraints": {"allowed_schedulers": ["beta", "totally_fake"]},
        }
        _, _, svc = _make_bundle_dir(tmp_path, generation_block=block)
        with pytest.raises(BundleConfigError, match="totally_fake"):
            svc.get_generation_config("test_bundle", "260101-01")
