"""Unit tests for BundleIndexService — parsing and resolution logic only.

Git operations (clone/pull) require network access and are NOT tested here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.api.services.bundle_index import BundleIndexService, BundleNotFoundError
from src.core.bundle_config import BundleMapping, HardwareRequirements

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HW_YAML: dict = {
    "gpu_whitelist": ["RTX_4090", "A100_SXM4"],
    "min_disk_gb": 150,
    "min_network_upload_mbps": 100,
    "min_network_download_mbps": 500,
    "cuda_min_version": "12.1",
    "num_gpus": 1,
    "comfyui_port": 18188,
}


def _make_service(cache_dir: Path) -> BundleIndexService:
    return BundleIndexService(
        repo_url="https://github.com/gearbox/ai-bundles.git",
        github_token="ghp_test",
        sync_interval_minutes=15,
        cache_dir=cache_dir,
    )


def _write_bundle_yaml(bundle_path: Path, hw: dict | None = None) -> None:
    """Create {bundle_path}/current/bundle.yaml with the given hardware section."""
    current_dir = bundle_path / "current"
    current_dir.mkdir(parents=True, exist_ok=True)
    data: dict = {"hardware": hw if hw is not None else _HW_YAML}
    (current_dir / "bundle.yaml").write_text(yaml.dump(data))


def _write_index(cache_dir: Path, entries: list[dict]) -> None:
    (cache_dir / "bundle-index.yaml").write_text(yaml.dump({"bundles": entries}))


# ---------------------------------------------------------------------------
# Fixture: populated fake repo
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Populate tmp_path as a minimal ai-bundles repo."""
    _write_bundle_yaml(tmp_path / "bundles" / "wan_2_i2v")
    _write_bundle_yaml(tmp_path / "bundles" / "wan_2_t2v")
    _write_index(
        tmp_path,
        [
            {
                "name": "wan_2_i2v",
                "path": "bundles/wan_2_i2v",
                "model_type": "aisha-i2v",
                "default_bundle": True,
            },
            {
                "name": "wan_2_t2v",
                "path": "bundles/wan_2_t2v",
                "model_type": "aisha-t2v",
                "default_bundle": True,
            },
        ],
    )
    return tmp_path


# ---------------------------------------------------------------------------
# _parse_index / resolve_bundle
# ---------------------------------------------------------------------------


class TestParseIndex:
    def test_parses_default_bundles(self, fake_repo: Path) -> None:
        svc = _make_service(fake_repo)
        svc._parse_index()

        assert "aisha-i2v" in svc._model_index
        assert "aisha-t2v" in svc._model_index

    def test_parses_all_bundles_into_bundle_index(self, fake_repo: Path) -> None:
        svc = _make_service(fake_repo)
        svc._parse_index()

        assert "wan_2_i2v" in svc._bundle_index
        assert "wan_2_t2v" in svc._bundle_index

    def test_non_default_bundle_excluded_from_model_index(self, tmp_path: Path) -> None:
        _write_bundle_yaml(tmp_path / "bundles" / "exp_bundle")
        _write_index(
            tmp_path,
            [
                {
                    "name": "exp_bundle",
                    "path": "bundles/exp_bundle",
                    "model_type": "aisha-exp",
                    "default_bundle": False,
                }
            ],
        )
        svc = _make_service(tmp_path)
        svc._parse_index()

        assert "aisha-exp" not in svc._model_index
        assert "exp_bundle" in svc._bundle_index

    def test_missing_index_file_is_noop(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path)
        svc._parse_index()  # should not raise
        assert svc._model_index == {}
        assert svc._bundle_index == {}


class TestResolveBundle:
    def test_returns_correct_mapping(self, fake_repo: Path) -> None:
        svc = _make_service(fake_repo)
        svc._parse_index()

        mapping = svc.resolve_bundle("aisha-i2v")

        assert isinstance(mapping, BundleMapping)
        assert mapping.bundle_name == "wan_2_i2v"
        assert mapping.bundle_version is None
        assert mapping.hardware.min_disk_gb == 150
        assert mapping.hardware.gpu_whitelist == ("RTX_4090", "A100_SXM4")

    def test_raises_for_unknown_model_type(self, fake_repo: Path) -> None:
        svc = _make_service(fake_repo)
        svc._parse_index()

        with pytest.raises(BundleNotFoundError, match="no_such_type"):
            svc.resolve_bundle("no_such_type")

    def test_raises_when_index_empty(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path)
        # No sync, index is empty

        with pytest.raises(BundleNotFoundError):
            svc.resolve_bundle("aisha-i2v")


# ---------------------------------------------------------------------------
# resolve_bundle_override
# ---------------------------------------------------------------------------


class TestResolveBundleOverride:
    def test_name_only_returns_mapping_with_none_version(self, fake_repo: Path) -> None:
        svc = _make_service(fake_repo)
        svc._parse_index()

        mapping = svc.resolve_bundle_override("wan_2_i2v")
        assert mapping.bundle_name == "wan_2_i2v"
        assert mapping.bundle_version is None

    def test_name_colon_version_returns_pinned_version(self, fake_repo: Path) -> None:
        svc = _make_service(fake_repo)
        svc._parse_index()

        mapping = svc.resolve_bundle_override("wan_2_i2v:260105-01")
        assert mapping.bundle_name == "wan_2_i2v"
        assert mapping.bundle_version == "260105-01"

    def test_version_override_preserves_hardware(self, fake_repo: Path) -> None:
        svc = _make_service(fake_repo)
        svc._parse_index()

        mapping = svc.resolve_bundle_override("wan_2_i2v:260105-01")
        assert mapping.hardware.min_disk_gb == 150

    def test_raises_for_unknown_bundle_name(self, fake_repo: Path) -> None:
        svc = _make_service(fake_repo)
        svc._parse_index()

        with pytest.raises(BundleNotFoundError, match="no_such_bundle"):
            svc.resolve_bundle_override("no_such_bundle")

    def test_raises_for_unknown_bundle_with_version(self, fake_repo: Path) -> None:
        svc = _make_service(fake_repo)
        svc._parse_index()

        with pytest.raises(BundleNotFoundError):
            svc.resolve_bundle_override("no_such_bundle:260101-01")


# ---------------------------------------------------------------------------
# _parse_hardware
# ---------------------------------------------------------------------------


class TestParseHardware:
    def test_extracts_all_fields(self, tmp_path: Path) -> None:
        hw = {
            "gpu_whitelist": ["RTX_3090", "RTX_4090"],
            "min_disk_gb": 200,
            "min_network_upload_mbps": 50,
            "min_network_download_mbps": 1000,
            "cuda_min_version": "12.4",
            "num_gpus": 2,
            "comfyui_port": 8188,
        }
        _write_bundle_yaml(tmp_path / "my_bundle", hw)

        svc = _make_service(tmp_path)
        result: HardwareRequirements = svc._parse_hardware(tmp_path / "my_bundle")

        assert result.gpu_whitelist == ("RTX_3090", "RTX_4090")
        assert result.min_disk_gb == 200
        assert result.min_network_upload_mbps == 50
        assert result.min_network_download_mbps == 1000
        assert result.cuda_min_version == "12.4"
        assert result.num_gpus == 2
        assert result.comfyui_port == 8188

    def test_default_comfyui_port(self, tmp_path: Path) -> None:
        hw = {k: v for k, v in _HW_YAML.items() if k != "comfyui_port"}
        _write_bundle_yaml(tmp_path / "my_bundle", hw)

        svc = _make_service(tmp_path)
        result = svc._parse_hardware(tmp_path / "my_bundle")
        assert result.comfyui_port == 18188

    def test_raises_on_missing_hardware_section(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "bad_bundle" / "current"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "bundle.yaml").write_text(yaml.dump({"name": "bad"}))

        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="hardware"):
            svc._parse_hardware(tmp_path / "bad_bundle")

    def test_raises_on_missing_bundle_yaml(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "empty_bundle"
        bundle_dir.mkdir()

        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="bundle.yaml"):
            svc._parse_hardware(bundle_dir)


# ---------------------------------------------------------------------------
# _build_repo_url_with_token
# ---------------------------------------------------------------------------


class TestBuildRepoUrl:
    def test_injects_token(self) -> None:
        svc = BundleIndexService(
            repo_url="https://github.com/gearbox/ai-bundles.git",
            github_token="ghp_abc123",
            sync_interval_minutes=15,
        )
        url = svc._build_repo_url_with_token()
        assert url == "https://ghp_abc123@github.com/gearbox/ai-bundles.git"

    def test_no_token_returns_original_url(self) -> None:
        svc = BundleIndexService(
            repo_url="https://github.com/gearbox/ai-bundles.git",
            github_token="",
            sync_interval_minutes=15,
        )
        url = svc._build_repo_url_with_token()
        assert url == "https://github.com/gearbox/ai-bundles.git"
