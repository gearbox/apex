"""Unit tests for BundleIndexService — parsing, resolution, and HTTP fetch logic."""

from __future__ import annotations

import io
import tarfile as tarfile_module
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from pydantic import ValidationError

from src.api.services.bundle_index import (
    BundleIndexService,
    BundleNotFoundError,
    _parse_github_url,
)
from src.core.bundle_config import DEFAULT_COMFYUI_PORT, BundleMapping, ReadinessMarker
from src.core.config import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HW_YAML: dict[str, object] = {
    "gpu_whitelist": ["RTX_4090", "A100_SXM4"],
    "min_disk_gb": 150,
    "min_network_upload_mbps": 100,
    "min_network_download_mbps": 500,
    "cuda_min_version": "12.1",
    "num_gpus": 1,
    "comfyui_port": DEFAULT_COMFYUI_PORT,
}


_TEST_CAPS: dict[str, Any] = {
    "default_comfyui_port": 18188,
    "max_download_bytes": 10 * 1024 * 1024,  # 10 MB
    "max_member_count": 1000,
    "max_member_size_bytes": 5 * 1024 * 1024,
    "max_uncompressed_bytes": 50 * 1024 * 1024,
}


def _make_service(cache_dir: Path, **kwargs: Any) -> BundleIndexService:
    defaults: dict[str, Any] = {
        "repo_url": "https://github.com/gearbox/ai-bundles.git",
        "github_token": "ghp_test",
        "branch": "master",
        "sync_interval_minutes": 15,
        "cache_dir": cache_dir,
        **_TEST_CAPS,
    } | kwargs
    return BundleIndexService(**defaults)


def _write_bundle_yaml(
    bundle_path: Path,
    hw: dict[str, object] | None = None,
    *,
    readiness_marker: dict[str, object] | None = None,
) -> None:
    """Create {bundle_path}/current/bundle.yaml with the given hardware section."""
    current_dir = bundle_path / "current"
    current_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {"hardware": hw if hw is not None else _HW_YAML}
    if readiness_marker is not None:
        data["readiness_marker"] = readiness_marker
    (current_dir / "bundle.yaml").write_text(yaml.dump(data))


def _write_index(cache_dir: Path, entries: list[dict[str, object]]) -> None:
    (cache_dir / "bundle-index.yaml").write_text(yaml.dump({"bundles": entries}))


def _make_test_tarball(top_dir: str, files: dict[str, str]) -> bytes:
    """Build an in-memory .tar.gz containing `files` under `top_dir/`."""
    buf = io.BytesIO()
    with tarfile_module.open(fileobj=buf, mode="w:gz") as tf:
        for path, content in files.items():
            data = content.encode()
            info = tarfile_module.TarInfo(name=f"{top_dir}/{path}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


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
# _parse_github_url
# ---------------------------------------------------------------------------


class TestParseGithubUrl:
    def test_https_with_git_suffix(self) -> None:
        assert _parse_github_url("https://github.com/gearbox/ai-bundles.git") == (
            "gearbox",
            "ai-bundles",
        )

    def test_https_without_git_suffix(self) -> None:
        assert _parse_github_url("https://github.com/gearbox/ai-bundles") == (
            "gearbox",
            "ai-bundles",
        )

    def test_https_with_trailing_slash(self) -> None:
        assert _parse_github_url("https://github.com/gearbox/ai-bundles/") == (
            "gearbox",
            "ai-bundles",
        )

    def test_ssh_form_rejected(self) -> None:
        with pytest.raises(ValueError, match="SSH form is no longer supported"):
            _parse_github_url("git@github.com:gearbox/ai-bundles.git")

    def test_gitlab_rejected(self) -> None:
        with pytest.raises(ValueError, match="https://github.com/"):
            _parse_github_url("https://gitlab.com/gearbox/ai-bundles.git")

    def test_single_segment_rejected(self) -> None:
        with pytest.raises(ValueError, match="<owner>/<repo>"):
            _parse_github_url("https://github.com/just-one-segment")

    def test_empty_owner_rejected(self) -> None:
        with pytest.raises(ValueError, match="<owner>/<repo>"):
            _parse_github_url("https://github.com//empty-owner")

    def test_dash_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="https://github.com/"):
            _parse_github_url("-not-a-url")


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

    def test_empty_index_file_is_noop(self, tmp_path: Path) -> None:
        """Empty YAML file → safe_load returns None, must not crash."""
        (tmp_path / "bundle-index.yaml").write_text("")
        svc = _make_service(tmp_path)
        svc._parse_index()  # must not raise
        assert svc._model_index == {}
        assert svc._bundle_index == {}

    def test_non_dict_index_file_is_noop(self, tmp_path: Path) -> None:
        """YAML list at top level → not a dict, must not crash."""
        (tmp_path / "bundle-index.yaml").write_text("- this is a list\n- not a mapping\n")
        svc = _make_service(tmp_path)
        svc._parse_index()  # must not raise
        assert svc._model_index == {}
        assert svc._bundle_index == {}

    def test_mapping_without_bundles_key_is_noop(self, tmp_path: Path) -> None:
        """Valid mapping file without 'bundles' key — behaves like empty index."""
        (tmp_path / "bundle-index.yaml").write_text(yaml.dump({"something_else": 1}))
        svc = _make_service(tmp_path)
        svc._parse_index()  # must not raise
        assert svc._model_index == {}
        assert svc._bundle_index == {}

    def test_bundles_not_a_list_is_noop(self, tmp_path: Path) -> None:
        """If 'bundles' is a mapping instead of a list, don't crash."""
        (tmp_path / "bundle-index.yaml").write_text(
            yaml.dump({"bundles": {"not": "a list"}}),
        )
        svc = _make_service(tmp_path)
        svc._parse_index()
        assert svc._model_index == {}
        assert svc._bundle_index == {}

    def test_bad_bundle_entry_is_skipped(self, tmp_path: Path) -> None:
        """A single malformed entry must not poison the whole index."""
        _write_bundle_yaml(tmp_path / "bundles" / "good")
        # Missing 'path' field in second entry
        _write_index(
            tmp_path,
            [
                {
                    "name": "good",
                    "path": "bundles/good",
                    "model_type": "aisha-good",
                    "default_bundle": True,
                },
                {
                    "name": "bad",
                    "model_type": "aisha-bad",
                    "default_bundle": True,
                },  # missing 'path'
            ],
        )
        svc = _make_service(tmp_path)
        svc._parse_index()
        # Good bundle parsed, bad bundle skipped
        assert "aisha-good" in svc._model_index
        assert "aisha-bad" not in svc._model_index
        assert "good" in svc._bundle_index
        assert "bad" not in svc._bundle_index

    def test_bundle_with_invalid_hardware_is_skipped(self, tmp_path: Path) -> None:
        """If a bundle's bundle.yaml has invalid hardware, skip only that bundle."""
        _write_bundle_yaml(tmp_path / "bundles" / "good")
        # Write an invalid bundle.yaml for the second bundle
        bad_dir = tmp_path / "bundles" / "bad" / "current"
        bad_dir.mkdir(parents=True)
        (bad_dir / "bundle.yaml").write_text(yaml.dump({"name": "bad", "hardware": "garbage"}))
        _write_index(
            tmp_path,
            [
                {
                    "name": "good",
                    "path": "bundles/good",
                    "model_type": "aisha-good",
                    "default_bundle": True,
                },
                {
                    "name": "bad",
                    "path": "bundles/bad",
                    "model_type": "aisha-bad",
                    "default_bundle": True,
                },
            ],
        )
        svc = _make_service(tmp_path)
        svc._parse_index()
        assert "aisha-good" in svc._model_index
        assert "aisha-bad" not in svc._model_index

    def test_non_mapping_entry_is_skipped(self, tmp_path: Path) -> None:
        """A YAML list where entries are strings instead of mappings."""
        _write_bundle_yaml(tmp_path / "bundles" / "good")
        (tmp_path / "bundle-index.yaml").write_text(
            yaml.dump(
                {
                    "bundles": [
                        "not a dict",
                        {
                            "name": "good",
                            "path": "bundles/good",
                            "model_type": "aisha-good",
                            "default_bundle": True,
                        },
                    ]
                }
            )
        )
        svc = _make_service(tmp_path)
        svc._parse_index()
        assert "good" in svc._bundle_index


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
        result, _ = svc._parse_hardware(tmp_path / "my_bundle")

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
        result, _ = svc._parse_hardware(tmp_path / "my_bundle")
        assert result.comfyui_port == DEFAULT_COMFYUI_PORT

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

    def test_raises_with_field_name_on_missing_int_field(self, tmp_path: Path) -> None:
        """Missing required int field produces a message naming the field and file."""
        hw = {k: v for k, v in _HW_YAML.items() if k != "min_disk_gb"}
        _write_bundle_yaml(tmp_path / "bad", hw)
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="hardware.min_disk_gb"):
            svc._parse_hardware(tmp_path / "bad")

    def test_raises_on_non_integer_field_value(self, tmp_path: Path) -> None:
        """Non-coercible values for int fields produce a clear error."""
        hw = {**_HW_YAML, "min_disk_gb": "one hundred"}
        _write_bundle_yaml(tmp_path / "bad", hw)
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="min_disk_gb.*integer"):
            svc._parse_hardware(tmp_path / "bad")

    def test_rejects_float_for_int_field(self, tmp_path: Path) -> None:
        """YAML floats (e.g. 1.5, 42.0) must be rejected, not silently truncated.

        ``int(1.5)`` would produce ``1`` — silently accepting a misconfiguration.
        """
        hw = {**_HW_YAML, "min_disk_gb": 150.5}
        _write_bundle_yaml(tmp_path / "bad", hw)
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="min_disk_gb.*integer"):
            svc._parse_hardware(tmp_path / "bad")

    def test_rejects_decimal_string_for_int_field(self, tmp_path: Path) -> None:
        """String like '150.5' must be rejected (int('150.5') also raises)."""
        hw = {**_HW_YAML, "min_disk_gb": "150.5"}
        _write_bundle_yaml(tmp_path / "bad", hw)
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="min_disk_gb.*integer"):
            svc._parse_hardware(tmp_path / "bad")

    def test_rejects_bool_for_int_field(self, tmp_path: Path) -> None:
        """``bool`` is an ``int`` subclass — must be explicitly rejected."""
        hw = {**_HW_YAML, "num_gpus": True}
        _write_bundle_yaml(tmp_path / "bad", hw)
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="num_gpus.*bool"):
            svc._parse_hardware(tmp_path / "bad")

    def test_accepts_integer_valued_string(self, tmp_path: Path) -> None:
        """``'100'`` is legitimate — YAML sometimes quotes numbers. Must accept."""
        hw = {**_HW_YAML, "min_disk_gb": "100"}
        _write_bundle_yaml(tmp_path / "ok", hw)
        svc = _make_service(tmp_path)
        result, _ = svc._parse_hardware(tmp_path / "ok")
        assert result.min_disk_gb == 100

    def test_raises_on_invalid_cuda_min_version(self, tmp_path: Path) -> None:
        """Non-numeric cuda_min_version is rejected at parse time (upstream of VastAI)."""
        hw = {**_HW_YAML, "cuda_min_version": "12.1+"}
        _write_bundle_yaml(tmp_path / "bad", hw)
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="cuda_min_version"):
            svc._parse_hardware(tmp_path / "bad")

    def test_raises_on_non_list_gpu_whitelist(self, tmp_path: Path) -> None:
        hw = {**_HW_YAML, "gpu_whitelist": "RTX_4090"}  # string instead of list
        _write_bundle_yaml(tmp_path / "bad", hw)
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="gpu_whitelist.*list"):
            svc._parse_hardware(tmp_path / "bad")

    def test_raises_on_non_string_gpu_whitelist_entry(self, tmp_path: Path) -> None:
        """YAML loading ``[4090]`` unquoted produces an int — must be rejected."""
        hw = {**_HW_YAML, "gpu_whitelist": ["RTX_4090", 4090, "A100_SXM4"]}
        _write_bundle_yaml(tmp_path / "bad", hw)
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match=r"gpu_whitelist\[1\].*string"):
            svc._parse_hardware(tmp_path / "bad")

    def test_raises_on_non_integer_comfyui_port(self, tmp_path: Path) -> None:
        """comfyui_port must go through _require_int so errors are field-specific."""
        hw = {**_HW_YAML, "comfyui_port": "not a number"}
        _write_bundle_yaml(tmp_path / "bad", hw)
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="hardware.comfyui_port.*integer"):
            svc._parse_hardware(tmp_path / "bad")

    def test_comfyui_port_default_applied_when_missing(self, tmp_path: Path) -> None:
        """Missing comfyui_port uses the default 8188, still goes through _require_int."""
        hw = {k: v for k, v in _HW_YAML.items() if k != "comfyui_port"}
        _write_bundle_yaml(tmp_path / "no_port", hw)
        svc = _make_service(tmp_path)
        result, _ = svc._parse_hardware(tmp_path / "no_port")
        assert result.comfyui_port == DEFAULT_COMFYUI_PORT

    def test_template_hash_id_when_set_string(self, tmp_path: Path) -> None:
        hw = {**_HW_YAML, "template_hash_id": "4e17788f74f075dd9aab7d0d4427968f"}
        _write_bundle_yaml(tmp_path / "b", hw)
        svc = _make_service(tmp_path)
        result, _ = svc._parse_hardware(tmp_path / "b")
        assert result.template_hash_id == "4e17788f74f075dd9aab7d0d4427968f"

    def test_template_hash_id_absent_returns_none(self, tmp_path: Path) -> None:
        _write_bundle_yaml(tmp_path / "b", dict(_HW_YAML))
        svc = _make_service(tmp_path)
        result, _ = svc._parse_hardware(tmp_path / "b")
        assert result.template_hash_id is None

    def test_template_hash_id_empty_string_rejected(self, tmp_path: Path) -> None:
        hw = {**_HW_YAML, "template_hash_id": ""}
        _write_bundle_yaml(tmp_path / "b", hw)
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="template_hash_id.*non-empty"):
            svc._parse_hardware(tmp_path / "b")

    def test_template_hash_id_whitespace_only_rejected(self, tmp_path: Path) -> None:
        hw = {**_HW_YAML, "template_hash_id": "   "}
        _write_bundle_yaml(tmp_path / "b", hw)
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="template_hash_id.*non-empty"):
            svc._parse_hardware(tmp_path / "b")

    def test_template_hash_id_non_string_rejected(self, tmp_path: Path) -> None:
        hw = {**_HW_YAML, "template_hash_id": 12345}
        _write_bundle_yaml(tmp_path / "b", hw)
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="template_hash_id.*string"):
            svc._parse_hardware(tmp_path / "b")


# ---------------------------------------------------------------------------
# _parse_readiness_marker / TestParseReadinessMarker
# ---------------------------------------------------------------------------


class TestParseReadinessMarker:
    def _write(
        self,
        bundle_path: Path,
        hw: dict[str, object] | None = None,
        *,
        readiness_marker: dict[str, object] | None = None,
    ) -> None:
        _write_bundle_yaml(bundle_path, hw, readiness_marker=readiness_marker)

    def test_marker_absent_returns_none(self, tmp_path: Path) -> None:
        self._write(tmp_path / "b")
        svc = _make_service(tmp_path)
        _, marker = svc._parse_hardware(tmp_path / "b")
        assert marker is None

    def test_marker_with_node_class_parsed(self, tmp_path: Path) -> None:
        self._write(
            tmp_path / "b",
            readiness_marker={"node_class": "WanImageToVideo"},
        )
        svc = _make_service(tmp_path)
        _, marker = svc._parse_hardware(tmp_path / "b")
        assert isinstance(marker, ReadinessMarker)
        assert marker.node_class == "WanImageToVideo"

    def test_marker_without_node_class_raises(self, tmp_path: Path) -> None:
        self._write(tmp_path / "b", readiness_marker={})
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="node_class"):
            svc._parse_hardware(tmp_path / "b")

    def test_marker_node_class_empty_raises(self, tmp_path: Path) -> None:
        self._write(tmp_path / "b", readiness_marker={"node_class": ""})
        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="node_class"):
            svc._parse_hardware(tmp_path / "b")

    def test_marker_not_a_mapping_raises(self, tmp_path: Path) -> None:
        self._write(tmp_path / "b", readiness_marker=None)
        # Write the YAML manually so readiness_marker is a bare string
        current_dir = tmp_path / "b" / "current"
        current_dir.mkdir(parents=True, exist_ok=True)
        data: dict[str, object] = {
            "hardware": dict(_HW_YAML),
            "readiness_marker": "WanImageToVideo",
        }
        (current_dir / "bundle.yaml").write_text(yaml.dump(data))

        svc = _make_service(tmp_path)
        with pytest.raises(ValueError, match="must be a mapping"):
            svc._parse_hardware(tmp_path / "b")

    def test_bundle_mapping_carries_readiness_marker(self, tmp_path: Path) -> None:
        """BundleMapping produced by _build_entry includes the marker."""
        _write_bundle_yaml(
            tmp_path / "bundles" / "wan_2_i2v",
            readiness_marker={"node_class": "WanImageToVideo"},
        )
        _write_index(
            tmp_path,
            [
                {
                    "name": "wan_2_i2v",
                    "path": "bundles/wan_2_i2v",
                    "model_type": "aisha-i2v",
                    "default_bundle": True,
                }
            ],
        )
        svc = _make_service(tmp_path)
        svc._parse_index()
        mapping = svc.resolve_bundle("aisha-i2v")
        assert isinstance(mapping.readiness_marker, ReadinessMarker)
        assert mapping.readiness_marker.node_class == "WanImageToVideo"


# ---------------------------------------------------------------------------
# start() / sync() error behavior
# ---------------------------------------------------------------------------


class TestSyncErrorBehavior:
    async def test_start_raises_on_initial_sync_failure(self, tmp_path: Path) -> None:
        """If initial sync fails, start() must re-raise and not leave the service running."""
        svc = _make_service(tmp_path)

        async def boom() -> None:
            raise RuntimeError("fetch failed")

        svc._fetch_and_extract = boom  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="fetch failed"):
            await svc.start()

        assert svc._running is False
        assert svc._sync_task is None

    async def test_sync_swallows_errors(self, tmp_path: Path) -> None:
        """The public sync() method logs errors but does not raise (background-safe)."""
        svc = _make_service(tmp_path)

        async def boom() -> None:
            raise RuntimeError("transient fetch failure")

        svc._fetch_and_extract = boom  # type: ignore[method-assign]

        # Must not raise — background loop depends on this
        await svc.sync()

    async def test_redact_token_removes_pat(self, tmp_path: Path) -> None:
        """If the PAT appears in an error body, _redact_token must remove it."""
        svc = _make_service(tmp_path, github_token="ghp_secret_abc")
        redacted = svc._redact_token("fatal: unable to access 'https://ghp_secret_abc@github.com/'")
        assert "ghp_secret_abc" not in redacted
        assert "***" in redacted

    async def test_redact_token_handles_empty_or_missing_token(self, tmp_path: Path) -> None:
        """Empty token or empty text must be a noop (no traceback)."""
        svc = _make_service(tmp_path, github_token="")
        assert svc._redact_token("any text") == "any text"
        assert svc._redact_token("") == ""


# ---------------------------------------------------------------------------
# _fetch_and_extract
# ---------------------------------------------------------------------------


def _make_mock_transport(
    status_code: int,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> httpx.MockTransport:
    """Build an httpx.MockTransport that returns a single fixed response."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            content=body,
            headers=headers or {},
        )

    return httpx.MockTransport(handler)


class TestFetchAndExtract:
    def _svc_with_transport(
        self, tmp_path: Path, transport: httpx.MockTransport
    ) -> BundleIndexService:
        client = httpx.AsyncClient(transport=transport)
        return _make_service(tmp_path, http_client=client)

    async def test_fetches_and_extracts_on_first_sync(self, tmp_path: Path) -> None:
        """200 response: cache_dir populated, etag set, auth header sent."""
        tarball = _make_test_tarball(
            "gearbox-ai-bundles-abc123",
            {"bundle-index.yaml": "bundles: []"},
        )
        received_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            received_headers.update(dict(request.headers))
            return httpx.Response(200, content=tarball, headers={"ETag": '"v1"'})

        transport = httpx.MockTransport(handler)
        svc = self._svc_with_transport(tmp_path, transport)
        await svc._fetch_and_extract()

        assert (tmp_path / "bundle-index.yaml").exists()
        assert svc._etag == '"v1"'
        assert received_headers.get("authorization") == "Bearer ghp_test"

    async def test_304_short_circuits_extraction(self, tmp_path: Path) -> None:
        """Second call with same ETag returns 304; cache_dir is untouched."""
        tarball = _make_test_tarball(
            "gearbox-ai-bundles-abc123",
            {"bundle-index.yaml": "bundles: []"},
        )
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if_none_match = request.headers.get("if-none-match")
            if if_none_match == '"v1"':
                return httpx.Response(304)
            return httpx.Response(200, content=tarball, headers={"ETag": '"v1"'})

        transport = httpx.MockTransport(handler)
        svc = self._svc_with_transport(tmp_path, transport)

        # First call: 200
        await svc._fetch_and_extract()
        assert svc._etag == '"v1"'
        first_mtime = (tmp_path / "bundle-index.yaml").stat().st_mtime

        # Second call: 304
        await svc._fetch_and_extract()
        assert call_count == 2
        assert (tmp_path / "bundle-index.yaml").stat().st_mtime == first_mtime

    async def test_strips_top_level_owner_repo_sha_directory(self, tmp_path: Path) -> None:
        """Files under gearbox-ai-bundles-abc123/ land at cache_dir root."""
        tarball = _make_test_tarball(
            "gearbox-ai-bundles-abc123",
            {"bundle-index.yaml": "bundles: []", "extra.txt": "hello"},
        )
        transport = _make_mock_transport(200, tarball, {"ETag": '"v1"'})
        svc = self._svc_with_transport(tmp_path, transport)
        await svc._fetch_and_extract()

        assert (tmp_path / "bundle-index.yaml").exists()
        assert (tmp_path / "extra.txt").exists()
        # No nested prefix directory
        nested = tmp_path / "gearbox-ai-bundles-abc123"
        assert not nested.exists()

    async def test_atomic_swap_on_extraction(self, tmp_path: Path) -> None:
        """Stale cache_dir content is replaced; staging and old dirs are cleaned up."""
        # Pre-populate cache_dir with stale content
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "stale.txt").write_text("old")

        tarball = _make_test_tarball(
            "gearbox-ai-bundles-abc123",
            {"bundle-index.yaml": "bundles: []"},
        )
        transport = _make_mock_transport(200, tarball, {"ETag": '"v1"'})
        svc = self._svc_with_transport(tmp_path, transport)
        await svc._fetch_and_extract()

        assert not (tmp_path / "stale.txt").exists()
        assert (tmp_path / "bundle-index.yaml").exists()
        # Staging and old dirs must be cleaned up
        assert not tmp_path.with_name(f"{tmp_path.name}.staging").exists()
        assert not tmp_path.with_name(f"{tmp_path.name}.old").exists()

    async def test_404_raises_with_clear_message(self, tmp_path: Path) -> None:
        """404 response raises RuntimeError mentioning the status code."""
        transport = _make_mock_transport(404, b'{"message":"Not Found"}')
        svc = self._svc_with_transport(tmp_path, transport)

        with pytest.raises(RuntimeError, match="404"):
            await svc._fetch_and_extract()

    async def test_401_raises_with_redacted_token(self, tmp_path: Path) -> None:
        """401 response body containing the token must have the token redacted."""
        token = "ghp_secret_xyz"
        body = f'{{"message":"Bad credentials","token":"{token}"}}'.encode()
        transport = _make_mock_transport(401, body)

        client = httpx.AsyncClient(transport=transport)
        svc = _make_service(tmp_path, github_token=token, http_client=client)

        with pytest.raises(RuntimeError) as exc_info:
            await svc._fetch_and_extract()

        assert token not in str(exc_info.value)
        assert "401" in str(exc_info.value)

    async def test_unsafe_tarball_member_filtered(self, tmp_path: Path) -> None:
        """Members with path traversal are rejected; file not written to cache_dir."""
        buf = io.BytesIO()
        with tarfile_module.open(fileobj=buf, mode="w:gz") as tf:
            # Well-formed top-level dir
            dir_info = tarfile_module.TarInfo(name="gearbox-ai-bundles-abc123")
            dir_info.type = tarfile_module.DIRTYPE
            tf.addfile(dir_info)
            # Normal file inside top-level
            data = b"bundles: []"
            normal = tarfile_module.TarInfo(name="gearbox-ai-bundles-abc123/bundle-index.yaml")
            normal.size = len(data)
            tf.addfile(normal, io.BytesIO(data))
            # Escape attempt
            evil_data = b"evil"
            evil = tarfile_module.TarInfo(name="../escape.txt")
            evil.size = len(evil_data)
            tf.addfile(evil, io.BytesIO(evil_data))
        tarball = buf.getvalue()

        transport = _make_mock_transport(200, tarball, {"ETag": '"v1"'})
        svc = self._svc_with_transport(tmp_path, transport)
        await svc._fetch_and_extract()

        # The evil file must not appear anywhere under cache_dir's parent
        assert not (tmp_path.parent / "escape.txt").exists()
        assert not (tmp_path / "escape.txt").exists()
        assert (tmp_path / "bundle-index.yaml").exists()

    async def test_unsafe_tarball_symlink_filtered(self, tmp_path: Path) -> None:
        """Symlink entries whose target escapes cache_dir are rejected; not created."""
        buf = io.BytesIO()
        with tarfile_module.open(fileobj=buf, mode="w:gz") as tf:
            # Well-formed top-level dir
            dir_info = tarfile_module.TarInfo(name="gearbox-ai-bundles-abc123")
            dir_info.type = tarfile_module.DIRTYPE
            tf.addfile(dir_info)
            # Normal file inside top-level
            data = b"bundles: []"
            normal = tarfile_module.TarInfo(name="gearbox-ai-bundles-abc123/bundle-index.yaml")
            normal.size = len(data)
            tf.addfile(normal, io.BytesIO(data))
            # Symlink whose target escapes the destination directory
            unsafe_symlink = tarfile_module.TarInfo(name="gearbox-ai-bundles-abc123/unsafe-symlink")
            unsafe_symlink.type = tarfile_module.SYMTYPE
            unsafe_symlink.linkname = "../../outside-cache-dir-target"
            tf.addfile(unsafe_symlink)
        tarball = buf.getvalue()

        transport = _make_mock_transport(200, tarball, {"ETag": '"v1"'})
        svc = self._svc_with_transport(tmp_path, transport)
        await svc._fetch_and_extract()

        # The symlink must not appear in cache_dir (as file or symlink)
        symlink_path = tmp_path / "unsafe-symlink"
        assert not symlink_path.exists()
        assert not symlink_path.is_symlink()
        # Normal content is still present
        assert (tmp_path / "bundle-index.yaml").exists()

    async def test_malformed_tarball_layout_raises(self, tmp_path: Path) -> None:
        """Tarball with two top-level dirs raises RuntimeError."""
        buf = io.BytesIO()
        with tarfile_module.open(fileobj=buf, mode="w:gz") as tf:
            data = b"x"
            for name in ("dir_one/file.txt", "dir_two/file.txt"):
                info = tarfile_module.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        tarball = buf.getvalue()

        transport = _make_mock_transport(200, tarball, {"ETag": '"v1"'})
        svc = self._svc_with_transport(tmp_path, transport)

        with pytest.raises(RuntimeError, match="unexpected tarball layout"):
            await svc._fetch_and_extract()

    async def test_owns_http_client_when_not_provided(self, tmp_path: Path) -> None:
        """Without http_client, service creates and closes its own client."""
        svc = _make_service(tmp_path)
        assert svc._owned_client is not None
        svc._running = True  # simulate started state so stop() proceeds
        await svc.stop()
        assert svc._owned_client is None

    async def test_borrows_http_client_when_provided(self, tmp_path: Path) -> None:
        """With http_client, stop() does not close the provided client."""
        external_client = httpx.AsyncClient()
        svc = _make_service(tmp_path, http_client=external_client)
        assert svc._owned_client is None

        svc._running = True  # simulate started state so stop() proceeds
        await svc.stop()

        # External client is still usable
        assert not external_client.is_closed
        await external_client.aclose()

    async def test_branch_used_in_url(self, tmp_path: Path) -> None:
        """The branch parameter appears in the tarball API URL."""
        tarball = _make_test_tarball(
            "gearbox-ai-bundles-abc123",
            {"bundle-index.yaml": "bundles: []"},
        )
        captured_url: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_url.append(str(request.url))
            return httpx.Response(200, content=tarball, headers={"ETag": '"v1"'})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        svc = _make_service(tmp_path, branch="develop", http_client=client)
        await svc._fetch_and_extract()

        assert captured_url, "No request was made"
        assert "/tarball/develop" in captured_url[0]


# ---------------------------------------------------------------------------
# Malicious tarball fixture helpers
# ---------------------------------------------------------------------------


def _make_oversized_member_tarball(top_dir: str, member_size: int) -> bytes:
    """Build a tarball with one member of `member_size` zero bytes."""
    buf = io.BytesIO()
    with tarfile_module.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile_module.TarInfo(name=f"{top_dir}/big.bin")
        info.size = member_size
        tf.addfile(info, io.BytesIO(b"\x00" * member_size))
    return buf.getvalue()


def _make_many_members_tarball(top_dir: str, count: int) -> bytes:
    """Build a tarball with `count` empty members."""
    buf = io.BytesIO()
    with tarfile_module.open(fileobj=buf, mode="w:gz") as tf:
        for i in range(count):
            info = tarfile_module.TarInfo(name=f"{top_dir}/file_{i}.txt")
            info.size = 0
            tf.addfile(info, io.BytesIO(b""))
    return buf.getvalue()


def _make_lying_size_tarball(top_dir: str, claimed_size: int, actual_size: int) -> bytes:
    """Build a tarball where TarInfo.size lies about the body size.

    Standard tarfile.addfile won't produce this directly (it pads/truncates to
    the declared size). We construct the tar manually: write the header with
    claimed_size, then write actual_size bytes of body, padded to 512-byte blocks.
    """
    import gzip

    # Build TarInfo header with claimed_size
    info = tarfile_module.TarInfo(name=f"{top_dir}/lying.bin")
    info.size = claimed_size
    header_bytes = info.tobuf()

    raw = io.BytesIO()
    raw.write(header_bytes)
    raw.write(b"\x00" * actual_size)
    # Pad body to 512-byte block boundary
    padding = (-actual_size) % 512
    raw.write(b"\x00" * padding)
    # End-of-archive: two zero blocks
    raw.write(b"\x00" * 1024)

    return gzip.compress(raw.getvalue())


# ---------------------------------------------------------------------------
# Tarball decompression-bomb hardening tests
# ---------------------------------------------------------------------------


class TestTarballHardening:
    """Tests for the five-layer decompression-bomb defense."""

    def _svc_with_transport(
        self,
        tmp_path: Path,
        transport: httpx.MockTransport,
        **cap_overrides: Any,
    ) -> BundleIndexService:
        client = httpx.AsyncClient(transport=transport)
        caps = {**_TEST_CAPS, **cap_overrides}
        return _make_service(tmp_path, http_client=client, **caps)

    # --- Cap #1: HTTP download size ---

    async def test_download_size_cap_aborts_stream(self, tmp_path: Path) -> None:
        """Tarball larger than max_download_bytes raises mid-stream."""
        # Use a cap small enough that our tiny tarball still exceeds it
        big_tarball = _make_test_tarball(
            "gearbox-ai-bundles-abc123", {"bundle-index.yaml": "bundles: []"}
        )
        transport = _make_mock_transport(200, big_tarball, {"ETag": '"v1"'})
        svc = self._svc_with_transport(tmp_path, transport, max_download_bytes=len(big_tarball) - 1)

        with pytest.raises(RuntimeError, match="max download size"):
            await svc._fetch_and_extract()

        assert not tmp_path.exists() or not (tmp_path / "bundle-index.yaml").exists()

    async def test_content_length_header_exceeds_cap_short_circuits(self, tmp_path: Path) -> None:
        """Content-Length header above cap raises before stream is fully consumed."""
        body_consumed: list[bool] = [False]

        def handler(_request: httpx.Request) -> httpx.Response:
            body_consumed[0] = True
            return httpx.Response(
                200,
                content=b"x" * 10,
                headers={"Content-Length": "100000000", "ETag": '"v1"'},
            )

        transport = httpx.MockTransport(handler)
        svc = self._svc_with_transport(tmp_path, transport, max_download_bytes=1024)

        with pytest.raises(RuntimeError, match="Content-Length"):
            await svc._fetch_and_extract()

    # --- Cap #2: member count ---

    async def test_member_count_cap_rejects_many_files_tarball(self, tmp_path: Path) -> None:
        """Tarball with more members than max_member_count raises."""
        cap = 10
        tarball = _make_many_members_tarball("gearbox-ai-bundles-abc123", count=cap + 1)
        transport = _make_mock_transport(200, tarball, {"ETag": '"v1"'})
        svc = self._svc_with_transport(tmp_path, transport, max_member_count=cap)

        with pytest.raises(RuntimeError, match="member count"):
            await svc._fetch_and_extract()

    # --- Cap #3: per-member declared size ---

    async def test_per_member_declared_size_cap_rejects_oversized_member(
        self, tmp_path: Path
    ) -> None:
        """Member with TarInfo.size > max_member_size_bytes raises before extraction."""
        cap = 1024
        tarball = _make_oversized_member_tarball("gearbox-ai-bundles-abc123", member_size=cap + 1)
        transport = _make_mock_transport(200, tarball, {"ETag": '"v1"'})
        svc = self._svc_with_transport(tmp_path, transport, max_member_size_bytes=cap)

        with pytest.raises(RuntimeError, match="declares size"):
            await svc._fetch_and_extract()

    # --- Cap #4: total uncompressed size ---

    async def test_total_uncompressed_size_cap_rejects_combined_overflow(
        self, tmp_path: Path
    ) -> None:
        """Multiple members each under per-member cap but combined over total cap raise."""
        per_member_cap = 2048
        total_cap = 3000  # less than 2 * per_member_cap

        buf = io.BytesIO()
        with tarfile_module.open(fileobj=buf, mode="w:gz") as tf:
            for i in range(2):
                data = b"\x00" * per_member_cap
                info = tarfile_module.TarInfo(name=f"gearbox-ai-bundles-abc123/file_{i}.bin")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        tarball = buf.getvalue()

        transport = _make_mock_transport(200, tarball, {"ETag": '"v1"'})
        svc = self._svc_with_transport(
            tmp_path,
            transport,
            max_member_size_bytes=per_member_cap,
            max_uncompressed_bytes=total_cap,
        )

        with pytest.raises(RuntimeError, match="total uncompressed size"):
            await svc._fetch_and_extract()

    # --- Cap #5: lying-size defense ---

    def test_lying_size_tarball_caught_during_extraction(self, tmp_path: Path) -> None:
        """_extract_member_capped raises when actual bytes read exceed the cap.

        Python's tarfile.extractfile() is itself bounded by TarInfo.size, so a
        tarball-level lying-size bomb is neutralised by Python before we even see
        it. This unit test bypasses that layer to verify that _extract_member_capped
        independently enforces the cap on actual reads — defence in depth against
        future CPython changes or other callers.
        """
        from unittest.mock import MagicMock

        cap = 512
        svc = _make_service(tmp_path, max_member_size_bytes=cap)

        # Member with a small declared size that passes the metadata check (cap #3).
        member = tarfile_module.TarInfo(name="gearbox-ai-bundles-abc123/lying.bin")
        member.size = 100  # declared 100 < cap 512 → metadata check passes

        # Inject a stream that has more data than the cap (simulating a lying body).
        fake_tf = MagicMock()
        fake_tf.extractfile.return_value = io.BytesIO(b"\x00" * (cap + 100))

        staging = tmp_path / "staging"
        staging.mkdir()

        with pytest.raises(RuntimeError, match="during extraction"):
            svc._extract_member_capped(fake_tf, member, staging)

    # --- Legitimate bundle should pass ---

    async def test_caps_at_defaults_dont_reject_legitimate_bundle(self, tmp_path: Path) -> None:
        """Default caps pass a small, realistic tarball without error."""
        from src.core.config import Settings

        settings = Settings()
        tarball = _make_test_tarball(
            "gearbox-ai-bundles-abc123",
            {"bundle-index.yaml": "bundles: []", "README.md": "hello"},
        )
        transport = _make_mock_transport(200, tarball, {"ETag": '"v1"'})
        client = httpx.AsyncClient(transport=transport)
        svc = BundleIndexService(
            repo_url="https://github.com/gearbox/ai-bundles.git",
            github_token="ghp_test",
            branch="master",
            sync_interval_minutes=15,
            cache_dir=tmp_path,
            default_comfyui_port=settings.comfyui_port,
            max_download_bytes=settings.ai_bundles_max_download_bytes,
            max_member_count=settings.ai_bundles_max_member_count,
            max_member_size_bytes=settings.ai_bundles_max_member_size_bytes,
            max_uncompressed_bytes=settings.ai_bundles_max_uncompressed_bytes,
            http_client=client,
        )
        await svc._fetch_and_extract()
        assert (tmp_path / "bundle-index.yaml").exists()

    # --- Cleanup on failure ---

    async def test_partial_extraction_cleaned_up_on_cap_breach(self, tmp_path: Path) -> None:
        """Staging dir is removed when a cap is breached mid-extraction."""
        cap = 10
        tarball = _make_many_members_tarball("gearbox-ai-bundles-abc123", count=cap + 1)
        transport = _make_mock_transport(200, tarball, {"ETag": '"v1"'})
        svc = self._svc_with_transport(tmp_path, transport, max_member_count=cap)

        with pytest.raises(RuntimeError):
            await svc._fetch_and_extract()

        staging = tmp_path.with_name(f"{tmp_path.name}.staging")
        assert not staging.exists()

    # --- Path traversal still rejected after refactor ---

    async def test_path_traversal_member_still_rejected_after_refactor(
        self, tmp_path: Path
    ) -> None:
        """data_filter still rejects ../escape members in streaming mode."""
        buf = io.BytesIO()
        with tarfile_module.open(fileobj=buf, mode="w:gz") as tf:
            dir_info = tarfile_module.TarInfo(name="gearbox-ai-bundles-abc123")
            dir_info.type = tarfile_module.DIRTYPE
            tf.addfile(dir_info)
            data = b"bundles: []"
            normal = tarfile_module.TarInfo(name="gearbox-ai-bundles-abc123/bundle-index.yaml")
            normal.size = len(data)
            tf.addfile(normal, io.BytesIO(data))
            evil_data = b"evil"
            evil = tarfile_module.TarInfo(name="../escape.txt")
            evil.size = len(evil_data)
            tf.addfile(evil, io.BytesIO(evil_data))
        tarball = buf.getvalue()

        transport = _make_mock_transport(200, tarball, {"ETag": '"v1"'})
        svc = self._svc_with_transport(tmp_path, transport)
        await svc._fetch_and_extract()

        assert not (tmp_path.parent / "escape.txt").exists()
        assert (tmp_path / "bundle-index.yaml").exists()

    # --- Directory members ---

    async def test_streaming_mode_handles_directory_members(self, tmp_path: Path) -> None:
        """Directories inside the tarball are created correctly in streaming mode."""
        buf = io.BytesIO()
        with tarfile_module.open(fileobj=buf, mode="w:gz") as tf:
            top = tarfile_module.TarInfo(name="gearbox-ai-bundles-abc123")
            top.type = tarfile_module.DIRTYPE
            tf.addfile(top)
            subdir = tarfile_module.TarInfo(name="gearbox-ai-bundles-abc123/subdir")
            subdir.type = tarfile_module.DIRTYPE
            tf.addfile(subdir)
            data = b"bundles: []"
            f = tarfile_module.TarInfo(name="gearbox-ai-bundles-abc123/bundle-index.yaml")
            f.size = len(data)
            tf.addfile(f, io.BytesIO(data))
        tarball = buf.getvalue()

        transport = _make_mock_transport(200, tarball, {"ETag": '"v1"'})
        svc = self._svc_with_transport(tmp_path, transport)
        await svc._fetch_and_extract()

        assert (tmp_path / "bundle-index.yaml").exists()


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------


class TestInitValidation:
    def test_rejects_zero_sync_interval(self, tmp_path: Path) -> None:
        """sync_interval_minutes=0 would cause a tight-loop background sync."""
        with pytest.raises(ValueError, match="sync_interval_minutes must be a positive"):
            BundleIndexService(
                repo_url="https://github.com/x/y.git",
                github_token="",
                branch="master",
                sync_interval_minutes=0,
                cache_dir=tmp_path,
                **_TEST_CAPS,
            )

    def test_rejects_negative_sync_interval(self, tmp_path: Path) -> None:
        """asyncio.sleep() with a negative value raises; fail fast instead."""
        with pytest.raises(ValueError, match="sync_interval_minutes must be a positive"):
            BundleIndexService(
                repo_url="https://github.com/x/y.git",
                github_token="",
                branch="master",
                sync_interval_minutes=-5,
                cache_dir=tmp_path,
                **_TEST_CAPS,
            )

    def test_branch_is_required(self, tmp_path: Path) -> None:
        """Omitting branch raises TypeError (required parameter)."""
        with pytest.raises(TypeError):
            BundleIndexService(  # type: ignore[call-arg]
                repo_url="https://github.com/x/y.git",
                github_token="",
                sync_interval_minutes=15,
                cache_dir=tmp_path,
                **_TEST_CAPS,
            )

    def test_invalid_url_raises_at_construction(self, tmp_path: Path) -> None:
        """URL validation fires in __init__, not lazily at first sync."""
        with pytest.raises(ValueError, match="SSH form is no longer supported"):
            BundleIndexService(
                repo_url="git@github.com:gearbox/ai-bundles.git",
                github_token="",
                branch="master",
                sync_interval_minutes=15,
                cache_dir=tmp_path,
                **_TEST_CAPS,
            )

    def test_rejects_zero_cap_values(self, tmp_path: Path) -> None:
        """Cap fields must be positive; zero or negative raise ValueError."""
        for bad_field in (
            "max_download_bytes",
            "max_member_count",
            "max_member_size_bytes",
            "max_uncompressed_bytes",
        ):
            caps = {**_TEST_CAPS, bad_field: 0}
            with pytest.raises(ValueError, match=bad_field):
                BundleIndexService(
                    repo_url="https://github.com/x/y.git",
                    github_token="",
                    branch="master",
                    sync_interval_minutes=15,
                    cache_dir=tmp_path,
                    **caps,
                )

    def test_settings_caps_validation_rejects_below_minimum(self) -> None:
        """Settings ge=1024 / ge=1 constraints reject values below the minimum."""
        with pytest.raises(ValidationError):
            Settings(ai_bundles_max_download_bytes=0)
