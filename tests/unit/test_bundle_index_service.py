"""Unit tests for BundleIndexService — parsing and resolution logic only.

Git operations (clone/pull) require network access and are NOT tested here.
"""

from __future__ import annotations

import base64
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


# ---------------------------------------------------------------------------
# _auth_extra_header
# ---------------------------------------------------------------------------


class TestAuthExtraHeader:
    def _decoded_header(self, args: list[str]) -> str:
        """Extract and base64-decode the Authorization header from git -c args."""
        # args look like ["-c", "http.extraHeader=Authorization: Basic <b64>"]
        assert len(args) == 2
        assert args[0] == "-c"
        prefix = "http.extraHeader=Authorization: Basic "
        assert args[1].startswith(prefix)
        b64 = args[1][len(prefix) :]
        return base64.b64decode(b64).decode()

    def test_injects_header_for_https_url(self) -> None:
        svc = BundleIndexService(
            repo_url="https://github.com/gearbox/ai-bundles.git",
            github_token="ghp_abc123",
            sync_interval_minutes=15,
        )
        args = svc._auth_extra_header()
        assert self._decoded_header(args) == "x-access-token:ghp_abc123"

    def test_empty_token_returns_no_args(self) -> None:
        svc = BundleIndexService(
            repo_url="https://github.com/gearbox/ai-bundles.git",
            github_token="",
            sync_interval_minutes=15,
        )
        assert svc._auth_extra_header() == []

    def test_non_https_url_with_token_returns_no_args(self) -> None:
        """Non-HTTPS URLs (e.g. SSH) can't use HTTP Basic auth; no header injected."""
        svc = BundleIndexService(
            repo_url="git@github.com:gearbox/ai-bundles.git",
            github_token="ghp_abc123",
            sync_interval_minutes=15,
        )
        assert svc._auth_extra_header() == []

    def test_token_not_in_plaintext_args(self) -> None:
        """The raw PAT must not appear verbatim in the argv."""
        svc = BundleIndexService(
            repo_url="https://github.com/gearbox/ai-bundles.git",
            github_token="ghp_secret_value_xyz",
            sync_interval_minutes=15,
        )
        args = svc._auth_extra_header()
        joined = " ".join(args)
        assert "ghp_secret_value_xyz" not in joined
        # But it should be recoverable via base64 decode
        assert "ghp_secret_value_xyz" in self._decoded_header(args)


# ---------------------------------------------------------------------------
# start() / sync() error behavior
# ---------------------------------------------------------------------------


class TestSyncErrorBehavior:
    async def test_start_raises_on_initial_sync_failure(self, tmp_path: Path) -> None:
        """If initial sync fails, start() must re-raise and not leave the service running."""
        svc = _make_service(tmp_path)

        def boom() -> None:
            raise RuntimeError("git clone failed")

        svc._git_sync = boom  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="git clone failed"):
            await svc.start()

        # Service must not be marked as running and no background task should exist
        assert svc._running is False
        assert svc._sync_task is None

    async def test_sync_swallows_errors(self, tmp_path: Path) -> None:
        """The public sync() method logs errors but does not raise (background-safe)."""
        svc = _make_service(tmp_path)

        def boom() -> None:
            raise RuntimeError("transient git pull failure")

        svc._git_sync = boom  # type: ignore[method-assign]

        # Must not raise — background loop depends on this
        await svc.sync()

    async def test_scrub_token_removes_pat(self, tmp_path: Path) -> None:
        """If git echoes the PAT in stderr, _scrub_token must redact it."""
        svc = BundleIndexService(
            repo_url="https://github.com/gearbox/ai-bundles.git",
            github_token="ghp_secret_abc",
            sync_interval_minutes=15,
            cache_dir=tmp_path,
        )
        scrubbed = svc._scrub_token("fatal: unable to access 'https://ghp_secret_abc@github.com/'")
        assert "ghp_secret_abc" not in scrubbed
        assert "***" in scrubbed


# ---------------------------------------------------------------------------
# _validate_repo_url
# ---------------------------------------------------------------------------


class TestValidateRepoUrl:
    def test_accepts_https_url(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path)
        svc._validate_repo_url()  # must not raise

    def test_accepts_ssh_url(self, tmp_path: Path) -> None:
        svc = BundleIndexService(
            repo_url="git@github.com:gearbox/ai-bundles.git",
            github_token="",
            sync_interval_minutes=15,
            cache_dir=tmp_path,
        )
        svc._validate_repo_url()

    def test_accepts_ssh_scheme_url(self, tmp_path: Path) -> None:
        svc = BundleIndexService(
            repo_url="ssh://git@github.com/gearbox/ai-bundles.git",
            github_token="",
            sync_interval_minutes=15,
            cache_dir=tmp_path,
        )
        svc._validate_repo_url()

    def test_rejects_dash_prefix_argument_injection(self, tmp_path: Path) -> None:
        """Leading '-' would be parsed as a git option flag, not a URL."""
        svc = BundleIndexService(
            repo_url="--upload-pack=malicious",
            github_token="",
            sync_interval_minutes=15,
            cache_dir=tmp_path,
        )
        with pytest.raises(ValueError, match="must not start with '-'"):
            svc._validate_repo_url()

    def test_rejects_unsupported_scheme(self, tmp_path: Path) -> None:
        svc = BundleIndexService(
            repo_url="file:///etc/passwd",
            github_token="",
            sync_interval_minutes=15,
            cache_dir=tmp_path,
        )
        with pytest.raises(ValueError, match="https://.*git@.*ssh://"):
            svc._validate_repo_url()
