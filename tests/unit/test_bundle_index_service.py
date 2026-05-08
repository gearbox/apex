"""Unit tests for BundleIndexService — parsing, resolution, and HTTP fetch logic."""

from __future__ import annotations

import io
import tarfile as tarfile_module
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from src.api.services.bundle_index import (
    BundleIndexService,
    BundleNotFoundError,
    _parse_github_url,
)
from src.core.bundle_config import BundleMapping, HardwareRequirements

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
    "comfyui_port": 18188,
}


def _make_service(cache_dir: Path, **kwargs: Any) -> BundleIndexService:
    defaults: dict[str, Any] = {
        "repo_url": "https://github.com/gearbox/ai-bundles.git",
        "github_token": "ghp_test",
        "branch": "master",
        "sync_interval_minutes": 15,
        "cache_dir": cache_dir,
    } | kwargs
    return BundleIndexService(**defaults)


def _write_bundle_yaml(bundle_path: Path, hw: dict[str, object] | None = None) -> None:
    """Create {bundle_path}/current/bundle.yaml with the given hardware section."""
    current_dir = bundle_path / "current"
    current_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {"hardware": hw if hw is not None else _HW_YAML}
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
        result = svc._parse_hardware(tmp_path / "ok")
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
        """Missing comfyui_port uses the default 18188, still goes through _require_int."""
        hw = {k: v for k, v in _HW_YAML.items() if k != "comfyui_port"}
        _write_bundle_yaml(tmp_path / "no_port", hw)
        svc = _make_service(tmp_path)
        result = svc._parse_hardware(tmp_path / "no_port")
        assert result.comfyui_port == 18188


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
            )

    def test_branch_is_required(self, tmp_path: Path) -> None:
        """Omitting branch raises TypeError (required parameter)."""
        with pytest.raises(TypeError):
            BundleIndexService(  # type: ignore[call-arg]
                repo_url="https://github.com/x/y.git",
                github_token="",
                sync_interval_minutes=15,
                cache_dir=tmp_path,
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
            )
