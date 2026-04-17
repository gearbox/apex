"""Bundle index service: syncs ai-bundles repo and resolves ModelType → BundleMapping."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import structlog
import yaml

from src.core.bundle_config import BundleMapping, HardwareRequirements

logger = structlog.get_logger()


class BundleNotFoundError(Exception):
    """No bundle found for the requested model type or bundle spec."""


@dataclass
class _BundleIndexEntry:
    bundle_name: str
    bundle_path: str  # relative path inside the repo
    model_type: str
    is_default: bool
    mapping: BundleMapping


class BundleIndexService:
    """Reads the ai-bundles git repository to resolve ModelType → bundle mappings.

    Syncs the ai-bundles repo on startup (shallow clone) and periodically (git pull).
    Parses bundle-index.yaml for model_type mappings and individual bundle.yaml
    files for hardware requirements.

    Thread-safety: reads are from an in-memory cache; writes (sync) are infrequent
    and replace the cache atomically.
    """

    def __init__(
        self,
        repo_url: str,
        github_token: str,
        sync_interval_minutes: int,
        cache_dir: Path | None = None,
    ) -> None:
        self._repo_url = repo_url
        self._github_token = github_token
        self._sync_interval = sync_interval_minutes
        self._cache_dir = cache_dir or Path("/tmp/apex-bundle-cache")
        # model_type -> entry (only default_bundle=true entries)
        self._model_index: dict[str, _BundleIndexEntry] = {}
        # bundle_name -> entry (all entries)
        self._bundle_index: dict[str, _BundleIndexEntry] = {}
        self._running = False
        self._sync_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Initial sync + start periodic background sync."""
        if self._running:
            return
        self._running = True
        await self.sync()
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(
            "bundle_index.started",
            cache_dir=str(self._cache_dir),
            sync_interval_minutes=self._sync_interval,
        )

    async def stop(self) -> None:
        """Stop the background sync task."""
        if not self._running:
            return
        self._running = False
        if self._sync_task is not None:
            self._sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sync_task
            self._sync_task = None
        logger.info("bundle_index.stopped")

    async def sync(self) -> None:
        """Clone or pull the ai-bundles repo and rebuild the in-memory index.

        Runs git operations in a thread executor to avoid blocking the event loop.
        """
        try:
            await asyncio.to_thread(self._git_sync)
            self._parse_index()
            logger.info(
                "bundle_index.synced",
                model_types=list(self._model_index.keys()),
                bundle_count=len(self._bundle_index),
            )
        except Exception:
            logger.exception("bundle_index.sync_error")

    def resolve_bundle(self, model_type: str) -> BundleMapping:
        """Resolve the default bundle for a ModelType.

        Args:
            model_type: ModelType enum value (e.g. "aisha-video").

        Returns:
            BundleMapping with bundle name, version, and hardware requirements.

        Raises:
            BundleNotFoundError: If no default bundle exists for this model type.
        """
        entry = self._model_index.get(model_type)
        if entry is None:
            raise BundleNotFoundError(f"No default bundle configured for model type '{model_type}'")
        return entry.mapping

    def resolve_bundle_override(self, bundle_spec: str) -> BundleMapping:
        """Resolve an admin-specified bundle override.

        Args:
            bundle_spec: Bundle name or "name:version" string.

        Returns:
            BundleMapping for the specified bundle.

        Raises:
            BundleNotFoundError: If the bundle doesn't exist in the index.
        """
        if ":" in bundle_spec:
            bundle_name, version = bundle_spec.split(":", 1)
        else:
            bundle_name = bundle_spec
            version = None

        entry = self._bundle_index.get(bundle_name)
        if entry is None:
            raise BundleNotFoundError(f"Bundle '{bundle_name}' not found in index")

        if version is None:
            return entry.mapping

        # Return a new BundleMapping with the specified version
        return BundleMapping(
            bundle_name=entry.mapping.bundle_name,
            bundle_version=version,
            hardware=entry.mapping.hardware,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_index(self) -> None:
        """Parse bundle-index.yaml and build the in-memory index."""
        index_file = self._cache_dir / "bundle-index.yaml"
        if not index_file.exists():
            logger.warning("bundle_index.index_file_missing", path=str(index_file))
            return

        with index_file.open() as fh:
            data = yaml.safe_load(fh)

        model_index: dict[str, _BundleIndexEntry] = {}
        bundle_index: dict[str, _BundleIndexEntry] = {}

        for item in data.get("bundles", []):
            name: str = item["name"]
            path: str = item["path"]
            model_type: str = item["model_type"]
            is_default: bool = bool(item.get("default_bundle", False))

            bundle_path = self._cache_dir / path
            hardware = self._parse_hardware(bundle_path)

            mapping = BundleMapping(
                bundle_name=name,
                bundle_version=None,
                hardware=hardware,
            )
            entry = _BundleIndexEntry(
                bundle_name=name,
                bundle_path=path,
                model_type=model_type,
                is_default=is_default,
                mapping=mapping,
            )

            bundle_index[name] = entry
            if is_default:
                model_index[model_type] = entry

        # Atomic replacement
        self._model_index = model_index
        self._bundle_index = bundle_index

    def _parse_hardware(self, bundle_path: Path) -> HardwareRequirements:
        """Parse hardware requirements from a bundle.yaml file.

        Args:
            bundle_path: Path to the bundle directory (containing the 'current' symlink).

        Returns:
            HardwareRequirements parsed from the hardware section.

        Raises:
            ValueError: If the hardware section is missing or malformed.
        """
        bundle_yaml = bundle_path / "current" / "bundle.yaml"
        if not bundle_yaml.exists():
            raise ValueError(f"bundle.yaml not found at {bundle_yaml}")

        with bundle_yaml.open() as fh:
            data = yaml.safe_load(fh)

        hw = data.get("hardware")
        if hw is None:
            raise ValueError(f"Missing 'hardware' section in {bundle_yaml}")

        gpu_whitelist_raw = hw.get("gpu_whitelist", [])
        return HardwareRequirements(
            gpu_whitelist=tuple(gpu_whitelist_raw),
            min_disk_gb=int(hw["min_disk_gb"]),
            min_network_upload_mbps=int(hw["min_network_upload_mbps"]),
            min_network_download_mbps=int(hw["min_network_download_mbps"]),
            cuda_min_version=str(hw["cuda_min_version"]),
            num_gpus=int(hw["num_gpus"]),
            comfyui_port=int(hw.get("comfyui_port", 18188)),
        )

    def _build_repo_url_with_token(self) -> str:
        """Build HTTPS URL with embedded PAT for git operations."""
        if self._github_token and self._repo_url.startswith("https://"):
            return self._repo_url.replace("https://", f"https://{self._github_token}@", 1)
        return self._repo_url

    def _git_sync(self) -> None:
        """Perform the actual git clone or pull (runs in thread executor)."""
        repo_dir = self._cache_dir
        url = self._build_repo_url_with_token()

        if (repo_dir / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(repo_dir), "pull", "--ff-only"],
                capture_output=True,
                text=True,
            )
        else:
            repo_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", "--depth=1", url, str(repo_dir)],
                capture_output=True,
                text=True,
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"git operation failed (exit {result.returncode}): {result.stderr.strip()}"
            )

    async def _sync_loop(self) -> None:
        """Periodic background sync loop."""
        while self._running:
            await asyncio.sleep(self._sync_interval * 60)
            if self._running:
                await self.sync()
