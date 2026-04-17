"""Bundle index service: syncs ai-bundles repo and resolves ModelType → BundleMapping."""

from __future__ import annotations

import asyncio
import base64
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
    and serialized by an asyncio.Lock so concurrent sync() calls don't race on
    the on-disk cache directory.
    """

    # Max wall time for any single git invocation. Prevents a hung/prompting
    # git process from blocking the thread pool indefinitely.
    _GIT_TIMEOUT_SECONDS = 120

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
        self._sync_lock = asyncio.Lock()

    async def start(self) -> None:
        """Initial sync + start periodic background sync.

        Raises:
            Exception: If the initial clone or index parse fails. The service
                must have a usable index after start() returns; subsequent
                background sync failures are logged but do not stop the loop.
        """
        if self._running:
            return
        self._running = True
        try:
            await self._sync_once(raise_on_error=True)
        except Exception:
            # Initial sync failed — roll back and re-raise so startup fails loudly
            self._running = False
            raise
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

        Errors are logged but not raised — use this for background/manual
        refreshes where stale data is preferable to crashing. For startup,
        use start() which re-raises.
        """
        await self._sync_once(raise_on_error=False)

    async def _sync_once(self, *, raise_on_error: bool) -> None:
        """Single sync pass. Raises when raise_on_error=True, else logs and swallows.

        Runs git operations in a thread executor to avoid blocking the event loop.
        Serialized by an asyncio.Lock so concurrent callers don't race on the
        shared on-disk cache.
        """
        async with self._sync_lock:
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
                if raise_on_error:
                    raise

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
        """Parse bundle-index.yaml and build the in-memory index.

        Malformed individual bundle entries are skipped with a warning so one
        bad bundle doesn't poison the entire index.
        """
        index_file = self._cache_dir / "bundle-index.yaml"
        if not index_file.exists():
            logger.warning("bundle_index.index_file_missing", path=str(index_file))
            return

        with index_file.open() as fh:
            data = yaml.safe_load(fh)

        if data is None:
            logger.warning("bundle_index.index_file_empty", path=str(index_file))
            return
        if not isinstance(data, dict):
            logger.warning(
                "bundle_index.index_file_invalid_structure",
                path=str(index_file),
                got_type=type(data).__name__,
            )
            return

        bundles_raw = data.get("bundles", [])
        if not isinstance(bundles_raw, list):
            logger.warning(
                "bundle_index.bundles_not_a_list",
                path=str(index_file),
                got_type=type(bundles_raw).__name__,
            )
            return

        model_index: dict[str, _BundleIndexEntry] = {}
        bundle_index: dict[str, _BundleIndexEntry] = {}
        skipped = 0

        for item in bundles_raw:
            if not isinstance(item, dict):
                logger.warning(
                    "bundle_index.entry_not_a_mapping",
                    got_type=type(item).__name__,
                )
                skipped += 1
                continue
            try:
                entry = self._build_entry(item)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "bundle_index.entry_invalid",
                    bundle_name=item.get("name") if isinstance(item, dict) else None,
                    error=str(exc),
                )
                skipped += 1
                continue

            bundle_index[entry.bundle_name] = entry
            if entry.is_default:
                model_index[entry.model_type] = entry

        if skipped:
            logger.warning("bundle_index.entries_skipped", skipped=skipped)

        # Atomic replacement
        self._model_index = model_index
        self._bundle_index = bundle_index

    def _build_entry(self, item: dict[str, object]) -> _BundleIndexEntry:
        """Build a single index entry from a raw YAML mapping.

        Raises:
            KeyError: If a required field is missing.
            ValueError: If the bundle's hardware section is invalid.
        """
        for field in ("name", "path", "model_type"):
            if field not in item:
                raise KeyError(f"bundle-index.yaml entry missing required field '{field}'")

        name = str(item["name"])
        path = str(item["path"])
        model_type = str(item["model_type"])
        is_default = bool(item.get("default_bundle", False))

        bundle_path = self._cache_dir / path
        hardware = self._parse_hardware(bundle_path)

        mapping = BundleMapping(
            bundle_name=name,
            bundle_version=None,
            hardware=hardware,
        )
        return _BundleIndexEntry(
            bundle_name=name,
            bundle_path=path,
            model_type=model_type,
            is_default=is_default,
            mapping=mapping,
        )

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

        if not isinstance(data, dict):
            raise ValueError(
                f"{bundle_yaml}: expected a YAML mapping at top level, got {type(data).__name__}"
            )

        hw = data.get("hardware")
        if hw is None:
            raise ValueError(f"{bundle_yaml}: missing 'hardware' section")
        if not isinstance(hw, dict):
            raise ValueError(
                f"{bundle_yaml}: 'hardware' must be a mapping, got {type(hw).__name__}"
            )

        def _require_int(field: str) -> int:
            if field not in hw:
                raise ValueError(f"{bundle_yaml}: missing required field 'hardware.{field}'")
            value = hw[field]
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{bundle_yaml}: 'hardware.{field}' must be an integer, got "
                    f"{value!r} ({type(value).__name__})"
                ) from exc

        if "cuda_min_version" not in hw:
            raise ValueError(f"{bundle_yaml}: missing required field 'hardware.cuda_min_version'")
        cuda_min = str(hw["cuda_min_version"])
        try:
            float(cuda_min)
        except ValueError as exc:
            raise ValueError(
                f"{bundle_yaml}: 'hardware.cuda_min_version' must be a numeric string "
                f"(e.g. '12.1'), got {cuda_min!r}"
            ) from exc

        gpu_whitelist_raw = hw.get("gpu_whitelist", [])
        if not isinstance(gpu_whitelist_raw, list):
            raise ValueError(
                f"{bundle_yaml}: 'hardware.gpu_whitelist' must be a list, got "
                f"{type(gpu_whitelist_raw).__name__}"
            )

        return HardwareRequirements(
            gpu_whitelist=tuple(gpu_whitelist_raw),
            min_disk_gb=_require_int("min_disk_gb"),
            min_network_upload_mbps=_require_int("min_network_upload_mbps"),
            min_network_download_mbps=_require_int("min_network_download_mbps"),
            cuda_min_version=cuda_min,
            num_gpus=_require_int("num_gpus"),
            comfyui_port=int(hw.get("comfyui_port", 18188)),
        )

    def _auth_extra_header(self) -> list[str]:
        """Return git `-c http.extraHeader=...` args for PAT auth, or empty list.

        Uses the GitHub-recommended pattern — same as ``actions/checkout`` —
        which authenticates via HTTP header instead of URL embedding. This avoids:

        - Persisting the token to ``.git/config`` (URL embedding does this on clone)
        - Leaking the token if git echoes the remote URL in errors

        The token is still passed in argv (visible to ``ps`` on the local host
        during the brief window of the subprocess), but it is never written to
        disk and the clone URL remains clean.
        """
        if not self._github_token or not self._repo_url.startswith("https://"):
            return []
        # GitHub accepts x-access-token:<PAT> as HTTP Basic; the token itself
        # is the password, and x-access-token is a recognized sentinel username.
        credentials = f"x-access-token:{self._github_token}".encode()
        encoded = base64.b64encode(credentials).decode()
        return ["-c", f"http.extraHeader=Authorization: Basic {encoded}"]

    def _validate_repo_url(self) -> None:
        """Reject repo URLs that would be interpreted as CLI options by git.

        Defense-in-depth: repo_url comes from operator-controlled settings, but
        a URL starting with ``-`` (or ``--``) would be parsed as an option flag
        rather than a positional URL argument, potentially triggering unintended
        git behavior. Accept only HTTPS and SSH forms.
        """
        url = self._repo_url
        if url.startswith("-"):
            raise ValueError(f"repo_url must not start with '-': {url!r}")
        if not (url.startswith("https://") or url.startswith("git@") or url.startswith("ssh://")):
            raise ValueError(f"repo_url must be an https://, git@, or ssh:// URL, got {url!r}")

    def _git_sync(self) -> None:
        """Perform the actual git clone or pull (runs in thread executor).

        Security notes:
        - ``shell=False`` (default) + list argv: no shell metacharacters can
          trigger command injection; each argv element is passed as-is to
          execve(). This is why Semgrep's ``dangerous-subprocess-use-audit``
          warning is a false positive here — but we still validate inputs.
        - ``self._repo_url`` is operator-controlled (env var), validated via
          ``_validate_repo_url`` to reject leading-dash argument injection.
        - ``self._cache_dir`` is operator-controlled.
        - ``auth_args`` is either empty or a ``-c http.extraHeader=...`` pair
          with a base64-encoded token (never raw user input).
        - ``--`` separator ensures subsequent positional args are parsed as
          URLs/paths even if git adds new option flags in future versions.
        - Bounded timeout prevents a hung/prompting git process from blocking
          the thread pool indefinitely.
        """
        self._validate_repo_url()
        repo_dir = self._cache_dir
        auth_args = self._auth_extra_header()

        if (repo_dir / ".git").exists():
            cmd = [
                "git",
                *auth_args,
                "-C",
                str(repo_dir),
                "pull",
                "--ff-only",
            ]
        else:
            repo_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "git",
                *auth_args,
                "clone",
                "--depth=1",
                "--",  # lock down positional args against future git option additions
                self._repo_url,
                str(repo_dir),
            ]

        try:
            # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            # Safe: shell=False, static-prefix list argv, repo_url validated above.
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._GIT_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"git operation timed out after {self._GIT_TIMEOUT_SECONDS}s"
            ) from exc

        if result.returncode != 0:
            # Scrub stderr of any accidental token echo before raising/logging
            stderr = self._scrub_token(result.stderr.strip())
            raise RuntimeError(f"git operation failed (exit {result.returncode}): {stderr}")

    def _scrub_token(self, text: str) -> str:
        """Best-effort removal of the PAT from text before logging/raising."""
        if self._github_token and self._github_token in text:
            return text.replace(self._github_token, "***")
        return text

    async def _sync_loop(self) -> None:
        """Periodic background sync loop."""
        while self._running:
            await asyncio.sleep(self._sync_interval * 60)
            if self._running:
                await self.sync()
