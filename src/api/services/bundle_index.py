"""Bundle index service: syncs ai-bundles repo and resolves ModelType → BundleMapping."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import anyio
import httpx
import structlog
import yaml

from src.core.bundle_config import (
    DEFAULT_COMFYUI_PORT,
    BundleMapping,
    HardwareRequirements,
    ReadinessMarker,
)

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


def _parse_github_url(url: str) -> tuple[str, str]:
    """Parse a GitHub HTTPS URL into (owner, repo).

    Raises ValueError for URLs that are not parseable as a GitHub
    HTTPS repo URL — including git@... SSH forms, which are no
    longer supported now that we use the REST API.
    """
    if not url.startswith("https://github.com/"):
        raise ValueError(
            f"ai_bundles_repo_url must be an https://github.com/ URL "
            f"(SSH form is no longer supported), got: {url!r}"
        )
    path = url.removeprefix("https://github.com/").removesuffix(".git").strip("/")
    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"ai_bundles_repo_url must be of the form "
            f"https://github.com/<owner>/<repo>, got: {url!r}"
        )
    return parts[0], parts[1]


class BundleIndexService:
    """Reads the ai-bundles repo to resolve ModelType → bundle mappings.

    Fetches the ai-bundles tree from GitHub's tarball API on startup and
    periodically thereafter. Parses bundle-index.yaml for model_type mappings
    and individual bundle.yaml files for hardware requirements.

    Thread-safety: reads are from an in-memory cache; writes (sync) are infrequent
    and serialized by an asyncio.Lock so concurrent sync() calls don't race on
    the on-disk cache directory.
    """

    _FETCH_TIMEOUT_SECONDS = 60

    def __init__(
        self,
        repo_url: str,
        github_token: str,
        branch: str,
        sync_interval_minutes: int,
        cache_dir: Path | None = None,
        *,
        default_comfyui_port: int,
        max_download_bytes: int,
        max_member_count: int,
        max_member_size_bytes: int,
        max_uncompressed_bytes: int,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if sync_interval_minutes <= 0:
            raise ValueError(
                f"sync_interval_minutes must be a positive integer, got {sync_interval_minutes!r}"
            )
        for name, val in (
            ("default_comfyui_port", default_comfyui_port),
            ("max_download_bytes", max_download_bytes),
            ("max_member_count", max_member_count),
            ("max_member_size_bytes", max_member_size_bytes),
            ("max_uncompressed_bytes", max_uncompressed_bytes),
        ):
            if val <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val!r}")
        self._default_comfyui_port = default_comfyui_port
        self._owner, self._repo = _parse_github_url(repo_url)
        self._github_token = github_token
        self._branch = branch
        self._sync_interval = sync_interval_minutes
        self._cache_dir = cache_dir or Path("/tmp/apex-bundle-cache")
        # model_type -> entry (only default_bundle=true entries)
        self._model_index: dict[str, _BundleIndexEntry] = {}
        # bundle_name -> entry (all entries)
        self._bundle_index: dict[str, _BundleIndexEntry] = {}
        self._running = False
        self._sync_task: asyncio.Task[None] | None = None
        self._sync_lock = asyncio.Lock()
        # ETag for change detection; in-memory only (reset on pod restart).
        self._etag: str | None = None
        # Decompression-bomb defense caps.
        self._max_download_bytes = max_download_bytes
        self._max_member_count = max_member_count
        self._max_member_size_bytes = max_member_size_bytes
        self._max_uncompressed_bytes = max_uncompressed_bytes
        # HTTP client ownership: borrowed clients are not closed by stop().
        self._http_client = http_client
        self._owned_client: httpx.AsyncClient | None = None
        if http_client is None:
            self._owned_client = httpx.AsyncClient()

    async def start(self) -> None:
        """Initial sync + start periodic background sync.

        Raises:
            Exception: If the initial fetch or index parse fails. The service
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
        if self._owned_client is not None:
            await self._owned_client.aclose()
            self._owned_client = None
        logger.info("bundle_index.stopped")

    async def sync(self) -> None:
        """Fetch the ai-bundles tree from GitHub and rebuild the in-memory index.

        Errors are logged but not raised — use this for background/manual
        refreshes where stale data is preferable to crashing. For startup,
        use start() which re-raises.
        """
        await self._sync_once(raise_on_error=False)

    async def _sync_once(self, *, raise_on_error: bool) -> None:
        """Single sync pass. Raises when raise_on_error=True, else logs and swallows."""
        async with self._sync_lock:
            try:
                await self._fetch_and_extract()
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
            readiness_marker=entry.mapping.readiness_marker,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_and_extract(self) -> None:
        """Fetch the ai-bundles tree from GitHub's tarball API and extract.

        On 304 Not Modified, returns early without touching the cache dir.
        On 200 OK, downloads the tarball, extracts to a staging dir,
        atomically swaps it with cache_dir, and updates self._etag.
        """
        url = f"https://api.github.com/repos/{self._owner}/{self._repo}/tarball/{self._branch}"
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "apex-bundle-index/1.0",
        }
        if self._github_token:
            headers["Authorization"] = f"Bearer {self._github_token}"
        if self._etag is not None:
            headers["If-None-Match"] = self._etag

        client = self._http_client if self._http_client is not None else self._owned_client
        assert client is not None  # guaranteed by __init__

        tmp_path: Path | None = None
        new_etag: str | None = None
        try:
            async with client.stream(
                "GET",
                url,
                headers=headers,
                follow_redirects=True,
                timeout=self._FETCH_TIMEOUT_SECONDS,
            ) as response:
                if response.status_code == 304:
                    logger.debug("bundle_index.unchanged", etag=self._etag)
                    return
                if response.status_code != 200:
                    body = await response.aread()
                    raise RuntimeError(
                        f"github tarball fetch failed (status {response.status_code}): "
                        f"{self._redact_token(body.decode('utf-8', errors='replace'))[:500]}"
                    )

                new_etag = response.headers.get("ETag")

                # Early-exit if Content-Length header exceeds the cap.
                declared_length = response.headers.get("Content-Length")
                if declared_length is not None:
                    with contextlib.suppress(ValueError):
                        if int(declared_length) > self._max_download_bytes:
                            raise RuntimeError(
                                f"github tarball Content-Length ({declared_length}) exceeds "
                                f"max download size ({self._max_download_bytes})"
                            )
                # Stream into a temp file on the same filesystem as cache_dir
                # (required for the intra-filesystem rename in _extract_to_cache).
                self._cache_dir.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz", dir=self._cache_dir.parent)
                os.close(fd)
                tmp_path = Path(tmp_name)
                total_bytes = 0
                async with await anyio.open_file(tmp_path, "wb") as tmp_file:
                    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                        total_bytes += len(chunk)
                        if total_bytes > self._max_download_bytes:
                            raise RuntimeError(
                                f"github tarball exceeds max download size "
                                f"({self._max_download_bytes} bytes); aborted at {total_bytes} bytes"
                            )
                        await tmp_file.write(chunk)

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"github tarball fetch timed out after {self._FETCH_TIMEOUT_SECONDS}s"
            ) from exc

        # tmp_path is always set here: 304 returned above, !200 raised above.
        if tmp_path is None:  # appease mypy — unreachable at runtime
            return

        try:
            await asyncio.to_thread(self._extract_to_cache, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        self._etag = new_etag
        logger.info("bundle_index.fetched", etag=new_etag)

    def _extract_to_cache(self, tarball_path: Path) -> None:
        """Extract `tarball_path` into self._cache_dir atomically.

        The tarball's top-level entry is a single `<owner>-<repo>-<sha>/`
        directory; we strip that prefix so the extracted tree matches
        what `git clone` produced (bundle-index.yaml at the cache_dir
        root, etc.).

        Atomic strategy: extract to `<cache_dir>.staging/`, then rename it
        onto `<cache_dir>`. The old version is moved aside first and removed
        after the swap. Concurrent `_parse_index` callers serialized by
        `_sync_lock` (see `_sync_once`) never observe a partial tree.
        """
        staging = self._cache_dir.with_name(f"{self._cache_dir.name}.staging")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=False)

        extracted_count = 0
        extracted_total_bytes = 0

        try:
            with tarfile.open(tarball_path, mode="r|gz") as tf:
                for member in tf:
                    # --- Cap #2: member count ---
                    if extracted_count >= self._max_member_count:
                        raise RuntimeError(
                            f"tarball member count exceeds limit "
                            f"({self._max_member_count}); aborted"
                        )

                    # --- Path-traversal / symlink-escape filter ---
                    try:
                        filtered = tarfile.data_filter(member, str(staging))
                    except tarfile.FilterError as exc:
                        logger.warning(
                            "bundle_index.tarball_member_rejected",
                            member=member.name,
                            reason=str(exc),
                        )
                        continue
                    if filtered is None:
                        continue
                    member = filtered

                    # --- Cap #3: per-member declared size ---
                    if member.size > self._max_member_size_bytes:
                        raise RuntimeError(
                            f"tarball member {member.name!r} declares size "
                            f"{member.size} > limit {self._max_member_size_bytes}"
                        )

                    # --- Cap #4: total declared uncompressed size ---
                    if extracted_total_bytes + member.size > self._max_uncompressed_bytes:
                        raise RuntimeError(
                            f"tarball total uncompressed size exceeds limit "
                            f"({self._max_uncompressed_bytes}); aborted at "
                            f"{extracted_total_bytes + member.size} bytes"
                        )

                    # --- Cap #5: actual extracted bytes per file (lying-size defense) ---
                    actual_bytes = self._extract_member_capped(tf, member, staging)

                    extracted_total_bytes += actual_bytes
                    extracted_count += 1
        except Exception:
            # Clean up partial staging on any failure so next sync starts fresh.
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

        logger.info(
            "bundle_index.extracted",
            member_count=extracted_count,
            total_bytes=extracted_total_bytes,
        )

        # Find the top-level <owner>-<repo>-<sha>/ directory and promote its
        # contents to the staging root.
        children = list(staging.iterdir())
        if len(children) != 1 or not children[0].is_dir():
            raise RuntimeError(
                f"unexpected tarball layout: top level had {len(children)} entries, "
                f"expected exactly 1 directory"
            )
        inner = children[0]
        for item in inner.iterdir():
            item.rename(staging / item.name)
        inner.rmdir()

        # Atomic swap: rename old → trash, new → cache_dir, rmtree(trash).
        trash: Path | None = None
        if self._cache_dir.exists():
            trash = self._cache_dir.with_name(f"{self._cache_dir.name}.old")
            if trash.exists():
                shutil.rmtree(trash)
            self._cache_dir.rename(trash)

        staging.rename(self._cache_dir)
        if trash is not None:
            shutil.rmtree(trash)

    def _extract_member_capped(
        self,
        tf: tarfile.TarFile,
        member: tarfile.TarInfo,
        dest: Path,
    ) -> int:
        """Extract one tar member with a per-member byte cap on actual reads.

        Returns the number of bytes actually written. For non-regular members
        (directories, symlinks), returns 0.

        Why we don't just use tf.extract(member, dest):
          tf.extract trusts member.size and reads exactly that many bytes from
          the stream. A malicious tarball can declare size=1024 but encode 10 GB
          of payload; reading "exactly 1024 bytes" still consumes the full
          bomb on the underlying gzip stream because tarfile reads the entire
          block. We need to bound the actual decompressed read.
        """
        # Directories, symlinks, hardlinks, devices: no body. Use built-in extract.
        if not member.isreg():
            tf.extract(member, dest, set_attrs=False)  # noqa: S202  # filter applied above
            return 0

        target = dest / member.name
        target.parent.mkdir(parents=True, exist_ok=True)

        src = tf.extractfile(member)
        if src is None:
            # extractfile returns None for non-regular members; defensive guard.
            return 0

        cap = self._max_member_size_bytes
        written = 0
        with target.open("wb") as out:
            while True:
                # cap+1 lets us detect overflow on the last chunk without off-by-one.
                remaining = cap - written + 1
                if remaining <= 0:
                    raise RuntimeError(
                        f"tarball member {member.name!r} exceeds per-member "
                        f"size cap ({cap} bytes) during extraction"
                    )
                chunk = src.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                written += len(chunk)
                if written > cap:
                    raise RuntimeError(
                        f"tarball member {member.name!r} exceeds per-member "
                        f"size cap ({cap} bytes) during extraction "
                        f"(declared size was {member.size})"
                    )
                out.write(chunk)

        if member.mode is not None:
            target.chmod(member.mode & 0o777)
        return written

    def _redact_token(self, text: str) -> str:
        """Replace any occurrence of the PAT in `text` with '***'.

        Defensive: GitHub's API does not echo the Authorization header
        in error bodies, but a misconfigured proxy or future API change
        could. Redact unconditionally before logging.
        """
        if not text or not self._github_token:
            return text
        return text.replace(self._github_token, "***")

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
        hardware, readiness_marker = self._parse_hardware(bundle_path)

        mapping = BundleMapping(
            bundle_name=name,
            bundle_version=None,
            hardware=hardware,
            readiness_marker=readiness_marker,
        )
        return _BundleIndexEntry(
            bundle_name=name,
            bundle_path=path,
            model_type=model_type,
            is_default=is_default,
            mapping=mapping,
        )

    def _parse_hardware(
        self, bundle_path: Path
    ) -> tuple[HardwareRequirements, ReadinessMarker | None]:
        """Parse hardware requirements and readiness marker from a bundle.yaml file.

        Args:
            bundle_path: Path to the bundle directory (containing the 'current' symlink).

        Returns:
            Tuple of (HardwareRequirements, ReadinessMarker | None).

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
            """Strict: accepts int (not bool) or integer-valued string. Rejects float.

            Rationale: ``int()`` silently coerces ``True`` → ``1``, ``"3.14"`` →
            ValueError (good) but ``3.14`` → ``3`` (silent truncation of a
            config error). YAML also loads ``42`` as int and ``42.0`` as float,
            so we can distinguish them.
            """
            if field not in hw:
                raise ValueError(f"{bundle_yaml}: missing required field 'hardware.{field}'")
            value = hw[field]
            # bool is a subclass of int — reject it explicitly
            if isinstance(value, bool):
                raise ValueError(
                    f"{bundle_yaml}: 'hardware.{field}' must be an integer, got bool {value!r}"
                )
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                try:
                    return int(value)  # accepts "42", rejects "42.0" / "1e3"
                except ValueError as exc:
                    raise ValueError(
                        f"{bundle_yaml}: 'hardware.{field}' must be an integer, got {value!r} (str)"
                    ) from exc
            raise ValueError(
                f"{bundle_yaml}: 'hardware.{field}' must be an integer, "
                f"got {value!r} ({type(value).__name__})"
            )

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
        for idx, entry in enumerate(gpu_whitelist_raw):
            if not isinstance(entry, str):
                raise ValueError(
                    f"{bundle_yaml}: 'hardware.gpu_whitelist[{idx}]' must be a string, "
                    f"got {entry!r} ({type(entry).__name__})"
                )

        # Apply default for optional int fields before validation so _require_int
        # produces consistent error messages for invalid values.
        hw.setdefault("comfyui_port", DEFAULT_COMFYUI_PORT)

        template_hash_id = hw.get("template_hash_id")
        if template_hash_id is not None:
            if not isinstance(template_hash_id, str):
                raise ValueError(
                    f"{bundle_yaml}: 'hardware.template_hash_id' must be a string, got "
                    f"{type(template_hash_id).__name__}"
                )
            if not template_hash_id.strip():
                raise ValueError(f"{bundle_yaml}: 'hardware.template_hash_id' must be non-empty")

        hardware = HardwareRequirements(
            gpu_whitelist=tuple(gpu_whitelist_raw),
            min_disk_gb=_require_int("min_disk_gb"),
            min_network_upload_mbps=_require_int("min_network_upload_mbps"),
            min_network_download_mbps=_require_int("min_network_download_mbps"),
            cuda_min_version=cuda_min,
            num_gpus=_require_int("num_gpus"),
            comfyui_port=_require_int("comfyui_port"),
            template_hash_id=template_hash_id,
        )
        readiness_marker = self._parse_readiness_marker(data, bundle_yaml)
        return hardware, readiness_marker

    def _parse_readiness_marker(
        self, data: dict[str, object], bundle_yaml: Path
    ) -> ReadinessMarker | None:
        """Parse optional readiness_marker block. Returns None if absent.

        Raises ValueError if the key is present but malformed (e.g., missing node_class).
        """
        marker = data.get("readiness_marker")
        if marker is None:
            return None
        if not isinstance(marker, dict):
            raise ValueError(
                f"{bundle_yaml}: 'readiness_marker' must be a mapping, got {type(marker).__name__}"
            )
        node_class = marker.get("node_class")
        if not isinstance(node_class, str) or not node_class.strip():
            raise ValueError(
                f"{bundle_yaml}: 'readiness_marker.node_class' must be a non-empty string"
            )
        return ReadinessMarker(node_class=node_class)

    async def _sync_loop(self) -> None:
        """Periodic background sync loop."""
        while self._running:
            await asyncio.sleep(self._sync_interval * 60)
            if self._running:
                await self.sync()
