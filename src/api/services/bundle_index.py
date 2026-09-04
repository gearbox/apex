"""Bundle index service: syncs ai-bundles repo and resolves ModelType → BundleMapping."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import httpx
import structlog
import yaml

from src.api.services.workflow.binder import bind_workflow
from src.api.services.workflow.capabilities import derive_capabilities
from src.api.services.workflow.contract import MODEL_TYPE_MEDIA, BoundWorkflow, BundleCapabilities
from src.api.services.workflow.parser import parse_workflow_map
from src.core.bundle_config import (
    BundleDefinitionError,
    BundleMapping,
    HardwareRequirements,
    ReadinessMarker,
)
from src.core.enums import Resolution, Sampler, Scheduler
from src.core.generation_config import (
    BundleConfigError,
    BundleGenerationConfig,
    GenerationConstraints,
    GenerationDefaults,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.config import Settings

logger = structlog.get_logger()


def _safe_enum[E: StrEnum](enum_cls: type[E], value: str | None, fallback: E) -> E:
    """Parse an enum value, returning fallback + warning on invalid input."""
    if value is None:
        return fallback
    try:
        return enum_cls(value)
    except ValueError:
        logger.warning(
            "bundle_index.invalid_settings_enum",
            enum=enum_cls.__name__,
            value=value,
            using=fallback.value,
        )
        return fallback


class BundleNotFoundError(Exception):
    """No bundle found for the requested model type or bundle spec."""


@dataclass
class _BundleIndexEntry:
    bundle_name: str
    bundle_path: str  # relative path inside the repo
    model_type: str
    is_default: bool
    mapping: BundleMapping
    bound_workflow: BoundWorkflow
    capabilities: BundleCapabilities


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
        settings: Settings | None = None,
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
        self._cache_dir = cache_dir or Path(tempfile.gettempdir()) / "apex-bundle-cache"
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
        self._settings = settings
        self._generation_config_cache: dict[str, BundleGenerationConfig] = {}
        self._invalid_generation_config_paths: dict[str, Path] = {}
        self._model_filenames_cache: dict[tuple[str, str], list[str]] = {}
        self._declared_bytes_cache: dict[str, int | None] = {}
        self._on_resync: list[Callable[[], None]] = []

    def register_on_resync(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked after the bundle cache is refreshed (content change only)."""
        self._on_resync.append(callback)

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

    def get_bundle_path(self, bundle_name: str) -> Path:
        """Return the on-disk root path for a named bundle (without version segment).

        Args:
            bundle_name: The bundle name (e.g. "qwen_rapid_aio").

        Returns:
            Absolute path to the bundle root directory in the local cache.

        Raises:
            BundleNotFoundError: If the bundle is not in the index.
        """
        entry = self._bundle_index.get(bundle_name)
        if entry is None:
            raise BundleNotFoundError(f"Bundle '{bundle_name}' not found in index")
        return self._cache_dir / entry.bundle_path

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

        # Return a new BundleMapping with the specified version. declared_model_bytes
        # is recomputed (not copied from entry.mapping) — a version override reads a
        # different bundle.yaml, whose declared sizes may differ from 'current'.
        return BundleMapping(
            bundle_name=entry.mapping.bundle_name,
            bundle_version=version,
            hardware=entry.mapping.hardware,
            readiness_marker=entry.mapping.readiness_marker,
            capabilities=entry.mapping.capabilities,
            declared_model_bytes=self.get_declared_model_bytes(entry.mapping.bundle_name, version),
        )

    def get_model_filenames(
        self, bundle_name: str, bundle_version: str | None, model_type: str
    ) -> list[str] | None:
        """Return model filenames declared by one ``models[].model_type``.

        Returns:
            A list of model filenames (possibly empty when the bundle
            genuinely declares zero checkpoints). Returns None on any lookup
            or read/parse error so callers can distinguish a transient failure
            from a deliberate zero-checkpoint bundle. Never raises.
        """
        try:
            bundle_dir = self.get_bundle_path(bundle_name)
        except BundleNotFoundError:
            logger.debug(
                "bundle_index.model_filenames.bundle_not_found",
                bundle_name=bundle_name,
                bundle_version=bundle_version,
                model_type=model_type,
            )
            return None

        bundle_yaml_path = bundle_dir / (bundle_version or "current") / "bundle.yaml"
        cache_key = (str(bundle_yaml_path), model_type)
        cached = self._model_filenames_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            with bundle_yaml_path.open() as fh:
                data = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError) as exc:
            logger.info(
                "bundle_index.model_filenames.read_error",
                bundle_name=bundle_name,
                bundle_version=bundle_version,
                model_type=model_type,
                error=str(exc),
            )
            return None

        if not isinstance(data, dict):
            return None

        filenames: list[str] = []
        for model in data.get("models", []) or []:
            if isinstance(model, dict) and model.get("model_type") == model_type:
                filenames.extend(
                    str(file_entry["filename"])
                    for file_entry in model.get("files", []) or []
                    if isinstance(file_entry, dict) and "filename" in file_entry
                )
        self._model_filenames_cache[cache_key] = filenames
        return filenames

    def get_declared_model_bytes(self, bundle_name: str, bundle_version: str | None) -> int | None:
        """Sum declared models[].files[].size_bytes across the whole bundle (F8).

        Deliberately bundle-wide, not scoped to one ``models[].model_type`` — unlike
        get_model_filenames's asset-category filter (e.g. "checkpoints"), this is a
        disk-headroom figure for everything a provision command's download would
        pull, so every declared file counts.

        Returns:
            The exact sum when every declared file declares size_bytes. None on any
            lookup/read/parse error, AND the moment any file's size_bytes is missing
            or null — a partial sum is worse than no sum, because it under-reports
            and would let a batch-headroom guard downstream pass a batch that will
            not actually fit. Never raises.
        """
        try:
            bundle_dir = self.get_bundle_path(bundle_name)
        except BundleNotFoundError:
            logger.debug(
                "bundle_index.declared_model_bytes.bundle_not_found",
                bundle_name=bundle_name,
                bundle_version=bundle_version,
            )
            return None

        bundle_yaml_path = bundle_dir / (bundle_version or "current") / "bundle.yaml"
        cache_key = str(bundle_yaml_path)
        if cache_key in self._declared_bytes_cache:
            return self._declared_bytes_cache[cache_key]

        result = self._sum_declared_model_bytes(bundle_yaml_path)
        self._declared_bytes_cache[cache_key] = result
        return result

    def _sum_declared_model_bytes(self, bundle_yaml_path: Path) -> int | None:
        """Core summation given an already-resolved bundle.yaml path — no caching.

        Split out from get_declared_model_bytes so _build_entry can call it directly
        with the bundle_path it already computed locally: at build time self._bundle_index
        doesn't contain this entry yet (the atomic swap happens after the whole index
        is built), so get_bundle_path/get_declared_model_bytes would wrongly report
        the bundle as not-yet-found. Returns None when no files are declared: zero
        would incorrectly assert a known model-footprint size.
        """
        try:
            with bundle_yaml_path.open() as fh:
                data = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError) as exc:
            logger.info(
                "bundle_index.declared_model_bytes.read_error",
                bundle_yaml_path=str(bundle_yaml_path),
                error=str(exc),
            )
            return None

        if not isinstance(data, dict):
            return None

        total = 0
        files_summed = 0
        for model in data.get("models", []) or []:
            if not isinstance(model, dict):
                continue
            for file_entry in model.get("files", []) or []:
                if not isinstance(file_entry, dict):
                    continue
                size_bytes = file_entry.get("size_bytes")
                if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
                    return None
                total += size_bytes
                files_summed += 1
        return total if files_summed else None

    def get_capabilities(
        self, bundle_name: str, bundle_version: str | None = None
    ) -> BundleCapabilities | None:
        """Return capabilities captured when the bundle index was built."""
        _ = bundle_version
        entry = self._bundle_index.get(bundle_name)
        return entry.capabilities if entry is not None else None

    def get_bound_workflow(
        self, bundle_name: str, bundle_version: str | None = None
    ) -> BoundWorkflow | None:
        """Return the eagerly-bound graph associated with an indexed bundle."""
        _ = bundle_version
        entry = self._bundle_index.get(bundle_name)
        return entry.bound_workflow if entry is not None else None

    def get_bound_workflow_for_path(
        self, bundle_dir: Path, bundle_version: str | None = None
    ) -> BoundWorkflow | None:
        """Resolve a bound graph by the provisioned bundle root path."""
        _ = bundle_version
        try:
            resolved = bundle_dir.resolve()
        except OSError:
            resolved = bundle_dir
        for entry in self._bundle_index.values():
            try:
                entry_path = (self._cache_dir / entry.bundle_path).resolve()
            except OSError:
                entry_path = self._cache_dir / entry.bundle_path
            if entry_path == resolved:
                return entry.bound_workflow
        return None

    def get_generation_config(
        self, bundle_name: str, bundle_version: str | None
    ) -> BundleGenerationConfig:
        """Load and cache the ``generation:`` section from the bundle's bundle.yaml.

        Falls back to global Settings + permissive constraints when the section
        (or individual keys) are absent, so existing bundles keep working.

        Raises:
            BundleConfigError: If an enum string in the YAML is invalid.
        """
        entry = self._bundle_index.get(bundle_name)
        if entry is None:
            invalid_path = self._invalid_generation_config_paths.get(bundle_name)
            if invalid_path is not None:
                return self._parse_generation_config(
                    invalid_path / (bundle_version or "current") / "bundle.yaml"
                )
            return self._fallback_generation_config()

        bundle_path = self._cache_dir / entry.bundle_path
        version_dir = bundle_path / (bundle_version or "current")
        cache_key = str(version_dir / "bundle.yaml")

        if cache_key in self._generation_config_cache:
            return self._generation_config_cache[cache_key]

        config = self._parse_generation_config(
            bundle_path / (bundle_version or "current") / "bundle.yaml"
        )
        self._generation_config_cache[cache_key] = config
        return config

    def _fallback_generation_config(self) -> BundleGenerationConfig:
        """Return Settings-derived defaults + permissive constraints."""
        s = self._settings
        return BundleGenerationConfig(
            defaults=GenerationDefaults(
                resolution=_safe_enum(
                    Resolution, s.default_resolution if s else None, Resolution.STANDARD
                ),
                steps=s.default_steps if s else 12,
                cfg=s.default_cfg if s else 1.1,
                sampler=_safe_enum(Sampler, s.default_sampler if s else None, Sampler.EULER),
                scheduler=_safe_enum(Scheduler, s.default_scheduler if s else None, Scheduler.BETA),
                denoise=s.default_denoise if s else 1.0,
            ),
            constraints=GenerationConstraints(
                max_megapixels=1.0,
                latent_multiple=16,
                max_edge=1536,
                min_steps=1,
                max_steps=s.max_steps if s else 20,
                min_cfg=0.0,
                max_cfg=30.0,
                allowed_samplers=frozenset(),
                allowed_schedulers=frozenset(),
                max_batch_size=4,
            ),
        )

    def _parse_generation_config(
        self, bundle_yaml_path: Path, *, strict: bool = False
    ) -> BundleGenerationConfig:
        """Parse the ``generation:`` block from a bundle.yaml file.

        Missing keys fall back to Settings or permissive defaults.
        Unknown enum values or non-numeric YAML values raise BundleConfigError.
        With ``strict=True``, malformed config structure also raises so index
        capabilities are never derived from fallback values.
        """
        fallback = self._fallback_generation_config()

        if not bundle_yaml_path.exists():
            return fallback

        try:
            with bundle_yaml_path.open() as fh:
                data = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError) as exc:
            if strict:
                raise BundleConfigError(
                    f"{bundle_yaml_path}: unable to read generation config: {exc}"
                ) from exc
            logger.warning(
                "bundle_index.generation_config_unreadable",
                path=str(bundle_yaml_path),
                error=str(exc),
            )
            return fallback

        if not isinstance(data, dict):
            if strict:
                raise BundleConfigError(
                    f"{bundle_yaml_path}: generation config root must be a mapping, "
                    f"got {type(data).__name__}"
                )
            logger.warning(
                "bundle_index.generation_config_invalid_root",
                path=str(bundle_yaml_path),
                got_type=type(data).__name__,
            )
            return fallback

        gen = data.get("generation")
        if gen is None:
            logger.debug(
                "bundle_index.generation_config_section_absent", path=str(bundle_yaml_path)
            )
            return fallback

        if not isinstance(gen, dict):
            if not strict:
                logger.warning(
                    "bundle_index.generation_config_section_malformed",
                    path=str(bundle_yaml_path),
                    got_type=type(gen).__name__,
                )
                return fallback
            raise BundleConfigError(
                f"{bundle_yaml_path}: generation must be a mapping, got {type(gen).__name__}"
            )

        raw_defaults = gen.get("defaults", {})
        raw_constraints = gen.get("constraints", {})
        if raw_defaults is None:
            raw_defaults = {}
        if raw_constraints is None:
            raw_constraints = {}
        if not isinstance(raw_defaults, dict):
            if not strict:
                logger.warning(
                    "bundle_index.generation_config_invalid_defaults",
                    path=str(bundle_yaml_path),
                    got_type=type(raw_defaults).__name__,
                )
                return fallback
            raise BundleConfigError(
                f"{bundle_yaml_path}: generation.defaults must be a mapping, "
                f"got {type(raw_defaults).__name__}"
            )
        if not isinstance(raw_constraints, dict):
            if not strict:
                logger.warning(
                    "bundle_index.generation_config_invalid_constraints",
                    path=str(bundle_yaml_path),
                    got_type=type(raw_constraints).__name__,
                )
                return fallback
            raise BundleConfigError(
                f"{bundle_yaml_path}: generation.constraints must be a mapping, "
                f"got {type(raw_constraints).__name__}"
            )

        fd = fallback.defaults
        fc = fallback.constraints

        def _parse_int(raw: dict[str, Any], key: str, default: int) -> int:
            if key not in raw or raw[key] is None:
                return default
            try:
                return int(raw[key])
            except (TypeError, ValueError) as exc:
                raise BundleConfigError(
                    f"{bundle_yaml_path}: {raw[key]!r} is not a valid integer for generation.{key}"
                ) from exc

        def _parse_float(raw: dict[str, Any], key: str, default: float) -> float:
            if key not in raw or raw[key] is None:
                return default
            try:
                return float(raw[key])
            except (TypeError, ValueError) as exc:
                raise BundleConfigError(
                    f"{bundle_yaml_path}: {raw[key]!r} is not a valid float for generation.{key}"
                ) from exc

        def _parse_sampler(key: str, default: Sampler) -> Sampler:
            v = raw_defaults.get(key)
            if v is None:
                return default
            try:
                return Sampler(str(v))
            except ValueError as exc:
                raise BundleConfigError(
                    f"{bundle_yaml_path}: unknown sampler {v!r} in generation.defaults.{key}"
                ) from exc

        def _parse_scheduler(key: str, default: Scheduler) -> Scheduler:
            v = raw_defaults.get(key)
            if v is None:
                return default
            try:
                return Scheduler(str(v))
            except ValueError as exc:
                raise BundleConfigError(
                    f"{bundle_yaml_path}: unknown scheduler {v!r} in generation.defaults.{key}"
                ) from exc

        def _parse_resolution(key: str, default: Resolution) -> Resolution:
            v = raw_defaults.get(key)
            if v is None:
                return default
            try:
                return Resolution(str(v))
            except ValueError as exc:
                raise BundleConfigError(
                    f"{bundle_yaml_path}: unknown resolution tier {v!r} in generation.defaults.{key}"
                ) from exc

        def _parse_samplers_list(key: str) -> frozenset[Sampler]:
            v = raw_constraints.get(key)
            if not v:
                return frozenset()
            result: list[Sampler] = []
            for item in v:
                try:
                    result.append(Sampler(str(item)))
                except ValueError as exc:
                    raise BundleConfigError(
                        f"{bundle_yaml_path}: unknown sampler {item!r} in generation.constraints.{key}"
                    ) from exc
            return frozenset(result)

        def _parse_schedulers_list(key: str) -> frozenset[Scheduler]:
            v = raw_constraints.get(key)
            if not v:
                return frozenset()
            result: list[Scheduler] = []
            for item in v:
                try:
                    result.append(Scheduler(str(item)))
                except ValueError as exc:
                    raise BundleConfigError(
                        f"{bundle_yaml_path}: unknown scheduler {item!r} in generation.constraints.{key}"
                    ) from exc
            return frozenset(result)

        defaults = GenerationDefaults(
            resolution=_parse_resolution("resolution", fd.resolution),
            steps=_parse_int(raw_defaults, "steps", fd.steps),
            cfg=_parse_float(raw_defaults, "cfg", fd.cfg),
            sampler=_parse_sampler("sampler", fd.sampler),
            scheduler=_parse_scheduler("scheduler", fd.scheduler),
            denoise=_parse_float(raw_defaults, "denoise", fd.denoise),
        )
        constraints = GenerationConstraints(
            max_megapixels=_parse_float(raw_constraints, "max_megapixels", fc.max_megapixels),
            latent_multiple=_parse_int(raw_constraints, "latent_multiple", fc.latent_multiple),
            max_edge=_parse_int(raw_constraints, "max_edge", fc.max_edge),
            min_steps=_parse_int(raw_constraints, "min_steps", fc.min_steps),
            max_steps=_parse_int(raw_constraints, "max_steps", fc.max_steps),
            min_cfg=_parse_float(raw_constraints, "min_cfg", fc.min_cfg),
            max_cfg=_parse_float(raw_constraints, "max_cfg", fc.max_cfg),
            allowed_samplers=_parse_samplers_list("allowed_samplers"),
            allowed_schedulers=_parse_schedulers_list("allowed_schedulers"),
            max_batch_size=_parse_int(raw_constraints, "max_batch_size", fc.max_batch_size),
        )

        c = constraints
        if c.latent_multiple <= 0:
            raise BundleConfigError(
                f"{bundle_yaml_path}: latent_multiple must be positive, got {c.latent_multiple}"
            )
        if c.max_megapixels <= 0:
            raise BundleConfigError(
                f"{bundle_yaml_path}: max_megapixels must be positive, got {c.max_megapixels}"
            )
        if c.max_edge < c.latent_multiple:
            raise BundleConfigError(
                f"{bundle_yaml_path}: max_edge ({c.max_edge}) < latent_multiple ({c.latent_multiple})"
            )
        if c.min_steps > c.max_steps:
            raise BundleConfigError(
                f"{bundle_yaml_path}: min_steps ({c.min_steps}) > max_steps ({c.max_steps})"
            )
        if c.min_cfg > c.max_cfg:
            raise BundleConfigError(
                f"{bundle_yaml_path}: min_cfg ({c.min_cfg}) > max_cfg ({c.max_cfg})"
            )
        if c.max_batch_size <= 0:
            raise BundleConfigError(
                f"{bundle_yaml_path}: max_batch_size must be positive, got {c.max_batch_size}"
            )

        return BundleGenerationConfig(defaults=defaults, constraints=constraints)

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
        if client is None:
            raise RuntimeError(
                "BundleIndexService has no HTTP client — invariant violated in __init__"
            )

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

        try:
            extracted_count, extracted_total_bytes = self._extract_tarball_members(
                tarball_path, staging
            )
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

        self._generation_config_cache.clear()
        self._model_filenames_cache.clear()
        self._declared_bytes_cache.clear()
        logger.info("bundle_index.workflow_and_generation_caches_cleared")
        for cb in self._on_resync:
            try:
                cb()
            except Exception:
                logger.exception("bundle_index.on_resync_callback_failed")

    def _extract_tarball_members(self, tarball_path: Path, staging: Path) -> tuple[int, int]:
        """Extract tarball members after applying safety limits and filters."""
        extracted_count = 0
        extracted_total_bytes = 0

        with tarfile.open(tarball_path, mode="r|gz") as tf:
            for member in tf:
                # --- Cap #2: member count ---
                if extracted_count >= self._max_member_count:
                    raise RuntimeError(
                        f"tarball member count exceeds limit ({self._max_member_count}); aborted"
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

                # --- Cap #3: per-member declared size ---
                if filtered.size > self._max_member_size_bytes:
                    raise RuntimeError(
                        f"tarball member {filtered.name!r} declares size "
                        f"{filtered.size} > limit {self._max_member_size_bytes}"
                    )

                # --- Cap #4: total declared uncompressed size ---
                if extracted_total_bytes + filtered.size > self._max_uncompressed_bytes:
                    raise RuntimeError(
                        f"tarball total uncompressed size exceeds limit "
                        f"({self._max_uncompressed_bytes}); aborted at "
                        f"{extracted_total_bytes + filtered.size} bytes"
                    )

                # --- Cap #5: actual extracted bytes per file (lying-size defense) ---
                actual_bytes = self._extract_member_capped(tf, filtered, staging)

                extracted_total_bytes += actual_bytes
                extracted_count += 1

        return extracted_count, extracted_total_bytes

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
            tf.extract(member, dest, set_attrs=False)  # filter applied above
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
        invalid_generation_config_paths: dict[str, Path] = {}
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
            except (BundleConfigError, KeyError, ValueError) as exc:
                logger.warning(
                    "bundle_index.entry_invalid",
                    bundle_name=item.get("name") if isinstance(item, dict) else None,
                    error=str(exc),
                )
                if isinstance(exc, BundleConfigError):
                    name = item.get("name")
                    path = item.get("path")
                    if isinstance(name, str) and isinstance(path, str):
                        invalid_generation_config_paths[name] = self._cache_dir / path
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
        self._invalid_generation_config_paths = invalid_generation_config_paths

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
        bound_workflow = self._parse_workflow(bundle_path, model_type)
        # Keep malformed generation config out of the published index. A
        # fallback is useful for direct legacy reads, but its capability
        # limits are not the bundle's and must not be advertised as such.
        generation = self._parse_generation_config(
            bundle_path / "current" / "bundle.yaml", strict=True
        )
        capabilities = derive_capabilities(bound_workflow, generation)

        declared_writable = {
            f"{role.value}.{parameter}"
            for role, node in bound_workflow.map.nodes.items()
            for parameter in node.inputs
        }
        no_request_source = sorted(declared_writable - capabilities.writable)
        if no_request_source:
            logger.warning(
                "workflow.capability.no_request_source",
                bundle_name=name,
                parameters=no_request_source,
            )

        # Read directly from the locally-known bundle_path (mirrors _parse_hardware/
        # _parse_workflow above) rather than calling get_declared_model_bytes: the
        # index isn't populated yet at build time, so that lookup would 404.
        declared_bytes_yaml_path = bundle_path / "current" / "bundle.yaml"
        declared_model_bytes = self._sum_declared_model_bytes(declared_bytes_yaml_path)
        self._declared_bytes_cache[str(declared_bytes_yaml_path)] = declared_model_bytes

        mapping = BundleMapping(
            bundle_name=name,
            bundle_version=None,
            hardware=hardware,
            readiness_marker=readiness_marker,
            capabilities=capabilities,
            declared_model_bytes=declared_model_bytes,
        )
        return _BundleIndexEntry(
            bundle_name=name,
            bundle_path=path,
            model_type=model_type,
            is_default=is_default,
            mapping=mapping,
            bound_workflow=bound_workflow,
            capabilities=capabilities,
        )

    def _parse_workflow(self, bundle_path: Path, model_type: str) -> BoundWorkflow:
        """Parse and bind ``workflow:`` against the declared API graph eagerly."""
        bundle_yaml = bundle_path / "current" / "bundle.yaml"
        if not bundle_yaml.exists():
            raise BundleDefinitionError(f"bundle.yaml not found at {bundle_yaml}")
        try:
            with bundle_yaml.open() as fh:
                data = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError) as exc:
            raise BundleDefinitionError(
                f"{bundle_yaml}: cannot read bundle definition: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise BundleDefinitionError(
                f"{bundle_yaml}: expected a YAML mapping at top level, got {type(data).__name__}"
            )
        workflow_data = data.get("workflow")
        if not isinstance(workflow_data, dict):
            raise BundleDefinitionError(f"{bundle_yaml}: missing required 'workflow' mapping")
        workflow_file = data.get("workflow_api_file")
        if (
            not isinstance(workflow_file, str)
            or not workflow_file.strip()
            or "/" in workflow_file
            or "\\" in workflow_file
            or workflow_file in {".", ".."}
        ):
            raise BundleDefinitionError(
                f"{bundle_yaml}: workflow_api_file must be a plain relative filename"
            )
        api_path = bundle_yaml.parent / workflow_file
        try:
            with api_path.open() as fh:
                api_graph = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise BundleDefinitionError(
                f"{api_path}: cannot read workflow API JSON: {exc}"
            ) from exc
        if not isinstance(api_graph, dict):
            raise BundleDefinitionError(
                f"{api_path}: expected an API graph mapping, got {type(api_graph).__name__}"
            )
        map_ = parse_workflow_map(workflow_data, bundle_yaml)
        expected_media = MODEL_TYPE_MEDIA.get(model_type)
        if expected_media is not None and map_.media is not expected_media:
            raise BundleDefinitionError(
                f"{bundle_yaml}: workflow.media {map_.media.value!r} does not match "
                f"model_type {model_type!r}"
            )
        return bind_workflow(map_, api_graph)

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
            raise BundleDefinitionError(
                f"{bundle_yaml}: expected a YAML mapping at top level, got {type(data).__name__}"
            )

        hw = data.get("hardware")
        if hw is None:
            raise BundleDefinitionError(f"{bundle_yaml}: missing 'hardware' section")
        if not isinstance(hw, dict):
            raise BundleDefinitionError(
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
                raise BundleDefinitionError(
                    f"{bundle_yaml}: missing required field 'hardware.{field}'"
                )
            value = hw[field]
            # bool is a subclass of int — reject it explicitly
            if isinstance(value, bool):
                raise BundleDefinitionError(
                    f"{bundle_yaml}: 'hardware.{field}' must be an integer, got bool {value!r}"
                )
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                try:
                    return int(value)  # accepts "42", rejects "42.0" / "1e3"
                except ValueError as exc:
                    raise BundleDefinitionError(
                        f"{bundle_yaml}: 'hardware.{field}' must be an integer, got {value!r} (str)"
                    ) from exc
            raise BundleDefinitionError(
                f"{bundle_yaml}: 'hardware.{field}' must be an integer, "
                f"got {value!r} ({type(value).__name__})"
            )

        if "cuda_min_version" not in hw:
            raise BundleDefinitionError(
                f"{bundle_yaml}: missing required field 'hardware.cuda_min_version'"
            )
        cuda_min = str(hw["cuda_min_version"])
        try:
            float(cuda_min)
        except ValueError as exc:
            raise BundleDefinitionError(
                f"{bundle_yaml}: 'hardware.cuda_min_version' must be a numeric string "
                f"(e.g. '12.1'), got {cuda_min!r}"
            ) from exc

        gpu_whitelist_raw = hw.get("gpu_whitelist", [])
        if not isinstance(gpu_whitelist_raw, list):
            raise BundleDefinitionError(
                f"{bundle_yaml}: 'hardware.gpu_whitelist' must be a list, got "
                f"{type(gpu_whitelist_raw).__name__}"
            )
        for idx, entry in enumerate(gpu_whitelist_raw):
            if not isinstance(entry, str):
                raise BundleDefinitionError(
                    f"{bundle_yaml}: 'hardware.gpu_whitelist[{idx}]' must be a string, "
                    f"got {entry!r} ({type(entry).__name__})"
                )

        # Apply default for optional int fields before validation so _require_int
        # produces consistent error messages for invalid values.
        hw.setdefault("comfyui_port", self._default_comfyui_port)

        template_hash_id = hw.get("template_hash_id")
        if template_hash_id is not None:
            if not isinstance(template_hash_id, str):
                raise BundleDefinitionError(
                    f"{bundle_yaml}: 'hardware.template_hash_id' must be a string, got "
                    f"{type(template_hash_id).__name__}"
                )
            if not template_hash_id.strip():
                raise BundleDefinitionError(
                    f"{bundle_yaml}: 'hardware.template_hash_id' must be non-empty"
                )

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
            raise BundleDefinitionError(
                f"{bundle_yaml}: 'readiness_marker' must be a mapping, got {type(marker).__name__}"
            )
        node_class = marker.get("node_class")
        if not isinstance(node_class, str) or not node_class.strip():
            raise BundleDefinitionError(
                f"{bundle_yaml}: 'readiness_marker.node_class' must be a non-empty string"
            )
        return ReadinessMarker(node_class=node_class)

    async def _sync_loop(self) -> None:
        """Periodic background sync loop."""
        while self._running:
            await asyncio.sleep(self._sync_interval * 60)
            if self._running:
                await self.sync()
