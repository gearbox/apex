"""Small cached facade over already-indexed bound workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.api.services.workflow.applier import apply as apply_bound_workflow

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from src.api.services.bundle_index import BundleIndexService
    from src.api.services.workflow.contract import BoundWorkflow, MediaSlot


class WorkflowNotFoundError(Exception):
    """The requested bundle version has no indexed workflow."""


class WorkflowService:
    """Load indexed bound workflows and apply request values to their graphs."""

    def __init__(self, bundle_index: BundleIndexService | None = None) -> None:
        self._bundles = bundle_index
        self._workflow_cache: dict[str, BoundWorkflow] = {}

    def invalidate_cache(self) -> None:
        self._workflow_cache.clear()

    @staticmethod
    def _version_key(bundle_dir: Path, bundle_version: str | None) -> str:
        version_dir = bundle_dir / (bundle_version or "current")
        try:
            return str(version_dir.resolve())
        except OSError:
            return str(version_dir)

    def load(self, bundle_dir: Path, bundle_version: str | None) -> BoundWorkflow:
        """Return the eagerly-bound workflow cached by its resolved version dir."""
        if self._bundles is None:
            raise WorkflowNotFoundError("Workflow service has no bundle index")
        key = self._version_key(bundle_dir, bundle_version)
        bound = self._workflow_cache.get(key)
        if bound is None:
            bound = self._bundles.get_bound_workflow_for_path(bundle_dir, bundle_version)
            if bound is None:
                raise WorkflowNotFoundError(f"No indexed workflow for {key}")
            self._workflow_cache[key] = bound
        return bound

    def apply(
        self,
        bound: BoundWorkflow,
        request: object,
        *,
        media_filenames: Mapping[MediaSlot, list[str]],
        filename_prefix: str,
        bundle_name: str,
        bundle_version: str | None,
    ) -> dict[str, Any]:
        """Configure an indexed graph, including all declared model loaders."""
        if self._bundles is None:
            raise WorkflowNotFoundError("Workflow service has no bundle index")
        bundles = self._bundles
        return apply_bound_workflow(
            bound,
            request,
            media_filenames=media_filenames,
            filename_prefix=filename_prefix,
            model_filenames=lambda model_type: bundles.get_model_filenames(
                bundle_name, bundle_version, model_type
            ),
        )
