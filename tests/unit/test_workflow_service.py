"""Behaviour tests for the cached bound-workflow facade."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.api.services.workflow.service import WorkflowNotFoundError, WorkflowService


def test_load_without_a_bundle_index_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(WorkflowNotFoundError, match="has no bundle index"):
        WorkflowService().load(tmp_path / "bundle", None)


def test_load_names_the_resolved_path_when_the_index_has_no_workflow(tmp_path: Path) -> None:
    bundle_index = MagicMock()
    bundle_index.get_bound_workflow_for_path.return_value = None
    bundle_dir = tmp_path / "bundle"

    with pytest.raises(WorkflowNotFoundError, match=str((bundle_dir / "current").resolve())):
        WorkflowService(bundle_index).load(bundle_dir, None)


def test_load_caches_a_resolved_version_and_invalidation_forces_a_refetch(tmp_path: Path) -> None:
    bundle_index = MagicMock()
    bound = MagicMock(name="bound-workflow")
    bundle_index.get_bound_workflow_for_path.return_value = bound
    service = WorkflowService(bundle_index)
    bundle_dir = tmp_path / "bundle"

    assert service.load(bundle_dir, "260101-01") is bound
    assert service.load(bundle_dir, "260101-01") is bound
    bundle_index.get_bound_workflow_for_path.assert_called_once_with(bundle_dir, "260101-01")

    service.invalidate_cache()

    assert service.load(bundle_dir, "260101-01") is bound
    assert bundle_index.get_bound_workflow_for_path.call_count == 2


def test_repointed_current_uses_the_new_resolved_version_after_index_resync(tmp_path: Path) -> None:
    """A current symlink identifies the concrete version cache entry.

    An index resync clears cached bound graphs.  Repointing ``current`` also
    changes the resolved key, so it can never silently return a graph for the
    old version even before that callback runs.
    """
    bundle_dir = tmp_path / "bundle"
    version_one = bundle_dir / "260101-01"
    version_two = bundle_dir / "260102-01"
    version_one.mkdir(parents=True)
    version_two.mkdir()
    current = bundle_dir / "current"
    current.symlink_to(version_one.name, target_is_directory=True)

    old_bound = MagicMock(name="old-bound-workflow")
    new_bound = MagicMock(name="new-bound-workflow")
    bundle_index = MagicMock()
    bundle_index.get_bound_workflow_for_path.side_effect = [old_bound, new_bound]
    service = WorkflowService(bundle_index)

    assert service.load(bundle_dir, None) is old_bound
    current.unlink()
    current.symlink_to(version_two.name, target_is_directory=True)

    # The resolved version directory is part of the cache key.  This is
    # stronger than the stale-cache failure mode: the old graph is not served
    # even in the short interval before the resync callback invalidates cache.
    assert service.load(bundle_dir, None) is new_bound
    assert bundle_index.get_bound_workflow_for_path.call_count == 2

    service.invalidate_cache()
    bundle_index.get_bound_workflow_for_path.side_effect = None
    bundle_index.get_bound_workflow_for_path.return_value = new_bound
    assert service.load(bundle_dir, None) is new_bound
    assert bundle_index.get_bound_workflow_for_path.call_count == 3


def test_version_key_falls_back_to_the_unresolved_path_on_resolution_error(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    with patch.object(Path, "resolve", side_effect=OSError("unavailable")):
        assert WorkflowService._version_key(bundle_dir, "260101-01") == str(
            bundle_dir / "260101-01"
        )


def test_apply_requires_an_index_and_passes_a_bundle_scoped_model_lookup() -> None:
    bound = MagicMock(name="bound-workflow")
    with pytest.raises(WorkflowNotFoundError, match="has no bundle index"):
        WorkflowService().apply(
            bound,
            object(),
            media_filenames={},
            filename_prefix="gen",
            bundle_name="wan",
            bundle_version="260101-01",
        )

    bundle_index = MagicMock()
    bundle_index.get_model_filenames.return_value = ["wan.safetensors"]
    service = WorkflowService(bundle_index)
    with patch("src.api.services.workflow.service.apply_bound_workflow", return_value={}) as apply:
        assert (
            service.apply(
                bound,
                object(),
                media_filenames={},
                filename_prefix="gen",
                bundle_name="wan",
                bundle_version="260101-01",
            )
            == {}
        )

    model_filenames = apply.call_args.kwargs["model_filenames"]
    assert model_filenames("unet") == ["wan.safetensors"]
    bundle_index.get_model_filenames.assert_called_once_with("wan", "260101-01", "unet")
