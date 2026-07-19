"""Unit tests for src/api/services/library_capabilities.py — pure action resolver."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.api.services.library_capabilities import LibraryAction, resolve_library_actions
from src.core.enums import OutputMediaType
from src.core.library_ref import LibraryAssetSource
from src.core.product import ProductConfig

pytestmark = pytest.mark.unit


def _product_config() -> ProductConfig:
    return MagicMock(spec=ProductConfig)


_ALWAYS = (
    LibraryAction.FAVORITE,
    LibraryAction.RENAME,
    LibraryAction.DOWNLOAD,
    LibraryAction.DELETE,
)


class TestAlwaysPresentActions:
    @pytest.mark.parametrize("media_type", list(OutputMediaType))
    @pytest.mark.parametrize("source", list(LibraryAssetSource))
    @pytest.mark.parametrize("has_generation_metadata", [True, False])
    def test_always_actions_present(
        self,
        media_type: OutputMediaType,
        source: LibraryAssetSource,
        has_generation_metadata: bool,
    ) -> None:
        actions = resolve_library_actions(
            media_type=media_type,
            source=source,
            has_generation_metadata=has_generation_metadata,
            product_config=_product_config(),
        )
        for action in _ALWAYS:
            assert action in actions


class TestImageActions:
    def test_image_gets_image_specific_actions(self) -> None:
        actions = resolve_library_actions(
            media_type=OutputMediaType.IMAGE,
            source=LibraryAssetSource.UPLOAD,
            has_generation_metadata=False,
            product_config=_product_config(),
        )
        assert LibraryAction.REMIX in actions
        assert LibraryAction.CREATE_VARIATION in actions
        assert LibraryAction.ANIMATE in actions
        assert LibraryAction.USE_AS_REFERENCE in actions
        assert LibraryAction.USE_AS_FIRST_FRAME in actions
        assert LibraryAction.USE_AS_LAST_FRAME in actions

    def test_image_does_not_get_video_only_actions(self) -> None:
        actions = resolve_library_actions(
            media_type=OutputMediaType.IMAGE,
            source=LibraryAssetSource.UPLOAD,
            has_generation_metadata=False,
            product_config=_product_config(),
        )
        assert LibraryAction.EXTEND not in actions
        assert LibraryAction.EXTRACT_FRAME not in actions


class TestVideoActions:
    def test_video_gets_video_specific_actions(self) -> None:
        actions = resolve_library_actions(
            media_type=OutputMediaType.VIDEO,
            source=LibraryAssetSource.UPLOAD,
            has_generation_metadata=False,
            product_config=_product_config(),
        )
        assert LibraryAction.REMIX in actions
        assert LibraryAction.EXTEND in actions
        assert LibraryAction.EXTRACT_FRAME in actions

    def test_video_does_not_get_image_only_actions(self) -> None:
        actions = resolve_library_actions(
            media_type=OutputMediaType.VIDEO,
            source=LibraryAssetSource.UPLOAD,
            has_generation_metadata=False,
            product_config=_product_config(),
        )
        assert LibraryAction.CREATE_VARIATION not in actions
        assert LibraryAction.ANIMATE not in actions
        assert LibraryAction.USE_AS_REFERENCE not in actions
        assert LibraryAction.USE_AS_FIRST_FRAME not in actions
        assert LibraryAction.USE_AS_LAST_FRAME not in actions


class TestGenerationMetadataActions:
    @pytest.mark.parametrize("media_type", list(OutputMediaType))
    def test_has_generation_metadata_unlocks_view_and_reproduce(
        self, media_type: OutputMediaType
    ) -> None:
        actions = resolve_library_actions(
            media_type=media_type,
            source=LibraryAssetSource.OUTPUT,
            has_generation_metadata=True,
            product_config=_product_config(),
        )
        assert LibraryAction.VIEW_SETTINGS in actions
        assert LibraryAction.REPRODUCE in actions

    @pytest.mark.parametrize("media_type", list(OutputMediaType))
    def test_no_generation_metadata_excludes_view_and_reproduce(
        self, media_type: OutputMediaType
    ) -> None:
        actions = resolve_library_actions(
            media_type=media_type,
            source=LibraryAssetSource.UPLOAD,
            has_generation_metadata=False,
            product_config=_product_config(),
        )
        assert LibraryAction.VIEW_SETTINGS not in actions
        assert LibraryAction.REPRODUCE not in actions


class TestOrderingStability:
    def test_ordering_deterministic_across_calls(self) -> None:
        first = resolve_library_actions(
            media_type=OutputMediaType.VIDEO,
            source=LibraryAssetSource.OUTPUT,
            has_generation_metadata=True,
            product_config=_product_config(),
        )
        second = resolve_library_actions(
            media_type=OutputMediaType.VIDEO,
            source=LibraryAssetSource.OUTPUT,
            has_generation_metadata=True,
            product_config=_product_config(),
        )
        assert first == second

    def test_always_actions_come_first_in_declared_order(self) -> None:
        actions = resolve_library_actions(
            media_type=OutputMediaType.IMAGE,
            source=LibraryAssetSource.OUTPUT,
            has_generation_metadata=True,
            product_config=_product_config(),
        )
        assert actions[:4] == _ALWAYS

    def test_generation_metadata_actions_come_last(self) -> None:
        actions = resolve_library_actions(
            media_type=OutputMediaType.IMAGE,
            source=LibraryAssetSource.OUTPUT,
            has_generation_metadata=True,
            product_config=_product_config(),
        )
        assert actions[-2:] == (LibraryAction.VIEW_SETTINGS, LibraryAction.REPRODUCE)


class TestFullMatrix:
    @pytest.mark.parametrize("media_type", list(OutputMediaType))
    @pytest.mark.parametrize("source", list(LibraryAssetSource))
    @pytest.mark.parametrize("has_generation_metadata", [True, False])
    def test_matrix_is_pure_and_no_io(
        self,
        media_type: OutputMediaType,
        source: LibraryAssetSource,
        has_generation_metadata: bool,
    ) -> None:
        """Full 2x2x2 matrix: no exceptions, always a non-empty tuple, no duplicates."""
        actions = resolve_library_actions(
            media_type=media_type,
            source=source,
            has_generation_metadata=has_generation_metadata,
            product_config=_product_config(),
        )
        assert isinstance(actions, tuple)
        assert len(actions) == len(set(actions))
        assert len(actions) > 0
