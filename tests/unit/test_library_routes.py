"""Unit tests for LibraryController route handlers (mocked LibraryService).

Calls handler functions directly via ``Controller.method.fn(...)`` (see
test_billing_routes.py for the same convention) — this exercises routing
logic (status codes, error envelopes, NotFoundException) without a DB.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from litestar.exceptions import NotFoundException
from litestar.status_codes import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from src.api.routes.library import LibraryController
from src.api.schemas.library import (
    BulkDelete,
    BulkOperationItemResult,
    BulkOperationResult,
    LibraryAssetDetail,
    LibraryAssetPatch,
    LibraryDescendants,
    LibraryGroupDetail,
    LibraryLineageGraph,
    LineageNode,
)
from src.api.schemas.media import MediaObject, MediaOriginal
from src.api.schemas.pagination import CursorPage
from src.api.services.library import (
    LibraryBulkValidationError,
    LibraryProjectNotFoundError,
    LibraryTagNotFoundError,
    LibraryValidationError,
)
from src.core.enums import OutputMediaType
from src.core.library_ref import LibraryAssetSource
from src.core.product_registry import VEX_CONFIG

pytestmark = pytest.mark.unit


def _detail(**overrides: object) -> LibraryAssetDetail:
    from src.api.schemas.library import LibraryDescendants
    from src.api.schemas.media import MediaObject, MediaOriginal
    from src.core.enums import OutputMediaType
    from src.core.library_ref import LibraryAssetSource

    defaults: dict[str, object] = {
        "asset_ref": f"upload:{uuid4()}",
        "source": LibraryAssetSource.UPLOAD,
        "media": MediaObject(
            media_type=OutputMediaType.IMAGE,
            original=MediaOriginal(
                url="/v1/content/uploads/x",
                content_type="image/png",
                size_bytes=1,
            ),
            variants=[],
        ),
        "created_at": MagicMock(),
        "expires_at": MagicMock(),
        "display_title": None,
        "original_filename": "photo.png",
        "is_favorite": False,
        "duration_ms": None,
        "job_id": None,
        "output_count": None,
        "model": None,
        "generation_type": None,
        "available_actions": (),
        "prompt": None,
        "negative_prompt": None,
        "provider": None,
        "aspect_ratio": None,
        "token_cost": None,
        "completed_at": None,
        "lineage": None,
        "descendants": LibraryDescendants(job_count=0, frame_count=0),
    } | overrides
    return LibraryAssetDetail(**defaults)  # type: ignore[arg-type]


class TestListAssets:
    async def test_invalid_cursor_returns_400(self) -> None:
        service = AsyncMock()
        service.list_assets.side_effect = ValueError("bad cursor")

        response = await LibraryController.list_assets.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            product_config=VEX_CONFIG,
            session=MagicMock(),
            library_service=service,
        )
        assert response.status_code == HTTP_400_BAD_REQUEST
        assert response.content.error == "invalid_cursor"

    async def test_success_returns_page(self) -> None:
        page: CursorPage[LibraryAssetDetail] = CursorPage(
            items=[], limit=30, has_more=False, next_cursor=None
        )
        service = AsyncMock()
        service.list_assets.return_value = page

        response = await LibraryController.list_assets.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            product_config=VEX_CONFIG,
            session=MagicMock(),
            library_service=service,
        )
        assert response.content is page


class TestGetAssetDetail:
    async def test_none_returns_404(self) -> None:
        service = AsyncMock()
        service.get_asset_detail.return_value = None

        response = await LibraryController.get_asset_detail.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            product_config=VEX_CONFIG,
            asset_ref="bogus",
            session=MagicMock(),
            library_service=service,
        )
        assert response.status_code == HTTP_404_NOT_FOUND
        assert response.content.error == "not_found"

    async def test_found_returns_detail(self) -> None:
        detail = _detail()
        service = AsyncMock()
        service.get_asset_detail.return_value = detail

        response = await LibraryController.get_asset_detail.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            product_config=VEX_CONFIG,
            asset_ref=detail.asset_ref,
            session=MagicMock(),
            library_service=service,
        )
        assert response.content is detail


class TestGetGroupDetail:
    async def test_none_returns_404(self) -> None:
        service = AsyncMock()
        service.get_group_detail.return_value = None

        response = await LibraryController.get_group_detail.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            job_id=uuid4(),
            session=MagicMock(),
            library_service=service,
        )
        assert response.status_code == HTTP_404_NOT_FOUND

    async def test_found_returns_detail(self) -> None:
        group_detail = MagicMock(spec=LibraryGroupDetail)
        service = AsyncMock()
        service.get_group_detail.return_value = group_detail

        response = await LibraryController.get_group_detail.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            job_id=uuid4(),
            session=MagicMock(),
            library_service=service,
        )
        assert response.content is group_detail


class TestPatchAsset:
    async def test_validation_error_returns_400(self) -> None:
        service = AsyncMock()
        service.patch_asset.side_effect = LibraryValidationError("too long")

        response = await LibraryController.patch_asset.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            product_config=VEX_CONFIG,
            asset_ref="upload:x",
            data=LibraryAssetPatch(display_title="x" * 300),
            session=MagicMock(),
            library_service=service,
        )
        assert response.status_code == HTTP_400_BAD_REQUEST
        assert response.content.error == "validation_error"

    async def test_none_returns_404(self) -> None:
        service = AsyncMock()
        service.patch_asset.return_value = None

        response = await LibraryController.patch_asset.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            product_config=VEX_CONFIG,
            asset_ref="upload:bogus",
            data=LibraryAssetPatch(display_title="x"),
            session=MagicMock(),
            library_service=service,
        )
        assert response.status_code == HTTP_404_NOT_FOUND

    async def test_unknown_project_id_returns_404(self) -> None:
        service = AsyncMock()
        service.patch_asset.side_effect = LibraryProjectNotFoundError(uuid4())

        response = await LibraryController.patch_asset.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            product_config=VEX_CONFIG,
            asset_ref="upload:x",
            data=LibraryAssetPatch(project_id=uuid4()),
            session=MagicMock(),
            library_service=service,
        )
        assert response.status_code == HTTP_404_NOT_FOUND
        assert response.content.error == "not_found"

    async def test_unknown_tag_id_returns_404(self) -> None:
        service = AsyncMock()
        service.patch_asset.side_effect = LibraryTagNotFoundError([uuid4()])

        response = await LibraryController.patch_asset.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            product_config=VEX_CONFIG,
            asset_ref="upload:x",
            data=LibraryAssetPatch(tag_ids=[uuid4()]),
            session=MagicMock(),
            library_service=service,
        )
        assert response.status_code == HTTP_404_NOT_FOUND
        assert response.content.error == "not_found"

    async def test_success_returns_detail(self) -> None:
        detail = _detail(display_title="New Title")
        service = AsyncMock()
        service.patch_asset.return_value = detail

        response = await LibraryController.patch_asset.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            product_config=VEX_CONFIG,
            asset_ref=detail.asset_ref,
            data=LibraryAssetPatch(display_title="New Title"),
            session=MagicMock(),
            library_service=service,
        )
        assert response.content is detail


class TestBulkApply:
    async def test_success_returns_result(self) -> None:
        ref = "upload:x"
        result = BulkOperationResult(
            op="set_favorite",
            results=[BulkOperationItemResult(asset_ref=ref, success=True)],
            succeeded=1,
            failed=0,
        )
        service = AsyncMock()
        service.bulk_apply.return_value = result

        response = await LibraryController.bulk_apply.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            data=BulkDelete(asset_refs=[ref]),
            session=MagicMock(),
            library_service=service,
            content_proxy=MagicMock(),
        )
        assert response.content is result

    async def test_invalid_refs_returns_400(self) -> None:
        service = AsyncMock()
        service.bulk_apply.side_effect = LibraryBulkValidationError(["upload:bogus"])

        response = await LibraryController.bulk_apply.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            data=BulkDelete(asset_refs=["upload:bogus"]),
            session=MagicMock(),
            library_service=service,
            content_proxy=MagicMock(),
        )
        assert response.status_code == HTTP_400_BAD_REQUEST
        assert response.content.error == "invalid_asset_refs"
        assert response.content.detail == {"invalid_refs": ["upload:bogus"]}

    async def test_unknown_project_id_returns_404(self) -> None:
        service = AsyncMock()
        service.bulk_apply.side_effect = LibraryProjectNotFoundError(uuid4())

        response = await LibraryController.bulk_apply.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            data=BulkDelete(asset_refs=["upload:x"]),
            session=MagicMock(),
            library_service=service,
            content_proxy=MagicMock(),
        )
        assert response.status_code == HTTP_404_NOT_FOUND
        assert response.content.error == "not_found"

    async def test_unknown_tag_id_returns_404(self) -> None:
        service = AsyncMock()
        service.bulk_apply.side_effect = LibraryTagNotFoundError([uuid4()])

        response = await LibraryController.bulk_apply.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            data=BulkDelete(asset_refs=["upload:x"]),
            session=MagicMock(),
            library_service=service,
            content_proxy=MagicMock(),
        )
        assert response.status_code == HTTP_404_NOT_FOUND
        assert response.content.error == "not_found"


class TestGetAssetLineage:
    async def test_none_returns_404(self) -> None:
        service = AsyncMock()
        service.get_lineage_graph.return_value = None

        response = await LibraryController.get_asset_lineage.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            asset_ref="upload:bogus",
            session=MagicMock(),
            library_service=service,
        )
        assert response.status_code == HTTP_404_NOT_FOUND
        assert response.content.error == "not_found"

    async def test_found_returns_graph(self) -> None:
        node = LineageNode(
            asset_ref=f"upload:{uuid4()}",
            source=LibraryAssetSource.UPLOAD,
            media=MediaObject(
                media_type=OutputMediaType.IMAGE,
                original=MediaOriginal(
                    url="/v1/content/uploads/x", content_type="image/png", size_bytes=1
                ),
                variants=[],
            ),
            created_at=MagicMock(),
        )
        graph = LibraryLineageGraph(
            focus=node,
            ancestors=(),
            descendants=(),
            descendant_totals=LibraryDescendants(job_count=0, frame_count=0),
            ancestors_truncated=False,
            descendants_truncated=False,
        )
        service = AsyncMock()
        service.get_lineage_graph.return_value = graph

        response = await LibraryController.get_asset_lineage.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            asset_ref=node.asset_ref,
            session=MagicMock(),
            library_service=service,
        )
        assert response.content is graph


class TestFavoriteIdempotency:
    async def test_add_favorite_found_both_calls_succeed(self) -> None:
        service = AsyncMock()
        service.set_favorite.return_value = True

        # Idempotent: calling twice with the same (found) result never raises.
        await LibraryController.add_favorite.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            asset_ref="upload:x",
            session=MagicMock(),
            library_service=service,
        )
        await LibraryController.add_favorite.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            asset_ref="upload:x",
            session=MagicMock(),
            library_service=service,
        )
        assert service.set_favorite.await_count == 2

    async def test_add_favorite_not_found_raises_404(self) -> None:
        service = AsyncMock()
        service.set_favorite.return_value = False

        with pytest.raises(NotFoundException):
            await LibraryController.add_favorite.fn(
                MagicMock(),
                current_user_id=uuid4(),
                product_id="vex",
                asset_ref="upload:x",
                session=MagicMock(),
                library_service=service,
            )

    async def test_remove_favorite_found_both_calls_succeed(self) -> None:
        service = AsyncMock()
        service.set_favorite.return_value = True

        await LibraryController.remove_favorite.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            asset_ref="upload:x",
            session=MagicMock(),
            library_service=service,
        )
        await LibraryController.remove_favorite.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            asset_ref="upload:x",
            session=MagicMock(),
            library_service=service,
        )
        assert service.set_favorite.await_count == 2

    async def test_remove_favorite_not_found_raises_404(self) -> None:
        service = AsyncMock()
        service.set_favorite.return_value = False

        with pytest.raises(NotFoundException):
            await LibraryController.remove_favorite.fn(
                MagicMock(),
                current_user_id=uuid4(),
                product_id="vex",
                asset_ref="upload:x",
                session=MagicMock(),
                library_service=service,
            )


class TestDeleteAsset:
    async def test_found_deletes(self) -> None:
        service = AsyncMock()
        service.delete_asset.return_value = True

        await LibraryController.delete_asset.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            asset_ref="upload:x",
            session=MagicMock(),
            library_service=service,
            content_proxy=MagicMock(),
        )
        service.delete_asset.assert_awaited_once()

    async def test_not_found_raises_404(self) -> None:
        service = AsyncMock()
        service.delete_asset.return_value = False

        with pytest.raises(NotFoundException):
            await LibraryController.delete_asset.fn(
                MagicMock(),
                current_user_id=uuid4(),
                product_id="vex",
                asset_ref="upload:x",
                session=MagicMock(),
                library_service=service,
                content_proxy=MagicMock(),
            )
