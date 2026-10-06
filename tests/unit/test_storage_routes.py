"""Unit tests for StorageController route handlers.

Tests call ``Handler.fn(self, ...)`` directly to exercise handler logic
without spinning up Litestar's HTTP layer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_502_BAD_GATEWAY,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from src.api.schemas.media import MediaObject, MediaOriginal
from src.api.schemas.user_content import ImageAccess, UploadedImage
from src.api.services.frame_lineage import FrameLineage, FrameLineageReason
from src.api.services.user_content import (
    UserContentError,
    UserContentFrameLineageError,
    UserContentNotFoundError,
    UserContentStorageError,
    UserContentTooLargeError,
    UserContentUnavailableError,
    UserContentValidationError,
)
from src.core.enums import OutputMediaType
from src.core.library_ref import AssetRef, LibraryAssetSource

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_media() -> MediaObject:
    return MediaObject(
        media_type=OutputMediaType.IMAGE,
        original=MediaOriginal(
            url="/v1/content/uploads/abc",
            width=800,
            height=600,
            content_type="image/png",
            size_bytes=1024,
        ),
        variants=[],
    )


def _make_uploaded_image() -> UploadedImage:
    return UploadedImage(
        id=uuid4(),
        storage_key="users/u/uploads/id.png",
        filename="photo.png",
        content_type="image/png",
        size_bytes=1024,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(days=7),
        media=_make_media(),
    )


def _make_db_image(**overrides: object) -> MagicMock:
    img = MagicMock()
    img.id = uuid4()
    img.storage_key = f"users/u/uploads/{img.id}.png"
    img.original_filename = "photo.png"
    img.content_type = "image/png"
    img.size_bytes = 1024
    img.created_at = datetime.now(UTC)
    img.expires_at = datetime.now(UTC) + timedelta(days=7)
    img.width = 800
    img.height = 600
    img.thumbnail_max_edge = None
    for k, v in overrides.items():
        setattr(img, k, v)
    return img


def _make_db_output(**overrides: object) -> MagicMock:
    out = MagicMock()
    out.id = uuid4()
    out.job_id = uuid4()
    out.storage_key = f"users/u/outputs/j/{out.id}.png"
    out.content_type = "image/png"
    out.size_bytes = 2048
    out.output_index = 0
    out.created_at = datetime.now(UTC)
    out.expires_at = datetime.now(UTC) + timedelta(days=7)
    out.width = 512
    out.height = 512
    out.thumbnail_max_edge = None
    for k, v in overrides.items():
        setattr(out, k, v)
    return out


def _upload_form(
    content_type: str = "image/png",
    filename: str = "photo.png",
    data: bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
    *,
    source_asset_ref: str | None = None,
    source_timestamp_ms: str | None = None,
) -> MagicMock:
    upload_file = AsyncMock()
    upload_file.content_type = content_type
    upload_file.filename = filename
    upload_file.read = AsyncMock(return_value=data)
    form = MagicMock()
    form.data = upload_file
    form.source_asset_ref = source_asset_ref
    form.source_timestamp_ms = source_timestamp_ms
    return form


# ---------------------------------------------------------------------------
# upload_image
# ---------------------------------------------------------------------------


class TestUploadImageHandler:
    async def test_invalid_content_type_returns_400(self) -> None:
        from src.api.routes.storage import StorageController

        user_content = AsyncMock()
        response = await StorageController.upload_image.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            data=_upload_form(content_type="image/gif"),
        )

        assert response.status_code == HTTP_400_BAD_REQUEST
        user_content.upload_image.assert_not_awaited()

    async def test_file_too_large_returns_400(self) -> None:
        from src.api.routes.storage import MAX_UPLOAD_SIZE, StorageController

        big_data = b"\x89PNG\r\n" + b"\x00" * (MAX_UPLOAD_SIZE + 1)
        user_content = AsyncMock()
        response = await StorageController.upload_image.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            data=_upload_form(data=big_data),
        )

        assert response.status_code == HTTP_400_BAD_REQUEST

    async def test_empty_file_returns_400(self) -> None:
        from src.api.routes.storage import StorageController

        user_content = AsyncMock()
        response = await StorageController.upload_image.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            data=_upload_form(data=b""),
        )

        assert response.status_code == HTTP_400_BAD_REQUEST

    async def test_successful_upload_returns_201(self) -> None:
        from src.api.routes.storage import StorageController

        uploaded = _make_uploaded_image()
        user_content = AsyncMock()
        user_content.upload_image = AsyncMock(return_value=uploaded)

        response = await StorageController.upload_image.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            data=_upload_form(),
        )

        assert response.status_code == HTTP_201_CREATED

    async def test_validation_error_returns_400(self) -> None:
        from src.api.routes.storage import StorageController

        user_content = AsyncMock()
        user_content.upload_image = AsyncMock(side_effect=UserContentValidationError("bad format"))

        response = await StorageController.upload_image.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            data=_upload_form(),
        )

        assert response.status_code == HTTP_400_BAD_REQUEST

    async def test_too_large_error_returns_413(self) -> None:
        """F4/D4: UserContentTooLargeError (pixel cap exceeded) maps to 413,
        distinct from the generic 400 for other validation failures."""
        from src.api.routes.storage import StorageController

        user_content = AsyncMock()
        user_content.upload_image = AsyncMock(
            side_effect=UserContentTooLargeError("Input image exceeds maximum pixel count")
        )

        response = await StorageController.upload_image.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            data=_upload_form(),
        )

        assert response.status_code == HTTP_413_REQUEST_ENTITY_TOO_LARGE

    async def test_content_error_returns_400(self) -> None:
        from src.api.routes.storage import StorageController

        user_content = AsyncMock()
        user_content.upload_image = AsyncMock(side_effect=UserContentError("upload failed"))

        response = await StorageController.upload_image.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            data=_upload_form(),
        )

        assert response.status_code == HTTP_400_BAD_REQUEST

    async def test_cyrillic_filename_returns_201_not_500(self) -> None:
        """Issue B regression: a non-latin multipart filename must succeed,
        not surface as an opaque 500 from a UnicodeEncodeError deep in R2."""
        from src.api.routes.storage import StorageController

        uploaded = _make_uploaded_image()
        user_content = AsyncMock()
        user_content.upload_image = AsyncMock(return_value=uploaded)

        response = await StorageController.upload_image.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            data=_upload_form(filename="тестовое фото.png"),
        )

        assert response.status_code == HTTP_201_CREATED

    async def test_storage_error_returns_502_upstream_error(self) -> None:
        """D-B3: UserContentStorageError maps to 502, not 500 and not 400 —
        an R2 outage is an upstream failure, not the client's fault. The
        message must not echo the underlying storage exception text."""
        from src.api.routes.storage import StorageController

        user_content = AsyncMock()
        user_content.upload_image = AsyncMock(
            side_effect=UserContentStorageError("Storage backend unavailable (StorageUploadError)")
        )

        response = await StorageController.upload_image.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            data=_upload_form(),
        )

        assert response.status_code == HTTP_502_BAD_GATEWAY
        assert response.content.error == "upstream_error"
        assert response.content.message == "Storage backend unavailable"
        assert "StorageUploadError" not in response.content.message

    async def test_media_unavailable_returns_503_service_unavailable(self) -> None:
        """Ingest capacity/operational failures are local unavailability (503),
        distinct from an upstream R2 failure (502)."""
        from src.api.routes.storage import StorageController

        user_content = AsyncMock()
        user_content.upload_image = AsyncMock(
            side_effect=UserContentUnavailableError("video preparation is temporarily unavailable")
        )

        response = await StorageController.upload_image.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            data=_upload_form(),
        )

        assert response.status_code == HTTP_503_SERVICE_UNAVAILABLE
        assert response.content.error == "service_unavailable"
        assert response.content.message == "Media processing is temporarily unavailable"

    async def test_none_filename_defaults_to_data_png(self) -> None:
        from src.api.routes.storage import StorageController

        uploaded = _make_uploaded_image()
        user_content = AsyncMock()
        user_content.upload_image = AsyncMock(return_value=uploaded)

        response = await StorageController.upload_image.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            data=_upload_form(filename=None),  # type: ignore[arg-type]
        )

        assert response.status_code == HTTP_201_CREATED
        call_kwargs = user_content.upload_image.call_args.kwargs
        assert call_kwargs["filename"] == "data.png"


# Fixed (not uuid4): parametrize ids must be identical across xdist workers.
_LINEAGE_SOURCE_ID = UUID("0f6c1c3e-6a54-4d0e-9d57-3c2d9b6f7a11")


class TestUploadFrameLineage:
    """A client-captured frame names its source video via two optional multipart fields."""

    _LINEAGE_ERROR = "invalid_frame_lineage"
    _LINEAGE_MESSAGE = "Frame source is not available"

    @staticmethod
    async def _upload(user_content: AsyncMock, form: MagicMock) -> object:
        from src.api.routes.storage import StorageController

        return await StorageController.upload_image.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            data=form,
        )

    async def test_valid_lineage_is_parsed_and_passed_to_the_service(self) -> None:
        source_id = uuid4()
        user_content = AsyncMock()
        user_content.upload_image = AsyncMock(return_value=_make_uploaded_image())

        response = await self._upload(
            user_content,
            _upload_form(source_asset_ref=f"output:{source_id}", source_timestamp_ms="1500"),
        )

        assert response.status_code == HTTP_201_CREATED  # type: ignore[attr-defined]
        assert user_content.upload_image.call_args.kwargs["lineage"] == FrameLineage(
            source=AssetRef(source=LibraryAssetSource.OUTPUT, asset_id=source_id),
            timestamp_ms=1500,
        )

    async def test_no_lineage_fields_is_an_ordinary_upload(self) -> None:
        user_content = AsyncMock()
        user_content.upload_image = AsyncMock(return_value=_make_uploaded_image())

        response = await self._upload(user_content, _upload_form())

        assert response.status_code == HTTP_201_CREATED  # type: ignore[attr-defined]
        assert user_content.upload_image.call_args.kwargs["lineage"] is None

    @pytest.mark.parametrize(
        ("asset_ref", "timestamp"),
        [
            (f"upload:{_LINEAGE_SOURCE_ID}", None),  # partial
            (None, "100"),  # partial
            ("not-a-ref", "100"),  # malformed ref
            (f"upload:{_LINEAGE_SOURCE_ID}", "1e3"),  # malformed timestamp
            (f"upload:{_LINEAGE_SOURCE_ID}", "-1"),
            (f"upload:{_LINEAGE_SOURCE_ID}", "12.0"),
            (f"upload:{_LINEAGE_SOURCE_ID}", ""),
            (f"upload:{_LINEAGE_SOURCE_ID}", "99999999999"),  # oversized
        ],
    )
    async def test_malformed_or_partial_lineage_is_rejected_before_any_work(
        self, asset_ref: str | None, timestamp: str | None
    ) -> None:
        user_content = AsyncMock()
        form = _upload_form(source_asset_ref=asset_ref, source_timestamp_ms=timestamp)

        response = await self._upload(user_content, form)

        assert response.status_code == HTTP_400_BAD_REQUEST  # type: ignore[attr-defined]
        assert response.content.error == self._LINEAGE_ERROR  # type: ignore[attr-defined]
        assert response.content.message == self._LINEAGE_MESSAGE  # type: ignore[attr-defined]
        user_content.upload_image.assert_not_called()
        form.data.read.assert_not_called()  # rejected before the body is read

    @pytest.mark.parametrize("content_type", ["video/mp4", "video/webm", "video/quicktime"])
    async def test_video_upload_with_lineage_is_rejected_before_the_service(
        self, content_type: str
    ) -> None:
        user_content = AsyncMock()
        form = _upload_form(
            content_type=content_type,
            source_asset_ref=f"upload:{uuid4()}",
            source_timestamp_ms="0",
        )

        response = await self._upload(user_content, form)

        assert response.status_code == HTTP_400_BAD_REQUEST  # type: ignore[attr-defined]
        assert response.content.error == self._LINEAGE_ERROR  # type: ignore[attr-defined]
        user_content.upload_image.assert_not_called()

    @pytest.mark.parametrize("reason", list(FrameLineageReason))
    async def test_every_service_reason_maps_to_the_same_public_error(
        self, reason: FrameLineageReason
    ) -> None:
        """No existence oracle: status, code and message never vary by reason."""
        user_content = AsyncMock()
        user_content.upload_image = AsyncMock(side_effect=UserContentFrameLineageError(reason))

        response = await self._upload(
            user_content,
            _upload_form(source_asset_ref=f"upload:{uuid4()}", source_timestamp_ms="10"),
        )

        assert response.status_code == HTTP_400_BAD_REQUEST  # type: ignore[attr-defined]
        assert response.content.error == self._LINEAGE_ERROR  # type: ignore[attr-defined]
        assert response.content.message == self._LINEAGE_MESSAGE  # type: ignore[attr-defined]
        assert reason.value not in response.content.message  # type: ignore[attr-defined]

    async def test_lineage_error_is_not_swallowed_by_the_generic_validation_branch(self) -> None:
        """P2 — ``UserContentFrameLineageError`` subclasses ``UserContentValidationError``."""
        assert issubclass(UserContentFrameLineageError, UserContentValidationError)
        user_content = AsyncMock()
        user_content.upload_image = AsyncMock(
            side_effect=UserContentFrameLineageError(FrameLineageReason.SOURCE_UNAVAILABLE)
        )

        response = await self._upload(
            user_content,
            _upload_form(source_asset_ref=f"upload:{uuid4()}", source_timestamp_ms="10"),
        )

        assert response.content.error != "validation_error"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# get_upload_access
# ---------------------------------------------------------------------------


class TestGetUploadAccessHandler:
    async def test_returns_200_with_presigned_url(self) -> None:
        from src.api.routes.storage import StorageController

        access = ImageAccess(
            storage_key="users/u/uploads/id.png",
            presigned_url="https://r2.example.com/key?sig=abc",
            content_type="image/png",
            size_bytes=1024,
            expires_in_seconds=3600,
        )
        user_content = AsyncMock()
        user_content.get_upload_access = AsyncMock(return_value=access)

        response = await StorageController.get_upload_access.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            image_id=uuid4(),
        )

        assert response.status_code == HTTP_200_OK

    async def test_returns_404_when_not_found(self) -> None:
        from src.api.routes.storage import StorageController

        user_content = AsyncMock()
        user_content.get_upload_access = AsyncMock(side_effect=UserContentNotFoundError())

        response = await StorageController.get_upload_access.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            image_id=uuid4(),
        )

        assert response.status_code == HTTP_404_NOT_FOUND


# ---------------------------------------------------------------------------
# download_upload
# ---------------------------------------------------------------------------


class TestDownloadUploadHandler:
    async def test_returns_200_with_raw_bytes(self) -> None:
        from src.api.routes.storage import StorageController

        img = _make_db_image()
        user_content = AsyncMock()
        user_content.get_upload = AsyncMock(return_value=img)
        user_content.download_upload = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n")

        response = await StorageController.download_upload.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            image_id=uuid4(),
        )

        assert response.status_code == HTTP_200_OK

    async def test_returns_404_when_image_not_in_db(self) -> None:
        from src.api.routes.storage import StorageController

        user_content = AsyncMock()
        user_content.get_upload = AsyncMock(return_value=None)

        response = await StorageController.download_upload.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            image_id=uuid4(),
        )

        assert response.status_code == HTTP_404_NOT_FOUND

    async def test_returns_404_when_storage_missing(self) -> None:
        from src.api.routes.storage import StorageController

        img = _make_db_image()
        user_content = AsyncMock()
        user_content.get_upload = AsyncMock(return_value=img)
        user_content.download_upload = AsyncMock(side_effect=UserContentNotFoundError())

        response = await StorageController.download_upload.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            image_id=uuid4(),
        )

        assert response.status_code == HTTP_404_NOT_FOUND


# ---------------------------------------------------------------------------
# Output endpoints
# ---------------------------------------------------------------------------


class TestOutputHandlers:
    async def test_get_output_access_returns_200(self) -> None:
        from src.api.routes.storage import StorageController

        access = ImageAccess(
            storage_key="users/u/outputs/j/id.png",
            presigned_url="https://r2.example.com/out?sig=abc",
            content_type="image/png",
            size_bytes=2048,
            expires_in_seconds=3600,
        )
        user_content = AsyncMock()
        user_content.get_output_access = AsyncMock(return_value=access)

        response = await StorageController.get_output_access.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            output_id=uuid4(),
        )

        assert response.status_code == HTTP_200_OK

    async def test_get_output_access_returns_404(self) -> None:
        from src.api.routes.storage import StorageController

        user_content = AsyncMock()
        user_content.get_output_access = AsyncMock(side_effect=UserContentNotFoundError())

        response = await StorageController.get_output_access.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            output_id=uuid4(),
        )

        assert response.status_code == HTTP_404_NOT_FOUND

    async def test_download_output_returns_200(self) -> None:
        from src.api.routes.storage import StorageController

        out = _make_db_output()
        user_content = AsyncMock()
        user_content.get_output = AsyncMock(return_value=out)
        user_content.download_output = AsyncMock(return_value=b"\x89PNG")

        response = await StorageController.download_output.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            output_id=uuid4(),
        )

        assert response.status_code == HTTP_200_OK

    async def test_download_output_returns_404_when_not_in_db(self) -> None:
        from src.api.routes.storage import StorageController

        user_content = AsyncMock()
        user_content.get_output = AsyncMock(return_value=None)

        response = await StorageController.download_output.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            output_id=uuid4(),
        )

        assert response.status_code == HTTP_404_NOT_FOUND

    async def test_download_output_returns_404_when_storage_missing(self) -> None:
        from src.api.routes.storage import StorageController

        out = _make_db_output()
        user_content = AsyncMock()
        user_content.get_output = AsyncMock(return_value=out)
        user_content.download_output = AsyncMock(side_effect=UserContentNotFoundError())

        response = await StorageController.download_output.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            output_id=uuid4(),
        )

        assert response.status_code == HTTP_404_NOT_FOUND

    async def test_list_outputs_returns_empty_page(self) -> None:
        from src.api.routes.storage import StorageController

        user_content = AsyncMock()
        user_content.list_user_outputs = AsyncMock(return_value=[])
        user_content.batch_output_derivatives = AsyncMock(return_value={})

        result = await StorageController.list_outputs.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
        )

        assert result.items == []
        assert result.has_more is False

    async def test_list_outputs_with_cursor(self) -> None:
        from src.api.routes.storage import StorageController
        from src.api.schemas.pagination import encode_cursor

        cursor = encode_cursor(datetime.now(UTC), uuid4())
        user_content = AsyncMock()
        user_content.list_user_outputs = AsyncMock(return_value=[])
        user_content.batch_output_derivatives = AsyncMock(return_value={})

        result = await StorageController.list_outputs.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            cursor=cursor,
        )

        assert result.items == []
        call_kwargs = user_content.list_user_outputs.call_args.kwargs
        assert call_kwargs["cursor_ts"] is not None

    async def test_list_outputs_has_more_when_extra_items(self) -> None:
        from src.api.routes.storage import StorageController

        outs = [_make_db_output() for _ in range(3)]
        user_content = AsyncMock()
        user_content.list_user_outputs = AsyncMock(return_value=outs)
        user_content.batch_output_derivatives = AsyncMock(return_value={})

        result = await StorageController.list_outputs.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            limit=2,
        )

        assert result.has_more is True
        assert len(result.items) == 2
        assert result.next_cursor is not None

    async def test_list_job_outputs_returns_200(self) -> None:
        from src.api.routes.storage import StorageController

        outs = [_make_db_output()]
        user_content = AsyncMock()
        user_content.list_job_outputs = AsyncMock(return_value=outs)
        user_content.batch_output_derivatives = AsyncMock(return_value={})

        response = await StorageController.list_job_outputs.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            job_id=uuid4(),
        )

        assert response.status_code == HTTP_200_OK

    async def test_list_job_outputs_returns_404_when_not_found(self) -> None:
        from src.api.routes.storage import StorageController

        user_content = AsyncMock()
        user_content.list_job_outputs = AsyncMock(side_effect=UserContentNotFoundError())

        response = await StorageController.list_job_outputs.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            user_content=user_content,
            job_id=uuid4(),
        )

        assert response.status_code == HTTP_404_NOT_FOUND
