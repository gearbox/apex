"""Tests for GrokJobService image-path thumbnail generation (FIX-1)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.grok import GrokImageResult
from src.api.services.grok.job_service import GrokJobService
from src.api.services.image_thumbnail import GeneratedThumbnail, ThumbnailResult
from src.core.thumbnails import ThumbnailSpec

pytestmark = pytest.mark.unit

_FAKE_WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20
_SM_SPEC = ThumbnailSpec("sm", 150)
_MD_SPEC = ThumbnailSpec("md", 512)


def _make_thumbnails() -> list[GeneratedThumbnail]:
    return [
        GeneratedThumbnail(
            spec=_SM_SPEC, result=ThumbnailResult(data=_FAKE_WEBP, width=100, height=56)
        ),
        GeneratedThumbnail(
            spec=_MD_SPEC, result=ThumbnailResult(data=_FAKE_WEBP, width=400, height=225)
        ),
    ]


def _make_service() -> tuple[GrokJobService, MagicMock]:
    storage = MagicMock()
    storage.build_storage_key = MagicMock(return_value="users/u/outputs/j/f.jpg")
    storage._settings = MagicMock()
    storage._settings.bucket_name = "bucket"

    client_ctx = AsyncMock()
    client_mock = AsyncMock()
    client_ctx.__aenter__ = AsyncMock(return_value=client_mock)
    client_ctx.__aexit__ = AsyncMock(return_value=False)
    storage._get_client = MagicMock(return_value=client_ctx)

    grok_client = MagicMock()
    svc = GrokJobService(grok_client=grok_client, storage=storage, retention_days=7)

    image_data = b"\xff\xd8\xff\xe0jpeg"
    http_mock = AsyncMock()
    response_mock = MagicMock()
    response_mock.raise_for_status = MagicMock()
    response_mock.content = image_data
    response_mock.headers = {"content-type": "image/jpeg"}
    http_mock.get = AsyncMock(return_value=response_mock)
    svc._http_client = http_mock
    return svc, client_mock


async def test_store_image_result_creates_sm_and_md_thumbnails() -> None:
    svc, client_mock = _make_service()

    output_id = uuid4()
    sm_thumb_id = uuid4()
    md_thumb_id = uuid4()

    created: list[dict[str, object]] = []

    async def capture_create(**kwargs: object) -> MagicMock:
        created.append(dict(kwargs))
        m = MagicMock()
        m.id = uuid4()
        return m

    output_repo = MagicMock()
    output_repo.create = capture_create

    result = GrokImageResult(
        url="https://cdn.xai.com/image.jpg", base64_data=None, revised_prompt=None
    )

    with (
        patch(
            "src.api.services.grok.job_service.new_id",
            side_effect=[output_id, sm_thumb_id, md_thumb_id],
        ),
        patch(
            "src.api.services.grok.job_service.make_image_thumbnails",
            new=AsyncMock(return_value=_make_thumbnails()),
        ),
    ):
        await svc._store_image_result(
            session=AsyncMock(),
            output_repo=output_repo,  # type: ignore[arg-type]
            user_id=uuid4(),
            job_id=uuid4(),
            result=result,
            output_index=0,
            input_image_id=None,
            product_id="vex",
        )

    assert len(created) == 3, f"Expected 3 creates, got {len(created)}: {created}"
    parent_create, sm_create, md_create = created

    assert "is_thumbnail" not in parent_create

    assert sm_create["is_thumbnail"] is True
    assert sm_create["parent_output_id"] == output_id
    assert sm_create["thumbnail_max_edge"] == 150
    assert sm_create["width"] == 100
    assert sm_create["height"] == 56
    assert sm_create["expires_at"] == parent_create["expires_at"]

    assert md_create["is_thumbnail"] is True
    assert md_create["parent_output_id"] == output_id
    assert md_create["thumbnail_max_edge"] == 512
    assert md_create["expires_at"] == parent_create["expires_at"]

    assert client_mock.put_object.await_count == 3


async def test_store_image_result_thumbnail_failure_does_not_fail_parent() -> None:
    svc, client_mock = _make_service()

    created: list[dict[str, object]] = []

    async def capture_create(**kwargs: object) -> MagicMock:
        created.append(dict(kwargs))
        m = MagicMock()
        m.id = uuid4()
        return m

    output_repo = MagicMock()
    output_repo.create = capture_create

    # First put_object call (the parent output) succeeds; subsequent calls
    # (thumbnail variants) raise.
    client_mock.put_object = AsyncMock(
        side_effect=[None, RuntimeError("r2 down"), RuntimeError("r2 down")]
    )

    result = GrokImageResult(
        url="https://cdn.xai.com/image.jpg", base64_data=None, revised_prompt=None
    )

    with (
        patch("src.api.services.grok.job_service.new_id", side_effect=[uuid4(), uuid4(), uuid4()]),
        patch(
            "src.api.services.grok.job_service.make_image_thumbnails",
            new=AsyncMock(return_value=_make_thumbnails()),
        ),
    ):
        await svc._store_image_result(
            session=AsyncMock(),
            output_repo=output_repo,  # type: ignore[arg-type]
            user_id=uuid4(),
            job_id=uuid4(),
            result=result,
            output_index=0,
            input_image_id=None,
            product_id="vex",
        )

    # Only the parent output row was created; both thumbnail variants failed
    # and were skipped without raising.
    assert len(created) == 1
    assert "is_thumbnail" not in created[0]
