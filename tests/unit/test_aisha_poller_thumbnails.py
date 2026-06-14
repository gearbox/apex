"""Tests for AishaJobPoller thumbnail generation and type filtering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.image_thumbnail import ThumbnailResult
from src.workers.aisha_job_poller import AishaJobPoller, AishaPollerConfig

pytestmark = pytest.mark.unit

_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # not a real PNG, but enough for mocking
_FAKE_WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20


def _make_config(**kwargs: object) -> AishaPollerConfig:
    defaults: dict[str, object] = {
        "enabled": True,
        "tick_interval_seconds": 0.01,
        "max_concurrent_polls": 4,
        "job_age_warning_seconds": 300,
        "job_age_timeout_seconds": 1800,
        "comfyui_request_timeout_seconds": 5.0,
        "tunnel_allowed_suffix": "gpu-domain.com",
        "tunnel_allowed_prefix": "gpu-",
        "retention_days": 7,
    } | kwargs
    return AishaPollerConfig(**defaults)  # type: ignore[arg-type]


def _make_job() -> MagicMock:
    gpu_session = MagicMock()
    gpu_session.id = uuid4()
    gpu_session.tunnel_hostname = "gpu-node1.gpu-domain.com"

    job = MagicMock()
    job.id = uuid4()
    job.user_id = uuid4()
    job.product_id = "vex"
    job.status = "queued"
    job.external_request_id = "prompt-abc"
    job.gpu_session_id = gpu_session.id
    job.gpu_session = gpu_session
    job.started_at = datetime.now(UTC) - timedelta(seconds=10)
    job.created_at = datetime.now(UTC) - timedelta(seconds=10)
    return job


def _make_r2_upload_result(file_id: object = None) -> MagicMock:
    result = MagicMock()
    result.id = file_id or uuid4()
    result.storage_key = f"users/u/outputs/j/{result.id}.png"
    return result


def _make_poller(r2: MagicMock | None = None) -> AishaJobPoller:
    session_factory = MagicMock()
    session_factory.return_value = AsyncMock()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return AishaJobPoller(
        session_factory=session_factory,
        event_bus=None,
        billing_service=AsyncMock(),
        r2_storage=r2,
        config=_make_config(),
    )


# ---------------------------------------------------------------------------
# _collect_image_infos — type filtering
# ---------------------------------------------------------------------------


class TestCollectImageInfosFiltering:
    def test_only_output_type_is_included(self) -> None:
        entry = {
            "outputs": {
                "1": {
                    "images": [
                        {"filename": "out.png", "type": "output"},
                        {"filename": "prev.png", "type": "temp"},
                    ]
                }
            }
        }
        result = AishaJobPoller._collect_image_infos(entry)
        assert len(result) == 1
        assert result[0]["filename"] == "out.png"

    def test_images_without_type_key_are_excluded(self) -> None:
        entry = {
            "outputs": {
                "1": {
                    "images": [
                        {"filename": "no_type.png"},  # no "type" key
                        {"filename": "good.png", "type": "output"},
                    ]
                }
            }
        }
        result = AishaJobPoller._collect_image_infos(entry)
        assert len(result) == 1
        assert result[0]["filename"] == "good.png"

    def test_multiple_output_type_images_all_included(self) -> None:
        entry = {
            "outputs": {
                "1": {
                    "images": [
                        {"filename": "a.png", "type": "output"},
                        {"filename": "b.png", "type": "output"},
                    ]
                },
                "2": {
                    "images": [
                        {"filename": "c.png", "type": "temp"},
                    ]
                },
            }
        }
        result = AishaJobPoller._collect_image_infos(entry)
        assert len(result) == 2

    def test_empty_outputs_returns_empty(self) -> None:
        result = AishaJobPoller._collect_image_infos({"outputs": {}})
        assert result == []


# ---------------------------------------------------------------------------
# _download_and_upload — thumbnail generation
# ---------------------------------------------------------------------------


class TestDownloadAndUploadThumbnails:
    async def test_returns_full_and_thumb_on_success(self) -> None:
        full_id = uuid4()
        thumb_id = uuid4()
        r2 = AsyncMock()
        r2.upload = AsyncMock(
            side_effect=[
                _make_r2_upload_result(full_id),
                _make_r2_upload_result(thumb_id),
            ]
        )

        poller = _make_poller(r2=r2)
        job = _make_job()
        client = AsyncMock()
        client.get_image = AsyncMock(return_value=_FAKE_PNG)

        expires_at = datetime.now(UTC) + timedelta(days=7)
        img_info = {"filename": "out.png", "subfolder": "", "type": "output"}

        thumb_result = ThumbnailResult(data=_FAKE_WEBP, width=128, height=128)
        dims_mock = MagicMock()
        dims_mock.width = 512
        dims_mock.height = 512

        with (
            patch(
                "src.workers.aisha_job_poller.read_dimensions",
                new=AsyncMock(return_value=dims_mock),
            ),
            patch(
                "src.workers.aisha_job_poller.make_image_thumbnail",
                new=AsyncMock(return_value=thumb_result),
            ),
        ):
            results = await poller._download_and_upload(
                client=client,
                job=job,
                img_info=img_info,
                output_index=0,
                expires_at=expires_at,
            )

        assert len(results) == 2
        full, thumb = results
        assert full.is_thumbnail is False
        assert full.width == 512
        assert full.height == 512
        assert full.parent_output_id is None
        assert thumb.is_thumbnail is True
        assert thumb.parent_output_id == full.id
        assert thumb.format == "webp"
        assert thumb.width == 128
        assert thumb.height == 128
        assert r2.upload.call_count == 2

    async def test_returns_only_full_when_thumbnail_fails(self) -> None:
        full_id = uuid4()
        r2 = AsyncMock()
        r2.upload = AsyncMock(return_value=_make_r2_upload_result(full_id))

        poller = _make_poller(r2=r2)
        job = _make_job()
        client = AsyncMock()
        client.get_image = AsyncMock(return_value=_FAKE_PNG)

        expires_at = datetime.now(UTC) + timedelta(days=7)
        img_info = {"filename": "out.png", "subfolder": "", "type": "output"}

        with (
            patch(
                "src.workers.aisha_job_poller.read_dimensions",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "src.workers.aisha_job_poller.make_image_thumbnail",
                new=AsyncMock(return_value=None),
            ),
        ):
            results = await poller._download_and_upload(
                client=client,
                job=job,
                img_info=img_info,
                output_index=0,
                expires_at=expires_at,
            )

        assert len(results) == 1
        assert results[0].is_thumbnail is False

    async def test_returns_only_full_when_thumb_r2_upload_fails(self) -> None:
        full_id = uuid4()
        r2 = AsyncMock()
        r2.upload = AsyncMock(
            side_effect=[
                _make_r2_upload_result(full_id),
                Exception("R2 upload failed"),
            ]
        )

        poller = _make_poller(r2=r2)
        job = _make_job()
        client = AsyncMock()
        client.get_image = AsyncMock(return_value=_FAKE_PNG)

        expires_at = datetime.now(UTC) + timedelta(days=7)
        img_info = {"filename": "out.png", "subfolder": "", "type": "output"}

        thumb_result = ThumbnailResult(data=_FAKE_WEBP, width=128, height=128)

        with (
            patch(
                "src.workers.aisha_job_poller.read_dimensions",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "src.workers.aisha_job_poller.make_image_thumbnail",
                new=AsyncMock(return_value=thumb_result),
            ),
        ):
            results = await poller._download_and_upload(
                client=client,
                job=job,
                img_info=img_info,
                output_index=0,
                expires_at=expires_at,
            )

        assert len(results) == 1
        assert results[0].is_thumbnail is False

    async def test_returns_empty_when_r2_not_configured(self) -> None:
        poller = _make_poller(r2=None)
        job = _make_job()
        client = AsyncMock()
        expires_at = datetime.now(UTC) + timedelta(days=7)

        results = await poller._download_and_upload(
            client=client,
            job=job,
            img_info={"filename": "out.png", "type": "output"},
            output_index=0,
            expires_at=expires_at,
        )
        assert results == []

    async def test_thumb_output_index_mirrors_parent(self) -> None:
        full_id = uuid4()
        thumb_id = uuid4()
        r2 = AsyncMock()
        r2.upload = AsyncMock(
            side_effect=[
                _make_r2_upload_result(full_id),
                _make_r2_upload_result(thumb_id),
            ]
        )

        poller = _make_poller(r2=r2)
        job = _make_job()
        client = AsyncMock()
        client.get_image = AsyncMock(return_value=_FAKE_PNG)

        thumb_result = ThumbnailResult(data=_FAKE_WEBP, width=64, height=64)

        with (
            patch(
                "src.workers.aisha_job_poller.read_dimensions",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "src.workers.aisha_job_poller.make_image_thumbnail",
                new=AsyncMock(return_value=thumb_result),
            ),
        ):
            results = await poller._download_and_upload(
                client=client,
                job=job,
                img_info={"filename": "out.png", "subfolder": "", "type": "output"},
                output_index=3,
                expires_at=datetime.now(UTC) + timedelta(days=7),
            )

        assert len(results) == 2
        assert results[0].output_index == 3
        assert results[1].output_index == 3
