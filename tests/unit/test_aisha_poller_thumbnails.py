"""Tests for AishaJobPoller thumbnail generation and type filtering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.image_thumbnail import GeneratedThumbnail, ThumbnailResult
from src.api.services.media_ingest import PreparedImage
from src.core.enums import MediaFormat
from src.core.media_hash import HashSample, HashSet, PdqHash
from src.core.thumbnails import ThumbnailSpec
from src.workers.aisha_job_poller import AishaJobPoller, AishaPollerConfig

pytestmark = pytest.mark.unit

_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # not a real PNG, but enough for mocking
_FAKE_WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20

_SM_SPEC = ThumbnailSpec("sm", 150)
_MD_SPEC = ThumbnailSpec("md", 512)


def _make_generated_thumbnail(
    spec: ThumbnailSpec, width: int = 64, height: int = 64
) -> GeneratedThumbnail:
    return GeneratedThumbnail(
        spec=spec,
        result=ThumbnailResult(data=_FAKE_WEBP, width=width, height=height),
    )


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


def _prepared_image() -> PreparedImage:
    return PreparedImage(
        data=_FAKE_PNG,
        format=MediaFormat.PNG,
        width=512,
        height=512,
        hash_set=HashSet(
            profile_id="pdq-image-rgb-white-v1",
            sampling_profile="still-v1",
            samples=(HashSample(pdq=PdqHash(bits=b"\x00" * 32, quality=0), sample_index=0),),
        ),
        orientation_baked=False,
        converted=False,
    )


def _make_poller(r2: MagicMock | None = None) -> AishaJobPoller:
    session_factory = MagicMock()
    session_factory.return_value = AsyncMock()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    media_ingestor = MagicMock()
    media_ingestor.prepare_image = AsyncMock(return_value=_prepared_image())
    return AishaJobPoller(
        session_factory=session_factory,
        event_bus=None,
        billing_service=AsyncMock(),
        r2_storage=r2,
        config=_make_config(),
        redis_client_factory=MagicMock(),
        media_ingestor=media_ingestor,
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
# _download_and_upload — multi-size thumbnail generation
# ---------------------------------------------------------------------------


class TestDownloadAndUploadThumbnails:
    async def test_returns_full_and_both_thumbs_on_success(self) -> None:
        """make_image_thumbnails returns sm+md, so we get full + 2 thumbnails."""
        full_id = uuid4()
        sm_id = uuid4()
        md_id = uuid4()
        r2 = AsyncMock()
        r2.upload = AsyncMock(
            side_effect=[
                _make_r2_upload_result(full_id),
                _make_r2_upload_result(sm_id),
                _make_r2_upload_result(md_id),
            ]
        )

        poller = _make_poller(r2=r2)
        job = _make_job()
        client = AsyncMock()
        client.get_image = AsyncMock(return_value=_FAKE_PNG + b"private-metadata-canary")

        expires_at = datetime.now(UTC) + timedelta(days=7)
        img_info = {"filename": "out.png", "subfolder": "", "type": "output"}

        thumbnails = [
            _make_generated_thumbnail(_SM_SPEC, width=100, height=100),
            _make_generated_thumbnail(_MD_SPEC, width=400, height=400),
        ]
        with (
            patch(
                "src.workers.aisha_job_poller.make_image_thumbnails",
                new=AsyncMock(return_value=thumbnails),
            ),
        ):
            results = await poller._download_and_upload(
                client=client,
                job=job,
                img_info=img_info,
                output_index=0,
                expires_at=expires_at,
            )

        assert len(results.outputs) == 3
        assert r2.upload.await_args_list[0].kwargs["data"] == _FAKE_PNG
        full = results.outputs[0]
        sm_thumb = results.outputs[1]
        md_thumb = results.outputs[2]

        assert full.is_thumbnail is False
        assert full.width == 512
        assert full.height == 512
        assert full.parent_output_id is None
        assert full.thumbnail_max_edge is None

        assert sm_thumb.is_thumbnail is True
        assert sm_thumb.parent_output_id == full.id
        assert sm_thumb.thumbnail_max_edge == 150
        assert sm_thumb.width == 100
        assert sm_thumb.height == 100

        assert md_thumb.is_thumbnail is True
        assert md_thumb.parent_output_id == full.id
        assert md_thumb.thumbnail_max_edge == 512
        assert r2.upload.call_count == 3

    async def test_returns_only_full_when_thumbnails_empty(self) -> None:
        """make_image_thumbnails returns [] (decode failure) → only full output."""
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
                "src.workers.aisha_job_poller.make_image_thumbnails",
                new=AsyncMock(return_value=[]),
            ),
        ):
            results = await poller._download_and_upload(
                client=client,
                job=job,
                img_info=img_info,
                output_index=0,
                expires_at=expires_at,
            )

        assert len(results.outputs) == 1
        assert results.outputs[0].is_thumbnail is False

    async def test_skips_thumb_when_r2_upload_fails(self) -> None:
        """If a thumb's R2 upload raises, that size is skipped; others succeed."""
        full_id = uuid4()
        md_id = uuid4()
        r2 = AsyncMock()
        r2.upload = AsyncMock(
            side_effect=[
                _make_r2_upload_result(full_id),
                Exception("R2 upload failed for sm"),
                _make_r2_upload_result(md_id),
            ]
        )

        poller = _make_poller(r2=r2)
        job = _make_job()
        client = AsyncMock()
        client.get_image = AsyncMock(return_value=_FAKE_PNG)

        expires_at = datetime.now(UTC) + timedelta(days=7)
        img_info = {"filename": "out.png", "subfolder": "", "type": "output"}

        thumbnails = [
            _make_generated_thumbnail(_SM_SPEC),
            _make_generated_thumbnail(_MD_SPEC),
        ]

        with (
            patch(
                "src.workers.aisha_job_poller.make_image_thumbnails",
                new=AsyncMock(return_value=thumbnails),
            ),
        ):
            results = await poller._download_and_upload(
                client=client,
                job=job,
                img_info=img_info,
                output_index=0,
                expires_at=expires_at,
            )

        # sm skipped (upload error), md succeeded → full + md
        assert len(results.outputs) == 2
        assert results.outputs[0].is_thumbnail is False
        assert results.outputs[1].thumbnail_max_edge == 512

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
        assert results.outputs == ()
        assert results.retryable_failure is True

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

        thumbnails = [_make_generated_thumbnail(_MD_SPEC, width=64, height=64)]

        with (
            patch(
                "src.workers.aisha_job_poller.make_image_thumbnails",
                new=AsyncMock(return_value=thumbnails),
            ),
        ):
            results = await poller._download_and_upload(
                client=client,
                job=job,
                img_info={"filename": "out.png", "subfolder": "", "type": "output"},
                output_index=3,
                expires_at=datetime.now(UTC) + timedelta(days=7),
            )

        assert len(results.outputs) == 2
        assert results.outputs[0].output_index == 3
        assert results.outputs[1].output_index == 3
