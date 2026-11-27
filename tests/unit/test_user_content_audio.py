"""``UserContentService._upload_audio`` with the ingestor, storage and repositories mocked (S2a W4).

The real ingest, real storage and real database are exercised by
``tests/integration/test_audio_upload_http.py``; this file pins the failure mapping, the orphan
cleanup and what the service does and does not log.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import structlog

from src.api.services.frame_lineage import FrameLineage, FrameLineageReason
from src.api.services.media_ingest import (
    AudioCode,
    InvalidAudioError,
    InvalidMediaError,
    MediaProcessingError,
    PreparedAudio,
    UnsupportedAudioError,
)
from src.api.services.storage import StorageUploadError, StorageValidationError
from src.api.services.user_content import (
    UserContentAudioError,
    UserContentFrameLineageError,
    UserContentService,
    UserContentStorageError,
    UserContentUnavailableError,
    UserContentValidationError,
)
from src.core.enums import MediaFormat
from src.core.library_ref import AssetRef, LibraryAssetSource

pytestmark = pytest.mark.unit

_TAG_CANARY = "SECRET-FILE-NAME-CANARY"


def _prepared(format_: MediaFormat = MediaFormat.FLAC) -> PreparedAudio:
    return PreparedAudio(
        data=b"stored-audio",
        format=format_,
        codec="flac",
        sample_rate_hz=44100,
        channels=2,
        bit_rate=900000,
        duration_ms=61_000,
        peaks=bytes(2048),
        peaks_version=1,
        detail_peaks=bytes(6_100),
        detail_peaks_version=1,
    )


def _service() -> tuple[UserContentService, AsyncMock, AsyncMock]:
    storage = AsyncMock()
    upload = MagicMock()
    upload.id = uuid4()
    upload.storage_key = f"users/u/uploads/{upload.id}.flac"
    storage.upload = AsyncMock(return_value=upload)
    storage.delete = AsyncMock(return_value=True)
    session = AsyncMock()
    # the profile repository does ``(await session.execute(...)).scalar_one()``
    session.execute = AsyncMock(return_value=MagicMock(scalar_one=MagicMock(return_value=uuid4())))
    ingestor = MagicMock()
    ingestor.prepare_audio = AsyncMock(return_value=_prepared())
    service = UserContentService(
        storage=storage, session=session, product_id="vex", media_ingestor=ingestor
    )
    row = MagicMock()
    row.id = upload.id
    row.storage_key = upload.storage_key
    row.original_filename = f"{upload.id}.flac"
    row.content_type = "audio/flac"
    row.size_bytes = len(b"stored-audio")
    row.created_at = datetime.now(UTC)
    row.expires_at = datetime.now(UTC) + timedelta(days=7)
    row.width = None
    row.height = None
    row.duration_ms = 61_000
    row.format = "flac"
    row.thumbnail_max_edge = None
    service._image_repo = AsyncMock()
    service._image_repo.create = AsyncMock(return_value=row)
    service._output_repo = AsyncMock()
    return service, storage, ingestor.prepare_audio


def _create(service: UserContentService) -> AsyncMock:
    """The mocked ``UserImageRepository.create`` that ``_service`` installed."""
    return cast("AsyncMock", service._image_repo.create)


async def _upload(service: UserContentService, **overrides: object) -> object:
    arguments: dict[str, object] = {
        "user_id": uuid4(),
        "data": b"audio-bytes",
        "filename": f"{_TAG_CANARY}.flac",
        "content_type": "audio/flac",
    }
    arguments.update(overrides)
    return await service.upload_image(**arguments)  # type: ignore[arg-type]


class TestRouting:
    @pytest.mark.parametrize(
        "content_type",
        [
            "audio/mpeg",
            "audio/mp4",
            "audio/wav",
            "audio/flac",
            "audio/mp3",
            "audio/x-m4a",
            "audio/aac",
            "audio/x-wav",
            "audio/wave",
            "audio/x-flac",
        ],
    )
    async def test_every_audio_type_goes_to_audio_ingest_only(self, content_type: str) -> None:
        service, _, prepare_audio = _service()
        ingestor = service._media_ingestor

        with (
            patch.object(ingestor, "prepare_image", new_callable=AsyncMock) as prepare_image,
            patch.object(ingestor, "prepare_video", new_callable=AsyncMock) as prepare_video,
        ):
            result = await _upload(service, content_type=content_type)

        prepare_audio.assert_awaited_once_with(b"audio-bytes")
        prepare_image.assert_not_awaited()
        prepare_video.assert_not_awaited()
        assert result.media.media_type.value == "audio"  # type: ignore[attr-defined]

    async def test_audio_with_a_frame_lineage_is_refused_before_any_work(self) -> None:
        service, storage, prepare_audio = _service()
        lineage = FrameLineage(
            source=AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=uuid4()), timestamp_ms=0
        )

        with pytest.raises(UserContentFrameLineageError) as raised:
            await _upload(service, lineage=lineage)

        assert raised.value.reason is FrameLineageReason.AUDIO_FILE
        prepare_audio.assert_not_awaited()
        storage.upload.assert_not_awaited()


class TestSuccess:
    async def test_the_row_and_profile_are_written_from_the_prepared_facts(self) -> None:
        service, storage, _ = _service()

        with patch(
            "src.api.services.user_content.LibraryAudioProfileRepository.insert",
            AsyncMock(return_value=uuid4()),
        ) as insert:
            result = await _upload(service)

        kwargs = _create(service).call_args.kwargs
        assert kwargs["content_type"] == "audio/flac"
        assert kwargs["format"] == "flac"
        assert kwargs["duration_ms"] == 61_000
        assert kwargs["size_bytes"] == len(b"stored-audio")
        assert kwargs["is_thumbnail"] is False if "is_thumbnail" in kwargs else True
        assert "width" not in kwargs or kwargs["width"] is None
        assert "height" not in kwargs or kwargs["height"] is None
        profile = insert.call_args.kwargs
        assert profile["codec"] == "flac"
        assert profile["sample_rate_hz"] == 44100
        assert profile["channels"] == 2
        assert profile["bit_rate"] == 900000
        assert profile["peaks"] == bytes(2048)
        assert profile["peaks_version"] == 1
        assert profile["detail_peaks"] == bytes(6_100)
        assert profile["detail_peaks_version"] == 1
        assert storage.upload.await_args.kwargs["data"] == b"stored-audio"
        assert storage.upload.await_args.kwargs["content_type"] == "audio/flac"
        assert result.media.variants == []  # type: ignore[attr-defined]
        storage.delete.assert_not_awaited()

    async def test_the_logs_carry_no_file_name_and_the_audio_profile_event_has_the_facts(
        self,
    ) -> None:
        service, _, _ = _service()

        with (
            structlog.testing.capture_logs() as logs,
            patch(
                "src.api.services.user_content.LibraryAudioProfileRepository.insert",
                AsyncMock(return_value=uuid4()),
            ),
        ):
            await _upload(service)

        assert _TAG_CANARY not in repr(logs)
        events = {entry["event"]: entry for entry in logs}
        profile = events["media.audio_profile"]
        assert profile["format"] == "flac"
        assert profile["codec"] == "flac"
        assert profile["sample_rate_hz"] == 44100
        assert profile["channels"] == 2
        assert profile["size_bytes"] == len(b"stored-audio")
        assert profile["duration_ms"] == 61_000
        assert set(profile) <= {
            "event",
            "log_level",
            "origin",
            "asset_ref",
            "format",
            "codec",
            "sample_rate_hz",
            "channels",
            "size_bytes",
            "duration_ms",
        }
        assert events["user_content.audio_uploaded"]["duration_ms"] == 61_000


class TestIngestRefusals:
    @pytest.mark.parametrize(
        "error",
        [
            UnsupportedAudioError(AudioCode.FORMAT_NOT_SUPPORTED, "format"),
            UnsupportedAudioError(AudioCode.HAS_VIDEO, "video"),
            UnsupportedAudioError(AudioCode.CODEC_NOT_SUPPORTED, "codec"),
            InvalidAudioError(AudioCode.INVALID, "invalid"),
            InvalidAudioError(
                AudioCode.DURATION_OUT_OF_RANGE, "range", {"min_seconds": 5, "max_seconds": 900}
            ),
        ],
    )
    async def test_a_typed_refusal_keeps_its_code_and_detail_and_stores_nothing(
        self, error: InvalidMediaError
    ) -> None:
        service, storage, prepare_audio = _service()
        prepare_audio.side_effect = error

        with pytest.raises(UserContentAudioError) as raised:
            await _upload(service)

        assert raised.value.code == error.code.value  # type: ignore[attr-defined]
        assert raised.value.detail == error.detail  # type: ignore[attr-defined]
        assert isinstance(raised.value, UserContentValidationError)
        storage.upload.assert_not_awaited()
        _create(service).assert_not_awaited()

    async def test_an_untyped_refusal_is_a_plain_validation_error(self) -> None:
        service, storage, prepare_audio = _service()
        prepare_audio.side_effect = InvalidMediaError(
            "audio input exceeds the configured byte limit"
        )

        with pytest.raises(UserContentValidationError) as raised:
            await _upload(service)

        assert not isinstance(raised.value, UserContentAudioError)
        storage.upload.assert_not_awaited()

    async def test_no_capacity_is_unavailable_not_a_client_error(self) -> None:
        service, storage, prepare_audio = _service()
        prepare_audio.side_effect = MediaProcessingError("audio preparation capacity is exhausted")

        with pytest.raises(UserContentUnavailableError, match="temporarily unavailable"):
            await _upload(service)

        storage.upload.assert_not_awaited()


class TestStorageAndDatabaseFailures:
    async def test_a_storage_validation_error_is_a_validation_error(self) -> None:
        service, storage, _ = _service()
        storage.upload.side_effect = StorageValidationError("too big")

        with pytest.raises(UserContentValidationError, match="too big"):
            await _upload(service)

    async def test_a_storage_failure_is_a_storage_error_with_no_row(self) -> None:
        service, storage, _ = _service()
        storage.upload.side_effect = StorageUploadError("boom")

        with pytest.raises(UserContentStorageError):
            await _upload(service)

        _create(service).assert_not_awaited()

    async def test_a_database_failure_deletes_the_stored_object_and_reraises(self) -> None:
        service, storage, _ = _service()
        stored = storage.upload.return_value
        _create(service).side_effect = RuntimeError("database went away")

        with pytest.raises(RuntimeError, match="database went away"):
            await _upload(service)

        storage.delete.assert_awaited_once_with(stored.storage_key)

    async def test_a_profile_failure_deletes_the_stored_object_too(self) -> None:
        service, storage, _ = _service()
        stored = storage.upload.return_value

        with (
            patch(
                "src.api.services.user_content.LibraryAudioProfileRepository.insert",
                AsyncMock(side_effect=ValueError("check violated")),
            ),
            pytest.raises(ValueError, match="check violated"),
        ):
            await _upload(service)

        storage.delete.assert_awaited_once_with(stored.storage_key)

    async def test_a_failing_cleanup_never_replaces_the_original_error(self) -> None:
        service, storage, _ = _service()
        _create(service).side_effect = RuntimeError("the real failure")
        storage.delete.side_effect = RuntimeError("the cleanup failure")

        with (
            structlog.testing.capture_logs() as logs,
            pytest.raises(RuntimeError, match="the real failure"),
        ):
            await _upload(service)

        assert [entry["event"] for entry in logs] == ["user_content.orphan_cleanup_failed"]
        assert "cleanup failure" not in repr(logs)
