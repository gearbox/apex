"""Audio uploads over real multipart HTTP: real ingest (ffmpeg), real database, R2 stubbed (S2a).

Invariants under test (W11.1): the stored object is exactly the input's audio stream (bit-identical,
no tag, no attached picture); the sniffed kind decides, never the client's MIME; each refusal is its
typed ``422`` and leaves **no row and no object**; an over-limit body is refused **before** ingest
runs; a database failure after the object was stored deletes the object again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from litestar import Litestar
from litestar.di import Provide
from litestar.testing import AsyncTestClient
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.product import ProductMiddleware
from src.api.routes.storage import StorageController
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.media_ingest import MediaIngestService
from src.api.services.token_revocation import TokenRevocationService
from src.api.services.user_content import UserContentService
from src.core.product_registry import VEX_CONFIG
from src.db.models.library import LibraryAudioProfile
from src.db.models.media_hash import MediaHash
from src.db.models.storage import UserImage
from src.db.models.user import User
from tests.audio_support import (
    CONTAINERS,
    SECRET_TAGS,
    AudioFixtures,
    has_attached_picture,
    probe,
    require_ffmpeg,
    run,
    stream_md5,
    user_tags,
)
from tests.legal_support import make_legal_registry
from tests.unit.helpers import hermetic_settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from httpx import Response
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.core.config import Settings

pytestmark = pytest.mark.asyncio

_SECRET = "test_secret_key_for_testing_only_256bits_long"
_PRODUCT_ID = "vex"
_REGISTRY = make_legal_registry()
_MIME = {
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "wav": "audio/wav",
    "flac": "audio/flac",
}


@pytest.fixture(autouse=True)
def _ffmpeg() -> None:
    require_ffmpeg()


@dataclass
class _Storage:
    """An in-memory R2: keeps what was uploaded, so the test can inspect the stored object."""

    objects: dict[str, bytes] = field(default_factory=dict)
    uploads: list[dict[str, Any]] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    fail_upload: Exception | None = None
    fail_delete: Exception | None = None

    def build(self) -> MagicMock:
        storage = MagicMock()

        async def upload(**kwargs: Any) -> MagicMock:
            if self.fail_upload is not None:
                raise self.fail_upload
            object_id = uuid4()
            extension = kwargs["content_type"].split("/")[-1]
            key = f"users/{kwargs['user_id']}/uploads/{object_id}.{extension}"
            self.objects[key] = kwargs["data"]
            self.uploads.append(kwargs)
            return MagicMock(id=object_id, storage_key=key)

        async def delete_object(key: str) -> bool:
            if self.fail_delete is not None:
                raise self.fail_delete
            self.deleted.append(key)
            self.objects.pop(key, None)
            return True

        storage.upload = AsyncMock(side_effect=upload)
        storage.delete = AsyncMock(side_effect=delete_object)
        return storage


@dataclass
class _Harness:
    app: Litestar
    storage: _Storage
    user_id: UUID
    ingestor: MediaIngestService
    engine: AsyncEngine


def _ingestor(**overrides: Any) -> MediaIngestService:
    options: dict[str, Any] = {
        "max_image_megapixels": 10,
        "max_input_bytes": 20 * 1024 * 1024,
        "max_audio_bytes": 80 * 1024 * 1024,
    }
    options.update(overrides)
    return MediaIngestService(**options)


def _build(
    engine: AsyncEngine, settings: Settings, ingestor: MediaIngestService, storage: _Storage
) -> Litestar:
    mock_storage = storage.build()

    async def provide_session() -> AsyncGenerator[AsyncSession]:
        # Mirrors ``get_db_session``: a request-scoped session that commits on success and rolls
        # back when the handler raises.
        async with AsyncSession(bind=engine, expire_on_commit=False) as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    def provide_user_content(session: AsyncSession) -> UserContentService:
        return UserContentService(
            mock_storage,
            session,
            product_id=_PRODUCT_ID,
            retention_days=7,
            media_ingestor=ingestor,
        )

    app = Litestar(
        route_handlers=[StorageController],
        middleware=[ProductMiddleware],
        dependencies={
            "session": Provide(provide_session),
            "user_content": Provide(provide_user_content, sync_to_thread=False),
            "settings": Provide(lambda: settings, sync_to_thread=False),
        },
        request_max_body_size=settings.max_request_body_bytes,
    )
    app.state["jwt_service"] = JWTService(JWTConfig(secret_key=_SECRET))
    app.state["token_revocation"] = TokenRevocationService(None, max_token_ttl_seconds=0)
    app.state["legal_registry"] = _REGISTRY
    return app


@pytest.fixture
async def user_id(db_engine: AsyncEngine) -> AsyncGenerator[UUID]:
    uid = uuid4()
    async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
        session.add(
            User(
                id=uid,
                email=f"audio-{uuid4().hex[:8]}@example.com",
                password_hash="x" * 64,
                is_active=True,
                product_id=_PRODUCT_ID,
            )
        )
        await session.commit()
    yield uid
    async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
        await session.execute(delete(User).where(User.id == uid))
        await session.commit()


@pytest.fixture
def harness(db_engine: AsyncEngine, user_id: UUID) -> _Harness:
    storage = _Storage()
    ingestor = _ingestor()
    app = _build(db_engine, hermetic_settings(), ingestor, storage)
    return _Harness(app, storage, user_id, ingestor, db_engine)


def _headers(app: Litestar, user_id: UUID) -> dict[str, str]:
    digest = _REGISTRY.required_digest(VEX_CONFIG, today=datetime.now(UTC).date())
    token, _ = app.state["jwt_service"].create_access_token(
        user_id, product_id=_PRODUCT_ID, legal_digest=digest
    )
    return {"Authorization": f"Bearer {token}", "X-Product-Id": _PRODUCT_ID}


async def _upload(
    h: _Harness,
    data: bytes,
    *,
    content_type: str,
    filename: str = "song.bin",
    fields: dict[str, str] | None = None,
) -> Response:
    async with AsyncTestClient(app=h.app) as client:
        return await client.post(
            "/v1/storage/upload",
            headers=_headers(h.app, h.user_id),
            files={"data": (filename, data, content_type)},
            data=fields or {},
        )


async def _rows(engine: AsyncEngine, user_id: UUID) -> tuple[int, int]:
    async with AsyncSession(bind=engine) as session:
        uploads = (
            await session.execute(
                select(func.count()).select_from(UserImage).where(UserImage.user_id == user_id)
            )
        ).scalar_one()
        profiles = (
            await session.execute(
                select(func.count())
                .select_from(LibraryAudioProfile)
                .join(UserImage, UserImage.id == LibraryAudioProfile.upload_id)
                .where(UserImage.user_id == user_id)
            )
        ).scalar_one()
    return uploads, profiles


class TestEachContainer:
    @pytest.mark.parametrize("container", list(CONTAINERS))
    async def test_it_is_stored_as_the_input_s_stream_without_tags_or_picture(
        self, harness: _Harness, audio_files: AudioFixtures, tmp_path: Path, container: str
    ) -> None:
        source = audio_files.tone(container)

        response = await _upload(
            harness, source.read_bytes(), content_type=_MIME[container], filename="My Song.bin"
        )

        assert response.status_code == 201, response.text
        body = response.json()
        # the stored object, as ingest wrote it
        ((key, stored_bytes),) = harness.storage.objects.items()
        stored = tmp_path / f"stored.{CONTAINERS[container][0]}"
        stored.write_bytes(stored_bytes)
        info = probe(stored)
        assert user_tags(info) == {}
        assert not has_attached_picture(info)
        assert [s["codec_type"] for s in info["streams"]] == ["audio"]  # type: ignore[index, union-attr]
        for value in SECRET_TAGS.values():
            assert value.encode() not in stored_bytes
        assert stream_md5(stored) == stream_md5(source), "the audio was re-encoded"
        # the response
        assert body["media"]["media_type"] == "audio"
        assert body["media"]["original"]["content_type"] == _MIME[container]
        assert body["media"]["original"]["width"] is None
        assert body["media"]["original"]["height"] is None
        assert abs(body["media"]["original"]["duration_ms"] - 6000) <= 60
        assert body["media"]["variants"] == []
        # the rows
        async with AsyncSession(bind=harness.engine) as session:
            row = (
                await session.execute(select(UserImage).where(UserImage.id == UUID(body["id"])))
            ).scalar_one()
            profile = (
                await session.execute(
                    select(LibraryAudioProfile).where(LibraryAudioProfile.upload_id == row.id)
                )
            ).scalar_one()
            hashes = (
                await session.execute(
                    select(func.count()).select_from(MediaHash).where(MediaHash.source_id == row.id)
                )
            ).scalar_one()
        assert row.storage_key == key
        assert row.content_type == _MIME[container]
        assert row.format == CONTAINERS[container][0]
        assert row.size_bytes == len(stored_bytes)
        assert row.width is None
        assert row.height is None
        assert row.is_thumbnail is False
        assert row.parent_image_id is None
        assert row.duration_ms == body["media"]["original"]["duration_ms"]
        assert row.expires_at > datetime.now(UTC)
        assert row.display_filename == "My Song.bin"
        assert row.original_filename == f"{row.id}.{CONTAINERS[container][0]}"
        assert hashes == 0, "audio has no perceptual hash ledger row"
        assert len(profile.peaks) == 2048
        assert profile.peaks_version == 1
        assert profile.sample_rate_hz == 44100
        assert profile.channels == 1
        assert profile.output_id is None
        facts = info["streams"][0]  # type: ignore[index]
        assert profile.codec == facts["codec_name"]  # type: ignore[index]
        # FLAC streams report no bit rate of their own: the container's is the fallback
        assert profile.bit_rate == int(facts.get("bit_rate") or info["format"]["bit_rate"])  # type: ignore[index, union-attr, attr-defined]
        # no thumbnail and no poster: exactly one object, one row
        assert len(harness.storage.uploads) == 1
        assert await _rows(harness.engine, harness.user_id) == (1, 1)

    @pytest.mark.parametrize(
        ("declared", "container", "expected"),
        [
            ("audio/mp3", "mp3", "audio/mpeg"),
            ("audio/x-m4a", "m4a", "audio/mp4"),
            ("audio/aac", "m4a", "audio/mp4"),
            ("audio/x-wav", "wav", "audio/wav"),
            ("audio/wave", "wav", "audio/wav"),
            ("audio/x-flac", "flac", "audio/flac"),
        ],
    )
    async def test_the_client_aliases_are_accepted_and_stored_as_the_real_type(
        self,
        harness: _Harness,
        audio_files: AudioFixtures,
        declared: str,
        container: str,
        expected: str,
    ) -> None:
        response = await _upload(
            harness, audio_files.tone(container).read_bytes(), content_type=declared
        )

        assert response.status_code == 201, response.text
        assert response.json()["media"]["original"]["content_type"] == expected

    async def test_the_sniffed_kind_decides_never_the_client_s_mime(
        self, harness: _Harness, audio_files: AudioFixtures
    ) -> None:
        """A FLAC declared as MP3 is stored as FLAC."""
        response = await _upload(
            harness, audio_files.tone("flac").read_bytes(), content_type="audio/mpeg"
        )

        assert response.status_code == 201
        assert response.json()["media"]["original"]["content_type"] == "audio/flac"

    async def test_no_tag_value_reaches_the_response(
        self, harness: _Harness, audio_files: AudioFixtures
    ) -> None:
        response = await _upload(
            harness, audio_files.tone("mp3").read_bytes(), content_type="audio/mpeg"
        )

        for value in SECRET_TAGS.values():
            assert value not in response.text


def _stereo_wav(path: Path, *, seconds: int) -> None:
    """A stereo 16-bit 44.1 kHz WAV: about 176 kB per second."""
    run(
        [
            require_ffmpeg(),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    )


class TestRefusals:
    @pytest.mark.parametrize(
        ("fixture", "declared", "code", "detail"),
        [
            ("video_mp4", "audio/mp4", "audio_has_video", None),
            (
                "short",
                "audio/wav",
                "audio_duration_out_of_range",
                {"min_seconds": 5, "max_seconds": 900},
            ),
            ("ogg_opus", "audio/mpeg", "audio_format_not_supported", None),
            ("pcm_float", "audio/wav", "audio_codec_not_supported", None),
            ("picture_only", "audio/mpeg", "audio_format_not_supported", None),
        ],
    )
    async def test_each_refusal_is_its_typed_422_and_leaves_nothing_behind(
        self,
        harness: _Harness,
        audio_files: AudioFixtures,
        fixture: str,
        declared: str,
        code: str,
        detail: dict[str, int] | None,
    ) -> None:
        data = getattr(audio_files, fixture).read_bytes()

        response = await _upload(harness, data, content_type=declared)

        assert response.status_code == 422, response.text
        body = response.json()
        assert body["error"] == code
        assert body["status_code"] == 422
        assert body["detail"] == detail
        assert harness.storage.uploads == []
        assert harness.storage.objects == {}
        assert await _rows(harness.engine, harness.user_id) == (0, 0)

    async def test_garbage_declared_as_audio_is_invalid_audio(self, harness: _Harness) -> None:
        response = await _upload(harness, b"definitely not audio" * 50, content_type="audio/mpeg")

        assert response.status_code == 422
        assert response.json()["error"] == "invalid_audio"
        assert await _rows(harness.engine, harness.user_id) == (0, 0)

    async def test_an_audio_type_outside_the_four_is_not_an_allowed_upload(
        self, harness: _Harness, audio_files: AudioFixtures
    ) -> None:
        response = await _upload(
            harness, audio_files.ogg_opus.read_bytes(), content_type="audio/ogg"
        )

        assert response.status_code == 400
        assert response.json()["error"] == "invalid_file_type"

    async def test_audio_bytes_declared_as_an_image_are_never_stored_as_audio(
        self, harness: _Harness, audio_files: AudioFixtures
    ) -> None:
        response = await _upload(
            harness, audio_files.tone("mp3").read_bytes(), content_type="image/png"
        )

        assert response.status_code == 400  # the image path: not a decodable image
        assert await _rows(harness.engine, harness.user_id) == (0, 0)
        assert harness.storage.objects == {}

    async def test_an_empty_audio_body_is_refused(self, harness: _Harness) -> None:
        response = await _upload(harness, b"", content_type="audio/mpeg")

        assert response.status_code == 400
        assert response.json()["error"] == "empty_file"

    async def test_audio_is_never_a_captured_frame(
        self, harness: _Harness, audio_files: AudioFixtures
    ) -> None:
        response = await _upload(
            harness,
            audio_files.tone("mp3").read_bytes(),
            content_type="audio/mpeg",
            fields={"source_asset_ref": f"upload:{uuid4()}", "source_timestamp_ms": "0"},
        )

        assert response.status_code == 400
        assert response.json()["error"] == "invalid_frame_lineage"
        assert response.json()["message"] == "Frame source is not available"
        assert harness.storage.uploads == []


class TestPerKindSizeLimits:
    """The route picks the limit from the declared type and refuses **before** ingest runs."""

    @staticmethod
    def _spy(h: _Harness) -> AsyncMock:
        spy = AsyncMock(wraps=h.ingestor.prepare_audio)
        h.ingestor.prepare_audio = spy  # type: ignore[method-assign]
        return spy

    async def test_an_audio_body_over_its_limit_is_refused_before_ingest(
        self, db_engine: AsyncEngine, user_id: UUID, audio_files: AudioFixtures
    ) -> None:
        storage = _Storage()
        ingestor = _ingestor()
        # audio 1 MB, images 3 MB: the app admits a 3 MB body, the route holds audio to 1 MB
        settings = hermetic_settings(audio_upload_max_mb=1, max_upload_size_mb=3)
        harness = _Harness(
            _build(db_engine, settings, ingestor, storage), storage, user_id, ingestor, db_engine
        )
        spy = self._spy(harness)
        big = audio_files.root / "big.wav"
        _stereo_wav(big, seconds=12)
        assert 1024 * 1024 < big.stat().st_size < 3 * 1024 * 1024

        response = await _upload(harness, big.read_bytes(), content_type="audio/wav")

        assert response.status_code == 400
        body = response.json()
        assert body["error"] == "file_too_large"
        assert body["message"] == "Maximum size: 1MB"
        spy.assert_not_awaited()
        assert storage.uploads == []
        assert await _rows(db_engine, user_id) == (0, 0)

    async def test_an_image_is_held_to_the_image_limit_with_that_limit_in_the_message(
        self, db_engine: AsyncEngine, user_id: UUID
    ) -> None:
        storage = _Storage()
        ingestor = _ingestor()
        settings = hermetic_settings(audio_upload_max_mb=3, max_upload_size_mb=1)
        harness = _Harness(
            _build(db_engine, settings, ingestor, storage), storage, user_id, ingestor, db_engine
        )

        image = await _upload(harness, b"x" * (2 * 1024 * 1024), content_type="image/png")
        video = await _upload(harness, b"x" * (2 * 1024 * 1024), content_type="video/mp4")

        for response in (image, video):
            assert response.status_code == 400
            assert response.json()["error"] == "file_too_large"
            assert response.json()["message"] == "Maximum size: 1MB"
        assert storage.uploads == []

    async def test_audio_may_be_bigger_than_an_image_is_allowed_to_be(
        self, db_engine: AsyncEngine, user_id: UUID, audio_files: AudioFixtures
    ) -> None:
        storage = _Storage()
        ingestor = _ingestor()
        settings = hermetic_settings(audio_upload_max_mb=5, max_upload_size_mb=1)
        harness = _Harness(
            _build(db_engine, settings, ingestor, storage), storage, user_id, ingestor, db_engine
        )
        big = audio_files.root / "between.wav"
        _stereo_wav(big, seconds=12)
        size = big.stat().st_size
        assert 1024 * 1024 < size < 5 * 1024 * 1024  # over the image limit, under the audio one

        response = await _upload(harness, big.read_bytes(), content_type="audio/wav")

        assert response.status_code == 201, response.text
        # and the same body declared as an image is refused by the *image* limit
        as_image = await _upload(harness, big.read_bytes(), content_type="image/png")
        assert as_image.status_code == 400
        assert as_image.json()["error"] == "file_too_large"

    async def test_a_body_above_the_largest_limit_never_reaches_a_handler(
        self, db_engine: AsyncEngine, user_id: UUID
    ) -> None:
        storage = _Storage()
        ingestor = _ingestor()
        settings = hermetic_settings(audio_upload_max_mb=2, max_upload_size_mb=1)
        harness = _Harness(
            _build(db_engine, settings, ingestor, storage), storage, user_id, ingestor, db_engine
        )
        spy = self._spy(harness)

        response = await _upload(harness, b"x" * (3 * 1024 * 1024), content_type="audio/mpeg")

        assert response.status_code == 413
        spy.assert_not_awaited()
        assert storage.uploads == []

    async def test_ingest_rechecks_the_size_against_the_audio_budget(
        self, audio_files: AudioFixtures
    ) -> None:
        """A client declaring a video type at audio-budget size is stopped by ingest's own check."""
        from src.api.services.media_ingest import InvalidMediaError

        service = _ingestor(max_audio_bytes=1000)
        with pytest.raises(InvalidMediaError, match="byte limit"):
            await service.prepare_audio(audio_files.tone("wav").read_bytes())


class TestFailuresLeaveNothingBehind:
    async def test_a_database_failure_after_the_object_was_stored_deletes_the_object(
        self, harness: _Harness, audio_files: AudioFixtures
    ) -> None:
        with patch(
            "src.api.services.user_content.LibraryAudioProfileRepository.insert",
            AsyncMock(side_effect=RuntimeError("database went away")),
        ):
            response = await _upload(
                harness, audio_files.tone("flac").read_bytes(), content_type="audio/flac"
            )

        assert response.status_code == 500
        assert len(harness.storage.uploads) == 1
        assert len(harness.storage.deleted) == 1  # the orphan was removed again
        assert harness.storage.objects == {}
        assert await _rows(harness.engine, harness.user_id) == (0, 0)

    async def test_a_failing_cleanup_does_not_replace_the_original_error(
        self, harness: _Harness, audio_files: AudioFixtures
    ) -> None:
        harness.storage.fail_delete = RuntimeError("R2 is down too")
        with patch(
            "src.api.services.user_content.LibraryAudioProfileRepository.insert",
            AsyncMock(side_effect=ValueError("the database error")),
        ):
            response = await _upload(
                harness, audio_files.tone("flac").read_bytes(), content_type="audio/flac"
            )

        # still the request's own failure (a 500 from the catch-all), not the cleanup's
        assert response.status_code == 500
        assert "R2 is down too" not in response.text
        assert await _rows(harness.engine, harness.user_id) == (0, 0)

    async def test_a_storage_failure_stores_no_row_and_answers_502(
        self, db_engine: AsyncEngine, user_id: UUID, audio_files: AudioFixtures
    ) -> None:
        from src.api.services.storage import StorageUploadError

        storage = _Storage(fail_upload=StorageUploadError("boom"))
        ingestor = _ingestor()
        harness = _Harness(
            _build(db_engine, hermetic_settings(), ingestor, storage),
            storage,
            user_id,
            ingestor,
            db_engine,
        )

        response = await _upload(
            harness, audio_files.tone("flac").read_bytes(), content_type="audio/flac"
        )

        assert response.status_code == 502
        assert response.json()["error"] == "upstream_error"
        assert await _rows(db_engine, user_id) == (0, 0)

    async def test_capacity_exhaustion_is_a_503_not_a_client_error(
        self, db_engine: AsyncEngine, user_id: UUID, audio_files: AudioFixtures
    ) -> None:
        import asyncio

        storage = _Storage()
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()  # a video preparation holds the only slot
        ingestor = _ingestor(video_slots=semaphore, admission_wait_seconds=0.05)
        harness = _Harness(
            _build(db_engine, hermetic_settings(), ingestor, storage),
            storage,
            user_id,
            ingestor,
            db_engine,
        )

        response = await _upload(
            harness, audio_files.tone("mp3").read_bytes(), content_type="audio/mpeg"
        )

        assert response.status_code == 503
        assert response.json()["error"] == "service_unavailable"
        assert storage.uploads == []
