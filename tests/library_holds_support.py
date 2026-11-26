"""Builders for the asset-hold and cast integration tests (S1-P3).

Two flavours, because the two kinds of test need different things:

* ``make_output`` works on any session (the savepoint-scoped ``db_session`` too) and flushes: for
  service-level tests that run in one transaction.
* ``HoldWorld`` commits through a real session factory: for the races, the retention sweep (which
  opens its own sessions) and the HTTP tests, where a second connection must see the rows. It
  deletes every user it created (cascading holds, assets, cast) when the test ends - which also
  proves, every time, that a user hard-delete with holds in place succeeds (``NO ACTION``).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator  # noqa: TC003 - Litestar resolves the hint at runtime
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

from litestar import Litestar
from litestar.di import Provide
from litestar.exceptions import HTTPException
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002 - resolved by Litestar at runtime

from src.api.app import (
    asset_held_handler,
    global_exception_handler,
    http_exception_handler,
    job_in_progress_handler,
    legal_acceptance_required_handler,
    studio_error_handler,
)
from src.api.dependencies.common import (
    get_library_service,
    get_product_config,
    get_product_id,
    get_studio_artist_service,
)
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.content_proxy import ContentProxyService
from src.api.services.legal.errors import LegalAcceptanceRequiredError
from src.api.services.library_holds import AssetHeldError, AssetHoldService
from src.api.services.token_revocation import TokenRevocationService
from src.api.services.unified_jobs import JobInProgressError
from src.core.enums import GenerationType, JobStatus, MediaKind
from src.core.library_ref import AssetRef, LibraryAssetSource
from src.core.product_registry import VEX_CONFIG
from src.db.models.library import LibraryAssetHold, LibraryAudioProfile
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage
from src.db.models.user import User
from src.studio.errors import StudioError
from src.studio.services.cast import CastService
from src.studio.services.timeline import TimelineService
from src.studio.services.track import TrackService
from src.studio.services.track_audio import TrackAudioService
from src.studio.services.version import DocumentVersionService
from tests.legal_support import TEST_LEGAL_DIGEST, TEST_LEGAL_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from litestar.types import ASGIApp, Receive, Scope, Send
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.core.product import ProductConfig

__all__ = [
    "SECRET",
    "HoldWorld",
    "bearer",
    "build_app",
    "future",
    "make_output",
    "make_upload",
    "output_ref",
    "past",
    "upload_ref",
]


def past(hours: int = 1) -> datetime:
    return datetime.now(UTC) - timedelta(hours=hours)


def future(days: int = 7) -> datetime:
    return datetime.now(UTC) + timedelta(days=days)


IMAGES = frozenset({MediaKind.IMAGE})
"""What a cast reference accepts: the ``accept`` of most hold tests."""
AUDIO = frozenset({MediaKind.AUDIO})
"""What a track audio version accepts."""


def upload_ref(image: UserImage) -> AssetRef:
    return AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=image.id)


def output_ref(output: GenerationOutput) -> AssetRef:
    return AssetRef(source=LibraryAssetSource.OUTPUT, asset_id=output.id)


async def make_upload(
    session: AsyncSession,
    user: User,
    *,
    expires_at: datetime | None = None,
    content_type: str = "image/png",
    is_thumbnail: bool = False,
    parent: UserImage | None = None,
    product_id: str | None = None,
) -> UserImage:
    """Insert and flush an upload (or a thumbnail of ``parent``)."""
    image_id = uuid4()
    image = UserImage(
        id=image_id,
        user_id=user.id,
        product_id=product_id or user.product_id,
        storage_key=f"users/{user.id}/uploads/{image_id}.bin",
        original_filename="photo.png",
        content_type=content_type,
        size_bytes=100,
        format="png",
        width=64,
        height=64,
        is_thumbnail=is_thumbnail,
        parent_image_id=parent.id if parent is not None else None,
        thumbnail_max_edge=150 if is_thumbnail else None,
        expires_at=expires_at or future(),
    )
    session.add(image)
    await session.flush()
    return image


AUDIO_FORMATS = {
    "audio/flac": "flac",
    "audio/mpeg": "mp3",
    "audio/mp4": "m4a",
    "audio/wav": "wav",
}


async def make_audio_upload(
    session: AsyncSession,
    user: User,
    *,
    expires_at: datetime | None = None,
    content_type: str = "audio/flac",
    duration_ms: int | None = 6000,
    product_id: str | None = None,
    peaks: bytes | None = None,
    with_profile: bool = True,
    detail_peaks: bytes | None = None,
) -> UserImage:
    """Insert and flush an audio upload and its ``library_audio_profiles`` row.

    What the upload service writes for audio: a ``user_images`` row with no width / height and a
    ``duration_ms``, plus the profile (probed facts and the 2,048-byte peaks). ``detail_peaks``
    is the 20 ms level; left out, it is NULL, as for audio stored before S3a (not backfilled).
    """
    image_id = uuid4()
    extension = AUDIO_FORMATS[content_type]
    image = UserImage(
        id=image_id,
        user_id=user.id,
        product_id=product_id or user.product_id,
        storage_key=f"users/{user.id}/uploads/{image_id}.{extension}",
        original_filename=f"{image_id}.{extension}",
        content_type=content_type,
        size_bytes=1000,
        format=extension,
        width=None,
        height=None,
        duration_ms=duration_ms,
        is_thumbnail=False,
        expires_at=expires_at or future(),
    )
    session.add(image)
    await session.flush()
    if with_profile:
        session.add(
            LibraryAudioProfile(
                id=uuid4(),
                upload_id=image_id,
                codec="flac",
                sample_rate_hz=44100,
                channels=2,
                bit_rate=900000,
                peaks=peaks if peaks is not None else bytes(range(256)) * 8,
                peaks_version=1,
                detail_peaks=detail_peaks,
                detail_peaks_version=None if detail_peaks is None else 1,
            )
        )
        await session.flush()
    return image


async def make_job(
    session: AsyncSession,
    user: User,
    *,
    status: JobStatus = JobStatus.COMPLETED,
    is_deleted: bool = False,
) -> GenerationJob:
    job = GenerationJob(
        id=uuid4(),
        user_id=user.id,
        product_id=user.product_id,
        name="Hold Test Job",
        status=status.value,
        generation_type=GenerationType.T2I.value,
        prompt="a test image",
        provider="grok",
        is_deleted=is_deleted,
    )
    session.add(job)
    await session.flush()
    return job


async def make_output(
    session: AsyncSession,
    user: User,
    job: GenerationJob,
    *,
    expires_at: datetime | None = None,
    content_type: str = "image/png",
    is_thumbnail: bool = False,
    parent: GenerationOutput | None = None,
    output_index: int = 0,
) -> GenerationOutput:
    """Insert and flush an output (or a thumbnail of ``parent``) of ``job``."""
    output_id = uuid4()
    output = GenerationOutput(
        id=output_id,
        user_id=user.id,
        job_id=job.id,
        product_id=user.product_id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{output_id}.bin",
        content_type=content_type,
        size_bytes=100,
        format="png",
        output_index=output_index,
        width=64,
        height=64,
        is_thumbnail=is_thumbnail,
        parent_output_id=parent.id if parent is not None else None,
        thumbnail_max_edge=150 if is_thumbnail else None,
        expires_at=expires_at or future(),
    )
    session.add(output)
    await session.flush()
    return output


@dataclass
class HoldWorld:
    """Committed users and assets, removed (with everything hanging off them) afterwards."""

    factory: async_sessionmaker[AsyncSession]
    _users: list[UUID] = field(default_factory=list)

    async def user(self, *, product_id: str = "vex") -> User:
        async with self.factory() as session:
            user = User(
                id=uuid4(),
                email=f"holds-{uuid4().hex}@example.com",
                password_hash="x",
                product_id=product_id,
            )
            session.add(user)
            await session.commit()
        self._users.append(user.id)
        return user

    async def upload(self, user: User, **kwargs: object) -> UserImage:
        async with self.factory() as session:
            image = await make_upload(session, user, **kwargs)  # type: ignore[arg-type]
            await session.commit()
            return image

    async def audio(self, user: User, **kwargs: object) -> UserImage:
        """A committed audio upload with its profile."""
        async with self.factory() as session:
            image = await make_audio_upload(session, user, **kwargs)  # type: ignore[arg-type]
            await session.commit()
            return image

    async def job(self, user: User, **kwargs: object) -> GenerationJob:
        async with self.factory() as session:
            job = await make_job(session, user, **kwargs)  # type: ignore[arg-type]
            await session.commit()
            return job

    async def output(self, user: User, job: GenerationJob, **kwargs: object) -> GenerationOutput:
        async with self.factory() as session:
            output = await make_output(session, user, job, **kwargs)  # type: ignore[arg-type]
            await session.commit()
            return output

    async def upload_expires_at(self, image: UserImage) -> datetime:
        async with self.factory() as session:
            return (
                await session.execute(select(UserImage.expires_at).where(UserImage.id == image.id))
            ).scalar_one()

    async def output_expires_at(self, output: GenerationOutput) -> datetime:
        async with self.factory() as session:
            return (
                await session.execute(
                    select(GenerationOutput.expires_at).where(GenerationOutput.id == output.id)
                )
            ).scalar_one()

    async def upload_exists(self, image: UserImage) -> bool:
        async with self.factory() as session:
            return (
                await session.execute(select(UserImage.id).where(UserImage.id == image.id))
            ).scalar_one_or_none() is not None

    async def output_exists(self, output: GenerationOutput) -> bool:
        async with self.factory() as session:
            return (
                await session.execute(
                    select(GenerationOutput.id).where(GenerationOutput.id == output.id)
                )
            ).scalar_one_or_none() is not None

    async def hold_count(self) -> int:
        """Holds that belong to the users of this world."""
        async with self.factory() as session:
            return (
                await session.execute(
                    select(text("count(*)"))
                    .select_from(LibraryAssetHold)
                    .where(LibraryAssetHold.user_id.in_(self._users))
                )
            ).scalar_one()

    async def cleanup(self) -> None:
        """Hard-delete the users: the cascade removes holds and assets in one statement."""
        async with self.factory() as session, session.begin():
            await session.execute(delete(User).where(User.id.in_(self._users)))
        self._users.clear()


# ---------------------------------------------------------------------------------------------
# HTTP: a small app with the real controllers, guards, DI providers and handlers
# ---------------------------------------------------------------------------------------------

SECRET = "test_secret_key_for_testing_only_256bits_long"
RETENTION_DAYS = 7


def bearer(
    user_id: UUID, *, product: str = "vex", digest: str | None = TEST_LEGAL_DIGEST
) -> dict[str, str]:
    """An ``Authorization`` header for ``user_id`` (a stale ``digest`` makes mutations 428)."""
    token, _ = JWTService(JWTConfig(secret_key=SECRET)).create_access_token(
        user_id, product_id=product, legal_digest=digest
    )
    return {"Authorization": f"Bearer {token}"}


def _scope_for(config: ProductConfig) -> Callable[[ASGIApp], ASGIApp]:
    def factory(app: ASGIApp) -> ASGIApp:
        async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] in ("http", "websocket"):
                state = scope.setdefault("state", {})
                state["product_config"] = config
                state["product_id"] = config.slug
            await app(scope, receive, send)

        return middleware

    return factory


def make_holds(session: AsyncSession) -> AssetHoldService:
    """The library's hold service on ``session`` (the studio services take it as a dependency)."""
    return AssetHoldService(session, retention_days=RETENTION_DAYS)


def provide_cast_service(session: AsyncSession) -> CastService:
    """``get_studio_cast_service`` without ``Settings`` (a unit test holds that wiring)."""
    return CastService(session, make_holds(session))


def provide_track_service(session: AsyncSession) -> TrackService:
    """``get_studio_track_service`` without ``Settings`` (a unit test holds that wiring)."""
    return TrackService(session, make_holds(session))


def provide_version_service(session: AsyncSession) -> DocumentVersionService:
    """``get_studio_version_service`` (a plain function: Litestar resolves its hints at runtime)."""
    return DocumentVersionService(session)


def provide_track_audio_service(session: AsyncSession) -> TrackAudioService:
    """``get_studio_track_audio_service`` without ``Settings`` (a unit test holds that wiring)."""
    return TrackAudioService(session, make_holds(session))


def provide_timeline_service(session: AsyncSession) -> TimelineService:
    """``get_studio_timeline_service`` (a plain function: Litestar resolves its hints)."""
    return TimelineService(session)


def _unused_job_service() -> MagicMock:
    """Stands in for ``UnifiedJobService`` (only the delete route is exercised)."""
    return MagicMock()


def build_app(
    factory: async_sessionmaker[AsyncSession],
    controllers: Sequence[type],
    *,
    config: ProductConfig = VEX_CONFIG,
) -> Litestar:
    """The given controllers behind the real guards, handlers and per-request DI.

    The session provider mirrors ``get_db_session`` (commit on success, rollback when the handler
    raises), which is what makes a 409 leave nothing behind. Rows are committed: ``AsyncTestClient``
    runs handlers on another event loop, so the savepoint ``db_session`` cannot be shared.
    """

    async def provide_session() -> AsyncGenerator[AsyncSession]:
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    storage = MagicMock()
    storage.delete = AsyncMock()

    def provide_content_proxy() -> ContentProxyService:
        return ContentProxyService(
            storage=storage,
            settings=SimpleNamespace(content_url_ttl=60),  # type: ignore[arg-type]
        )

    def dep(fn: Callable[..., Any]) -> Provide:
        return Provide(fn, sync_to_thread=False)

    app = Litestar(
        route_handlers=list(controllers),
        dependencies={
            "session": Provide(provide_session),
            "product_id": dep(get_product_id),
            "product_config": dep(get_product_config),
            "content_proxy": dep(provide_content_proxy),
            "library_service": dep(get_library_service),
            "unified_job_service": dep(_unused_job_service),
            "studio_artist_service": dep(get_studio_artist_service),
            "studio_cast_service": dep(provide_cast_service),
            "studio_track_service": dep(provide_track_service),
            "studio_version_service": dep(provide_version_service),
            "studio_track_audio_service": dep(provide_track_audio_service),
            "studio_timeline_service": dep(provide_timeline_service),
        },
        exception_handlers={
            HTTPException: http_exception_handler,
            StudioError: studio_error_handler,
            AssetHeldError: asset_held_handler,
            JobInProgressError: job_in_progress_handler,
            LegalAcceptanceRequiredError: legal_acceptance_required_handler,
            Exception: global_exception_handler,
        },
        middleware=[_scope_for(config)],
    )
    app.state["jwt_service"] = JWTService(JWTConfig(secret_key=SECRET))
    app.state["token_revocation"] = TokenRevocationService(None, max_token_ttl_seconds=0)
    app.state["legal_registry"] = TEST_LEGAL_REGISTRY
    return app
