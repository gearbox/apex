"""Wire-level frame lineage: real multipart through ``POST /v1/storage/upload``.

The route unit tests hand-build ``UploadForm`` and the service tests call
``UserContentService`` directly, so neither exercises Litestar's multipart
parsing of ``source_asset_ref`` / ``source_timestamp_ms``. These tests post real
multipart bodies through the real ``StorageController`` (product middleware, auth
and legal guards) against a real database, with only R2 stubbed. They pin the
framework guarantee that form values reach ``UploadForm`` as raw ``str`` (``""``
included), which ``parse_frame_lineage`` relies on to reject them uniformly.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from litestar import Litestar
from litestar.di import Provide
from litestar.status_codes import HTTP_201_CREATED, HTTP_400_BAD_REQUEST
from litestar.testing import TestClient
from PIL import Image
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.product import ProductMiddleware
from src.api.routes.storage import StorageController
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.token_revocation import TokenRevocationService
from src.api.services.user_content import UserContentService
from src.core.product_registry import VEX_CONFIG
from src.db.models.storage import UserImage
from src.db.models.user import User
from tests.legal_support import make_legal_registry

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from httpx import Response
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.api.services.media_ingest import MediaIngestService

pytestmark = pytest.mark.asyncio

_SECRET = "test_secret_key_for_testing_only_256bits_long"
_PRODUCT_ID = "vex"
_DURATION_MS = 10_000
_REGISTRY = make_legal_registry()
_LINEAGE_ERROR = {
    "error": "invalid_frame_lineage",
    "message": "Frame source is not available",
    "status_code": HTTP_400_BAD_REQUEST,
}


def _png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 48), (10, 20, 30)).save(out, format="PNG")
    return out.getvalue()


def _storage() -> MagicMock:
    storage = MagicMock()

    async def upload(**kwargs: Any) -> MagicMock:
        object_id = uuid4()
        ext = kwargs["content_type"].split("/")[-1]
        return MagicMock(id=object_id, storage_key=f"test/{object_id}.{ext}")

    storage.upload = AsyncMock(side_effect=upload)
    return storage


def _app(engine: AsyncEngine, media_ingestor: MediaIngestService) -> Litestar:
    storage = _storage()

    async def provide_session() -> AsyncGenerator[AsyncSession]:
        # Mirrors ``get_db_session``: a request-scoped session that commits on success.
        async with AsyncSession(bind=engine, expire_on_commit=False) as session:
            yield session
            await session.commit()

    def provide_user_content(session: AsyncSession) -> UserContentService:
        return UserContentService(
            storage,
            session,
            product_id=_PRODUCT_ID,
            retention_days=7,
            media_ingestor=media_ingestor,
        )

    app = Litestar(
        route_handlers=[StorageController],
        middleware=[ProductMiddleware],
        dependencies={
            "session": Provide(provide_session),
            "user_content": Provide(provide_user_content, sync_to_thread=False),
        },
    )
    app.state["jwt_service"] = JWTService(JWTConfig(secret_key=_SECRET))
    app.state["token_revocation"] = TokenRevocationService(None, max_token_ttl_seconds=0)
    app.state["legal_registry"] = _REGISTRY
    return app


@dataclass(frozen=True)
class _Seed:
    user_id: UUID
    video_id: UUID


@pytest.fixture
async def seed(db_engine: AsyncEngine) -> AsyncGenerator[_Seed]:
    """A committed user with one video upload; the user's cascade removes every frame."""
    user_id, video_id = uuid4(), uuid4()
    async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
        session.add(
            User(
                id=user_id,
                email=f"mp-{uuid4().hex[:8]}@example.com",
                password_hash="x" * 64,
                is_active=True,
                product_id=_PRODUCT_ID,
            )
        )
        await session.flush()
        session.add(
            UserImage(
                id=video_id,
                user_id=user_id,
                storage_key=f"users/{user_id}/uploads/{video_id}.mp4",
                original_filename=f"{video_id}.mp4",
                content_type="video/mp4",
                size_bytes=1000,
                format="mp4",
                width=1280,
                height=720,
                duration_ms=_DURATION_MS,
                expires_at=datetime.now(UTC) + timedelta(days=1),
                product_id=_PRODUCT_ID,
            )
        )
        await session.commit()
    yield _Seed(user_id=user_id, video_id=video_id)
    async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
        await session.execute(delete(User).where(User.id == user_id))
        await session.commit()


def _post_frame(app: Litestar, user_id: UUID, fields: dict[str, str]) -> Response:
    digest = _REGISTRY.required_digest(VEX_CONFIG, today=datetime.now(UTC).date())
    token, _ = app.state["jwt_service"].create_access_token(
        user_id, product_id=_PRODUCT_ID, legal_digest=digest
    )
    with TestClient(app=app) as client:
        return client.post(
            "/v1/storage/upload",
            headers={"Authorization": f"Bearer {token}", "X-Product-Id": _PRODUCT_ID},
            files={"data": ("frame.png", _png(), "image/png")},
            data=fields,
        )


async def _lineage_columns(
    engine: AsyncEngine, upload_id: str
) -> tuple[UUID | None, UUID | None, int | None]:
    async with AsyncSession(bind=engine) as session:
        row = await session.get(UserImage, UUID(upload_id))
        assert row is not None
        return row.source_upload_id, row.source_output_id, row.source_timestamp_ms


async def test_multipart_frame_with_valid_lineage_is_stored_with_lineage(
    db_engine: AsyncEngine, media_ingestor: MediaIngestService, seed: _Seed
) -> None:
    response = _post_frame(
        _app(db_engine, media_ingestor),
        seed.user_id,
        {"source_asset_ref": f"upload:{seed.video_id}", "source_timestamp_ms": "0"},
    )

    assert response.status_code == HTTP_201_CREATED
    assert await _lineage_columns(db_engine, response.json()["id"]) == (seed.video_id, None, 0)


async def test_multipart_timestamp_without_source_is_rejected(
    db_engine: AsyncEngine, media_ingestor: MediaIngestService, seed: _Seed
) -> None:
    response = _post_frame(
        _app(db_engine, media_ingestor), seed.user_id, {"source_timestamp_ms": "0"}
    )

    assert response.status_code == HTTP_400_BAD_REQUEST
    assert {k: response.json()[k] for k in _LINEAGE_ERROR} == _LINEAGE_ERROR


async def test_multipart_empty_timestamp_is_rejected_not_treated_as_absent(
    db_engine: AsyncEngine, media_ingestor: MediaIngestService, seed: _Seed
) -> None:
    response = _post_frame(
        _app(db_engine, media_ingestor),
        seed.user_id,
        {"source_asset_ref": f"upload:{seed.video_id}", "source_timestamp_ms": ""},
    )

    assert response.status_code == HTTP_400_BAD_REQUEST
    assert {k: response.json()[k] for k in _LINEAGE_ERROR} == _LINEAGE_ERROR


async def test_multipart_frame_without_lineage_fields_is_an_ordinary_upload(
    db_engine: AsyncEngine, media_ingestor: MediaIngestService, seed: _Seed
) -> None:
    response = _post_frame(_app(db_engine, media_ingestor), seed.user_id, {})

    assert response.status_code == HTTP_201_CREATED
    assert await _lineage_columns(db_engine, response.json()["id"]) == (None, None, None)
