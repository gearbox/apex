"""W2-c / W2-d — an address registered on two products never crosses the product boundary.

W2-c drives ``POST /v1/auth/forgot-password`` through ``ProductMiddleware`` and the real
route/DI shape against PostgreSQL; W2-d drives ``UserService.update_profile``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import structlog.testing
from litestar import Litestar
from litestar.di import Provide
from litestar.testing import TestClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.common import get_product_config, get_product_id
from src.api.dependencies.request_context import provide_request_context
from src.api.middleware.product import ProductMiddleware
from src.api.routes.auth import AuthController
from src.api.services.age_verification import AgeVerificationService
from src.api.services.email_verification import EmailVerificationService
from src.api.services.token_revocation import TokenRevocationService
from src.api.services.user import EmailAlreadyExistsError, UserService
from src.core.config import Settings
from src.core.product_registry import (
    SYNTHARA_CONFIG,
    VEX_CONFIG,
    get_product_config_by_slug,
)
from src.db.models.auth_tokens import PasswordResetToken
from src.db.models.user import User
from src.db.repositories.user import UserRepository
from src.db.repositories.user_identity import UserIdentityRepository
from tests.email_support import RecordingEmailService
from tests.legal_support import make_legal_acceptance_service
from tests.revocation_support import make_session_termination, make_session_termination_factory_noop

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine

SHARED_EMAIL = "both-brands@example.com"
VEX_APP = "https://app.vex.test"
SYNTHARA_APP = "https://app.synthara.test"


async def _seed(engine: AsyncEngine, *, email: str, product_id: str) -> UUID:
    user_id = uuid4()
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add(
            User(
                id=user_id,
                email=email,
                password_hash="x" * 64,
                display_name=f"{product_id} user",
                product_id=product_id,
                is_active=True,
                locale="en",
            )
        )
        await session.commit()
    return user_id


async def _cleanup(engine: AsyncEngine, user_ids: list[UUID]) -> None:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(
            delete(PasswordResetToken).where(PasswordResetToken.user_id.in_(user_ids))
        )
        await session.execute(delete(User).where(User.id.in_(user_ids)))
        await session.commit()


@pytest.fixture
async def two_products(db_engine: AsyncEngine) -> AsyncGenerator[dict[str, UUID]]:
    ids = {
        "vex": await _seed(db_engine, email=SHARED_EMAIL, product_id="vex"),
        "synthara": await _seed(db_engine, email=SHARED_EMAIL, product_id="synthara"),
    }
    yield ids
    await _cleanup(db_engine, list(ids.values()))


def _app(engine: AsyncEngine, email: RecordingEmailService) -> Litestar:
    settings = Settings(app_url_vex=VEX_APP, app_url_synthara=SYNTHARA_APP, debug=True)
    service = EmailVerificationService(
        email_service=email,
        app_url_for=settings.app_url_for,
        brand_for=lambda slug: get_product_config_by_slug(slug).display_name,
        session_termination_factory=make_session_termination_factory_noop(),
    )

    async def session_provider() -> AsyncGenerator[AsyncSession]:
        async with AsyncSession(bind=engine, expire_on_commit=False) as session:
            yield session

    stub: Any = MagicMock()
    return Litestar(
        route_handlers=[AuthController],
        middleware=[ProductMiddleware],
        dependencies={
            "session": Provide(session_provider),
            "email_verification_service": Provide(lambda: service, sync_to_thread=False),
            "product_config": Provide(get_product_config, sync_to_thread=False),
            "product_id": Provide(get_product_id, sync_to_thread=False),
            "request_context": Provide(provide_request_context, sync_to_thread=False),
            "settings": Provide(lambda: settings, sync_to_thread=False),
            # Only needed to satisfy the controller's other handlers' signatures.
            "auth_service": Provide(lambda: stub, sync_to_thread=False),
            "jwt_service": Provide(lambda: stub, sync_to_thread=False),
            "oauth_registry": Provide(lambda: stub, sync_to_thread=False),
            "current_user_id": Provide(uuid4, sync_to_thread=False),
        },
    )


def _forgot(client: TestClient[Any], origin: str) -> Any:
    return client.post(
        "/v1/auth/forgot-password",
        json={"email": SHARED_EMAIL},
        headers={"Origin": origin},
    )


@pytest.mark.parametrize(
    ("origin", "expected_link", "brand", "other_link"),
    [
        ("https://vex.pics", VEX_APP, "vex.pics", SYNTHARA_APP),
        ("https://synthara.app", SYNTHARA_APP, "Synthara", VEX_APP),
    ],
    ids=["vex_host", "synthara_host"],
)
def test_w2_c_forgot_password_with_the_same_address_on_both_products(
    db_engine: AsyncEngine,
    two_products: dict[str, UUID],  # noqa: ARG001
    origin: str,
    expected_link: str,
    brand: str,
    other_link: str,
) -> None:
    email = RecordingEmailService()

    with (
        structlog.testing.capture_logs() as logs,
        TestClient(app=_app(db_engine, email)) as client,
    ):
        response = _forgot(client, origin)

    assert response.status_code == 200
    assert [
        entry["event"] for entry in logs if entry["event"] == "auth.forgot_password_failed"
    ] == []
    (message,) = email.sent  # exactly one email — MultipleResultsFound used to swallow it
    assert message.to == SHARED_EMAIL
    assert f"{expected_link}/reset-password?token=" in message.text_body
    assert other_link not in message.text_body
    assert brand in message.subject
    assert "Apex" not in f"{message.subject}{message.text_body}{message.html_body}"


async def test_w2_c_the_reset_token_belongs_to_the_requesting_products_account(
    db_engine: AsyncEngine, two_products: dict[str, UUID]
) -> None:
    email = RecordingEmailService()
    settings_app = _app(db_engine, email)

    with TestClient(app=settings_app) as client:
        assert _forgot(client, "https://synthara.app").status_code == 200

    async with AsyncSession(bind=db_engine) as session:
        owners = (await session.execute(PasswordResetToken.__table__.select())).all()
    owner_ids = {row.user_id for row in owners} & set(two_products.values())
    assert owner_ids == {two_products["synthara"]}


def _user_service(session: AsyncSession) -> UserService:
    repo = UserRepository(session)
    return UserService(
        repository=repo,
        password_service=MagicMock(),
        age_verification_service=AgeVerificationService(),
        legal_acceptance_service=make_legal_acceptance_service(),
        identity_repository=UserIdentityRepository(session),
        session_termination=make_session_termination(
            user_repo=repo,
            token_revocation=TokenRevocationService(None, max_token_ttl_seconds=0),
            session=session,
        ),
    )


async def test_w2_d_email_change_only_collides_within_the_same_product(
    db_session: AsyncSession, make_user: Any
) -> None:
    vex_user = await make_user(email="mover@example.com", product_id="vex")
    vex_user_id = vex_user.id
    await make_user(email="synthara-only@example.com", product_id="synthara")
    await make_user(email="vex-taken@example.com", product_id="vex")
    config = MagicMock()
    config.age_gate = VEX_CONFIG.age_gate
    service = _user_service(db_session)

    # An address only the other brand knows is free for this product...
    profile = await service.update_profile(
        vex_user_id, product_config=config, email="synthara-only@example.com"
    )
    assert profile.email == "synthara-only@example.com"

    # ...but one already used on this product is still rejected.
    with pytest.raises(EmailAlreadyExistsError):
        await service.update_profile(
            vex_user_id, product_config=config, email="vex-taken@example.com"
        )


async def test_w2_d_synthara_user_can_take_an_address_only_vex_uses(
    db_session: AsyncSession, make_user: Any
) -> None:
    synthara_user = await make_user(email="s-mover@example.com", product_id="synthara")
    synthara_user_id = synthara_user.id
    await make_user(email="vex-only@example.com", product_id="vex")
    config = MagicMock()
    config.age_gate = SYNTHARA_CONFIG.age_gate

    profile = await _user_service(db_session).update_profile(
        synthara_user_id, product_config=config, email="vex-only@example.com"
    )

    assert profile.email == "vex-only@example.com"
