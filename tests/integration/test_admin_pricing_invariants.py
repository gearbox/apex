"""R3 / P2-1, P2-2: pricing-rule invariants are a 422 from the service, never a 500 or a latent
end-user failure (real PostgreSQL 16, real ``AdminController`` + ``PricingService``).

0ed11cd probes (inverted here):

* ``PATCH {"per_second_token_cost": 0}`` on a per-second-only rule reached
  ``chk_pricing_costs_valid`` and surfaced as an unhandled ``IntegrityError`` (500).
* ``POST`` a ``t2i`` rule with ``per_second_token_cost > 0`` / ``video_resolution`` was
  accepted; every later image generation for that model then failed ``quote()``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import httpx
import pytest
from litestar import Litestar
from litestar.datastructures import State
from litestar.di import Provide
from sqlalchemy import select
from structlog.testing import capture_logs

from src.api.app import invalid_pricing_rule_handler, legal_acceptance_required_handler
from src.api.dependencies.common import get_product_config, get_product_id
from src.api.middleware.product import ProductMiddleware
from src.api.routes.admin import AdminController
from src.api.security import JWTConfig, JWTService
from src.api.services.billing_errors import InvalidPricingRuleError
from src.api.services.legal.errors import LegalAcceptanceRequiredError
from src.api.services.pricing import PricingService
from src.api.services.token_revocation import TokenRevocationService
from src.core.enums import UserRole
from src.core.product_registry import VEX_CONFIG
from src.core.uid import new_id
from src.db.models.billing import PricingRule
from tests.legal_support import make_legal_registry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.user import User
    from tests.integration.conftest import UserFactory

pytestmark = pytest.mark.asyncio

JWT_SECRET = "integration-pricing-secret-key-32-bytes"


class _Api:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.jwt = JWTService(JWTConfig(secret_key=JWT_SECRET))
        self.registry = make_legal_registry()
        self.service = PricingService()

    def client(self) -> httpx.AsyncClient:
        app = Litestar(
            route_handlers=[AdminController],
            middleware=[ProductMiddleware],
            dependencies={
                "product_config": Provide(get_product_config, sync_to_thread=False),
                "product_id": Provide(get_product_id, sync_to_thread=False),
                "session": Provide(lambda: self.session, sync_to_thread=False),
                "pricing_service": Provide(lambda: self.service, sync_to_thread=False),
            },
            exception_handlers={
                LegalAcceptanceRequiredError: legal_acceptance_required_handler,
                InvalidPricingRuleError: invalid_pricing_rule_handler,
            },
            state=State(
                {
                    "jwt_service": self.jwt,
                    "token_revocation": TokenRevocationService(None, max_token_ttl_seconds=0),
                    "legal_registry": self.registry,
                }
            ),
        )
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=cast("Any", app)),
            base_url="http://testserver.local",
        )

    def headers(self, user: User) -> dict[str, str]:
        token, _ = self.jwt.create_access_token(
            user.id,
            product_id=user.product_id,
            legal_digest=self.registry.required_digest(VEX_CONFIG, today=datetime.now(UTC).date()),
        )
        return {"Authorization": f"Bearer {token}", "X-Product-Id": user.product_id}


@pytest.fixture
async def admin(make_user: UserFactory) -> User:
    user = await make_user(email=f"pricing-{uuid4().hex[:10]}@example.com", product_id="vex")
    user.role = UserRole.ADMIN
    return user


async def _insert_rule(session: AsyncSession, admin: User, **fields: Any) -> PricingRule:
    """Insert straight through the ORM, bypassing the service (legacy / seeded rows)."""
    values: dict[str, Any] = {
        "id": new_id(),
        "provider": "alibaba",
        "generation_type": "t2v",
        "model": f"probe-{uuid4().hex[:8]}",
        "token_cost": 0,
        "input_token_cost": 0,
        "per_second_token_cost": 5,
        "video_resolution": None,
        "is_active": True,
        "created_by": admin.id,
    } | fields
    rule = PricingRule(**values)
    session.add(rule)
    await session.flush()
    return rule


def _no_error_level_events(logs: Sequence[Mapping[str, Any]]) -> bool:
    return not [e for e in logs if e.get("log_level") in {"error", "critical"}]


async def test_patch_to_zero_per_second_is_a_422_and_leaves_the_row_untouched(
    db_session: AsyncSession, admin: User
) -> None:
    api = _Api(db_session)
    rule = await _insert_rule(db_session, admin, token_cost=0, per_second_token_cost=5)

    with capture_logs() as logs:
        async with api.client() as client:
            response = await client.patch(
                f"/v1/admin/pricing/{rule.id}",
                json={"per_second_token_cost": 0},
                headers=api.headers(admin),
            )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_pricing_rule"
    await db_session.refresh(rule)
    assert (rule.token_cost, rule.per_second_token_cost) == (0, 5)
    assert _no_error_level_events(logs)


async def test_patch_that_zeroes_the_last_cost_via_token_cost_is_a_422(
    db_session: AsyncSession, admin: User
) -> None:
    api = _Api(db_session)
    rule = await _insert_rule(db_session, admin, token_cost=7, per_second_token_cost=0)

    async with api.client() as client:
        response = await client.patch(
            f"/v1/admin/pricing/{rule.id}", json={"token_cost": 0}, headers=api.headers(admin)
        )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_pricing_rule"


async def test_patch_with_a_negative_cost_is_rejected_by_the_schema_not_the_database(
    db_session: AsyncSession, admin: User
) -> None:
    api = _Api(db_session)
    rule = await _insert_rule(db_session, admin, token_cost=7, per_second_token_cost=0)

    async with api.client() as client:
        response = await client.patch(
            f"/v1/admin/pricing/{rule.id}", json={"token_cost": -1}, headers=api.headers(admin)
        )

    assert response.status_code == 400


async def test_a_valid_patch_still_applies(db_session: AsyncSession, admin: User) -> None:
    api = _Api(db_session)
    rule = await _insert_rule(db_session, admin, token_cost=0, per_second_token_cost=5)

    async with api.client() as client:
        response = await client.patch(
            f"/v1/admin/pricing/{rule.id}",
            json={"per_second_token_cost": 9, "notes": "raised"},
            headers=api.headers(admin),
        )

    assert response.status_code == 200
    assert response.json()["per_second_token_cost"] == 9


async def test_a_broken_legacy_rule_can_still_be_deactivated(
    db_session: AsyncSession, admin: User
) -> None:
    """A patch that does not touch costs is never blocked by the rule's existing state."""
    api = _Api(db_session)
    broken = await _insert_rule(
        db_session, admin, generation_type="t2i", token_cost=1, per_second_token_cost=3
    )

    async with api.client() as client:
        response = await client.patch(
            f"/v1/admin/pricing/{broken.id}", json={"is_active": False}, headers=api.headers(admin)
        )

    assert response.status_code == 200
    assert response.json()["is_active"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"per_second_token_cost": 3}, id="per-second-on-image"),
        pytest.param({"video_resolution": "1080p"}, id="resolution-on-image"),
    ],
)
@pytest.mark.parametrize("generation_type", ["t2i", "i2i"])
async def test_image_rules_with_video_only_pricing_are_a_422(
    db_session: AsyncSession,
    admin: User,
    generation_type: str,
    overrides: dict[str, Any],
) -> None:
    api = _Api(db_session)
    body = {
        "provider": "grok",
        "generation_type": generation_type,
        "model": f"probe-image-{uuid4().hex[:8]}",
        "token_cost": 1,
    } | overrides

    async with api.client() as client:
        response = await client.post("/v1/admin/pricing", json=body, headers=api.headers(admin))

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_pricing_rule"
    stored = (
        await db_session.scalars(select(PricingRule).where(PricingRule.model == body["model"]))
    ).all()
    assert stored == []


async def test_unknown_generation_type_is_a_422(db_session: AsyncSession, admin: User) -> None:
    api = _Api(db_session)

    async with api.client() as client:
        response = await client.post(
            "/v1/admin/pricing",
            json={"provider": "grok", "generation_type": "t2x", "model": "m", "token_cost": 1},
            headers=api.headers(admin),
        )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_pricing_rule"


async def test_a_per_second_t2v_1080p_rule_is_created(
    db_session: AsyncSession, admin: User
) -> None:
    api = _Api(db_session)

    async with api.client() as client:
        response = await client.post(
            "/v1/admin/pricing",
            json={
                "provider": "alibaba",
                "generation_type": "t2v",
                "model": f"probe-{uuid4().hex[:8]}",
                "token_cost": 0,
                "per_second_token_cost": 4,
                "video_resolution": "1080p",
            },
            headers=api.headers(admin),
        )

    assert response.status_code == 201
    assert response.json()["video_resolution"] == "1080p"


async def test_quote_still_fails_loudly_for_a_legacy_image_per_second_rule(
    db_session: AsyncSession, admin: User
) -> None:
    """The fail-loud guard stays for programming errors; admin input can no longer reach it."""
    model = f"probe-image-{uuid4().hex[:8]}"
    await _insert_rule(
        db_session,
        admin,
        provider="grok",
        generation_type="t2i",
        model=model,
        token_cost=1,
        per_second_token_cost=3,
    )

    with pytest.raises(ValueError, match="per-second rule requires billed_seconds"):
        await PricingService().quote(
            "grok", "t2i", model, n=1, source_media_count=0, session=db_session
        )
