"""auth_guard legal enforcement (C9) and the pinned exemption set (C10)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from uuid import uuid4

os.environ.setdefault("JWT_SECRET_KEY", "test-legal-guard-audit-key-32-bytes-long")

import pytest
from litestar import Litestar, delete, get, patch, post
from litestar.datastructures import State
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED
from litestar.testing import TestClient

from src.api.app import create_app, legal_acceptance_required_handler
from src.api.middleware.product import ProductMiddleware
from src.api.security import auth_guard
from src.api.security.guards import LEGAL_EXEMPT_OPT, _enforce_legal_acceptance
from src.api.security.jwt import JWTConfig, JWTService, TokenPayload
from src.api.services.legal.errors import LegalAcceptanceRequiredError
from src.api.services.token_revocation import TokenRevocationService
from src.core.product_registry import SYNTHARA_CONFIG, VEX_CONFIG
from tests.legal_support import make_legal_registry

if TYPE_CHECKING:
    from litestar.handlers import BaseRouteHandler

pytestmark = pytest.mark.unit

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
REGISTRY = make_legal_registry()
CURRENT_DIGEST = REGISTRY.required_digest(VEX_CONFIG, today=datetime.now(UTC).date())

# The reviewed set of non-safe handlers that bypass legal enforcement.
# Adding one here is a deliberate, reviewed change (see guards.LEGAL_EXEMPT_OPT).
_REVIEWED_LEGAL_EXEMPT: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/v1/legal/acceptances"),
        ("DELETE", "/v1/users/me"),
        # Password rotation must remain available to secure a compromised account.
        ("POST", "/v1/users/me/password"),
        ("POST", "/v1/users/me/logout-all"),
        ("POST", "/v1/auth/resend-verification"),
        ("POST", "/v1/auth/content-cookie"),
        ("POST", "/v1/events/sse-ticket"),
    }
)


@post("/mutate", guards=[auth_guard])
async def _mutate() -> dict[str, str]:
    return {"ok": "yes"}


@patch("/mutate", guards=[auth_guard])
async def _patch_mutate() -> dict[str, str]:
    return {"ok": "yes"}


@delete("/mutate", guards=[auth_guard], status_code=HTTP_200_OK)
async def _delete_mutate() -> dict[str, str]:
    return {"ok": "yes"}


@get("/read", guards=[auth_guard])
async def _read() -> dict[str, str]:
    return {"ok": "yes"}


@post("/exempt", guards=[auth_guard], opt={LEGAL_EXEMPT_OPT: True})
async def _exempt() -> dict[str, str]:
    return {"ok": "yes"}


@pytest.fixture
def jwt_service() -> JWTService:
    return JWTService(JWTConfig(secret_key=TEST_SECRET))


def _app(jwt_service: JWTService) -> Litestar:
    return Litestar(
        route_handlers=[_mutate, _patch_mutate, _delete_mutate, _read, _exempt],
        middleware=[ProductMiddleware],
        exception_handlers={LegalAcceptanceRequiredError: legal_acceptance_required_handler},
        state=State(
            {
                "jwt_service": jwt_service,
                "token_revocation": TokenRevocationService(None, max_token_ttl_seconds=0),
                "legal_registry": REGISTRY,
            }
        ),
    )


def _headers(
    jwt_service: JWTService, *, product: str = "vex", lgl: str | None = None
) -> dict[str, str]:
    token, _ = jwt_service.create_access_token(uuid4(), product_id=product, legal_digest=lgl)
    return {"Authorization": f"Bearer {token}", "X-Product-Id": product}


class TestGuardBlocksMutations:
    """C9 — stale/absent lgl → 428 on non-safe methods, never on GET."""

    @pytest.mark.parametrize("lgl", [None, "0000000000000000"])
    @pytest.mark.parametrize("method", ["post", "patch", "delete"])
    def test_stale_or_absent_digest_gets_428(
        self, jwt_service: JWTService, lgl: str | None, method: str
    ) -> None:
        with TestClient(app=_app(jwt_service)) as client:
            resp = getattr(client, method)("/mutate", headers=_headers(jwt_service, lgl=lgl))
        assert resp.status_code == 428
        body = resp.json()
        assert body["error"] == "legal_acceptance_required"
        assert body["status_code"] == 428

    @pytest.mark.parametrize("lgl", [None, "0000000000000000"])
    def test_get_passes_without_digest(self, jwt_service: JWTService, lgl: str | None) -> None:
        with TestClient(app=_app(jwt_service)) as client:
            resp = client.get("/read", headers=_headers(jwt_service, lgl=lgl))
        assert resp.status_code == HTTP_200_OK

    def test_current_digest_passes(self, jwt_service: JWTService) -> None:
        with TestClient(app=_app(jwt_service)) as client:
            resp = client.post("/mutate", headers=_headers(jwt_service, lgl=CURRENT_DIGEST))
        assert resp.status_code == HTTP_201_CREATED

    def test_exempt_handler_passes_without_digest(self, jwt_service: JWTService) -> None:
        with TestClient(app=_app(jwt_service)) as client:
            resp = client.post("/exempt", headers=_headers(jwt_service))
        assert resp.status_code == HTTP_201_CREATED

    def test_product_without_requirements_passes(self, jwt_service: JWTService) -> None:
        with TestClient(app=_app(jwt_service)) as client:
            resp = client.post("/mutate", headers=_headers(jwt_service, product="synthara"))
        assert resp.status_code == HTTP_201_CREATED

    def test_missing_registry_is_a_wiring_error(self, jwt_service: JWTService) -> None:
        app = _app(jwt_service)
        del app.state["legal_registry"]
        with TestClient(app=app, raise_server_exceptions=False) as client:
            resp = client.post("/mutate", headers=_headers(jwt_service))
        assert resp.status_code == 500


def _real_handlers() -> list[tuple[str, str, BaseRouteHandler]]:
    app = create_app()
    found: list[tuple[str, str, BaseRouteHandler]] = []
    for route in app.routes:
        for handler in getattr(route, "route_handlers", []):
            found.extend((m, route.path, handler) for m in sorted(handler.http_methods))
    return found


class TestExemptionAudit:
    """C10 — the exact exemption set is pinned, and each exempt route passes stale."""

    def test_exempt_set_matches_reviewed_list(self) -> None:
        actual = {
            (method, path)
            for method, path, handler in _real_handlers()
            if handler.opt.get(LEGAL_EXEMPT_OPT)
        }
        assert actual == _REVIEWED_LEGAL_EXEMPT, (
            "legal_exempt usage changed — every exemption must be reviewed and added to "
            f"_REVIEWED_LEGAL_EXEMPT. actual={sorted(actual)}"
        )

    def test_exempt_routes_are_all_auth_guarded(self) -> None:
        for method, path, handler in _real_handlers():
            if (method, path) in _REVIEWED_LEGAL_EXEMPT:
                assert auth_guard in handler.resolve_guards(), (method, path)

    @pytest.mark.parametrize(("method", "path"), sorted(_REVIEWED_LEGAL_EXEMPT))
    def test_exempt_route_passes_with_stale_digest(self, method: str, path: str) -> None:
        handler = next(h for m, p, h in _real_handlers() if (m, p) == (method, path))
        payload = TokenPayload(
            sub=str(uuid4()), exp=0, iat=0, jti="j", product_id="vex", legal_digest="stale"
        )
        _enforce_legal_acceptance(_connection(method), handler, payload)

    def test_non_exempt_mutation_is_blocked_with_stale_digest(self) -> None:
        handler = next(h for m, p, h in _real_handlers() if (m, p) == ("PATCH", "/v1/users/me"))
        payload = TokenPayload(
            sub=str(uuid4()), exp=0, iat=0, jti="j", product_id="vex", legal_digest="stale"
        )
        with pytest.raises(LegalAcceptanceRequiredError):
            _enforce_legal_acceptance(_connection("POST"), handler, payload)

    def test_missing_product_scope_is_a_wiring_error(self) -> None:
        handler = MagicMock()
        handler.opt = {}
        connection = _connection("POST")
        del connection.state["product_config"]
        payload = TokenPayload(sub=str(uuid4()), exp=0, iat=0, jti="j", product_id="vex")

        with pytest.raises(RuntimeError, match="Product scope missing"):
            _enforce_legal_acceptance(connection, handler, payload)

    def test_get_with_missing_product_scope_returns_before_lookup(self) -> None:
        handler = MagicMock()
        handler.opt = {}
        connection = _connection("GET")
        del connection.state["product_config"]
        payload = TokenPayload(sub=str(uuid4()), exp=0, iat=0, jti="j", product_id="vex")

        _enforce_legal_acceptance(connection, handler, payload)


def _connection(method: str, product: Any = VEX_CONFIG) -> MagicMock:
    connection = MagicMock()
    connection.scope = {"method": method}
    connection.state = {"product_config": product}
    connection.app.state = {"legal_registry": REGISTRY}
    return connection


def test_synthara_connection_never_enforced() -> None:
    handler = MagicMock()
    handler.opt = {}
    payload = TokenPayload(sub=str(uuid4()), exp=0, iat=0, jti="j", product_id="synthara")
    _enforce_legal_acceptance(_connection("POST", SYNTHARA_CONFIG), handler, payload)
