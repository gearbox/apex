"""HTTP-level tests for /v1/legal/* and the register route's legal errors.

Drives a minimal real Litestar app (ProductMiddleware + real exception
handlers + DI) so signature models, path/query parsing and error envelopes
run for real. Contracts: C16 (public endpoints), C4 (API returns the
normalised sha), and the envelope shapes behind C6.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from litestar import Litestar
from litestar.datastructures import State
from litestar.di import Provide
from litestar.testing import TestClient

from src.api.app import (
    legal_acceptance_required_handler,
    legal_document_not_found_handler,
    legal_submission_incomplete_handler,
    legal_version_stale_handler,
)
from src.api.dependencies.common import get_product_config, get_product_id
from src.api.middleware.product import ProductMiddleware
from src.api.routes.auth import AuthController
from src.api.routes.legal import LegalController
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.auth import AuthService
from src.api.services.legal.acceptance import LegalAcceptanceService, RequestContext
from src.api.services.legal.errors import (
    LegalAcceptanceRequiredError,
    LegalDocumentNotFoundError,
    LegalSubmissionIncompleteError,
    LegalVersionStaleError,
)
from src.api.services.legal.registry import LegalDocumentRegistry
from src.api.services.token_revocation import TokenRevocationService
from src.core.config import Settings
from src.core.enums import LegalDocumentType, Product
from src.db.repositories.legal import LegalAcceptanceRepository
from tests.legal_support import accept_all_current, make_legal_document, make_legal_registry

if TYPE_CHECKING:
    from src.db.models.legal import LegalAcceptance

pytestmark = pytest.mark.unit

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
VEX = {"X-Product-Id": "vex"}
SYNTHARA = {"X-Product-Id": "synthara"}
FUTURE = date(2999, 1, 1)
CRLF_BODY = "# Terms\r\n\r\nWindows line endings.\r\n"
LF_BODY = CRLF_BODY.replace("\r\n", "\n")


def _settings() -> Settings:
    return Settings()


def _registry() -> LegalDocumentRegistry:
    return make_legal_registry(
        make_legal_document(LegalDocumentType.TERMS, content=LF_BODY),
        make_legal_document(LegalDocumentType.TERMS, FUTURE, requires_reacceptance=True),
        make_legal_document(LegalDocumentType.PRIVACY),
        make_legal_document(LegalDocumentType.SENSITIVE_DATA_CONSENT),
    )


def _app(
    registry: LegalDocumentRegistry,
    *,
    legal_service: LegalAcceptanceService | None = None,
    auth_service: Any = None,
    jwt_service: JWTService | None = None,
) -> Litestar:
    jwt = jwt_service or JWTService(JWTConfig(secret_key=TEST_SECRET))
    deps: dict[str, Provide] = {
        "product_config": Provide(get_product_config, sync_to_thread=False),
        "product_id": Provide(get_product_id, sync_to_thread=False),
        "legal_registry": Provide(lambda: registry, sync_to_thread=False),
        "request_context": Provide(
            lambda: RequestContext(ip_address="203.0.113.9", user_agent="ua"),
            sync_to_thread=False,
        ),
        "jwt_service": Provide(lambda: jwt, sync_to_thread=False),
        "settings": Provide(_settings, sync_to_thread=False),
    }
    if legal_service is not None:
        deps["legal_acceptance_service"] = Provide(lambda: legal_service, sync_to_thread=False)
    if auth_service is not None:
        deps["auth_service"] = Provide(lambda: auth_service, sync_to_thread=False)
    return Litestar(
        route_handlers=[LegalController, AuthController],
        middleware=[ProductMiddleware],
        dependencies=deps,
        exception_handlers={
            LegalDocumentNotFoundError: legal_document_not_found_handler,
            LegalSubmissionIncompleteError: legal_submission_incomplete_handler,
            LegalVersionStaleError: legal_version_stale_handler,
            LegalAcceptanceRequiredError: legal_acceptance_required_handler,
        },
        state=State(
            {
                "jwt_service": jwt,
                "token_revocation": TokenRevocationService(None, max_token_ttl_seconds=0),
                "legal_registry": registry,
            }
        ),
    )


class TestPublicDocumentEndpoints:
    """C16 — public, product-scoped, 404 for unknown/unrequired."""

    def test_current_document_without_auth(self) -> None:
        registry = _registry()
        with TestClient(app=_app(registry)) as client:
            resp = client.get("/v1/legal/documents/terms", headers=VEX)
        assert resp.status_code == 200
        body = resp.json()
        expected = registry.current(Product.VEX, LegalDocumentType.TERMS, today=date(2030, 1, 1))
        assert body == {
            "doc_type": "terms",
            "version": "2020-01-01",
            "requires_reacceptance": True,
            "sha256": expected.sha256,
            "content_md": LF_BODY,
        }
        assert resp.headers["etag"] == f'"{expected.sha256}"'
        assert resp.headers["cache-control"] == "public, max-age=300"

    def test_exact_version_including_future(self) -> None:
        with TestClient(app=_app(_registry())) as client:
            resp = client.get(
                "/v1/legal/documents/terms", params={"version": "2999-01-01"}, headers=VEX
            )
        assert resp.status_code == 200
        assert resp.json()["version"] == "2999-01-01"

    @pytest.mark.parametrize(
        ("path", "params", "headers"),
        [
            ("/v1/legal/documents/cookies", None, VEX),
            ("/v1/legal/documents/terms", {"version": "2021-05-05"}, VEX),
            ("/v1/legal/documents/terms", None, SYNTHARA),
        ],
        ids=["unknown-type", "unknown-version", "type-not-required-by-product"],
    )
    def test_not_found(
        self, path: str, params: dict[str, str] | None, headers: dict[str, str]
    ) -> None:
        with TestClient(app=_app(_registry())) as client:
            resp = client.get(path, params=params, headers=headers)
        assert resp.status_code == 404
        assert resp.json()["error"] == "legal_document_not_found"

    def test_current_lists_required_set_only(self) -> None:
        registry = _registry()
        with TestClient(app=_app(registry)) as client:
            vex = client.get("/v1/legal/current", headers=VEX)
            synthara = client.get("/v1/legal/current", headers=SYNTHARA)
        assert vex.status_code == 200
        docs = vex.json()["documents"]
        assert [d["doc_type"] for d in docs] == sorted(t.value for t in LegalDocumentType)
        assert all(d["version"] == "2020-01-01" for d in docs)
        assert synthara.json() == {"documents": []}

    def test_api_sha_matches_crlf_source_normalised(self, tmp_path: Any) -> None:
        """C4 — a CRLF file is served (and hashed) as LF."""
        root = tmp_path / "legal"
        for t in LegalDocumentType:
            path = root / "vex" / t.value / "2020-01-01.md"
            path.parent.mkdir(parents=True)
            path.write_bytes(CRLF_BODY.encode())
        (root / "manifest.toml").write_text(
            "[vex]\n"
            + "".join(
                f"{t.value} = [ {{ version = 2020-01-01, requires_reacceptance = true }} ]\n"
                for t in LegalDocumentType
            )
        )
        from src.core.product_registry import VEX_CONFIG

        registry = LegalDocumentRegistry.load(
            root, products=[VEX_CONFIG], environment="development", today=date(2030, 1, 1)
        )
        lf_sha = make_legal_document(LegalDocumentType.TERMS, content=LF_BODY).sha256
        with TestClient(app=_app(registry)) as client:
            body = client.get("/v1/legal/documents/terms", headers=VEX).json()
        assert body["sha256"] == lf_sha
        assert body["content_md"] == LF_BODY


def _acceptance_service(
    registry: LegalDocumentRegistry, latest: dict[LegalDocumentType, LegalAcceptance] | None = None
) -> tuple[LegalAcceptanceService, MagicMock]:
    repo = MagicMock(spec=LegalAcceptanceRepository)
    repo.latest_per_type = AsyncMock(return_value=latest or {})
    return LegalAcceptanceService(registry=registry, repository=repo), repo


class TestAuthenticatedEndpoints:
    def test_status_requires_auth(self) -> None:
        registry = make_legal_registry()
        service, _ = _acceptance_service(registry)
        with TestClient(app=_app(registry, legal_service=service)) as client:
            resp = client.get("/v1/legal/status", headers=VEX)
        assert resp.status_code == 401

    def test_status_reports_unsatisfied(self) -> None:
        registry = make_legal_registry()
        service, _ = _acceptance_service(registry)
        jwt = JWTService(JWTConfig(secret_key=TEST_SECRET))
        token, _ = jwt.create_access_token(uuid4(), product_id="vex")
        with TestClient(app=_app(registry, legal_service=service, jwt_service=jwt)) as client:
            resp = client.get(
                "/v1/legal/status", headers={**VEX, "Authorization": f"Bearer {token}"}
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["all_satisfied"] is False
        assert {d["doc_type"] for d in body["documents"]} == {t.value for t in LegalDocumentType}
        assert all(d["accepted_version"] is None for d in body["documents"])

    def test_acceptances_is_exempt_and_records_reaccept(self) -> None:
        registry = make_legal_registry()
        service, repo = _acceptance_service(registry)
        jwt = JWTService(JWTConfig(secret_key=TEST_SECRET))
        token, _ = jwt.create_access_token(uuid4(), product_id="vex")  # no lgl at all
        payload = {
            "accepted_documents": [
                {"doc_type": d.doc_type.value, "version": d.version.isoformat(), "sha256": d.sha256}
                for d in accept_all_current(registry)
            ]
        }
        with TestClient(app=_app(registry, legal_service=service, jwt_service=jwt)) as client:
            resp = client.post(
                "/v1/legal/acceptances",
                json=payload,
                headers={**VEX, "Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200, resp.text
        rows = repo.add_many.call_args.args[0]
        assert {r.source for r in rows} == {"reaccept"}
        assert {r.ip_address for r in rows} == {"203.0.113.9"}

    def test_acceptances_stale_version_is_409_with_current(self) -> None:
        registry = make_legal_registry()
        service, repo = _acceptance_service(registry)
        jwt = JWTService(JWTConfig(secret_key=TEST_SECRET))
        token, _ = jwt.create_access_token(uuid4(), product_id="vex")
        payload = {
            "accepted_documents": [
                {"doc_type": d.doc_type.value, "version": "2019-01-01", "sha256": d.sha256}
                for d in accept_all_current(registry)
            ]
        }
        with TestClient(app=_app(registry, legal_service=service, jwt_service=jwt)) as client:
            resp = client.post(
                "/v1/legal/acceptances",
                json=payload,
                headers={**VEX, "Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 409
        body = resp.json()
        assert body["error"] == "legal_version_stale"
        assert len(body["detail"]["current"]) == 3
        repo.add_many.assert_not_called()


class TestRegisterLegalErrors:
    """Envelope shapes for C6 — the DB-level 'no user row' half is integration."""

    def _post(self, auth_service: Any, accepted_documents: list[dict[str, str]]) -> Any:
        registry = make_legal_registry()
        with TestClient(app=_app(registry, auth_service=auth_service)) as client:
            return client.post(
                "/v1/auth/register",
                json={
                    "email": "new@example.com",
                    "password": "pw-123456",
                    "accepted_documents": accepted_documents,
                },
                headers=VEX,
            )

    def test_incomplete_is_422(self) -> None:
        auth_service = MagicMock(spec=AuthService)
        auth_service.register = AsyncMock(
            side_effect=LegalSubmissionIncompleteError(
                missing=[LegalDocumentType.PRIVACY], unexpected=[], duplicated=[]
            )
        )
        resp = self._post(auth_service, [])
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"] == "legal_acceptance_incomplete"
        assert body["detail"]["missing"] == ["privacy"]

    def test_stale_is_409(self) -> None:
        auth_service = MagicMock(spec=AuthService)
        current = [{"doc_type": "terms", "version": "2020-01-01", "sha256": "a" * 64}]
        auth_service.register = AsyncMock(side_effect=LegalVersionStaleError(current))
        resp = self._post(auth_service, [])
        assert resp.status_code == 409
        assert resp.json()["detail"] == {"current": current}

    def test_missing_accepted_documents_field_is_rejected(self) -> None:
        auth_service = MagicMock(spec=AuthService)
        registry = make_legal_registry()
        with TestClient(app=_app(registry, auth_service=auth_service)) as client:
            resp = client.post(
                "/v1/auth/register",
                json={"email": "new@example.com", "password": "pw-123456"},
                headers=VEX,
            )
        assert resp.status_code == 400
        auth_service.register.assert_not_called()

    def test_register_passes_documents_and_context(self) -> None:
        registry = make_legal_registry()
        auth_service = MagicMock(spec=AuthService)
        user = MagicMock()
        user.id = uuid4()
        tokens = MagicMock(
            access_token="a",
            refresh_token="r",
            expires_in=900,
            expires_at=datetime.now(UTC),
        )
        auth_service.register = AsyncMock(return_value=(user, tokens))
        docs = accept_all_current(registry)
        resp = self._post(
            auth_service,
            [
                {"doc_type": d.doc_type.value, "version": d.version.isoformat(), "sha256": d.sha256}
                for d in docs
            ],
        )
        assert resp.status_code == 201, resp.text
        kwargs = auth_service.register.call_args.kwargs
        assert list(kwargs["accepted_documents"]) == docs
        assert kwargs["context"] == RequestContext(ip_address="203.0.113.9", user_agent="ua")
