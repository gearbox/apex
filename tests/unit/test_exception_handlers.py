"""Tests for exception handlers in src/api/app.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from litestar import Request
from litestar.exceptions import HTTPException
from structlog.testing import capture_logs

from src.api.app import (
    account_inactive_handler,
    account_not_found_handler,
    global_exception_handler,
    http_exception_handler,
    idempotency_conflict_handler,
    insufficient_balance_handler,
    moderation_error_handler,
    organization_balance_handler,
    organization_permission_handler,
    payment_verification_handler,
    price_not_found_handler,
    refund_not_eligible_handler,
)
from src.api.services.billing_errors import (
    AccountInactiveError,
    AccountNotFoundError,
    InsufficientBalanceError,
    ModerationError,
    OrganizationBalanceError,
    OrganizationPermissionError,
    PaymentVerificationError,
    PriceNotFoundError,
    RefundNotEligibleError,
)
from src.api.services.idempotency import IdempotencyConflictError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_request(path: str = "/v1/test", method: str = "POST") -> Request[Any, Any, Any]:
    """Build a minimal mock Request with .url.path and .method."""
    req = MagicMock(spec=Request)
    req.method = method
    req.url = MagicMock()
    req.url.path = path
    return req


# ---------------------------------------------------------------------------
# http_exception_handler
# ---------------------------------------------------------------------------


class TestHttpExceptionHandler:
    def test_4xx_returns_error_envelope(self) -> None:
        resp = self._call_handler_with_exception("Not found", 404)
        assert resp.status_code == 404
        assert resp.content.error == "not_found"  # type: ignore[union-attr]

    def test_4xx_logs_warning(self) -> None:
        req = _mock_request()
        exc = HTTPException(detail="Bad input", status_code=400)

        with capture_logs() as cap:
            http_exception_handler(req, exc)

        log = next(r for r in cap if r["event"] == "http.error")
        assert log["log_level"] == "warning"
        assert log["status_code"] == 400

    def test_5xx_logs_error_with_exc_info(self) -> None:
        req = _mock_request()
        exc = HTTPException(detail="Server error", status_code=500)

        with capture_logs() as cap:
            resp = http_exception_handler(req, exc)

        assert resp.status_code == 500
        log = next(r for r in cap if r["event"] == "http.error")
        assert log["log_level"] == "error"

    def test_unknown_status_code_uses_generic_error(self) -> None:
        resp = self._call_handler_with_exception("I'm a teapot", 418)
        assert resp.content.error == "error"  # type: ignore[union-attr]
        assert resp.status_code == 418

    def _call_handler_with_exception(self, detail, status_code):
        req = _mock_request()
        exc = HTTPException(detail=detail, status_code=status_code)
        with capture_logs():
            result = http_exception_handler(req, exc)
        return result


# ---------------------------------------------------------------------------
# Business logic exception handlers
# ---------------------------------------------------------------------------


class TestBusinessExceptionHandlers:
    def test_insufficient_balance_logs_and_returns_402(self) -> None:
        req = _mock_request()
        exc = InsufficientBalanceError(balance=10, required=100)

        with capture_logs() as cap:
            resp = insufficient_balance_handler(req, exc)

        assert resp.status_code == 402
        log = next(r for r in cap if r["event"] == "billing.insufficient_balance")
        assert log["balance"] == 10
        assert log["required"] == 100

    def test_account_not_found_logs_and_returns_404(self) -> None:
        req = _mock_request()
        exc = AccountNotFoundError("No account")

        with capture_logs() as cap:
            resp = account_not_found_handler(req, exc)

        assert resp.status_code == 404
        assert any(r["event"] == "billing.account_not_found" for r in cap)

    def test_account_inactive_logs_and_returns_403(self) -> None:
        req = _mock_request()
        exc = AccountInactiveError("Suspended")

        with capture_logs() as cap:
            resp = account_inactive_handler(req, exc)

        assert resp.status_code == 403
        assert any(r["event"] == "billing.account_inactive" for r in cap)

    def test_refund_not_eligible_logs_and_returns_409(self) -> None:
        req = _mock_request()
        exc = RefundNotEligibleError("Already refunded")

        with capture_logs() as cap:
            resp = refund_not_eligible_handler(req, exc)

        assert resp.status_code == 409
        assert any(r["event"] == "billing.refund_not_eligible" for r in cap)

    def test_price_not_found_logs_and_returns_404(self) -> None:
        req = _mock_request()
        exc = PriceNotFoundError("No rule")

        with capture_logs() as cap:
            resp = price_not_found_handler(req, exc)

        assert resp.status_code == 404
        assert any(r["event"] == "billing.price_not_found" for r in cap)

    def test_moderation_error_logs_with_provider_and_policy(self) -> None:
        req = _mock_request()
        exc = ModerationError(provider="grok", policy="safety")

        with capture_logs() as cap:
            resp = moderation_error_handler(req, exc)

        assert resp.status_code == 422
        log = next(r for r in cap if r["event"] == "moderation.rejected")
        assert log["provider"] == "grok"
        assert log["policy"] == "safety"

    def test_payment_verification_logs_and_returns_400(self) -> None:
        req = _mock_request()
        exc = PaymentVerificationError("Bad signature")

        with capture_logs() as cap:
            resp = payment_verification_handler(req, exc)

        assert resp.status_code == 400
        assert any(r["event"] == "payment.verification_failed" for r in cap)

    def test_organization_permission_logs_and_returns_403(self) -> None:
        req = _mock_request()
        exc = OrganizationPermissionError("Not admin")

        with capture_logs() as cap:
            resp = organization_permission_handler(req, exc)

        assert resp.status_code == 403
        assert any(r["event"] == "organization.permission_denied" for r in cap)

    def test_organization_balance_logs_with_balance(self) -> None:
        req = _mock_request()
        exc = OrganizationBalanceError(balance=500)

        with capture_logs() as cap:
            resp = organization_balance_handler(req, exc)

        assert resp.status_code == 409
        log = next(r for r in cap if r["event"] == "organization.balance_nonzero")
        assert log["balance"] == 500


# ---------------------------------------------------------------------------
# Idempotency conflict handler
# ---------------------------------------------------------------------------


class TestIdempotencyConflictHandler:
    def test_returns_409_with_retry_after(self) -> None:
        req = _mock_request()
        exc = IdempotencyConflictError("Key in flight")

        with capture_logs():
            resp = idempotency_conflict_handler(req, exc)

        assert resp.status_code == 409
        assert resp.headers.get("Retry-After") == "1"

    def test_logs_at_info_level(self) -> None:
        req = _mock_request()
        exc = IdempotencyConflictError("Key in flight")

        with capture_logs() as cap:
            idempotency_conflict_handler(req, exc)

        log = next(r for r in cap if r["event"] == "idempotency.conflict")
        assert log["log_level"] == "info"


# ---------------------------------------------------------------------------
# Global exception handler (catch-all)
# ---------------------------------------------------------------------------


class TestGlobalExceptionHandler:
    def test_returns_500_error_envelope(self) -> None:
        resp = self._call_global_handler_with_exception("something broke")
        assert resp.status_code == 500
        assert resp.content.error == "internal_error"  # type: ignore[union-attr]

    def test_does_not_leak_exception_details(self) -> None:
        resp = self._call_global_handler_with_exception("secret internal info")
        assert "secret" not in resp.content.message  # type: ignore[union-attr]
        assert resp.content.message == "An unexpected error occurred."  # type: ignore[union-attr]

    def _call_global_handler_with_exception(self, arg0):
        req = _mock_request()
        exc = RuntimeError(arg0)
        with capture_logs():
            result = global_exception_handler(req, exc)
        return result

    def test_logs_error_with_exc_type(self) -> None:
        req = _mock_request()
        exc = ValueError("bad value")

        with capture_logs() as cap:
            global_exception_handler(req, exc)

        log = next(r for r in cap if r["event"] == "unhandled_exception")
        assert log["log_level"] == "error"
        assert log["exc_type"] == "ValueError"
        assert log["path"] == "/v1/test"
        assert log["method"] == "POST"

    def test_handles_nested_exception_class(self) -> None:
        """Verify qualname works for nested/inner exception classes."""
        req = _mock_request()

        class CustomError(Exception):
            pass

        exc = CustomError("custom")

        with capture_logs() as cap:
            global_exception_handler(req, exc)

        log = next(r for r in cap if r["event"] == "unhandled_exception")
        assert "CustomError" in log["exc_type"]
