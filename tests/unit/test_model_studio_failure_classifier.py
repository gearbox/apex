"""C14: the Model Studio classifier table, decided on structured signals only."""

from __future__ import annotations

from typing import Any

import pytest

from src.api.services.generation.provider_failures import ProviderFailureKind
from src.api.services.model_studio.failure_classifier import ModelStudioFailureClassifier
from src.core.enums import Provider

pytestmark = pytest.mark.unit

K = ProviderFailureKind
_classifier = ModelStudioFailureClassifier()


@pytest.mark.parametrize(
    ("code", "status", "kind", "retryable"),
    [
        # moderation
        ("DataInspectionFailed", 400, K.MODERATION_REJECTED, False),
        ("data_inspection_failed", None, K.MODERATION_REJECTED, False),
        ("IPInfringementSuspect", None, K.MODERATION_REJECTED, False),
        ("CustomRoleBlocked", None, K.MODERATION_REJECTED, False),
        # invalid request
        ("InvalidParameter", 400, K.INVALID_REQUEST, False),
        ("InvalidParameter.DurationOutOfRange", 400, K.INVALID_REQUEST, False),
        ("InvalidFile.Format", 400, K.INVALID_REQUEST, False),
        ("InvalidImageFormat", 400, K.INVALID_REQUEST, False),
        ("InvalidURL", 400, K.INVALID_REQUEST, False),
        ("BadRequest.EmptyInput", 400, K.INVALID_REQUEST, False),
        ("invalid_request_error", 400, K.INVALID_REQUEST, False),
        ("invalid_request_error.foo", 400, K.INVALID_REQUEST, False),
        # rate limited
        ("Throttling", 429, K.RATE_LIMITED, True),
        ("Throttling.RateQuota", 429, K.RATE_LIMITED, True),
        ("LimitRequests", 429, K.RATE_LIMITED, True),
        ("limit_requests", 429, K.RATE_LIMITED, True),
        ("limit_burst_rate", 429, K.RATE_LIMITED, True),
        ("insufficient_quota", 429, K.RATE_LIMITED, True),
        (None, 429, K.RATE_LIMITED, True),
        # authentication
        ("InvalidApiKey", 401, K.AUTHENTICATION_FAILED, False),
        ("invalid_api_key", 401, K.AUTHENTICATION_FAILED, False),
        ("AccessDenied", 403, K.AUTHENTICATION_FAILED, False),
        ("AccessDenied.Unpurchased", 403, K.AUTHENTICATION_FAILED, False),
        ("Workspace.AccessDenied", 403, K.AUTHENTICATION_FAILED, False),
        ("Model.AccessDenied", 403, K.AUTHENTICATION_FAILED, False),
        ("App.AccessDenied", 403, K.AUTHENTICATION_FAILED, False),
        ("Endpoint.AccessDenied", 403, K.AUTHENTICATION_FAILED, False),
        ("NOT AUTHORIZED", 403, K.AUTHENTICATION_FAILED, False),
        (None, 401, K.AUTHENTICATION_FAILED, False),
        (None, 403, K.AUTHENTICATION_FAILED, False),
        # provider unavailable (billing/account states and server errors)
        ("Arrearage", 400, K.PROVIDER_UNAVAILABLE, True),
        ("AllocationQuota.FreeTierOnly", 403, K.PROVIDER_UNAVAILABLE, True),
        ("PrepaidBillOverdue", 400, K.PROVIDER_UNAVAILABLE, True),
        ("PostpaidBillOverdue", 400, K.PROVIDER_UNAVAILABLE, True),
        ("BudgetLimitExceeded", 400, K.PROVIDER_UNAVAILABLE, True),
        ("CommodityNotPurchased", 403, K.PROVIDER_UNAVAILABLE, True),
        ("isv.OUT_OF_SERVICE", 400, K.PROVIDER_UNAVAILABLE, True),
        ("ModelNotFound", 404, K.PROVIDER_UNAVAILABLE, True),
        ("model_not_found", 404, K.PROVIDER_UNAVAILABLE, True),
        ("InternalError", 500, K.PROVIDER_UNAVAILABLE, True),
        ("InternalError.Algo", 500, K.PROVIDER_UNAVAILABLE, True),
        (None, 500, K.PROVIDER_UNAVAILABLE, True),
        (None, 503, K.PROVIDER_UNAVAILABLE, True),
        # nothing recognized
        ("SomethingNew", 400, K.UNKNOWN, False),
        (None, 400, K.UNKNOWN, False),
        (None, None, K.UNKNOWN, False),
    ],
)
def test_c14_structured_signals_map_to_the_documented_kind(
    code: str | None, status: int | None, kind: ProviderFailureKind, retryable: bool
) -> None:
    failure = _classifier.classify(code=code, http_status=status)

    assert failure.kind is kind
    assert failure.retryable is retryable
    assert failure.provider is Provider.ALIBABA
    assert failure.provider_error_code == code
    assert failure.provider_status_code == status


@pytest.mark.parametrize(
    "code", ["Arrearage", "PrepaidBillOverdue", "BudgetLimitExceeded", "CommodityNotPurchased"]
)
def test_c14_billing_code_under_http_429_is_provider_unavailable_not_rate_limited(
    code: str,
) -> None:
    failure = _classifier.classify(code=code, http_status=429)

    assert failure.kind is K.PROVIDER_UNAVAILABLE


def test_c14_a_recognized_code_wins_over_the_http_status() -> None:
    assert _classifier.classify(code="DataInspectionFailed", http_status=500).kind is (
        K.MODERATION_REJECTED
    )


def test_c14_timeout_and_connection_failures_are_retryable() -> None:
    timeout = _classifier.timeout(provider_request_accepted=None)
    unavailable = _classifier.unavailable(provider_request_accepted=None)

    assert (timeout.kind, timeout.retryable) == (K.TIMEOUT, True)
    assert (unavailable.kind, unavailable.retryable) == (K.PROVIDER_UNAVAILABLE, True)


def test_c14_malformed_response_is_not_retryable() -> None:
    failure = _classifier.malformed(provider_request_accepted=True)

    assert (failure.kind, failure.retryable) == (K.MALFORMED_RESPONSE, False)


@pytest.mark.parametrize("accepted", [True, False, None])
def test_c14_acceptance_is_carried_through_unchanged(accepted: bool | None) -> None:
    failure = _classifier.classify(code="InvalidParameter", provider_request_accepted=accepted)

    assert failure.provider_request_accepted is accepted


def test_c14_free_text_is_never_inspected() -> None:
    """The classifier has no message parameter; prose cannot influence the outcome."""
    failure = _classifier.classify(code="SomethingUnlisted", http_status=None)

    assert failure.kind is K.UNKNOWN
    assert "inappropriate" not in failure.sanitized_message
    extra: dict[str, Any] = {"message": "Input contains inappropriate content"}
    with pytest.raises(TypeError):
        _classifier.classify(code="SomethingUnlisted", **extra)


def test_public_message_is_the_fixed_catalog_text() -> None:
    failure = _classifier.classify(code="DataInspectionFailed", provider_request_id="r-1")

    assert failure.sanitized_message == (
        "The requested content was rejected by the AI provider's safety system. "
        "Modify the prompt or input and try again."
    )
    assert failure.public_code == "provider_moderation_rejected"
    assert failure.provider_request_id == "r-1"
