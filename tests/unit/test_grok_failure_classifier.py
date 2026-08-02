"""Focused tests for the centralized Grok provider-failure classifier."""

from __future__ import annotations

from typing import Literal

import pytest

from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
from src.api.services.generation.provider_failures import ProviderFailureKind
from src.api.services.grok.failure_classifier import GrokFailureClassifier

_CONFIRMED_MESSAGE = "Image did not respect moderation rules; URL is not available."


@pytest.fixture
def classifier() -> GrokFailureClassifier:
    return GrokFailureClassifier()


@pytest.mark.parametrize(
    "payload",
    [
        {"message": _CONFIRMED_MESSAGE},
        {"message": "  image DID NOT respect moderation rules; url IS not available.  "},
        {"message": f"xAI rejected the request: {_CONFIRMED_MESSAGE} Please try again."},
        {"error": {"detail": _CONFIRMED_MESSAGE}},
    ],
)
def test_classifies_confirmed_moderation_message(
    classifier: GrokFailureClassifier,
    payload: dict[str, object],
) -> None:
    failure = classifier.classify(payload)

    assert failure.kind == ProviderFailureKind.MODERATION_REJECTED
    assert failure.provider_request_accepted is True
    assert failure.public_code == "provider_moderation_rejected"


def test_unrelated_provider_error_is_not_moderation(classifier: GrokFailureClassifier) -> None:
    failure = classifier.classify({"error": {"code": "UPSTREAM_UNAVAILABLE", "message": "retry"}})

    assert failure.kind == ProviderFailureKind.PROVIDER_UNAVAILABLE


def test_structured_provider_code_takes_precedence_over_message(
    classifier: GrokFailureClassifier,
) -> None:
    failure = classifier.classify(
        {"error": {"code": "INVALID_REQUEST", "message": _CONFIRMED_MESSAGE}}
    )

    assert failure.kind == ProviderFailureKind.INVALID_REQUEST
    assert failure.provider_request_accepted is None


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("429 Too Many Requests", ProviderFailureKind.RATE_LIMITED),
        ("HTTP 429: request quota exceeded", ProviderFailureKind.RATE_LIMITED),
        ("invalid image url", ProviderFailureKind.INVALID_REQUEST),
        ("invalid image input", ProviderFailureKind.INVALID_REQUEST),
        ("400 Bad Request", ProviderFailureKind.INVALID_REQUEST),
    ],
)
def test_classifies_precise_xai_prose_fallbacks(
    classifier: GrokFailureClassifier,
    message: str,
    kind: ProviderFailureKind,
) -> None:
    assert classifier.classify({"message": message}).kind == kind


@pytest.mark.parametrize(
    "message",
    ["generate artwork", "rate card updated", "invalidate cache", "connection style"],
)
def test_prose_fallback_does_not_match_bare_substrings(
    classifier: GrokFailureClassifier,
    message: str,
) -> None:
    assert classifier.classify({"message": message}).kind == ProviderFailureKind.UNKNOWN


def test_structured_code_wins_over_conflicting_precise_prose(
    classifier: GrokFailureClassifier,
) -> None:
    failure = classifier.classify({"code": "UNAVAILABLE", "message": "invalid image url; HTTP 429"})

    assert failure.kind == ProviderFailureKind.PROVIDER_UNAVAILABLE


@pytest.mark.parametrize(
    "message",
    [
        "moderation service unavailable",
        "moderation endpoint timed out",
        "content policy service connection failed",
        "safety system internal error",
    ],
)
def test_moderation_service_failures_are_not_billable_rejections(
    classifier: GrokFailureClassifier,
    message: str,
) -> None:
    failure = classifier.classify({"message": message})

    assert failure.kind != ProviderFailureKind.MODERATION_REJECTED
    assert failure.provider_request_accepted is None


def test_sanitized_failure_never_exposes_provider_payload_or_secret(
    classifier: GrokFailureClassifier,
) -> None:
    secret = "xai-secret-should-never-reach-a-client"
    failure = classifier.classify({"message": f"{_CONFIRMED_MESSAGE} authorization={secret}"})

    assert secret not in failure.sanitized_message
    assert secret not in failure.public_code
    assert _CONFIRMED_MESSAGE not in failure.sanitized_message


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        ("MODERATION_SERVICE_UNAVAILABLE", ProviderFailureKind.PROVIDER_UNAVAILABLE),
        ("SAFETY_SERVICE_INTERNAL", ProviderFailureKind.PROVIDER_UNAVAILABLE),
        ("CONTENT_POLICY_TIMEOUT", ProviderFailureKind.TIMEOUT),
        ("MODERATION_RATE_LIMITED", ProviderFailureKind.RATE_LIMITED),
    ],
)
def test_compound_moderation_infrastructure_codes_are_non_billable(
    classifier: GrokFailureClassifier,
    code: str,
    kind: ProviderFailureKind,
) -> None:
    failure = classifier.classify({"code": code})

    assert failure.kind == kind
    assert failure.provider_request_accepted is None
    assert failure.billable is False


@pytest.mark.parametrize(
    ("setting", "expected_billable"),
    [("charge", True), ("refund", False)],
)
def test_grok_moderation_billing_policy_is_injected(
    classifier: GrokFailureClassifier,
    setting: Literal["charge", "refund"],
    expected_billable: bool,
) -> None:
    failure = classifier.classify({"code": "MODERATION_REJECTED"}, provider_request_accepted=True)
    registry = ProviderBillingPolicyRegistry.with_grok_moderation_policy(setting)

    assert registry.apply(failure).billable is expected_billable
