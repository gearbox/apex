"""Moderation detection — protocol-based, pluggable per provider."""

from __future__ import annotations

import dataclasses
from typing import Protocol


@dataclasses.dataclass(frozen=True)
class ModerationResult:
    """Result of moderation classification."""

    is_moderated: bool  # True = content policy violation (policy-driven charge/refund)
    is_provider_error: bool  # True = infra/network failure → always refund
    reason: str | None = None


class ModerationDetector(Protocol):
    """Protocol for provider-specific moderation detection."""

    def classify(
        self,
        provider_response: dict | None,
        exception: Exception | None,
    ) -> ModerationResult: ...


class GrokModerationDetector:
    """Moderation detector for Grok API responses.

    For images: response dict contains 'respect_moderation' bool.
    For videos: polling result's video sub-object contains respect_moderation.

    respect_moderation = True  → passed, no billing issue
    respect_moderation = False → content moderated (policy-driven: charge or refund)
    exception raised           → provider/infra error → always refund
    """

    def classify(
        self,
        provider_response: dict | None,
        exception: Exception | None,
    ) -> ModerationResult:
        if exception is not None:
            return ModerationResult(
                is_moderated=False,
                is_provider_error=True,
                reason=str(exception),
            )
        if provider_response is not None:
            respect = provider_response.get("respect_moderation", True)
            if not respect:
                return ModerationResult(
                    is_moderated=True,
                    is_provider_error=False,
                    reason="content_moderated",
                )
        return ModerationResult(is_moderated=False, is_provider_error=False)


class ComfyUIModerationDetector:
    """Moderation detector for ComfyUI.

    No content moderation yet for ComfyUI.
    All failures treated as provider errors → always refund.
    """

    def classify(
        self,
        provider_response: dict | None,  # noqa: ARG002
        exception: Exception | None,
    ) -> ModerationResult:
        if exception is not None:
            return ModerationResult(
                is_moderated=False,
                is_provider_error=True,
                reason=str(exception),
            )
        return ModerationResult(is_moderated=False, is_provider_error=False)


MODERATION_DETECTORS: dict[str, ModerationDetector] = {
    "grok": GrokModerationDetector(),
    "comfyui": ComfyUIModerationDetector(),
}
