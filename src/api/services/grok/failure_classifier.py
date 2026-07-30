"""Centralized, safe classification of xAI Grok failures.

Structured provider codes win over message matching. Message matching exists
only as a backwards-compatible fallback for the production moderation payload
that xAI may return through different gRPC/SDK wrappers.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.api.services.generation.provider_failures import ProviderFailure, ProviderFailureKind
from src.core.enums import Provider

_MODERATION_CODE_MARKERS = ("MODERATION", "SAFETY", "CONTENT_POLICY", "POLICY_VIOLATION")
_INVALID_CODE_MARKERS = ("INVALID_ARGUMENT", "INVALID_REQUEST", "BAD_REQUEST")
_RATE_LIMIT_CODE_MARKERS = ("RESOURCE_EXHAUSTED", "RATE_LIMIT", "TOO_MANY_REQUESTS")
_AUTH_CODE_MARKERS = ("UNAUTHENTICATED", "UNAUTHORIZED", "PERMISSION_DENIED", "FORBIDDEN")
_UNAVAILABLE_CODE_MARKERS = ("UNAVAILABLE", "SERVICE_UNAVAILABLE", "INTERNAL", "UPSTREAM")
_TIMEOUT_CODE_MARKERS = ("DEADLINE_EXCEEDED", "TIMEOUT", "TIMED_OUT")
_MALFORMED_CODE_MARKERS = ("MALFORMED", "INVALID_RESPONSE", "PARSE_ERROR")


class GrokFailureClassifier:
    """Normalize Grok SDK, gRPC, and deferred-response errors."""

    def classify(
        self,
        source: object,
        *,
        provider_request_accepted: bool | None = None,
        provider_request_id: str | None = None,
    ) -> ProviderFailure:
        """Classify ``source`` without retaining raw provider payload data."""
        codes, messages, status_code = self._extract(source)
        kind = self._classify_kind(codes, messages, status_code)

        # A Grok moderation rejection is a definitive post-processing result.
        # The production billing policy therefore treats it as accepted unless
        # a caller supplied a more precise provider-acceptance value.
        if kind == ProviderFailureKind.MODERATION_REJECTED and provider_request_accepted is None:
            provider_request_accepted = True

        return ProviderFailure(
            kind=kind,
            provider=Provider.GROK,
            sanitized_message=ProviderFailure.safe_message_for_kind(kind),
            provider_status_code=status_code,
            provider_error_code=codes[0] if codes else None,
            retryable=kind
            in {
                ProviderFailureKind.RATE_LIMITED,
                ProviderFailureKind.PROVIDER_UNAVAILABLE,
                ProviderFailureKind.TIMEOUT,
            },
            provider_request_accepted=provider_request_accepted,
            provider_request_id=provider_request_id,
        )

    def _classify_kind(
        self,
        codes: list[str],
        messages: list[str],
        status_code: int | None,
    ) -> ProviderFailureKind:
        # Structured codes are authoritative and intentionally evaluated
        # before any free-text fallback.
        for code in codes:
            kind = self._kind_from_code(code)
            if kind is not None:
                return kind

        if status_code in (401, 403):
            return ProviderFailureKind.AUTHENTICATION_FAILED
        if status_code == 429:
            return ProviderFailureKind.RATE_LIMITED
        if status_code is not None and status_code >= 500:
            return ProviderFailureKind.PROVIDER_UNAVAILABLE
        if status_code is not None and 400 <= status_code < 500:
            return ProviderFailureKind.INVALID_REQUEST

        combined = " ".join(messages).casefold()
        normalized = " ".join(combined.replace(";", " ").replace(".", " ").split())
        if "respect moderation rules" in normalized and "url is not available" in normalized:
            return ProviderFailureKind.MODERATION_REJECTED
        if any(
            marker in normalized for marker in ("moderation", "safety system", "content policy")
        ):
            return ProviderFailureKind.MODERATION_REJECTED
        if any(marker in normalized for marker in ("deadline exceeded", "timed out", "timeout")):
            return ProviderFailureKind.TIMEOUT
        if any(marker in normalized for marker in ("rate limit", "resource exhausted")):
            return ProviderFailureKind.RATE_LIMITED
        if any(marker in normalized for marker in ("unauthenticated", "unauthorized", "forbidden")):
            return ProviderFailureKind.AUTHENTICATION_FAILED
        if any(marker in normalized for marker in ("invalid argument", "invalid request")):
            return ProviderFailureKind.INVALID_REQUEST
        if any(marker in normalized for marker in ("malformed", "parse error", "invalid response")):
            return ProviderFailureKind.MALFORMED_RESPONSE
        if any(marker in normalized for marker in ("unavailable", "connection", "upstream")):
            return ProviderFailureKind.PROVIDER_UNAVAILABLE
        return ProviderFailureKind.UNKNOWN

    @staticmethod
    def _kind_from_code(code: str) -> ProviderFailureKind | None:
        upper_code = code.upper()
        marker_sets: tuple[tuple[tuple[str, ...], ProviderFailureKind], ...] = (
            (_MODERATION_CODE_MARKERS, ProviderFailureKind.MODERATION_REJECTED),
            (_INVALID_CODE_MARKERS, ProviderFailureKind.INVALID_REQUEST),
            (_RATE_LIMIT_CODE_MARKERS, ProviderFailureKind.RATE_LIMITED),
            (_AUTH_CODE_MARKERS, ProviderFailureKind.AUTHENTICATION_FAILED),
            (_TIMEOUT_CODE_MARKERS, ProviderFailureKind.TIMEOUT),
            (_MALFORMED_CODE_MARKERS, ProviderFailureKind.MALFORMED_RESPONSE),
            (_UNAVAILABLE_CODE_MARKERS, ProviderFailureKind.PROVIDER_UNAVAILABLE),
        )
        return next(
            (
                kind
                for markers, kind in marker_sets
                if any(marker in upper_code for marker in markers)
            ),
            None,
        )

    def _extract(self, source: object) -> tuple[list[str], list[str], int | None]:
        codes: list[str] = []
        messages: list[str] = []
        statuses: list[int] = []
        self._walk(source, codes=codes, messages=messages, statuses=statuses, seen=set())
        return codes, messages, statuses[0] if statuses else None

    def _walk(
        self,
        value: object,
        *,
        codes: list[str],
        messages: list[str],
        statuses: list[int],
        seen: set[int],
    ) -> None:
        if value is None or len(seen) > 32:
            return
        if isinstance(value, str):
            messages.append(value)
            return
        if isinstance(value, int) and not isinstance(value, bool):
            if 100 <= value <= 599:
                statuses.append(value)
            return

        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if isinstance(value, Mapping):
            for key, nested in value.items():
                self._consume_field(
                    str(key),
                    nested,
                    codes=codes,
                    messages=messages,
                    statuses=statuses,
                    seen=seen,
                )
            return

        for field_name in (
            "error_code",
            "code",
            "grpc_code",
            "status_code",
            "status",
            "message",
            "detail",
            "details",
            "error",
            "response",
            "data",
        ):
            nested = getattr(value, field_name, None)
            if callable(nested):
                try:
                    nested = nested()
                except Exception:
                    nested = None
            if nested is not None:
                self._consume_field(
                    field_name,
                    nested,
                    codes=codes,
                    messages=messages,
                    statuses=statuses,
                    seen=seen,
                )

        args = getattr(value, "args", ())
        if isinstance(args, tuple):
            for arg in args:
                self._walk(arg, codes=codes, messages=messages, statuses=statuses, seen=seen)

    def _consume_field(
        self,
        field_name: str,
        value: object,
        *,
        codes: list[str],
        messages: list[str],
        statuses: list[int],
        seen: set[int],
    ) -> None:
        normalized_name = field_name.casefold()
        if normalized_name in {"error_code", "code", "grpc_code"}:
            if isinstance(value, str):
                codes.append(value)
                return
            name = getattr(value, "name", None)
            if isinstance(name, str):
                codes.append(name)
                return
            codes.append(str(value))
            return
        if normalized_name in {"status", "status_code"} and isinstance(value, int):
            if 100 <= value <= 599:
                statuses.append(value)
            return
        if normalized_name in {"message", "detail", "details"} and isinstance(value, str):
            messages.append(value)
            return
        self._walk(value, codes=codes, messages=messages, statuses=statuses, seen=seen)
