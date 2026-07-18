"""Billing and payment error types."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.product import PaymentProvider


class BillingError(Exception):
    """Base billing error."""


class InsufficientBalanceError(BillingError):
    """User does not have enough tokens. → HTTP 402 Payment Required"""

    def __init__(self, balance: int, required: int) -> None:
        self.balance = balance
        self.required = required
        super().__init__(f"Insufficient balance: have {balance}, need {required}")


class AccountNotFoundError(BillingError):
    """Token account not found. → HTTP 404"""


class AccountInactiveError(BillingError):
    """Token account is suspended/inactive. → HTTP 403"""


class RefundNotEligibleError(BillingError):
    """No debit found for job, or already refunded. → HTTP 409"""


class PriceNotFoundError(BillingError):
    """No active pricing rule for this provider/type/model. → HTTP 500"""


class ModerationError(Exception):
    """Generation was moderated by the provider. → HTTP 422"""

    def __init__(self, provider: str, policy: str) -> None:
        self.provider = provider
        self.policy = policy
        super().__init__(f"Content moderated by {provider} (policy: {policy})")


class PaymentVerificationReason(StrEnum):
    """Why a webhook/IPN failed verification — the alerting-stable field.

    Distinguishes signature/format failures (dashboard IPN-format toggle,
    secret mismatch) from payload-shape failures, so an incident doesn't
    require redeploying with extra logging to find out which check failed.
    """

    SIGNATURE_MISMATCH = "signature_mismatch"
    MISSING_SIGNATURE_HEADER = "missing_signature_header"
    MALFORMED_JSON = "malformed_json"
    MISSING_FIELD = "missing_field"
    MALFORMED_ORDER_ID = "malformed_order_id"
    PRODUCT_MISMATCH = "product_mismatch"
    AMOUNT_FIELDS_INVALID = "amount_fields_invalid"


class PaymentVerificationError(Exception):
    """Webhook signature verification failed. → HTTP 400

    ``context`` holds pre-sanitized diagnostic fields only (see D2 in
    ipn-verification-observability-prompt.md) — never the secret, full body,
    full signatures, amounts, or order_id contents.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: PaymentVerificationReason,
        context: dict[str, str] | None = None,
    ) -> None:
        self.reason = reason
        self.context = context or {}
        super().__init__(message)


class PaymentCatalogError(Exception):
    """A provider currency-catalog endpoint returned an unexpected shape or host."""


class LogoCacheError(Exception):
    """A currency logo failed to download, validate, or upload. Non-fatal to sync (D10)."""


class LogoStorageError(LogoCacheError):
    """A currency logo failed because the R2 assets bucket itself is unavailable.

    Distinct from a plain LogoCacheError (bad/oversized/wrong-type download) so the
    sync service can short-circuit remaining logo attempts for the run (P1-2) instead
    of retrying a dead storage backend once per ticker.
    """


class UnsupportedProviderError(BillingError):
    """No gateway implementation is registered for a provider."""

    def __init__(self, provider: PaymentProvider) -> None:
        self.provider = provider
        super().__init__(f"Unsupported payment provider: {provider.value}")


class UnknownProviderError(BillingError):
    """Provider is outside a product's statically configured capability set."""

    def __init__(self, provider: PaymentProvider) -> None:
        self.provider = provider
        super().__init__(f"Payment provider not available for this product: {provider.value}")


class PaymentProviderDisabledError(BillingError):
    """Provider is supported by a product but disabled at runtime. → HTTP 409"""

    def __init__(self, provider: PaymentProvider) -> None:
        self.provider = provider
        super().__init__(f"Payment provider is disabled: {provider.value}")


class TopUpAmountError(ValueError):
    """Top-up amount is outside configured payment bounds. → HTTP 400"""


class PayCurrencySuppressedError(BillingError):
    """Pinned pay_currency is admin-suppressed (provider-side zombie ticker). → HTTP 400"""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        super().__init__(f"Pay currency is suppressed: {ticker}")


class OrganizationPermissionError(Exception):
    """Insufficient org role for this action. → HTTP 403"""


class OrganizationBalanceError(BillingError):
    """Organization has a non-zero balance and cannot be deleted. → HTTP 409"""

    def __init__(self, balance: int) -> None:
        self.balance = balance
        super().__init__(
            f"Organization balance is not 0 (current: {balance}). "
            "Use force_delete=true to override."
        )
