"""Billing and payment error types."""


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


class PaymentVerificationError(Exception):
    """Webhook signature verification failed. → HTTP 400"""


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
