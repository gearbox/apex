"""Age verification service.

Validates age claims on demand (at profile-update time) and computes the
desired (age_verified_at, date_of_birth) values to persist. Stateless —
monotonic / write-once enforcement lives in UserService which owns the row.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from src.core.product import AgeGatePolicy, ProductConfig


class AgeVerificationError(Exception):
    """Raised when age verification fails."""


class AgeVerificationService:
    """Validates age claims according to the product's age-gate policy.

    Stateless — all state is passed in per call.
    """

    def verify(
        self,
        product_config: ProductConfig,
        *,
        age_confirmed: bool | None = None,
        date_of_birth: date | None = None,
    ) -> tuple[datetime | None, date | None]:
        """Validate an age claim and return values to persist.

        Args:
            product_config: Product configuration with age_gate policy.
            age_confirmed: Whether user checked the "I am 18+" checkbox.
                           Required for CHECKBOX policy when any age input is provided.
            date_of_birth: User's date of birth.
                           Required for DATE_OF_BIRTH policy when any age input is provided.

        Returns:
            Tuple of (age_verified_at, date_of_birth) to apply to the User record.
            (None, None) when no age input was supplied or the product has no age gate.

        Raises:
            AgeVerificationError: If the user does not meet age requirements.
        """
        # No age input at all — no-op regardless of policy.
        # Allows profile PATCHes that don't touch age fields to pass through.
        if age_confirmed is None and date_of_birth is None:
            return None, None

        policy = product_config.age_gate

        if policy == AgeGatePolicy.NONE:
            return None, None

        if policy == AgeGatePolicy.CHECKBOX:
            if not age_confirmed:
                raise AgeVerificationError(
                    "You must confirm you are 18 or older to use this platform"
                )
            return datetime.now(UTC), None

        if policy == AgeGatePolicy.DATE_OF_BIRTH:
            if date_of_birth is None:
                raise AgeVerificationError("Date of birth is required to verify your age")
            today = datetime.now(UTC).date()
            # Compare (month, day) tuples to handle leap-year boundaries correctly
            age = (
                today.year
                - date_of_birth.year
                - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))
            )
            if age < 18:
                raise AgeVerificationError("You must be at least 18 years old to use this platform")
            return datetime.now(UTC), date_of_birth

        # Unknown policy — fail safe (deny)
        raise AgeVerificationError(f"Unknown age gate policy: {policy}")
