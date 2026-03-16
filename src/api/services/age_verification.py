"""Age verification service.

Enforces per-product age gate policies during registration.

Currently supports:
- NONE: No age verification required (skip entirely).
- CHECKBOX: User confirms age via a boolean flag.
- DATE_OF_BIRTH: User provides DOB, validated server-side (age >= 18).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from src.core.product import AgeGatePolicy, ProductConfig


class AgeVerificationError(Exception):
    """Raised when age verification fails."""


class AgeVerificationService:
    """Service for age gate enforcement at registration time.

    Stateless — all state is passed in per call.
    """

    def verify(
        self,
        product_config: ProductConfig,
        *,
        age_confirmed: bool | None = None,
        date_of_birth: date | None = None,
    ) -> tuple[datetime | None, date | None]:
        """Verify age according to product policy.

        Args:
            product_config: Product configuration with age_gate policy.
            age_confirmed: Whether user checked the "I am 18+" checkbox.
                           Required for CHECKBOX policy.
            date_of_birth: User's date of birth.
                           Required for DATE_OF_BIRTH policy.

        Returns:
            Tuple of (age_verified_at, date_of_birth) to set on the User record.
            Both will be None if the product has no age gate.

        Raises:
            AgeVerificationError: If the user does not meet age requirements.
            ValueError: If required fields are missing for the policy.
        """
        policy = product_config.age_gate

        if policy == AgeGatePolicy.NONE:
            return None, None

        if policy == AgeGatePolicy.CHECKBOX:
            if age_confirmed is None:
                raise AgeVerificationError(
                    "You must confirm you are 18 or older to use this platform"
                )
            if not age_confirmed:
                raise AgeVerificationError(
                    "You must confirm you are 18 or older to use this platform"
                )
            return datetime.now(UTC), None

        if policy == AgeGatePolicy.DATE_OF_BIRTH:
            if date_of_birth is None:
                raise AgeVerificationError("Date of birth is required to verify your age")
            today = datetime.now(UTC).date()
            # Calculate age: compare (year, month, day) tuples to handle leap years
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
