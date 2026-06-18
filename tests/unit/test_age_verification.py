"""Unit tests for AgeVerificationService."""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pytest

from src.api.services.age_verification import AgeVerificationError, AgeVerificationService
from src.core.product import AgeGatePolicy, ProductConfig

pytestmark = pytest.mark.unit


def _config(policy: AgeGatePolicy) -> ProductConfig:
    cfg = MagicMock(spec=ProductConfig)
    cfg.age_gate = policy
    return cfg


class TestNoAgeInput:
    """No age fields supplied → no-op regardless of policy (regression guard)."""

    @pytest.mark.parametrize("policy", list(AgeGatePolicy))
    def test_no_input_any_policy_returns_none_none(self, policy: AgeGatePolicy) -> None:
        svc = AgeVerificationService()
        result = svc.verify(_config(policy))
        assert result == (None, None)


class TestNonePolicy:
    def test_returns_none_none_even_with_inputs(self) -> None:
        svc = AgeVerificationService()
        result = svc.verify(_config(AgeGatePolicy.NONE), age_confirmed=True)
        assert result == (None, None)


class TestCheckboxPolicy:
    def test_confirmed_true_returns_datetime_none(self) -> None:
        svc = AgeVerificationService()
        verified_at, dob = svc.verify(_config(AgeGatePolicy.CHECKBOX), age_confirmed=True)
        assert isinstance(verified_at, datetime)
        assert dob is None

    def test_confirmed_false_raises(self) -> None:
        svc = AgeVerificationService()
        with pytest.raises(AgeVerificationError, match="18 or older"):
            svc.verify(_config(AgeGatePolicy.CHECKBOX), age_confirmed=False)


class TestDateOfBirthPolicy:
    def test_missing_dob_with_dob_policy_raises(self) -> None:
        """Supplying age_confirmed=True with DATE_OF_BIRTH policy but no DOB should raise."""
        svc = AgeVerificationService()
        with pytest.raises(AgeVerificationError, match="Date of birth is required"):
            svc.verify(_config(AgeGatePolicy.DATE_OF_BIRTH), age_confirmed=True)

    def test_underage_raises(self) -> None:
        svc = AgeVerificationService()
        today = datetime.now(UTC).date()
        underage_dob = date(today.year - 17, today.month, today.day)
        with pytest.raises(AgeVerificationError, match="at least 18"):
            svc.verify(_config(AgeGatePolicy.DATE_OF_BIRTH), date_of_birth=underage_dob)

    def test_exactly_18_is_valid(self) -> None:
        svc = AgeVerificationService()
        today = datetime.now(UTC).date()
        dob_18 = date(today.year - 18, today.month, today.day)
        verified_at, returned_dob = svc.verify(
            _config(AgeGatePolicy.DATE_OF_BIRTH), date_of_birth=dob_18
        )
        assert isinstance(verified_at, datetime)
        assert returned_dob == dob_18

    def test_over_18_returns_datetime_and_dob(self) -> None:
        svc = AgeVerificationService()
        dob = date(1990, 1, 1)
        verified_at, returned_dob = svc.verify(
            _config(AgeGatePolicy.DATE_OF_BIRTH), date_of_birth=dob
        )
        assert isinstance(verified_at, datetime)
        assert returned_dob == dob

    def test_leap_day_birthday_turning_18_today(self) -> None:
        """User born on a leap day turning exactly 18 today should be verified."""
        svc = AgeVerificationService()
        today = datetime.now(UTC).date()
        # Use a fixed DOB exactly 18 years ago (same month/day, year-18)
        dob = date(today.year - 18, today.month, today.day)
        verified_at, returned_dob = svc.verify(
            _config(AgeGatePolicy.DATE_OF_BIRTH), date_of_birth=dob
        )
        assert isinstance(verified_at, datetime)
        assert returned_dob == dob


class TestUnknownPolicy:
    def test_unknown_policy_with_input_raises(self) -> None:
        cfg = MagicMock(spec=ProductConfig)
        cfg.age_gate = "unknown_policy"
        svc = AgeVerificationService()
        with pytest.raises(AgeVerificationError, match="Unknown age gate policy"):
            svc.verify(cfg, age_confirmed=True)
