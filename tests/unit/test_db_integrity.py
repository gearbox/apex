"""Regression coverage for database-integrity translation helpers."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from src.db.integrity import violated_constraint

pytestmark = pytest.mark.unit

_EMAIL = "ix_users_email_product"
_IDENTITY = "uq_user_identities_product_provider_subject"


class _DriverError(Exception):
    def __init__(self, message: str, constraint_name: str | None = None) -> None:
        super().__init__(message)
        self.constraint_name = constraint_name


def _integrity_error(original: BaseException) -> IntegrityError:
    return IntegrityError("INSERT", {}, original)


def test_r2_d_finds_constraint_name_on_nested_cause() -> None:
    """R2-d — asyncpg's nested driver exception exposes the constraint name."""
    outer = _DriverError("adapter error")
    outer.__cause__ = _DriverError("unique violation", _EMAIL)

    assert violated_constraint(_integrity_error(outer), frozenset({_EMAIL})) == _EMAIL


def test_r2_d_matches_only_quoted_candidate_names() -> None:
    """R2-d — message fallback is exact and constrained to caller-provided names."""
    quoted = _integrity_error(_DriverError(f'duplicate key violates unique constraint "{_EMAIL}"'))
    unquoted = _integrity_error(_DriverError(f"duplicate key violates {_EMAIL}"))

    assert violated_constraint(quoted, frozenset({_EMAIL})) == _EMAIL
    assert violated_constraint(quoted, frozenset({_IDENTITY})) is None
    assert violated_constraint(unquoted, frozenset({_EMAIL})) is None
