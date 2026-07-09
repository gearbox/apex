"""Tests for SQLAlchemy model metadata registration."""

from __future__ import annotations

from src.db.models import Base

EXPECTED_METADATA_TABLES = frozenset(
    {
        "admin_audit_log",
        "admin_permissions",
        "email_verification_tokens",
        "generation_jobs",
        "generation_models",
        "generation_outputs",
        "gpu_sessions",
        "health_snapshots",
        "idempotency_keys",
        "organization_members",
        "organizations",
        "password_reset_tokens",
        "payments",
        "payment_provider_state",
        "pricing_catalog",
        "push_subscriptions",
        "refresh_tokens",
        "token_accounts",
        "token_transactions",
        "user_images",
        "users",
    }
)


def test_critical_tables_are_registered_in_sqlalchemy_metadata() -> None:
    """Alembic autogenerate only sees model tables registered in Base.metadata."""
    missing_tables = EXPECTED_METADATA_TABLES - set(Base.metadata.tables)
    assert not missing_tables
