"""Billing and organization database models."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.models.storage import Base

if TYPE_CHECKING:
    from src.db.models.user import User


class Organization(Base):
    """Organization for enterprise billing."""

    __tablename__ = "organizations"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    owner_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    # Relationships
    owner: Mapped[User] = relationship("User", foreign_keys=[owner_id])
    members: Mapped[list[OrganizationMember]] = relationship(
        "OrganizationMember",
        back_populates="organization",
        cascade="all, delete-orphan",
    )
    token_account: Mapped[TokenAccount | None] = relationship(
        "TokenAccount",
        back_populates="organization",
        uselist=False,
    )

    def __repr__(self) -> str:
        return f"<Organization {self.id} name={self.name}>"


class OrganizationMember(Base):
    """Organization membership with role."""

    __tablename__ = "organization_members"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    organization_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    # Relationships
    organization: Mapped[Organization] = relationship(
        "Organization",
        back_populates="members",
    )
    user: Mapped[User] = relationship("User", foreign_keys=[user_id])

    __table_args__ = (UniqueConstraint("organization_id", "user_id", name="uq_org_member"),)

    def __repr__(self) -> str:
        return (
            f"<OrganizationMember org={self.organization_id} user={self.user_id} role={self.role}>"
        )


class TokenAccount(Base):
    """Token account for billing — personal or enterprise."""

    __tablename__ = "token_accounts"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    account_type: Mapped[str] = mapped_column(String(20), nullable=False)
    user_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=True,
    )
    organization_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        unique=True,
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    # Relationships
    user: Mapped[User | None] = relationship(
        "User",
        back_populates="token_account",
        foreign_keys=[user_id],
    )
    organization: Mapped[Organization | None] = relationship(
        "Organization",
        back_populates="token_account",
        foreign_keys=[organization_id],
    )
    transactions: Mapped[list[TokenTransaction]] = relationship(
        "TokenTransaction",
        back_populates="account",
        cascade="all, delete-orphan",
    )
    payments: Mapped[list[Payment]] = relationship(
        "Payment",
        back_populates="account",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            """
            (account_type = 'personal' AND user_id IS NOT NULL AND organization_id IS NULL)
            OR
            (account_type = 'enterprise' AND organization_id IS NOT NULL AND user_id IS NULL)
            """,
            name="chk_account_owner",
        ),
    )

    def __repr__(self) -> str:
        return f"<TokenAccount {self.id} type={self.account_type}>"


class TokenTransaction(Base):
    """Append-only immutable ledger for token transactions."""

    __tablename__ = "token_transactions"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    account_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("token_accounts.id"),
        nullable=False,
    )
    transaction_type: Mapped[str] = mapped_column(String(30), nullable=False)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    balance_after: Mapped[int] = mapped_column(BigInteger, nullable=False)
    job_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("generation_jobs.id"),
        nullable=True,
    )
    payment_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("payments.id"),
        nullable=True,
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    created_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True,
    )

    # Relationships
    account: Mapped[TokenAccount] = relationship(
        "TokenAccount",
        back_populates="transactions",
    )

    __table_args__ = (
        CheckConstraint("amount != 0", name="chk_amount_nonzero"),
        Index("ix_token_transactions_account_created", "account_id", "created_at"),
        Index("ix_token_transactions_job_id", "job_id"),
        Index("ix_token_transactions_payment_id", "payment_id"),
    )

    def __repr__(self) -> str:
        return f"<TokenTransaction {self.id} type={self.transaction_type} amount={self.amount}>"


class PricingRule(Base):
    """Pricing catalog entry for generation costs."""

    __tablename__ = "pricing_catalog"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    generation_type: Mapped[str] = mapped_column(String(20), nullable=False)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    token_cost: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    effective_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    effective_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        CheckConstraint("token_cost > 0", name="chk_token_cost_positive"),
        CheckConstraint(
            "effective_until IS NULL OR effective_until > effective_from",
            name="chk_effective_range",
        ),
        UniqueConstraint(
            "provider",
            "generation_type",
            "model",
            "effective_from",
            name="uq_pricing_rule",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<PricingRule {self.id} {self.provider}/{self.generation_type} cost={self.token_cost}>"
        )


class Payment(Base):
    """Payment record for token purchases."""

    __tablename__ = "payments"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    account_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("token_accounts.id"),
        nullable=False,
    )
    payment_provider: Mapped[str] = mapped_column(String(30), nullable=False)
    external_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    amount_usd: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        nullable=False,
    )
    tokens_granted: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        server_default=text("'USD'"),
    )
    provider_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_by: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )

    # Relationships
    account: Mapped[TokenAccount] = relationship(
        "TokenAccount",
        back_populates="payments",
    )

    __table_args__ = (
        CheckConstraint("amount_usd > 0", name="chk_amount_positive"),
        CheckConstraint("tokens_granted > 0", name="chk_tokens_positive"),
        Index("ix_payments_account_id", "account_id"),
        Index("ix_payments_status_created", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Payment {self.id} provider={self.payment_provider} "
            f"status={self.status} usd={self.amount_usd}>"
        )
