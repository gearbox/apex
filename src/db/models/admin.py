"""Admin permission and audit models."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import Base


class AdminPermissionGrant(Base):
    """Granular permission granted to an admin user by a superadmin."""

    __tablename__ = "admin_permissions"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    permission: Mapped[str] = mapped_column(String(50), nullable=False)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False)
    granted_by: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id", "permission", "product_id", name="uq_admin_perm_user_perm_product"
        ),
        Index("ix_admin_perm_user_product", "user_id", "product_id"),
    )


class AdminAuditLog(Base):
    """Append-only audit log for admin role and permission changes.

    Covers: role grants/revokes, permission grants/revokes.
    Written by both API and CLI paths.
    """

    __tablename__ = "admin_audit_log"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    # Who performed the action
    actor_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )
    # Who was affected
    target_user_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True,
    )
    product_id: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="e.g. role.grant, role.revoke, permission.grant, permission.revoke",
    )
    detail: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Human-readable detail, e.g. 'role changed from user to admin'",
    )
    source: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment="'api' or 'cli'",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        Index("ix_audit_target_product", "target_user_id", "product_id"),
        Index("ix_audit_product_created_id", "product_id", "created_at", "id"),
    )
