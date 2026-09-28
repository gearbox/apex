"""External identity-provider links (OAuth/OIDC subjects) for users."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Runtime import required: SQLAlchemy resolves Mapped[] annotations at runtime.
from src.core.product import OAuthProvider  # noqa: TC001
from src.db.models.base import Base

if TYPE_CHECKING:
    from src.db.models.user import User


class UserIdentity(Base):
    """One user's link to one provider subject (e.g. a Google ``sub``).

    Deliberately stores no email or profile data — ``users.email`` is the only
    copy (minimal-traceability stance). ``provider`` is a plain ``String``,
    not a PG enum, so adding a provider never needs ``ALTER TYPE``.

    Uniqueness:
      - ``(product_id, provider, subject)`` — one subject maps to one account
        per product (accounts are product-scoped, so the same Google account
        may sign up on each product separately).
      - ``(user_id, provider)`` — a user links at most one subject per provider.

    Self-closure (``DELETE /v1/users/me``) deletes a user's identities in the
    same transaction so the subject can sign up again; admin deactivation
    keeps them (the account may be reactivated).
    """

    __tablename__ = "user_identities"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    provider: Mapped[OAuthProvider] = mapped_column(String(32), nullable=False)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship("User", back_populates="identities")

    __table_args__ = (
        UniqueConstraint(
            "product_id", "provider", "subject", name="uq_user_identities_product_provider_subject"
        ),
        UniqueConstraint("user_id", "provider", name="uq_user_identities_user_provider"),
    )

    def __repr__(self) -> str:
        return f"<UserIdentity {self.id} user={self.user_id} provider={self.provider}>"
