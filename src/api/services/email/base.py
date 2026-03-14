"""Abstract base class for email service implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.api.services.email.i18n import get_subject, render_template


@dataclass(frozen=True, slots=True)
class EmailMessage:
    """Immutable value object representing an outgoing email.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        html_body: HTML email body.
        text_body: Plain-text fallback (strongly recommended for deliverability).
        from_address: Sender address override (uses settings default when None).
        from_name: Sender display name override.
        reply_to: Reply-to address (optional).
        tags: Provider-specific tags for tracking/analytics (optional).
    """

    to: str
    subject: str
    html_body: str
    text_body: str
    from_address: str | None = None
    from_name: str | None = None
    reply_to: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


class EmailService(ABC):
    """Abstract base class for transactional email providers.

    All implementations must be safe to call concurrently from async code.
    Concrete providers:
      - ``LogEmailService``    — logs email to stdout (dev/test)
      - ``ResendEmailService`` — Resend API (production)
    """

    @abstractmethod
    async def send(self, message: EmailMessage) -> None:
        """Send a single transactional email.

        Args:
            message: The email to send.

        Raises:
            EmailDeliveryError: If delivery fails after retries.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience factory methods — keeps template logic off the routes
    # ------------------------------------------------------------------

    async def send_verification_email(
        self,
        *,
        to: str,
        display_name: str | None,
        verification_url: str,
        expires_hours: int = 24,
        locale: str = "en",
        app_name: str = "Apex",
    ) -> None:
        """Send an email verification message.

        Args:
            to: Recipient email.
            display_name: User's display name (falls back to email prefix).
            verification_url: The full URL the user must visit to verify.
            expires_hours: How many hours until the link expires.
            locale: ISO 639-1 locale code for template selection.
            app_name: Public-facing product name for branding.
        """
        name = display_name or to.split("@")[0]
        context: dict[str, str | int] = {
            "display_name": name,
            "verification_url": verification_url,
            "expires_hours": expires_hours,
            "app_name": app_name,
        }
        html, text = render_template("verify_email", locale, context)
        await self.send(
            EmailMessage(
                to=to,
                subject=get_subject("verify_email", locale, app_name=app_name),
                html_body=html,
                text_body=text,
                tags={"type": "verification"},
            )
        )

    async def send_password_reset_email(
        self,
        *,
        to: str,
        display_name: str | None,
        reset_url: str,
        expires_minutes: int = 30,
        locale: str = "en",
        app_name: str = "Apex",
    ) -> None:
        """Send a password reset email.

        Args:
            to: Recipient email.
            display_name: User's display name.
            reset_url: The full URL the user must visit to reset their password.
            expires_minutes: How many minutes until the link expires.
            locale: ISO 639-1 locale code for template selection.
            app_name: Public-facing product name for branding.
        """
        name = display_name or to.split("@")[0]
        context: dict[str, str | int] = {
            "display_name": name,
            "reset_url": reset_url,
            "expires_minutes": expires_minutes,
            "app_name": app_name,
        }
        html, text = render_template("reset_password", locale, context)
        await self.send(
            EmailMessage(
                to=to,
                subject=get_subject("reset_password", locale, app_name=app_name),
                html_body=html,
                text_body=text,
                tags={"type": "password_reset"},
            )
        )


class EmailDeliveryError(Exception):
    """Raised when an email cannot be delivered."""

    def __init__(self, message: str, *, provider: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.cause = cause
