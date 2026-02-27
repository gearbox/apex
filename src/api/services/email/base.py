"""Abstract base class for email service implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
    ) -> None:
        """Send an email verification message.

        Args:
            to: Recipient email.
            display_name: User's display name (falls back to email prefix).
            verification_url: The full URL the user must visit to verify.
            expires_hours: How many hours until the link expires.
        """
        name = display_name or to.split("@")[0]
        await self.send(
            EmailMessage(
                to=to,
                subject="Verify your Apex account",
                text_body=_verification_text(name, verification_url, expires_hours),
                html_body=_verification_html(name, verification_url, expires_hours),
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
    ) -> None:
        """Send a password reset email.

        Args:
            to: Recipient email.
            display_name: User's display name.
            reset_url: The full URL the user must visit to reset their password.
            expires_minutes: How many minutes until the link expires.
        """
        name = display_name or to.split("@")[0]
        await self.send(
            EmailMessage(
                to=to,
                subject="Reset your Apex password",
                text_body=_reset_text(name, reset_url, expires_minutes),
                html_body=_reset_html(name, reset_url, expires_minutes),
                tags={"type": "password_reset"},
            )
        )


# ---------------------------------------------------------------------------
# Simple inline templates — good enough for MVP.
# Move to Jinja2 / React Email when branding matters.
# ---------------------------------------------------------------------------


def _verification_text(name: str, url: str, expires_hours: int) -> str:
    return (
        f"Hi {name},\n\n"
        f"Please verify your Apex account by visiting the link below:\n\n"
        f"{url}\n\n"
        f"This link expires in {expires_hours} hours.\n\n"
        f"If you did not create an account, you can safely ignore this email.\n\n"
        f"— The Apex Team"
    )


def _verification_html(name: str, url: str, expires_hours: int) -> str:
    return f"""<!DOCTYPE html>
<html>
<body style="font-family:sans-serif;max-width:600px;margin:0 auto;padding:24px;">
  <h2>Verify your Apex account</h2>
  <p>Hi {name},</p>
  <p>Click the button below to verify your email address.</p>
  <p style="margin:32px 0;">
    <a href="{url}"
       style="background:#6366f1;color:#fff;padding:12px 24px;
              border-radius:6px;text-decoration:none;font-weight:600;">
      Verify Email
    </a>
  </p>
  <p style="color:#6b7280;font-size:14px;">
    Or copy and paste this URL into your browser:<br/>
    <a href="{url}" style="color:#6366f1;">{url}</a>
  </p>
  <p style="color:#6b7280;font-size:14px;">
    This link expires in {expires_hours} hours.
    If you did not create an account, you can safely ignore this email.
  </p>
</body>
</html>"""


def _reset_text(name: str, url: str, expires_minutes: int) -> str:
    return (
        f"Hi {name},\n\n"
        f"We received a request to reset your Apex password.\n\n"
        f"Click the link below to choose a new password:\n\n"
        f"{url}\n\n"
        f"This link expires in {expires_minutes} minutes.\n\n"
        f"If you did not request a password reset, you can safely ignore this email.\n\n"
        f"— The Apex Team"
    )


def _reset_html(name: str, url: str, expires_minutes: int) -> str:
    return f"""<!DOCTYPE html>
<html>
<body style="font-family:sans-serif;max-width:600px;margin:0 auto;padding:24px;">
  <h2>Reset your password</h2>
  <p>Hi {name},</p>
  <p>We received a request to reset your Apex password.
     Click the button below to choose a new password.</p>
  <p style="margin:32px 0;">
    <a href="{url}"
       style="background:#6366f1;color:#fff;padding:12px 24px;
              border-radius:6px;text-decoration:none;font-weight:600;">
      Reset Password
    </a>
  </p>
  <p style="color:#6b7280;font-size:14px;">
    Or copy and paste this URL into your browser:<br/>
    <a href="{url}" style="color:#6366f1;">{url}</a>
  </p>
  <p style="color:#6b7280;font-size:14px;">
    This link expires in {expires_minutes} minutes.
    If you did not request a password reset, you can safely ignore this email.
  </p>
</body>
</html>"""


class EmailDeliveryError(Exception):
    """Raised when an email cannot be delivered."""

    def __init__(self, message: str, *, provider: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.cause = cause
