"""Resend email service for production transactional email.

Uses the official ``resend`` Python SDK which wraps the Resend REST API.
Resend is the recommended provider for new projects in 2025:
- Excellent deliverability out of the box
- Generous free tier (3 000 emails/month)
- Simple API surface — single ``resend.Emails.send()`` call
- Native DKIM/SPF management via Resend dashboard

Install: ``uv add resend``
Docs:    https://resend.com/docs/send-with-python
"""

from __future__ import annotations

import resend
import structlog

from .base import EmailDeliveryError, EmailMessage, EmailService

logger = structlog.get_logger(__name__)


class ResendEmailService(EmailService):
    """Transactional email via the Resend API.

    Args:
        api_key: Resend API key (``re_...``).
        from_address: Default sender address, e.g. ``noreply@yourdomain.com``.
            Must be a verified domain in your Resend dashboard.
        from_name: Default sender display name, e.g. ``Apex``.

    Raises:
        ImportError: If the ``resend`` package is not installed.
    """

    def __init__(
        self,
        *,
        api_key: str,
        from_address: str,
        from_name: str = "Apex",
    ) -> None:
        try:
            import resend  # noqa: F401
            # validate at construction, not import time
        except ImportError as exc:
            raise ImportError(
                "The 'resend' package is required for ResendEmailService. "
                "Install it with: uv add resend"
            ) from exc

        self._api_key = api_key
        self._from_address = from_address
        self._from_name = from_name

    async def send(self, message: EmailMessage) -> None:
        """Send an email via the Resend API.

        The Resend SDK is synchronous internally, so we call it directly
        here (it's a thin HTTP wrapper — blocking time is negligible).
        For high-throughput scenarios, wrap in ``asyncio.to_thread()``.

        Args:
            message: Email to send.

        Raises:
            EmailDeliveryError: If Resend returns an error response.
        """
        resend.api_key = self._api_key

        sender_address = message.from_address or self._from_address
        sender_name = message.from_name or self._from_name
        from_field = f"{sender_name} <{sender_address}>"

        params: resend.Emails.SendParams = {
            "from": from_field,
            "to": [message.to],
            "subject": message.subject,
            "html": message.html_body,
            "text": message.text_body,
        }

        if message.reply_to:
            params["reply_to"] = message.reply_to

        if message.tags:
            # Resend expects tags as list of {name, value} dicts
            params["tags"] = [{"name": k, "value": v} for k, v in message.tags.items()]

        try:
            result = resend.Emails.send(params)
            logger.info(
                "email.sent",
                to=message.to,
                subject=message.subject,
                resend_id=result.get("id"),
            )
        except Exception as exc:
            logger.exception(
                "email.send_failed",
                to=message.to,
                subject=message.subject,
                error=str(exc),
            )
            raise EmailDeliveryError(
                f"Resend delivery failed: {exc}",
                provider="resend",
                cause=exc,
            ) from exc
