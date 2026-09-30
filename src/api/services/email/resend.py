"""Resend email service for production transactional email.

Uses the official ``resend`` Python SDK which wraps the Resend REST API.
Resend is the recommended provider for new projects in 2025:
- Excellent deliverability out of the box
- Generous free tier (3 000 emails/month)
- Simple API surface — single ``await resend.Emails.send_async()`` call
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

    Sends through the SDK's native ``httpx``-backed async path so a slow Resend
    round-trip never blocks the event loop.

    Args:
        api_key: Resend API key (``re_...``).
        from_address: Default sender address, e.g. ``noreply@yourdomain.com``.
            Must be a verified domain in your Resend dashboard.
        send_timeout_seconds: Timeout for each Resend HTTP request.

    Raises:
        ImportError: If the ``resend`` package or its async (``httpx``) client
            is not installed.
    """

    def __init__(
        self,
        *,
        api_key: str,
        from_address: str,
        send_timeout_seconds: int = 10,
    ) -> None:
        try:
            import resend
            # validate at construction, not import time
        except ImportError as exc:
            raise ImportError(
                "The 'resend' package is required for ResendEmailService. "
                "Install it with: uv add resend"
            ) from exc
        try:
            from resend.http_client_httpx import HTTPXClient
        except ImportError as exc:
            # Fail loud: without an async client, send_async cannot work at all.
            raise ImportError(
                "ResendEmailService needs 'httpx' for the Resend async client. "
                "Install it with: uv add httpx"
            ) from exc

        # Process-global SDK state: the key and async client are module attributes
        # of ``resend``, so one Resend account per process (there is only one).
        resend.api_key = api_key
        resend.default_async_http_client = HTTPXClient(timeout=send_timeout_seconds)

        self._from_address = from_address

    async def send(self, message: EmailMessage) -> None:
        """Send an email via the Resend API without blocking the event loop.

        Args:
            message: Email to send.

        Raises:
            ValueError: If the message carries no ``from_name`` (the sender
                name is per-product; there is no global default).
            EmailDeliveryError: If Resend returns an error response.
        """
        if not message.from_name:
            raise ValueError("EmailMessage.from_name is required: the sender name is per-product")
        sender_address = message.from_address or self._from_address
        from_field = f"{message.from_name} <{sender_address}>"

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

        # Transactional-email logs must not hold user email addresses.
        recipient_domain = message.to.rsplit("@", 1)[-1]
        try:
            result = await resend.Emails.send_async(params)
            logger.info(
                "email.sent",
                recipient_domain=recipient_domain,
                subject=message.subject,
                resend_id=result.get("id"),
            )
        except Exception as exc:
            logger.exception(
                "email.send_failed",
                recipient_domain=recipient_domain,
                subject=message.subject,
                error=str(exc),
            )
            raise EmailDeliveryError(
                f"Resend delivery failed: {exc}",
                provider="resend",
                cause=exc,
            ) from exc
