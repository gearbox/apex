"""Log-based email service for development and testing.

Writes email content to the logger instead of actually sending it.
Zero external dependencies — use this in local dev and test environments.
"""

from __future__ import annotations

import structlog

from .base import EmailMessage, EmailService

logger = structlog.get_logger(__name__)


class LogEmailService(EmailService):
    """Email service that logs messages instead of sending them.

    Useful for:
    - Local development (see email content in console/logs)
    - Integration tests (no real emails sent)
    - Staging environments without email provider credentials

    The full email body is logged at DEBUG level so it doesn't flood
    INFO-level output, but the delivery event is logged at INFO.
    """

    async def send(self, message: EmailMessage) -> None:
        """Log the email message instead of sending it.

        Args:
            message: The email to log.
        """
        logger.info(
            "email_not_sent to=%s subject=%r provider=log",
            message.to,
            message.subject,
        )
        logger.debug(
            "email_content\n"
            "  to=%s\n"
            "  subject=%r\n"
            "  tags=%s\n"
            "  ---TEXT---\n%s\n"
            "  ---HTML (truncated)---\n%s",
            message.to,
            message.subject,
            message.tags,
            message.text_body,
            message.html_body[:500] + ("..." if len(message.html_body) > 500 else ""),
        )
