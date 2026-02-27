"""Email service package.

Provides a provider-agnostic email abstraction used for transactional emails
(verification, password reset, etc.).

Usage:
    - ``LogEmailService``   — logs to stdout (local dev / testing, zero deps)
    - ``ResendEmailService`` — production via Resend API (resend Python SDK)

Wire up via ``init_services()`` in ``dependencies/common.py`` depending on
``settings.email_provider``.
"""

from .base import EmailMessage, EmailService
from .log import LogEmailService
from .resend import ResendEmailService

__all__ = [
    "EmailMessage",
    "EmailService",
    "LogEmailService",
    "ResendEmailService",
]
