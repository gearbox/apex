"""Shared email test doubles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.services.email import EmailService

if TYPE_CHECKING:
    from src.api.services.email.base import EmailMessage


class RecordingEmailService(EmailService):
    """Keeps every message the (real) template layer renders; nothing is sent."""

    def __init__(self) -> None:
        self.sent: list[EmailMessage] = []

    async def send(self, message: EmailMessage) -> None:
        self.sent.append(message)
