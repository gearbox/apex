"""Integration tests for locale-aware email rendering via Jinja2."""

from __future__ import annotations

from src.api.services.email.base import EmailMessage
from src.api.services.email.log import LogEmailService


async def test_verification_email_uses_locale_and_app_name() -> None:
    """send_verification_email renders content in the specified locale with app_name."""
    service = LogEmailService()
    sent_messages: list[EmailMessage] = []

    original_send = service.send

    async def capture_send(message: EmailMessage) -> None:
        sent_messages.append(message)
        await original_send(message)

    service.send = capture_send  # type: ignore[assignment]

    await service.send_verification_email(
        to="user@example.com",
        display_name="Тест",
        verification_url="https://app.example.com/verify?token=abc",
        locale="ru",
        app_name="MyBrand",
    )

    assert len(sent_messages) == 1
    msg = sent_messages[0]
    assert "MyBrand" in msg.subject
    assert "Подтвердите" in msg.subject or "Подтвердить" in msg.html_body
    assert "MyBrand" in msg.text_body
    assert "Apex" not in msg.text_body  # no hardcoded name


async def test_password_reset_email_uses_locale_and_app_name() -> None:
    """send_password_reset_email renders content in the specified locale with app_name."""
    service = LogEmailService()
    sent_messages: list[EmailMessage] = []

    original_send = service.send

    async def capture_send(message: EmailMessage) -> None:
        sent_messages.append(message)
        await original_send(message)

    service.send = capture_send  # type: ignore[assignment]

    await service.send_password_reset_email(
        to="user@example.com",
        display_name="Test",
        reset_url="https://app.example.com/reset?token=xyz",
        locale="sr",
        app_name="MyBrand",
    )

    assert len(sent_messages) == 1
    msg = sent_messages[0]
    assert "MyBrand" in msg.subject
    assert "Resetuj" in msg.subject or "Resetuj" in msg.html_body
    assert "MyBrand" in msg.text_body
