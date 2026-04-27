"""Unit tests for ResendEmailService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.api.services.email.base import EmailDeliveryError, EmailMessage
from src.api.services.email.resend import ResendEmailService

pytestmark = pytest.mark.unit


def _make_message(**kwargs: object) -> EmailMessage:
    return EmailMessage(
        to=str(kwargs.get("to", "user@example.com")),
        subject=str(kwargs.get("subject", "Test Subject")),
        html_body=str(kwargs.get("html_body", "<p>Hello</p>")),
        text_body=str(kwargs.get("text_body", "Hello")),
        from_address=kwargs.get("from_address", None),  # type: ignore[arg-type]
        from_name=kwargs.get("from_name", None),  # type: ignore[arg-type]
        reply_to=kwargs.get("reply_to", None),  # type: ignore[arg-type]
        tags=kwargs.get("tags", None),  # type: ignore[arg-type]
    )


@pytest.fixture
def svc() -> ResendEmailService:
    return ResendEmailService(
        api_key="re_test_key",
        from_address="noreply@example.com",
        from_name="Test",
    )


class TestInit:
    def test_raises_import_error_when_resend_missing(self) -> None:
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "resend":
                raise ImportError("no module named resend")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="resend"):
                ResendEmailService(api_key="k", from_address="a@b.com")


class TestSend:
    async def test_sends_basic_message(self, svc: ResendEmailService) -> None:
        msg = _make_message()
        mock_result = {"id": "email-123"}

        with patch("resend.Emails.send", return_value=mock_result):
            await svc.send(msg)  # should not raise

    async def test_uses_override_from_address(self, svc: ResendEmailService) -> None:
        msg = _make_message(from_address="custom@example.com", from_name="Custom")
        mock_result = {"id": "email-456"}

        with patch("resend.Emails.send", return_value=mock_result) as mock_send:
            await svc.send(msg)

        params = mock_send.call_args[0][0]
        assert "Custom <custom@example.com>" == params["from"]

    async def test_includes_reply_to_when_set(self, svc: ResendEmailService) -> None:
        msg = _make_message(reply_to="reply@example.com")

        with patch("resend.Emails.send", return_value={"id": "x"}) as mock_send:
            await svc.send(msg)

        params = mock_send.call_args[0][0]
        assert params["reply_to"] == "reply@example.com"

    async def test_includes_tags_as_list_of_dicts(self, svc: ResendEmailService) -> None:
        msg = _make_message(tags={"env": "test", "type": "welcome"})

        with patch("resend.Emails.send", return_value={"id": "x"}) as mock_send:
            await svc.send(msg)

        params = mock_send.call_args[0][0]
        assert {"name": "env", "value": "test"} in params["tags"]
        assert {"name": "type", "value": "welcome"} in params["tags"]

    async def test_raises_email_delivery_error_on_exception(self, svc: ResendEmailService) -> None:
        msg = _make_message()

        with patch("resend.Emails.send", side_effect=Exception("API error")):
            with pytest.raises(EmailDeliveryError, match="Resend delivery failed"):
                await svc.send(msg)

    async def test_omits_reply_to_when_not_set(self, svc: ResendEmailService) -> None:
        msg = _make_message()

        with patch("resend.Emails.send", return_value={"id": "x"}) as mock_send:
            await svc.send(msg)

        params = mock_send.call_args[0][0]
        assert "reply_to" not in params

    async def test_omits_tags_when_not_set(self, svc: ResendEmailService) -> None:
        msg = _make_message()

        with patch("resend.Emails.send", return_value={"id": "x"}) as mock_send:
            await svc.send(msg)

        params = mock_send.call_args[0][0]
        assert "tags" not in params
