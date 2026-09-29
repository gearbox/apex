"""Unit tests for ResendEmailService."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import resend
import structlog

from src.api.services.email.base import EmailDeliveryError, EmailMessage
from src.api.services.email.resend import ResendEmailService

pytestmark = pytest.mark.unit


def _make_message(**kwargs: object) -> EmailMessage:
    return EmailMessage(
        to=str(kwargs.get("to", "user@example.com")),
        subject=str(kwargs.get("subject", "Test Subject")),
        html_body=str(kwargs.get("html_body", "<p>Hello</p>")),
        text_body=str(kwargs.get("text_body", "Hello")),
        from_address=kwargs.get("from_address"),  # type: ignore[arg-type]
        from_name=kwargs.get("from_name"),  # type: ignore[arg-type]
        reply_to=kwargs.get("reply_to"),  # type: ignore[arg-type]
        tags=kwargs.get("tags"),  # type: ignore[arg-type]
    )


@pytest.fixture
def svc() -> ResendEmailService:
    return ResendEmailService(
        api_key="re_test_key",
        from_address="noreply@example.com",
        from_name="Test",
    )


@pytest.fixture(autouse=True)
def _sync_send_must_not_be_used() -> Any:
    """K8 — the blocking ``Emails.send`` must never be reached from ``send``."""
    with patch(
        "resend.Emails.send", side_effect=AssertionError("blocking Emails.send was called")
    ) as blocked:
        yield blocked


class TestInit:
    def test_raises_import_error_when_resend_missing(self) -> None:
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> object:
            if name == "resend":
                raise ImportError("no module named resend")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="resend"),
        ):
            ResendEmailService(api_key="k", from_address="a@b.com")

    def test_raises_when_no_async_client_available(self) -> None:
        """K8 — fail loud at construction if the httpx-backed client can't be imported."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> object:
            if name == "resend.http_client_httpx":
                raise ImportError("no module named httpx")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="httpx"),
        ):
            ResendEmailService(api_key="k", from_address="a@b.com")

    def test_configures_sdk_once_at_construction(self) -> None:
        ResendEmailService(api_key="re_abc", from_address="a@b.com", send_timeout_seconds=3.5)

        assert resend.api_key == "re_abc"
        client = resend.default_async_http_client
        assert client is not None
        assert client._timeout == 3.5  # type: ignore[attr-defined]


class TestSend:
    async def test_awaits_send_async(self, svc: ResendEmailService) -> None:
        msg = _make_message()

        with patch("resend.Emails.send_async", new=AsyncMock(return_value={"id": "e1"})) as sent:
            await svc.send(msg)

        sent.assert_awaited_once()

    async def test_uses_override_from_address(self, svc: ResendEmailService) -> None:
        msg = _make_message(from_address="custom@example.com", from_name="Custom")

        with patch("resend.Emails.send_async", new=AsyncMock(return_value={"id": "e2"})) as sent:
            await svc.send(msg)

        params = sent.call_args[0][0]
        assert params["from"] == "Custom <custom@example.com>"

    async def test_includes_reply_to_when_set(self, svc: ResendEmailService) -> None:
        msg = _make_message(reply_to="reply@example.com")

        with patch("resend.Emails.send_async", new=AsyncMock(return_value={"id": "x"})) as sent:
            await svc.send(msg)

        params = sent.call_args[0][0]
        assert params["reply_to"] == "reply@example.com"

    async def test_includes_tags_as_list_of_dicts(self, svc: ResendEmailService) -> None:
        msg = _make_message(tags={"env": "test", "type": "welcome"})

        with patch("resend.Emails.send_async", new=AsyncMock(return_value={"id": "x"})) as sent:
            await svc.send(msg)

        params = sent.call_args[0][0]
        assert {"name": "env", "value": "test"} in params["tags"]
        assert {"name": "type", "value": "welcome"} in params["tags"]

    async def test_raises_email_delivery_error_on_exception(self, svc: ResendEmailService) -> None:
        msg = _make_message()

        with (
            patch("resend.Emails.send_async", new=AsyncMock(side_effect=Exception("API error"))),
            pytest.raises(EmailDeliveryError, match="Resend delivery failed"),
        ):
            await svc.send(msg)

    async def test_omits_reply_to_when_not_set(self, svc: ResendEmailService) -> None:
        msg = _make_message()

        with patch("resend.Emails.send_async", new=AsyncMock(return_value={"id": "x"})) as sent:
            await svc.send(msg)

        params = sent.call_args[0][0]
        assert "reply_to" not in params

    async def test_omits_tags_when_not_set(self, svc: ResendEmailService) -> None:
        msg = _make_message()

        with patch("resend.Emails.send_async", new=AsyncMock(return_value={"id": "x"})) as sent:
            await svc.send(msg)

        params = sent.call_args[0][0]
        assert "tags" not in params


class TestLogging:
    async def test_logs_contain_no_email_address(self, svc: ResendEmailService) -> None:
        """K8 — success and failure logs carry the recipient's domain, never the address."""
        msg = _make_message(to="someone.private@customer.example")

        with structlog.testing.capture_logs() as logs:
            with patch("resend.Emails.send_async", new=AsyncMock(return_value={"id": "e1"})):
                await svc.send(msg)
            with (
                patch("resend.Emails.send_async", new=AsyncMock(side_effect=RuntimeError("boom"))),
                pytest.raises(EmailDeliveryError),
            ):
                await svc.send(msg)

        assert {e["event"] for e in logs} == {"email.sent", "email.send_failed"}
        for entry in logs:
            assert entry["recipient_domain"] == "customer.example"
            assert "someone.private" not in repr(entry)
            assert "@" not in repr(entry)


class TestConcurrency:
    async def test_send_does_not_block_the_event_loop(self, svc: ResendEmailService) -> None:
        """K9 — while a send is in flight, other tasks keep running."""

        async def slow_send(_params: object) -> dict[str, str]:
            await asyncio.sleep(0.2)
            return {"id": "slow"}

        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        with patch("resend.Emails.send_async", new=slow_send):
            ticker_task = asyncio.create_task(ticker())
            try:
                await svc.send(_make_message())
            finally:
                ticker_task.cancel()

        # A blocked loop would have let the ticker run ~0 times during the 0.2 s send.
        assert ticks >= 5
