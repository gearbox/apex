"""Unit tests for HttpxTelegramSender: the Telegram Bot API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.api.services.telegram.sender import HttpxTelegramSender, TelegramSendError


def _make_sender() -> HttpxTelegramSender:
    return HttpxTelegramSender(bot_token="123:abc", send_timeout_seconds=5.0)


def _response(
    *, status_code: int = 200, json_body: object | None = None, is_error: bool = False
) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.is_error = is_error
    response.json = MagicMock(return_value=json_body if json_body is not None else {"ok": True})
    return response


class TestSendMessage:
    async def test_success_posts_expected_payload(self) -> None:
        sender = _make_sender()
        sender._client.post = AsyncMock(return_value=_response())  # type: ignore[method-assign]

        await sender.send_message(chat_id=42, text="hello")

        sender._client.post.assert_awaited_once()
        args, kwargs = sender._client.post.call_args
        assert args[0].endswith("/sendMessage")
        assert kwargs["json"] == {
            "chat_id": 42,
            "text": "hello",
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

    async def test_network_error_raises_telegram_send_error(self) -> None:
        sender = _make_sender()
        sender._client.post = AsyncMock(  # type: ignore[method-assign]
            side_effect=httpx.ConnectError("dns failed")
        )

        with pytest.raises(TelegramSendError, match="Telegram request failed"):
            await sender.send_message(chat_id=42, text="hello")

    async def test_ok_false_response_raises_telegram_send_error(self) -> None:
        sender = _make_sender()
        sender._client.post = AsyncMock(  # type: ignore[method-assign]
            return_value=_response(json_body={"ok": False, "description": "bot was blocked"})
        )

        with pytest.raises(TelegramSendError, match="bot was blocked"):
            await sender.send_message(chat_id=42, text="hello")

    async def test_non_json_response_raises_telegram_send_error(self) -> None:
        sender = _make_sender()
        response = _response()
        response.json = MagicMock(side_effect=ValueError("not json"))
        sender._client.post = AsyncMock(return_value=response)  # type: ignore[method-assign]

        with pytest.raises(TelegramSendError, match="non-JSON response"):
            await sender.send_message(chat_id=42, text="hello")

    async def test_rate_limited_then_retry_succeeds(self) -> None:
        sender = _make_sender()
        rate_limited = _response(
            status_code=429, json_body={"parameters": {"retry_after": 1}}, is_error=True
        )
        success = _response()
        sender._client.post = AsyncMock(side_effect=[rate_limited, success])  # type: ignore[method-assign]

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            await sender.send_message(chat_id=42, text="hello")

        assert sender._client.post.await_count == 2
        mock_sleep.assert_awaited_once_with(1.0)

    async def test_rate_limited_retry_failure_raises(self) -> None:
        sender = _make_sender()
        rate_limited = _response(
            status_code=429, json_body={"parameters": {"retry_after": 1}}, is_error=True
        )
        sender._client.post = AsyncMock(  # type: ignore[method-assign]
            side_effect=[rate_limited, httpx.ConnectError("dns failed")]
        )

        with (
            patch("asyncio.sleep", new=AsyncMock()),
            pytest.raises(TelegramSendError, match="Telegram retry request failed"),
        ):
            await sender.send_message(chat_id=42, text="hello")

    async def test_rate_limited_retry_after_defaults_to_max_when_body_not_json(self) -> None:
        sender = _make_sender()
        rate_limited = _response(status_code=429, is_error=True)
        rate_limited.json = MagicMock(side_effect=ValueError("not json"))
        success = _response()
        sender._client.post = AsyncMock(side_effect=[rate_limited, success])  # type: ignore[method-assign]

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            await sender.send_message(chat_id=42, text="hello")

        mock_sleep.assert_awaited_once_with(30.0)

    async def test_rate_limited_retry_after_is_capped_at_max(self) -> None:
        sender = _make_sender()
        rate_limited = _response(
            status_code=429, json_body={"parameters": {"retry_after": 999}}, is_error=True
        )
        success = _response()
        sender._client.post = AsyncMock(side_effect=[rate_limited, success])  # type: ignore[method-assign]

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            await sender.send_message(chat_id=42, text="hello")

        mock_sleep.assert_awaited_once_with(30.0)


class TestGetMe:
    async def test_success_returns_username(self) -> None:
        sender = _make_sender()
        sender._client.get = AsyncMock(  # type: ignore[method-assign]
            return_value=_response(json_body={"ok": True, "result": {"username": "my_bot"}})
        )

        username = await sender.get_me()

        assert username == "my_bot"

    async def test_network_error_raises_telegram_send_error(self) -> None:
        sender = _make_sender()
        sender._client.get = AsyncMock(side_effect=httpx.ConnectError("dns failed"))  # type: ignore[method-assign]

        with pytest.raises(TelegramSendError, match="getMe request failed"):
            await sender.get_me()

    async def test_missing_username_raises_telegram_send_error(self) -> None:
        sender = _make_sender()
        sender._client.get = AsyncMock(  # type: ignore[method-assign]
            return_value=_response(json_body={"ok": True, "result": {}})
        )

        with pytest.raises(TelegramSendError, match="missing username"):
            await sender.get_me()


class TestGetUpdates:
    async def test_success_decodes_updates(self) -> None:
        sender = _make_sender()
        sender._client.get = AsyncMock(  # type: ignore[method-assign]
            return_value=_response(
                json_body={
                    "ok": True,
                    "result": [
                        {"update_id": 1, "message": {"chat": {"id": 100}, "text": "/start tok"}},
                        {"update_id": 2},
                    ],
                }
            )
        )

        updates = await sender.get_updates(offset=None, timeout_seconds=25)

        assert len(updates) == 2
        assert updates[0].update_id == 1
        assert updates[0].message is not None
        assert updates[0].message.text == "/start tok"
        assert updates[1].message is None

    async def test_offset_included_only_when_provided(self) -> None:
        sender = _make_sender()
        sender._client.get = AsyncMock(  # type: ignore[method-assign]
            return_value=_response(json_body={"ok": True, "result": []})
        )

        await sender.get_updates(offset=None, timeout_seconds=25)
        _, kwargs = sender._client.get.call_args
        assert "offset" not in kwargs["params"]

        await sender.get_updates(offset=7, timeout_seconds=25)
        _, kwargs = sender._client.get.call_args
        assert kwargs["params"]["offset"] == 7

    async def test_network_error_raises_telegram_send_error(self) -> None:
        sender = _make_sender()
        sender._client.get = AsyncMock(side_effect=httpx.ConnectError("dns failed"))  # type: ignore[method-assign]

        with pytest.raises(TelegramSendError, match="getUpdates request failed"):
            await sender.get_updates(offset=None, timeout_seconds=25)

    async def test_non_ok_response_raises_telegram_send_error(self) -> None:
        sender = _make_sender()
        sender._client.get = AsyncMock(  # type: ignore[method-assign]
            return_value=_response(status_code=401, json_body={"ok": False}, is_error=True)
        )

        with pytest.raises(TelegramSendError, match="getUpdates failed"):
            await sender.get_updates(offset=None, timeout_seconds=25)


class TestAclose:
    async def test_closes_underlying_client(self) -> None:
        sender = _make_sender()
        sender._client.aclose = AsyncMock()  # type: ignore[method-assign]

        await sender.aclose()

        sender._client.aclose.assert_awaited_once()


class TestConstruction:
    def test_base_url_embeds_bot_token(self) -> None:
        sender = HttpxTelegramSender(bot_token="123:abc", send_timeout_seconds=5.0)
        assert sender._base_url == "https://api.telegram.org/bot123:abc"
