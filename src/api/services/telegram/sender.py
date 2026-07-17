"""Telegram Bot API client — behind a Protocol so tests never touch the network.

Mirrors ``src.api.services.push``'s ``WebPushSender`` protocol + concrete
sender pattern: a thin ``Protocol`` describing what the dispatcher/poller
need, and one concrete ``httpx``-backed implementation.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol

import httpx
import msgspec
import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

_API_BASE = "https://api.telegram.org"
_MAX_RETRY_AFTER_SECONDS = 30.0


class TelegramSendError(Exception):
    """A Telegram Bot API call failed (network error, non-2xx, or ok=false)."""


class _TelegramChat(msgspec.Struct, kw_only=True):
    id: int


class _TelegramMessage(msgspec.Struct, kw_only=True):
    chat: _TelegramChat
    text: str | None = None


class TelegramUpdate(msgspec.Struct, kw_only=True):
    """Decodes only the fields TelegramLinkPoller needs from a getUpdates result."""

    update_id: int
    message: _TelegramMessage | None = None


class TelegramSender(Protocol):
    async def send_message(self, *, chat_id: int, text: str) -> None: ...

    async def get_me(self) -> str:
        """Returns the bot's username (without the leading '@')."""
        ...

    async def get_updates(
        self, *, offset: int | None, timeout_seconds: int
    ) -> list[TelegramUpdate]: ...


class HttpxTelegramSender:
    """``TelegramSender`` backed by ``httpx`` against the Telegram Bot API.

    The bot token is unwrapped from ``SecretStr`` only here, at construction
    (composition-root discipline) — no ``SecretStr`` value crosses into
    services/workers.
    """

    def __init__(self, *, bot_token: str, send_timeout_seconds: float) -> None:
        self._base_url = f"{_API_BASE}/bot{bot_token}"
        self._send_timeout_seconds = send_timeout_seconds
        self._client = httpx.AsyncClient(timeout=send_timeout_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def send_message(self, *, chat_id: int, text: str) -> None:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        await self._post_with_retry("/sendMessage", payload, chat_id=chat_id, text_len=len(text))

    async def get_me(self) -> str:
        try:
            response = await self._client.get(f"{self._base_url}/getMe")
        except httpx.HTTPError as exc:
            raise TelegramSendError("getMe request failed") from exc

        body = self._decode_ok(response, method="getMe")
        if username := body.get("result", {}).get("username"):
            return str(username)
        raise TelegramSendError("getMe response missing username")

    async def get_updates(
        self, *, offset: int | None, timeout_seconds: int
    ) -> list[TelegramUpdate]:
        params: dict[str, int] = {"timeout": timeout_seconds}
        if offset is not None:
            params["offset"] = offset

        # Telegram holds the connection open server-side for up to
        # `timeout_seconds` — the HTTP-level timeout must exceed that.
        request_timeout = timeout_seconds + 10.0
        try:
            response = await self._client.get(
                f"{self._base_url}/getUpdates",
                params=params,
                timeout=request_timeout,
            )
        except httpx.HTTPError as exc:
            raise TelegramSendError("getUpdates request failed") from exc

        body = self._decode_ok(response, method="getUpdates")
        raw_updates: Sequence[object] = body.get("result", [])
        return [msgspec.convert(u, type=TelegramUpdate) for u in raw_updates]

    async def _post_with_retry(
        self, path: str, payload: dict[str, object], *, chat_id: int, text_len: int
    ) -> None:
        try:
            response = await self._client.post(f"{self._base_url}{path}", json=payload)
        except httpx.HTTPError as exc:
            raise TelegramSendError("Telegram request failed") from exc

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            retry_after = self._extract_retry_after(response)
            logger.warning(
                "telegram.rate_limited",
                chat_id=chat_id,
                retry_after=retry_after,
            )
            await asyncio.sleep(min(retry_after, _MAX_RETRY_AFTER_SECONDS))
            try:
                response = await self._client.post(f"{self._base_url}{path}", json=payload)
            except httpx.HTTPError as exc:
                raise TelegramSendError("Telegram retry request failed") from exc

        self._decode_ok(response, method=path, chat_id=chat_id, text_len=text_len)

    @staticmethod
    def _extract_retry_after(response: httpx.Response) -> float:
        try:
            body = response.json()
        except ValueError:
            return _MAX_RETRY_AFTER_SECONDS
        retry_after = body.get("parameters", {}).get("retry_after")
        return float(retry_after) if retry_after is not None else _MAX_RETRY_AFTER_SECONDS

    @staticmethod
    def _decode_ok(
        response: httpx.Response, *, method: str, **log_fields: object
    ) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise TelegramSendError(f"{method}: non-JSON response") from exc

        if response.is_error or not body.get("ok", False):
            logger.warning(
                "telegram.api_error",
                method=method,
                status_code=response.status_code,
                **log_fields,
            )
            raise TelegramSendError(
                f"{method} failed: status={response.status_code} "
                f"description={body.get('description')}"
            )
        return dict(body)
