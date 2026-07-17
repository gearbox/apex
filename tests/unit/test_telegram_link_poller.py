"""Unit tests for TelegramLinkPoller: /start token confirmation flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

from src.api.services.telegram.sender import TelegramUpdate, _TelegramChat, _TelegramMessage
from src.workers.telegram_link_poller import TelegramLinkPoller


def _update(update_id: int, text: str | None, chat_id: int = 100) -> TelegramUpdate:
    message = (
        _TelegramMessage(chat=_TelegramChat(id=chat_id), text=text) if text is not None else None
    )
    return TelegramUpdate(update_id=update_id, message=message)


def _make_poller(sender: object) -> TelegramLinkPoller:
    return TelegramLinkPoller(
        sender=sender,  # type: ignore[arg-type]
        session_factory=AsyncMock(),  # type: ignore[arg-type]
        poll_timeout_seconds=25,
        redis_enabled=False,
    )


class _FakeSession:
    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def commit(self) -> None:
        return None


async def test_start_command_confirms_token_and_replies_success() -> None:
    sender = AsyncMock()
    poller = _make_poller(sender)
    poller._session_factory = _FakeSession  # type: ignore[method-assign]

    fake_link = AsyncMock()
    fake_link.user_id = uuid4()

    with patch("src.workers.telegram_link_poller.AdminNotificationRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo.confirm_link_by_token = AsyncMock(return_value=fake_link)
        mock_repo_cls.return_value = mock_repo

        await poller._handle_update(_update(1, "/start abc123", chat_id=555))

    mock_repo.confirm_link_by_token.assert_awaited_once()
    args, _ = mock_repo.confirm_link_by_token.call_args
    assert args[0] == "abc123"
    assert args[1] == 555
    sender.send_message.assert_awaited_once()
    _, kwargs = sender.send_message.call_args
    assert kwargs["chat_id"] == 555
    assert "linked" in kwargs["text"].lower()


async def test_start_command_with_invalid_token_replies_invalid() -> None:
    sender = AsyncMock()
    poller = _make_poller(sender)
    poller._session_factory = _FakeSession  # type: ignore[method-assign]

    with patch("src.workers.telegram_link_poller.AdminNotificationRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo.confirm_link_by_token = AsyncMock(return_value=None)
        mock_repo_cls.return_value = mock_repo

        await poller._handle_update(_update(1, "/start expired-token", chat_id=555))

    _, kwargs = sender.send_message.call_args
    assert "invalid" in kwargs["text"].lower() or "expired" in kwargs["text"].lower()


async def test_non_start_message_is_ignored_without_reply() -> None:
    sender = AsyncMock()
    poller = _make_poller(sender)

    await poller._handle_update(_update(1, "hello there"))

    sender.send_message.assert_not_awaited()


async def test_message_without_text_is_ignored() -> None:
    sender = AsyncMock()
    poller = _make_poller(sender)

    await poller._handle_update(_update(1, None))

    sender.send_message.assert_not_awaited()


async def test_run_once_advances_offset_and_processes_updates() -> None:
    sender = AsyncMock()
    sender.get_updates = AsyncMock(return_value=[_update(5, "hello"), _update(6, "/start tok")])
    poller = _make_poller(sender)
    poller._session_factory = _FakeSession  # type: ignore[method-assign]

    with patch("src.workers.telegram_link_poller.AdminNotificationRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo.confirm_link_by_token = AsyncMock(return_value=None)
        mock_repo_cls.return_value = mock_repo

        await poller.run_once()

    assert poller._next_offset == 7
    sender.get_updates.assert_awaited_once_with(offset=None, timeout_seconds=25)
