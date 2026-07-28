"""Unit tests for TelegramLinkPoller: /start token confirmation flow."""

from __future__ import annotations

from http import HTTPStatus
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.api.services.telegram.sender import (
    TelegramSendError,
    TelegramUpdate,
    _TelegramChat,
    _TelegramMessage,
)
from src.workers.telegram_link_poller import _CONFLICT_BACKOFF_SECONDS, TelegramLinkPoller


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
    """Unknown token from an unlinked chat (no link row at all) still gets the invalid reply."""
    sender = AsyncMock()
    poller = _make_poller(sender)
    poller._session_factory = _FakeSession  # type: ignore[method-assign]

    with patch("src.workers.telegram_link_poller.AdminNotificationRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo.confirm_link_by_token = AsyncMock(return_value=None)
        mock_repo.get_link_by_chat_id = AsyncMock(return_value=None)
        mock_repo_cls.return_value = mock_repo

        await poller._handle_update(_update(1, "/start expired-token", chat_id=555))

    mock_repo.get_link_by_chat_id.assert_awaited_once_with(555)
    sender.send_message.assert_awaited_once()
    _, kwargs = sender.send_message.call_args
    assert "invalid" in kwargs["text"].lower() or "expired" in kwargs["text"].lower()


async def test_replayed_consumed_token_from_already_linked_chat_sends_no_reply() -> None:
    """F4: a restart-replayed, already-consumed /start from a linked chat must not
    tell the admin their token is invalid — they successfully linked minutes ago."""
    sender = AsyncMock()
    poller = _make_poller(sender)
    poller._session_factory = _FakeSession  # type: ignore[method-assign]

    existing_link = AsyncMock()

    with patch("src.workers.telegram_link_poller.AdminNotificationRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo.confirm_link_by_token = AsyncMock(return_value=None)
        mock_repo.get_link_by_chat_id = AsyncMock(return_value=existing_link)
        mock_repo_cls.return_value = mock_repo

        await poller._handle_update(_update(1, "/start already-consumed", chat_id=555))

    mock_repo.get_link_by_chat_id.assert_awaited_once_with(555)
    sender.send_message.assert_not_awaited()


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


async def test_confirm_failure_is_logged_and_no_reply_sent() -> None:
    sender = AsyncMock()
    poller = _make_poller(sender)
    poller._session_factory = _FakeSession  # type: ignore[method-assign]

    with patch("src.workers.telegram_link_poller.AdminNotificationRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo.confirm_link_by_token = AsyncMock(side_effect=RuntimeError("db gone"))
        mock_repo_cls.return_value = mock_repo

        await poller._handle_update(_update(1, "/start tok", chat_id=555))  # must not raise

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


async def test_conflict_does_not_propagate_and_backs_off() -> None:
    sender = AsyncMock()
    sender.get_updates = AsyncMock(
        side_effect=TelegramSendError("conflict", status_code=HTTPStatus.CONFLICT)
    )
    poller = _make_poller(sender)

    with patch.object(poller, "_interruptible_sleep", new=AsyncMock()) as mock_sleep:
        await poller.run_once()  # must not raise

    mock_sleep.assert_awaited_once_with(_CONFLICT_BACKOFF_SECONDS)
    assert poller._consecutive_conflicts == 1


async def test_non_conflict_telegram_error_propagates() -> None:
    sender = AsyncMock()
    sender.get_updates = AsyncMock(
        side_effect=TelegramSendError("unauthorized", status_code=HTTPStatus.UNAUTHORIZED)
    )
    poller = _make_poller(sender)

    with pytest.raises(TelegramSendError):
        await poller.run_once()


async def test_conflict_does_not_reset_next_offset() -> None:
    sender = AsyncMock()
    sender.get_updates = AsyncMock(
        side_effect=TelegramSendError("conflict", status_code=HTTPStatus.CONFLICT)
    )
    poller = _make_poller(sender)
    poller._next_offset = 42

    with patch.object(poller, "_interruptible_sleep", new=AsyncMock()):
        await poller.run_once()

    assert poller._next_offset == 42


async def test_conflict_logged_once_per_relog_interval_with_consecutive_count() -> None:
    sender = AsyncMock()
    sender.get_updates = AsyncMock(
        side_effect=TelegramSendError("conflict", status_code=HTTPStatus.CONFLICT)
    )
    poller = _make_poller(sender)

    times = iter([1000.0, 1010.0, 1400.0])  # 2nd call within the 300s relog window, 3rd past it

    with (
        patch.object(poller, "_interruptible_sleep", new=AsyncMock()),
        patch(
            "src.workers.telegram_link_poller.time.monotonic",
            side_effect=lambda: next(times),
        ),
        patch("src.workers.telegram_link_poller.logger") as mock_logger,
    ):
        await poller.run_once()
        await poller.run_once()
        await poller.run_once()

    assert mock_logger.error.call_count == 2  # 1st occurrence + 1st past the relog interval
    first_call, second_call = mock_logger.error.call_args_list
    assert first_call.kwargs["consecutive"] == 1
    assert second_call.kwargs["consecutive"] == 3


async def test_conflict_cleared_logged_on_recovery() -> None:
    sender = AsyncMock()
    sender.get_updates = AsyncMock(
        side_effect=[TelegramSendError("conflict", status_code=HTTPStatus.CONFLICT), []]
    )
    poller = _make_poller(sender)

    with (
        patch.object(poller, "_interruptible_sleep", new=AsyncMock()),
        patch("src.workers.telegram_link_poller.logger") as mock_logger,
    ):
        await poller.run_once()
        await poller.run_once()

    mock_logger.info.assert_any_call("telegram.getupdates_conflict_cleared", consecutive=1)
    assert poller._consecutive_conflicts == 0
