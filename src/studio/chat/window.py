"""The history window of a chat reply (P2c-5, P2c-21). Pure.

A chat has **no rolling summary**: a reply sees the last few messages and nothing older, and a long
conversation moves on to a new chat. The window is the new user message plus up to 19 earlier
``complete`` messages, newest first, **while** the running total of characters (the new message
included) stays within 16,000; then it is put back in chronological order with the new message last.
It stops at the first message that does not fit - never skipping ahead to an older, smaller one,
which would leave a hole in the middle of the conversation.

A failed or pending message has no text and is not history (the caller leaves it out), so two user
messages may follow each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from src.llm.port import LlmMessage
from src.studio.constraints import CHAT_WINDOW_BUDGET, CHAT_WINDOW_MESSAGES

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["select_window"]

type WireRole = Literal["user", "assistant"]


def select_window(
    history_newest_first: Iterable[tuple[WireRole, str]],
    new_message: str,
    *,
    max_messages: int = CHAT_WINDOW_MESSAGES,
    budget: int = CHAT_WINDOW_BUDGET,
) -> tuple[LlmMessage, ...]:
    """The conversation a reply is given, oldest first, the new user message last.

    Args:
        history_newest_first: The chat's earlier ``complete`` messages as ``(wire role, text)``,
            newest first. Read lazily, only as far as the window needs.
        new_message: The user message being answered. It is always included (a message is at most
            4,000 characters, far inside the budget).
        max_messages: Most messages in the window, the new one included.
        budget: Most characters in the window, the new one included.

    Returns:
        The window in chronological order.
    """
    taken: list[LlmMessage] = []
    total = len(new_message)
    for role, text in history_newest_first:
        if len(taken) + 1 >= max_messages or total + len(text) > budget:
            break
        taken.append(LlmMessage(role, text))
        total += len(text)
    taken.reverse()
    taken.append(LlmMessage("user", new_message))
    return tuple(taken)
