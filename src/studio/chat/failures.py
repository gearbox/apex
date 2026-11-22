"""Why a chat reply or a memory draft ends ``failed`` - the codes stored in ``error_code``.

Every failure is refunded in full (P2c-12): there is no reply to charge for. A reply a client
merely stopped *listening* to is not a failure - it completes and is charged.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "ERR_CONTENT_POLICY",
    "ERR_DRAFT_INVALID",
    "ERR_LLM_UNAVAILABLE",
    "ERR_REPLY_EMPTY",
    "ERR_REPLY_TIMEOUT",
    "ERR_SERVER_SHUTDOWN",
]

ERR_LLM_UNAVAILABLE: Final = "llm_unavailable"
"""The provider could not be reached, kept failing, refused the request or cut the stream."""
ERR_CONTENT_POLICY: Final = "content_policy_refused"
"""The model declined (the sentinel, or the provider's own ``content_filter``)."""
ERR_REPLY_EMPTY: Final = "reply_empty"
"""The stream ended with no text."""
ERR_REPLY_TIMEOUT: Final = "reply_timeout"
"""The deadline passed - noticed by the task, or by the sweep for a task that never finished."""
ERR_SERVER_SHUTDOWN: Final = "server_shutdown"
"""The process stopped while the reply was being written."""
ERR_DRAFT_INVALID: Final = "memory_draft_invalid"
"""The model's draft was not a valid memory even after the one fix call."""
