"""HTML escaping for Telegram ``parse_mode="HTML"`` messages.

Telegram's HTML mode only requires escaping ``&``, ``<``, ``>`` (see
https://core.telegram.org/bots/api#html-style). Every interpolated value in
a message must pass through here — IDs/enums are safe today, but escape
anyway so a future field addition can't accidentally become an injection.
"""

from __future__ import annotations


def escape(value: str) -> str:
    """Escape a value for interpolation into a Telegram HTML-mode message."""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
