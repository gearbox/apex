"""Limits of a stored media file that more than one bounded context has to agree on.

Pure module: no I/O, no framework or database imports. Media ingest refuses an audio file longer
than ``AUDIO_MAX_SECONDS``, and a studio timeline's times are bounded by the same cap (S3-3), so
the number is defined once, here: neither side can move it alone.
"""

from __future__ import annotations

from typing import Final

__all__ = ["AUDIO_MAX_MS", "AUDIO_MAX_SECONDS"]

AUDIO_MAX_SECONDS: Final = 900
"""The longest audio file accepted (15 minutes). A module constant, not a setting."""

AUDIO_MAX_MS: Final = AUDIO_MAX_SECONDS * 1000
"""The same cap in milliseconds: the upper bound of every time in a realized timeline."""
