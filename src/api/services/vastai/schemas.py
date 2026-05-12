"""Vast.ai API response schemas."""

from __future__ import annotations

import msgspec


class VastAIOffer(msgspec.Struct, forbid_unknown_fields=False):
    """A GPU rental offer from Vast.ai search results.

    Only fields apex actively consumes are declared. Other fields Vast.ai
    returns are silently ignored via forbid_unknown_fields=False.

    This insulates apex from Vast.ai's frequent API field-shape changes —
    including the 2026-05 incident where apex declared 'verified: bool' but
    the live response has no 'verified' field at all (only 'verification: str').
    """

    id: int
    gpu_name: str
    dph_total: float
    num_gpus: int | None = None  # not read downstream; kept for diagnostics

    @property
    def dph_total_micros(self) -> int:
        """Cost in microdollars (1_000_000 = $1.00).

        Uses ``round()`` rather than ``int()`` to avoid off-by-one errors from
        IEEE-754 float representation. For example ``0.258607 * 1_000_000`` is
        ``258606.99999999997`` in float64, which ``int()`` truncates to
        ``258606``; ``round()`` produces the correct ``258607``.
        """
        return round(self.dph_total * 1_000_000)


class VastAIInstance(msgspec.Struct, forbid_unknown_fields=False):
    """A running or stopped Vast.ai instance."""

    id: int
    actual_status: str | None = None
    cur_state: str | None = None


class SearchOffersResponse(msgspec.Struct, forbid_unknown_fields=False):
    """Vast.ai search offers API response envelope."""

    offers: list[VastAIOffer]


class CreateInstanceResponse(msgspec.Struct, forbid_unknown_fields=False):
    """Vast.ai create instance API response envelope."""

    new_contract: int
    success: bool
