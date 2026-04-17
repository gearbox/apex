"""Vast.ai API response schemas."""

from __future__ import annotations

import msgspec


class VastAIOffer(msgspec.Struct, forbid_unknown_fields=False):
    """A GPU rental offer from Vast.ai search results."""

    id: int
    gpu_name: str
    num_gpus: int
    gpu_ram: int
    disk_space: float
    dph_total: float
    inet_up: float
    inet_down: float
    cuda_max_good: float
    verified: bool
    geolocation: str | None = None

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
    status_msg: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    public_ipaddr: str | None = None
    ports: dict[str, object] | None = None
    cur_state: str | None = None
    dph_total: float | None = None


class SearchOffersResponse(msgspec.Struct, forbid_unknown_fields=False):
    """Vast.ai search offers API response envelope."""

    offers: list[VastAIOffer]


class CreateInstanceResponse(msgspec.Struct, forbid_unknown_fields=False):
    """Vast.ai create instance API response envelope."""

    new_contract: int
    success: bool
