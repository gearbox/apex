"""msgspec Structs for Cloudflare API responses."""

from __future__ import annotations

import msgspec


class CFApiResult(msgspec.Struct, kw_only=True, forbid_unknown_fields=False):
    """Base Cloudflare API response wrapper."""

    success: bool
    errors: list[dict[str, object]] = []
    messages: list[dict[str, object]] = []


class TunnelCreateData(msgspec.Struct, kw_only=True, forbid_unknown_fields=False):
    """Data portion of a tunnel create response."""

    id: str
    name: str
    token: str


class CreateTunnelResult(CFApiResult):
    """Response from POST /accounts/{id}/cfd_tunnel."""

    result: TunnelCreateData | None = None


class DNSRecordData(msgspec.Struct, kw_only=True, forbid_unknown_fields=False):
    """Data portion of a DNS record response."""

    id: str
    name: str
    type: str
    content: str


class CreateDNSResult(CFApiResult):
    """Response from POST /zones/{id}/dns_records."""

    result: DNSRecordData | None = None


class TunnelListEntry(msgspec.Struct, kw_only=True, forbid_unknown_fields=False):
    """Entry in a tunnel list response."""

    id: str
    name: str
    status: str


class TunnelListResult(CFApiResult):
    """Response from GET /accounts/{id}/cfd_tunnel."""

    result: list[TunnelListEntry] = []
