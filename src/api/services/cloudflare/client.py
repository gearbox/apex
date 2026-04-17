"""Async Cloudflare Tunnel API client."""

from __future__ import annotations

import httpx
import msgspec
import structlog

from .exceptions import (
    CloudflareError,
    DNSRecordError,
    TunnelConfigError,
    TunnelCreationError,
    TunnelDeletionError,
)
from .schemas import CreateDNSResult, CreateTunnelResult, TunnelListEntry, TunnelListResult

logger = structlog.get_logger(__name__)


class CloudflareTunnelClient:
    """Async client for Cloudflare Tunnel API.

    Manages the full tunnel lifecycle for GPU sessions:
    - Create named tunnel → get tunnel token
    - Configure tunnel ingress (route hostname → localhost:port)
    - Create DNS CNAME record (hostname → tunnel UUID)
    - Delete tunnel and DNS record on cleanup
    """

    _CF_API_BASE = "https://api.cloudflare.com/client/v4"

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        api_token: str,
        account_id: str,
        zone_id: str,
        tunnel_domain: str,
    ) -> None:
        self._http = http_client
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        self._account_id = account_id
        self._zone_id = zone_id
        self._tunnel_domain = tunnel_domain

    async def create_tunnel(self, name: str) -> tuple[str, str]:
        """Create a named tunnel.

        Args:
            name: Tunnel name (e.g. "gpu-session-01jf8x3k").

        Returns:
            Tuple of (tunnel_id, tunnel_token).

        Raises:
            TunnelCreationError: If CF API returns an error.
        """
        url = f"{self._CF_API_BASE}/accounts/{self._account_id}/cfd_tunnel"
        payload = msgspec.json.encode({"name": name, "config_src": "cloudflare"})
        resp = await self._http.post(url, headers=self._headers, content=payload)

        if not resp.is_success:
            logger.error("cloudflare.tunnel.create_failed", status=resp.status_code, name=name)
            raise TunnelCreationError(
                f"Failed to create tunnel '{name}': HTTP {resp.status_code}",
                status_code=resp.status_code,
            )

        try:
            result = msgspec.json.decode(resp.content, type=CreateTunnelResult)
        except msgspec.DecodeError as exc:
            logger.error(
                "cloudflare.tunnel.create_decode_error",
                error=str(exc),
                status=resp.status_code,
                name=name,
            )
            raise TunnelCreationError(
                f"Failed to parse Cloudflare response while creating tunnel '{name}' "
                f"(HTTP {resp.status_code}): {exc}",
                status_code=resp.status_code,
            ) from exc

        if not result.success or result.result is None:
            logger.error("cloudflare.tunnel.create_api_error", errors=result.errors, name=name)
            raise TunnelCreationError(
                f"CF API error creating tunnel '{name}': {result.errors}",
            )

        logger.info("cloudflare.tunnel.created", tunnel_id=result.result.id, name=name)
        return result.result.id, result.result.token

    async def configure_tunnel_ingress(
        self, tunnel_id: str, hostname: str, local_port: int
    ) -> None:
        """Configure tunnel ingress rules (remote-managed).

        Args:
            tunnel_id: CF tunnel UUID.
            hostname: Public hostname (e.g. "01jf8x3k.gpu.cloudin.space").
            local_port: Local port to forward to (e.g. 18188 for ComfyUI).

        Raises:
            TunnelConfigError: If configuration fails.
        """
        url = (
            f"{self._CF_API_BASE}/accounts/{self._account_id}/cfd_tunnel/{tunnel_id}/configurations"
        )
        payload = msgspec.json.encode(
            {
                "config": {
                    "ingress": [
                        {"hostname": hostname, "service": f"http://localhost:{local_port}"},
                        {"service": "http_status:404"},
                    ]
                }
            }
        )
        resp = await self._http.put(url, headers=self._headers, content=payload)

        if not resp.is_success:
            logger.error(
                "cloudflare.tunnel.configure_failed",
                status=resp.status_code,
                tunnel_id=tunnel_id,
                hostname=hostname,
            )
            raise TunnelConfigError(
                f"Failed to configure tunnel '{tunnel_id}': HTTP {resp.status_code}",
                status_code=resp.status_code,
            )

        logger.info(
            "cloudflare.tunnel.configured",
            tunnel_id=tunnel_id,
            hostname=hostname,
            local_port=local_port,
        )

    async def create_dns_record(self, hostname: str, tunnel_id: str) -> str:
        """Create CNAME DNS record pointing hostname to tunnel.

        Args:
            hostname: Full hostname (e.g. "01jf8x3k.gpu.cloudin.space").
            tunnel_id: CF tunnel UUID.

        Returns:
            DNS record ID (for deletion later).

        Raises:
            DNSRecordError: If DNS creation fails.
        """
        url = f"{self._CF_API_BASE}/zones/{self._zone_id}/dns_records"
        payload = msgspec.json.encode(
            {
                "type": "CNAME",
                "name": hostname,
                "content": f"{tunnel_id}.cfargotunnel.com",
                "proxied": True,
            }
        )
        resp = await self._http.post(url, headers=self._headers, content=payload)

        if not resp.is_success:
            logger.error(
                "cloudflare.dns.create_failed",
                status=resp.status_code,
                hostname=hostname,
                tunnel_id=tunnel_id,
            )
            raise DNSRecordError(
                f"Failed to create DNS record for '{hostname}': HTTP {resp.status_code}",
                status_code=resp.status_code,
            )

        try:
            result = msgspec.json.decode(resp.content, type=CreateDNSResult)
        except msgspec.DecodeError as exc:
            logger.error(
                "cloudflare.dns.create_decode_error",
                error=str(exc),
                status=resp.status_code,
                hostname=hostname,
            )
            raise DNSRecordError(
                f"Failed to parse Cloudflare response while creating DNS record "
                f"for '{hostname}' (HTTP {resp.status_code}): {exc}",
                status_code=resp.status_code,
            ) from exc

        if not result.success or result.result is None:
            logger.error("cloudflare.dns.create_api_error", errors=result.errors, hostname=hostname)
            raise DNSRecordError(
                f"CF API error creating DNS record for '{hostname}': {result.errors}",
            )

        logger.info(
            "cloudflare.dns.created",
            dns_record_id=result.result.id,
            hostname=hostname,
        )
        return result.result.id

    async def delete_tunnel(self, tunnel_id: str) -> None:
        """Delete a tunnel. Idempotent — ignores 404.

        Args:
            tunnel_id: CF tunnel UUID.

        Raises:
            TunnelDeletionError: If deletion fails with a non-404 error.
        """
        url = f"{self._CF_API_BASE}/accounts/{self._account_id}/cfd_tunnel/{tunnel_id}"
        resp = await self._http.delete(url, headers=self._headers)

        if resp.status_code == 404:
            logger.info("cloudflare.tunnel.delete_already_gone", tunnel_id=tunnel_id)
            return

        if not resp.is_success:
            logger.error(
                "cloudflare.tunnel.delete_failed", status=resp.status_code, tunnel_id=tunnel_id
            )
            raise TunnelDeletionError(
                f"Failed to delete tunnel '{tunnel_id}': HTTP {resp.status_code}",
                status_code=resp.status_code,
            )

        logger.info("cloudflare.tunnel.deleted", tunnel_id=tunnel_id)

    async def delete_dns_record(self, dns_record_id: str) -> None:
        """Delete a DNS record. Idempotent — ignores 404.

        Args:
            dns_record_id: CF DNS record UUID.

        Raises:
            DNSRecordError: If deletion fails with a non-404 error.
        """
        url = f"{self._CF_API_BASE}/zones/{self._zone_id}/dns_records/{dns_record_id}"
        resp = await self._http.delete(url, headers=self._headers)

        if resp.status_code == 404:
            logger.info("cloudflare.dns.delete_already_gone", dns_record_id=dns_record_id)
            return

        if not resp.is_success:
            logger.error(
                "cloudflare.dns.delete_failed",
                status=resp.status_code,
                dns_record_id=dns_record_id,
            )
            raise DNSRecordError(
                f"Failed to delete DNS record '{dns_record_id}': HTTP {resp.status_code}",
                status_code=resp.status_code,
            )

        logger.info("cloudflare.dns.deleted", dns_record_id=dns_record_id)

    async def list_tunnels(self, name_prefix: str | None = None) -> list[TunnelListEntry]:
        """List tunnels, optionally filtered by name prefix.

        Used by orphan cleanup worker.

        Args:
            name_prefix: Optional prefix to filter tunnel names.

        Returns:
            List of tunnel entries.

        Raises:
            CloudflareError: If the list request fails.
        """
        url = f"{self._CF_API_BASE}/accounts/{self._account_id}/cfd_tunnel"
        params: dict[str, str] = {"is_deleted": "false"}
        if name_prefix is not None:
            params["name"] = name_prefix

        resp = await self._http.get(url, headers=self._headers, params=params)

        if not resp.is_success:
            logger.error("cloudflare.tunnel.list_failed", status=resp.status_code)
            raise CloudflareError(
                f"Failed to list tunnels: HTTP {resp.status_code}",
                status_code=resp.status_code,
            )

        try:
            result = msgspec.json.decode(resp.content, type=TunnelListResult)
        except msgspec.DecodeError as exc:
            logger.error(
                "cloudflare.tunnel.list_decode_error",
                error=str(exc),
                status=resp.status_code,
            )
            raise CloudflareError(
                f"Failed to parse Cloudflare response while listing tunnels "
                f"(HTTP {resp.status_code}): {exc}",
                status_code=resp.status_code,
            ) from exc

        if not result.success:
            logger.error(
                "cloudflare.tunnel.list_api_error",
                errors=result.errors,
                status=resp.status_code,
            )
            raise CloudflareError(
                f"CF API error listing tunnels: {result.errors}",
                status_code=resp.status_code,
            )

        return result.result

    async def create_session_tunnel(
        self, session_id_short: str, comfyui_port: int
    ) -> tuple[str, str, str, str]:
        """High-level: create tunnel + configure + DNS for a GPU session.

        Args:
            session_id_short: First 8 chars of session UUIDv7.
            comfyui_port: ComfyUI port (typically 18188).

        Returns:
            Tuple of (tunnel_id, tunnel_token, dns_record_id, hostname).
        """
        tunnel_name = f"gpu-session-{session_id_short}"
        hostname = f"{session_id_short}.{self._tunnel_domain}"
        tunnel_id, tunnel_token = await self.create_tunnel(tunnel_name)
        await self.configure_tunnel_ingress(tunnel_id, hostname, comfyui_port)
        dns_record_id = await self.create_dns_record(hostname, tunnel_id)
        return tunnel_id, tunnel_token, dns_record_id, hostname

    async def delete_session_tunnel(self, tunnel_id: str, dns_record_id: str) -> None:
        """High-level: delete tunnel + DNS for a GPU session.

        Idempotent — safe to call even if resources were partially created.

        Args:
            tunnel_id: CF tunnel UUID.
            dns_record_id: CF DNS record UUID.
        """
        await self.delete_dns_record(dns_record_id)
        await self.delete_tunnel(tunnel_id)
