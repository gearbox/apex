"""HTTP schemas for the public provisioning endpoints (bootstrap script + failure webhook)."""

from __future__ import annotations

import msgspec


class ProvisionerFailureWebhookBody(msgspec.Struct, kw_only=True):
    """Vast.ai provisioner failure callback, POSTed after max_retries are exhausted.

    Deliberately does NOT set forbid_unknown_fields=True: this is a third-party
    (Vast.ai provisioner) wire contract that may grow fields apex doesn't use yet —
    msgspec's default of ignoring unknown fields is the correct tolerant-reader
    behavior here, not an oversight.
    """

    action: str
    manifest: str
    error: str
    container_id: str
    timestamp: str
