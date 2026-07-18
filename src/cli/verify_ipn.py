"""CLI tool for offline NowPayments IPN signature verification — D4 self-check.

Captures a real IPN body from the dashboard/tunnel, replays it through the
exact production verification path (``NowPaymentsGateway.verify_webhook``)
with no network or DB access, and prints the reason-coded result with the
same D2 diagnostics the API logs on a failed webhook. Turns "capture one
body and test offline" into a one-command workflow instead of
redeploy-and-guess.

Usage:
    python -m src.cli.verify_ipn --body capture.json --signature <sig> --product vex
    NOWPAYMENTS_IPN_SIGNATURE=<sig> python -m src.cli.verify_ipn --body capture.json --product vex

See docs/guides/nowpayments_ipn_debugging.md for how to capture a raw IPN
body and how to read each reason code.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console

from src.api.services.billing_errors import PaymentVerificationError
from src.api.services.payments.contracts import WebhookEnvelope
from src.api.services.payments.nowpayments_gateway import NowPaymentsGateway
from src.core.config import Settings

console = Console()

app = typer.Typer(
    name="verify-ipn",
    help="Offline NowPayments IPN signature verification — D4 self-check",
    no_args_is_help=True,
)

_SIGNATURE_ENV_VAR = "NOWPAYMENTS_IPN_SIGNATURE"

# Defined at module level (not inline in the signature) so ruff's B008 immutable-calls
# exemption for typer.Option applies — inline works for str/int defaults but not Path.
_BODY_OPTION = typer.Option(
    ..., "--body", "-b", exists=True, help="Path to a raw captured IPN body file"
)


async def _verify(*, raw_body: bytes, signature: str, product: str) -> None:
    settings = Settings()
    gateway = NowPaymentsGateway(settings)
    envelope = WebhookEnvelope(
        raw_body=raw_body,
        headers={"x-nowpayments-sig": signature},
        product_id=product,
    )

    try:
        outcome = await gateway.verify_webhook(envelope)
    except PaymentVerificationError as exc:
        console.print(f"[red]FAILED[/red] reason=[bold]{exc.reason.value}[/bold]")
        console.print(f"  message: {exc}")
        for key, value in exc.context.items():
            console.print(f"  {key}: {value}")
        raise typer.Exit(1) from None

    console.print("[green]OK[/green] — signature verified")
    console.print(f"  status: {outcome.status.value if outcome.status else None}")
    console.print(f"  lookup: {outcome.lookup.by}={outcome.lookup.value}")
    if outcome.amount_paid is not None:
        console.print(f"  amount_paid/amount_due: {outcome.amount_paid}/{outcome.amount_due}")
    if outcome.settled_currency is not None:
        console.print(f"  settled_currency: {outcome.settled_currency}")


@app.command()
def check(
    body: Path = _BODY_OPTION,
    signature: str = typer.Option(
        "",
        "--signature",
        "-s",
        help=f"x-nowpayments-sig header value (falls back to ${_SIGNATURE_ENV_VAR})",
    ),
    product: str = typer.Option(..., "--product", "-p", help="Product slug (vex, synthara)"),
) -> None:
    """Verify a captured IPN body+signature through the production verification path."""
    sig = signature or os.environ.get(_SIGNATURE_ENV_VAR, "")
    if not sig:
        console.print(
            f"[red]No signature provided — pass --signature or set ${_SIGNATURE_ENV_VAR}[/red]"
        )
        raise typer.Exit(1)

    asyncio.run(_verify(raw_body=body.read_bytes(), signature=sig, product=product))


if __name__ == "__main__":
    app()
