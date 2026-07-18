# Debugging NowPayments IPN verification failures

Every rejected IPN logs a `payment.verification_failed` event with a
`reason` field (see `PaymentVerificationReason` in
`src/api/services/billing_errors.py`) plus safe, non-secret diagnostics —
signature prefixes, a body hash, and the parsed payload's key list. That's
usually enough to tell operational misconfiguration (dashboard IPN-format
toggle, wrong secret in the deployed env) apart from a code-level bug
without redeploying anything.

## 1. Capture a raw IPN body

Resend the IPN from the NowPayments dashboard against an endpoint you
control (a capture route, a request-bin, or a `cloudflared`/`ngrok` tunnel
dump). Save the **exact raw bytes** of the request body to a file — do not
re-serialize it through `json.dumps`, since that can change byte-for-byte
formatting and invalidate the signature comparison. Also note the
`x-nowpayments-sig` header value from the same request.

## 2. Run the offline self-check

```bash
python -m src.cli.verify_ipn --body capture.json --signature <sig-from-header> --product vex
# or, to avoid the signature landing in shell history:
NOWPAYMENTS_IPN_SIGNATURE=<sig-from-header> python -m src.cli.verify_ipn --body capture.json --product vex
```

This runs the exact production verification path
(`NowPaymentsGateway.verify_webhook`) against the `Settings` resolved from
your local/deployed environment — no network calls, no DB access.

## 3. Reading the reason codes

| Reason | Meaning |
|---|---|
| `missing_signature_header` | No `x-nowpayments-sig` header at all — check the capture, not NowPayments' signer. |
| `signature_mismatch` | Header present but doesn't match the locally computed HMAC. See below. |
| `malformed_json` | Body isn't valid JSON, or isn't a JSON object. |
| `malformed_order_id` | `order_id` isn't the expected `{"payment_id": ..., ...}` JSON, or `payment_id` isn't a UUID. |
| `amount_fields_invalid` | `actually_paid`/`pay_amount` missing, non-numeric, or non-positive on a settled (`finished`/`partially_paid`) status. |
| `product_mismatch` | The payment's `product_id` doesn't match the IPN callback URL's resolved product. |
| `missing_field` | Reserved for other required-field gaps. |

### Interpreting `signature_mismatch`

The context carries `received_sig_prefix` and `computed_sig_prefix` (first
8 hex chars of each), `body_sha256_prefix`, and `secret_source`
(`per_product` or `legacy_global` — never the secret value itself).

- **`body_sha256_prefix` matches a known-good capture, but prefixes still
  differ:** the HMAC canonicalization is correct and the body is intact —
  the deployed IPN secret doesn't match what's configured in the
  NowPayments dashboard for this product, **or** the dashboard's IPN
  format is set to "Classic" instead of "All-Strings". **All-Strings is
  mandatory** for this implementation (see `NowPaymentsGateway.verify_webhook`
  docstring) — `parse_float=str, parse_int=str` only round-trips
  byte-identically for All-Strings bodies; Classic re-serializes numbers
  differently and fails every signature check regardless of whether the
  secret is correct.
- **`body_sha256_prefix` differs from what you expect:** the capture itself
  is different from what NowPayments actually sent (e.g. a proxy/tunnel
  mutated it) — re-capture before concluding anything about the secret.

A clean run (`OK`) prints the resolved `status`, `lookup`, and settlement
amounts, confirming the whole verification path — including the
`Settings.nowpayments_ipn_secret_for()` resolution — end to end.
