# Frontend Contract — Payments, Providers & Currency Catalog

**Backend:** `gearbox/apex` `master` @ `64f378b271d2` (full payment/currency arc merged). All schemas below are transcribed from the live code at this SHA, not from design docs. Regenerate the typed client (`gen:api`) against this or a later master before implementing — this document explains *semantics and flows* the OpenAPI schema can't express.
**Audience:** apex-frontend (SvelteKit checkout + superadmin panel).

---

## 1. The checkout flow at a glance

```
page load ──► GET /v1/billing/providers        (which payment methods to render)
          ──► GET /v1/billing/currencies       (crypto currency picker options)
          ──► GET /v1/billing/topup/options    (amount bounds, rate, discount tiers)

user picks amount (+ optionally a currency) and a provider
          ──► POST /v1/billing/topup/nowpayments  (or /topup/stripe)
          ◄── 201 { invoice_url, payment_id }
redirect user to invoice_url (NowPayments-hosted page / Stripe Checkout)

payment settles asynchronously (webhook → tokens credited server-side)
          ◄── SSE `balance.updated` on /v1/events/stream   (the "it worked" signal)
```

The FE never talks to NowPayments/Stripe APIs directly and never handles crypto addresses, amounts due, or payment statuses on-chain — the hosted invoice page owns all of that.

## 2. Discovery endpoints (public, unauthenticated, product-scoped)

Product scoping is automatic via the request's origin/host (middleware) — no product parameter.

### `GET /v1/billing/providers`

```json
[
  { "provider": "nowpayments", "display_order": 0 },
  { "provider": "stripe", "display_order": 1 }
]
```

- `provider` ∈ `"stripe" | "nowpayments"` today; **treat as an open string enum** — new providers will appear here without an FE release. Render only providers you have UI for; ignore unknown values gracefully.
- Already filtered server-side to *effective* providers (product capability ∩ superadmin runtime state). **Do not hardcode the provider list anywhere** — a superadmin can disable Stripe at runtime and it simply disappears from this response.
- Sort by `display_order` ascending (server pre-sorts; don't rely on it).
- Empty list is legal (everything disabled): render a "payments temporarily unavailable" state, not an error.

### `GET /v1/billing/currencies`

The crypto currency picker for the NowPayments path. DB-cached mirror of the NowPayments dashboard, refreshed every 3h + on admin demand.

```json
[
  { "ticker": "USDCMATIC", "name": "USD Coin", "network": "MATIC", "logo_url": "https://assets.…/payment-currency-logos/ab12….svg" },
  { "ticker": "USDTTRC20", "name": "Tether",   "network": "TRX",   "logo_url": null }
]
```

Field-by-field contract:

| field | type | FE behavior |
|---|---|---|
| `ticker` | string, uppercase | The value you send back as `pay_currency`. Never synthesize tickers; only echo what this endpoint returned. |
| `name` | string \| null | null ⇒ show the ticker as the label |
| `network` | string \| null | null ⇒ omit the network chip. Display data only — never branch logic on it. |
| `logo_url` | string \| null | null ⇒ generic coin icon. Absolute URL on the assets domain; immutable content (`Cache-Control: immutable`) — safe to cache forever, use plain `<img>`. |

**The one rule that matters: `[]` ⇒ hide the picker entirely and omit `pay_currency` from the top-up request.** The NowPayments invoice page then shows its own currency picker — checkout still works. An empty catalog is a *degraded UI*, never an error state. Same fallback if this request fails: proceed without a picker.

Recommended UX: sort/group client-side (e.g. stablecoins first, group by `network`), and always offer an explicit "Other / choose on the payment page" option that omits `pay_currency` — the backend supports any currency NowPayments can convert, not just the listed ones.

### `GET /v1/billing/topup/options`

```json
{ "min_amount_usd": 5, "max_amount_usd": 1000, "tokens_per_usd": 100, "tiers": [ { "threshold_usd": 20, "discount_pct": 5 } ] }
```

`tiers` ascending by `threshold_usd` — use as preset cards. Enforce min/max client-side for UX; the server re-validates (400 with a message on violation).

## 3. Creating a top-up (authenticated)

Both endpoints: `POST`, auth required, and an **`Idempotency-Key` header is required**.

### Idempotency-Key rules

- Generate a UUID v4 **per user intent** (per click of "Pay"), not per HTTP attempt. Retries of the same intent (network flake, timeout) reuse the same key.
- Same key + same body ⇒ replay: you get the **original response back** (same 201 body — same `payment_id`, same `invoice_url`). Safe to blindly retry.
- Same key + **different body** ⇒ `409` (see error table). This means "your retry logic is buggy", not "try again".
- Same key while the first attempt is still in flight ⇒ `409` (concurrent). Disable the pay button while a request is pending.

### `POST /v1/billing/topup/nowpayments`

```json
{ "amount_usd": 25, "pay_currency": "USDCMATIC" }
```

- `amount_usd`: integer ≥ 1 (real bounds from `/topup/options`).
- `pay_currency`: **optional**. Send a ticker from `/currencies` to pre-pin the invoice to that currency; omit the field (or send nothing) to let the user pick on the NowPayments page. Don't send `""` — omit.
- Unknown fields are rejected (`forbid_unknown_fields`) — don't spread extra props into the body.

`201`:
```json
{ "invoice_url": "https://nowpayments.io/payment/?iid=…", "payment_id": "018f…" }
```

Redirect (same tab recommended on mobile) to `invoice_url`. Keep `payment_id` for correlating history/status.

### `POST /v1/billing/topup/stripe`

```json
{ "amount_usd": 25 }
```

`201`:
```json
{ "checkout_url": "https://checkout.stripe.com/…", "session_id": "cs_…", "payment_id": "018f…" }
```

Stripe redirects back to the product's configured success/cancel URLs (`frontend_origin` + paths) — the FE routes for those pages must exist; they receive no payment metadata beyond what you stashed client-side, so persist `payment_id` before redirecting.

### Error table (both top-up endpoints)

| status | body | meaning | FE action |
|---|---|---|---|
| 400 | `{"detail": "…"}` | amount out of bounds / invalid `pay_currency` (NowPayments rejected the ticker) | show message, let user correct |
| 401 | standard auth error | not logged in | auth flow |
| 409 | `{"code": "payment_provider_disabled", "provider": "stripe"}` | provider disabled at runtime after your page loaded | re-fetch `/providers`, re-render methods, toast "this payment method is currently unavailable" |
| 409 | `{"code": "idempotency_conflict", …}` (detail text varies: reused key with different body, or concurrent in-flight) | client-side retry bug or double-submit | treat as terminal for this key; generate a fresh key only for a genuinely new user intent |
| 5xx | — | server/provider failure; idempotency record marked failed | safe to retry with the **same** key |

Distinguish the two 409s by `code`.

## 4. Knowing the payment succeeded

Crypto settlement takes minutes (network confirmations + IPN). The FE signal is **SSE**, not the redirect back from the invoice page (users may never return to the tab).

- Stream: `GET /v1/events/stream` (existing SSE infra with ticket auth + `Last-Event-ID` reconnection — reuse the current client).
- Event: envelope `{ "event_type": "balance.updated", "payload": …, "timestamp": …, "event_id": … }` with payload:

```json
{ "account_id": "018f…", "balance": 12500, "delta": 2500, "transaction_type": "topup" }
```

- On `balance.updated` with `transaction_type: "topup"`: refresh the balance display (use `balance` directly — it's authoritative post-credit) and celebrate.
- **Partial payments are real**: a user who underpays gets proportionally credited, and a later completing payment credits the remainder — so **multiple `balance.updated` events for one `payment_id`'s lifecycle are normal**. Just apply each `balance` as it arrives; never sum `delta`s yourself.
- Fallback when SSE is unavailable: poll `GET /v1/billing/balance` (`{ account_id, account_type, balance, organization_name }`).

Payment history for a receipts/status UI: `GET /v1/billing/transactions?limit=&type=&cursor=` → `CursorPage<TransactionResponse>` — cursor-only (`next_cursor` opaque string, `null` = last page; no total counts, no offsets — don't build page-number UI). `TransactionResponse.payment_id` links a credit back to the top-up. Payment rows themselves (`PaymentResponse`) expose `status` ∈ `pending | partially_paid | completed | failed | refunded` and `amount_usd` as a **string** (decimal — don't parseFloat for display, render verbatim); `currency` starts as `"USD"` and flips to the actually-paid ticker (e.g. `"USDTTRC20"`) once the first IPN arrives — display it as "paid with", not as the invoice denomination.

## 5. Superadmin panel endpoints

All under `/v1/admin/payments`, superadmin JWT required (401/403 otherwise), product-scoped like everything else.

### `GET /v1/admin/payments/providers` → full `ProviderInfo` list

Includes disabled providers and `credentials_configured: bool` (render a "keys missing" warning badge; it's informational — the backend still fails loud at charge time).

### `PATCH /v1/admin/payments/providers/{provider}`

```json
{ "is_enabled": false }            // or { "display_order": 2 }, or both
```

At least one field required (400 if empty). `display_order` ∈ [0, 1000]. 404 for unknown providers or ones outside the product's capability (can't runtime-enable what has no wiring). Returns the updated `ProviderInfo`. Every change is audit-logged server-side.

### `GET /v1/admin/payments/currencies`

Full catalog **including unavailable rows** — the admin diff view against the public list:

```json
[ { "ticker": "USDCMATIC", "provider": "nowpayments", "is_available": true, "name": "USD Coin", "network": "MATIC", "logo_key": "payment-currency-logos/ab12….svg", "logo_source_url": "https://nowpayments.io/images/coins/usdc.svg", "logo_synced_at": "2026-07-16T…", "last_seen_at": "2026-07-16T…" } ]
```

`is_available: false` rows = currencies unchecked in the NowPayments dashboard since last seen — render greyed out with `last_seen_at`.

### `POST /v1/admin/payments/currencies/refresh`

No body. Synchronous; give it a spinner (it calls NowPayments live and may fetch logos).

- `201/200` → `[ { "provider": "nowpayments", "upserted": 42, "deactivated": 1 } ]` — show the counts.
- `502` with detail → NowPayments/config failure; previous catalog untouched. Message: "refresh failed, showing last known list".

Admin workflow to surface in the UI: *edit currencies in the NowPayments dashboard → click refresh here → verify the list*. The 3h background sync makes the button optional but the immediate path is this.

## 6. Invariants the FE must uphold (the contract's spirit)

1. **No hardcoded provider or ticker lists, anywhere.** Both are runtime data from the endpoints above; that's the entire point of the backend design.
2. **Absence over empty string** for `pay_currency`.
3. **Degrade, don't break**: empty providers ⇒ unavailable-state; empty currencies ⇒ no picker + omit `pay_currency`; null logo ⇒ generic icon; null name/network ⇒ ticker-only label; SSE down ⇒ poll balance.
4. **Idempotency keys are per-intent**, and the two 409 codes mean different things.
5. **Balance from the server is truth** — render `balance`, never accumulate `delta`s.
6. Ticker/network/name are **display data** — no client logic may branch on their values (e.g. no "if network === MATIC" special cases).

## 7. Gap checklist to close (suggested order)

- [ ] `gen:api` against current master; verify the new types (`PublicCurrency`, `PublicPaymentProvider`, `TopUpNowPaymentsRequest.pay_currency?`, `AdminCurrency`, `SyncResult`) appear.
- [ ] Checkout: provider method list driven by `/providers` (delete any hardcoded method array).
- [ ] Checkout: currency picker component (`/currencies`, grouping by `network`, "Other" option, empty-state fallback) wired into the NowPayments top-up body.
- [ ] Post-payment UX: SSE `balance.updated` handler + pending-payment state keyed by `payment_id`; balance-poll fallback.
- [ ] Error handling: both 409 codes, 400 detail surfacing, provider-disabled re-fetch.
- [ ] Admin panel: provider toggle/order UI, catalog table with unavailable rows, refresh button with counts/502 handling.
- [ ] Stripe success/cancel routes exist and read the client-persisted `payment_id`.
