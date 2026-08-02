# Backend API Reference — Apex REST API

> _Last updated: 2026-07-28 — **Review remediation r2: serialize push-subscription creation
> against bulk revocation** (§15b): `POST /v1/push/subscriptions` could commit a fresh
> subscription row *after* a concurrent bulk revocation (logout-all, password change/reset,
> deactivation, refresh-token reuse detection) had already run its cleanup — `push_subscriptions.
> user_id` carries `ON DELETE CASCADE`, so the insert's implicit `FOR KEY SHARE` lock on the user
> row blocks behind the bulk path's `FOR UPDATE`, then proceeds to insert regardless once the bulk
> transaction commits. The device the revocation was meant to unsubscribe stayed subscribed
> indefinitely. Fixed by having the handler acquire the same user-row lock
> (`UserRepository.lock_user_for_session_change`) *before* re-checking revocation and upserting:
> whichever side wins the lock, the other observes a consistent outcome — either the bulk delete's
> subsequent snapshot includes the new row, or the insert's re-check sees the epoch the bulk path
> just wrote and rejects with `401`. Lock ordering is user-row-only in this handler, so it cannot
> invert the user-row -> refresh-token-row ordering the bulk paths already rely on. Fails open on
> Redis unavailability, consistent with every other revocation check in this codebase — a cache
> outage must not block push registration. This required exposing the decoded JWT `TokenPayload`
> (previously discarded once `auth_guard` returned) via `connection.state["token_payload"]` and a
> new `get_current_token_payload` DI dependency, following the existing `current_user_id` pattern;
> `optional_auth_guard` mirrors this for consistency, explicitly setting `None` on the anonymous
> path. Proven by a 50-iteration-per-ordering concurrency test against a real database
> (`tests/integration/test_push_subscription_revocation_race.py`) forcing both lock-acquisition
> orderings deterministically via an `asyncio.Event` rather than relying on scheduling luck.
> **Follow-up filed, not fixed here**: revocation is enforced only at the guard boundary, so any
> other mutating handler can still commit work authorized by a token revoked mid-request — push
> subscriptions were singled out because the row outlives the request and keeps delivering to a
> device, which is what makes this instance worth a dedicated fix rather than accepting the
> general gap.
>
> _Prior (2026-07-27): **Review remediation: push-cleanup ops alert wiring** (§15c, §17):
> the push-subscription-cleanup work below shipped `OpsEventType.PUSH_SUBSCRIPTIONS_CLEANUP_FAILED`
> from all three affected services, but never wired it past that point — no `NotificationClass`
> member, no `telegram/mapping.py` branch, no catalog entry, no platform-scope membership, so the
> event reached no operator (the same gap previously found and fixed for
> `TOKEN_REVOCATION_FAILED`). Fixed by following that precedent exactly: new
> `NotificationClass.PUSH_SUBSCRIPTIONS_CLEANUP_FAILED` (`"push_subscriptions.cleanup_failed"`),
> platform-scoped, with a mapping branch and catalog entry, plus a one-time preference seed
> (migration `032`) for admins who already have a Telegram link — see the Notification Classes
> table below. `OpsEventType`'s docstring now points at `NotificationClass`'s wiring checklist, so
> a developer adding a new ops event from that enum has a reason to open the other one. Also in
> this pass: the three near-identical `_delete_push_subscriptions` methods (`AuthService`,
> `UserService`, `EmailVerificationService`) were consolidated into one module-level helper
> (`src.api.services.push_cleanup.delete_user_push_subscriptions`) — the `None`-session guard
> stayed at the two call sites that need it (`AuthService`/`UserService`), and existing
> `{source}.push_subscriptions_deleted` / `_cleanup_failed` log event names are unchanged, so no
> log-based alert or dashboard keyed on them needs updating. `PushSubscriptionRepository.
> delete_all_for_user` no longer carries a `type: ignore` — its `CursorResult` is now typed
> explicitly and a driver-reported `-1` (no count available) is clamped to `0` before it reaches
> logs or the ops payload. A missing `session` on `AuthService`/`UserService` (construction without
> one, which only happens in tests today — both DI providers always pass one) now logs a warning
> instead of silently skipping the cleanup. **Rejected**: Sourcery's suggestion to inject
> `PushSubscriptionRepository` instead of constructing it inline — every other repository in this
> codebase (e.g. `admin_management.py`) is constructed inline from the session, so special-casing
> just this one would make these three services the outlier; a wholesale inject-vs-construct
> refactor is a separate change with its own review.
>
> _Prior (2026-07-27): **Push-subscription cleanup on bulk session revocation** (§2, §3,
> §15b): closes a gap where no server-side path ever deleted a user's Web Push subscriptions.
> Previously only the client-initiated `DELETE /v1/push/subscriptions` (the caller's own endpoint)
> and the dispatcher's own expired-subscription pruning ever removed a row — a user who hit
> "log out all devices" because they suspected compromise left the attacker's device subscribed
> indefinitely, since nothing else ever touched the table. `PushSubscriptionRepository` gained
> `delete_all_for_user(user_id) -> int`, now called from all **five** bulk-revocation sites
> alongside their existing `TokenRevocationService.revoke_user_sessions` call (not inside it —
> different failure semantics, no dependency on Redis availability):
> `POST /v1/users/me/logout-all`, `POST /v1/users/me/password`, `POST /v1/auth/reset-password`,
> `DELETE /v1/users/me`, and refresh-token reuse detection. **`POST /v1/auth/logout`
> (single-device) is deliberately unchanged** — it still only ever touches the calling device's own
> subscription client-side, exactly as before. Push deletion runs inside a SAVEPOINT and is
> best-effort: a failure never blocks the primary action (the password change/reset/logout-all/
> deactivation still succeeds) and is reported via the new platform-scoped
> `NotificationClass`-adjacent ops event `ops.push.subscriptions_cleanup_failed`
> (`PushSubscriptionsCleanupFailedOpsPayload`), logged truthfully rather than assumed successful.
> No request/response shape changes; no frontend action required to adopt this, though the
> client-side detach-before-revoke workaround in `ChangePasswordModal`/`LogoutAllModal` is now
> redundant and can be simplified to a local-only `PushManager` unsubscribe.
>
> _Prior (2026-07-27): **`Clear-Site-Data` coverage for session-ending endpoints** (§2, §3):
> a frontend request to change `Cache-Control` on `/v1/content/...` to `private, no-store` was
> declined — it would force a full re-fetch of every thumbnail on every library grid render,
> reproducing the parallel-request saturation behind a prior mobile bug, and would nullify the
> video prewarm design's HTTP-cache reuse. The residue concern behind that request (private images
> sitting in a shared device's HTTP cache after an account switch) is instead addressed by
> extending `Clear-Site-Data: "cache", "storage"` — previously sent only by single-device
> `POST /v1/auth/logout` — to every other endpoint that ends the caller's own session:
> `POST /v1/users/me/logout-all`, `POST /v1/users/me/password`, `POST /v1/auth/reset-password`,
> and `DELETE /v1/users/me`. All five now share one constant (`CLEAR_SITE_DATA_HEADER`,
> `src/api/security/response_headers.py`). Also newly documented: `POST /v1/auth/logout` returning
> `200` + the header **regardless of whether `refresh_token` was valid, unknown, or already
> revoked** is a deliberate, permanent contract, not incidental behavior — a client that discovers
> its session was revoked remotely has no other way to purge this origin's HTTP cache, so it fires
> a best-effort logout purely to receive this header. `Cache-Control` on `/v1/content/...` is
> unchanged. No request/response shapes or status codes changed for any of these five endpoints.
>
> _Prior (2026-07-26): **Token-revocation-failure alerting** (§15c, §17): closes the last
> outstanding gap in [#142](https://github.com/gearbox/apex/issues/142). A failed bulk
> access-token revocation (the Redis write behind `logout_all`/`change_password`/
> `deactivate_account`/`reset_password`/`token_reuse_detected`/`refresh_race_detected` failing
> while Redis is otherwise configured) now reaches operators instead of only being logged. New
> `NotificationClass.TOKEN_REVOCATION_FAILED` (`"token_revocation.failed"`) appears in `GET
> /v1/admin/notifications/classes` (§15c, §17) as **platform-scoped** — delivered to every
> subscribed admin/superadmin regardless of product, like the two health classes, since a
> revocation-backend degradation isn't specific to one product. Once subscribed, an admin
> receives a Telegram message naming the failing `op` and stating that the affected user's
> existing access tokens and content cookies remain valid until they expire — `op` matters: a
> failure during `token_reuse_detected` is materially more serious than one during a routine
> `logout_all`. **Admins must keep this class selected**: migration `029` seeds a subscription
> for every admin who already has a Telegram link, so existing installs get immediate coverage
> without anyone touching preferences — but subscription is still row-presence, so the next
> full-set `PUT /v1/admin/notifications/preferences` from a seeded admin that omits this class
> un-subscribes them, exactly like any other class. This was a deliberate choice over making the
> class bypass preferences entirely (which would special-case one class out of the uniform
> subscription model): `GET /v1/admin/health`'s `TokenRevocationChecker` remains a second,
> preference-independent channel surfacing the same degradation, so the seed-plus-this-note is
> judged sufficient rather than mandatory-delivery. Backend-only — no request/response shape
> changes, no frontend action required.
>
> _Prior (2026-07-26): **Access-token revocation, remediation pass** (§2, §3): closes the
> remaining gaps in [#142](https://github.com/gearbox/apex/issues/142) found in review. Three fixes,
> no request/response shape changes:
> (1) `POST /v1/auth/reset-password` now also bulk-revokes live access tokens/content cookies via
> `TokenRevocationService.revoke_user_sessions`, matching the authenticated change-password path —
> previously the account-recovery flow a user reaches *because* they suspect compromise left a
> stolen access token or content cookie live for its full remaining lifetime.
> (2) Refresh-token reuse detection (`AuthService.refresh_tokens`'s theft-detection branch) now
> bulk-revokes the same way before raising, so its "All sessions have been invalidated" message is
> now accurate rather than aspirational.
> (3) `optional_auth_guard` (used by `GET /v1/providers`) now also consults `TokenRevocationService`
> — a revoked token previously still authenticated on this route, since only `auth_guard` and
> `content_auth_guard` checked revocation. It degrades a revoked token to anonymous, exactly like it
> already does for a product-mismatched token, and — unlike the two guards above — tolerates a
> missing revocation service (treats as not-revoked) rather than raising, preserving its
> never-401 contract for an anonymous-capable route. Also documented: single-device
> `POST /v1/auth/logout` cannot revoke that device's `apex_content` cookie server-side (see the
> endpoint note below) — a pre-existing limitation, not a regression, now made explicit. No frontend
> action required for any of this.
>
> _Prior (2026-07-25): **Access-token revocation** (§2, §3): closes
> [#142](https://github.com/gearbox/apex/issues/142). `POST /v1/auth/logout`,
> `POST /v1/users/me/logout-all`, `POST /v1/users/me/password`, and `DELETE /v1/users/me` now invalidate
> live access tokens and the `apex_content` cookie **immediately**, not just refresh tokens — previously a
> stolen access token (or a cookie minted before the 24h TTL raise) survived an explicit "log out
> everywhere" for its full remaining lifetime. Two mechanisms, both backed by Redis (`TokenRevocationService`,
> `src/api/services/token_revocation.py`): a per-user "revoke all sessions" epoch (bulk sites — logout-all,
> password change, deactivation) and a per-token denylist keyed by the token's own `jti` (single-device
> logout only, from the `Authorization` header presented to that call). Both `auth_guard` and
> `content_auth_guard` now consult this on every request — one Redis round-trip, after the (free) product
> check and before setting connection state. **Redis dependency, fail-open**: this is a deliberate security
> posture, not a gap — with `REDIS_URL` unset, or Redis transiently down, revocation silently degrades to
> the pre-#142 refresh-token-only behavior (logged once at startup / on each backend error as
> `authrev.backend_unavailable`) rather than 401ing every authenticated request. No response shapes, request
> shapes, or status codes changed — this is a behind-the-scenes tightening of what "logged out" means.
> No frontend action required.
>
> _Prior (2026-07-24): **Content Proxy performance & streaming** (§9): `GET /v1/content/outputs/{id}`
> and `GET /v1/content/uploads/{id}` now support single-range `Range` requests (`206 Partial Content` /
> `416 Range Not Satisfiable`, `Accept-Ranges: bytes` on every 200/206 — multipart ranges are treated as a
> full 200) and honor `If-None-Match` for `304 Not Modified` (checked after ownership, before any R2
> traffic). Every request now makes at most one R2 round-trip — the standalone `head_object` call is gone;
> `Content-Type`/`Content-Length`/`Content-Range` all come from the single GetObject response, and an
> out-of-range request is rejected using the DB-recorded size before R2 is ever touched. New endpoint
> `POST /v1/auth/content-cookie` (§2, Bearer-only, 204, rate-limited 20/minute) re-mints the `apex_content`
> cookie without a full token refresh — frontend should prefer it over `silentRefresh` for recovering
> image/video auth. No response body contracts changed; frontend should regenerate types (`gen:api`) to
> pick up the new endpoint._
>
> _Prior (2026-07-22): **Breaking change:** replaced **Gallery** (§10) and the uploads-list
> endpoint with a unified **Library** API (new §10) — one paginated, cursor-based read model over
> `user_images` + `generation_outputs`, addressed by a typed `asset_ref` (`"upload:<uuid>"` /
> `"output:<uuid>"`). `GET /v1/gallery/`, `GET /v1/gallery/{job_id}`, `GET /v1/storage/uploads`, and
> `DELETE /v1/content/{content_id}` are all **removed** — no redirects, no deprecation window
> (pre-prod). New surface: `GET /v1/library/` (grid, with `source`/`media_type`/`model`/`favorite`/
> `project_id`/`tag_id`/`expiring`/`query`/`created_from`/`created_to` filters and
> `newest`/`oldest`/`expiring_soon` sort), `GET /v1/library/assets/{asset_ref}` (detail),
> `GET /v1/library/assets/{asset_ref}/lineage` (bounded ancestor/descendant graph),
> `GET /v1/library/groups/{job_id}` (generation-group detail — the old `GalleryGroupDetail`, ported
> as-is), `PATCH`/`PUT favorite`/`DELETE favorite`/`DELETE` on a single asset, `POST
> /v1/library/assets/bulk` (favorite/project/tags/delete up to 100 refs in one call), and full CRUD
> for `/v1/library/projects` and `/v1/library/tags`. Every grid item carries a server-resolved
> `available_actions: LibraryAction[]` — no more inferring allowed actions from `source`/media type
> on the client. Favorites and display titles are per-asset (`library_asset_metadata`, migration
> `025`); projects (one per asset, migration `026`) and tags (many-to-many, migration `027`) are
> user-created groupings layered on top. Frontend must regenerate types (`gen:api`), drop all
> `/app/gallery` and `/app/uploads` routes/references, and switch to `/app/library`. See new §10 for
> the full contract; `GalleryBadge`/`GallerySourceType` (§17) are replaced by `LibraryBadge`/
> `LibraryGroupSourceType`/`LibrarySort`/`LibraryAssetSource`/`LibraryAction`._
>
> _Prior (2026-07-17 — Added **Admin Ops Notifications (Telegram)** (new §15c): admins/superadmins can subscribe to operational events — `user.registered`, `generation.created`, `gpu_node.started`, `generation.failed` (product-scoped) and `health.degraded`/`health.restored` (platform-scoped) — and receive them as Telegram messages via a one-time `t.me` deep-link flow. New endpoints under `/v1/admin/notifications/*`: `GET /classes` (catalog), `GET`/`PUT /preferences` (per-class subscribe + optional `min_interval_seconds` throttle, full-set replace), `GET /preferences/{user_id}` (superadmin-only read of another admin's set), `GET`/`POST /telegram`/`DELETE /telegram` (link status, create/rotate deep link, unlink). Backend-only — no frontend/consumer-facing surface changes. New `NotificationClass` enum (§17). `AuditLogEntry.action` (§14) gained `notification_prefs.update`, `telegram.link_requested`, `telegram.unlinked`. Requires `TELEGRAM_BOT_TOKEN` + `REDIS_URL` server-side to actually deliver; preference endpoints work regardless._
>
> _Prior (2026-07-17): Added **Currency Suppression**: a superadmin deny-list for provider-side "zombie" tickers (NowPayments confirmed a data bug where `merchant/coins` can report currencies they've effectively delisted and won't fix). New `PATCH /v1/admin/payments/currencies/{provider}/{ticker}` (§14) toggles `is_suppressed` on a catalog row — suppressed tickers are immediately excluded from `GET /v1/billing/currencies` (§11), which now gained `AdminCurrency.is_suppressed` on the admin GET, and pinning a suppressed ticker on `POST /v1/billing/topup/nowpayments` (§11) now returns `400 { "code": "pay_currency_suppressed", "pay_currency": "<TICKER>" }`. Suppression survives every catalog sync (the sync code never reads/writes the flag) and requires an already-seen ticker (404 otherwise — no pre-emptive/pattern suppression). See the "Provider-side zombie currencies" ops note under §14. Frontend should regenerate types (`gen:api`) and handle the new 400 by re-fetching `/currencies` and re-prompting the user._
>
> _Prior (2026-07-16): Added the DB-cached **Payment Currency Catalog**: public `GET /v1/billing/currencies` (§11) returns available tickers with display name/network/R2-hosted logo, synced from NowPayments `merchant/coins` (availability) + `full-currencies` (metadata) by a periodic worker (default 3h) and on-demand via superadmin `GET/POST /v1/admin/payments/currencies[/refresh]` (§14). Empty list ⇒ hide the picker and omit `pay_currency`; catalog state never gates checkout. No hardcoded ticker list exists anywhere in the contract. Frontend should regenerate types (`gen:api`)._
>
> _Prior (2026-07-16): `pay_currency` on `POST /v1/billing/topup/nowpayments` (§11) is now **optional** (was required). Omit it to let the customer pick any currency/network NowPayments supports on the hosted invoice page instead of pinning one; blank/whitespace is treated the same as omitted. `PaymentResponse.currency` (admin §14) is `"USD"` at charge time for an unpinned invoice and is patched to the customer's actual settled ticker (e.g. `"USDCMATIC"`, `"USDTTRC20"`) once the first IPN reports it — this can happen on intermediate statuses, not just completion. This is backwards-compatible: existing callers that always send `pay_currency` see no behavior change. Frontend should regenerate types (`gen:api`) to pick up the now-optional field._
>
> _Prior (2026-07-13): **Breaking change:** fixed i2i aspect-ratio distortion. `aspect_ratio` on `POST /v1/generate` (§4) is now **optional** — `None`/omitted means "provider default" for t2i (1:1 image, 16:9 video) and "follow the source image's aspect" for i2i. For i2i, an explicit `aspect_ratio` is now capability-gated per model: `GET /v1/providers` `ImageConstraints` (§5) gained `edit_aspect_ratios` — an empty list means the model cannot reshape on edit (it would silently stretch the source) and any explicit `aspect_ratio` on an i2i request for that model now returns `400 validation_error`; `grok-imagine-image` currently has an empty list, `aisha-image` supports the full ratio list. t2i requests are now also validated against the model's `aspect_ratios` list (previously unenforced — any value silently passed through). `aspect_ratio: null` on job/gallery responses (§6, §10) now additionally means "generation followed the source image's aspect" for i2i jobs, alongside its prior meanings. Frontend must regenerate types (`gen:api`) and stop hardcoding an `aspect_ratio` default on i2i requests for non-reshaping models._
>
> _Prior (2026-07-12): Added **Video Frame Extraction** (new §9b): `POST /v1/frames/preview` and `POST /v1/frames/extract` run as free, non-billed background jobs (no `Idempotency-Key`) against either a `GenerationOutput` video or a user-uploaded video; poll `GET /v1/frames/jobs/{id}` until `completed`/`failed`. Preview frames are presigned R2 URLs generated fresh per poll (never cache beyond the current session — see §9b); extracted frames become ordinary uploads (standard `MediaObject`, stable `/v1/content/uploads/{id}` URLs) with new source-video lineage. Also: `POST /v1/storage/upload` (§8) now accepts video (`video/mp4`, `video/webm`, `video/quicktime`, ≤20MB, ffprobe-validated server-side — the declared `Content-Type` is never trusted); the Content Proxy (§9) inline-safe `Content-Type` allowlist grew to match. New enums `FrameExtractionKind`/`FrameExtractionStatus` (§17); `MediaFormat` gained `webm`/`mov`. Frontend must regenerate types (`gen:api`) — see `docs/contracts/video-frame-extraction.md` for the full contract._
>
> _Prior (2026-07-12): Gallery items now expose `expires_at` (§10): `GalleryGridItem.expires_at` (sourced from the cover output) and `GalleryOutputItem.expires_at` (per-output), matching the existing `ImageListItem`/`OutputListItem` contract in Storage (§8). Frontend can now render a "Delete in N days/hours/minutes" badge directly from the gallery grid/detail responses without a separate Storage lookup. Also: content retention is now actively enforced by a periodic sweeper — see the new retention note in §8. Frontend must regenerate types (`gen:api`)._
>
> _Prior (2026-07-10): Added the payment gateway protocol and per-product runtime provider registry. Checkout clients must discover enabled providers through public `GET /v1/billing/providers`; superadmins manage capability members through `GET/PATCH /v1/admin/payments/providers`. Disabling a provider blocks new charges with a stable `409` body but never blocks webhook settlement. Provider changes are appended to the admin audit log with a nullable `target_user_id`._
>
> _Prior (2026-06-30): MediaObject contract tightening (§5b): `ImageVariant.width`/`height` are now required non-null integers (serializer skips and logs any legacy dimensionless variant row rather than emitting null). `MediaObject.variants` is now required (was optional with a default) — OpenAPI marks it in `required`. Content cookie `Domain` is now omitted (host-only) in dev mode so the `apex_content` cookie is stored correctly over `http://localhost`; production posture unchanged (`Domain=<product>`, `Secure`). Frontend must re-run `gen:api` to pick up the updated types, then drop `?? []` on `variants` and `.filter(v => v.width)` guards. Prior (2026-06-29): Cursor pagination on audit log (§14): `GET /v1/admin/manage/audit` now returns `CursorPage<AuditLogEntry>` instead of a bare array. Pass `cursor=next_cursor` for subsequent pages. Frontend must regenerate types and switch to cursor-scroll. Prior (2026-06-27): Unified Image Variants (§6, §8, §10): `MediaObject` replaces all per-output presigned URL fields across the Jobs, Storage, and Gallery APIs. Jobs API no longer presigns URLs — all content URLs are stable content-proxy paths. Gallery cover logic now always uses the job's own primary output (no longer sources input images). Upload thumbnails (sm=150px, md=512px WEBP) generated automatically on upload._
>
> _Prior (2026-06-26): removed `bundle_name` and `bundle_version` from the frontend-facing API surface (§7, §15). Prior (2026-06-17): synced the doc with `master` — Aisha generation parameter system (§4), quality-tier capabilities (§5), per-output `thumbnail_url` (§6), GPU-session provisioning fields + internal callback (§7), corrected billing public-endpoint behaviour (§11), corrected model-capability matrix and new enums (§17), corrected `POST /v1/auth/register` contract (§2)._

> **Source:** `gearbox/apex` repository
> **Framework:** Litestar 2.5+ / Python 3.13
> **Schema:** `GET /docs/openapi.json` from running backend (Litestar OpenAPIConfig has `path="/docs"`)
> **Last synced:** 2026-07-13 — `master` @ `20bec2ec71a432f8cdb1325da4f816fe38641a7a` (+ pending i2i aspect-ratio capability fix, v0.26.0)
>
> _2026-07-08: **Breaking change** — replaced fixed token packages with tiered free-amount top-up (§11). `POST /v1/billing/topup/{stripe,nowpayments}` now take `{ amount_usd: int }` instead of `{ package_id: string }`; `GET /v1/billing/packages` is removed, replaced by `GET /v1/billing/topup/options`. Frontend must regenerate types (`gen:api`) and update the top-up UI to a preset-amounts + free-input flow (separate prompt)._
>
> _2026-07-08: Added Web Push notifications (§15b) — `GET /v1/push/vapid-public-key`, `POST /v1/push/subscriptions`, `DELETE /v1/push/subscriptions`. Delivers notifications even when the app is closed, unlike SSE (§15). Frontend must regenerate types._

This document captures the API surface that the frontend depends on. It is a **stable reference**, not a live mirror. When endpoints change in the backend, update this document and regenerate `types.ts`.

---

## Idempotency

Four mutation endpoints require an `Idempotency-Key` header to prevent duplicate operations on network retries:

- `POST /v1/generate/`
- `POST /v1/billing/topup/stripe`
- `POST /v1/billing/topup/nowpayments`
- `POST /v1/admin/accounts/{account_id}/adjust`

### How it works

1. The client generates a unique key (UUIDv4 recommended, max 64 characters) and sends it as `Idempotency-Key: <key>`.
2. On the **first** request, the server processes the operation and caches the response.
3. On **retry** with the same key, the server returns the **original cached response** without re-executing the operation.
4. Keys are scoped to `(user_id, product_id)` — the same key from different users or products does not collide.
5. Keys expire after 24 hours (configurable via `IDEMPOTENCY_KEY_TTL_HOURS`).
6. Generation's synchronous final results (201, billable 422, and normalized provider failures) persist the job state, billing ledger result, resource ID, HTTP status, response body, and completed idempotency key in one transaction. A crash cannot leave a durable debit behind a reclaimable `processing` key.
7. A key stuck `processing` before a final outcome (the original request's connection dropped, its worker crashed, etc.) for longer than `IDEMPOTENCY_PROCESSING_STALE_SECONDS` (default 120s) can be reclaimed by the next retry. A retry within that window gets `409 idempotency_conflict`.
8. If committing a completed outcome succeeded but its acknowledgement was lost, a retry returns the stored outcome and retries its post-commit notifications. Notification failures are logged and do not change the completed outcome, billing, or replay response.

### Error responses

| Status | Error code | Meaning |
|--------|------------|---------|
| `400` | `validation_error` | `Idempotency-Key` header missing or exceeds 64 characters |
| `409` | `idempotency_conflict` | Same key is currently being processed and is still within the staleness window (see point 6 above), or was reused with a different request body. Retry after 1 second (`Retry-After: 1` header included). |

---

## Pagination

All list endpoints return a **unified `CursorPage<T>`** shape:

```typescript
interface CursorPage<T> {
  items: T[];
  limit: number;               // echoed page size
  has_more: boolean;           // true when there are additional pages
  next_cursor: string | null;  // opaque cursor token for the next page; null if none
}
```

All list endpoints use **keyset (cursor) pagination** exclusively. There are no `total` count fields and no `offset` parameters.

- Pass `cursor=<next_cursor>` from the previous response to fetch the next page.
- The cursor is an opaque, URL-safe base64 token; do not parse or construct it manually.
- `next_cursor` is `null` when `has_more` is `false`.

```
// Page 1
GET /v1/jobs?limit=20
Response: { items: [...], limit: 20, has_more: true, next_cursor: "eyJ..." }

// Page 2 — stable even if new jobs were added between requests
GET /v1/jobs?limit=20&cursor=eyJ...
Response: { items: [...], limit: 20, has_more: false, next_cursor: null }
```

---

## 0. Multi-Product Architecture

The backend serves two distinct products from the same codebase:

| Product | Slug | Domains | Audience | Content |
|---------|------|---------|----------|---------|
| **example.com** | `vex` | `vex-domain.com`, `www.vex-domain.com`, `app.vex-domain.com` | Consumer / creator | Permissive — NSFW-capable models available |
| **Synthara** | `synthara` | `synthara-domain.com`, `www.synthara-domain.com`, `app.synthara-domain.com` | Enterprise / business | SFW only, professional |

### Product Resolution

Every request is resolved to a product via:
1. `Origin` header domain (preferred)
2. `Host` header domain
3. `X-Product-Id` header (`vex` or `synthara`) — dev fallback
4. `localhost` / `127.0.0.1` / `0.0.0.0` → uses `DEFAULT_PRODUCT` env var (default: `vex`)

If no product can be resolved: `400 Bad Request` with `{ "error": "unknown_product" }`.

The response always includes `X-Product-Id` header for debugging.

### User Scoping

Accounts are **product-scoped** — the same email address can register independently on both products. Users cannot authenticate across products.

### JWT Token Scoping

JWT tokens embed a `product_id` claim. Tokens issued for one product are rejected on the other product.

---

## 1. Base URL & CORS

- **Local dev:** `http://localhost:8000`
- **Production:** Determined by `Origin`/`Host` header per-product (no single base URL)
- **CORS:** Backend allows `*` origins in dev; tighten for production

---

## 2. Authentication

### 2.1 Token Model

- **Access token:** Short-lived JWT (default 15 min), passed in `Authorization: Bearer <token>`
- **Refresh token:** Longer-lived (default 7 days), opaque string stored server-side
- **Rotation:** Every refresh issues a new refresh token and invalidates the old one
- **Family tracking:** If a revoked refresh token is reused, ALL tokens in that family are revoked (theft detection)
- **Password hashing:** Argon2id (server-side)

### 2.2 Auth Endpoints

#### `GET /v1/auth/product-info`

```
Response: {
  product: string,              // "vex" | "synthara"
  display_name: string,         // e.g. "example.com"
  age_gate: string,             // "none" | "checkbox" | "date_of_birth"
  allowed_auth_methods: string[],  // e.g. ["email_password", "google_oauth"]
  content_rating: string,       // "sfw" | "permissive"
  payment_providers: string[]   // e.g. ["stripe", "nowpayments"]
}
Note:     Public endpoint — no auth needed. Frontend calls this on load.
```

#### `POST /v1/auth/register`

```
Request:  { email: string, password: string, display_name?: string }
Response: { access_token, refresh_token, token_type: "bearer", expires_in: int, expires_at: datetime,
            content_cookie_expires_at: datetime }
Status:   201 Created
Errors:   400 (validation_error | email_exists)
Note:     The request body no longer carries age fields — `age_confirmed` / `date_of_birth`
          are NOT accepted by this endpoint anymore, and the previous
          `403 age_verification_required` response is no longer emitted here.
          Age-gate requirements are still advertised via GET /v1/auth/product-info
          (`age_gate`) for the frontend to enforce client-side before submitting.
          `email_exists` is returned as 400 (not 409).
```

#### `POST /v1/auth/login`

```
Request:  { email: string, password: string }
Response: { access_token, refresh_token, token_type: "bearer", expires_in: int, expires_at: datetime,
            content_cookie_expires_at: datetime }
Errors:   401 (invalid_credentials | account_inactive)
```

#### `POST /v1/auth/refresh`

```
Request:  { refresh_token: string }
Response: { access_token, refresh_token, token_type: "bearer", expires_in: int, expires_at: datetime,
            content_cookie_expires_at: datetime }
Errors:   401 (token revoked/expired/invalid | token_reuse_detected | account_inactive)
Note:     A `token_reuse_detected` 401 (a revoked refresh token replayed) is one of the five
          bulk-revocation sites — it also deletes every Web Push subscription the user has,
          same as logout-all, so a confirmed theft signal doesn't leave the thief's device
          subscribed.
```

#### `POST /v1/auth/logout`

```
Request:  { refresh_token: string }
Response: { message: string }
Status:   Always 200, even if refresh_token is unknown, malformed-but-syntactically-valid, or
          already revoked — a deliberate, permanent contract (not incidental behavior). A client
          that discovers its session was revoked remotely (logout-all/password change/reset
          elsewhere) has no way to clear this origin's HTTP cache itself, so it fires a
          best-effort logout purely to receive the Clear-Site-Data header below. The uniform 200
          is also the correct privacy posture — it never reveals whether a given refresh token
          was ever valid.
Headers:  Clear-Site-Data: "cache", "storage" — purges this origin's HTTP cache/storage on the
          calling device. Shared across all session-ending endpoints (see the changelog entry
          above); "executionContexts" is deliberately omitted (would force a page reload).
Note:     Revokes the specific refresh token. If an Authorization: Bearer header is also present
          (optional — this route isn't guarded, since the access token may already be expired),
          its jti is denylisted for its remaining lifetime (issue #142), so that specific access
          token 401s on its next use while any other device's token is unaffected.
Limitation: this device's apex_content cookie is cleared client-side only — it is scoped
          Path=/v1/content (never sent to this endpoint) and carries a different jti than the
          access token, so it cannot be revoked server-side here. A stolen cookie survives
          single-device logout until it expires (up to CONTENT_COOKIE_TTL_HOURS) or until a
          bulk-revocation event (logout-all, password change/reset, deactivation). Users who
          suspect theft should use logout-all or change/reset their password, not rely on
          single-device logout.
Note:     Does NOT delete the caller's Web Push subscription server-side — deliberately, since
          this ends only one device's session and other devices must keep receiving push
          notifications. The client is expected to unsubscribe its own endpoint locally
          (DELETE /v1/push/subscriptions, §15b) before calling this. Contrast with the five
          bulk-revocation endpoints (logout-all, password change/reset, deactivation, and
          refresh-token reuse detection), which delete every subscription the user has.
```

#### `POST /v1/auth/verify-email`

```
Request:  { token: string }  // from email link query param (20-100 chars)
Response: { message: string }
Errors:   400 (invalid_token | expired)
```

#### `POST /v1/auth/forgot-password`

```
Request:  { email: string }
Response: { message: string }  // always 200 (doesn't reveal if email exists)
Rate:     3/hour
```

#### `POST /v1/auth/reset-password`

```
Request:  { token: string (20-100 chars), new_password: string (8-128 chars) }
Response: { message: string }
Errors:   400 (invalid_token | expired)
Headers:  (200 only) Clear-Site-Data: "cache", "storage" — the calling device ends its own
          session here too, and this is the compromised-account recovery path.
Note:     One of the five bulk-revocation sites — also deletes every Web Push subscription the
          user has (§15b), same as logout-all. Best-effort: a failure here never blocks the
          password reset itself from succeeding.
```

#### `POST /v1/auth/resend-verification` *(authenticated)*

```
Response: { message: string }
Rate:     3/hour
```

#### `POST /v1/auth/content-cookie` *(authenticated, Bearer-only)*

```
Response: { expires_at: datetime }   // ContentCookieResponse — UTC, ISO-8601
Status:   200 OK
Errors:   401 (missing/invalid/expired Bearer token)
Rate:     20/minute
Note:     Re-mints the apex_content cookie (same attributes login/register/refresh set —
          see §9) for the caller's user_id + the current request's product, without
          rotating the refresh token. `expires_at` is the cookie's absolute expiry —
          derived from the same helper as the Set-Cookie `Max-Age`, so the two can never
          diverge. With the cookie's TTL now measured in days (§9), this endpoint's role
          has shifted: it's no longer mainly about dodging a refresh-token rotation, but is
          the *proactive* keep-alive path — clients should schedule their next re-mint from
          `expires_at` (or from TokenResponse.content_cookie_expires_at) well before the
          cookie actually lapses, rather than waiting for a 401. The content cookie itself
          does NOT authorize this endpoint — only a valid Bearer access token does.
```

---

## 3. User Profile

All endpoints below require `Authorization: Bearer <access_token>`.

#### `GET /v1/users/me`

```
Response: {
  id: UUID,
  email: string,
  display_name: string | null,
  subscription_tier: SubscriptionTier,
  locale: SupportedLocale,
  role: UserRole,
  is_active: bool,
  created_at: datetime,
  updated_at: datetime,
  age_verified: bool,                  // true once the user has passed the age gate
  age_verified_at: datetime | null,    // timestamp of first successful verification; null if never
  date_of_birth: date | null           // stored only for DATE_OF_BIRTH-policy products; else null
}
```

#### `PATCH /v1/users/me`

```
Request:  {
  display_name?: string | null,
  email?: string,
  locale?: SupportedLocale,
  age_confirmed?: bool,        // "I am 18+" checkbox — for CHECKBOX-policy products
  date_of_birth?: date         // for DATE_OF_BIRTH-policy products (e.g. vex)
}
Response: same as GET /v1/users/me
Errors:   400 (email_exists | validation_error)
Note:     Age capture is policy-driven by the active product's age_gate (see GET /v1/auth/product-info):
            • Omitting both age fields is a no-op — ordinary profile edits never touch age state.
            • CHECKBOX product: age_confirmed=true sets age_verified_at=now(); age_confirmed=false → 400 validation_error.
            • DATE_OF_BIRTH product (vex): a DOB computing to ≥18 verifies; <18 → 400 validation_error.
              DOB is captured at point of use — never at registration — when POST /v1/generate returns 403 age_verification_required.
          Verification is monotonic: once age_verified_at is set it is never cleared, and re-confirming
          is an idempotent 200 (the original timestamp is preserved).
          date_of_birth is write-once: submitting a different value once one is stored → 400 validation_error
          (an identical value is a no-op).
          This is the capture path for the per-model age gate enforced at POST /v1/generate (§4).
```

#### `POST /v1/users/me/password`

```
Request:  { current_password: string, new_password: string }
Response: { message: string }
Errors:   400 invalid_password
Headers:  (200 only) Clear-Site-Data: "cache", "storage" — the caller's own session ends here too.
Note:     Revokes ALL refresh tokens, plus all live access tokens and the content cookie
          (issue #142) — the most security-sensitive of the three bulk-revocation sites,
          since a password change is often a reaction to suspected compromise. Also deletes
          every Web Push subscription the user has (§15b) — best-effort, never blocks the
          password change itself from succeeding.
```

#### `DELETE /v1/users/me`

```
Response: { message: string, deactivated_at: datetime }
Headers:  Clear-Site-Data: "cache", "storage" — the caller's own session ends here too.
Note:     Soft delete — account can be recovered. Revokes ALL refresh tokens, plus all live
          access tokens and the content cookie (issue #142). Also deletes every Web Push
          subscription the user has (§15b) — best-effort, never blocks deactivation itself
          from succeeding.
```

#### `GET /v1/users/me/stats`

```
Response: {
  total_jobs: int,
  completed_jobs: int,
  failed_jobs: int,
  total_outputs: int,
  total_uploads: int,
  storage_used_bytes: int
}
```

> **Deprecated:** `GET /v1/users/me/jobs` has been removed.
> Use `GET /v1/jobs` (Section 6) which provides the same data plus filtering
> by `status`, `provider`, and `generation_type`, and the full output list per job.

#### `POST /v1/users/me/logout-all`

```
Response: { message: string }
Headers:  Clear-Site-Data: "cache", "storage" — the calling device ends its own session here too.
          Other devices whose sessions this call ends never receive this response at all — they
          discover the revocation via a 401 on their next request.
Note:     Revokes ALL refresh tokens, plus all live access tokens and the content cookie
          (issue #142) — the access token used to make this very call also stops working
          from the next request onward. See the module docstring on TokenRevocationService
          (src/api/services/token_revocation.py) for the Redis-backed epoch mechanism and its
          fail-open posture when Redis is unset or transiently unavailable. Also deletes every
          Web Push subscription the user has (§15b) — best-effort, never blocks logout-all
          itself from succeeding.
```

---

## 4. Generation *(authenticated)*

### Unified Endpoint (primary)

#### `POST /v1/generate`

Single endpoint for all generation types and providers.

```
Request: {
  prompt: string (1–4096 chars),
  generation_type: GenerationType,
  model: ModelType,
  input_image_id?: UUID,          // required for i2i / i2v / flf2v if source_output_id not set
  source_output_id?: UUID,        // alternative to input_image_id — use an existing generation output as input
                                  // mutually exclusive with input_image_id
  source_images?: Array<{         // Grok I2I multi-reference inputs (1–4 items); backend resolves refs to provider URLs
    input_image_id?: UUID,        // exactly one of input_image_id or source_output_id per item
    source_output_id?: UUID
  }>,                             // mutually exclusive with top-level input_image_id/source_output_id
  input_video_url?: string,       // required for v2v (public URL)
  negative_prompt?: string (≤2048 chars),  // applied by Aisha; stored but ignored by Grok
  aspect_ratio?: AspectRatio | null,  // omit/null ⇒ provider default for t2i (1:1 image, 16:9 video);
                                      //   for i2i, omit/null ⇒ output follows the source image's aspect.
                                      //   For i2i, an explicit value is only accepted when the model's
                                      //   edit_aspect_ratios (see GET /v1/providers, §5) includes it —
                                      //   otherwise 400 validation_error (the model would silently stretch
                                      //   the source to fit instead of recomposing it).
  n?: int (1–10, default 1),      // number of outputs; clamped to model max (see ModelType.max_images)
  name?: string,                  // auto-generated from prompt[:50] if omitted

  // --- Video-only (ignored for image generation) ---
  duration?: int (1–15, default 5),
  resolution?: VideoResolution (default "720p"),

  // --- Aisha image sizing (image models only). Tier XOR explicit width+height. ---
  image_resolution?: Resolution,  // quality tier: "draft" | "standard" | "high" | "ultra"
                                  //   → maps to a target megapixel budget (see §17 Resolution)
                                  //   mutually exclusive with width+height; omit both ⇒ model's default tier
  width?: int (256–4096),         // explicit width; MUST be paired with height
  height?: int (256–4096),        // explicit height; MUST be paired with width

  // --- Aisha sampler overrides (image models only). Omit ⇒ per-model bundle default. ---
  seed?: int,                     // reproducibility seed; auto-generated if omitted
  steps?: int (1–150),            // inference steps; bundle clamps per-model max
  cfg?: float (0.0–30.0),         // CFG scale
  sampler?: Sampler,              // ComfyUI sampler name (see §17 Sampler)
  scheduler?: Scheduler,          // ComfyUI scheduler name (see §17 Scheduler)
  denoise?: float (0.0–1.0)       // denoise strength
}
Response: JobCreatedResponse
Status:   201 Created
Errors:   400 (model_disabled | validation_error | generation_failed | not_implemented | provider_invalid_request), 402 insufficient_balance, 403 (model_not_allowed | age_verification_required), 409 (idempotency_conflict | no_active_gpu_session), 422 provider_moderation_rejected, 429 (rate_limited | provider_rate_limited), 502 (provider_malformed_response | provider_output_not_delivered), 503 (service_unavailable | provider_timeout | provider_unavailable | provider_authentication_failed | provider_unknown)
Headers:  Idempotency-Key: <string> (required, max 64 chars)
Note:     source_output_id enables "remix from Library" — the backend resolves lineage automatically
          (source_job_id + source_output_id) and records it on the new job.
          source_images is storage-reference based (1–4 items); clients send upload/output IDs, not public URLs.
          If source_images contains output references and no top-level source_output_id is set,
          lineage is recorded from the first output-typed item in list order.
          Tokens charged scale as (token_cost + input_token_cost × k) × n, where k is the
          input-image count: 0 for text-to-image, 1 for input_image_id/source_output_id, or
          source_images.length for multi-reference image inputs.
          Idempotency-Key prevents duplicate jobs on network retries — supply a UUIDv4 per submission attempt.
          Aisha (ComfyUI) models require an active GPU session — start one via
          POST /v1/sessions before submitting an Aisha generation, otherwise 409 no_active_gpu_session is returned.

          Aisha sizing rules (enforced server-side; violations return 400 validation_error):
            • width and height must be supplied together (one without the other is rejected).
            • image_resolution and explicit width/height are mutually exclusive.
            • Omitting all three uses the model's default tier (aisha-image default: "standard").
          The Aisha sampler/sizing fields above are accepted only for Aisha image models; for Grok
          models they are ignored (Grok determines sizing/sampling server-side).

          Age-verification gate: models with requires_age_verification=true (currently aisha-image and
          aisha-video — see GET /v1/providers) require the user to be age-verified. If age_verified_at
          is not set, the request returns 403 age_verification_required and no job is created / no tokens
          are charged. Capture verification first via PATCH /v1/users/me (§3). The gate is per-model and
          authoritative regardless of the product's age_gate policy.

          Aspect-ratio capability gate: t2i requests validate aspect_ratio against the model's
          aspect_ratios list (GET /v1/providers). i2i requests validate against the model's
          edit_aspect_ratios list instead — a separate, usually smaller/empty capability, since
          reshaping the *output* canvas of an edit is a different (and not universally supported)
          operation from generating a fresh canvas from scratch. An unsupported value on either path
          returns 400 validation_error with an actionable message (e.g. "omit aspect_ratio to preserve
          the source aspect"); no job is created and no tokens are charged.

          Provider moderation: a synchronous Grok safety rejection returns 422
          provider_moderation_rejected with a safe actionable message. It differs from Apex's own
          pre-submission moderation (`moderation`): Grok already accepted and production-observed
          billing applies to this rejected generation, so the reservation remains spent. Re-use the
          same Idempotency-Key to replay this 422 safely; do not submit the same intent again.

          Grok video is asynchronous. Both its worker and poll-on-read `GET /v1/jobs/{id}` use the
          same terminal settlement path.  On completion the first caller takes a short PostgreSQL
          finalization lease before downloading or writing R2 objects; it then commits the output rows
          and completed status together.  A crashed claimant is recoverable after the lease expires;
          losing callers write no outputs and remove any objects if their claim expires mid-flight.
          Partial unique indexes enforce one full output per job/index and one thumbnail per
          parent/size bucket across both providers. A worker is optional for correctness and
          recommended for proactive completion. Repeated GETs and worker/read-through races publish
          one terminal event and settle billing once.

          A deferred xAI `FAILED` state is authoritative even if its normalized failure kind is
          `provider_rate_limited` or `provider_timeout`; it is settled immediately. By contrast, a
          transport-level rate limit/timeout while calling xAI's poll API is transient and leaves the
          job in progress for the next poll.

          The active `GROK_MODERATION_BILLING_POLICY` controls an accepted moderation rejection:
          `charge` retains the reservation and `refund` compensates it. Only exact moderation results
          or call sites with a confirmed accepted deferred request can use that policy; ambiguous or
          moderation-service infrastructure failures are non-billable. Other normalized provider failures refund once unless an explicit provider policy
          says otherwise. Synchronous provider failures are
          returned with stable normalized codes: invalid request 400, provider rate limit 429,
          malformed response 502, and timeout/unavailable/authentication/unknown 503. Their messages
          are fixed safe messages, never provider diagnostics.

          A failed billable generation and its refund are one database transaction. If refund creation
          fails, the terminal failure rolls back, the job stays in flight, and a later poll or sweep
          retries settlement; workers isolate that error to the affected job.

          Aisha terminal failures use public-safe codes and fixed text: infrastructure failures use
          `provider_unavailable` / "Generation infrastructure is temporarily unavailable.", ComfyUI
          execution failures use `provider_execution_failed` / "The generation engine could not
          complete the request.", timeouts use `provider_timeout` / "Generation timed out before the
          compute service returned a result.", and swept compute sessions use
          `generation_session_terminated` / "Generation stopped because the compute session ended."
          Raw Aisha, ComfyUI, and session diagnostics remain internal only.
```

### JobCreatedResponse Schema

```
{
  job_id: UUID,
  status: JobStatus,
  name: string,
  model: ModelType,
  generation_type: GenerationType,
  created_at: datetime,
  message?: string,
  tokens_charged?: int,
  balance_remaining?: int
}
```

---

## 5. Providers *(auth-optional)*

#### `GET /v1/providers`

Auth-optional: unauthenticated callers get the full capabilities catalog; authenticated callers additionally receive `user_context` with their subscription tier.

Models are **filtered by the current product** — Synthara only returns SFW-safe models; Vex returns all enabled models.

```
Response: {
  providers: ProviderInfo[],
  user_context: UserContext | null   // null when unauthenticated
}

ProviderInfo: {
  provider: string,             // e.g. "aisha", "grok"
  name: string,                 // e.g. "Aisha", "xAI Grok"
  available: bool,              // true when the provider is fully configured and able to serve requests
                                // (registry membership — e.g. Grok requires both XAI_API_KEY and R2)
  provisioning_mode: string,    // "always_on" — cloud API, usable immediately when available
                                // "on_demand" — requires a per-user GPU session (POST /v1/sessions)
  models: ModelInfo[]
}

ModelInfo: {
  model_key: string,                 // matches ModelType value
  name: string,
  description: string,
  capabilities: string[],            // e.g. ["t2i", "i2i"]
  is_enabled: bool,
  max_images: int,                   // max outputs per request
  max_prompt_length: int,
  supports_negative_prompt: bool,
  aspect_ratios: string[],           // e.g. ["1:1", "16:9"] — t2i only; see ImageConstraints.edit_aspect_ratios for i2i
  requires_age_verification: bool,   // true ⇒ user must be age-verified (PATCH /v1/users/me) before
                                     //   generating; enforced at POST /v1/generate. Collect the 18+
                                     //   confirmation before starting a (billable) GPU session.
  image: ImageConstraints | null,    // null for video-only models
  video: VideoConstraints | null,    // null for image-only models
  session_state: string | null       // per-user readiness; only populated for authenticated requests
                                     //   on on_demand provider models. Values: ModelSessionState.
                                     //   null when unauthenticated or for always_on providers.
                                     //   "none" → no live session; start one via POST /v1/sessions.
                                     //   "active" → ready to generate; submit POST /v1/generate.
}

ImageConstraints: {
  min_height: int | null,            // null = not user-controllable
  max_height: int | null,
  default_height: int | null,
  output_resolutions: string[] | null,  // informational; null = backend-determined
  supported_tiers: string[] | null,  // image quality tiers, e.g. ["draft","standard","high","ultra"]
                                      // null for models with fixed sizing (e.g. Grok)
  default_tier: string | null,       // default quality tier; null for fixed-sizing models
  tier_megapixels: { [tier: string]: number } | null,  // target megapixel budget per tier
                                      // actual W×H depends on model + aspect ratio
  edit_aspect_ratios: string[]       // aspect ratios this model can reshape TO during image editing
                                      // (i2i). Empty array ⇒ the model cannot reshape on edit — clients
                                      // must omit aspect_ratio on i2i requests for this model, and the
                                      // output follows the source image's own aspect. e.g. grok-imagine-image
                                      // → [] (accepts the param on edits but stretches instead of
                                      // recomposing); aisha-image → full AspectRatio list (recomposes
                                      // natively onto the requested canvas).
}

VideoConstraints: {
  max_duration: int,                 // maximum video duration in seconds
  resolutions: string[]              // e.g. ["480p", "720p"]
}

UserContext: {
  subscription_tier: string          // e.g. "free", "pro"
}
```

> **Deprecated flat format** (`providers` + `models` as a flat list) was removed in v2.

---

### Rate Limit Errors

The unified generation endpoint may return `429 Too Many Requests` when a model's global rate limit is exceeded:

```
Status:  429 Too Many Requests
Headers: Retry-After: <seconds>
Body: {
  error: "rate_limited",
  message: "Rate limit exceeded for model '...' ...",
  status_code: 429,
  detail: { retry_after: int }
}
```

---

## 5b. Shared Media Types

### `MediaObject`

All image- and video-bearing responses use a unified `MediaObject` envelope. URLs are stable content-proxy paths — never presigned and never expiring within the resource's lifetime.

```typescript
interface MediaObject {
  media_type: "image" | "video";
  original: MediaOriginal;
  variants: ImageVariant[];  // preview rasters, ascending by width; always present, may be empty
}

interface MediaOriginal {
  url: string;          // "/v1/content/outputs/{id}" or "/v1/content/uploads/{id}"
  width: number | null;
  height: number | null;
  content_type: string; // "image/png", "image/jpeg", "image/webp", "video/mp4", etc.
  size_bytes: number;
}

interface ImageVariant {
  label: string;   // "sm" (150px longest edge) or "md" (512px longest edge)
  width: number;   // actual pixel width of this variant (always non-null)
  height: number;  // actual pixel height of this variant (always non-null)
  url: string;     // "/v1/content/outputs/{id}" or "/v1/content/uploads/{id}"
}
```

**Variant labels:**

| `label` | Max longest edge | Content type | Use case |
|---------|-----------------|--------------|----------|
| `sm` | 150 px | `image/webp` | Thumbnails, grid cells |
| `md` | 512 px | `image/webp` | Preview, lightbox |

For **video** outputs: `original.url` is the MP4 source; `variants` are poster-frame rasters (extracted first frame → sm + md WEBP).

For **image** outputs and **uploads**: `original.url` is the full-resolution source; `variants` are downscaled WEBP previews.

---

## 6. Jobs *(authenticated)*

#### `GET /v1/jobs`

```
Query:    status?, provider?, generation_type?, limit? (default 20), cursor? (opaque token)
Response: CursorPage<UnifiedJobResponse>
```

#### `GET /v1/jobs/{job_id}`

```
Response: UnifiedJobResponse
Errors:   404
Note:     For queued/running Grok video jobs this is poll-on-read. It also enforces
          GROK_VIDEO_MAX_POLL_TIME from the immutable submission timestamp: an overdue job is
          failed with provider_timeout and refunded once. Terminal provider outcomes use the same
          settlement path as the Grok video worker, so no background worker is required for
          correctness.
```

#### `DELETE /v1/jobs/{job_id}`

```
Response: 204 No Content
Note:     Soft-hide from history
```

### UnifiedJobResponse Schema

```typescript
interface UnifiedJobResponse {
  id: string;               // UUID
  name: string;
  status: JobStatus;
  provider: "grok" | "aisha";
  model: string | null;
  generation_type: GenerationType;
  prompt: string;
  negative_prompt: string | null;
  aspect_ratio: string | null;  // null ⇒ i2i generation that followed the source image's aspect
                                 // (no explicit aspect_ratio was requested/capability-approved)
  token_cost: number | null;
  created_at: string;       // ISO datetime
  started_at: string | null;
  completed_at: string | null;
  outputs: JobOutputItem[]; // empty while processing
  error: string | null;       // public-safe failure text only
  failure_code: string | null; // stable code, e.g. "provider_moderation_rejected"
}

interface JobOutputItem {
  id: string;               // UUID
  output_index: number;     // 0-based position within the batch
  media: MediaObject;       // original asset + sm/md WEBP preview variants
}
```

> **Breaking change (2026-06-27):** `JobOutputItem` no longer carries `url`, `content_type`,
> `format`, `size_bytes`, `thumbnail_url`, or `is_thumbnail`. The full media envelope
> (`original` + `variants`) is in `media: MediaObject`. Jobs API URLs are now stable
> content-proxy paths — **no presigned URLs**. `UnifiedJobResponse.thumbnail_url` is removed.

`error` is populated only from the public-safe failure-message boundary. The legacy/internal
`GenerationJob.error_message` column is never returned by this endpoint. A legacy failed row that
does not yet have a public-safe message returns `"Generation failed. Please try again."`; it never
falls back to raw diagnostics.

Grok failure normalization checks an explicit provider/gRPC code first, then a numeric HTTP status,
then only narrow prose markers (`too many requests`, standalone `429`, `invalid image URL` or
`invalid image input`, and `bad request`). Other wording is classified as unknown rather than by
broad terms such as `invalid`, `rate`, `timeout`, or `connection`.

Materialization attempts and storage-cleanup records are scoped to a product. The retention worker
reconciles each configured product independently; reconciliation, output lookup, and cleanup-outbox
queries always include that product scope.

---

## 7. GPU Sessions *(authenticated)*

Aisha (ComfyUI) models run on per-user, per-model GPU instances provisioned on Vast.ai. Before submitting an Aisha generation, the user must start a session for that model; sessions are billed by uptime (active + minimum-session floor) and can be paused (storage-only) or stopped (full teardown).

> **Grok models do not require a GPU session** — they run on xAI's hosted API and are billed per-generation.

### Lifecycle

```
            ┌─────────┐
            │ pending │ — request accepted, Vast.ai not yet provisioning
            └────┬────┘
                 ▼
         ┌──────────────┐
         │ provisioning │ — instance starting, ComfyUI not yet reachable
         └──────┬───────┘
                ▼
            ┌────────┐                     ┌────────┐
            │ active │ ◄─── resuming ◄──── │ paused │
            └───┬────┘                     └────────┘
                │                              ▲
                │   ┌─────────────────────┐    │
                ├──►│ stale (unreachable) │────┘ (auto-recovers if probe succeeds)
                │   └─────────────────────┘
                ▼
           ┌──────────┐
           │ stopping │ — teardown in progress
           └────┬─────┘
                ▼
           ┌─────────┐                     ┌────────┐
           │ stopped │                     │ failed │ (terminal — provisioning error)
           └─────────┘                     └────────┘
```

Lifecycle events are pushed in real time via SSE — see [§15 Real-Time Events](#15-real-time-events-sse--pubsub) (`gpu_session.status_changed`).

### Schemas

```typescript
GpuSessionResponse: {
  id: UUID,
  user_id: UUID,
  product_id: string,                          // "vex" | "synthara"
  status: GpuSessionStatus,                    // see Enums
  model_type: ModelType,                       // e.g. "aisha-image"
  tunnel_hostname: string | null,              // Cloudflare tunnel — set once active
  vastai_gpu_name: string | null,              // e.g. "RTX 4090"
  vastai_cost_per_hour_micros: int | null,     // cost in millionths of USD per hour
  created_at: datetime,
  started_at: datetime | null,                 // first transition to active
  paused_at: datetime | null,
  resumed_at: datetime | null,
  stopped_at: datetime | null,
  error_message: string | null,                // populated when status == "failed"
  in_flight_job_count: int,                    // queued+running Aisha jobs on this session
                                               // — non-zero only for active sessions; 0 otherwise
  provisioning_phase: string | null,           // latest phase reported via the node callback
                                               // (e.g. "downloading", "ready"); null before first callback
  provisioning_progress: object | null         // latest progress blob from the node callback
                                               // (download bytes/files, message, etc.); null before first callback
}

ListSessionsResponse: { sessions: GpuSessionResponse[] }

StopConfirmationResponse: {
  session_id: UUID,
  model_type: ModelType,
  vastai_gpu_name: string | null,
  vastai_cost_per_hour_micros: int | null,
  active_duration_seconds: int,
  paused_duration_seconds: int,                // billed at storage rate (future)
  estimated_final_tokens: int,                 // total token cost if stopped now
  message: string
}
```

### Endpoints

#### `POST /v1/sessions/`

Start a new GPU session for a model. The session begins in `pending` and transitions to `provisioning` → `active` over ~30–90 seconds. Only one active session per `(user_id, product_id, model_type)` is allowed.

```
Request: {
  model: ModelType,            // "aisha-image" | "aisha-video"
  bundle_override?: string     // ADMIN-only — pin a specific bundle "name" or "name:version".
                               // Non-admins receive 403; ignored otherwise.
}
Response: GpuSessionResponse
Status:   201 Created
Errors:   402 insufficient_balance,
          403 (bundle_override is admin-only),
          409 session_already_exists (active session for same user+product+model),
          503 (no_gpu_capacity | provisioning_failed)
Note:     `in_flight_job_count` is always 0 in the creation response.
```

#### `GET /v1/sessions/`

List the current user's sessions for the active product.

```
Query:    include_terminal? (bool, default false)  — include stopped/failed sessions
Response: ListSessionsResponse
Note:     Sessions are scoped to the resolved product. Default behavior is "show only live" — pass
          include_terminal=true for history views.
```

#### `GET /v1/sessions/{session_id}`

Fetch a single session, including the up-to-date `in_flight_job_count` for active sessions.

```
Response: GpuSessionResponse
Errors:   404 (session not found or not owned by current user)
Note:     `in_flight_job_count` reflects the live count of QUEUED+RUNNING Aisha jobs on this session.
          Frontends should disable the Pause button when `in_flight_job_count > 0`.
          For non-active statuses the value is always 0 (sweeps run on transition out of active).
```

#### `POST /v1/sessions/{session_id}/pause`

Pause an active session — Vast.ai instance is stopped, persistent disk is retained. Refused if any in-flight jobs exist; the user must wait for them to complete or cancel them first.

```
Response: GpuSessionResponse                    (status -> "paused")
Errors:   404 session_not_found,
          409 invalid_state (session is not in 'active' status),
          409 jobs_in_flight (one or more queued/running jobs)

// 409 jobs_in_flight error body:
{
  "error": "jobs_in_flight",
  "message": "Cannot pause: 2 in-flight job(s) on this session",
  "status_code": 409,
  "detail": { "in_flight_count": 2 }
}
```

#### `POST /v1/sessions/{session_id}/resume`

Resume a paused session — restart the Vast.ai instance with retained disk. Transitions through `resuming` → `active`.

```
Response: GpuSessionResponse                    (status -> "resuming" then "active" via SSE)
Errors:   404 session_not_found,
          409 invalid_state (session is not in 'paused' status)
```

#### `POST /v1/sessions/{session_id}/stop`

Two-call stop flow — the first call returns a cost confirmation, the second call (with `confirmed: true`) executes the stop. This prevents accidental teardown and lets the user see the final billable amount before committing.

```
Request: { confirmed: bool }    // false (or omitted) -> dry-run / preview;
                                // true -> execute teardown

// First call (confirmed: false) — preview:
Response: StopConfirmationResponse
Status:   200 OK

// Second call (confirmed: true) — execute:
Response: GpuSessionResponse                    (status -> "stopping" then "stopped")
Status:   200 OK

Errors:   404 session_not_found,
          409 invalid_state (session is in a terminal status)

Note:     The two-call pattern is stateless on the server — the first call does NOT lock state.
          The frontend should display StopConfirmationResponse, then issue the second call
          with `confirmed: true` to actually stop.
```

### Billing & Credit Guard

GPU sessions are billed by uptime. All debit transactions carry a `metadata.type` field identifying the charge kind:

| `metadata.type` | When created |
|-----------------|-------------|
| `generation` | Aisha or Grok generation charged at submission |
| `gpu_session_reservation` | Base reservation debit when session starts |
| `gpu_session_metered` | Per-cycle metered debit from `SessionCreditGuard` (clamped to balance) |
| `gpu_session_overage` | Finalization overage — additional tokens owed at session stop |

**Credit guard cycle** (`SessionCreditGuard`, runs on the health-snapshot worker cadence):

1. For each `active` or `stale` session, settle a metered debit equal to `ceil(interval_min × rate)` tokens — clamped to the current balance (never goes negative).
2. Compute a **floor** = `ceil(interval_min × rate × safety_factor)` (default safety factor 1.5). Sessions with `balance ≤ floor` are auto-terminated.
3. Classify the warning level:
   - `balance ≤ floor` → terminate immediately (emits `critical` warning then stops)
   - `balance ≤ critical_threshold` (default 10-min runway) → `critical`
   - `balance ≤ warning_threshold` (default 20-min runway) → `warning`
   - otherwise → no warning
4. Emit `gpu_session.credit_warning` SSE events **once per upward transition** (de-escalates when balance recovers, e.g. after a top-up).

**Finalization** (runs after the session reaches `stopped`): computes total billable minutes, compares against total settled tokens (base reservation + all metered debits), then either settles any remaining overage or issues a partial refund — no debt invariant holds throughout.

### Frontend Usage Pattern

```typescript
// 1. Start a session for an Aisha model:
const session = await api.post<GpuSessionResponse>('/v1/sessions/', {
  model: 'aisha-image',
});

// 2. Subscribe to gpu_session.status_changed via SSE.
//    Wait for status === 'active' before submitting generations.

// 3. Submit generations against /v1/generate (the backend routes Aisha
//    requests to the user's active session for that model automatically).

// 4. Pause / resume / stop as needed:
await api.post(`/v1/sessions/${session.id}/pause`);
await api.post(`/v1/sessions/${session.id}/resume`);

// Two-call stop:
const preview = await api.post<StopConfirmationResponse>(
  `/v1/sessions/${session.id}/stop`,
  { confirmed: false },
);
// ...show preview to user, get confirmation...
await api.post(`/v1/sessions/${session.id}/stop`, { confirmed: true });
```

### Internal — GPU node provisioning callback *(node-to-backend; not for frontend use)*

GPU nodes (Aisha's `ProvisioningReporter`) push provisioning progress to the backend over this internal endpoint. It is **not** part of the authenticated frontend surface and is documented here only for completeness — the frontend observes provisioning progress via the `provisioning_phase` / `provisioning_progress` fields on `GpuSessionResponse` (poll `GET /v1/sessions/{id}` or react to the `gpu_session.status_changed` SSE event).

#### `POST /v1/internal/gpu-sessions/{session_id}/provisioning`

```
Auth:    Authorization: Bearer <node callback token>
         The controller has NO JWT guard. The presented token is validated in-handler by
         comparing its SHA-256 hash against the session's stored callback_token_hash
         (constant-time compare). Each session has its own single-purpose callback token.

Request: {
  session_id: UUID,           // MUST equal the {session_id} path param
  phase: string,              // e.g. "starting", "downloading", "ready", "failed"
  message?: string,           // human-readable status (default "")
  download?: {                // present during the "downloading" phase
    bytes_done: int,
    bytes_total: int,
    files_done: int,
    files_total: int
  },
  elapsed_seconds?: int,      // default 0
  error?: string | null,      // populated on failure
  ts: datetime                // event timestamp (used for stale-callback rejection)
}

Response: { ok: true }
Status:   200 OK   — for ALL non-auth outcomes, including status-gated and stale-ts no-ops,
                     so the node never retries on a benign race.
Errors:   401 unauthorized (missing / empty / invalid Bearer token),
          400 bad_request   (body session_id does not match the path session_id)
```

---

## 8. Storage *(authenticated)*

> **Library is the primary read surface.** `GET /v1/library/` (§10) supersedes both the removed
> `GET /v1/storage/uploads` list and, for most UI purposes, `GET /v1/storage/outputs` below —
> it's the single paginated grid over uploads + outputs with favorites/projects/tags/filters.
> The endpoints in this section remain for upload creation, single-item presigned access, raw
> byte download, and storage stats — none of that is replaced by Library.

> **Retention:** every upload and output row carries `expires_at`, set at creation to `now +
> RETENTION_DAYS` (default 7 days). A periodic background sweeper (`ContentRetentionWorker`)
> deletes expired rows and their R2 objects on a fixed interval. Once swept: the item drops out
> of Storage/Library list responses, and `GET /v1/content/...` (§9) for that ID returns `404`.
> `expires_at` is a plain timestamp (not a countdown) so the frontend can derive and tick a
> "Delete in N days/hours/minutes" badge client-side — see `ImageListItem`/`OutputListItem`
> below and `LibraryAssetItem` (§10).

### Uploads

#### `POST /v1/storage/upload`

```
Request:  multipart/form-data, field "data" (max 20MB)
          Images:  PNG, JPEG, WebP, HEIC/HEIF, AVIF — non-PNG/JPEG/WebP inputs
                    are converted to PNG.
          Videos:  MP4, WebM, QuickTime (.mov) — stored as-is, never re-encoded.
Response: {
  id: UUID,
  filename: string,
  created_at: datetime,
  expires_at: datetime,
  media: MediaObject    // original + sm/md WEBP variants (generated synchronously)
}
Status:   201 Created
Errors:   400 (invalid_file_type | file_too_large | empty_file | validation_error)
Note:     Returns image id used for I2I/I2V generation requests, or (for
          videos) as source_upload_id on POST /v1/frames/preview|extract (§9b).
          Thumbnail/poster generation is non-fatal; variants may be empty on failure.

          Videos are probed server-side (ffprobe) before acceptance — the
          declared Content-Type is never trusted. A validation_error 400 is
          returned if the bytes aren't a decodable video, or if duration
          exceeds the server's configured maximum (default 300s):
            { "error": "validation_error",
              "message": "File is not a decodable video", "status_code": 400 }
            { "error": "validation_error",
              "message": "Video duration 620.0s exceeds maximum 300s", "status_code": 400 }

          Video duration is not currently exposed on any response — poll a
          preview job (§9b) to learn frame timestamps within the clip.
```

> **Removed (2026-07-22):** `GET /v1/storage/uploads` (list, and its `ImageListItem` response
> schema) — use `GET /v1/library/?source=upload` (§10) instead.

#### `GET /v1/storage/uploads/{image_id}`

```
Query:    expires_in? (60–86400 seconds, default 3600)
Response: {
  id: UUID,
  storage_key: string,
  presigned_url: string,
  content_type: string,
  size_bytes: int,
  expires_in_seconds: int
}
Errors:   404 not_found
```

#### `GET /v1/storage/uploads/{image_id}/download`

```
Response: Raw bytes (with appropriate Content-Type header)
Errors:   404 not_found
```

### Outputs

#### `GET /v1/storage/outputs`

```
Query:    limit? (1–100, default 50), cursor? (opaque token)
Response: CursorPage<OutputListItem>

OutputListItem: {
  id: UUID,
  job_id: UUID,
  output_index: int,
  created_at: datetime,
  expires_at: datetime,
  media: MediaObject    // original + sm/md WEBP variants
}
```

#### `GET /v1/storage/outputs/{output_id}`

```
Query:    expires_in? (60–86400 seconds, default 3600)
Response: { id, storage_key, presigned_url, content_type, size_bytes, expires_in_seconds }
Errors:   404 not_found
```

#### `GET /v1/storage/outputs/{output_id}/download`

```
Response: Raw bytes (with appropriate Content-Type header)
Errors:   404 not_found
```

#### `GET /v1/storage/jobs/{job_id}/outputs`

```
Response: CursorPage<OutputListItem>  // has_more=false, no cursor (returns all outputs)
Errors:   404
```

### Statistics

#### `GET /v1/storage/stats`

```
Response: {
  upload_count: int,
  output_count: int,
  total_bytes: int,
  total_mb: float
}
```

---

## 9. Content Proxy *(authenticated)*

Provides stable, non-expiring authenticated URLs for user content. The server resolves ownership, checks product scoping, then streams bytes directly from R2. **No presigned URLs are exposed** — the client only ever sees `/v1/content/...` paths.

> **Why use this instead of presigned URLs?** Content proxy URLs are permanent (for the lifetime of the resource), cacheable with `Cache-Control: private, max-age=<ttl>, immutable`, and enforce per-request authorization. They are the preferred URL format for Library and any UI that persists content references.

> **Why not `Cache-Control: private, no-store`?** Raised and declined (2026-07-27): `no-store` would re-fetch every thumbnail on every library grid render/page switch, reproducing a prior mobile bug (parallel-request saturation causing blank thumbnails) and nullifying the video prewarm design's HTTP-cache reuse. The residue concern it was meant to address — private images surviving in a shared device's HTTP cache after an account switch — is instead closed by `Clear-Site-Data: "cache", "storage"` on every session-ending endpoint (§2, §3) plus client-side session isolation. See the content route's module docstring (`src/api/routes/content.py`) for the full reasoning.

### Auth: the `apex_content` cookie

Requests here accept either a Bearer access token or the `apex_content` cookie (`HttpOnly`, `Secure`, `SameSite=Lax`, `Path=/v1/content`) — see §2 for how it's minted/re-minted. Its lifetime is `content_cookie_ttl_hours` (default **24h**, configurable up to **168h**/7d) — raised from a 1h default specifically so the cookie survives a suspended PWA: with no API traffic there's no `/v1/auth/refresh` to re-attach it, so a short TTL ages out during suspension and the first batch of `<img>` requests on resume all 401 before any JSON call can trigger recovery. This is deliberately asymmetric with the 15-minute access token (§2.1) — the content token is `type: "content"` (structurally rejected by the access-token decoder), product-scoped, and every request here still performs the full ownership check below regardless of which credential was presented; its blast radius is read access to the bearer's own media on one product. As of issue #142, `content_auth_guard` also consults `TokenRevocationService`: `POST /v1/auth/logout` clears the cookie client-side *and* denylists a presenting access token's own jti, while `logout-all`/password-change/deactivation (§3) reject any token — access or content — issued before that event, closing the exposure window the 24h TTL raise opened.

### Response Headers

All successful (200/206) responses include:
- `Content-Type` — the stored R2 `ContentType` **only if it's on the inline-safe allowlist** (`image/png`, `image/jpeg`, `image/webp`, `video/mp4`, `video/webm`, `video/quicktime`); otherwise `application/octet-stream`
- `Content-Length` — bytes in *this* response body (the full object size on 200, the served range's length on 206)
- `Cache-Control: private, max-age=10800, immutable` — 3-hour client cache (default; configurable via `CONTENT_URL_TTL`)
- `ETag: "<content_id>"` — the output/upload UUID, for conditional requests
- `X-Content-Id: <content_id>` — same UUID, without quotes
- `X-Content-Type-Options: nosniff` — always present, blocks MIME-sniffing
- `Content-Disposition: inline` for inline-safe content types, `attachment` otherwise — a stored content-type outside the allowlist (e.g. `text/html`, `image/svg+xml`) is forced to download rather than rendered inline, even though it streams with a coerced `Content-Type`
- `Accept-Ranges: bytes` — advertised on every 200 and 206, so clients know range requests are supported before issuing one

#### Range requests (single range only)

Both endpoints below honor a `Range: bytes=<start>-<end>` request header — the mechanism `<video>`/`<audio>` elements use for seeking and resumable playback:
- A satisfiable range → `206 Partial Content`, body is just that byte slice, `Content-Range: bytes <start>-<end>/<size>`, `Content-Length` is the slice length. Open-ended (`bytes=500-`) and suffix (`bytes=-500`) forms are supported; an end beyond the object's size is clamped to the last byte rather than rejected.
- A range whose start is at or beyond the object's size → `416 Range Not Satisfiable`, `Content-Range: bytes */<size>`, no body.
- **Multipart ranges are out of scope** — a comma-separated `Range` header (multiple ranges in one request) is treated as if no `Range` header were sent: a normal full `200`.
- No `Range` header, or a malformed one → full body, `200 OK`.

#### Conditional GET

Both endpoints honor `If-None-Match` against the resource's `ETag`. A match (including the `*` wildcard) short-circuits to `304 Not Modified` with the `ETag` and `Cache-Control` headers and no body — this happens *after* the ownership/product check (a 304 never leaks whether foreign content exists) and *before* any R2 traffic.

#### `GET /v1/content/outputs/{output_id}`

```
Path:     output_id (UUID)
Headers:  Range?: bytes=<start>-<end>, If-None-Match?: "<etag>"
Response: 200 Raw bytes | 206 Partial Content | 304 Not Modified (no body)
Errors:   404 not_found (ownership check failed or wrong product),
          416 range_not_satisfiable (Range start at/beyond object size),
          502 upstream_error (R2 fetch failed)
Note:     Only returns outputs owned by the authenticated user and matching the current product.
```

#### `GET /v1/content/uploads/{image_id}`

```
Path:     image_id (UUID)
Headers:  Range?: bytes=<start>-<end>, If-None-Match?: "<etag>"
Response: 200 Raw bytes | 206 Partial Content | 304 Not Modified (no body)
Errors:   404 not_found (ownership check failed or wrong product),
          416 range_not_satisfiable (Range start at/beyond object size),
          502 upstream_error (R2 fetch failed)
Note:     Only returns uploads owned by the authenticated user and matching the current product.
```

> **Removed (2026-07-22):** `DELETE /v1/content/{content_id}` — deletion is now typed via
> `DELETE /v1/library/assets/{asset_ref}` (§10), which delegates to the same
> `ContentProxyService.delete_content` logic (R2 removal + DB record removal + lineage
> `SET NULL`) but resolves the target table from the `asset_ref` prefix instead of trying both.

---

## 9b. Video Frame Extraction *(authenticated)*

> Full contract: `docs/contracts/video-frame-extraction.md`.

Takes any video — a `GenerationOutput` (Grok T2V/I2V) or a user-uploaded video (§8, `video/*` content type) — and either previews it as a low-res frame strip or extracts full-resolution frames at chosen timestamps. **Free** — no token charge, no `Idempotency-Key` header on any endpoint below. Both endpoints return `202` immediately with a `job_id`; the actual ffmpeg work runs on a background worker (`FrameExtractionWorker`) — poll `GET /v1/frames/jobs/{job_id}` until `status` is `completed` or `failed`.

Exactly one of `source_output_id` / `source_upload_id` must be set on every request below (`400 invalid_source` otherwise); the resolved source must be owned by the caller, belong to the current product, and have a video content type (`400 not_a_video` / `404 not_found` otherwise).

#### `POST /v1/frames/preview`

```
Request:  {
  source_output_id?: UUID | null,   // exactly one of these two
  source_upload_id?: UUID | null,
  frame_count?: int                 // 2-60, default 12
}
Response: { job_id: UUID, status: "queued" }
Status:   202 Accepted
Errors:   400 invalid_source | not_a_video, 404 not_found
```

#### `POST /v1/frames/extract`

```
Request:  {
  source_output_id?: UUID | null,
  source_upload_id?: UUID | null,
  timestamps_ms: int[]              // 1-50 entries, each >= 0
}
Response: { job_id: UUID, status: "queued" }
Status:   202 Accepted
Errors:   400 invalid_source | not_a_video, 404 not_found
Note:     Whether each timestamp is within the video's actual duration is
          checked once the worker probes the file (after the job starts
          running) — an out-of-range timestamp fails the *job*
          (status=failed, precise error), not the request.
```

#### `GET /v1/frames/jobs/{job_id}`

```
Response: {
  job_id: UUID,
  kind: "preview" | "extract",
  status: "queued" | "running" | "completed" | "failed",
  created_at: datetime,
  started_at: datetime | null,
  finished_at: datetime | null,
  error: string | null,             // populated only when status=failed
  source: { type: "output" | "upload", id: UUID },
  preview?: {                       // present iff kind=preview AND status=completed
    frames: [ { index: int, timestamp_ms: int, url: string } ],
    expires_in_seconds: int
  },
  extracted?: {                     // present iff kind=extract AND status=completed
    frames: [ { timestamp_ms: int, upload_id: UUID, media: MediaObject } ]
  }
}
Errors:   404 not_found (job doesn't exist or isn't owned by the caller)
Note:     preview.frames[].url is a presigned R2 URL generated FRESH on every
          call — never persisted, never the same URL twice. expires_in_seconds
          is that response's TTL (default 3600s); re-poll for fresh URLs
          rather than caching. Preview frames live at a non-authenticated,
          top-level R2 prefix that expires via an R2 lifecycle rule (default
          2 days) — there is no /v1/content/... proxy indirection for them
          (by design: stateless, no DB rows).

          extracted.frames[].media is the same MediaObject as everything else
          (§5b) — its urls are stable /v1/content/uploads/{id} proxy paths,
          cacheable indefinitely, same as any other upload. Once an extract
          job completes its frames are ordinary uploads: same download (§8),
          same delete (DELETE /v1/library/assets/upload:{id}, §10), same
          retention/expiry. Deleting the source video does NOT delete
          frames already extracted from it.
```

---

## 10. Library *(authenticated)*

Library is the unified, asset-oriented replacement for the old Gallery + My Uploads split: one
paginated read model over `user_images` + `generation_outputs` (the tables themselves stay
separate — Library is a query-time UNION, not a new content table). Every asset is addressed by a
typed **asset reference**: `"upload:<uuid>"` or `"output:<uuid>"` (`LibraryAssetSource` = `upload` |
`output`). Parsing is strict — malformed refs, unknown sources, or bad UUID segments are rejected,
never silently truncated.

- Uses the same **cursor pagination** as all other list endpoints, but with a 3-part cursor
  (`created_at`/`expires_at`, source rank, id) that gives a strict total order across the
  upload/output UNION — a cursor encoded under one `sort` is rejected if replayed under another.
- Every grid item carries a server-resolved `available_actions: LibraryAction[]` — a pure,
  table-driven function of media type, source, and whether the asset has generation metadata (see
  `LibraryAction` below). Do not infer allowed actions from `source`/media type client-side.
- Content URLs in responses are always `/v1/content/...` paths (permanent, auth-gated).
- Favorites and display titles (`library_asset_metadata`) are lazily created on first mutation —
  an asset with no favorite/title/project/tags has no metadata row at all until you set one.
- **Projects**: at most one per asset (nullable FK, `ON DELETE SET NULL` — deleting a project
  unassigns its assets rather than touching them). **Tags**: many-to-many (join table, `ON DELETE
  CASCADE` — deleting a tag removes its assignments). Both are user-scoped, name-unique
  case-insensitively per owner (`409` on conflict), and independently CRUD'd below.
- Deleting an asset or letting it expire via the retention sweeper also purges its
  `library_asset_metadata` and `library_asset_tags` rows (both are polymorphic — no FK exists to
  cascade the delete automatically).

#### `GET /v1/library/`

```
Query:    limit? (1–50, default 30),
          cursor? (opaque token),
          source? ("upload" | "output"),
          media_type? ("image" | "video"),
          model? (string — model key; implies output-only),
          favorite? (bool),
          project_id? (UUID),
          tag_id? (UUID),
          expiring? (bool — true: expires_at within 7 days; false: beyond),
          query? (string, ≤200 chars — case-insensitive substring match over
                  display_title / original_filename / prompt),
          created_from? (datetime), created_to? (datetime),
          sort? ("newest" | "oldest" | "expiring_soon", default "newest")
Response: CursorPage<LibraryAssetItem>
Errors:   400 invalid_cursor
```

#### `GET /v1/library/assets/{asset_ref}`

```
Path:     asset_ref (string, e.g. "upload:<uuid>" or "output:<uuid>")
Response: LibraryAssetDetail
Errors:   404 not_found (malformed ref, not owned, or wrong product)
```

#### `GET /v1/library/assets/{asset_ref}/lineage`

```
Path:     asset_ref (string)
Response: LibraryLineageGraph
Errors:   404 not_found
Note:     Ancestor walk is depth-capped at 10 hops (nearest-first, one parent
          per step); descendants are immediate only (not recursive), capped
          at 50 per relation (job outputs / extracted frames, counted
          separately). `ancestors_truncated` / `descendants_truncated` flag
          when a cap clipped the real graph. `descendant_totals` gives the
          full (uncapped) counts regardless of the capped `descendants` list.
          Bounded total query count — acceptable for this on-demand detail
          endpoint, not the list hot path.
```

#### `GET /v1/library/groups/{job_id}`

```
Path:     job_id (UUID)
Response: LibraryGroupDetail
Errors:   404 not_found (job not completed, wrong user, or wrong product)
Note:     Generation-group detail — one GenerationJob with its full output
          list. This is the old GalleryGroupDetail, relocated as-is (D6):
          the grid is per-asset, but a job's outputs still stack into one
          detail view reachable from any of its LibraryAssetItem rows via
          `job_id` + `output_count`.
```

#### `PATCH /v1/library/assets/{asset_ref}`

```
Request:  LibraryAssetPatch — every field is tri-state (absent = leave
          unchanged); see schema below for per-field null/set semantics
Response: LibraryAssetDetail
Errors:   400 validation_error, 404 not_found (asset, project, or tag)
```

#### `PUT /v1/library/assets/{asset_ref}/favorite`

```
Response: 204 No Content
Errors:   404 not_found
Note:     Idempotent — marks favorite=true.
```

#### `DELETE /v1/library/assets/{asset_ref}/favorite`

```
Response: 204 No Content
Errors:   404 not_found
Note:     Idempotent — clears favorite.
```

#### `DELETE /v1/library/assets/{asset_ref}`

```
Response: 204 No Content
Errors:   404 not_found
Note:     Permanently deletes the file from R2 and the DB record (via the
          same ContentProxyService.delete_content used by the old
          DELETE /v1/content/{id}, §9), purges library_asset_metadata /
          library_asset_tags rows, and SETs NULL any lineage references.
```

#### `POST /v1/library/assets/bulk`

```
Request:  BulkOperation — a tagged union discriminated by "type":
  { type: "set_favorite", asset_refs: string[1-100], value: bool }
  { type: "set_project",  asset_refs: string[1-100], project_id: UUID | null }
  { type: "add_tags",     asset_refs: string[1-100], tag_ids: UUID[1-10] }
  { type: "remove_tags",  asset_refs: string[1-100], tag_ids: UUID[1-10] }
  { type: "delete",       asset_refs: string[1-100] }
Response: BulkOperationResult
Errors:   400 invalid_asset_refs (detail.invalid_refs lists every offending
            ref — malformed, missing, not owned, or wrong product),
          404 not_found (set_project's project_id, or any add/remove_tags
            tag_id, doesn't exist / isn't owned by the caller),
          422 tag_cap_exceeded (an add_tags op would push an asset past 20
            tags; detail.asset_refs lists the offenders, detail.cap = 20)
Note:     Every ref is validated BEFORE anything executes — a single bad
          ref fails the whole request, never a silent partial skip.
          Duplicate refs (including mixed-case UUID duplicates) collapse
          to one occurrence. All ops except delete are naturally
          idempotent; delete is idempotent up to "already gone" — a retry
          on an already-deleted ref surfaces it in invalid_refs rather
          than silently succeeding twice.
```

#### `GET /v1/library/projects/`  •  `POST /v1/library/projects/`

```
GET  Query:    limit? (1–50, default 30), cursor? (opaque token)
     Response: CursorPage<LibraryProjectListItem>
POST Request:  LibraryProjectCreate { name: string(1-100), description?: string }
     Response: LibraryProject
     Status:   201 Created
     Errors:   400 validation_error, 409 project_name_conflict
```

#### `GET /v1/library/projects/{id}`  •  `PATCH /v1/library/projects/{id}`  •  `DELETE /v1/library/projects/{id}`

```
GET    Response: LibraryProject                          Errors: 404 not_found
PATCH  Request:  LibraryProjectPatch (tri-state name/description)
       Response: LibraryProject
       Errors:   400 validation_error, 404 not_found, 409 project_name_conflict
DELETE Response: 204 No Content                           Errors: 404 not_found
       Note:     Assigned assets are unassigned (project_id → null via
                 ON DELETE SET NULL), never deleted.
```

#### `GET /v1/library/tags/`  •  `POST /v1/library/tags/`

```
GET  Query:    limit? (1–50, default 30), cursor? (opaque token)
     Response: CursorPage<LibraryTagListItem>
POST Request:  LibraryTagCreate { name: string(1-50) }
     Response: LibraryTag
     Status:   201 Created
     Errors:   400 validation_error, 409 tag_name_conflict
```

#### `GET /v1/library/tags/{id}`  •  `PATCH /v1/library/tags/{id}`  •  `DELETE /v1/library/tags/{id}`

```
GET    Response: LibraryTag                               Errors: 404 not_found
PATCH  Request:  LibraryTagPatch (tri-state name)
       Response: LibraryTag
       Errors:   400 validation_error, 404 not_found, 409 tag_name_conflict
DELETE Response: 204 No Content                           Errors: 404 not_found
       Note:     Asset tag assignments cascade-delete (ON DELETE CASCADE).
```

### Library Schemas

```typescript
interface LibraryAssetItem {
  asset_ref: string;              // "upload:<uuid>" | "output:<uuid>"
  source: LibraryAssetSource;
  media: MediaObject;             // original + sm/md variants
  created_at: string;
  expires_at: string;             // retention-cleanup deletion timestamp
  display_title: string | null;
  original_filename: string | null;  // upload-only
  is_favorite: boolean;
  duration_ms: number | null;     // upload-only, video
  job_id: string | null;          // output-only
  output_count: number | null;    // output-only: non-thumbnail outputs in the same job
  model: string | null;           // output-only
  generation_type: GenerationType | null; // output-only
  available_actions: LibraryAction[];
  project_id: string | null;
  project_name: string | null;    // denormalized, batched lookup
  tags: { id: string; name: string }[];
}

interface LibraryAssetDetail extends LibraryAssetItem {
  prompt: string | null;
  negative_prompt: string | null;
  provider: string | null;
  aspect_ratio: string | null;
  token_cost: number | null;
  completed_at: string | null;
  lineage: LibraryLineage | null;      // single-level frame-extraction lineage
  descendants: { job_count: number; frame_count: number };
}

interface LibraryLineage {
  source_asset_ref: string | null;
  source_job_id: string | null;        // set when the source was a generation output
  source_timestamp_ms: number | null;  // frame-extraction timestamp within the source video
}

interface LibraryAssetPatch {
  display_title?: string | null;  // absent=unchanged, null=clear, string=set (max 255)
  project_id?: string | null;     // absent=unchanged, null=unassign, UUID=assign (must be owned)
  tag_ids?: string[];             // absent=unchanged; replace-set semantics — [] clears all
                                   // tags, a list sets the exact set (max 20, all must be owned)
}

interface LibraryGroupDetail {
  job_id: string;
  badge: LibraryBadge;             // "prompt" (t2i/t2v) or "image" (i2i/i2v/flf2v/v2v)
  input_media: MediaObject | null; // present when badge == "image"
  prompt: string;
  negative_prompt: string | null;
  outputs: LibraryOutputItem[];    // non-thumbnail outputs, ordered by output_index
  media_type: OutputMediaType;
  model: string | null;
  provider: string;
  generation_type: GenerationType;
  aspect_ratio: string | null;     // null ⇒ i2i job that followed the source image's aspect
  token_cost: number | null;
  created_at: string;
  completed_at: string | null;
  lineage: LibraryGroupLineage | null;
}

interface LibraryOutputItem {
  id: string;
  asset_ref: string;               // always "output:<id>"
  output_index: number;
  created_at: string;
  expires_at: string;
  media: MediaObject;
}

interface LibraryGroupLineage {
  source_type: LibraryGroupSourceType;  // "upload" or "output"
  source_upload_id: string | null;
  source_job_id: string | null;
  source_job_name: string | null;
  source_output_id: string | null;
}

interface LibraryLineageGraph {
  focus: LineageNode;
  ancestors: LineageEdge[];         // nearest-first, depth-capped at 10
  descendants: LineageEdge[];       // immediate only, capped at 50 per relation
  descendant_totals: { job_count: number; frame_count: number }; // uncapped
  ancestors_truncated: boolean;
  descendants_truncated: boolean;
}

interface LineageNode {
  asset_ref: string;
  source: LibraryAssetSource;
  media: MediaObject;
  created_at: string;
  model: string | null;
  generation_type: GenerationType | null;
}

interface LineageEdge {
  relation: "generated_from_upload" | "generated_from_output"
          | "frame_of_output" | "frame_of_upload";
  node: LineageNode;
  source_timestamp_ms: number | null;  // frame edges only
}

interface LibraryProject {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

interface LibraryProjectListItem extends LibraryProject {
  asset_count: number;   // batched, not per-row
}

interface LibraryTag {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
}

interface LibraryTagListItem extends LibraryTag {
  asset_count: number;   // batched, not per-row
}

interface BulkOperationResult {
  op: string;            // "set_favorite" | "set_project" | "add_tags" | "remove_tags" | "delete"
  results: { asset_ref: string; success: boolean }[];
  succeeded: number;
  failed: number;
}
```

### Library Action Resolution

`available_actions` on every `LibraryAssetItem`/`LibraryAssetDetail` is resolved server-side by a
pure, table-driven function of media type + whether the asset has generation metadata (i.e. is an
output) — never inferred from `source` on the client:

| Always | Image-only | Video-only | Has generation metadata (output) |
|--------|-----------|------------|-----------------------------------|
| `favorite`, `rename`, `download`, `delete` | `remix`, `create_variation`, `animate`, `use_as_reference`, `use_as_first_frame`, `use_as_last_frame` | `remix`, `extend`, `extract_frame` | `view_settings`, `reproduce` |

### Library Badge Logic

| Badge value | Generation types |
|-------------|-----------------|
| `"prompt"` | `t2i`, `t2v` — text-only input |
| `"image"` | `i2i`, `i2v`, `flf2v`, `v2v` — image/video input |

---

## 11. Billing *(authenticated)*

#### `GET /v1/billing/balance`

```
Response: {
  account_id: UUID,
  account_type: "personal" | "enterprise",
  balance: int,
  organization_name: string | null
}
```

#### `GET /v1/billing/transactions`

```
Query:    limit? (default 50), type? ("debit" | "credit" | "refund" | "admin_adjustment"), cursor? (opaque token)
Response: CursorPage<TransactionResponse>

TransactionResponse: {
  id: UUID,
  transaction_type: string,
  amount: int,                   // negative for debits
  balance_after: int,
  description: string | null,
  metadata: object,
  job_id: UUID | null,
  payment_id: UUID | null,
  created_at: datetime,
  created_by: UUID | null
}
```

#### `GET /v1/billing/pricing`

```
Response: PricingRuleResponse[]

PricingRuleResponse: {
  id: UUID,
  provider: string,
  generation_type: string,
  model: string | null,
  token_cost: int,               // per-output token cost
  input_token_cost: int,         // per input image, charged per output sample
  is_active: bool,
  effective_from: datetime,
  effective_until: datetime | null,
  notes: string | null
}
```

Total generation charge is `(token_cost + input_token_cost × k) × n`, where `n` is the
requested output count and `k` is the input-image count: `0` for T2I, `1` when
`input_image_id` or `source_output_id` is set, or `source_images.length` when
`source_images` is set.

#### `GET /v1/billing/topup/options`

```
Response: TopUpOptionsResponse

TopUpOptionsResponse: {
  min_amount_usd: int,
  max_amount_usd: int,
  tokens_per_usd: int,
  tiers: TopUpTierResponse[]     // ascending by threshold_usd; presets for the UI cards
}

TopUpTierResponse: {
  threshold_usd: int,
  discount_pct: int
}
```

Note: an `amount_usd` qualifies for the highest tier whose `threshold_usd <= amount_usd` (0% if
none match). The discount reduces the price paid, not the token count — see the `POST` endpoints
below. This same tier table is the single source of truth for both this endpoint and the actual
charge (`src/core/topup_pricing.py`), so the UI summary and the charge can never disagree.

#### `GET /v1/billing/account`

```
Response: { preferred_account: AccountType | null, message: string }
```

#### `POST /v1/billing/account`

```
Request:  { account: "personal" | "enterprise" }
Response: { preferred_account: AccountType | null, message: string }
Note:     Sets preferred billing account (personal vs enterprise)
```

#### `POST /v1/billing/topup/stripe`

```
Request:  { amount_usd: int }   // whole USD, the nominal credits amount before discount
Response: { checkout_url: string, session_id: string, payment_id: UUID }
Status:   201 Created
Headers:  Idempotency-Key: <string> (required, max 64 chars)
Errors:   409 idempotency_conflict
          409 { "code": "payment_provider_disabled", "provider": "stripe" }
          400 if amount_usd is outside the configured min/max top-up bounds
              (see GET /v1/billing/topup/options)
Note:     Redirect user to checkout_url for Stripe Checkout. The amount charged
          (Stripe unit_amount, and the stored Payment.amount_usd) is amount_usd with
          the resolved tier discount applied; tokens_granted is always the full,
          pre-discount value (amount_usd * tokens_per_usd).
          Supply a fresh UUIDv4 Idempotency-Key per checkout attempt to prevent duplicate payments.
```

#### `POST /v1/billing/topup/nowpayments`

```
Request:  { amount_usd: int, pay_currency?: string }
Response: { invoice_url: string, payment_id: UUID }
Status:   201 Created
Headers:  Idempotency-Key: <string> (required, max 64 chars)
Errors:   409 idempotency_conflict
          409 { "code": "payment_provider_disabled", "provider": "nowpayments" }
          400 if amount_usd is outside the configured min/max top-up bounds
          400 { "code": "pay_currency_suppressed", "pay_currency": "<TICKER>" } when a pinned
              pay_currency has been superadmin-suppressed (see §14) — never raised when
              pay_currency is omitted; a stale currency picker should re-fetch
              GET /v1/billing/currencies and ask the user to pick again
Note:     Same discount-on-price semantics as the Stripe path. NowPayments IPN
          under/overpayments are credited proportionally to actually_paid/amount_usd
          (uncapped on overpayment) — never held for manual review.

          pay_currency is optional. Pass the exact ticker returned by
          GET /v1/billing/currencies (e.g. "USDCMATIC") to pin the
          invoice to that currency/network (unchanged behavior) — typically sourced
          from GET /v1/billing/currencies. Omit it (or send blank/whitespace) to let
          the customer pick any currency NowPayments supports on the hosted invoice
          page — the checkout UI is never required to fetch or render a currency
          list itself; the catalog is advisory UI only (see GET /v1/billing/currencies).

          Payment.currency is "USD" at charge time when pay_currency was omitted
          (mirrors price_currency — the invoice is USD-denominated until paid) and
          is patched to the customer's actual settled ticker once the first IPN
          reports it, even on intermediate (waiting/confirming) statuses. Poll
          GET /v1/billing/... payment/transaction records after the invoice_url
          redirect to observe the final currency; do not assume it stays "USD".
```

### Billing — Public (no auth)

#### `GET /v1/billing/providers`

```
Auth:     none
Response: Array<{
  provider: "stripe" | "nowpayments",
  display_order: int
}>
```

Returns the ordered effective provider set for the product resolved from the request host/header.
Effective means the provider is present in the product's static capability set and is not disabled
by a runtime override. An absent override row means enabled. Disabled providers are omitted.
Checkout UIs should render this list rather than hardcoding provider availability.

#### `GET /v1/billing/currencies`

```
Auth:     none
Response: Array<{
  ticker: string,           // uppercased provider ticker, e.g. "BTC", "USDCMATIC"
  name: string | null,
  network: string | null,   // uppercased provider network code
  logo_url: string | null   // served from the R2 public assets domain, never nowpayments.io
}>
```

DB-cached currency catalog for the product's catalog-capable payment providers (currently
NowPayments only), refreshed by a periodic worker (every `PAYMENT_CURRENCY_SYNC_INTERVAL_SECONDS`,
default 3h) and on-demand by superadmin (`POST /v1/admin/payments/currencies/refresh`). Only rows
that are both `is_available` and **not** superadmin-suppressed are returned, ordered by ticker. No
hardcoded ticker list exists anywhere in this contract — availability is decided solely by the
provider's own dashboard-checked list, and suppression is a superadmin-authored deny-list layered
on top (see §14) for tickers the provider wrongly reports as available.

FE contract:
- **Empty array** (cold cache, or every catalog-capable provider disabled/unconfigured) ⇒ hide the
  currency picker and omit `pay_currency` from `POST /v1/billing/topup/nowpayments` — NowPayments'
  own hosted invoice-page picker takes over. Checkout has zero dependency on catalog state.
- `logo_url: null` ⇒ render a generic coin icon.
- `name`/`network: null` ⇒ render a ticker-only label.

`GET /v1/billing/topup/options` and `GET /v1/billing/pricing` remain authenticated.

### Billing — Webhooks (no auth)

#### `POST /v1/billing/webhooks/stripe`

```
Request:  Stripe webhook payload (raw body + Stripe-Signature header)
Note:     Internal endpoint for Stripe payment events
          Webhook settlement remains active when Stripe is runtime-disabled.
```

#### `POST /v1/billing/webhooks/nowpayments`

```
Request:  NowPayments webhook payload (raw body + x-nowpayments-sig header)
Note:     Internal endpoint for NowPayments events
          Webhook settlement remains active when NowPayments is runtime-disabled.
```

`NOWPAYMENTS_API_BASE` defaults to `https://api.nowpayments.io` and may be pointed at the
NowPayments sandbox without a code change.

---

## 12. Organizations *(authenticated)*

#### `POST /v1/organizations/`

```
Request:  { name: string }
Response: {
  organization: OrgResponse,
  account: AccountSummary,
  membership: MemberResponse
}
Status:   201 Created
```

#### `GET /v1/organizations/me`

```
Response: { organization: OrgResponse, role: OrgRole, balance: int }
Errors:   404 (not a member of any org)
```

#### `GET /v1/organizations/{org_id}`

```
Response: OrgResponse
Errors:   404
```

#### `GET /v1/organizations/{org_id}/members`

```
Response: MemberResponse[]
```

#### `POST /v1/organizations/{org_id}/members`

```
Request:  { user_id: UUID, role: "admin" | "member" }
Response: MemberResponse
Status:   201 Created
```

#### `PATCH /v1/organizations/{org_id}/members/{user_id}`

```
Request:  { role: "admin" | "member" }
Response: MemberResponse
```

#### `DELETE /v1/organizations/{org_id}/members/{user_id}`

```
Response: { message: string }
```

#### `DELETE /v1/organizations/{org_id}`

```
Query:    force_delete? (bool)
Response: { message: string }
Errors:   409 organization_balance_nonzero (unless force_delete=true)
```

### Organization Schemas

```
OrgResponse: {
  id: UUID,
  name: string,
  slug: string,
  owner_id: UUID,
  is_active: bool,
  created_at: datetime
}

AccountSummary: {
  account_id: UUID,
  account_type: AccountType,
  balance: int
}

MemberResponse: {
  id: UUID,
  user_id: UUID,
  role: OrgRole,
  joined_at: datetime
}
```

---

## 13. Admin *(authenticated — ADMIN or SUPERADMIN role)*

### Role Hierarchy

| Role | Access | Billing Adjust | Role Management |
|------|--------|----------------|-----------------|
| `superadmin` | All admin endpoints | Inherent | Can grant/revoke ADMIN and SUPERADMIN |
| `admin` | All admin endpoints | Only with explicit `billing_adjust` permission | None |
| `user` | No admin endpoints | — | — |

### User Management

#### `GET /v1/admin/users`

```
Query:    is_active? (bool), role? (string), email? (partial match, case-insensitive),
          limit? (default 50)
Response: CursorPage<AdminUserResponse>
Note:     SYSTEM role users are never returned regardless of filters.

AdminUserResponse: {
  id: UUID,
  email: string,
  display_name: string | null,
  role: string,               // UserRole value
  subscription_tier: string,  // SubscriptionTier value
  is_active: bool,
  email_verified_at: datetime | null,
  created_at: datetime,
  updated_at: datetime
}
```

#### `PATCH /v1/admin/users/{user_id}`

```
Request:  {
  role?: UserRole,            // "admin" or "user" only — not "superadmin"
  subscription_tier?: SubscriptionTier,
  is_active?: bool,
  locale?: SupportedLocale
}
          // All fields optional — only provided fields are updated
Response: AdminUserResponse
Errors:   400 (role=system or role=superadmin),
          403 (patching own account, or target user is a superadmin),
          404 (user not found)
Notes:    - Admins cannot modify their own account via this endpoint.
          - Cannot set role to "superadmin" via this endpoint — use POST /v1/admin/manage/roles/{user_id}/grant.
          - Cannot patch superadmin users via this endpoint — use /v1/admin/manage/ endpoints.
```

### Organization Management

#### `GET /v1/admin/organizations`

```
Query:    is_active? (bool), limit? (default 50)
Response: CursorPage<AdminOrgResponse>

AdminOrgResponse: {
  id: UUID,
  name: string,
  slug: string,
  owner_id: UUID,
  is_active: bool,
  member_count: int,
  token_balance: int,   // 0 if no token account
  created_at: datetime
}
```

### Account Management

#### `GET /v1/admin/accounts/{account_id}/balance`

```
Response: BalanceResponse (same as GET /billing/balance)
```

#### `GET /v1/admin/accounts/{account_id}/transactions`

```
Query:    limit?, type?
Response: CursorPage<TransactionResponse>
```

#### `POST /v1/admin/accounts/{account_id}/adjust`

```
Request:  { amount: int, description: string }
          // positive amount = credit, negative = debit
Response: { transaction: TransactionResponse, new_balance: int }
Headers:  Idempotency-Key: <string> (required, max 64 chars)
Errors:   401 (admin without billing_adjust permission), 409 idempotency_conflict
Notes:    - Requires SUPERADMIN role OR ADMIN role with explicit "billing_adjust" permission grant.
          - Idempotency is scoped to the admin's user_id — prevents duplicate adjustments on retry.
```

#### `GET /v1/admin/users/{user_id}/account`

```
Response: BalanceResponse
```

#### `GET /v1/admin/organizations/{org_id}/account`

```
Response: BalanceResponse
```

### Pricing Management

#### `GET /v1/admin/pricing`

```
Query:    active_only? (default true)
Response: PricingRuleResponse[]
```

#### `POST /v1/admin/pricing`

```
Request:  { provider: string, generation_type: string, model?: string | null, token_cost: int, input_token_cost?: int, notes?: string | null }
Response: PricingRuleResponse
Status:   201 Created
```

`token_cost` is the per-output cost. `input_token_cost` defaults to `0` and is charged per
input image per output sample.

#### `PATCH /v1/admin/pricing/{rule_id}`

```
Request:  { token_cost?: int, input_token_cost?: int, is_active?: bool, effective_until?: datetime | null, notes?: string | null }
Response: PricingRuleResponse
```

Patch fields are optional. Omitted nullable fields are left unchanged; explicit `null` clears
`effective_until` or `notes`.

#### `DELETE /v1/admin/pricing/{rule_id}`

```
Response: { message: string }
Note:     Deactivates the rule (sets is_active=false), does not hard-delete
```

### Payment Management

#### `GET /v1/admin/payments`

```
Query:    status?, payment_provider?, limit? (default 50)
Response: CursorPage<PaymentResponse>

PaymentResponse: {
  id: UUID,
  payment_provider: string,
  status: PaymentStatus,
  amount_usd: string,        // decimal string
  tokens_granted: int,
  currency: string,          // e.g. "USD"; for NowPayments with an unpinned
                             // pay_currency, "USD" until the first IPN reports
                             // the customer's chosen settlement ticker
                             // (e.g. "USDCMATIC"), then that value
  created_at: datetime,
  completed_at: datetime | null
}
```

#### `GET /v1/admin/payments/{payment_id}`

```
Response: PaymentResponse
Errors:   404
```

### Model Management

#### `GET /v1/admin/models`

```
Query:    enabled_only? (default false)
Response: {
  items: GenerationModelResponse[],
  total: int
}

GenerationModelResponse: {
  model_key: string,
  provider: string,
  name: string,
  description: string,
  is_enabled: bool,
  created_at: datetime,
  updated_at: datetime
}
```

#### `PATCH /v1/admin/models/{model_key}`

```
Request:  { is_enabled: bool }
Response: GenerationModelResponse
Errors:   404
```

---

## 14. Admin Management *(authenticated — SUPERADMIN only)*

All endpoints under `/v1/admin/manage` require the **SUPERADMIN** role. An ADMIN attempting to call these endpoints receives `401 Unauthorized`.

### Shared types

```typescript
AdminRoleResponse: {
  id: UUID,
  email: string,
  display_name: string | null,
  role: string,           // "superadmin" | "admin"
  permissions: string[],  // e.g. ["billing_adjust"]
  is_active: bool,
  created_at: datetime,
  updated_at: datetime
}

AuditLogEntry: {
  id: UUID,
  actor_id: UUID,
  target_user_id: UUID | null,
  action: string,   // role.*, permission.*, payment_provider.enable/disable/reorder,
                    // or notification_prefs.update / telegram.link_requested / telegram.unlinked (§15c)
  detail: string,   // human-readable, e.g. "Role changed from 'user' to 'admin'"
  source: string,   // "api" | "cli"
  created_at: datetime
}
```

### Endpoints

#### `GET /v1/admin/manage/admins`

```
Response: AdminRoleResponse[]
Note:     Returns all SUPERADMIN users first, then all ADMIN users, for the current product.
          Each entry includes the user's current permission grants.
```

#### `POST /v1/admin/manage/roles/{user_id}/grant`

```
Request:  { role: "admin" | "superadmin" }
Response: { message: string }
Errors:   403 (self-modification),
          400 (invalid role — must be "admin" or "superadmin"),
          404 (user not found in current product)
Note:     Writes an audit entry with source="api".
```

#### `POST /v1/admin/manage/roles/{user_id}/revoke`

```
Response: { message: string }
Errors:   403 (self-modification),
          400 (last superadmin — cannot leave the product with zero superadmins, or the user has no admin role),
          404 (user not found in current product)
Notes:    - Demotes the target user back to role "user".
          - Automatically revokes all permission grants for that user in the current product.
          - Writes an audit entry.
```

#### `POST /v1/admin/manage/permissions/{user_id}/grant`

```
Request:  { permission: "billing_adjust" }
Response: { message: string }
Errors:   400 (user is not an admin, or user not found)
Note:     Idempotent — granting an already-held permission is a no-op (returns 200, no duplicate entry).
          Writes an audit entry only on first grant.
```

#### `POST /v1/admin/manage/permissions/{user_id}/revoke`

```
Request:  { permission: "billing_adjust" }
Response: { message: string }
Note:     Idempotent — revoking a permission the user doesn't hold is a no-op (returns 200).
          Writes an audit entry only when a row was actually deleted.
```

#### `GET /v1/admin/manage/audit`

```
Query:    target_user_id? (UUID), limit? (default 50), cursor? (opaque token)
Response: CursorPage<AuditLogEntry>
Note:     Entries are returned newest-first. Optionally filter to a specific target user.
          Uses cursor (keyset) pagination — pass cursor=next_cursor from the previous
          response to fetch the next page. Breaking change from the previous bare
          AuditLogEntry[] response: the body is now wrapped in the standard CursorPage
          envelope (items / limit / has_more / next_cursor). Regenerate OpenAPI types
          and update the admin audit-log table in apex-frontend (gen:api → cursor scroll).
```

### Payment Provider Registry

All endpoints under `/v1/admin/payments/providers` require the **SUPERADMIN** role and are scoped
to the product resolved for the request.

```typescript
ProviderInfo: {
  provider: "stripe" | "nowpayments",
  is_enabled: boolean,            // effective runtime state
  display_order: number,
  credentials_configured: boolean // warning signal only; does not gate listing
}
```

#### `GET /v1/admin/payments/providers/`

```
Response: ProviderInfo[]
Note:     Includes every provider in the product's static capability set, including disabled
          providers, ordered by display_order then provider name.
```

#### `PATCH /v1/admin/payments/providers/{provider}`

```
Request:  { is_enabled?: boolean | null, display_order?: int | null }
Response: ProviderInfo
Errors:   400 when neither field is supplied
          404 when provider is unknown or outside the product's static capability set
Note:     Writes payment_provider.enable, payment_provider.disable, or
          payment_provider.reorder to the append-only audit log with target_user_id=null.
```

### Payment Currency Catalog

Superadmin management of the DB-cached currency catalog (see `GET /v1/billing/currencies` for the
public contract). All endpoints require **SUPERADMIN** and are scoped to the resolved product.

```typescript
AdminCurrency: {
  ticker: string,
  provider: "stripe" | "nowpayments",
  is_available: boolean,          // false = flipped unavailable by the most recent sync, row kept
  is_suppressed: boolean,         // true = superadmin deny-listed; excluded from the public picker
                                  // and from pinned top-ups regardless of is_available
  name: string | null,
  network: string | null,
  logo_key: string | null,        // R2 object key; null = no cached logo
  logo_source_url: string | null, // provider URL last successfully cached (change detector)
  logo_synced_at: string | null,  // ISO 8601
  last_seen_at: string            // ISO 8601, touched on every sync that includes this ticker
}

SyncResult: {
  provider: "stripe" | "nowpayments",
  upserted: number,
  deactivated: number
}
```

#### `GET /v1/admin/payments/currencies`

```
Response: AdminCurrency[]
Note:     Full catalog including unavailable and suppressed rows, ordered by ticker. Unlike the
          public endpoint, this never filters by is_available or is_suppressed — used to audit
          sync history and manage the deny-list.
```

#### `PATCH /v1/admin/payments/currencies/{provider}/{ticker}`

```
Request:  { is_suppressed: boolean }
Response: AdminCurrency
Errors:   404 when provider is unknown or outside the product's static capability set
          404 when ticker has never been seen for this (product, provider) pair — suppression
              requires an existing catalog row; there is no pre-emptive or pattern
              (e.g. "all *XTZ") suppression
Note:     ticker is case-insensitive (uppercased server-side before lookup). Takes effect
          immediately — GET /v1/billing/currencies excludes a newly suppressed ticker on its
          very next request, no catalog sync required, and the flag is never touched by
          POST /v1/admin/payments/currencies/refresh (a suppression survives every sync,
          including a deactivate→reappear cycle). Writes payment_currency.suppress or
          payment_currency.unsuppress to the audit log with target_user_id=null — only when
          is_suppressed actually changes; a PATCH that sets the value it already has is a
          no-op (200, unchanged row, no new audit row).
```

#### `POST /v1/admin/payments/currencies/refresh`

```
Response: SyncResult[]   // one entry per catalog-capable provider in the product's capability set
Errors:   502 when any provider's merchant/coins or full-currencies call fails — the previously
          synced catalog and logos are left untouched (no partial sync is ever committed)
Note:     Synchronous — runs list_merchant_currencies + list_full_currencies + logo caching inline
          and commits before responding. Writes payment_currencies.refresh to the audit log with
          target_user_id=null and a detail blob of per-provider upserted/deactivated counts.
          Never reads or writes is_suppressed — a refresh cannot resurrect a suppressed ticker.
```

#### Ops note: provider-side zombie currencies

NowPayments confirmed (support ticket) a data bug on their side: `merchant/coins` can report
tickers they have effectively delisted/killed. They will not fix it, and their only offered
remedy is a new account — which doesn't prevent recurrence. Since NowPayments would still create
invoices for such zombie currencies (stranding customer payments on a dead rail), suppression is
the authoritative-negative override: **workflow** is support confirms a specific dead ticker →
superadmin `PATCH`es it suppressed → it vanishes from the public picker immediately, no sync
needed. Suppression never invents an *allow* — an unsuppressed/unknown ticker on a pinned top-up
still passes through to NowPayments' own validation (their 4xx is the validator, unchanged).

**Residual risk (accepted, documented, not built around):** the NowPayments-hosted invoice page
(the customer-chooses flow used when `pay_currency` is omitted) renders *their* currency list —
Apex cannot filter that page, so a zombie ticker may still be selectable there. Suppression fully
protects the Apex-rendered picker and any pinned invoice. If this residual path ever strands a
real payment, the escalation is a frontend policy of pinned-currency-only checkout (forcing
`pay_currency` to always be set from `GET /v1/billing/currencies`) — not implemented today.

**FE addendum:** the admin panel gains a suppress/unsuppress toggle per catalog row (driven by
`PATCH /v1/admin/payments/currencies/{provider}/{ticker}`). Checkout must handle the new
`400 { "code": "pay_currency_suppressed", "pay_currency": "<TICKER>" }` from
`POST /v1/billing/topup/nowpayments` by re-fetching `GET /v1/billing/currencies` and asking the
user to pick a currency again — this is a narrow race window (the ticker was suppressed between
the picker load and the top-up submit), not a normal-path error.

---

## 15. Real-Time Events (SSE + Pub/Sub)

The backend supports real-time event streaming via **Server-Sent Events (SSE)** backed by Redis Pub/Sub. Because `EventSource` cannot send custom headers, authentication uses a short-lived **one-time ticket** pattern.

> **Requires Redis**: SSE is only active when `REDIS_URL` is configured on the server. When Redis is not configured, the `/v1/events/stream` endpoint returns `503 Service Unavailable`.

### Flow

```
1. Client (authenticated)  POST /v1/events/sse-ticket  →  { ticket: "abc123" }
2. Client                  GET  /v1/events/stream?ticket=abc123  →  text/event-stream
3. Server streams EventEnvelopes until client disconnects
```

### Auth & Ticket

#### `POST /v1/events/sse-ticket` *(authenticated)*

```
Response: { ticket: string }   // opaque URL-safe token
Status:   201 Created
Errors:   401 unauthorized
Rate:     10/minute per user
Note:     Ticket is single-use and expires in 30 seconds. Obtain a fresh ticket
          immediately before opening the SSE stream.
```

### Streaming Endpoint

#### `GET /v1/events/stream`

```
Query:   ticket=<string>   // required — one-time ticket from POST /v1/events/sse-ticket
Headers: Accept: text/event-stream
Response: 200 text/event-stream (chunked)
Errors:  401 (missing/expired/invalid ticket), 503 (Redis not configured)
Note:    Long-lived HTTP connection. Heartbeat comments sent every ~15 seconds
         to keep connection alive through proxies.
```

**SSE frame format** (each event):

```
id: <event_id>
event: <EventType>
data: <JSON-encoded inner payload>

```

**Heartbeat** (sent when idle to maintain connection):

```
: keepalive

```

### Event Types

| `event` field | Description | Payload type |
|---------------|-------------|--------------|
| `job.status_changed` | Job moved to a new status | `JobStatusPayload` |
| `job.progress` | Job progress update | `JobProgressPayload` |
| `gpu_session.status_changed` | GPU session moved to a new status (e.g. provisioning → active, active → paused, paused → resuming → active, → stopped) | `GpuSessionStatusPayload` |
| `gpu_session.credit_warning` | Session balance is low; emitted once per upward level transition (no warning → warning → critical). Cleared on balance recovery or termination. | `GpuSessionCreditWarningPayload` |
| `balance.updated` | Token balance changed (debit, credit, refund) | `BalanceUpdatedPayload` |
| `system.notification` | Broadcast system message (maintenance, outage) | `SystemNotificationPayload` |

### Event Payload Schemas

```typescript
// job.status_changed
interface JobStatusPayload {
  job_id: string;           // UUID
  status: JobStatus;        // new status
  previous_status: string;  // actual persisted pre-transition status (or "none" on first publish)
  generation_type: string;  // e.g. "t2v"
  provider: string;         // e.g. "grok"
  failure_code: string | null; // set for normalized terminal failures
  error_message: string | null; // public-safe terminal-failure message, never backend diagnostics
}

// job.progress
interface JobProgressPayload {
  job_id: string;
  progress_pct: number;     // 0–100
  generation_type: string;
}

// gpu_session.status_changed
interface GpuSessionStatusPayload {
  session_id: string;            // UUID
  status: GpuSessionStatus;      // new status
  previous_status: string;       // previous status
  model_type: string;            // e.g. "aisha-image"
  tunnel_hostname: string | null;
  error_message: string | null;  // populated when status == "failed"
  reason: string | null;         // machine-readable stop reason, e.g. "insufficient_credits"
}

// gpu_session.credit_warning
interface GpuSessionCreditWarningPayload {
  session_id: string;            // UUID
  level: "warning" | "critical"; // severity; only emitted on upward transitions
  minutes_remaining: number;     // estimated minutes left at current burn rate
  terminate_at: string | null;   // ISO datetime when session will auto-terminate (null if >24h)
  balance: number;               // current token balance at time of emission
}

// balance.updated
interface BalanceUpdatedPayload {
  account_id: string;       // UUID
  balance: number;          // new balance (tokens)
  delta: number;            // change (negative = debit)
  transaction_type: string; // "debit" | "credit" | "refund" | "admin_adjustment"
}

// system.notification
interface SystemNotificationPayload {
  level: string;            // "info" | "warning" | "critical"
  title: string;
  message: string;
  expires_at: string | null; // ISO datetime
}
```

### Channel Topology

| Channel | Subscribers | Events |
|---------|-------------|--------|
| `user:{user_id}` | Per-user | `job.status_changed`, `job.progress`, `gpu_session.status_changed`, `gpu_session.credit_warning`, `balance.updated` |
| `system:broadcast` | All connected clients | `system.notification` |

Each SSE connection subscribes to both the per-user channel and `system:broadcast`.

### Publish Points

Events are automatically published by the backend at:

| Event | Published when |
|-------|---------------|
| `job.status_changed` | Generation request submitted (status → `pending`) |
| `job.status_changed` | Grok video job completes, fails, or times out |
| `job.progress` | Grok video job enters `running` state |
| `gpu_session.status_changed` | GPU session transitions between any two states (start/provision/active/pause/resume/stop/fail) |
| `gpu_session.credit_warning` | `SessionCreditGuard` cycle detects balance at warning or critical level (emitted once per upward transition) |
| `balance.updated` | `check_and_reserve` (debit), `refund`, `credit`, `admin_adjustment`, `settle_session_usage` |
| `system.notification` | Admin calls `POST /v1/admin/broadcast` |

### Admin Broadcast

#### `POST /v1/admin/broadcast` *(admin only)*

```
Request: {
  level: "info" | "warning" | "critical",
  title: string,
  message: string,
  expires_at?: string | null  // ISO datetime
}
Response: { message: string }
Status:  200 OK
Note:    Publishes to system:broadcast channel — delivered to all active SSE connections
```

### Frontend Usage Example

```typescript
async function openEventStream(apiFetch: Fetcher) {
  // 1. Get a fresh ticket
  const { ticket } = await apiFetch<{ ticket: string }>(
    '/v1/events/sse-ticket', { method: 'POST' }
  );

  // 2. Open SSE stream
  const es = new EventSource(`/v1/events/stream?ticket=${ticket}`);

  es.addEventListener('job.status_changed', (e) => {
    const payload = JSON.parse(e.data);
    console.log('Job status:', payload.status);
  });

  es.addEventListener('balance.updated', (e) => {
    const payload = JSON.parse(e.data);
    console.log('New balance:', payload.balance);
  });

  es.addEventListener('system.notification', (e) => {
    const payload = JSON.parse(e.data);
    showBanner(payload.level, payload.title, payload.message);
  });

  es.onerror = () => {
    // Reconnect with a fresh ticket — the old ticket is expired/used
    es.close();
    setTimeout(() => openEventStream(apiFetch), 2000);
  };

  return es;
}
```

> **Reconnection note:** `EventSource` auto-reconnects on error, but the ticket is single-use and already expired. Always close and re-obtain a fresh ticket on error.

---

## 15b. Push Notifications (Web Push)

Backend-driven **Web Push** notifications, delivered via the browser Push API and a service worker — unlike SSE (§15), these arrive **even when the app is closed or the tab isn't open**. Requires the frontend PWA to register a service worker and a `PushSubscription`.

> **Requires VAPID + Redis**: Push is only active when `VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY`, `VAPID_SUBJECT`, and `REDIS_URL` are all configured on the server. When disabled, all three endpoints below return `503 Service Unavailable`.

### Flow

```
1. Client (authenticated)  GET  /v1/push/vapid-public-key       →  { public_key: "..." }
2. Client                  navigator.serviceWorker.register(...) + pushManager.subscribe({ applicationServerKey: public_key })
3. Client                  POST /v1/push/subscriptions           →  registers the PushSubscription with the backend
4. Server                  pushes notifications as relevant events occur (see mapping table below)
5. Client (on logout / permission revoked)  DELETE /v1/push/subscriptions
```

### Endpoints

#### `GET /v1/push/vapid-public-key` *(authenticated)*

```
Response: { public_key: string }   // base64url, pass directly as applicationServerKey
Status:   200 OK
Errors:   401 unauthorized, 503 (push not configured)
```

#### `POST /v1/push/subscriptions` *(authenticated)*

```
Request: {
  endpoint: string,               // from PushSubscription.toJSON().endpoint
  keys: { p256dh: string, auth: string },
  user_agent?: string | null
}
Response: { id: string, endpoint: string, created_at: string }
Status:   201 Created
Errors:   401 unauthorized, 422 validation_error, 503 (push not configured)
Note:     Upserts by endpoint. If the endpoint was previously registered under a
          different user (shared device, account switch), it is reassigned to the
          current user.
Note:     Serialized against the five bulk-revocation events below: the handler
          acquires the same user-row lock those paths take first, then re-checks
          revocation before upserting. A token revoked concurrently with this
          request (not just before it) is rejected with 401 rather than being
          allowed to register a subscription that would outlive the session —
          see "Server-Side Cleanup on Bulk Session Revocation" below.
```

#### `DELETE /v1/push/subscriptions` *(authenticated)*

```
Request: { endpoint: string }
Response: (empty body)
Status:   204 No Content
Errors:   401 unauthorized, 503 (push not configured)
Note:     Idempotent — returns 204 even if the endpoint was never registered, or
          already belongs to a different user (no ownership leak). Deletes only the
          caller's own endpoint — for every subscription a user has, see the
          bulk-revocation cleanup note below.
```

### Server-Side Cleanup on Bulk Session Revocation

Besides the client-initiated `DELETE /v1/push/subscriptions` above (one endpoint) and the
dispatcher's own pruning of expired subscriptions (a 404/410 from the push service, see
"Delivery Guarantees" below), five server-side events also delete **every** subscription a user has
(`PushSubscriptionRepository.delete_all_for_user`), run alongside their existing
`TokenRevocationService.revoke_user_sessions` bulk-revocation call:

- `POST /v1/users/me/logout-all` (§3)
- `POST /v1/users/me/password` (§3)
- `DELETE /v1/users/me` (§3)
- `POST /v1/auth/reset-password` (§2.2)
- Refresh-token reuse detection — a `token_reuse_detected` 401 on `POST /v1/auth/refresh` (§2.2)

This closes the gap the client cannot: a user who suspects compromise and revokes every session
from one device previously left the attacker's device subscribed indefinitely, since nothing
server-side ever deleted that row. **Single-device `POST /v1/auth/logout` deliberately does
NOT delete any subscription** — it ends only one session, and deleting every subscription would
silently kill push on the user's other devices; the client is expected to unsubscribe its own
endpoint locally before calling it. Cleanup is best-effort and isolated (a SAVEPOINT, not the
outer transaction): a failure never blocks the triggering action itself, and is reported via the
`ops.push.subscriptions_cleanup_failed` ops event rather than assumed successful.

**Race with a concurrent `POST /v1/push/subscriptions`**: any of the five events above can land
while a device is mid-registration. `POST /v1/push/subscriptions` acquires the same user-row lock
these events take first, then re-checks revocation before upserting — so whichever side wins the
lock, the other observes it: a subscription that wins the race is still caught by the bulk delete's
subsequent snapshot, and a bulk event that wins the race causes the registration to see the fresh
epoch and reject with 401 rather than insert a row the revocation already believes it cleaned up.

### Wire Payload Contract

Every push message body is exactly this JSON shape — the service worker's `push` event handler should parse it directly and call `registration.showNotification(payload.title, {...})`:

```typescript
interface PushNotificationPayload {
  title: string;
  body: string;
  url: string;    // relative deep link to open on notification click
  tag: string;     // used for OS-level notification coalescing (repeat tag replaces prior)
  category: "job" | "gpu_credit" | "system" | "balance";
  level: "info" | "warning" | "critical";
}
```

### Event → Notification Mapping

The backend maps a subset of the same real-time events used by SSE (§15) into push notifications. Mapping logic is pure and one-way — it never affects SSE delivery.

| SSE event | Pushed when | Notes |
|-----------|-------------|-------|
| `job.status_changed` | Only terminal states: `completed`, `failed` | `tag: "job-{job_id}"`, `url: "/app/library/groups/{job_id}"` |
| `gpu_session.credit_warning` | Every level (`warning`, `critical`) | `tag: "gpu-credit-{session_id}"` — repeated warnings coalesce instead of stacking |
| `system.notification` | Always — broadcast to **every** subscription | `tag: "system-notification"` |
| `balance.updated` | Only `delta > 0` **and** `transaction_type` is `credit` or `admin_adjustment` | Per-generation debits and refunds never push (avoids spam) |
| `job.progress`, `gpu_session.status_changed` | Never | Ignored entirely |

### Delivery Guarantees

- **Best-effort**, same as SSE — if the push dispatcher process is down or Redis drops the connection, notifications are lost (no replay/persistence queue).
- **Expired subscriptions** (the push service returns HTTP 404/410) are pruned automatically — no client action needed; the next `POST /v1/push/subscriptions` re-registers.
- **No per-category preferences yet** — a subscribed user receives all four categories above. Granular opt-out is planned for a future release.
- **No presence suppression** — a user with an open SSE connection still receives push notifications for the same event today.

### Frontend Usage Example

```typescript
async function registerPushNotifications(apiFetch: Fetcher) {
  const registration = await navigator.serviceWorker.register('/sw.js');

  const { public_key } = await apiFetch<{ public_key: string }>('/v1/push/vapid-public-key');

  const subscription = await registration.pushManager.subscribe({
    userVisibleOnly: true,
    applicationServerKey: public_key,
  });

  const { endpoint, keys } = subscription.toJSON() as {
    endpoint: string;
    keys: { p256dh: string; auth: string };
  };

  await apiFetch('/v1/push/subscriptions', {
    method: 'POST',
    body: JSON.stringify({ endpoint, keys, user_agent: navigator.userAgent }),
  });
}

// sw.js — service worker push handler
self.addEventListener('push', (event) => {
  const payload = event.data.json();
  event.waitUntil(
    self.registration.showNotification(payload.title, {
      body: payload.body,
      tag: payload.tag,
      data: { url: payload.url },
    })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(clients.openWindow(event.notification.data.url));
});
```

---

## 15c. Admin Ops Notifications (Telegram)

Backend-driven **operational alerting for admins/superadmins**, delivered as Telegram messages — separate from the consumer-facing SSE (§15) and Web Push (§15b) channels above. This is an admin-panel-only feature; it has no effect on and no dependency on the end-user-facing API surface.

> **Requires bot token + Redis**: Telegram delivery is only active when `TELEGRAM_BOT_TOKEN` and `REDIS_URL` are both configured on the server. When disabled, `POST /v1/admin/notifications/telegram/link` returns `503 Service Unavailable`. The preference endpoints (`/classes`, `/preferences`) work regardless — an admin can configure subscriptions ahead of Telegram being enabled.

### Notification Classes

| Class | Scope | Fires when |
|-------|-------|------------|
| `user.registered` | product | A new user registers on the admin's own product |
| `generation.created` | product | A new generation job is submitted |
| `gpu_node.started` | product | A GPU node finishes provisioning — `provisioning → active` only; resuming a paused session does **not** count as "started" |
| `generation.failed` | product | A generation job transitions to `failed`. Provider-authentication failures are reported separately under `provider_authentication.failed`, not here — see below. |
| `provider_authentication.failed` | platform | A generation provider rejected its API key/authentication — fires once per failed generation for as long as the credentials are broken, so it's seeded with a 300s throttle (not the usual 0) rather than flooding on every request during an incident |
| `health.degraded` | platform | A platform health subsystem becomes `degraded`, `unhealthy`, or `unknown` |
| `health.restored` | platform | A platform health subsystem recovers from a bad status back to a healthy one |
| `token_revocation.failed` | platform | A bulk access-token revocation (Redis write) failed while Redis is otherwise configured — the affected user's existing access tokens/content cookies remain valid until they expire |
| `push_subscriptions.cleanup_failed` | platform | A bulk-revocation event's push-subscription cleanup (`delete_all_for_user`) failed — the affected user's devices that should have been unsubscribed may still receive push notifications |

- **Product-scoped** classes (`user.registered`, `generation.created`, `gpu_node.started`, `generation.failed`) are delivered only to admins whose own account product matches the event's product — a `synthara` admin never sees a `vex` registration.
- **Platform-scoped** classes (`provider_authentication.failed`, `health.*`, `token_revocation.failed`, `push_subscriptions.cleanup_failed`) are delivered to every subscribed admin/superadmin regardless of product, since these describe the health/safety of the whole platform rather than any single product.
- `token_revocation.failed` ships with a one-time seed (migration `029`); `push_subscriptions.cleanup_failed` ships with the same treatment (migration `032`); `provider_authentication.failed` ships the same way in migration `034` (kept separate from the `033` migration that introduced the class's schema/mapping, because Alembic never re-runs an already-applied revision — appending the seed to `033` would silently skip any environment already at `033`) — every admin who already has a Telegram link gets a subscription automatically, so existing installs don't start blind. It's still an ordinary preference row after that — a subsequent full-set `PUT /v1/admin/notifications/preferences` that omits it un-subscribes the admin, same as any other class. Unlike `token_revocation.failed`, `push_subscriptions.cleanup_failed` and `provider_authentication.failed` have no second, preference-independent channel (no health checker watches them), which is why seeding — not just a release note — was judged necessary there.
- Subscription is **row-presence**, not a flag: `PUT /v1/admin/notifications/preferences` is a full-set replace — a class omitted from the request body is unsubscribed.
- Each subscribed class carries an optional `min_interval_seconds` throttle (default `0` = unthrottled, max `86400`). Messages suppressed during the cooldown are counted; the next delivered message for that class appends `(+N suppressed)` to the text.

### Telegram Linking Flow

```
1. Admin     POST /v1/admin/notifications/telegram/link        →  { deep_link, expires_at }
2. Admin     opens deep_link (https://t.me/<bot>?start=<token>) and taps Start in Telegram
3. Telegram  sends "/start <token>" to the bot
4. Backend   confirms the token (single-use), stores the chat_id, replies "✅ Telegram linked..."
5. Backend   delivers every subscribed notification class to that chat_id from then on
```

- Link tokens expire after `TELEGRAM_LINK_TOKEN_TTL_SECONDS` (default `900`s / 15 min) and are single-use — replaying a `/start <token>` after it's been consumed (or expired) gets "⚠️ Link token is invalid or expired."
- Re-requesting a link while already linked **rotates the token but does not clear the existing chat_id** — it's a way to re-link (e.g. after losing access to the chat), not an implicit unlink.
- `DELETE /v1/admin/notifications/telegram` unlinks immediately and stops all further deliveries to that admin.

### Endpoints

All endpoints require **ADMIN or SUPERADMIN** role, except reading another admin's preferences, which is **SUPERADMIN-only**.

#### `GET /v1/admin/notifications/classes`

```
Response: [ { notification_class: string, scope: "product" | "platform", description: string }, ... ]
Status:   200 OK
Note:     Static catalog derived from the NotificationClass enum (§17) — no DB round-trip.
```

#### `GET /v1/admin/notifications/preferences`

```
Response: { items: [ { notification_class: string, min_interval_seconds: int }, ... ] }
Status:   200 OK
Note:     The caller's own subscribed set. A class absent from `items` means "not subscribed".
```

#### `PUT /v1/admin/notifications/preferences`

```
Request:  { items: [ { notification_class: string, min_interval_seconds?: int }, ... ] }
Response: { items: [ ... ] }   // the updated set, echoed back
Status:   200 OK
Errors:   400 validation_error (unknown notification_class, or min_interval_seconds outside [0, 86400])
Note:     Full-set replace, idempotent. Send the complete desired set every time —
          omitted classes are unsubscribed, not left untouched.
```

#### `GET /v1/admin/notifications/preferences/{user_id}` *(SUPERADMIN only)*

```
Response: { items: [ ... ] }   // read-only view of another admin's preferences
Status:   200 OK
Errors:   403 forbidden (caller is ADMIN, not SUPERADMIN)
```

#### `GET /v1/admin/notifications/telegram`

```
Response: { linked: boolean, linked_at: string | null, chat_id_last4: string | null }
Status:   200 OK
Note:     chat_id_last4 is the last 4 digits of the Telegram chat id, for the admin to
          confirm which chat is linked without exposing the full id.
```

#### `POST /v1/admin/notifications/telegram/link`

```
Response: { deep_link: string, expires_at: string }
Status:   200 OK
Errors:   503 (Telegram not configured — no TELEGRAM_BOT_TOKEN/REDIS_URL)
Note:     Creates a new link token (or rotates an existing one) and returns the
          ready-to-open t.me deep link.
```

#### `DELETE /v1/admin/notifications/telegram`

```
Response: { message: string }
Status:   200 OK
Note:     Idempotent — succeeds even if no link exists.
```

### Message Format

Messages use Telegram's `parse_mode=HTML`, always prefixed with the product tag — `[vex]`, `[synthara]`, or `[platform]` for the platform-scoped health classes. Every interpolated value (ids, statuses) is HTML-escaped. Example:

```
[vex] ❌ Generation failed
job <code>3fa85f64-...</code> · grok/t2i
```

### Delivery Guarantees

- **Best-effort**, same as SSE (§15) and Push (§15b) — if the Telegram dispatcher process is down, or Redis drops, notifications are lost. No replay, no persistence queue.
- **Role and active-status are checked at delivery time** against the live user row, not cached on the subscription — a demoted or deactivated admin stops receiving immediately, with no preference cleanup required.
- **Throttling is per-process, in-memory** — `min_interval_seconds` cooldowns reset on a dispatcher restart. Accepted tradeoff for v1 rather than adding another Redis-backed key namespace.

---

## 16. Product Reference

### Products

| Slug | Display Name | Domains | Content Rating | Age Gate | Payment Providers | Org Feature |
|------|-------------|---------|---------------|----------|------------------|-------------|
| `vex` | Vex | vex-domain.com, www.vex-domain.com, app.vex-domain.com | permissive | date_of_birth | Stripe + NowPayments | No |
| `synthara` | Synthara | synthara-domain.com, www.synthara-domain.com, app.synthara-domain.com | sfw | none | Stripe only | Yes |

### AgeGatePolicy

Values: `"none"`, `"checkbox"`, `"date_of_birth"`

### ContentRating

Values: `"sfw"`, `"permissive"`

---

## 17. Enums Reference

### ModelType

| Value | Provider | T2I | I2I | T2V | I2V | V2V | FLF2V | Max Images | Age-gated |
|-------|----------|-----|-----|-----|-----|-----|-------|-----------|-----------|
| `grok-imagine-image` | grok | ✓ | ✓ | | | | | 10 | |
| `grok-2-image-1212` | grok | ✓ | | | | | | 10 | |
| `grok-imagine-video` | grok | | | ✓ | ✓ | ✓ | | 1 | |
| `aisha-image` | aisha | ✓ | ✓ | | | | | 4 | ✓ |
| `aisha-video` | aisha | | | ✓ | ✓ | | ✓ | 1 | ✓ |

**Age-gated** = `requires_age_verification=true` (exposed on `GET /v1/providers` → `ModelInfo`). The user must be age-verified via `PATCH /v1/users/me` before `POST /v1/generate` will accept the model; otherwise `403 age_verification_required`. The flag is per-model and authoritative regardless of the product's `age_gate` policy.

**Capability corrections vs. the previous revision of this doc:**
- `grok-imagine-video` does **not** support `flf2v` (Grok has no first-last-frame mode). Max video duration 15 s.
- `aisha-video` does **not** support `v2v` yet. It supports `t2v`, `i2v`, `flf2v`; max duration 10 s; aspect ratios limited to `1:1`, `16:9`, `9:16`.
- `aisha-image` exposes quality tiers (`draft`/`standard`/`high`/`ultra`, default `standard`) plus explicit `width`/`height`; `min_height` 256, `max_height` 2048, `default_height` 1024.

**Seeded enablement (default `is_enabled`; admins toggle via `PATCH /v1/admin/models/{model_key}`):**

| Model | Seeded `is_enabled` | Reason |
|-------|---------------------|--------|
| `grok-imagine-image` | `true` | Flagship image model |
| `grok-imagine-video` | `true` | Active video model |
| `grok-2-image-1212` | `false` | EOL after Grok Imagine Image; kept for reference/fallback |
| `aisha-image` | `false` | Seeded off; enable once the GPU image workflow is signed off |
| `aisha-video` | `false` | Seeded off until the video workflow is production-ready |

> `GET /v1/providers` reflects the **live** `is_enabled` value (not the seed), filtered by product.

### GenerationType

| Value | Description | Requires image? | Requires video? | Is video output? |
|-------|------------|-----------------|-----------------|------------------|
| `t2i` | Text → Image | No | No | No |
| `i2i` | Image → Image | Yes | No | No |
| `t2v` | Text → Video | No | No | Yes |
| `i2v` | Image → Video | Yes | No | Yes |
| `v2v` | Video → Video | No | Yes | Yes |
| `flf2v` | First-Last Frame → Video | Yes | No | Yes |

### JobStatus

| Value | Terminal? | Description |
|-------|----------|-------------|
| `pending` | No | Created, awaiting processing |
| `queued` | No | In queue |
| `running` | No | Actively generating |
| `completed` | Yes | Done, outputs available |
| `failed` | Yes | Error occurred |
| `cancelled` | Yes | User or system cancelled |
| `moderated` | Yes | Content moderated by provider |

**Polling strategy:** Poll `GET /v1/jobs/{id}` every 2s while status is `pending`, `queued`, or `running`. Stop on any terminal status. A Grok video GET settles terminal provider results itself using the same guarded path as the worker, so this remains correct when no worker is deployed. For real-time updates without polling, subscribe to the SSE `job.status_changed` event (see [§15 Real-Time Events](#15-real-time-events-sse--pubsub)).

### GpuSessionStatus

| Value | Terminal? | Description |
|-------|----------|-------------|
| `pending` | No | Session requested, Vast.ai node not yet provisioning |
| `provisioning` | No | Vast.ai node is starting up; ComfyUI not yet reachable |
| `active` | No | Node is up, ComfyUI is reachable — generations can be submitted |
| `stale` | No | Node was active but the latest health probe failed; auto-recovers if a subsequent probe succeeds |
| `paused` | No | User paused — Vast.ai instance stopped, persistent disk retained |
| `resuming` | No | User resumed — Vast.ai instance restarting |
| `stopping` | No | User-requested stop — teardown in progress |
| `stopped` | Yes | Session ended normally |
| `failed` | Yes | Provisioning or runtime failure (`error_message` populated) |

> Use the `gpu_session.status_changed` SSE event for real-time UI updates rather than polling.

### ProvisioningMode

How a provider's compute is made available. Surfaced as `ProviderInfo.provisioning_mode` on `GET /v1/providers`.

| Value | Description |
|-------|-------------|
| `always_on` | Cloud API — usable immediately whenever the provider is configured (e.g. Grok / xAI) |
| `on_demand` | Per-user GPU session required — start one via `POST /v1/sessions` before generating (e.g. Aisha / ComfyUI) |

### ModelSessionState

Per-user readiness of an `on_demand` model. Surfaced as `ModelInfo.session_state` on `GET /v1/providers` for authenticated requests only. Always `null` for `always_on` providers and unauthenticated callers.

| Value | Description |
|-------|-------------|
| `none` | No live session for this model — start one via `POST /v1/sessions` |
| `provisioning` | Session exists and is starting up (`pending` / `provisioning` / `resuming`) |
| `active` | Session is active and ComfyUI is reachable — generations can be submitted |
| `paused` | Session is paused (instance stopped, disk retained) — resume via `POST /v1/sessions/{id}/resume` |
| `stale` | Session was active but the last health probe failed; may self-recover |

> Prefer the `gpu_session.status_changed` SSE event for real-time state changes rather than polling `GET /v1/providers`.

### AspectRatio

Values: `"1:1"`, `"16:9"`, `"9:16"`, `"4:3"`, `"3:4"`, `"2:3"`, `"3:2"`

> `aspect_ratio` on `POST /v1/generate` (§4) is optional. Omitting it means "provider default" for
> t2i and "follow the source image's aspect" for i2i. For i2i, an explicit value is only accepted for
> models whose `edit_aspect_ratios` (`GET /v1/providers`, §5) includes it — see §4 for the full gate.

### VideoResolution

Values: `"480p"`, `"720p"`

> Used for the `resolution` field on video generations. Distinct from the image-quality `Resolution` tier below.

### Resolution *(image quality tier)*

Image quality tier for Aisha image generation. Sent as `image_resolution` on `POST /v1/generate`. Each tier maps to a target **megapixel budget**; the backend computes concrete `width × height` for the requested aspect ratio (snapped to the model's latent multiple and clamped to its max edge / max megapixels).

| Value | Target megapixels |
|-------|-------------------|
| `draft` | 0.25 MP |
| `standard` | 1.0 MP (default) |
| `high` | 2.0 MP |
| `ultra` | 4.0 MP |

> `tier_megapixels` in `GET /v1/providers` → `ImageConstraints` echoes this mapping per model. Mutually exclusive with explicit `width`/`height`.

### Sampler *(Aisha sampler override)*

ComfyUI sampler names accepted on `POST /v1/generate` (`sampler`, Aisha image only):

`euler`, `euler_ancestral`, `euler_cfg_pp`, `heun`, `dpm_2`, `dpm_2_ancestral`, `lms`, `dpmpp_2s_ancestral`, `dpmpp_sde`, `dpmpp_2m`, `dpmpp_2m_sde`, `dpmpp_3m_sde`, `ddim`, `uni_pc`, `uni_pc_bh2`, `lcm`, `res_multistep`

### Scheduler *(Aisha scheduler override)*

ComfyUI scheduler names accepted on `POST /v1/generate` (`scheduler`, Aisha image only):

`normal`, `karras`, `exponential`, `sgm_uniform`, `simple`, `ddim_uniform`, `beta`, `linear_quadratic`, `kl_optimal`

### MediaFormat

Values: `"png"`, `"jpeg"`, `"webp"` (images), `"mp4"`, `"webm"`, `"mov"` (video)

> Surfaced as the `format` field on job/library outputs. Generated image thumbnails are `webp`. `"webm"`/`"mov"` apply only to user-uploaded videos (§8/§9b) — generated video outputs are always `"mp4"`.

### AccountType

Values: `"personal"`, `"enterprise"`

### NotificationLevel

Values: `"info"`, `"warning"`, `"critical"`

> Shared severity enum used in `SystemNotificationPayload` (admin broadcast) and `GpuSessionCreditWarningPayload` (credit guard).

### TransactionType

Values: `"debit"`, `"credit"`, `"refund"`, `"admin_adjustment"`

### SubscriptionTier

Values: `"free"`, `"basic"`, `"pro"`, `"enterprise"`

### UserRole

Values: `"superadmin"`, `"admin"`, `"user"`

> `"system"` is an internal sentinel role — never returned by any API endpoint.
>
> `"superadmin"` has all admin capabilities plus the ability to manage roles and permissions. See [§14 Admin Management](#14-admin-management-authenticated--superadmin-only).

### AdminPermission

Values: `"billing_adjust"`

> Granular permissions that a superadmin can grant to ADMIN-role users. Currently only `billing_adjust` exists (enables `POST /v1/admin/accounts/{id}/adjust` for that admin).

### NotificationClass

| Value | Scope | Fires when |
|-------|-------|------------|
| `user.registered` | product | New user registration |
| `generation.created` | product | New generation job submitted |
| `gpu_node.started` | product | GPU node finishes provisioning (`provisioning → active` only) |
| `generation.failed` | product | Generation job transitions to `failed` (excludes provider-authentication failures, reported separately below) |
| `provider_authentication.failed` | platform | A generation provider rejected its API key/authentication |
| `health.degraded` | platform | A health subsystem becomes `degraded`/`unhealthy`/`unknown` |
| `health.restored` | platform | A health subsystem recovers |
| `token_revocation.failed` | platform | A bulk access-token revocation failed to write to Redis |
| `push_subscriptions.cleanup_failed` | platform | A bulk-revocation event's push-subscription cleanup failed |

> Admin ops-notification subscription classes — see [§15c Admin Ops Notifications (Telegram)](#15c-admin-ops-notifications-telegram) for the full subscribe/throttle/delivery model.

### OrgRole

Values: `"owner"`, `"admin"`, `"member"`

### PaymentStatus

Values: `"pending"`, `"partially_paid"`, `"completed"`, `"failed"`, `"refunded"`

### SupportedLocale

Values: `"en"` (English), `"ru"` (Russian), `"sr"` (Serbian Latin)

### OutputMediaType

Values: `"image"`, `"video"`

Used in Library to distinguish image vs. video assets and generation groups.

### LibraryAssetSource

Values: `"upload"`, `"output"`

Which table a Library asset lives in (`user_images` vs. `generation_outputs`). Prefixes every
`asset_ref` on the wire (`"upload:<uuid>"` / `"output:<uuid>"`) and is the `source=` filter value
on `GET /v1/library/`.

### LibrarySort

Values: `"newest"` (default), `"oldest"`, `"expiring_soon"`

Sort order for `GET /v1/library/`. `expiring_soon` orders ascending by `expires_at` (soonest first)
and is also the axis the `expiring` filter checks against (7-day fixed window).

### LibraryAction

Values: `"remix"`, `"create_variation"`, `"animate"`, `"extend"`, `"extract_frame"`,
`"use_as_reference"`, `"use_as_first_frame"`, `"use_as_last_frame"`, `"view_settings"`,
`"reproduce"`, `"favorite"`, `"rename"`, `"download"`, `"delete"`

Server-resolved per-asset action set — see the "Library Action Resolution" table in §10.

### LibraryBadge

Values: `"prompt"`, `"image"`

Indicates the primary input type for a library group:
- `"prompt"` — text-to-image or text-to-video (no image input)
- `"image"` — image/video input types (i2i, i2v, flf2v, v2v)

### LibraryGroupSourceType

Values: `"upload"`, `"output"`

Used in `LibraryGroupLineage.source_type` (§10) to indicate whether a generation job's input came
from a direct upload or a previous generation output.

### FrameExtractionKind

Values: `"preview"`, `"extract"`

The `kind` field on a video frame extraction job (§9b) — which of the two flows a job performs.

### FrameExtractionStatus

| Value | Terminal? | Description |
|-------|----------|-------------|
| `queued` | No | Job created, awaiting the worker |
| `running` | No | Worker has claimed the job and is running ffmpeg |
| `completed` | Yes | Done — `preview`/`extracted` populated on `GET /v1/frames/jobs/{id}` |
| `failed` | Yes | Error occurred — `error` populated with a human-readable message |

**Polling strategy:** Poll `GET /v1/frames/jobs/{id}` (§9b) every ~1s while status is `queued` or `running`. Jobs are short-lived (typically low single-digit seconds) — no SSE variant exists for this.

---

## 18. Error Response Format

Non-2xx responses normally use a single unified envelope:

```typescript
interface ApiError {
  error: string;        // machine-readable code (snake_case)
  message: string;      // human-readable, safe to show in UI
  status_code: number;  // mirrors the HTTP status
  detail?: Record<string, unknown> | null;  // optional structured context
}
```

The `error` code is always a stable snake_case string — treat it like an enum. Common values:

| HTTP | `error` | `detail` keys |
|------|---------|---------------|
| 400 | `bad_request`, `email_exists`, `invalid_token`, `invalid_password`, `validation_error`, `empty_file`, `file_too_large`, `invalid_file_type`, `upload_failed`, `payment_verification_failed`, `model_disabled`, `generation_failed`, `provider_invalid_request`, `unknown_product` | — |
| 401 | `unauthorized`, `invalid_credentials`, `account_inactive`, `token_reuse_detected` | — |
| 402 | `insufficient_balance` | `balance`, `required` |
| 403 | `forbidden`, `account_inactive`, `permission_denied`, `model_not_allowed`, `age_verification_required` | — |
| 404 | `not_found`, `account_not_found`, `price_not_found` | — |
| 409 | `conflict`, `refund_not_eligible`, `organization_balance_nonzero`, `no_active_gpu_session`, `session_already_exists`, `invalid_state`, `jobs_in_flight` | `balance`, `in_flight_count` |
| 422 | `validation_error`, `moderation`, `provider_moderation_rejected` | `provider`, `policy` (Apex moderation only) |
| 429 | `too_many_requests`, `rate_limited`, `provider_rate_limited` | `retry_after` (global rate limit only) |
| 502 | `provider_malformed_response`, `provider_output_not_delivered` | — |
| 503 | `service_unavailable`, `no_gpu_capacity`, `provisioning_failed`, `provider_timeout`, `provider_unavailable`, `provider_execution_failed`, `generation_session_terminated`, `provider_authentication_failed`, `provider_unknown` | — |

**Example responses:**

```json
// 401
{ "error": "invalid_credentials", "message": "Invalid email or password", "status_code": 401, "detail": null }

// 402
{ "error": "insufficient_balance", "message": "Insufficient balance: have 50, need 100", "status_code": 402, "detail": { "balance": 50, "required": 100 } }

// 404
{ "error": "not_found", "message": "Job not found", "status_code": 404, "detail": null }

// 422
{ "error": "moderation", "message": "Content moderated by grok (policy: nsfw)", "status_code": 422, "detail": { "provider": "grok", "policy": "nsfw" } }

// 422 — provider-side Grok moderation; upstream text is never exposed
{ "error": "provider_moderation_rejected", "message": "The requested content was rejected by the AI provider's safety system. Modify the prompt or input and try again.", "status_code": 422, "detail": null }
```

Provider disablement is the one deliberately compact compatibility response used by both top-up
routes:

```json
{ "code": "payment_provider_disabled", "provider": "stripe" }
```

**Frontend usage:**

```typescript
async function apiFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const err: ApiError = await res.json();
    throw err;  // catch by err.error code, display err.message
  }
  return res.json();
}
```

---

## 19. Content URLs

### Content Proxy URLs (preferred for Library / persistent UI)

Library responses and the `/v1/content/` endpoints return **content proxy URLs** — permanent, auth-gated paths:

- `GET /v1/content/outputs/{output_id}` — streams a generated output
- `GET /v1/content/uploads/{image_id}` — streams an uploaded image

These URLs:
- Are **stable** for the lifetime of the resource (no expiry)
- Return `Cache-Control: private, max-age=10800, immutable` (3-hour client cache, configurable via `CONTENT_URL_TTL`)
- Enforce ownership and product scoping on every request
- Are suitable for `<img src>`, `<video src>`, or background image CSS

**Frontend caching:** Because responses are `immutable`, browsers will serve cached bytes without revalidating for the `max-age` window. Use these URLs directly in `<img>` and `<video>` tags.

### Presigned URLs (jobs / storage endpoints)

- All R2 presigned URLs returned by `/v1/jobs` and `/v1/storage` endpoints are valid for **~1 hour** by default
- Do **not** aggressively cache them — use `staleTime` of ~30 minutes in TanStack Query
- URLs are generated on-demand when fetching jobs/outputs/uploads
- R2 storage key pattern:
  - Uploads: `users/{user_id}/uploads/{file_id}.{ext}`
  - Outputs: `users/{user_id}/outputs/{job_id}/{file_id}.{ext}`

---

## 20. Rate Limits

| Endpoint | Limit |
|----------|-------|
| `POST /auth/register` | 5/hour per IP |
| `POST /auth/login` | 10/minute per IP |
| `POST /auth/forgot-password` | 3/hour per IP |
| `POST /auth/resend-verification` | 3/hour per IP |
| `POST /v1/events/sse-ticket` | 10/minute per user |

Rate limit headers are **not currently exposed** in responses. The frontend should handle 429 responses gracefully with a user-friendly message.

---

## 21. Health Check

Three-tier health monitoring system. Use the appropriate endpoint for each consumer:

| Consumer | Endpoint | Auth |
|----------|----------|------|
| Docker HEALTHCHECK, load balancers | `GET /health/live` | None |
| CI readiness waits, traffic routing | `GET /health/ready` | None |
| Admin dashboards, monitoring | `GET /v1/admin/health/` | Admin JWT |
| Admin real-time stream | `GET /v1/admin/health/stream` | Admin JWT |
| Admin historical charts | `GET /v1/admin/health/history` | Admin JWT |

#### `GET /health/live`

Always returns 200 if the process is serving HTTP. Used by Docker HEALTHCHECK.

```
Response: { status: "alive" }
Note:     Always 200 — use this for Docker/container restart decisions only.
```

#### `GET /health/ready`

Checks PostgreSQL and Redis connectivity. Returns 200 if ready, 503 if not.

```
Response: { status: "ready" | "not_ready", checks: { postgres: string, redis?: string, r2?: string }, build_sha: string }
Status:   200 if ready, 503 if not ready
Note:     R2 is checked but excluded from the ready/not_ready determination (slow HeadBucket).
          Use this for CI readiness waits and traffic-routing decisions.
          build_sha is the git SHA baked into the running image (Settings.build_sha,
          default "unknown" if not injected at build time) — populated on both the
          ready and not-ready branches. The staging deploy workflow polls this field
          to confirm a redeploy actually swapped in the new image, not just that some
          container is answering on the ready path.
```

#### `GET /v1/admin/health/`

Full system health across all categories. Requires admin authentication.

```
Request:  Authorization: Bearer <admin_token>
Response: {
  status: "healthy" | "degraded" | "unhealthy" | "unknown",
  checked_at: string,         // ISO 8601
  infrastructure: {
    status: string,
    components: [{ name, status, latency_ms, message?, metadata? }]
    // Registered: postgres, redis (if configured), r2 (if configured)
  },
  platform_apis: {
    status: string,
    components: [{ name, status, latency_ms, message?, metadata? }]
    // Registered: vastai_api (inactive if VASTAI_API_KEY not set)
  },
  cloud_providers: {
    [product_id]: {
      status: string,
      components: [{ name, status, latency_ms, message?, metadata? }]
    }
    // Registered per product that uses Grok: { vex: { grok }, synthara: { grok } }
    // Only populated when XAI_API_KEY is configured.
    // Uses REST GET /v1/models probe (not gRPC) for lightweight health check.
  },
  gpu_sessions: {
    status: "healthy" | "degraded" | "unhealthy" | "inactive",
    total: number,    // active + stale sessions probed
    healthy: number,  // sessions with reachable ComfyUI endpoint
    stale: number,    // sessions that failed the /object_info probe
    message: string
  }
  // Phase 3: GpuSessionReconciler probes all active/stale sessions concurrently.
  // Returns "inactive" when no sessions exist; "degraded" when some are unreachable;
  // "unhealthy" when all are unreachable; "healthy" when all are reachable.
}
Status:   200
Errors:   401 (unauthorized), 403 (not admin)
```

Component `status` values: `healthy`, `degraded`, `unhealthy`, `unknown`, `inactive`.

**Status semantics for cloud providers / platform APIs:**

| HTTP response | Status | Meaning |
|---|---|---|
| 2xx, 3xx | `healthy` | API up, key valid |
| 401, 403 | `degraded` | API reachable, authentication failed — check API key |
| 429 | `healthy` | Rate-limited = API alive, transient condition |
| other 4xx | `degraded` | API reachable but returning unexpected client errors |
| 5xx | `unhealthy` | Server-side failure |
| Connection error | `unhealthy` | API unreachable |
| Key not set / whitespace-only | `inactive` | Not configured (VASTAI_API_KEY or XAI_API_KEY not set) |

Note: 401/403 returns immediately without trying fallback probes — if auth is wrong, all probes will fail the same way.

#### `GET /v1/admin/health/stream`

SSE stream of real-time health snapshots. Emits a `health.snapshot` event at each snapshot interval (default 60s), with `: keepalive` comments between events.

```
Request:  Authorization: Bearer <admin_token>
          (Use fetch() with ReadableStream — EventSource cannot send headers)
Response: Content-Type: text/event-stream

event: health.snapshot
data: { ...same structure as GET /v1/admin/health/ ... }

: keepalive          <- sent every 15s when no snapshot arrives

Status:   200 (streaming)
Errors:   401 (unauthorized), 403 (not admin)
Note:     Subscribes to Redis channel "health:stream". Falls back to direct polling
          when Redis is not configured.
```

#### `GET /v1/admin/health/history`

Historical health snapshots for dashboard charts. Stored by `HealthSnapshotWorker` each interval.

```
Request:  Authorization: Bearer <admin_token>
Params:   after  — ISO 8601 datetime, only snapshots after this time (optional)
          before — ISO 8601 datetime, only snapshots before this time (optional)
          limit  — max results, default 60, clamped to [1, 1440] (optional)
Response: [
  {
    checked_at: string,       // ISO 8601
    overall_status: string,   // healthy | degraded | unhealthy
    snapshot_data: object     // full DetailedHealthResponse dict
  },
  ...
]
Status:   200
Errors:   400 (malformed after/before datetime), 401 (unauthorized), 403 (not admin)
Note:     Results ordered by checked_at DESC. Default limit 60 = 1 hour at 1/min interval.
          Snapshots are retained for HEALTH_SNAPSHOT_RETENTION_DAYS (default 30 days).
          Trailing Z suffix is accepted (e.g. 2026-03-31T14:00:00Z).
```

**GPU session reconciler behaviour:**

The `gpu_sessions` section is populated by `GpuSessionReconciler`. On each health check cycle it:
1. Queries all `gpu_sessions` rows with `status IN ('active', 'stale')`.
2. Probes each node's `GET /object_info` endpoint (10 s timeout) concurrently.
3. Unreachable sessions → marked `stale` in DB (`stale_detected_at` set, `status = 'stale'`). Already-stale sessions are not re-marked (idempotent).
4. Previously-stale sessions that become reachable → cleared (`stale_detected_at = null`, `status = 'active'`) to self-heal transient network blips.

The registry timeout for this checker is 15 s (increased from the 5 s default for infrastructure checks) to accommodate concurrent 10 s probes.

---

## 22. OpenAPI Documentation Endpoints

**Gated behind `ENABLE_DOCS` — off by default.** The OpenAPI schema and every
endpoint under `/docs/` enumerate the full API surface, including
`/v1/internal/*` and admin routes, which is free reconnaissance if left
public. `Settings.enable_docs` (env: `ENABLE_DOCS`) defaults to `false`; when
false, `create_app()` passes `openapi_config=None` and every path below
returns 404. Set `ENABLE_DOCS=true` in dev/staging to expose them — **never
enable in production.** This is independent of `DEBUG`.

When enabled, the backend's `OpenAPIConfig` is configured with `path="/docs"`, so all schema and documentation UI endpoints live under `/docs/`:

| Endpoint | Description |
|----------|-------------|
| `GET /docs/openapi.json` | OpenAPI 3.1 schema (JSON) — **use this for type generation** |
| `GET /docs/openapi.yaml` | OpenAPI 3.1 schema (YAML) |
| `GET /docs/swagger` | Swagger UI |
| `GET /docs/redoc` | ReDoc UI |
| `GET /docs/elements` | Stoplight Elements UI |
| `GET /docs/rapidoc` | RapiDoc UI |

**Export command** (requires `ENABLE_DOCS=true` on the target server):
```bash
curl http://localhost:8000/docs/openapi.json > src/lib/api/schema.json
npx openapi-typescript src/lib/api/schema.json -o src/lib/api/types.ts
```
