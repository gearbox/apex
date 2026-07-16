# Backend API Reference — Apex REST API

> _Last updated: 2026-07-16 — Added the DB-cached **Payment Currency Catalog**: public `GET /v1/billing/currencies` (§11) returns available tickers with display name/network/R2-hosted logo, synced from NowPayments `merchant/coins` (availability) + `full-currencies` (metadata) by a periodic worker (default 3h) and on-demand via superadmin `GET/POST /v1/admin/payments/currencies[/refresh]` (§14). Empty list ⇒ hide the picker and omit `pay_currency`; catalog state never gates checkout. No hardcoded ticker list exists anywhere in the contract. Frontend should regenerate types (`gen:api`)._
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
6. If a key is stuck `processing` (the original request's connection dropped, its worker crashed, etc.) for longer than `IDEMPOTENCY_PROCESSING_STALE_SECONDS` (default 120s), the **next** retry with the same key reclaims it and proceeds — it does not need to wait out the full 24h TTL. A retry within that window still gets `409 idempotency_conflict` (treated as a genuinely concurrent in-flight request).

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
| **vex.pics** | `vex` | `vex.pics`, `www.vex.pics`, `app.vex.pics` | Consumer / creator | Permissive — NSFW-capable models available |
| **Synthara** | `synthara` | `synthara.app`, `www.synthara.app`, `app.synthara.app` | Enterprise / business | SFW only, professional |

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
  display_name: string,         // e.g. "vex.pics"
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
Response: { access_token, refresh_token, token_type: "bearer", expires_in: int, expires_at: datetime }
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
Response: { access_token, refresh_token, token_type: "bearer", expires_in: int, expires_at: datetime }
Errors:   401 (invalid_credentials | account_inactive)
```

#### `POST /v1/auth/refresh`

```
Request:  { refresh_token: string }
Response: { access_token, refresh_token, token_type: "bearer", expires_in: int, expires_at: datetime }
Errors:   401 (token revoked/expired/invalid | token_reuse_detected | account_inactive)
```

#### `POST /v1/auth/logout`

```
Request:  { refresh_token: string }
Response: { message: string }
Note:     Revokes the specific refresh token
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
```

#### `POST /v1/auth/resend-verification` *(authenticated)*

```
Response: { message: string }
Rate:     3/hour
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
Note:     Revokes ALL refresh tokens
```

#### `DELETE /v1/users/me`

```
Response: { message: string, deactivated_at: datetime }
Note:     Soft delete — account can be recovered
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
Errors:   400 (model_disabled | validation_error | generation_failed | not_implemented), 402 insufficient_balance, 403 (model_not_allowed | age_verification_required), 409 (idempotency_conflict | no_active_gpu_session), 429 rate_limited, 503 service_unavailable
Headers:  Idempotency-Key: <string> (required, max 64 chars)
Note:     source_output_id enables "remix from gallery" — the backend resolves lineage automatically
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

Models are **filtered by the current product** — Synthara only returns SFW-safe models; vex.pics returns all enabled models.

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
  error: string | null;
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

> **Retention:** every upload and output row carries `expires_at`, set at creation to `now +
> RETENTION_DAYS` (default 7 days). A periodic background sweeper (`ContentRetentionWorker`)
> deletes expired rows and their R2 objects on a fixed interval. Once swept: the item drops out
> of Storage/Gallery list responses, and `GET /v1/content/...` (§9) for that ID returns `404`.
> `expires_at` is a plain timestamp (not a countdown) so the frontend can derive and tick a
> "Delete in N days/hours/minutes" badge client-side — see `ImageListItem`/`OutputListItem`
> below and `GalleryGridItem`/`GalleryOutputItem` (§10).

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

#### `GET /v1/storage/uploads`

```
Query:    limit? (1–100, default 50), cursor? (opaque token)
Response: CursorPage<ImageListItem>

ImageListItem: {
  id: UUID,
  filename: string,
  created_at: datetime,
  expires_at: datetime,
  media: MediaObject    // original + sm/md WEBP variants
}
```

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

> **Why use this instead of presigned URLs?** Content proxy URLs are permanent (for the lifetime of the resource), cacheable with `Cache-Control: private, max-age=<ttl>, immutable`, and enforce per-request authorization. They are the preferred URL format for Gallery and any UI that persists content references.

### Response Headers

All successful responses include:
- `Content-Type` — the stored R2 `ContentType` **only if it's on the inline-safe allowlist** (`image/png`, `image/jpeg`, `image/webp`, `video/mp4`, `video/webm`, `video/quicktime`); otherwise `application/octet-stream`
- `Content-Length` — from R2 object metadata
- `Cache-Control: private, max-age=10800, immutable` — 3-hour client cache (default; configurable via `CONTENT_URL_TTL`)
- `ETag: "<content_id>"` — the output/upload UUID, for conditional requests
- `X-Content-Id: <content_id>` — same UUID, without quotes
- `X-Content-Type-Options: nosniff` — always present, blocks MIME-sniffing
- `Content-Disposition: inline` for inline-safe content types, `attachment` otherwise — a stored content-type outside the allowlist (e.g. `text/html`, `image/svg+xml`) is forced to download rather than rendered inline, even though it streams with a coerced `Content-Type`

#### `GET /v1/content/outputs/{output_id}`

```
Path:     output_id (UUID)
Response: 200 Raw bytes (chunked streaming, appropriate Content-Type)
Errors:   404 not_found (ownership check failed or wrong product),
          502 upstream_error (R2 fetch failed)
Note:     Only returns outputs owned by the authenticated user and matching the current product.
```

#### `GET /v1/content/uploads/{image_id}`

```
Path:     image_id (UUID)
Response: 200 Raw bytes (chunked streaming, appropriate Content-Type)
Errors:   404 not_found (ownership check failed or wrong product),
          502 upstream_error (R2 fetch failed)
Note:     Only returns uploads owned by the authenticated user and matching the current product.
```

#### `DELETE /v1/content/{content_id}`

```
Path:     content_id (UUID) — can be a generation output ID or upload ID
Response: 204 No Content
Errors:   404 not_found (content does not exist, not owned, or wrong product)
Note:     Permanently deletes the file from R2 and removes the DB record.
          Checks generation_outputs first, then user_images.
          Lineage references (source_output_id, input_image_id) are SET NULL automatically.
```

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
          same delete (DELETE /v1/content/{id}, §9), same retention/expiry.
          Deleting the source video does NOT delete frames already extracted
          from it.
```

---

## 10. Gallery *(authenticated)*

Gallery presents completed generation jobs as a visual grid. Each **gallery item** is one `GenerationJob` (a "group") with its cover image/video, metadata, and output list.

- Only `completed` jobs are returned.
- Results are ordered by `created_at DESC` (newest first).
- Uses the same **cursor pagination** as all other list endpoints.
- Content URLs in responses are always `/v1/content/...` paths (permanent, auth-gated).

#### `GET /v1/gallery/`

```
Query:    limit? (1–25, default 20),
          cursor? (opaque token),
          media_type? ("image" | "video"),
          generation_type? (GenerationType value),
          model? (string — model key)
Response: CursorPage<GalleryGridItem>
```

#### `GET /v1/gallery/{job_id}`

```
Path:     job_id (UUID)
Response: GalleryGroupDetail
Errors:   404 not_found (job not completed, wrong user, or wrong product)
```

### Gallery Schemas

```typescript
interface GalleryGridItem {
  job_id: string;           // UUID
  cover: MediaObject;       // always the job's own primary output + sm/md WEBP variants
                            // for video: original is the MP4; variants are poster frames
  badge: GalleryBadge;      // "prompt" (t2i/t2v) or "image" (i2i/i2v/flf2v/v2v)
  output_count: number;     // non-thumbnail outputs in this group
  generation_type: GenerationType;
  model: string | null;
  aspect_ratio: string | null; // e.g. "16:9"; null ⇒ i2i job that followed the source image's aspect
  prompt_snippet: string;   // first 100 chars of the prompt
  created_at: string;       // ISO datetime
  expires_at: string;       // ISO datetime — sourced from the cover output; all outputs in a
                            // group share the same retention window. Drive a "Delete in N
                            // days/hours/minutes" badge from this client-side (server sends a
                            // timestamp, not a countdown, so it survives caching and ticks live).
}

interface GalleryGroupDetail {
  job_id: string;           // UUID
  // Header
  badge: GalleryBadge;
  input_media: MediaObject | null; // source input envelope when badge == "image"
                                   // (remixed output or uploaded input image + variants)
  prompt: string;
  negative_prompt: string | null;
  // Outputs
  outputs: GalleryOutputItem[];    // non-thumbnail outputs, ordered by output_index
  // Metadata
  media_type: OutputMediaType;     // "image" or "video"
  model: string | null;
  provider: string;
  generation_type: GenerationType;
  aspect_ratio: string | null;  // null ⇒ i2i job that followed the source image's aspect
  token_cost: number | null;
  created_at: string;
  completed_at: string | null;
  // Lineage
  lineage: GalleryLineage | null;
}

interface GalleryOutputItem {
  id: string;               // UUID
  output_index: number;     // 0-based
  created_at: string;
  expires_at: string;       // ISO datetime — this output's own retention deletion timestamp
  media: MediaObject;       // original asset + sm/md WEBP variants
}

interface GalleryLineage {
  source_type: GallerySourceType;    // "upload" or "generation"
  source_upload_id: string | null;   // UUID; set when source_type == "upload"
  source_job_id: string | null;      // UUID; set when source_type == "generation"
  source_job_name: string | null;    // human-readable name of the source job
  source_output_id: string | null;   // UUID; specific output used as input
}
```

> **Breaking change (2026-06-27):** `GalleryGridItem` no longer has `cover_url`, `video_url`,
> or `media_type` — use `cover: MediaObject` instead. The cover is now always the job's own
> primary output (stops N near-identical tiles for N generations from one input).
> `GalleryGroupDetail.input_image_url` → `input_media: MediaObject | null`.
> `GalleryOutputItem` drops `url`, `thumbnail_url`, `content_type`, `media_type`, `format`,
> `size_bytes` — all in `media: MediaObject`.

> **New field (2026-07-12, additive):** `expires_at` on `GalleryGridItem` and
> `GalleryOutputItem` — brings the gallery contract to parity with `ImageListItem`/
> `OutputListItem` (§8), which have exposed `expires_at` all along. `DELETE
> /v1/content/{content_id}` (§9) already supported uploads before this change — no new
> delete endpoint was added for the uploads-delete UI; reuse the existing one.

### Gallery Badge Logic

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
Note:     Same discount-on-price semantics as the Stripe path. NowPayments IPN
          under/overpayments are credited proportionally to actually_paid/amount_usd
          (uncapped on overpayment) — never held for manual review.

          pay_currency is optional. Pass a ticker (e.g. "usdcmatic") to pin the
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
default 3h) and on-demand by superadmin (`POST /v1/admin/payments/currencies/refresh`). Only
`is_available` rows are returned, ordered by ticker. No hardcoded ticker list exists anywhere in
this contract — availability is decided solely by the provider's own dashboard-checked list.

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
  action: string,   // role.*, permission.*, or payment_provider.enable/disable/reorder
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
Note:     Full catalog including unavailable rows, ordered by ticker. Unlike the public
          endpoint, this never filters by is_available — used to audit sync history.
```

#### `POST /v1/admin/payments/currencies/refresh`

```
Response: SyncResult[]   // one entry per catalog-capable provider in the product's capability set
Errors:   502 when any provider's merchant/coins or full-currencies call fails — the previously
          synced catalog and logos are left untouched (no partial sync is ever committed)
Note:     Synchronous — runs list_merchant_currencies + list_full_currencies + logo caching inline
          and commits before responding. Writes payment_currencies.refresh to the audit log with
          target_user_id=null and a detail blob of per-provider upserted/deactivated counts.
```

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
  previous_status: string;  // previous status (or "none" on first publish)
  generation_type: string;  // e.g. "t2v"
  provider: string;         // e.g. "grok"
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
```

#### `DELETE /v1/push/subscriptions` *(authenticated)*

```
Request: { endpoint: string }
Response: (empty body)
Status:   204 No Content
Errors:   401 unauthorized, 503 (push not configured)
Note:     Idempotent — returns 204 even if the endpoint was never registered, or
          already belongs to a different user (no ownership leak).
```

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
| `job.status_changed` | Only terminal states: `completed`, `failed` | `tag: "job-{job_id}"`, `url: "/gallery/{job_id}"` |
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

## 16. Product Reference

### Products

| Slug | Display Name | Domains | Content Rating | Age Gate | Payment Providers | Org Feature |
|------|-------------|---------|---------------|----------|------------------|-------------|
| `vex` | vex.pics | vex.pics, www.vex.pics, app.vex.pics | permissive | date_of_birth | Stripe + NowPayments | No |
| `synthara` | Synthara | synthara.app, www.synthara.app, app.synthara.app | sfw | none | Stripe only | Yes |

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

**Polling strategy:** Poll `GET /v1/jobs/{id}` every 2s while status is `pending`, `queued`, or `running`. Stop on any terminal status. For real-time updates without polling, subscribe to the SSE `job.status_changed` event (see [§15 Real-Time Events](#15-real-time-events-sse--pubsub)).

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

> Surfaced as the `format` field on job/gallery outputs. Generated image thumbnails are `webp`. `"webm"`/`"mov"` apply only to user-uploaded videos (§8/§9b) — generated video outputs are always `"mp4"`.

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

### OrgRole

Values: `"owner"`, `"admin"`, `"member"`

### PaymentStatus

Values: `"pending"`, `"partially_paid"`, `"completed"`, `"failed"`, `"refunded"`

### SupportedLocale

Values: `"en"` (English), `"ru"` (Russian), `"sr"` (Serbian Latin)

### OutputMediaType

Values: `"image"`, `"video"`

Used in Gallery to distinguish image vs. video generation groups and outputs.

### GalleryBadge

Values: `"prompt"`, `"image"`

Indicates the primary input type for a gallery item:
- `"prompt"` — text-to-image or text-to-video (no image input)
- `"image"` — image/video input types (i2i, i2v, flf2v, v2v)

### GallerySourceType

Values: `"upload"`, `"generation"`

Used in `GalleryLineage.source_type` to indicate whether the input came from a direct upload or a previous generation output.

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
| 400 | `bad_request`, `email_exists`, `invalid_token`, `invalid_password`, `validation_error`, `empty_file`, `file_too_large`, `invalid_file_type`, `upload_failed`, `payment_verification_failed`, `model_disabled`, `generation_failed`, `unknown_product` | — |
| 401 | `unauthorized`, `invalid_credentials`, `account_inactive`, `token_reuse_detected` | — |
| 402 | `insufficient_balance` | `balance`, `required` |
| 403 | `forbidden`, `account_inactive`, `permission_denied`, `model_not_allowed`, `age_verification_required` | — |
| 404 | `not_found`, `account_not_found`, `price_not_found` | — |
| 409 | `conflict`, `refund_not_eligible`, `organization_balance_nonzero`, `no_active_gpu_session`, `session_already_exists`, `invalid_state`, `jobs_in_flight` | `balance`, `in_flight_count` |
| 422 | `validation_error`, `moderation` | `provider`, `policy` |
| 429 | `too_many_requests`, `rate_limited` | `retry_after` |
| 503 | `service_unavailable`, `no_gpu_capacity`, `provisioning_failed` | — |

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

### Content Proxy URLs (preferred for Gallery / persistent UI)

Gallery responses and the `/v1/content/` endpoints return **content proxy URLs** — permanent, auth-gated paths:

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
