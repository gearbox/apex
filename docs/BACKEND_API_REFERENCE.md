# Backend API Reference — Apex REST API

> **Source:** `gearbox/apex` repository
> **Framework:** Litestar 2.5+ / Python 3.13
> **Schema:** `GET /docs/openapi.json` from running backend (Litestar OpenAPIConfig has `path="/docs"`)
> **Last synced:** 2026-04-01

This document captures the API surface that the frontend depends on. It is a **stable reference**, not a live mirror. When endpoints change in the backend, update this document and regenerate `types.ts`.

---

## Idempotency

Three mutation endpoints require an `Idempotency-Key` header to prevent duplicate operations on network retries:

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

### Error responses

| Status | Error code | Meaning |
|--------|------------|---------|
| `400` | `validation_error` | `Idempotency-Key` header missing or exceeds 64 characters |
| `409` | `idempotency_conflict` | Same key is currently being processed (concurrent request in-flight) or was reused with a different request body. Retry after 1 second (`Retry-After: 1` header included). |

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
Request:  { email: string, password: string, display_name?: string, age_confirmed?: bool, date_of_birth?: date }
Response: { access_token, refresh_token, token_type: "bearer", expires_in: int, expires_at: datetime }
Status:   201 Created
Errors:   400 (validation), 409 email_exists, 403 age_verification_required
Note:     age_confirmed required for vex.pics (age_gate=checkbox). date_of_birth for date_of_birth mode.
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
  updated_at: datetime
}
```

#### `PATCH /v1/users/me`

```
Request:  { display_name?: string | null, email?: string, locale?: SupportedLocale }
Response: same as GET /v1/users/me
Errors:   400 email_exists
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
  input_video_url?: string,       // required for v2v (public URL)
  negative_prompt?: string,
  aspect_ratio?: AspectRatio (default "1:1"),
  n?: int (1–10, default 1),      // number of outputs
  name?: string,
  duration?: int (1–15, default 5),          // video only
  resolution?: VideoResolution (default "720p"),  // video only
  height?: int,                   // ComfyUI only
  seed?: int,                     // ComfyUI only
  steps?: int (1–20)              // ComfyUI only
}
Response: JobCreatedResponse
Status:   201 Created
Errors:   400 (model_disabled | validation_error | generation_failed), 402 insufficient_balance, 403 model_not_allowed (model unavailable on this product), 409 idempotency_conflict, 429 rate_limited, 503 service_unavailable
Headers:  Idempotency-Key: <string> (required, max 64 chars)
Note:     source_output_id enables "remix from gallery" — the backend resolves lineage automatically
          (source_job_id + source_output_id) and records it on the new job.
          Idempotency-Key prevents duplicate jobs on network retries — supply a UUIDv4 per submission attempt.
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
  provider: string,       // e.g. "aisha", "grok"
  name: string,           // e.g. "Aisha", "xAI Grok"
  available: bool,        // whether provider backend is reachable
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
  aspect_ratios: string[],           // e.g. ["1:1", "16:9"]
  image: ImageConstraints | null,    // null for video-only models
  video: VideoConstraints | null     // null for image-only models
}

ImageConstraints: {
  min_height: int | null,            // null = not user-controllable
  max_height: int | null,
  default_height: int | null,
  output_resolutions: string[] | null  // informational; null = backend-determined
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

```
{
  id: UUID,
  name: string,
  status: JobStatus,
  provider: "grok" | "aisha",
  model: string | null,
  generation_type: GenerationType,
  prompt: string,
  negative_prompt: string | null,
  aspect_ratio: string | null,
  token_cost: int | null,
  created_at: datetime,
  started_at: datetime | null,
  completed_at: datetime | null,
  outputs: JobOutputItem[],
  thumbnail_url: string | null,
  error: string | null
}

JobOutputItem: {
  id: UUID,
  url: string,              // presigned URL, valid ~1 hour
  content_type: string,     // "image/jpeg", "video/mp4", etc.
  format: string,           // "jpeg", "webp", "mp4"
  size_bytes: int,
  output_index: int,        // 0-based; -1 for thumbnails
  is_thumbnail: bool
}
```

---

## 7. Storage *(authenticated)*

### Uploads

#### `POST /v1/storage/upload`

```
Request:  multipart/form-data, field "data" (max 20MB, PNG/JPEG/WebP)
Response: {
  id: UUID,
  storage_key: string,
  filename: string,
  content_type: string,
  size_bytes: int,
  created_at: datetime,
  expires_at: datetime
}
Status:   201 Created
Errors:   400 (invalid_file_type | file_too_large | empty_file)
Note:     Returns image id used for I2I/I2V generation requests
```

#### `GET /v1/storage/uploads`

```
Query:    limit? (1–100, default 50), cursor? (opaque token)
Response: CursorPage<ImageListItem>

ImageListItem: {
  id: UUID,
  filename: string,
  content_type: string,
  size_bytes: int,
  created_at: datetime,
  expires_at: datetime
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

#### `DELETE /v1/storage/uploads/{image_id}`

```
Response: 204 No Content
```

### Outputs

#### `GET /v1/storage/outputs`

```
Query:    limit? (1–100, default 50), cursor? (opaque token)
Response: CursorPage<OutputListItem>

OutputListItem: {
  id: UUID,
  job_id: UUID,
  content_type: string,
  size_bytes: int,
  output_index: int,
  created_at: datetime,
  expires_at: datetime
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

## 8. Content Proxy *(authenticated)*

Provides stable, non-expiring authenticated URLs for user content. The server resolves ownership, checks product scoping, then streams bytes directly from R2. **No presigned URLs are exposed** — the client only ever sees `/v1/content/...` paths.

> **Why use this instead of presigned URLs?** Content proxy URLs are permanent (for the lifetime of the resource), cacheable with `Cache-Control: private, max-age=<ttl>, immutable`, and enforce per-request authorization. They are the preferred URL format for Gallery and any UI that persists content references.

### Response Headers

All successful responses include:
- `Content-Type` — from R2 object metadata
- `Content-Length` — from R2 object metadata
- `Cache-Control: private, max-age=10800, immutable` — 3-hour client cache (default; configurable via `CONTENT_URL_TTL`)
- `ETag: "<content_id>"` — the output/upload UUID, for conditional requests
- `X-Content-Id: <content_id>` — same UUID, without quotes

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

---

## 9. Gallery *(authenticated)*

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
  job_id: UUID;
  cover_url: string;        // "/v1/content/outputs/{id}" or "/v1/content/uploads/{id}"
  video_url: string | null; // present for video generation types; autoplay source
  badge: GalleryBadge;      // "prompt" (t2i/t2v) or "image" (i2i/i2v/flf2v/v2v)
  media_type: OutputMediaType; // "image" or "video"
  output_count: number;     // non-thumbnail outputs in this group
  generation_type: GenerationType;
  model: string | null;
  aspect_ratio: string | null; // e.g. "16:9"
  prompt_snippet: string;   // first 100 chars of the prompt
  created_at: string;       // ISO datetime
}

interface GalleryGroupDetail {
  job_id: UUID;
  // Header
  badge: GalleryBadge;
  input_image_url: string | null; // present when badge == "image"
  prompt: string;
  negative_prompt: string | null;
  // Outputs
  outputs: GalleryOutputItem[];   // non-thumbnail outputs, ordered by output_index
  // Metadata
  media_type: OutputMediaType;
  model: string | null;
  provider: string;
  generation_type: GenerationType;
  aspect_ratio: string | null;
  token_cost: number | null;
  created_at: string;
  completed_at: string | null;
  // Lineage
  lineage: GalleryLineage | null;
}

interface GalleryOutputItem {
  id: UUID;
  url: string;              // "/v1/content/outputs/{id}"
  thumbnail_url: string | null; // video poster frame URL (if applicable)
  content_type: string;     // "image/jpeg", "video/mp4", etc.
  media_type: OutputMediaType;
  format: string;           // "jpeg", "webp", "mp4"
  size_bytes: number;
  output_index: number;     // 0-based
  created_at: string;
}

interface GalleryLineage {
  source_type: GallerySourceType;    // "upload" or "generation"
  source_upload_id: UUID | null;     // set when source_type == "upload"
  source_job_id: UUID | null;        // set when source_type == "generation"
  source_job_name: string | null;    // human-readable name of the source job
  source_output_id: UUID | null;     // specific output used as input
}
```

### Cover URL Resolution

| Generation type | `cover_url` source | `video_url` |
|----------------|-------------------|-------------|
| `t2i` | Last generated output | `null` |
| `t2v` | Video thumbnail (poster frame) | Video output |
| `i2i` | Source output → input upload → last output | `null` |
| `i2v` | Source output → input upload → last output | Video output |
| `flf2v` | Source output → input upload → last output | Video output |
| `v2v` | Source output → thumbnail → last output | Video output |

### Gallery Badge Logic

| Badge value | Generation types |
|-------------|-----------------|
| `"prompt"` | `t2i`, `t2v` — text-only input |
| `"image"` | `i2i`, `i2v`, `flf2v`, `v2v` — image/video input |

---

## 10. Billing *(authenticated)*

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
  token_cost: int,
  is_active: bool,
  effective_from: datetime,
  effective_until: datetime | null,
  notes: string | null
}
```

#### `GET /v1/billing/packages`

```
Response: TokenPackageResponse[]

TokenPackageResponse: {
  id: string,
  name: string,
  tokens: int,
  bonus_tokens: int,
  total_tokens: int,
  price_usd: string      // decimal string e.g. "9.99"
}
```

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
Request:  { package_id: string }
Response: { checkout_url: string, session_id: string, payment_id: UUID }
Status:   201 Created
Headers:  Idempotency-Key: <string> (required, max 64 chars)
Errors:   409 idempotency_conflict
Note:     Redirect user to checkout_url for Stripe Checkout.
          Supply a fresh UUIDv4 Idempotency-Key per checkout attempt to prevent duplicate payments.
```

#### `POST /v1/billing/topup/nowpayments`

```
Request:  { package_id: string, pay_currency: string }
Response: { invoice_url: string, payment_id: UUID }
Status:   201 Created
Headers:  Idempotency-Key: <string> (required, max 64 chars)
Errors:   409 idempotency_conflict
```

### Billing — Public (no auth)

#### `GET /v1/billing/packages`

Same response as authenticated version.

#### `GET /v1/billing/pricing`

```
Response: {
  packages: TokenPackageResponse[],
  prices: PricingRulePublicResponse[]
}

PricingRulePublicResponse: {
  provider: string,
  generation_type: string,
  model: string | null,
  token_cost: int
}
```

### Billing — Webhooks (no auth)

#### `POST /v1/billing/webhooks/stripe`

```
Request:  Stripe webhook payload (raw body + Stripe-Signature header)
Note:     Internal endpoint for Stripe payment events
```

#### `POST /v1/billing/webhooks/nowpayments`

```
Request:  NowPayments webhook payload (raw body + x-nowpayments-sig header)
Note:     Internal endpoint for NowPayments events
```

---

## 11. Organizations *(authenticated)*

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

## 12. Admin *(authenticated — ADMIN or SUPERADMIN role)*

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
Request:  { provider: string, generation_type: string, model?: string, token_cost: int, notes?: string }
Response: PricingRuleResponse
Status:   201 Created
```

#### `PATCH /v1/admin/pricing/{rule_id}`

```
Request:  { token_cost?: int, is_active?: bool, effective_until?: datetime, notes?: string }
Response: PricingRuleResponse
```

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
  currency: string,          // e.g. "USD"
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

## 13. Admin Management *(authenticated — SUPERADMIN only)*

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
  target_user_id: UUID,
  action: string,   // "role.grant" | "role.revoke" | "permission.grant" | "permission.revoke"
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
Query:    target_user_id? (UUID), limit? (default 50)
Response: AuditLogEntry[]
Note:     Entries are returned newest-first. Optionally filter to a specific target user.
```

---

## 14. Real-Time Events (SSE + Pub/Sub)

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
| `user:{user_id}` | Per-user | `job.status_changed`, `job.progress`, `balance.updated` |
| `system:broadcast` | All connected clients | `system.notification` |

Each SSE connection subscribes to both the per-user channel and `system:broadcast`.

### Publish Points

Events are automatically published by the backend at:

| Event | Published when |
|-------|---------------|
| `job.status_changed` | Generation request submitted (status → `pending`) |
| `job.status_changed` | Grok video job completes, fails, or times out |
| `job.progress` | Grok video job enters `running` state |
| `balance.updated` | `check_and_reserve` (debit), `refund`, `credit`, `admin_adjustment` |
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

## 15. Product Reference

### Products

| Slug | Display Name | Domains | Content Rating | Age Gate | Payment Providers | Org Feature |
|------|-------------|---------|---------------|----------|------------------|-------------|
| `vex` | vex.pics | vex.pics, www.vex.pics, app.vex.pics | permissive | checkbox | Stripe + NowPayments | No |
| `synthara` | Synthara | synthara.app, www.synthara.app, app.synthara.app | sfw | none | Stripe only | Yes |

### AgeGatePolicy

Values: `"none"`, `"checkbox"`, `"date_of_birth"`

### ContentRating

Values: `"sfw"`, `"permissive"`

---

## 16. Enums Reference

### ModelType

| Value | Provider | T2I | I2I | T2V | I2V | V2V | FLF2V | Max Images |
|-------|----------|-----|-----|-----|-----|-----|-------|-----------|
| `grok-imagine-image` | grok | ✓ | ✓ | | | | | 10 |
| `grok-2-image-1212` | grok | ✓ | | | | | | 10 |
| `grok-imagine-video` | grok | | | ✓ | ✓ | ✓ | ✓ | 1 |
| `aisha-image` | aisha | ✓ | ✓ | | | | | 4 |
| `aisha-video` | aisha | | | ✓ | ✓ | ✓ | ✓ | 1 |

> `aisha-video` is seeded as disabled until the video workflow is production-ready.

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

**Polling strategy:** Poll `GET /v1/jobs/{id}` every 2s while status is `pending`, `queued`, or `running`. Stop on any terminal status.

### AspectRatio

Values: `"1:1"`, `"16:9"`, `"9:16"`, `"4:3"`, `"3:4"`, `"2:3"`, `"3:2"`

### VideoResolution

Values: `"480p"`, `"720p"`

### AccountType

Values: `"personal"`, `"enterprise"`

### TransactionType

Values: `"debit"`, `"credit"`, `"refund"`, `"admin_adjustment"`

### SubscriptionTier

Values: `"free"`, `"basic"`, `"pro"`, `"enterprise"`

### UserRole

Values: `"superadmin"`, `"admin"`, `"user"`

> `"system"` is an internal sentinel role — never returned by any API endpoint.
>
> `"superadmin"` has all admin capabilities plus the ability to manage roles and permissions. See [§13 Admin Management](#13-admin-management-authenticated--superadmin-only).

### AdminPermission

Values: `"billing_adjust"`

> Granular permissions that a superadmin can grant to ADMIN-role users. Currently only `billing_adjust` exists (enables `POST /v1/admin/accounts/{id}/adjust` for that admin).

### OrgRole

Values: `"owner"`, `"admin"`, `"member"`

### PaymentStatus

Values: `"pending"`, `"completed"`, `"failed"`, `"refunded"`

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

---

## 17. Error Response Format

All non-2xx responses use a single unified envelope:

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
| 403 | `forbidden`, `account_inactive`, `permission_denied`, `age_verification_required`, `model_not_allowed` | — |
| 404 | `not_found`, `account_not_found`, `price_not_found` | — |
| 409 | `conflict`, `refund_not_eligible`, `organization_balance_nonzero` | `balance` |
| 422 | `validation_error`, `moderation` | `provider`, `policy` |
| 429 | `too_many_requests`, `rate_limited` | `retry_after` |
| 503 | `service_unavailable` | — |

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

## 18. Content URLs

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

## 19. Rate Limits

| Endpoint | Limit |
|----------|-------|
| `POST /auth/register` | 5/hour per IP |
| `POST /auth/login` | 10/minute per IP |
| `POST /auth/forgot-password` | 3/hour per IP |
| `POST /auth/resend-verification` | 3/hour per IP |
| `POST /v1/events/sse-ticket` | 10/minute per user |

Rate limit headers are **not currently exposed** in responses. The frontend should handle 429 responses gracefully with a user-friendly message.

---

## 20. Health Check

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
Response: { status: "ready" | "not_ready", checks: { postgres: string, redis?: string, r2?: string } }
Status:   200 if ready, 503 if not ready
Note:     R2 is checked but excluded from the ready/not_ready determination (slow HeadBucket).
          Use this for CI readiness waits and traffic-routing decisions.
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

## 21. OpenAPI Documentation Endpoints

The backend's `OpenAPIConfig` is configured with `path="/docs"`, so all schema and documentation UI endpoints live under `/docs/`:

| Endpoint | Description |
|----------|-------------|
| `GET /docs/openapi.json` | OpenAPI 3.1 schema (JSON) — **use this for type generation** |
| `GET /docs/openapi.yaml` | OpenAPI 3.1 schema (YAML) |
| `GET /docs/swagger` | Swagger UI |
| `GET /docs/redoc` | ReDoc UI |
| `GET /docs/elements` | Stoplight Elements UI |
| `GET /docs/rapidoc` | RapiDoc UI |

**Export command:**
```bash
curl http://localhost:8000/docs/openapi.json > src/lib/api/schema.json
npx openapi-typescript src/lib/api/schema.json -o src/lib/api/types.ts
```
