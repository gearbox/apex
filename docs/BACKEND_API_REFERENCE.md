# Backend API Reference — Apex REST API

> **Source:** `gearbox/apex` repository
> **Framework:** Litestar 2.5+ / Python 3.13
> **Schema:** `GET /docs/openapi.json` from running backend (Litestar OpenAPIConfig has `path="/docs"`)
> **Last synced:** 2026-03-17

This document captures the API surface that the frontend depends on. It is a **stable reference**, not a live mirror. When endpoints change in the backend, update this document and regenerate `types.ts`.

---

## Pagination

All list endpoints return a **unified `PaginatedResponse<T>`** shape:

```typescript
interface PaginatedResponse<T> {
  items: T[];
  total: number;      // total matching records (regardless of cursor/offset)
  limit: number;      // echoed page size
  offset: number;     // echoed offset (0 when cursor-based paging is active)
  has_more: boolean;  // true when there are additional pages
  next_cursor: string | null;  // opaque cursor token for the next page; null if none
}
```

### Cursor-based pagination (keyset)

Priority list endpoints (`/v1/jobs`, `/v1/storage/outputs`,
`/v1/billing/transactions`, `/v1/storage/uploads`) support an optional `cursor`
query parameter alongside the existing `limit`/`offset` parameters.

- Pass `cursor=<next_cursor>` from the previous response to fetch the next page.
- When `cursor` is supplied, `offset` is **ignored**.
- The cursor is an opaque, URL-safe base64 token; do not parse or construct it manually.
- `next_cursor` is `null` when `has_more` is `false`.

```
// Page 1
GET /v1/jobs?limit=20
Response: { items: [...], total: 142, limit: 20, offset: 0, has_more: true, next_cursor: "eyJ..." }

// Page 2 (cursor-based — stable even if new jobs were added)
GET /v1/jobs?limit=20&cursor=eyJ...
Response: { items: [...], total: 142, limit: 20, offset: 0, has_more: true, next_cursor: "eyJ..." }
```

Traditional offset paging still works unchanged for all endpoints:

```
GET /v1/jobs?limit=20&offset=40
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
  input_image_id?: UUID,          // required for i2i / i2v / flf2v
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
Errors:   400 (model_disabled | validation_error | generation_failed), 402 insufficient_balance, 403 model_not_allowed (model unavailable on this product), 429 rate_limited, 503 service_unavailable
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
Query:    status?, provider?, generation_type?, limit? (default 20), offset? (default 0),
          cursor? (opaque token for keyset pagination)
Response: PaginatedResponse<UnifiedJobResponse>
  // has_more, next_cursor for cursor-based paging
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
Query:    limit? (1–100, default 50), offset? (default 0), cursor? (opaque token)
Response: PaginatedResponse<ImageListItem>
  // has_more, next_cursor for cursor-based paging

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
Query:    limit? (1–100, default 50), offset? (default 0), cursor? (opaque token)
Response: PaginatedResponse<OutputListItem>
  // has_more, next_cursor for cursor-based paging

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
Response: PaginatedResponse<OutputListItem>  // has_more=false, no cursor (returns all outputs)
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

## 8. Billing *(authenticated)*

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
Query:    limit? (default 50), offset?, type? ("debit" | "credit" | "refund" | "admin_adjustment"),
          cursor? (opaque token for keyset pagination)
Response: PaginatedResponse<TransactionResponse>
  // has_more, next_cursor for cursor-based paging

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
Note:     Redirect user to checkout_url for Stripe Checkout
```

#### `POST /v1/billing/topup/nowpayments`

```
Request:  { package_id: string, pay_currency: string }
Response: { invoice_url: string, payment_id: UUID }
Status:   201 Created
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

## 9. Organizations *(authenticated)*

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

## 10. Admin *(authenticated — admin only)*

### User Management

#### `GET /v1/admin/users`

```
Query:    is_active? (bool), role? (string), email? (partial match, case-insensitive),
          limit? (default 50), offset? (default 0)
Response: PaginatedResponse<AdminUserResponse>
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
  role?: UserRole,
  subscription_tier?: SubscriptionTier,
  is_active?: bool,
  locale?: SupportedLocale
}
          // All fields optional — only provided fields are updated
Response: AdminUserResponse
Errors:   400 (role=system), 403 (patching own account), 404 (user not found)
Note:     Admins cannot modify their own account via this endpoint.
```

### Organization Management

#### `GET /v1/admin/organizations`

```
Query:    is_active? (bool), limit? (default 50), offset? (default 0)
Response: PaginatedResponse<AdminOrgResponse>

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
Query:    limit?, offset?, type?
Response: PaginatedResponse<TransactionResponse>
```

#### `POST /v1/admin/accounts/{account_id}/adjust`

```
Request:  { amount: int, description: string }
          // positive amount = credit, negative = debit
Response: { transaction: TransactionResponse, new_balance: int }
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
Query:    status?, payment_provider?, limit? (default 50), offset?
Response: PaginatedResponse<PaymentResponse>

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

## 11. Real-Time Events (SSE + Pub/Sub)

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

## 12. Product Reference

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

## 13. Enums Reference

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

Values: `"admin"`, `"user"`

> `"system"` is an internal sentinel role — never returned by any API endpoint.

### OrgRole

Values: `"owner"`, `"admin"`, `"member"`

### PaymentStatus

Values: `"pending"`, `"completed"`, `"failed"`, `"refunded"`

### SupportedLocale

Values: `"en"` (English), `"ru"` (Russian), `"sr"` (Serbian Latin)

---

## 14. Error Response Format

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

## 15. Presigned URL Notes

- All R2 presigned URLs are valid for **~1 hour** by default
- Do **not** aggressively cache them — use `staleTime` of ~30 minutes in TanStack Query
- URLs are generated on-demand when fetching jobs/outputs/uploads
- R2 storage key pattern:
  - Uploads: `users/{user_id}/uploads/{file_id}.{ext}`
  - Outputs: `users/{user_id}/outputs/{job_id}/{file_id}.{ext}`

---

## 16. Rate Limits

| Endpoint | Limit |
|----------|-------|
| `POST /auth/register` | 5/hour per IP |
| `POST /auth/login` | 10/minute per IP |
| `POST /auth/forgot-password` | 3/hour per IP |
| `POST /auth/resend-verification` | 3/hour per IP |
| `POST /v1/events/sse-ticket` | 10/minute per user |

Rate limit headers are **not currently exposed** in responses. The frontend should handle 429 responses gracefully with a user-friendly message.

---

## 17. Health Check

#### `GET /health/`

```
Response: { status: "healthy" | "unhealthy", comfyui_connected: bool, version: string }
Note:     Public endpoint, no auth needed
```

---

## 18. OpenAPI Documentation Endpoints

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
