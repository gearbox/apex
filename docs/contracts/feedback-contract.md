# Frontend Contract — In-Product Problem Reports (Feedback)

> **Audience:** `gearbox/apex-frontend` (report dialog + admin feedback page).
> **Backend source:** branch `feat/feedback-reports` (migration `049_feedback_reports`).
> **Authority:** `gen:api` (OpenAPI `schema.json`) is authoritative for **types**. This document is authoritative for **semantics**: validation, error codes, what to send, and the admin triage lifecycle.

---

## 0. Summary

- Signed-in users can send a problem report from anywhere in the product. This is the vex Terms §11.1 "in-product reporting function".
- One endpoint: `POST /v1/feedback`. It returns `201 {id, status, created_at}`. The user gets no confirmation email and has no "my reports" view. Show a local "Thanks, we got it" state.
- The report text stays inside apex. Operators get a Telegram ping that contains **only IDs and the category**, and they read the text in the admin API (§5).
- **No attachments.** To point at a broken result, send `job_id` and/or `asset_ref`. The backend stores the reference, not a copy of the file.
- Anonymous reports are not supported. A user who can't sign in should contact the support email.

---

## 1. Types

```ts
type FeedbackCategory = "bug" | "generation" | "billing" | "account" | "content" | "other";

type FeedbackStatus = "open" | "in_progress" | "resolved" | "dismissed";

// POST /v1/feedback — request
interface FeedbackCreate {
  category: FeedbackCategory;
  message: string;          // 10–4000 code points after trimming; see §2
  job_id?: string | null;   // UUID of the caller's own generation job
  asset_ref?: string | null; // "upload:<uuid>" | "output:<uuid>" — the library's asset_ref
  client_path?: string | null; // location.pathname ONLY — see §2
  app_version?: string | null; // 1–64 chars, your build id
}

// POST /v1/feedback — 201 response
interface FeedbackCreated {
  id: string;               // UUID
  status: "open";
  created_at: string;       // ISO 8601
}
```

Unknown request fields are rejected with `400`. Omit a field or send `null` when you have nothing for it.

Suggested category labels:

| Value | Label |
|---|---|
| `bug` | Something isn't working |
| `generation` | Problem with a generation / result |
| `billing` | Payments, tokens, top-ups |
| `account` | Account, sign-in, settings |
| `content` | Content concern |
| `other` | Something else |

---

## 2. Validation rules

| Field | Rule | On failure |
|---|---|---|
| `category` | One of the six values above. | `400` |
| `message` | 10–4000 **Unicode code points after trimming**. The backend trims leading/trailing whitespace **before** it checks both bounds, so `"   hi   "` is rejected and a 4000-character message with a trailing newline or surrounding spaces is accepted. Count with `[...text.trim()].length`, not `text.length` (an emoji is 2 UTF-16 units but 1 code point; a `textarea` `maxLength` counts UTF-16 units). Must not contain NUL (`\u0000`). Every message-length and NUL violation returns `400 validation_error`. | `400 validation_error` |
| `job_id` | A UUID of a job the caller owns that is not deleted. | `404 job_not_found` |
| `asset_ref` | `upload:<uuid>` or `output:<uuid>`, owned by the caller. Pass the `asset_ref` from the library item as-is. | malformed → `400 validation_error`; missing/not owned → `404 asset_not_found` |
| `client_path` | Must start with `/`. At most 512 characters. No `?`, no `#`, no control characters. | `400` |
| `app_version` | 1–64 characters, no control characters. | `400` |

**`client_path` must be `location.pathname` only.** Never send `location.href`, the query string or the hash. Those can hold tokens (the OAuth callback fragment, reset links), and the server rejects `?` and `#` to enforce this.

**Ownership 404s don't leak.** Another user's job, a deleted job and an ID that doesn't exist all return the same `404` body. A report about a job the user has deleted returns `404 job_not_found`. Resend without `job_id`, or tell the user to describe the job in the message.

The server reads the `User-Agent` header itself (truncated to 512 characters). Don't put it in the body. The server does **not** store the client IP.

---

## 3. Errors

Every error uses the standard envelope: `{ error, message, status_code, detail? }`.

| Status | `error` | When | UX |
|---|---|---|---|
| 400 | `validation_error` | Message too short or too long after trimming, NUL in message, malformed `asset_ref` | Show inline on the message field |
| 400 | *(framework validation)* | Schema violation: bad category, bad `client_path`/`app_version`, unknown field | Treat as a client bug; log it |
| 401 | — | Not signed in / token expired | Normal refresh-then-retry |
| 404 | `job_not_found` / `asset_not_found` | See §2 | Offer to send without the attachment reference |
| 413 | `error` *(generic)* | Request body over 64 KiB. Cannot happen with a valid report | Treat as a client bug; log it |
| 429 | `rate_limited` | More than **10 reports per hour from one IP** (`RATE_LIMIT_FEEDBACK`). `detail.retry_after` is in seconds. | "You've sent several reports recently — please try again later." |

The `413` carries the generic `error: "error"` value because the global error-code table has no entry for 413 yet. Match on `status_code`, not on `error`.

**No `428` on this endpoint.** `POST /v1/feedback` is exempt from legal re-acceptance. A user blocked by a pending terms update can still report a problem. Don't route them to the re-acceptance screen first.

---

## 4. Rate limit

- The limit is `10/hour` per client IP by default. The server counts every attempt, including rejected ones (401/400), because the check runs before authentication.
- `/v1/feedback` and `/v1/feedback/` share one budget.
- Successful responses carry `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset`. A `429` also carries `Retry-After`.

---

## 5. Admin API

All three endpoints require an **ADMIN or SUPERADMIN** token. Each is scoped to the product of the request (`Origin`/`Host`/`X-Product-Id`), so a synthara report never shows up in the vex admin and the reverse. A report from another product returns the same `404` as a missing one. A non-admin gets `401` ("Admin access required"), the same as every other `/v1/admin/*` route.

```ts
interface FeedbackReportAdmin {
  id: string;
  category: FeedbackCategory;
  status: FeedbackStatus;
  message: string;            // raw user text — render as TEXT, never as HTML/markdown
  user_id: string | null;     // null once the user account is hard-deleted
  user_email: string | null;  // joined at read time; null once the user is gone
  job_id: string | null;      // null if the job was hard-deleted
  asset_ref: string | null;   // may point at an asset that retention has since deleted
  asset_url: string | null;   // "/v1/content/feedback/{id}" iff asset_ref is set — relative, prefix with API_BASE
  client_path: string | null;
  app_version: string | null;
  user_agent: string | null;
  admin_note: string | null;
  resolved_at: string | null; // set once, when the report enters resolved/dismissed
  resolved_by: string | null; // admin user id
  created_at: string;
  updated_at: string;
}
```

`message`, `admin_note`, `client_path` and `user_agent` are untrusted user input. Render them as plain text.

`asset_ref` has no foreign key, because content retention deletes outputs. `asset_url` is set whenever `asset_ref` is. Open it in a new tab (`API_BASE + asset_url`); it authenticates with the content cookie like library media and supports Range, so videos play. `404 asset_not_found` means the asset was deleted or expired by retention, or the reporter's account is gone. Opening an asset is audit-logged. Never build the owner URL `/v1/content/outputs/{id}` for an admin; it will always 404.

### `GET /v1/content/feedback/{report_id}`

Streams the asset a report points at. It requires an **ADMIN or SUPERADMIN** credential (Bearer access token or the `apex_content` cookie) and is scoped to the request's product. It lives under `/v1/content` so `<img>`, `<video>` and a new-tab navigation authenticate with the cookie (`Path=/v1/content`) and no `Authorization` header. Behaviour matches the other content routes (`Range`, `If-None-Match`, `Content-Disposition`), with these differences:

- `Cache-Control: private, no-store` (the owner routes use `immutable`), so an admin device does not keep other users' media in its HTTP cache.
- The asset is resolved as the **reporter**, not the admin. A report pointing at an asset its reporter does not own returns `404 asset_not_found`.
- One `admin_audit_log` row (`action: feedback.asset.view`, IDs only) is written when a view starts: a request without `Range`, or a `Range` that starts at byte 0. Later range chunks of the same playback are not logged again.

| Status | `error` | Meaning |
|---|---|---|
| 200 / 206 | | Full body / requested range |
| 304 | | `If-None-Match` matched |
| 401 | | Missing or invalid credential, or not an admin |
| 404 | `feedback_not_found` | No such report in this product |
| 404 | `asset_not_found` | Report has no `asset_ref`, the reporter was purged, or the asset was deleted or retention-expired |
| 416 | `range_not_satisfiable` | `Range` starts at or beyond the object size |
| 502 | `upstream_error` | R2 fetch failed |

### `GET /v1/admin/feedback`

Query: `status?`, `category?`, `limit` (1–100, default 30), `cursor?`. Returns `CursorPage<FeedbackReportAdmin>`, newest first. Pagination uses a cursor (`next_cursor` / `has_more`) and stays stable while new reports arrive. A malformed cursor returns `400 invalid_cursor`.

### `GET /v1/admin/feedback/{report_id}`

Returns `FeedbackReportAdmin`, or `404 feedback_not_found`.

### `PATCH /v1/admin/feedback/{report_id}`

```ts
interface FeedbackAdminPatch {
  status?: FeedbackStatus;
  admin_note?: string | null; // ≤ 4000 chars, no NUL; null clears the note
}
```

Omitted fields stay unchanged. An empty body returns `400 validation_error`. The response is the updated `FeedbackReportAdmin`.

Status lifecycle:

```
open ──► in_progress ──► resolved
  │            └───────► dismissed
  ├──────────────────►   resolved
  └──────────────────►   dismissed
```

- `resolved` and `dismissed` are **terminal**, and a report can't be reopened. `resolved_at` and `resolved_by` are written once, on entry to a terminal status.
- Any other transition returns `409 invalid_status_transition`, with `detail: {current, target}`. That includes setting the status the report already has. After a 409, reload the report. Another admin may have just resolved it.
- A note-only PATCH works in every status, including terminal ones.
- Unlike `POST /v1/feedback`, this endpoint **is** subject to legal enforcement. An admin with a stale `lgl` claim gets `428`.

---

## 6. Operator notification

When a report is saved, admins subscribed to the product-scoped `feedback.submitted` class get a Telegram message:

```
[vex] 📨 New feedback · billing
report <uuid>
user <uuid>
job <uuid>          (only when job_id was sent)
```

The message text, path, user agent and version are never sent to Telegram. Delivery is best-effort. The admin list is the source of truth, so a missed ping never loses a report.
