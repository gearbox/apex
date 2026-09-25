# Frontend Contract — Legal Documents & Acceptance

> **Audience:** `gearbox/apex-frontend`.
> **Backend source:** branch `feat/legal-documents` (migration `047_legal_acceptances`).
> **Authority:** `gen:api` (OpenAPI `schema.json`) is authoritative for **types**; this document is authoritative for **semantics**: the signup payload, the 428 flow, and the refresh-after-accept rule.

---

## 0. Summary

- The backend serves versioned **Terms of Use**, **Privacy Policy** and **Sensitive-data consent** per product. The markdown lives in the backend repo (`legal/{product}/{doc_type}/{YYYY-MM-DD}.md`), not in the frontend bundle.
- Published versions are immutable. The backend pins each normalized document's SHA-256 in `legal/manifest.toml` at startup; changing published text requires a new version file.
- **Signup must submit what the user accepted:** `accepted_documents` on `POST /v1/auth/register` is **required**.
- After the backend publishes a version that requires re-acceptance, **every mutating call** (`POST`/`PUT`/`PATCH`/`DELETE`) from a user who hasn't re-accepted returns **`428 legal_acceptance_required`**. `GET` requests are never blocked, so users can always read and download their own data.
- To clear a 428, call `POST /v1/legal/acceptances`, then **`POST /v1/auth/refresh`**. The old access token stays blocked until it is refreshed.

| Product | Required documents |
|---|---|
| `vex` | `terms`, `privacy`, `sensitive_data_consent` |
| `synthara` | none yet. Send `accepted_documents: []`; nothing is enforced. |

Always derive the list from `GET /v1/legal/current` instead of hardcoding it.

---

## 1. Types

```ts
type LegalDocumentType = "terms" | "privacy" | "sensitive_data_consent";

// Version = effective date (UTC), ISO "YYYY-MM-DD". Versions compare as dates.
type LegalVersion = string;

interface AcceptedDocument {          // what the client sends back
  doc_type: LegalDocumentType;
  version: LegalVersion;
  sha256: string;                     // 64 lowercase hex chars, copied verbatim from the API
}

interface LegalDocumentMeta {
  doc_type: LegalDocumentType;
  version: LegalVersion;
  sha256: string;
  requires_reacceptance: boolean;
}

interface LegalDocumentResponse extends LegalDocumentMeta {
  content_md: string;                 // markdown source, LF line endings
}

interface LegalStatusResponse {
  documents: {
    doc_type: LegalDocumentType;
    required_version: LegalVersion;   // minimum version that satisfies the requirement
    current_version: LegalVersion;    // what the user would accept now (≥ required_version)
    accepted_version: LegalVersion | null;
    accepted_at: string | null;       // ISO datetime
    satisfied: boolean;
  }[];
  all_satisfied: boolean;
}
```

**Never compute `sha256` on the client.** Echo back exactly the value the API returned for the version you rendered. It is the hash of the text the user saw, and the backend stores it as proof.

---

## 2. Endpoints

All endpoints are product-scoped by the usual `Origin` / `Host` / `X-Product-Id` resolution.

### `GET /v1/legal/current` (public)

Returns the current version of each **required** document: `{ documents: LegalDocumentMeta[] }`. For synthara, `documents` is `[]`.

### `GET /v1/legal/documents/{doc_type}` (public)

- Without a query parameter it returns the **current** version. With `?version=YYYY-MM-DD` it returns that exact version. A future-dated version can be fetched this way to show an advance notice, but it is never "current" before its date.
- Every response includes `ETag: "<sha256>"` and `Vary: Origin, X-Product-Id`. A request for an exact `?version=` has `Cache-Control: private, max-age=31536000, immutable`; the current alias has `Cache-Control: private, no-cache`. Either form honours a matching `If-None-Match` (including `W/`-prefixed validators) with `304` and the same cache headers, ETag, and Vary.
- Returns `404 legal_document_not_found` for an unknown type, a type the product doesn't require, or an unknown version.
- **Render `content_md` as markdown with sanitisation** (no raw HTML). The backend doesn't send the effective date inside the text, so render `version` yourself, for example "Effective 1 October 2026".
- **Acceptance UIs must fetch document bodies by exact version.** First fetch `/v1/legal/current`, then fetch each body with `?version=` set to that response's `version`, and submit that same response's `sha256`. Never use the current alias for a signup or re-acceptance screen; it is only for public page links.

### `GET /v1/legal/status` (Bearer)

Returns `LegalStatusResponse` for the caller. Use it to decide which documents the re-acceptance screen must show (`satisfied === false`).

### `POST /v1/legal/acceptances` (Bearer, works with a stale token)

- Body: `{ accepted_documents: AcceptedDocument[] }`. It must contain **exactly the full required set** at the **current** versions, including documents the user already accepted.
- Response: `200 LegalStatusResponse` with the new state.
- Idempotent: already-accepted current versions are skipped server-side. Writes to one user's acceptance ledger are serialized, so concurrent submissions cannot create duplicate evidence rows.
- **This endpoint does not issue a new token.** Call `POST /v1/auth/refresh` next (see §4).

### `POST /v1/auth/register` (changed)

```jsonc
{
  "email": "…",
  "password": "…",
  "display_name": "…",               // optional
  "accepted_documents": [            // REQUIRED. [] for synthara.
    { "doc_type": "terms", "version": "2026-10-01", "sha256": "…" },
    { "doc_type": "privacy", "version": "2026-10-01", "sha256": "…" },
    { "doc_type": "sensitive_data_consent", "version": "2026-10-01", "sha256": "…" }
  ]
}
```

- The request body now rejects unknown fields.
- The legal check runs **before** the email-exists check, so a stale form never creates an account.
- The backend records the client IP and user agent with each acceptance. The frontend doesn't need to send anything extra for this.

**UI requirement:** `sensitive_data_consent` is a separate consent, not part of the Terms of Use. Show it as its own **unticked** checkbox with its own text, fetched by its exact version from `/v1/legal/current`. Don't bundle it into the "I agree to the Terms" checkbox. How Terms and Privacy are presented is a product/legal decision. On the backend, `privacy` records acknowledgement, not consent. Send the entries only when the user has agreed to every required document.

### `DELETE /v1/users/me` (unchanged request)

Closing the account is how a user withdraws sensitive-data consent. The backend records the withdrawal automatically. The call works even when the user's token is legally stale.

---

## 3. Error codes

All errors use the standard `ErrorEnvelope`: `{ error, message, status_code, detail }`.

| Status | `error` | When | `detail` | Client action |
|---|---|---|---|---|
| 428 | `legal_acceptance_required` | Any non-GET call with a token whose legal digest isn't current | `null` | Start the re-acceptance flow (§4) |
| 409 | `legal_version_stale` | Register or acceptance submitted an old `version` or a wrong `sha256` | `{ current: [{doc_type, version, sha256}] }` | Refetch the documents, show them again, and resubmit |
| 422 | `legal_acceptance_incomplete` | Missing, extra or duplicated `doc_type` | `{ missing: [], unexpected: [], duplicated: [] }` | Bug in the form: submit exactly the `/current` set |
| 404 | `legal_document_not_found` | Unknown type or version, or a type the product doesn't require | `null` | — |

A `409 legal_version_stale` means a new version went live while the user had the form open. Refetch, show the new text, and ask again. Don't silently resubmit with the new hashes.

---

## 4. The 428 flow (re-acceptance)

Access tokens carry an `lgl` claim, a digest of the legal versions the user had accepted when the token was minted. On every mutating request, `auth_guard` compares it to the currently required digest. The check needs no database round-trip. As a result:

1. When a new version with `requires_reacceptance = true` takes effect (at **00:00 UTC** on its effective date), **every** existing access token becomes stale for mutations at that moment. That includes users who accept later in the day on another device, until they refresh.
2. On `428 legal_acceptance_required`:
   1. `GET /v1/legal/status` shows which types are unsatisfied.
   2. Fetch `/v1/legal/current`, then for each listed item fetch `GET /v1/legal/documents/{doc_type}?version=<version>` to render exactly the text whose `sha256` you will echo back.
   3. `POST /v1/legal/acceptances` with **all** required types at their current versions (take them from `/v1/legal/current`).
   4. **`POST /v1/auth/refresh`**. The new access token carries the updated `lgl`.
   5. Retry the original request, if appropriate.
3. **Refresh after accepting.** Skipping step 4 keeps returning 428, because the old token still carries the old digest.
4. Refreshing without accepting yields a token that still 428s. The backend computes `lgl` from the acceptance ledger, so the client can't obtain a passing token any other way.

Endpoints that work with a stale token (they never return 428):

- `POST /v1/legal/acceptances`
- `DELETE /v1/users/me`
- `POST /v1/users/me/password` (password rotation must remain available to secure an account)
- `POST /v1/users/me/logout-all`
- `POST /v1/auth/resend-verification`
- `POST /v1/auth/content-cookie`
- `POST /v1/events/sse-ticket`
- All `GET` requests, and all unauthenticated endpoints (`/v1/auth/login`, `/refresh`, `/logout`, …)

A version published with `requires_reacceptance = false` (for example a typo fix) becomes current without invalidating anything. `/v1/legal/current` returns the new text and hash, and new signups accept it.

---

## 5. Production document authoring

Production startup rejects unresolved drafting placeholders. Markdown links are safe to use in any of these forms when their reference definition exists in the same document: inline (`[text](url)`), full reference (`[text][label]`), collapsed reference (`[text][]`), and shortcut reference (`[label]`). Definitions (`[label]: url`) and GFM task-list markers (`- [x]`) are also allowed. Undefined bracket spans are treated as placeholders and fail production startup.

The manifest's `sha256` is the SHA-256 of the UTF-8 markdown after line endings are normalized to LF. Do not edit a published version in place: add a new dated file and its manifest entry. An operator may correct a manifest hash only for a version that has never been accepted anywhere.

---

## 6. Advance notice

Backend operators can commit a future-dated version up to 14 days ahead, as the Terms of Use require. It stays dormant until its date. There is currently **no endpoint that lists upcoming versions**. Emailing users about them is a planned backend follow-up. If the UI needs a banner, fetch the specific upcoming version by `?version=`.

---

## 7. Out of scope (next arcs)

- **OAuth signup:** a post-callback acceptance screen that reuses the same payload and endpoints.
- **Synthara documents:** `required_legal_documents` is empty until they exist.
- **Email notification** of upcoming versions.
