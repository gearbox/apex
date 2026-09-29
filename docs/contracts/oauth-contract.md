# Frontend Contract — OAuth Sign-In (Google)

> **Audience:** `gearbox/apex-frontend`.
> **Backend source:** branch `feat/oauth-google` (migration `048_user_identities`).
> **Authority:** `gen:api` (OpenAPI `schema.json`) is authoritative for **types**. This document is authoritative for **semantics**: the redirect sequence, the fragment grammar, cookie handling, and error UX.

---

## 0. Summary

- "Continue with Google" is a **full-page navigation** to the API, not a `fetch`. The API runs the whole OAuth exchange with Google itself. The SPA never sees a Google token.
- The API finishes on your route `/auth/callback` and puts the result in the **URL fragment** (`#…`). Depending on the result, the SPA then does one of three things:
  - **login:** redeems a one-time `code` at `POST /v1/auth/oauth/exchange` and receives the usual `TokenResponse`.
  - **signup:** shows a signup screen with the legal checkboxes, then calls `POST /v1/auth/oauth/complete-signup` and receives the `TokenResponse`.
  - **error:** shows the message for the `error` code (§4).
- A browser-binding cookie (`apex_oauth_tx`, HttpOnly, set by the API) ties every step to the browser that started the flow. Every POST below must send **`credentials: 'include'`**.
- No account exists until the user accepts the legal documents on the signup screen.

Show the Google button only when `GET /v1/auth/product-info` lists `"google_oauth"` in `allowed_auth_methods`. The backend only lists it when Google is configured for that product.

---

## 1. Sequence

```
SPA                              API (api.<brand>)                        Google
───                              ─────────────────                        ──────
window.location = /v1/auth/oauth/google/authorize?return_to=/library
                                 302 → accounts.google.com  (+ Set-Cookie apex_oauth_tx)
                                                                          user picks account
                                 GET /v1/auth/oauth/google/callback  ◄────┘
                                 302 → https://<brand>/auth/callback#result=…
/auth/callback:
  1. read location.hash, then history.replaceState(null, "", location.pathname)
  2. result=login  → POST /v1/auth/oauth/exchange {code}          → 200 TokenResponse
     result=signup → POST /v1/auth/oauth/signup-info {ticket}     → {email, provider}
                     (show legal checkboxes)
                     POST /v1/auth/oauth/complete-signup {…}       → 201 TokenResponse
     result=error  → show message for `error`
  3. navigate to return_to (or home)
```

---

## 2. Types

```ts
type OAuthProvider = "google";

type OAuthResult = "login" | "signup" | "error";

type OAuthErrorCode =
  | "oauth_cancelled"
  | "oauth_failed"
  | "flow_expired"
  | "email_unverified"
  | "account_inactive"
  | "identity_conflict"
  | "invalid_handoff"
  | "invalid_signup_ticket";

interface OAuthExchangeRequest { code: string }
interface OAuthSignupInfoRequest { ticket: string }
interface OAuthSignupInfoResponse { email: string; provider: OAuthProvider }
interface OAuthCompleteSignupRequest {
  ticket: string;
  accepted_documents: AcceptedDocument[];   // same as POST /v1/auth/register
  display_name?: string | null;             // 1–100 chars
}
// TokenResponse is unchanged — identical to /v1/auth/login.
```

`AcceptedDocument` is defined in `docs/contracts/legal-documents-contract.md` §1.

---

## 3. Endpoints

All endpoints are product-scoped through the usual `Origin` / `Host` / `X-Product-Id` resolution. The navigations (authorize, callback) resolve the product from the API host.

### `GET /v1/auth/oauth/{provider}/authorize?return_to=<path>` (navigation)

- Set `window.location.href` to this URL. Don't use `fetch` or a popup: the response is a 302 to Google that sets the binding cookie.
- `return_to` is optional. It must be a same-origin **path** that starts with a single `/`, has no whitespace and no `#`, and is at most 512 chars. Anything else returns `400 invalid_return_to`, including `//host`, `/\host`, and `https://…`.
- Returns `404` when the provider isn't enabled for this product. That shouldn't happen if the button follows `product-info`.
- Starting a sign-in **replaces** any earlier binding cookie, so a flow already open in another tab ends with `flow_expired`. Only one sign-in can be in progress per browser, and that is intended.
- The `apex_oauth_tx` binding cookie is set to the pending flow's lifetime here. On a successful callback it is re-minted to the resulting login handoff or signup ticket lifetime; it is not changed on callback errors.

### `GET /v1/auth/oauth/{provider}/callback` (Google → API)

This is the API's redirect target, and the SPA never calls it. It always responds `302` to `{app_url}/auth/callback#<fragment>` and never returns JSON.

**Fragment grammar** (`application/x-www-form-urlencoded`; parse with `new URLSearchParams(location.hash.slice(1))`):

| `result` | Other keys | Meaning |
|---|---|---|
| `login` | `code`, optional `return_to` | An existing account signed in (or was just linked or claimed, §4). Redeem `code` within **60 s**. |
| `signup` | `ticket`, optional `return_to` | New to this product. Show the signup screen. The ticket is valid for **15 min**. |
| `error` | `error` (an `OAuthErrorCode`) | Show §4 copy. |

`return_to` is URL-encoded. `URLSearchParams` decodes it. Navigate to it only after a successful exchange or signup, and treat it as a path on your own origin.

### `POST /v1/auth/oauth/exchange` — `credentials: 'include'`

- Body: `{ code }`.
- **200:** `TokenResponse` plus `Set-Cookie: apex_content` (same as login) and a Set-Cookie that clears `apex_oauth_tx`. Store the tokens exactly as after password login. Right after a claim (§4) this call can take up to about a second longer than usual, because the API waits out the second in which it ended the account's old sessions; the new tokens are valid as soon as it returns.
- **400 `invalid_handoff`:** the code is unknown, already used, expired, or was opened in another browser. Codes are single-use, so never retry the same one. Restart sign-in.
- **401 `account_inactive`:** the account was deactivated.

### `POST /v1/auth/oauth/signup-info` — `credentials: 'include'`

- Body: `{ ticket }`. Returns **200** `{ email, provider }` for the signup screen. For example, show "Create your account with person@example.com".
- It doesn't consume the ticket, so it is safe to call again (reload, back button).
- **400 `invalid_signup_ticket`:** the ticket expired, was already used, or belongs to another browser. Restart sign-in.

### `POST /v1/auth/oauth/complete-signup` — `credentials: 'include'`

- Body: `{ ticket, accepted_documents, display_name? }`.
- **201:** `TokenResponse` plus the `apex_content` cookie, with `apex_oauth_tx` cleared. The account's email is already verified. It has no password (see §5).
- **Legal errors come before the ticket is consumed.** `409 legal_version_stale` / `422 legal_acceptance_incomplete` behave exactly as for register (legal contract §3). Refetch `/v1/legal/current`, show the new text, and submit again with the **same ticket**.
- **400 `invalid_signup_ticket`:** expired or already used. A double-click is safe: exactly one request succeeds and the other gets this error. Restart sign-in.
- **400 `email_exists`:** someone registered this email on this product since the callback. The ticket is spent; send the user back to sign in. Signing in with Google again links that account (§4).
- **409 `identity_conflict`:** this Google account was linked to another account since the callback. Restart sign-in.

**Signup screen requirements:** render the legal documents exactly as the password signup form does. That means `GET /v1/legal/current`, each body fetched with `?version=`, and `sensitive_data_consent` as its own unticked checkbox. Submit `accepted_documents` in the same shape as `POST /v1/auth/register`. `display_name` is optional. The backend never imports the Google name or picture.

---

## 4. Error codes and UX copy

Use product branding in all copy. Never show the backend's internal name.

| `error` | Where | Suggested copy |
|---|---|---|
| `oauth_cancelled` | fragment | "Google sign-in was cancelled." (Optionally show no message at all.) |
| `oauth_failed` | fragment | "We couldn't complete sign-in with Google. Please try again." |
| `flow_expired` | fragment | "Your sign-in session expired or was started in another tab. Please try again." |
| `email_unverified` | fragment | "Your Google account's email address isn't verified. Verify it with Google, or sign up with email and password." |
| `account_inactive` | fragment, exchange (401) | "This account has been deactivated." |
| `identity_conflict` | fragment, complete-signup (409) | "This account is already linked to a different Google account." |
| `invalid_handoff` | exchange (400) | "This sign-in link has expired. Please sign in again." |
| `invalid_signup_ticket` | signup-info / complete-signup (400) | "Your sign-up session expired. Please start again." |

**Linking rule (backend semantics):** Google only reaches this step with `email_verified: true`, so it has proven the user owns the inbox. When that email matches an existing account on this product:

- **The account's email is verified:** sign-in links the Google identity to it. The password is kept and no session is touched. Both login methods work afterwards.
- **The account's email was never verified:** sign-in **claims** the account in one transaction. This defends against pre-account hijacking, where someone registers `victim@…` with their own password without owning the inbox. The claim marks the email verified, **clears the password** (`has_password` becomes `false`), ends **every** session of the account (refresh tokens, live access and content tokens, Web Push subscriptions), links the Google identity, and signs the user in. Whoever held the old password or tokens loses them. The rightful owner can add a password later with "Set a password" (§5).
- **The account is already linked to a different Google account:** `identity_conflict`, and nothing about the account changes (this is checked before any claim).

There is no "account exists but is unverified" error: the claim replaces it. If the user has other devices signed in to an unverified account, those devices are signed out by the claim, so the next request there returns `401` and the app should send the user to sign in.

---

## 5. Frontend requirements

1. **`credentials: 'include'`** on `exchange`, `signup-info`, and `complete-signup`. Without it the HttpOnly binding cookie isn't sent and every call fails with `invalid_handoff` / `invalid_signup_ticket`. CORS already allows credentials for the product origins.
2. **Strip the fragment first.** At the top of `/auth/callback`, read `location.hash`, then call `history.replaceState(null, "", location.pathname + location.search)` **before** making any request or rendering third-party content. This keeps `code` / `ticket` out of history, bookmarks, and analytics.
3. **Never send the fragment values anywhere except the matching endpoint.** Don't log them or put them in analytics or error reports.
4. **Show the legal checkboxes on the signup screen,** fetched from the legal endpoints as described in §3.
5. **Hide change-password when `GET /v1/users/me` returns `has_password: false`.** `POST /v1/users/me/password` returns `409 password_not_set` for these accounts. Offer "Set a password" through `POST /v1/auth/forgot-password` instead. Password login for such an account fails with the ordinary `401 invalid_credentials`. This also applies to an account just claimed through Google (§4): `has_password` is `false` afterwards. `GET /v1/users/me` also returns `email_verified` (a claimed account is verified).
6. **One flow per browser.** Starting a login in a second tab invalidates the first (§3). Treat `flow_expired` as "try again", not as a bug.
7. **Tokens are the usual ones.** After `exchange` / `complete-signup`, behave exactly as after `POST /v1/auth/login`: store the tokens, schedule the content-cookie re-mint, and refresh as usual. Tokens carry the `lgl` digest, so a freshly signed-up user doesn't hit `428` right away.

---

## 6. Account lifecycle notes

- Closing the account (`DELETE /v1/users/me`) unlinks Google, so the same Google account can later sign up again as a new account.
- An administrator deactivation keeps the link. Signing in with Google then returns `account_inactive` rather than starting a new signup.
- Accounts are product-scoped. The same Google account signs up separately on each product.

---

## 7. Out of scope (next arcs)

- **Sign in with Apple:** Synthara allows it, but it isn't wired yet. Its callback is a cross-site `form_post`, so the binding cookie will need `SameSite=None`.
- **Link / unlink** Google from account settings ("connected accounts").
