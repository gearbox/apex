# Provisioning contract — bootstrap script delivery + failure handling

This document defines the two public endpoints apex exposes to the Vast.ai
provisioner running on a GPU node, the `PROVISIONING_SCRIPT` / `PROVISIONER_*`
env-var contract that points a node at them, and the status codes both sides
must handle. It is a node/ops contract, not a frontend one — see
`session-state-ux-contract.md` for the frontend-facing runtime states these
changes feed into (`provisioning` -> `active` or a terminal `failed`).

Background: a reachable ComfyUI is not evidence of successful provisioning
(2026-09-13 staging incident — a node ran with the stock checkpoint after its
bootstrap script 404'd anonymously against a since-privatized repo). Every rule
below exists to close that gap.

## GET /v1/provisioning/scripts/{variant}/{ref}

```
GET /v1/provisioning/scripts/{variant}/{ref}?session={session_id}&token={callback_token}
```

No Litestar guard — the caller is the Vast.ai provisioner, which sends no
configurable headers, so auth rides entirely in the query string.

| Part | Contract |
|---|---|
| `variant` | One of `ScriptVariant` (`src/core/enums.py`): today only `comfyui`. Maps to a hard-coded `(repo, path)` pair in `SCRIPT_VARIANT_SOURCES` (`src/core/constants.py`) — **never** taken from the request body or query. |
| `ref` | Must match `PROVISIONING_REF_PATTERN` (`^(v\d+\.\d+\.\d+|[0-9a-f]{40})$`), or equal `Settings.provisioning_script_dev_ref` outside production. The configured `Settings.provisioning_script_ref` is checked against this same rule at boot and again during start-time preflight. `provisioning_script_dev_ref` must additionally be route-safe (T3, round-3 remediation): no `/`, `?`, `#`, `%`, or whitespace, and not `.`/`..` — it is interpolated raw as this single `{ref:str}` path segment, so e.g. a branch named `feature/bootstrap` would build a URL the router 404s on. Checked at Settings load (`validate_dev_ref_is_route_safe`) and again in `build_provisioning_callback_urls` itself, so an unsafe ref can never reach a node's boot env regardless of who validated it upstream. Use a slash-free branch name for dev testing. |
| `session` / `token` | `token`, SHA-256'd, must match the session's `callback_token_hash` (D4). Both required. |

Responses:

| Status | Code | Meaning |
|---|---|---|
| 200 | — | `text/plain; charset=utf-8` body, strong `ETag` = the script's sha256 |
| 304 | — | `If-None-Match` matched the current sha256; empty body |
| 400 | `bad_request` | Unknown variant, or `ref` fails validation |
| 401 | `unauthorized` | Missing/invalid `session`/`token`, or token doesn't match the session |
| 404 | `provisioning_script_ref_not_found` | Upstream GitHub 404 (bad ref/path) |
| 429 | `rate_limited` | Per-IP rate limit exceeded |
| 502 | `provisioning_script_unavailable` | Upstream GitHub 5xx or rate-limited |

Successful fetches are cached in Redis (long TTL for an immutable tag/SHA,
short TTL for the dev ref); failed fetches are **never** cached (D5).

## POST /v1/provisioning/webhook/{session_id}

```
POST /v1/provisioning/webhook/{session_id}?token={callback_token}
Content-Type: application/json

{
  "action": "continue",
  "manifest": "/path/to/manifest.yaml",
  "error": "provisioning failed after all retries",
  "container_id": "50885024",
  "timestamp": "2026-09-13T12:36:41"
}
```

Sent by the Vast.ai provisioner after its own `max_retries` are exhausted for
the onstart script (typically ~90-120s after the first failure, at the
defaults `max_retries=3`, `retry_delay=30`). `container_id` is a **string**;
`timestamp` is naive node-local time — apex stores it as received and never
uses it for ordering. Unknown fields are ignored (msgspec's default — the
upstream schema may grow).

No Litestar guard — same reasoning as the script endpoint. Auth is `token` in
the query string, validated the same way.

| Status | Body | Meaning |
|---|---|---|
| 200 | `{"ok": true}` | Applied, or a no-op for an already-terminal/stopping session after its token validates |
| 400 | error envelope | Malformed body |
| 401 | `unauthorized` | Session not found, or missing/invalid token. These cases intentionally have the same status and response body. |

On a valid call against a live session: the session transitions to `failed`
with `error_message` set only to the fixed reason constant
`_REASON_NODE_PROVISION_SCRIPT_FAILED` (`src/api/services/provisioning_webhook.py`),
its Vast.ai instance is destroyed, its Cloudflare tunnel is torn down, its base
reservation is refunded in full, and the usual `GPU_SESSION_STATUS_CHANGED` SSE
event fires — the same shape a client already handles for any other
provisioning failure. A `container_id` mismatch against the session's
`vastai_instance_id` is logged as a warning but never blocks the fail: the
token is the authority, not the container id.

This is terminal-once: it races against apex's own probe fail-fast (below) on
the same `gpu_sessions` row, serialized by a `SELECT ... FOR UPDATE`, so
whichever side reaches it first is the only one that tears down
infrastructure and refunds. The provisioner's free-text `error` and `manifest`
are observability-only: they are redacted (URL query strings, callback params,
and URL userinfo removed), bounded, and emitted only on the dedicated
`provisioning.webhook.failure_detail` log event. They never reach the session
row, SSE payload, or billing-refund metadata.

### Two-stage token check (S2, round-2 remediation)

`ProvisioningWebhookService.handle_failure` validates the presented `token`
**twice**, against two different reads of the row, because they authorize two
different things:

1. **Pre-lock check (authorization for the response).** The webhook loads the
   session row with a plain `get_by_id` (no lock) and validates the token
   against that detached read *before* it will reveal whether the session
   exists or its status — this is what produces the uniform `401` for an
   unknown session or a bad token, before any row lock is taken.
2. **Locked check (authorization for the action).** The presented token is
   then passed through to `GpuSessionService.fail_pre_active_session(...,
   expected_callback_token=token)`, which re-validates it against the row
   *after* acquiring `SELECT ... FOR UPDATE`. This closes a window the
   pre-lock check alone cannot: a delayed webhook call from an old,
   already-destroyed node can pass check 1 against the hash as it existed at
   read time, then a concurrent provisioning retry
   (`GpuProvisioningWorker._retry_with_new_node`) can commit a *replacement*
   instance with a fresh `callback_token_hash` before the webhook reaches its
   lock. Without check 2, the stale caller would tear down and refund the
   replacement node — a session mid-recovery, not the one that actually
   failed.

**When check 2 rejects (a rotated token), the webhook still returns `200`, not
`401`.** The token was valid when presented — the provisioner should not
retry — but the row it was validated against is no longer the row this call
is authorized to act on, so `fail_pre_active_session` returns `None` and
`handle_failure` never learns the difference between "applied" and "no-op on
a rotated token"; both paths reach the same unconditional `return HTTP_200_OK`.
Do not try to collapse these two checks into one — they run at different
times against different (detached vs. locked) data and guard different
things; see `fail_pre_active_session`'s docstring in
`src/api/services/gpu_session/service.py`.

`GpuProvisioningWorker`'s own probe-fail-fast path calls
`fail_pre_active_session` with `expected_callback_token=None`, which skips
check 2 entirely — its authorization is holding the worker's leader lease,
not a callback token.

## Trust boundary for node-supplied free text (S1 round-2, T1/T8 round-3 remediation)

**No node-supplied string is persisted, published, or billed against without
passing through `redact_secrets`/`redact_secrets_mapping`/`redact_known_secrets`
(`src/api/utils/redaction.py`) first.** This rule is enforced once, at each
receiver's trust boundary, not re-derived per call site — see each redaction
function's module docstring.

D3 put the callback token into the node's own environment
(`PROVISIONING_SCRIPT`/`PROVISIONER_WEBHOOK_URL` query strings, plus
`CF_TUNNEL_TOKEN`/`ACS_GITHUB_TOKEN`/`ACS_HF_TOKEN`/`ACS_CIVITAI_API_TOKEN` as
bare env vars), so any node-side code path that echoes a failing command, a
curl error, or an env dump into free-text telemetry can otherwise leak a
secret straight into a database row or the client's event stream. Two
receivers apply this today:

- **`POST /v1/provisioning/webhook/{session_id}`** (above): `error` and
  `manifest` are redacted before the single `provisioning.webhook.failure_detail`
  log line — this was closed in round 1.
- **Telemetry v2 operation events** (`OperationEventService.handle_event`,
  `src/api/services/gpu_session/operation_event_service.py` — the node's
  *every-tick* channel, not just its terminal failure callback): `message`,
  `error`, and the open-ended `progress`/`plan`/`summary` JSON bodies (a URL
  can hide at any depth in these) are all redacted before they reach
  `GpuSessionOperationRepository.apply_event` or
  `GpuSessionCommandRepository.mark_terminal` — persisted to
  `gpu_session_operations`/`gpu_session_commands.error` and published over
  SSE. `event.event_id` and `event.target.bundle_version` are also routed
  through `redact_known_secrets` (cheap — they're structurally simple, not
  free text, but an exact known-secret match must still never survive). This
  was the gap closed in round 2: round 1 fixed the webhook call site only,
  but D3's env-var change exposed the token on every path a node can reach,
  and telemetry events are the node's most-used path.

### What redaction guarantees, and what it does not (T1/T8, round-3 remediation)

The redactor is three layers, applied in order (see the module docstring for
the full design):

1. **Exact known-value replacement** — a plain `str.replace` against every
   plaintext secret apex holds at call time: `Settings.github_content_token`,
   `hf_token`, `civitai_api_token`. **This is the only layer that is a
   guarantee** — no false negatives for those exact values, regardless of how
   they're quoted, prefixed, or embedded.
2. **Key-based structural redaction** — in `redact_secrets_mapping`'s dict
   walker, a sensitive key's value is replaced wholesale before descending
   into it.
3. **A tokenizer over free text** — whitespace-delimited shape matching for
   `KEY=value`, `Authorization:`-style headers, and URLs. Layers 2 and 3 are
   **best-effort, not a guarantee** — round 3 found real leak forms the
   previous regex-based version of this layer missed (see the round-3
   remediation prompt's T1 finding for the empirical table).

**The per-session callback token is the one secret Layer 1 can never cover.**
Apex stores only `callback_token_hash` — the plaintext never exists anywhere
after minting except in the URLs handed to the node — so there is nothing for
Layer 1 to `str.replace` against at redaction time. It is protected *only* by
Layers 2/3's best-effort shape matching, which is also the layer round 3 had
to rebuild after finding it unreliable. It is also the token with the widest
exposure: D3 put it into two URLs (`PROVISIONING_SCRIPT`,
`PROVISIONER_WEBHOOK_URL`) that live in the node's own environment and that
failure text naturally quotes. The tunnel token is in the same position
(never persisted at all, so also outside Layer 1's reach) but has no
comparable exposure path today.

Two options were considered as a follow-up (not implemented in round 3):
splitting the callback token into a one-shot script-fetch token (consumed on
first serve, worthless after boot) plus a separate long-lived webhook token,
which would shrink the window instead of trying to redact it better; or
storing a short non-secret fingerprint alongside the hash so Layer 1 could
match it, which is cheaper but weakens the "only the hash is stored"
property. Do not treat redaction as a complete defence for this value in any
future change to this contract.

`redact_secrets` recognizes any `KEY=value`/`KEY:` where `KEY`
case-insensitively contains `token`, `secret`, `key`, `password`, `passwd`,
`credential`, `auth`, `cookie`, or `session` — not just the literal
`token=`/`session=` forms — so a name nobody anticipated yet is still covered
by Layers 2/3's best-effort matching.

## Env-var contract (`build_acs_env`, `src/api/services/gpu_session/_env_builder.py`)

Set on every `create_instance` call (initial start and provisioning retries):

| Env var | Prefix family | Meaning |
|---|---|---|
| `PROVISIONING_SCRIPT` | Vast's own onstart-fetch var name (unprefixed) | The full script URL above, with this session's `session`/`token` query params |
| `PROVISIONER_WEBHOOK_URL` | Provisioner setting override (unprefixed) | The full webhook URL above, with this session's `token` |
| `PROVISIONER_FAILURE_ACTION` | Provisioner setting override (unprefixed) | Always `"destroy"` — set in *addition* to the webhook, not instead of it, so a failure is never silent |
| `ACS_PROVISION_SCRIPT_SHA256` | `ACS_*` (apex/aisha) | sha256 of the script apex resolved for this session; the aisha script should echo it back in `acs.provision.ready` so apex can detect drift (tracked as a separate aisha-side task) |

`PROVISIONING_SCRIPT`/`PROVISIONER_*` never get the `ACS_` prefix — that
namespace is reserved for the apex/aisha-owned contract; these three are Vast
provisioner concepts. The Vast template itself no longer sets
`PROVISIONING_SCRIPT` (D2) — apex owns the ref pin via
`Settings.provisioning_script_ref` and rejects `POST /v1/sessions` with `503
provisioning_unavailable` before creating any resource if that ref is
unconfigured or fails to resolve (D6).

**Token scope (S5, round-2 remediation):** `ACS_GITHUB_TOKEN` above, and the
token `ProvisioningScriptService._fetch_from_github` presents to GitHub to
resolve the bootstrap script, are the same value: `Settings.github_content_token`
(env `AI_BUNDLES_GITHUB_TOKEN` — kept via `validation_alias` so no deployment's
env needs to change). It requires read access to **both**
`gearbox/ai-bundles` (bundle index cloning) and `gearbox/aisha` (bootstrap
script fetch). The field/env name predates the aisha-repo caller and, read on
its own, describes only the ai-bundles scope — an operator provisioning a
fresh environment from that name alone would reasonably issue an
ai-bundles-only-scoped PAT and get a `503 provisioning_unavailable` (a `404`
from GitHub, logged but not surfaced to the caller) on every session start.
Issue the PAT with both repos' read access.

`Settings.apex_callback_url` is required whenever bootstrap script delivery is
configured. It must be a non-empty absolute `http://` or `https://` origin,
with no path, query, fragment, or userinfo; trailing slashes normalize away at
settings load. Both URLs above are derived through one builder from that
normalized origin, so no path suffix or double slash can drift into one endpoint
but not the other.

**Retry/re-provision note:** each provisioning retry (`GpuProvisioningWorker._retry_or_fail`)
mints a fresh callback token for the new instance and rebuilds this env from
scratch, so the new `PROVISIONING_SCRIPT`/`PROVISIONER_WEBHOOK_URL` carry the
new token and the old instance's URLs stop validating (401) immediately.

**SECURITY:** every value above except `ACS_PROVISION_SCRIPT_SHA256` carries
the per-session callback token in its query string. Never log a full env dict,
a full `PROVISIONING_SCRIPT`/`PROVISIONER_WEBHOOK_URL` value, or any query
string containing `token=`/`session=` — logs should carry `session_id`,
`sha256_prefix`, and similar redacted fields only.

## `POST /v1/sessions` failure surface (Change 2)

`start_session` now validates config and resolves the bootstrap script
*before* creating the Cloudflare tunnel or any other external resource:

| Status | Code | When |
|---|---|---|
| 503 | `provisioning_unavailable` | `github_content_token`, `provisioning_script_ref`, or `apex_callback_url` unset, or the script ref fails to resolve (404/502 from GitHub) — no tunnel, no Vast.ai instance, no session row, no billing hold |

## gearbox/aisha coordination

- `docs/PROVISIONING.md` in `gearbox/aisha`: the template no longer sets
  `PROVISIONING_SCRIPT`; the release "tag pinning" step now means bumping
  `Settings.provisioning_script_ref` in apex.
- The aisha bootstrap script should echo `ACS_PROVISION_SCRIPT_SHA256` back in
  its `acs.provision.ready` telemetry so apex can detect drift between what it
  served and what actually ran. Tracked as a separate aisha-side task — this
  repo only emits the var.

## Staging rollout verification

The following deployment facts are intentionally not inferred from this code:

1. Start one cheap instance and inspect `/var/log/portal/provisioning.log` to
   prove that the instance-level `PROVISIONING_SCRIPT` points at the Apex script
   endpoint. Confirm an accompanying `provisioning.script.served` log event; an
   alert on its absence detects a template-level override winning unexpectedly.
2. Use that instance to force a script-fetch failure and verify that its webhook
   POST completes before `PROVISIONER_FAILURE_ACTION=destroy` terminates the
   node. Confirm the instance has the Vast credentials needed for self-destroy.
3. (S1, round-2) Force a node failure whose telemetry quotes the script URL
   (e.g. a bad `ACS_BUNDLE` so the aisha script's own error message embeds the
   `PROVISIONING_SCRIPT` value it tried to fetch), then read
   `gpu_session_operations.error` back from the database **by hand** — psql,
   not through any apex code path — and confirm the callback token is absent.
   Automated tests cover the redaction function and the write path; this step
   is the one check that the two are actually wired together in a real
   deployment.
4. (T1, round-3) Post telemetry containing each leak form from the round-3
   remediation's T1 finding table (quoted env assignment, `declare -x`,
   `Authorization: Bearer`, `Authorization: token`, a JSON-as-text `"token":`
   field) as the `message`/`error`/`summary` of one or more operation events,
   then read `gpu_session_operations` back **by hand** and confirm none of
   them survives. This is the round-3 equivalent of step 3 — the redactor was
   rebuilt in round 3, so the wiring-vs-function split above still holds, but
   the function itself needs re-proving against the new leak-form table.

These are required rollout checks because template environment precedence and
the provisioner's shutdown ordering are external Vast.ai behaviours, not
properties this repository can prove.
