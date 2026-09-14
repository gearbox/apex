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
| `ref` | Must match `PROVISIONING_REF_PATTERN` (`^(v\d+\.\d+\.\d+\|[0-9a-f]{40})$`), or equal `Settings.provisioning_script_dev_ref` outside production. |
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
| 200 | `{"ok": true}` | Applied, or a no-op (session not found / already terminal — D9, so the provisioner's own retry loop never gets a 404 to spam) |
| 400 | error envelope | Malformed body |
| 401 | `unauthorized` | Missing/invalid token |

On a valid call against a live session: the session transitions to `failed`
with `error_message` set to the upstream `error` string (reason constant
`_REASON_NODE_PROVISION_SCRIPT_FAILED`, `src/api/services/provisioning_webhook.py`),
its Vast.ai instance is destroyed, its Cloudflare tunnel is torn down, its base
reservation is refunded in full, and the usual `GPU_SESSION_STATUS_CHANGED` SSE
event fires — the same shape a client already handles for any other
provisioning failure. A `container_id` mismatch against the session's
`vastai_instance_id` is logged as a warning but never blocks the fail: the
token is the authority, not the container id.

This is terminal-once: it races against apex's own probe fail-fast (below) on
the same `gpu_sessions` row, serialized by a `SELECT ... FOR UPDATE`, so
whichever side reaches it first is the only one that tears down
infrastructure and refunds.

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
| 503 | `provisioning_unavailable` | `ai_bundles_github_token` or `provisioning_script_ref` unset, or the script ref fails to resolve (404/502 from GitHub) — no tunnel, no Vast.ai instance, no session row, no billing hold |

## gearbox/aisha coordination

- `docs/PROVISIONING.md` in `gearbox/aisha`: the template no longer sets
  `PROVISIONING_SCRIPT`; the release "tag pinning" step now means bumping
  `Settings.provisioning_script_ref` in apex.
- The aisha bootstrap script should echo `ACS_PROVISION_SCRIPT_SHA256` back in
  its `acs.provision.ready` telemetry so apex can detect drift between what it
  served and what actually ran. Tracked as a separate aisha-side task — this
  repo only emits the var.
