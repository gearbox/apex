# Frontend contract — model runtime, deployments, and operations

This document defines how `apex-frontend` renders on-demand Aisha models from
`GET /v1/providers`, session reads, and SSE. Generated OpenAPI types are the
source of truth for field types; this document defines lifecycle semantics and
cache-update rules.

## Inputs for an on-demand model card

| Field | Meaning |
|---|---|
| `provider.available` | The provider is configured and serviceable for every user. It is independent of a user's runtime. |
| `provider.provisioning_mode` | `"always_on"` or `"on_demand"`. Only the latter has a per-user runtime. |
| `model.runtime` | Authenticated user's current `ModelRuntimeResponse` for an on-demand model. It is `null` for an anonymous request. |
| `model.provisioning` | Configured display hints for the initial bootstrap and an additive attach. They are not estimates. |

Never collapse `available` and `runtime` into a single boolean. An unavailable
provider has no user action that can make it available; an available provider
whose runtime is `none` can be started by the user.

`runtime` has this shape:

```ts
interface ModelRuntimeResponse {
  state: "none" | "provisioning" | "active" | "suspended" | "removing"
       | "paused" | "stale" | "stopping";
  session_id: string | null;
  deployment_id: string | null;
  operation_id: string | null; // non-terminal work only
}
```

For an authenticated, available on-demand model, `runtime` is always present.
When `state === "none"`, every ID is null. `operation_id` is a lightweight
"work is in progress" handle; use `GET /v1/sessions/{session_id}` to read the
full `current_operation` projection.

## Card states and actions

Evaluate this order top-down:

| Condition | Card state | Primary action |
|---|---|---|
| `!model.is_enabled` | Disabled | None |
| `!provider.available` | Unavailable | None |
| `provisioning_mode === "always_on"` | Ready | Generate |
| On-demand and anonymous | Sign in required | Sign in |
| `runtime.state === "none"` | Needs session | Start session |
| `runtime.state === "provisioning"` | Provisioning | Show operation; allow cancel through session stop where applicable |
| `runtime.state === "active"` | Ready | Generate |
| `runtime.state === "suspended"` | Restarting | Show operation; generation disabled |
| `runtime.state === "removing"` | Removing | Generation disabled |
| `runtime.state === "paused"` | Paused | Resume or stop |
| `runtime.state === "stale"` | Unreachable | Stop; surface any safe error text from the session read |
| `runtime.state === "stopping"` | Stopping | Wait for terminal transition |

`Generate` is enabled only for an available model whose state is `active` (or
an available always-on model). Do not offer Start while a state other than
`none` occupies the model's live deployment slot.

## Provisioning hints

`model.provisioning` is present for Aisha models. These values are deliberately
coarse display hints and must never be combined with elapsed time to form an
ETA, replace operation telemetry, or drive a timeout.

| Model | Typical bootstrap | Typical additive attach |
|---|---:|---:|
| `aisha-image` | 600 s | 180 s |
| `aisha-image-lite` | 480 s | 120 s |
| `aisha-video` | 900 s | 300 s |

The API permits `null` for either value. Treat it as “no hint configured” and
show elapsed-only once an operation exists.

## Read model and operation association

`GET /v1/sessions/{id}` returns a primary deployment plus any sibling
deployments attached additively. Each `DeploymentResponse.current_operation`
is the current or latest durable `OperationResponse` for that deployment.

`OperationResponse.deployment_id` is informational only. A cohort
`comfyui_restart` legitimately governs multiple deployments, so its
`deployment_id` is `null`. When an operation update arrives, patch **every**
cached deployment whose `current_operation.id` equals the operation's `id`;
never route it by `deployment_id`.

## SSE synchronization

SSE is lossy: after every connect or reconnect, re-fetch the session detail
and treat REST as the complete source of truth. Frames can arrive out of order;
for operation frames, retain only a sequence greater than the cached sequence
for that operation.

### `gpu_session.status_changed`

```ts
interface GpuSessionStatusPayload {
  session_id: string;
  status: GpuSessionStatus;
  previous_status: GpuSessionStatus | "none";
  tunnel_hostname: string | null;
  error_message: string | null;
  reason: string | null;
}
```

This parent-session event intentionally has no `model_type`: a session may
contain several deployments. Re-fetch `GET /v1/providers` or the affected
session after it, then re-derive all cards for that session from REST.

### `gpu_session.deployment_status_changed`

```ts
interface GpuDeploymentStatusPayload {
  deployment_id: string;
  session_id: string;
  model_type: ModelType;
  status: DeploymentStatus;
  pending_restart: boolean;
  routing_suspended: boolean;
  operation_id: string | null;
  error_message: string | null;
}
```

Use this frame to fast-forward the affected deployment/card. It contains no
operation phase or progress and never exposes raw Aisha telemetry. `operation_id`
is only the join key for the typed operation stream below.

### `gpu_session.operation_updated`

The payload is exactly `OperationResponse`, the same safe projection returned
by REST. Its `phase` and `progress` are the only live operation telemetry the
frontend may interpret. Apply it by matching its `id` against cached
`current_operation.id` values as described above, including every member of a
cohort restart.

## Failed additive attaches

`RuntimeState` intentionally has no `failed` member. A failed deployment is
not live and is excluded from the `/v1/providers` runtime overlay, so the
affected model reports `runtime.state === "none"` after a failed attach. That
is indistinguishable from “never provisioned” on the catalog alone.

The failure is retained in `GET /v1/sessions/{id}` on the failed deployment and
its failed operation's `error.message`. If the UI needs to show a previous
attach failure, fetch the session detail; do not infer failure from a `none`
runtime or invent a client-side failed runtime state.

## Acceptance checks

- A cohort restart updates all matching deployment cards even though the
  operation's `deployment_id` is null.
- A deployment-status frame never causes the client to parse raw progress;
  phase/progress come only from `gpu_session.operation_updated`.
- A parent session-status frame triggers a REST refresh because it has no
  model type and may affect sibling deployments.
- A failed attach renders as `none` in the provider catalog; session detail is
  used to surface the persisted operation error when that context is needed.
