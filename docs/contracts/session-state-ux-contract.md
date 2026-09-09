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
| `model.runtime` | Authenticated user's current `ModelRuntimeResponse` for an on-demand model. It is `null` for always-on providers and for an unauthenticated request. |
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

`runtime: null` is not `state: "none"`: `null` means that the runtime
projection was not evaluated (always-on providers do not have a per-user
runtime, and unauthenticated callers do not receive one), while `state:
"none"` means that the authenticated on-demand runtime was evaluated and no
deployment is live. When `state === "none"`, every ID is null. `operation_id`
is a lightweight "work is in progress" handle; use `GET
/v1/sessions/{session_id}` to read the full `current_operation` projection.

Card state comes from the server-provided `model.runtime`. Live progress comes
from the operation cache, looked up by `runtime.operation_id`. Nothing is
derived by combining the providers `runtime.state` with a session-detail
deployment status; the frontend must not compute a third state from those two
REST projections.

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

The API permits `null` for either value. Treat it as “no hint configured” and
show elapsed-only once an operation exists.

## Read model and operation association

`GET /v1/sessions/{id}` returns a primary deployment plus any sibling
deployments attached additively. Each `DeploymentResponse.current_operation`
is the current or latest durable `OperationResponse` for that deployment.
Every embedded `OperationResponse` — `session.bootstrap_operation`, a
deployment's `current_operation`, a `202` mutation body, or an
`operation_updated` frame — is a view onto one canonical operation cache keyed
by operation id and merged by `revision`.

`OperationResponse.deployment_id` is an optional informational direct target.
It may identify the primary deployment for `session_bootstrap` and the target
for deployment-scoped operations. It may be `null` for operations governing
multiple deployments or the whole session, notably cohort restarts. It must
never be used to route an operation update to frontend deployment state.

Upsert every `operation_updated` frame into the operation cache keyed by
operation id whenever its `revision` is strictly greater than the cached
revision, **before** resolving any deployment association. Never discard a
newer operation solely because no cached deployment currently references it.
Deployment cards and `session.bootstrap_operation` then render from that cache
by id. Never route an operation by `deployment_id`; this includes cohort
restarts, which resolve through each deployment's restart pointer rather than
the operation's direct target.

When applying a session snapshot, replace deployment scalars wholesale — REST
is authoritative for them — but merge each embedded operation into the
operation cache under the same strictly-greater-`revision` rule. A snapshot
never lowers a cached operation's revision.

A re-provision creates a new `session.bootstrap_operation` with a new id. The
client learns of it through the `status_changed` invalidation and the refetch
that follows, not through a frame for the old id.

`/v1/providers` carries `runtime.operation_id` but not the operation itself. If
that id is absent from the operation cache, the coalesced session-detail
refetch hydrates it with the embedded full operation. Do not issue a separate
`GET /v1/sessions/{id}/operations/{operation_id}` for this; that endpoint is
the SSE-down fallback, not an operation-cache hydration path.

## Operation lifecycle and progress guarantees

A freshly created operation is a valid first state, not a loading failure:
`status` is `queued`, `revision` is `0`, and `phase`, `progress`, and
`started_at` are all `null`.

The `progress`, `work`, `items`, and `rate` objects are each fully populated or
`null`, never partial. `progress_pct` is `null` or within `[0, 100]`; a node may
truthfully report `work.completed > work.total` when some file sizes were
unknown, and the API retains that work measurement while rendering the derived
percentage as `100.0`.

`eta_seconds` is `null` unless it is derived from live throughput. The client
must never synthesize it from `work`, elapsed time, or `typical_*_seconds`.

## SSE synchronization

SSE is lossy: after every connect or reconnect, re-fetch the session detail.
Frames can arrive out of order; operation frames are retained in the canonical
operation cache by the strictly-greater-`revision` rule above.

Any durable change to `OperationResponse`, whatever its source (node telemetry,
command timeout, cancellation, or lifecycle cascade), produces a newer
`operation_updated` frame. This is what permits zero polling while SSE is
healthy.

### Event roles

| Event | Role |
|---|---|
| `gpu_session.operation_updated` | Authoritative incremental state; apply directly by `revision`. |
| `gpu_session.deployment_status_changed` | REST invalidation; refetch session detail. |
| `gpu_session.status_changed` | REST invalidation; refetch session detail. |

An event-triggered `GET` is not polling. The zero-polling guarantee is
unchanged.

After `gpu_session.status_changed` or
`gpu_session.deployment_status_changed`, invalidate both the affected
session-detail query and the authenticated `/v1/providers` runtime projection.
Coalesce provider invalidations alongside session invalidations under the same
per-session debounce: at most one session refetch may be in flight, use a
trailing-edge debounce of roughly 250 ms, and discard the response of any
refetch superseded by a later one.

`/v1/providers` and session detail are independent, unversioned snapshots.
Neither carries a revision, and they can land in either order, so transient
disagreement between them after an invalidation is expected rather than an
error. Rendering must not be gated on the two agreeing. The operation cache is
the only revision-tracked store; full operations embedded in a session
snapshot merge into it under the strictly-greater-`revision` rule, while the
providers snapshot supplies only the `runtime.operation_id` lookup.

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
contain several deployments. Treat this event as an invalidation signal.
Re-fetch `GET /v1/sessions/{session_id}` and render the affected model cards
from the server-provided `model.runtime` in authenticated `/v1/providers`.
Do not independently reconstruct `RuntimeState` from session or deployment
statuses. The payload may drive transient optimistic UI, but must not overwrite
a newer REST snapshot.

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

Treat this event as an invalidation signal. Re-fetch
`GET /v1/sessions/{session_id}` to reconcile the deployment and hydrate its
current operation from REST, while rendering card state from the
server-provided `model.runtime` in authenticated `/v1/providers`. Do not
independently reconstruct `RuntimeState` from session or deployment statuses.
The payload may drive transient optimistic UI, but must not overwrite a newer
REST snapshot. Multiple events for the same session may be coalesced into one
refresh. It contains no operation phase or progress and never exposes raw
Aisha telemetry. `operation_id` is only the join key for the typed operation
stream below.

### `gpu_session.operation_updated`

The payload is exactly `OperationResponse`, the same safe projection returned
by REST. Its `phase` and `progress` are the only live operation telemetry the
frontend may interpret. Upsert it into the operation cache before resolving
cached `current_operation.id` values as described above, including every
member of a cohort restart.

## Async deployment mutations

`POST` and `DELETE` deployment mutations return `202` with
`{deployment, operation}`. `DELETE` addresses a deployment UUID, is idempotent
while the deployment is `removing`, and requires `?force=true` to remove the
last live deployment.

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
- A `deployment_status_changed` frame arriving after a newer REST snapshot does
  not regress the card.
- An `operation_updated` frame whose id matches no cached deployment is
  retained and appears once the following refetch associates it.
- A burst of frames for one session produces a single refetch.
- An `operation_updated` frame replaces `session.bootstrap_operation` when the
  ids match, under the same revision rule.
- A deployment-status frame never causes the client to parse raw progress;
  phase/progress come only from `gpu_session.operation_updated`.
- A parent session-status frame triggers a REST refresh because it has no
  model type and may affect sibling deployments.
- An additive attach moves the card `none → provisioning → suspended → active`
  by following server-provided `model.runtime`, without client-side state
  derivation.
- A removal moves the card `active → removing → none` by following
  server-provided `model.runtime`, without client-side state derivation.
- Pause and resume move the card `active → paused → active` by following
  server-provided `model.runtime`, without client-side state derivation.
- A stop or failure affecting several models on one GPU updates every affected
  card from its server-provided `model.runtime`, without client-side state
  derivation.
- A failed attach renders as `none` in the provider catalog; session detail is
  used to surface the persisted operation error when that context is needed.
- A command-sweep timeout updates the operation card through SSE without a
  refetch.
- A `completed > total` telemetry payload renders at 100% rather than blanking
  progress.
