# Frontend contract — `session_state` → model-card UX state & actions

Companion to the backend change on `model-session-handling`. Ships in the **same PR**. Defines how `apex-frontend` (SvelteKit PWA) renders and acts on the three new/changed `GET /v1/providers` fields, and how it stays in sync via SSE.

---

## The three backend inputs

Each model card is a pure function of three catalog fields plus the client's auth state:

| Field | Source | Meaning |
|---|---|---|
| `provider.available` | `ProviderInfo.available` | Provider is **configured and serviceable in this deployment**. For `always_on` → cloud API wired (incl. storage). For `on_demand` → GPU stack wired (bundle index + workflow service). **Process-level**, identical for all users. |
| `provider.provisioning_mode` | `ProviderInfo.provisioning_mode` | `"always_on"` (cloud API, usable whenever available) or `"on_demand"` (requires a per-user GPU session before generating). |
| `model.session_state` | `ModelInfo.session_state` | Per-user readiness of an on-demand model: `null` \| `"none"` \| `"provisioning"` \| `"active"` \| `"paused"` \| `"stale"`. **Non-null only for `on_demand` models on an authenticated request.** |

### The orthogonality rule (folded from review finding #3)

**`available` and `session_state` are independent axes. Never collapse them into one "is it usable" boolean, and never treat `available=false` as "needs a session."**

- `available=false` → the provider can't serve *anybody* here (deployment-level). There is **no user action** that fixes it — do **not** offer "Start session"; starting one would fail.
- `available=true` + `session_state="none"` → provider is fine; the user simply hasn't spun up a session yet. This **is** actionable.

The previous backend always reported Aisha `available=true`; it now reports `false` wherever the GPU stack isn't wired (and Grok's condition is slightly stricter — it also requires storage). So a card can now legitimately be `provisioning_mode="on_demand"`, `available=false` — that's "temporarily unavailable," not "press Start."

---

## Canonical state derivation

Evaluate top-down; first match wins. `isAuthenticated` is the client's own auth flag.

| # | Condition | Card state |
|---|---|---|
| 1 | `!model.is_enabled` | `DISABLED` *(not emitted by `/v1/providers` today — it only lists enabled models — but handle defensively)* |
| 2 | `!provider.available` | `UNAVAILABLE` |
| 3 | `provisioning_mode === "always_on"` | `READY` |
| 4 | `on_demand` && `!isAuthenticated` | `SIGN_IN_REQUIRED` |
| 5 | `on_demand` && auth && `session_state === "none"` | `NEEDS_SESSION` |
| 6 | `on_demand` && auth && `session_state === "provisioning"` | `PROVISIONING` |
| 7 | `on_demand` && auth && `session_state === "active"` | `READY` |
| 8 | `on_demand` && auth && `session_state === "paused"` | `PAUSED` |
| 9 | `on_demand` && auth && `session_state === "stale"` | `STALE` |

Invariant to lean on: for an `on_demand` model that is `available` and authenticated, the backend always sets `session_state` to at least `"none"` (never `null`). So `null` `session_state` on an `on_demand` card ⟺ unauthenticated (row 4). Passing `isAuthenticated` explicitly is still clearer than inferring.

---

## Per-state UX & actions

`Generate` = the primary generation control. "enabled" means the user can submit a job.

| State | Badge / label | `Generate` | Primary CTA → action | Secondary | Driven by |
|---|---|---|---|---|---|
| `READY` (always_on) | none / "Ready" | **enabled** | — | — | static |
| `READY` (on_demand active) | "Session active" + running timer/cost | **enabled** | — | `Pause` → `POST /v1/sessions/{id}/pause`; `Stop` → `POST /v1/sessions/{id}/stop` | SSE |
| `NEEDS_SESSION` | "Needs GPU session" | disabled | **Start session** → `POST /v1/sessions { model }` (201) | cost/latency hint | SSE after start |
| `PROVISIONING` | "Starting…" + spinner (+ phase if fetched) | disabled | `Cancel` → `POST /v1/sessions/{id}/stop` | — | SSE |
| `PAUSED` | "Paused" | disabled | **Resume** → `POST /v1/sessions/{id}/resume` | `Stop` | SSE after resume |
| `STALE` | "Session unreachable" (warning) | disabled | **Stop & start new** → `POST …/stop` then `POST /v1/sessions { model }` | show `error_message` if present | SSE / manual |
| `SIGN_IN_REQUIRED` | "Sign in to use" | disabled | **Sign in** → auth flow, return to generate screen | — | — |
| `UNAVAILABLE` | "Temporarily unavailable" (muted) | disabled | none | optional tooltip: provider not configured | — |
| `DISABLED` | hidden/greyed | disabled | none | — | — |

Notes:
- `POST /v1/sessions` body field is **`model`** (the `model_key`); product is resolved server-side from the request host/middleware — do not send it.
- `PROVISIONING` may show finer detail (phase/progress) by fetching `GET /v1/sessions/{id}` once on entry; the catalog/SSE only carry the coarse status. Don't poll it in a tight loop — SSE drives the transition.
- The model card is the natural home for on-demand session state since `session_state` is per-model; `Stop`/`Pause` can live on the card or a sessions tray, your call.

---

## Shared status → state mapping (must mirror the backend)

The SSE payload carries the **raw `GpuSessionStatus`**, not the derived `ModelSessionState`. To keep SSE-driven and catalog-driven cards in agreement, the frontend needs the exact same mapping as backend `session_state_from_status`, **including `stopping → none`**:

```ts
// Mirror of backend ModelSessionState
export type SessionState = "none" | "provisioning" | "active" | "paused" | "stale";

// Raw lifecycle status as it arrives on SSE (GpuSessionStatus)
export type SessionStatus =
  | "pending" | "provisioning" | "active" | "stale"
  | "paused" | "resuming" | "stopping" | "stopped" | "failed";

export function sessionStateFromStatus(s: SessionStatus): SessionState {
  switch (s) {
    case "active":                                    return "active";
    case "pending":
    case "provisioning":
    case "resuming":                                  return "provisioning";
    case "paused":                                    return "paused";
    case "stale":                                     return "stale";
    case "stopping":
    case "stopped":
    case "failed":                                    return "none";
    // exhaustive: add a case if backend GpuSessionStatus grows
  }
}
```

Keep this in one module and unit-test it against the backend table so the two never drift (the backend has `TestSessionStateFromStatus` doing the equivalent).

---

## Real-time sync (no polling)

GPU session transitions are pushed over SSE; the frontend updates the relevant card in place rather than refetching `/v1/providers`.

1. `POST /v1/events/sse-ticket` → short-lived ticket (only if real-time is enabled; handle the "not available" response).
2. `GET /v1/events/stream?ticket=<ticket>` → `EventSource`.
3. Handle `event: gpu_session.status_changed`. Payload:

```ts
interface GpuSessionStatusPayload {
  session_id: string;
  status: SessionStatus;          // new status
  previous_status: SessionStatus;
  model_type: string;             // === ModelInfo.model_key — use to find the card
  bundle_name: string;
  tunnel_hostname?: string | null;
  error_message?: string | null;  // surface on stale/failed
}
```

On each event: `card[payload.model_type].session_state = sessionStateFromStatus(payload.status)` and re-derive the card state. Use `previous_status → status` for one-shot toasts (e.g. `provisioning → active` ⇒ "Session ready"; `* → stale` / `failed` ⇒ surface `error_message`).

**Fallbacks** (SSE ticket fails, EventSource drops, or real-time disabled): re-fetch `GET /v1/providers` (cheap, gives fresh `session_state`) or `GET /v1/sessions` on a slow interval / on window-focus. Treat the catalog as source of truth on (re)load; SSE only fast-forwards between loads.

---

## Decision points (recommended default + alternative)

**DA — Start trigger: explicit vs implicit.** *Recommended: explicit `Start session`, `Generate` stays disabled until `active`.* GPU sessions are billable (Vast.ai per-hour) and take time to provision; implicitly starting one behind a "Generate" click hides both the cost and a multi-second wait, which reads as a broken button. Explicit start makes intent and cost visible.
*Alternative:* one-click "Generate (starts a session)" gated behind a confirm that states the hourly cost and that generation begins once the node is ready. Choose this only if onboarding friction is the bigger concern; never auto-start silently.

**DB — Anonymous handling of on-demand models.** *Recommended: show them in `SIGN_IN_REQUIRED` (discoverable, drives signup), gate the action.* *Alternative:* hide on-demand models for anonymous users for a cleaner catalog. Cosmetic.

**DC — Stale recovery.** *Recommended: manual `Stop & start new`* — a stale node may be dead, and silent auto-restart re-bills without consent. Surface `error_message`. *Alternative:* if the backend later exposes a re-probe/heal action, offer `Retry` before `Stop & start new`.

**DD — Session controls location.** *Recommended: state + Pause/Stop on the model card* (since state is per-model), optionally mirrored in a global sessions tray. Pure UI.

---

## Acceptance checks

- A card with `provisioning_mode="on_demand"`, `available=false` renders `UNAVAILABLE` with **no** Start CTA (the finding-#3 regression: must not show "Start session").
- Anonymous load: every on-demand card is `SIGN_IN_REQUIRED`; every `always_on` available card is `READY`.
- Authenticated, no session: on-demand card is `NEEDS_SESSION`; pressing Start and receiving SSE `pending→provisioning→active` walks the card `NEEDS_SESSION → PROVISIONING → READY` with no manual refresh.
- `Generate` is enabled **iff** state is `READY`.
- `sessionStateFromStatus` covers all nine `GpuSessionStatus` values (unit-tested), matching the backend.
- Killing the SSE connection and reloading reproduces the same card states purely from `GET /v1/providers`.
