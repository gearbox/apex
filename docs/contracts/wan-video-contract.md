# Frontend Contract — Wan 3.0 Video (`wan3.0-video`)

> **Audience:** `gearbox/apex-frontend` (video generation UI).
> **Backend source:** branch `feat/wan3-model-studio` (migration `052_wan3_model_studio`).
> **Authority:** `gen:api` (OpenAPI `schema.json`) is authoritative for **types**. This document is authoritative for **semantics**: what each mode accepts, what to send, and what comes back. It is the only copy — do not duplicate it elsewhere.

---

## 0. Summary

- One new model, `wan3.0-video`, served by the new always-on provider `alibaba` (display name **"Alibaba Wan"**), through Alibaba Model Studio.
- Five modes: `t2v`, `i2v`, `flf2v`, `v2v` (edit **or** extend) and the new **`r2v`** (reference images → video).
- Same `POST /v1/generate` as every other model. Jobs are asynchronous: the response is `queued`, and the job completes through the normal job status / SSE flow, typically in **1–5 minutes**.
- Everything model-specific is discoverable on `GET /v1/providers` (§1). Build the UI from that payload, not from this document's numbers.
- Wan is **off until an admin enables it** and prices exist (§9). Until then it is simply absent from `GET /v1/providers`.

---

## 1. Discovery — `GET /v1/providers`

The `alibaba` provider appears with `provisioning_mode: "always_on"`. Its model:

```jsonc
{
  "model_key": "wan3.0-video",
  "name": "Wan 3.0",
  "is_enabled": true,
  "max_images": 1,                 // one video per request
  "max_prompt_length": 4096,
  "supports_negative_prompt": false,   // do not send negative_prompt
  "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"],
  "generation_modes": {            // enum order
    "t2v":   { "source_media": null },
    "i2v":   { "source_media": { "min": 1, "max": 1, "media_types": ["image"], "roles": ["first_frame"] } },
    "v2v":   { "source_media": { "min": 1, "max": 1, "media_types": ["video"], "roles": null } },
    "flf2v": { "source_media": { "min": 2, "max": 2, "media_types": ["image"], "roles": ["first_frame", "last_frame"] } },
    "r2v":   { "source_media": { "min": 1, "max": 8, "media_types": ["image"], "roles": null } }
  },
  "video": {
    "min_duration": 2,
    "max_duration": 30,
    "resolutions": ["480p", "720p", "1080p"],
    "supported_parameters": ["audio", "prompt_extend"],
    "input_video": {                // v2v source limits; null = the model takes no video source
      "formats": ["mov", "mp4"],    // sorted; filter the library picker by these
      "min_duration_ms": 1000,
      "max_duration_ms": 15000,
      "max_bytes": 100000000
    },
    "max_combined_duration": 30,
    "input_anchored_modes": ["flf2v", "i2v", "v2v"]
  },
  "seed_range": { "min": 0, "max": 2147483647 }   // null = no model-specific bound
}
```

`seed_range` (`{min, max}` or `null`) is a new field on the model itself, additive for every model: only Wan declares one today.

New wire fields on `video` (additive for every model): `min_duration`, `supported_parameters`, `input_video`, `max_combined_duration`, `input_anchored_modes`. Other video models report `supported_parameters: []`, `input_anchored_modes: []` and `null` limits. `input_video` replaces an earlier single per-clip seconds cap that was never released (hard cutover: there is no compatibility field); it is `null` for a model that takes no video source (Aisha video), and Grok reports all three library containers with no duration or size limit.

`input_anchored_modes` (sorted generation-mode keys) lists the modes whose output ratio **follows the input**: send no `aspect_ratio` for them. Every other mode accepts one even when `source_media` is present. It replaces an earlier per-model boolean that was never released (hard cutover: there is no compatibility field).

`generation_type` gains **`r2v`**. `duration` is now `1–30` at the schema level; the model's real bounds are `video.min_duration`–`video.max_duration` (Grok 1–15, Aisha 1–10, Wan 2–30). `resolution` gains `"1080p"` (Wan only).

---

## 2. Request examples

All requests also need `Idempotency-Key`. `source_media` items are library references (`"upload:<uuid>"` / `"output:<uuid>"`), sent in the order shown.

**Text to video**

```json
{ "model": "wan3.0-video", "generation_type": "t2v", "prompt": "A fox runs through snow at dawn",
  "duration": 5, "resolution": "720p", "aspect_ratio": "16:9", "audio": false }
```

**Image to video** (the image is the first frame)

```json
{ "model": "wan3.0-video", "generation_type": "i2v", "prompt": "The camera slowly pushes in",
  "source_media": [{ "asset_ref": "upload:…" }], "duration": 5 }
```

**First + last frame**

```json
{ "model": "wan3.0-video", "generation_type": "flf2v", "prompt": "Morning turns to night",
  "source_media": [{ "asset_ref": "upload:…first" }, { "asset_ref": "output:…last" }], "duration": 8 }
```

**Reference images** — see §3.

```json
{ "model": "wan3.0-video", "generation_type": "r2v",
  "prompt": "Image 1 and Image 2 sit at a café table and talk",
  "source_media": [{ "asset_ref": "upload:…" }, { "asset_ref": "upload:…" }], "duration": 6 }
```

**Video edit / extend** — see §4.

```json
{ "model": "wan3.0-video", "generation_type": "v2v",
  "prompt": "Extend this video: the camera keeps following the fox into the forest",
  "source_media": [{ "asset_ref": "output:…" }], "duration": 10 }
```

---

## 3. Reference images (`r2v`)

- Send **1–8** images in `source_media`; order is significant.
- In the prompt, refer to them as **"Image 1" … "Image N"** in `source_media` order (1-based). The model resolves those names; there is no other way to address a reference.
- Reference video and reference audio are **not** supported in `r2v` yet.
- `aspect_ratio` is **optional** in `r2v` (references are characters and objects, not frames, so square or portrait sheets are normal). Omitted ⇒ `adaptive`; set it to pick the output ratio, e.g. `"16:9"`.

## 4. Video edit and extend (`v2v`)

- Exactly **one** video in `source_media`. It must have a known duration (library videos do); a video without one is rejected with `422`.
- The source must be a container the model accepts, listed in `video.input_video.formats` (Wan: **mp4, mov**; a library **WebM** is not accepted). Filter the V2V library picker by that list rather than letting the user submit and fail.
- Whether the request **edits** or **extends** is decided **by the prompt wording**, so say it: *"Extend this video: …"* or *"Edit this video: change the jacket to red"*. There is no separate parameter.
- Limits (from `video.input_video`): the source is **1–15 s** (`min_duration_ms`, `max_duration_ms`) and at most **100 000 000 bytes** (`max_bytes`, decimal 100 MB — a Wan output re-used as a source can exceed it), and **source length (rounded up) + `duration` must be ≤ 30 s** (`video.max_combined_duration`). Violations are rejected (`422`) before anything is charged.
- **Not checked locally:** the provider also requires ≥ 16 fps and 240–4096 px per side for a video source (and rejects some image inputs, e.g. PNG alpha). The library does not carry dimensions or frame rate, so such a source passes validation and fails upstream as `provider_invalid_request`, which is refunded.
- Edits can come back close to the source length rather than `duration`; see §7.

## 5. Parameters

| Field | Wan behaviour |
|---|---|
| `duration` | 2–30 s. |
| `resolution` | `480p`, `720p` (default), `1080p`. |
| `aspect_ratio` | One of the model's five ratios. **Omit it in the modes listed in `video.input_anchored_modes`** (`i2v`, `flf2v`, `v2v`) — the output follows the input frame(s) or video, and sending a value is a `400 validation_error`. `t2v` and `r2v` accept it: omitted, `t2v` renders `16:9` and `r2v` renders `adaptive`. |
| `audio` | `true`/`false` to force; **`null` or omitted = the model's default**. |
| `prompt_extend` | `true`/`false`; `null` or omitted = the model's default (the provider rewrites your prompt with an LLM). Send `false` for precise directing. |
| `seed` | **0–2147483647**; omit it (or send `null`) for a random one. `-1` is rejected — there is one spelling of "random". The bounds are `seed_range` on `GET /v1/providers`. |
| `negative_prompt` | Not supported — do not send. |

Sending `audio` or `prompt_extend` to a model that does not list it in `video.supported_parameters` (any non-Wan model, or an image model) is a `422 unsupported_generation_parameter` that names the parameter. `null` is always accepted. The output is never watermarked.

## 6. Errors surfaced

Standard envelope; the codes you will see for Wan:

| Code | HTTP | Meaning |
|---|---|---|
| `validation_error` | 400 | Duration, resolution or aspect ratio outside what the model supports (including `aspect_ratio` sent in an input-anchored mode: `i2v`, `flf2v`, `v2v`), or a `seed` outside `seed_range` (including `-1`). |
| `validation_error` | 422 | `source_media` problems: wrong count/kind, a source video of unknown duration, a source container not accepted (`video.input_video.formats`), a source over `max_bytes`, shorter than `min_duration_ms` (1 s) or longer than `max_duration_ms` (15 s), or over the 30 s combined limit. |
| `unsupported_generation_parameter` | 422 | `audio` or `prompt_extend` sent to a model that does not list it in `video.supported_parameters`; the message names the parameter. |
| `price_not_found` | 404 | No pricing rule for this model/resolution yet (§9). |
| `provider_invalid_request` | 400 | The provider rejected the request shape. |
| `provider_moderation_rejected` | 422 | Content rejected by the provider's safety system. **Refunded** (the provider only bills successful generations). |
| `provider_rate_limited` | 429 | Provider throttling; retry later. |
| `provider_malformed_response`, `provider_output_not_delivered` | 502 | The provider produced something we could not read or fetch. `provider_output_not_delivered` means the provider generated the video and billed us but we could not obtain it (an unreadable file, or a result URL the provider no longer serves, i.e. HTTP 4xx), so the message states the job **was charged**. It is reported on the first poll; a transient download error (5xx, 408, 425, 429, network) is retried instead. |
| `provider_timeout`, `provider_unavailable`, `provider_authentication_failed`, `provider_unknown` | 503 | Provider-side trouble; refunded. |
| `insufficient_balance` | 402 | Balance below the quote. |
| `job_in_progress` | 409 | `DELETE /v1/jobs/{id}` on a `pending`, `queued` or `running` job. The job is already charged and cannot be stopped; delete it once it finishes. Hide or disable "Delete" for in-flight jobs. |

Failures that happen after the job is accepted arrive as the job's terminal `failed` state with the same `failure_code` and a public message.

An authentication or permission failure while *checking* an accepted job (rotated or revoked key, workspace permission) is **not** a job failure: the task keeps running and billing upstream, so the job stays `running` and is retried every poll (operators see `alibaba.poll_auth_failed` at `error` level). If the key is never fixed the job ends as `provider_timeout` after `WAN_VIDEO_MAX_POLL_TIME` and is refunded. A rejected key on *submission* still fails the request immediately with `provider_authentication_failed` (nothing was accepted, nothing is charged).

## 7. Cost

`tokens = (token_cost + input_token_cost × k + per_second_token_cost × s) × n`. For Wan, `n` is `1`, and `s` is `duration` — plus, for `v2v`, the source video's length rounded up to whole seconds (Wan bills the input as well). Prices are per resolution. `GET /v1/billing/pricing` lists the rules (`per_second_token_cost`, `video_resolution`).

The quote assumes the requested `duration`. An edit can finish near the source length, so the provider's billed seconds can differ from the quote; the backend only records this for now — there is no automatic extra charge or refund yet.

## 8. Accepted-task receipts (backend behaviour, no wire change)

A provider accepts a Wan task **inside the still-open request transaction**; nothing is durable until the route commits. A failure in between (the `QUEUED` write, the job-source write, idempotency completion, the commit itself) rolls back the job row and the reservation while the task keeps running upstream. The client sees `500 Generation failed due to an internal error.` — and **is not charged**: no ledger entry survives, and the backend never adopts-and-charges an orphan.

Mechanism (shared with Grok video):

1. Right after acceptance the task id is inserted into `accepted_task_receipts` and **committed on a dedicated two-connection pool** (never the request pool).
2. The request transaction then `DELETE`s that row. The delete holds the row lock until the request ends: a commit removes the receipt atomically with the job; a rollback brings it back, unlocked — an orphan by construction.
3. Immediately, if a job-service write failed after acceptance, the task is cancelled in-process (best chance it is still `PENDING`).
4. A per-provider reconciler (`alibaba_video_receipts`, `grok_video_receipts`) reads orphans with `FOR UPDATE SKIP LOCKED` — it can never touch an in-flight request's receipt — and **observes first, then acts**. The reconciler runs wherever workers run, independently of the poller (`WAN_VIDEO_WORKER_IN_PROCESS` / `GROK_VIDEO_WORKER_IN_PROCESS` only choose whether that process also polls).

| Receipt status | Meaning |
|---|---|
| `pending` | Recorded. Either an in-flight request still holds its lock, or it is an orphan not yet reconciled. |
| `cancelled` | Upstream `CANCELED` (cancelled immediately, or by the reconciler while `PENDING`). |
| `absorbed` | Not cancellable (running or finished upstream, Model Studio refused with `UnsupportedOperation`, or the provider has no cancel — Grok). Left to finish; `upstream_status` is recorded. |

Model Studio only cancels `PENDING` tasks (`POST /api/v1/tasks/{task_id}/cancel`; any other status is HTTP 400 `UnsupportedOperation`). xAI documents no cancel for video requests, so Grok orphans are always `absorbed`.

Ops alerts (Telegram, product-scoped, IDs only): `video_task.orphaned` once per receipt on its first resolution (`cancelled` or `absorbed`; an absorbed task may still bill the provider account — audit it by task id in the provider console) and `video_task.receipt_unrecorded` when the receipt itself could not be written (the task id is in the alert).

Frontend: nothing changes. A retry with the same `Idempotency-Key` after such a 500 submits a fresh task, as before.

## 9. Rollout order (admins)

1. Create the pricing rules: `POST /v1/admin/pricing` with `provider: "alibaba"`, `model: "wan3.0-video"`, the `generation_type`, `video_resolution` and `per_second_token_cost` taken from the Model Studio console. Pricing has **no `generation_type` wildcard**, so a rule is needed for every combination: **5 modes (`t2v`, `i2v`, `v2v`, `flf2v`, `r2v`) × 3 resolutions (`480p`, `720p`, `1080p`) = 15 rules**. A missing combination fails only that mode/resolution with `price_not_found`. Per-second or resolution-specific rules must target a video mode, and every rule needs `token_cost > 0` or `per_second_token_cost > 0` (`422 invalid_pricing_rule` otherwise).
2. Only then enable the model: `PATCH /v1/admin/models/wan3.0-video`.

Enabling first makes every request fail with a missing-price error (deliberately fail-loud).
