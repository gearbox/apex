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
- Wan is **off until an admin enables it** and prices exist (§8). Until then it is simply absent from `GET /v1/providers`.

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
    "max_input_video_seconds": 15,
    "max_combined_duration": 30,
    "aspect_ratio_with_input_media": false
  }
}
```

New wire fields on `video` (additive for every model): `min_duration`, `supported_parameters`, `max_input_video_seconds`, `max_combined_duration`, `aspect_ratio_with_input_media`. Other video models report `supported_parameters: []`, `aspect_ratio_with_input_media: true` and `null` limits.

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

## 4. Video edit and extend (`v2v`)

- Exactly **one** video in `source_media`. It must have a known duration (library videos do); a video without one is rejected with `422`.
- Whether the request **edits** or **extends** is decided **by the prompt wording**, so say it: *"Extend this video: …"* or *"Edit this video: change the jacket to red"*. There is no separate parameter.
- Limits (from `video`): the source is at most **15 s** (`max_input_video_seconds`), and **source length (rounded up) + `duration` must be ≤ 30 s** (`max_combined_duration`). Violations are rejected (`422`) before anything is charged.
- Edits can come back close to the source length rather than `duration`; see §7.

## 5. Parameters

| Field | Wan behaviour |
|---|---|
| `duration` | 2–30 s. |
| `resolution` | `480p`, `720p` (default), `1080p`. |
| `aspect_ratio` | One of the model's five ratios. **Omit it whenever `source_media` is present** (`aspect_ratio_with_input_media: false`) — the output follows the input, and sending a value is a `400 validation_error`. When omitted, text-to-video renders `16:9`. |
| `audio` | `true`/`false` to force; **`null` or omitted = the model's default**. |
| `prompt_extend` | `true`/`false`; `null` or omitted = the model's default (the provider rewrites your prompt with an LLM). Send `false` for precise directing. |
| `seed` | Forwarded when set; omitted = random. |
| `negative_prompt` | Not supported — do not send. |

Sending `audio` or `prompt_extend` to a model that does not list it in `video.supported_parameters` (any non-Wan model, or an image model) is a `422 unsupported_generation_parameter` that names the parameter. `null` is always accepted. The output is never watermarked.

## 6. Errors surfaced

Standard envelope; the codes you will see for Wan:

| Code | HTTP | Meaning |
|---|---|---|
| `validation_error` | 400 | Duration, resolution or aspect ratio outside what the model supports (including `aspect_ratio` sent with `source_media`). |
| `validation_error` | 422 | `source_media` problems: wrong count/kind, a source video of unknown duration, over 15 s, or over the 30 s combined limit. |
| `unsupported_generation_parameter` | 422 | `audio` or `prompt_extend` sent to a model that does not list it in `video.supported_parameters`; the message names the parameter. |
| `price_not_found` | 404 | No pricing rule for this model/resolution yet (§8). |
| `provider_invalid_request` | 400 | The provider rejected the request shape. |
| `provider_moderation_rejected` | 422 | Content rejected by the provider's safety system. **Refunded** (the provider only bills successful generations). |
| `provider_rate_limited` | 429 | Provider throttling; retry later. |
| `provider_malformed_response`, `provider_output_not_delivered` | 502 | The provider produced something we could not read or fetch. `provider_output_not_delivered` means the provider generated the video and billed us, so the message states the job **was charged**. |
| `provider_timeout`, `provider_unavailable`, `provider_authentication_failed`, `provider_unknown` | 503 | Provider-side trouble; refunded. |
| `insufficient_balance` | 402 | Balance below the quote. |

Failures that happen after the job is accepted arrive as the job's terminal `failed` state with the same `failure_code` and a public message.

## 7. Cost

`tokens = (token_cost + input_token_cost × k + per_second_token_cost × s) × n`. For Wan, `n` is `1`, and `s` is `duration` — plus, for `v2v`, the source video's length rounded up to whole seconds (Wan bills the input as well). Prices are per resolution. `GET /v1/billing/pricing` lists the rules (`per_second_token_cost`, `video_resolution`).

The quote assumes the requested `duration`. An edit can finish near the source length, so the provider's billed seconds can differ from the quote; the backend only records this for now — there is no automatic extra charge or refund yet.

## 8. Rollout order (admins)

1. Create pricing rules per resolution: `POST /v1/admin/pricing` with `provider: "alibaba"`, `model: "wan3.0-video"`, the `generation_type`, `video_resolution` and `per_second_token_cost` taken from the Model Studio console.
2. Only then enable the model: `PATCH /v1/admin/models/wan3.0-video`.

Enabling first makes every request fail with a missing-price error (deliberately fail-loud).
