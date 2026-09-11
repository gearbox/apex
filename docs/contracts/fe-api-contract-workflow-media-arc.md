# Frontend API contract — workflow-map & media-assets arc

**Backend source:** `gearbox/apex@7b79e4b` (B1 workflow-node-map, merged as #160, on top of
merged `feat/source-media-assets`)
**Status:** B1 merged; contract is stable. Regenerate `gen:api` before the next frontend release.

Two changes drive everything here:

1. **Generation inputs are now one ordered list of library asset references** instead of four
   image-shaped fields.
2. **Model capabilities are derived per-bundle at runtime**, not from a static table. The same
   `model_key` can advertise different capabilities after a bundle update, and the frontend must
   read them rather than assume them.

The practical consequence: **do not hardcode what a model supports.** Everything the UI needs to
enable, disable or bound a control now comes from `GET /v1/providers`.

---

## 1. `GET /v1/providers`

### 1.1 New: `generation_modes` (authoritative)

```jsonc
{
  "model_key": "grok-imagine-image",
  "generation_modes": {
    "t2i": { "source_media": null },
    "i2i": {
      "source_media": {
        "min": 1, "max": 4, "media_types": ["image"], "roles": null
      }
    }
  },
  "image": { /* unchanged: output + edit constraints */ }
}
```

`generation_modes` describes what each mode **accepts**. `image` / `video` continue to describe what it
**produces**. They are no longer the same question.

| Field | Meaning |
|---|---|
| `generation_modes[type].source_media` | `null` ⇒ this mode accepts no media input. |
| `min` / `max` | Bound the total number of source assets for this mode. |
| `media_types` | Allowed asset media kinds. Values: `"image"`, `"video"`. |
| `roles` | `null` ⇒ positions are interchangeable. Otherwise `roles[i]` names the slot filled by source `i` (`reference`, `first_frame`, `last_frame`, or `source`). |

The backend still requires an explicit `generation_type`, but the user does not select one. The
frontend resolves it from the user action, selected source media, and advertised modes:

```text
model.generation_modes  +  selected source media  +  user action
                            ↓
                frontend resolves generation_type
                            ↓
                POST /v1/generate { generation_type, source_media }
```

For on-demand (Aisha) models, `generation_modes` is the intersection of what the provider
implementation can execute and what the resolved bundle declares. A bundle whose graph has no image
loader offers only `t2i`, even though the model is registered for `t2i` and `i2i` — so
`zit.cyberrealistic` reports `generation_modes: { "t2i": { "source_media": null } }`. Read the
offered modes per request rather than caching a model→modes map for the session; a bundle update
changes them without a deploy.

Because the two sides intersect, anything advertised is executable: a bundle cannot widen a mode
beyond what the provider can run, and a mode with no satisfiable intersection is not offered at all.

The action supplies intent; the advertised contracts say which resolutions are legal. Cardinality
alone is not enough: with two images selected, “add reference” keeps `i2v` where advertised,
while “add end frame” selects `flf2v`.

Video workflows currently cannot declare `reference` slots. Multi-reference video input is not
representable yet, so every current video mode has named positional roles rather than `roles: null`.

### 1.2 Removed compatibility fields: `capabilities`, `inputs`, and `required_for`

These fields no longer exist. `generation_modes` is the only discovery input contract.

`inputs.source_media` was hazardous because it was a lossy union across modes: for
`grok-imagine-video`, it described `media_types: ["image", "video"]` with `max: 1`, a shape no
individual mode accepts.

### 1.3 New: `unsupported_parameters`

```jsonc
"unsupported_parameters": ["cfg", "denoise", "negative_prompt"]
```

Controls the resolved bundle cannot apply. **Disable these inputs.** Sending a value that differs
from the model's default for a listed parameter returns 422 (§3.2).

Closed vocabulary — these are the only values that can appear:

```
aspect_ratio  batch_size  cfg  denoise  height  image_resolution
negative_prompt  sampler  scheduler  seed  steps  width
```

Notes:
- `width` and `height` travel together. When they are unsupported, `aspect_ratio` and
  `image_resolution` are listed too, because neither can be honoured without writable dimensions.
- `batch_size` unsupported ⇒ `max_images` is `1`; the count selector should be hidden, not just
  clamped.
- `negative_prompt` appears here **and** `supports_negative_prompt` is `false`. They are
  consistent; use either, but prefer `unsupported_parameters` so one code path drives every
  control.

### 1.4 Changed semantics: `is_enabled`

For on-demand models, `is_enabled` is now `false` when the bundle index has not yet synced or the
bundle is unresolvable — even if the model is enabled in the database. The model still appears in
the list with its static constraints so a card can render, but it must be presented as
unavailable. Treat `is_enabled: false` as "render disabled", never as "hide".

### 1.5 Unchanged

`model_key`, `name`, `description`, `max_images`, `max_prompt_length`, `aspect_ratios`,
`requires_age_verification`, `session_state`, `image`, `video`.

---

## 2. `POST /v1/generate`

### 2.1 New: `source_media`

```jsonc
{
  "prompt": "a brown dog on a crowded crossroad",
  "generation_type": "i2i",
  "model": "aisha-image",
  "source_media": [
    { "asset_ref": "upload:3f2a…" },
    { "asset_ref": "output:9b71…" }
  ]
}
```

- **Order is significant** and is preserved end to end. The first entry is the primary reference.
- `asset_ref` format is `"upload:<uuid>"` or `"output:<uuid>"` — the same wire format already used
  by `GET /v1/library/assets/{asset_ref}/lineage`. Build it by prefixing, and never parse a raw
  UUID out of it for display.
- Schema bound is 1–8 items; the **real** bound is the selected
  `generation_modes[generation_type].source_media` contract (§1.1). Enforce the mode's bound
  client-side so the user gets feedback before a round trip.
- The backend resolves each reference, verifies ownership and product scope, and rejects
  thumbnails and duplicates.

### 2.2 Removed aliases

`source_media` is the only source input. `input_image_id`, `source_output_id`, and `source_images`
are removed and rejected as unknown request fields.

### 2.3 Unchanged request fields

`prompt`, `generation_type`, `model`, `negative_prompt`, `aspect_ratio`, `n`, `name`, `seed`,
`steps`, `cfg`, `denoise`, `sampler`, `scheduler`, `image_resolution`, `width`, `height`.

Their **acceptance** changed: any of them may now be rejected per-model via
`unsupported_parameters`.

### 2.4 V2V uses `source_media`

`v2v` takes exactly one owned video library asset in `source_media`. `input_video_url` has been
removed and is rejected as an unknown request field. This gives v2v the same ownership checks,
ordered provenance, and source expiry re-arming as every other source-media mode.

---

## 3. Errors

All errors use the existing `ErrorEnvelope` shape: `{ error, message, status_code }`.

### 3.1 `validation_error` — 422

Returned for source-media problems: malformed `asset_ref`, unresolvable or non-owned reference,
thumbnail reference, duplicate references, count outside `min`/`max`, wrong media kind, or media
supplied to a model that accepts none.

The `message` names the **position** in the list, not the asset id, so it is safe to surface
directly. A missing asset and an asset belonging to another user return an identical response by
design — do not try to distinguish them.

### 3.2 `unsupported_generation_parameter` — 422

```jsonc
{ "error": "unsupported_generation_parameter",
  "message": "Unsupported generation parameter(s): cfg, denoise",
  "status_code": 422 }
```

Raised **before billing**, so no tokens are reserved and nothing is charged.

Two things worth knowing:

- A parameter equal to the model's default never triggers this. Only a value that **differs** from
  the bundle default and is not writable is rejected. So leaving controls at their defaults is
  always safe.
- `generation_type` can appear as the offending parameter when the resolved effective mode set
  does not offer the requested type.

### 3.3 `not_implemented` — 400

This remains a provider-level refusal for unsupported features unrelated to an advertised
source-media shape. Nothing is charged; surface the returned message.

If the UI honours `unsupported_parameters`, the 422 in §3.2 should be unreachable. Treat it as a bug
signal (log it) rather than a routine user-facing state, but still render the message.

---

## 4. `GET /v1/library/groups/{job_id}`

### 4.1 New: `source_media`

```jsonc
"source_media": [
  { "position": 0, "asset_ref": "upload:3f2a…", "available": true,  "media": { /* MediaObject */ } },
  { "position": 1, "asset_ref": "output:9b71…", "available": false, "media": null }
]
```

Ordered, and **includes positions whose asset no longer exists**. Retention nulls the underlying
reference but keeps `asset_ref` and `position`.

This is what makes Re-Generate correct:

- All `available` ⇒ replay `source_media` verbatim as the new request's `source_media`.
- Any `available: false` ⇒ **do not silently replay a shorter list.** Show a placeholder at that
  position and either disable Re-Generate or require the user to substitute an asset. Replaying a
  two-reference edit as a one-reference edit produces a different image with no indication
  anything changed.

### 4.2 Removed `input_media`

`source_media` is the only generation-input authority on a library group. Its ordered entries and
`available` flags preserve more information than the removed single-media projection.

### 4.3 `duration_ms` — on the asset schemas, **not** on `MediaObject`

The only components exposing `duration_ms` are `LibraryAssetItem`
(`GET /v1/library/assets`), `LibraryAssetDetail`, and `FramePreviewResult`.

`MediaObject` does **not** carry it, so it is absent everywhere `MediaObject` is embedded:
`LibraryGroupDetail.source_media[].media`, job outputs and frames media.

Population:

| Asset | `duration_ms` |
|---|---|
| Uploaded video | populated — ffprobe measurement taken at upload |
| Uploaded image | null |
| Generated output (any) | always null — the column exists and is read, but nothing writes it |

Treat `null` as unknown. Never render it as zero, and do not use its presence to infer
media type — `media_type` on `MediaObject` is the discriminator.

### 4.4 `media_type`

The wire values are unchanged (`"image"`, `"video"`). The backing enum was renamed server-side;
no client change is required.

---

## 5. TypeScript

```ts
type MediaKind = 'image' | 'video';   // 'audio' is reserved; do not switch exhaustively

interface SourceMediaModeConstraints {
  min: number;
  max: number;
  media_types: MediaKind[];
  /** null means positions are interchangeable */
  roles: ('reference' | 'first_frame' | 'last_frame' | 'source')[] | null;
}

interface GenerationModeInfo {
  source_media: SourceMediaModeConstraints | null;
}

type UnsupportedParameter =
  | 'aspect_ratio' | 'batch_size' | 'cfg' | 'denoise' | 'height'
  | 'image_resolution' | 'negative_prompt' | 'sampler' | 'scheduler'
  | 'seed' | 'steps' | 'width';

interface ModelInfo {
  // …existing fields…
  generation_modes: Record<string, GenerationModeInfo>; // authoritative
  is_enabled: boolean;                  // false when the bundle index has not synced
  unsupported_parameters: UnsupportedParameter[];   // new
}

interface SourceMediaReference {
  asset_ref: string;                    // `upload:${uuid}` | `output:${uuid}`
}

interface GenerateRequest {
  // …existing fields…
  source_media?: SourceMediaReference[];
}

interface LibrarySourceMediaItem {
  position: number;
  asset_ref: string;
  available: boolean;
  media: MediaObject | null;
}

interface LibraryGroupDetail {
  // …existing fields…
  source_media: LibrarySourceMediaItem[];
}
```

`MediaKind` is deliberately typed as a union rather than an enum, and clients should **not** write
exhaustive `switch` statements over it. A third value is expected and adding one must not be a
breaking change.

---

## 6. Migration checklist

1. Regenerate `gen:api` before the next frontend release; confirm the emitted types match §5
   field for field.
2. Send `source_media` as the only source input.
3. Drive the media picker from `generation_modes[generation_type].source_media` — visibility from
   `!== null`, requiredness and cardinality from `min`/`max`, accepted kinds from `media_types`,
   and positional labels from `roles`. No hardcoded list of media-consuming types anywhere in the
   client.
4. Disable controls listed in `unsupported_parameters`. One mapping from parameter name to control,
   not per-model conditionals.
5. Render `is_enabled: false` on-demand models as unavailable rather than hiding them.
6. Stop inferring capability from `model_key`. Any `if (model === 'aisha-image')` guarding i2i,
   negative prompts or a batch selector is now wrong.
7. Rework Re-Generate onto `source_media`, handling `available: false` explicitly.
8. Treat `duration_ms: null` as unknown wherever the asset or preview schema exposes it.

## 7. What has not changed

- Authentication, idempotency headers, SSE job events, and job/output polling.
- Grok request and response shapes, beyond gaining `generation_modes` on discovery and moving v2v
  to owned `source_media`.
- Upload endpoints and the `MediaObject` shape.
- Pricing responses. Input count still drives price where applicable; it is now
  `source_media.length`.

## 8. Open questions for backend

Raise these before building against them rather than assuming:

- Whether `generation_modes` can change mid-session for a model the user already has open — and if so,
  whether the client should re-fetch `/v1/providers` on a session-state transition.
