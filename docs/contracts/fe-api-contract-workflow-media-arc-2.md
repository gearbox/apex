# Frontend API contract — workflow-map & media-assets arc

**Backend source:** `gearbox/apex@9bd09cc` (`feat/b1-workflow-node-map`, on top of merged
`feat/source-media-assets`)
**Status:** contract is stable; regenerate `gen:api` once B1 merges.

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

### 1.1 New: `inputs`

```jsonc
{
  "model_key": "grok-imagine-image",
  "capabilities": ["t2i", "i2i"],
  "inputs": {
    "source_media": {
      "min": 1, "max": 4, "media_types": ["image"],
      "required_for": ["i2i"]
    }
  },
  "image": { /* unchanged: output + edit constraints */ }
}
```

`inputs` describes what a model **accepts**. `image` / `video` continue to describe what it
**produces**. They are no longer the same question.

| Field | Meaning |
|---|---|
| `inputs.source_media` | `null` ⇒ the model accepts no media input at all. Hide the picker. |
| `min` / `max` | Bound the **total** number of source assets, across all kinds. |
| `media_types` | Allowed asset media kinds. Values: `"image"`, `"video"`. |
| `required_for` | The `generation_type` values, among this model's `capabilities`, that require source media. Always a subset of `capabilities`. |

**`min` does not mean "required".** A model with `capabilities: ["t2i", "i2i"]` and `min: 1` needs
no input for `t2i` and at least one for `i2i`. Requiredness comes from `required_for`:

```
required  = inputs.source_media?.required_for.includes(generation_type)
bounds    = inputs.source_media.{min,max}            // when required
accepted  = inputs.source_media.media_types
```

**Do not hardcode which generation types consume media.** `required_for` exists so that a new
media-consuming type is a backend change only. It is derived server-side from the same source of
truth the validator uses, so it cannot disagree with `capabilities`.

Note `v2v` never appears in `required_for`, even on a model whose `capabilities` include it: v2v
consumes `input_video_url` rather than an owned library asset (§2.4). That is correct, not an
omission.

### 1.2 New: `unsupported_parameters`

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

### 1.3 Changed semantics: `capabilities`

For on-demand (Aisha) models this is now the **intersection** of the static registry and the
resolved bundle's workflow map. A bundle whose graph has no image loader advertises `["t2i"]`
even though the model previously advertised `["t2i", "i2i"]`.

Concretely, `zit.cyberrealistic` reports `capabilities: ["t2i"]`,
`supports_negative_prompt: false` and `inputs.source_media: null`. If the UI offers i2i for that
model, every such request 422s.

### 1.4 Changed semantics: `is_enabled`

For on-demand models, `is_enabled` is now `false` when the bundle index has not yet synced or the
bundle is unresolvable — even if the model is enabled in the database. The model still appears in
the list with its static constraints so a card can render, but it must be presented as
unavailable. Treat `is_enabled: false` as "render disabled", never as "hide".

### 1.5 Unchanged

`model_key`, `name`, `description`, `max_images`, `max_prompt_length`, `aspect_ratios`,
`requires_age_verification`, `session_state`, `image`, `video`. Grok models are unaffected by all
of the above except that they now also carry `inputs`.

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
- Schema bound is 1–8 items; the **real** bound is `inputs.source_media.max` (§1.1). Enforce the
  model's bound client-side so the user gets feedback before a round trip.
- The backend resolves each reference, verifies ownership and product scope, and rejects
  thumbnails and duplicates.

### 2.2 Deprecated aliases — remove them

`input_image_id`, `source_output_id` and `source_images` still work for one minor release and are
normalised server-side. Mapping:

| Legacy | Replacement |
|---|---|
| `"input_image_id": "X"` | `"source_media": [{ "asset_ref": "upload:X" }]` |
| `"source_output_id": "X"` | `"source_media": [{ "asset_ref": "output:X" }]` |
| `"source_images": [{input_image_id: A}, {source_output_id: B}]` | `"source_media": [{asset_ref: "upload:A"}, {asset_ref: "output:B"}]` |

**Combining any alias with `source_media` is a 422.** There is no precedence rule — send one
shape or the other. Every alias use is logged server-side as
`generation.request.legacy_source_field`; the aliases are removed once that log goes quiet, so
migrating promptly is what sets the removal date.

### 2.3 Unchanged request fields

`prompt`, `generation_type`, `model`, `negative_prompt`, `aspect_ratio`, `n`, `name`, `seed`,
`steps`, `cfg`, `denoise`, `sampler`, `scheduler`, `image_resolution`, `width`, `height`.

Their **acceptance** changed: any of them may now be rejected per-model via
`unsupported_parameters`.

### 2.4 `input_video_url` — unchanged, still v2v-only

`v2v` continues to take a public URL rather than an owned asset. It is **not** part of
`source_media`, is not covered by `inputs.source_media`, and its generation type deliberately
requires no owned media. This is scheduled to move to `source_media` in a later arc; until then,
keep the v2v path as it is.

---

## 3. Errors

All errors use the existing `ErrorEnvelope` shape: `{ error, message, status_code }`.

### 3.1 `validation_error` — 422

Returned for source-media problems: malformed `asset_ref`, unresolvable or non-owned reference,
thumbnail reference, duplicate references, count outside `min`/`max`, wrong media kind, media
supplied to a model that accepts none, or an alias combined with `source_media`.

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
- `generation_type` can appear as the offending parameter when the resolved bundle does not
  support the requested type — which is the §1.3 case reaching the server.

### 3.3 `not_implemented` — 400

One provider-level refusal can arrive even for a request that satisfies every advertised
constraint: an Aisha model currently accepts **exactly one** source asset, while a bundle
declaring two reference slots advertises `max: 2`. A two-item `source_media` therefore passes the
count check and is refused downstream with `error: "not_implemented"` and a 400.

Nothing is charged — the idempotency record is marked failed before the response. Until the
backend lifts the single-asset limit, treat `max` as an upper bound the server may still refuse,
and surface the message rather than assuming any within-bounds count will succeed.

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

### 4.2 `input_media` — deprecated

Still present, equal to the first available item's `media`. Use `source_media` instead; a single
envelope cannot represent an ordered multi-reference job.

### 4.3 `duration_ms` on `MediaObject`

Now populated for outputs where the provider reports it. `null` means unknown — including every
image — and must not be rendered as zero.

### 4.4 `media_type`

The wire values are unchanged (`"image"`, `"video"`). The backing enum was renamed server-side;
no client change is required.

---

## 5. TypeScript

```ts
type MediaKind = 'image' | 'video';   // 'audio' is reserved; do not switch exhaustively

interface SourceMediaConstraints {
  min: number;
  max: number;
  media_types: MediaKind[];
  /** generation_type values that require source media; subset of ModelInfo.capabilities */
  required_for: string[];
}

interface ModelInputs {
  source_media: SourceMediaConstraints | null;
}

type UnsupportedParameter =
  | 'aspect_ratio' | 'batch_size' | 'cfg' | 'denoise' | 'height'
  | 'image_resolution' | 'negative_prompt' | 'sampler' | 'scheduler'
  | 'seed' | 'steps' | 'width';

interface ModelInfo {
  // …existing fields…
  capabilities: string[];               // now bundle-derived for on-demand models
  is_enabled: boolean;                  // false when the bundle index has not synced
  inputs: ModelInputs;                  // new
  unsupported_parameters: UnsupportedParameter[];   // new
}

interface SourceMediaReference {
  asset_ref: string;                    // `upload:${uuid}` | `output:${uuid}`
}

interface GenerateRequest {
  // …existing fields…
  source_media?: SourceMediaReference[];
  /** @deprecated use source_media */ input_image_id?: string;
  /** @deprecated use source_media */ source_output_id?: string;
  /** @deprecated use source_media */ source_images?: unknown[];
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
  /** @deprecated use source_media[0] */ input_media?: MediaObject | null;
}
```

`MediaKind` is deliberately typed as a union rather than an enum, and clients should **not** write
exhaustive `switch` statements over it. A third value is expected and adding one must not be a
breaking change.

---

## 6. Migration checklist

1. Regenerate `gen:api` after B1 merges; confirm the emitted types match §5 field for field.
2. Replace every `input_image_id` / `source_output_id` / `source_images` write with `source_media`.
   Never send both shapes.
3. Drive the media picker from `inputs.source_media` — visibility from `!== null`, cardinality
   from `min`/`max`, accepted kinds from `media_types`, requiredness from
   `required_for.includes(generation_type)`. No hardcoded list of media-consuming types anywhere
   in the client.
4. Disable controls listed in `unsupported_parameters`. One mapping from parameter name to control,
   not per-model conditionals.
5. Render `is_enabled: false` on-demand models as unavailable rather than hiding them.
6. Stop inferring capability from `model_key`. Any `if (model === 'aisha-image')` guarding i2i,
   negative prompts or a batch selector is now wrong.
7. Rework Re-Generate onto `source_media`, handling `available: false` explicitly.
8. Treat `duration_ms: null` as unknown.

## 7. What has not changed

- Authentication, idempotency headers, SSE job events, and job/output polling.
- Grok request and response shapes, beyond gaining `inputs` on discovery.
- `input_video_url` and the v2v flow.
- Upload endpoints and `MediaObject`, apart from `duration_ms` now being populated.
- Pricing responses. Input count still drives price where applicable; it is now
  `source_media.length`.

## 8. Open questions for backend

Raise these before building against them rather than assuming:

- Whether `capabilities` can change mid-session for a model the user already has open — and if so,
  whether the client should re-fetch `/v1/providers` on a session-state transition.
- Whether `inputs.source_media.max` for `aisha-image` will track a bundle that declares two
  reference slots. It does under B1's derivation, but a bundle update changes it without a deploy,
  so the picker must read it per request rather than caching it for the session.
