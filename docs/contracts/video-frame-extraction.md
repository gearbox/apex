# Frontend Contract — Video Frame Extraction

> **Audience:** `gearbox/apex-frontend` (SvelteKit 2 / Svelte 5).
> **Backend source of truth:** the video-frame-extraction merge (upload content-type expansion, `FrameExtractionJob` queue, `POST/GET /v1/frames/*`).
> **Authority:** `gen:api` (OpenAPI) is authoritative for **types**; this document is authoritative for **semantics**, the polling model, presigned-URL expiry for previews, and the new upload error cases. Run `gen:api` after this lands.

---

## 0. Feature summary

Users can take any video — a generated `GenerationOutput` (Grok T2V/I2V) or a
newly-supported **user-uploaded video** — and:

1. **Preview** — request a strip of N uniformly-spaced, downscaled frames.
   Async job → presigned WEBP URLs + the exact timestamp of each frame.
2. **Extract** — submit selected timestamps; each becomes a full-resolution
   PNG saved as a regular upload (with lineage back to the source video),
   immediately usable for i2i/i2v generation and downloadable via the
   existing content proxy.

Frame extraction is **free** — no token charge, no `Idempotency-Key` header
required on any of the three new endpoints.

---

## 1. Expanded upload content types

`POST /v1/storage/upload` now also accepts video files, up to the existing
20 MB cap (unchanged):

| Content-Type | Extension |
|---|---|
| `video/mp4` | `.mp4` |
| `video/webm` | `.webm` |
| `video/quicktime` | `.mov` |

The existing image types (`image/png`, `image/jpeg`, `image/webp`,
`image/heic`, `image/heif`, `image/avif`) are unchanged.

**New rejection case — video probe failure.** Uploaded video bytes are
probed server-side (ffprobe) before being accepted; the client-declared
`Content-Type` is never trusted. On probe failure the endpoint returns the
existing `400 validation_error` shape:

```json
{
  "error": "validation_error",
  "message": "File is not a decodable video",
  "status_code": 400
}
```

A video whose duration exceeds the server's configured maximum (default 300s)
is rejected the same way, with a message like:

```json
{
  "error": "validation_error",
  "message": "Video duration 620.0s exceeds maximum 300s",
  "status_code": 400
}
```

**Response shape is unchanged** (`UploadResponse` — `id`, `filename`,
`created_at`, `expires_at`, `media`). For an accepted video:

- `media.media_type` is `"video"`.
- `media.original.content_type` is the video's MIME type (as uploaded — video
  bytes are stored as-is, never re-encoded, unlike images).
- `media.variants` may contain a poster-frame WEBP thumbnail (same `sm`/`md`
  bucket convention as existing video output posters) — **best-effort**, may
  be `[]` if poster extraction failed. Never block on it.

---

## 2. New endpoints

All three require `Authorization: Bearer <token>` (the standard access
token guard — same as every other `/v1/*` write endpoint). None require
`Idempotency-Key`.

### `POST /v1/frames/preview`

Request a low-res, N-frame preview strip for a video (either a generation
output or an uploaded video).

```ts
interface FramePreviewRequest {
  source_output_id?: string | null; // UUID — exactly one of these two
  source_upload_id?: string | null; // UUID
  frame_count?: number; // 2-60, default 12
}
```

Exactly one of `source_output_id` / `source_upload_id` must be set — `400
invalid_source` otherwise. The source must be a video (by stored
`content_type`) and must belong to the authenticated user — `400
not_a_video` / `404 not_found` otherwise.

**Response `202`:**

```ts
interface FrameJobCreatedResponse {
  job_id: string; // UUID
  status: "queued";
}
```

### `POST /v1/frames/extract`

Request full-resolution frame extraction at specific timestamps.

```ts
interface FrameExtractRequest {
  source_output_id?: string | null;
  source_upload_id?: string | null;
  timestamps_ms: number[]; // 1-50 entries, each >= 0
}
```

Same source validation as `/preview`. `timestamps_ms` shape (count, `>= 0`)
is validated at the request layer (`422`-style `400` on a malformed body via
the standard msgspec validation path); whether each timestamp is actually
within the video's duration is validated **after** the job starts running
(the worker has to probe the file anyway) — see §4, a job can fail with a
precise out-of-range error. The valid range is `[0, duration_ms)` — the
upper bound is **exclusive**, matching the preview strip's timestamps
(`compute_uniform_timestamps` never returns the exact end either), since an
end-of-stream seek frequently decodes nothing.

**Use the server-probed `duration_ms`, not the browser's, as the bound.**
`FramePreviewResult.duration_ms` (see `GET /jobs/{job_id}` below) is the
ffprobe result for the source video — the same value the worker validates
`timestamps_ms` against. Clamp scrubber selections to `duration_ms - 1`.
Do **not** use `videoEl.duration` as the bound: the browser's own decoder
can disagree with ffprobe by a frame or two, and a scrubber pinned to the
browser's end-of-video can still produce an out-of-range job failure that
the client cannot reliably prevent.

**Response:** same `FrameJobCreatedResponse` shape as `/preview`, `status:
"queued"`.

### `GET /v1/frames/jobs/{job_id}`

Poll a job's status/result. Ownership-checked — `404` for a job belonging to
another user.

```ts
interface FrameJobSource {
  type: "output" | "upload";
  id: string; // UUID
}

interface PreviewFrame {
  index: number;
  timestamp_ms: number;
  url: string; // presigned R2 URL — see §3, TTL-bounded, never persisted
}

interface FramePreviewResult {
  frames: PreviewFrame[];
  expires_in_seconds: number; // TTL of every url above, as of this response
  duration_ms: number; // server-probed (ffprobe) duration of the source video
}

interface ExtractedFrame {
  timestamp_ms: number;
  upload_id: string; // UUID — same id you'd get from POST /v1/storage/upload
  media: MediaObject; // the standard media envelope — see thumbnails-gallery-contract.md
}

interface FrameExtractResult {
  frames: ExtractedFrame[];
}

interface FrameJobResponse {
  job_id: string;
  kind: "preview" | "extract";
  status: "queued" | "running" | "completed" | "failed";
  created_at: string; // ISO 8601
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null; // populated only when status === "failed"
  source: FrameJobSource;
  preview?: FramePreviewResult | null; // present only when kind=preview AND status=completed
  extracted?: FrameExtractResult | null; // present only when kind=extract AND status=completed
}
```

A job may transiently report `status: "failed"` and later `status:
"completed"` — but only in the narrow worker-death-plus-recovery window: the
stale-sweep worker fails any `frame_extraction_jobs` row stuck `running`
longer than `frame_extract_stale_running_seconds` (default 1800s) on the
assumption its worker died; if that worker was merely slow and finishes
afterward, its unconditional completion flips the row back. This only ever
goes `failed → completed`, never the reverse — the sweep's `WHERE status =
'running'` guard prevents it from touching an already-completed job. Clients
should treat `completed` as terminal-authoritative.

---

## 3. Polling model

Both `POST /preview` and `POST /extract` are **fire-and-forget** — they
return `202` immediately with a `job_id` and never block on ffmpeg work.
Poll `GET /v1/frames/jobs/{job_id}` until `status` is `"completed"` or
`"failed"`:

```ts
async function pollFrameJob(jobId: string): Promise<FrameJobResponse> {
  for (;;) {
    const res = await apiFetch(`/v1/frames/jobs/${jobId}`);
    const job: FrameJobResponse = await res.json();
    if (job.status === "completed" || job.status === "failed") return job;
    await sleep(1000); // jobs typically finish in low single-digit seconds
  }
}
```

There is no SSE/push notification for job progress — jobs are short-lived
(seconds), so a cheap DB-only polling GET is sufficient. If very long
uploaded videos become common later, this may grow an SSE variant; not
today.

On `status === "failed"`, `error` is a human-readable message (ffmpeg/ffprobe
failure, or an out-of-range timestamp for extract jobs) — safe to display.

---

## 4. Presigned URL expiry semantics (preview only)

`FramePreviewResult.frames[].url` is a **presigned R2 URL generated fresh on
every `GET /jobs/{id}` call** — it is never the same URL twice and is never
persisted server-side. `expires_in_seconds` tells you how long *that specific
response's* URLs are valid for (default 3600s); if you hold onto a stale
`FrameJobResponse` and the URLs expire, just re-poll `GET /jobs/{id}` — you
will get fresh URLs with a fresh TTL, keyed off the same `job_id`.

**Do not cache preview frame URLs beyond the current page session.** Unlike
`/v1/content/*` proxy URLs (stable, non-expiring, cacheable indefinitely —
see `thumbnails-gallery-contract.md`), preview frames live at a
non-authenticated, top-level R2 prefix (`frame-previews/...`) that expires
via an R2 lifecycle rule (default 2 days) — there is no proxy indirection for
these, by design (D3: preview frames are stateless, no DB rows).

**Extracted frames are different:** `ExtractedFrame.media` uses the exact
same `MediaObject` shape as every other upload — its `original.url` /
`variants[].url` are stable `/v1/content/uploads/{id}` proxy paths, cached
indefinitely per the existing contract. Once a `POST /extract` job
completes, the resulting frames behave exactly like anything from `POST
/v1/storage/upload` — same download semantics, same deletion endpoint (`DELETE
/v1/content/{id}`), same retention/expiry.

---

## 5. Lineage fields on upload media

Extracted frames are ordinary uploads with two new nullable lineage fields
**not currently exposed in `MediaObject`** (they're DB-internal, resolved via
`ExtractedFrame.upload_id`/`FrameJobResponse.source`, not embedded in every
upload response) — call out here because they explain *why* an "upload" can
now have a source video:

- An extracted frame's origin is always recoverable via the
  `FrameJobResponse` that produced it (`source.type`/`source.id` +
  `ExtractedFrame.timestamp_ms`), not via a field on the upload/gallery list
  endpoints themselves. If product wants "extracted from video" badges in
  the gallery grid, that's a follow-up — flag if needed, it's not in this
  contract.
- Deleting the source video (`DELETE /v1/content/{id}`) does **not** delete
  frames already extracted from it — they become ordinary, source-less
  uploads. No special handling needed client-side.
