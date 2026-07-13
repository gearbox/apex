# Frontend Contract — Video Frame Extraction

> **Audience:** `gearbox/apex-frontend` (SvelteKit 2 / Svelte 5).
> **Backend source of truth:** `gearbox/apex` branch `video-frame-extraction` @ `71a3fc241d213a1ab2ededc17da59d04bf433acc` (2026-07-12, master merged in) **plus** the `duration_ms` remediation commit (`feat(frames): expose probed duration_ms in preview job result`) — verify it is on the branch before integrating against §6. Every claim below was verified against that ref — routes (`src/api/routes/frames.py`), schemas (`src/api/schemas/frames.py`), worker (`src/api/services/frames/worker.py`), ffmpeg layer (`src/api/services/frames/ffmpeg.py`), upload path (`src/api/services/user_content.py`), and settings (`src/core/config.py`).
> **Authority:** `gen:api` (OpenAPI) is authoritative for **types**; this document is authoritative for **semantics** — the polling model, presigned-URL expiry for previews, timestamp bounds, and the new upload error cases. Run `gen:api` after the branch merges.

---

## 0. Feature summary

Users can take any video — a generated `GenerationOutput` (Grok T2V/I2V) or a newly-supported **user-uploaded video** — and:

1. **Preview** — request a strip of N uniformly-spaced, downscaled frames. Async job → presigned WEBP URLs + the exact timestamp of each frame.
2. **Extract** — submit selected timestamps; each becomes a **full-resolution PNG** saved as a regular upload (with lineage back to the source video), immediately usable as `input_image_id` for i2i/i2v generation and downloadable via the existing content proxy.

Frame extraction is **free** — no token charge, no `Idempotency-Key` header required on any of the three new endpoints.

---

## 1. Expanded upload content types

`POST /v1/storage/upload` now also accepts video files, up to the existing 20 MB request cap (unchanged):

| Content-Type | Extension |
|---|---|
| `video/mp4` | `.mp4` |
| `video/webm` | `.webm` |
| `video/quicktime` | `.mov` |

The existing image types (`image/png`, `image/jpeg`, `image/webp`, `image/heic`, `image/heif`, `image/avif`) are unchanged.

**Server-side probe — never trust the declared MIME.** Uploaded video bytes are probed with ffprobe before acceptance. Two new rejection cases, both the existing `400 validation_error` envelope:

```json
{ "error": "validation_error", "message": "File is not a decodable video", "status_code": 400 }
```

```json
{ "error": "validation_error", "message": "Video duration 620.0s exceeds maximum 300s", "status_code": 400 }
```

The duration cap is server-configured (`FRAME_EXTRACT_MAX_VIDEO_SECONDS`, default **300 s**). Surface the message verbatim — it is safe to display.

**Response shape is unchanged** (`UploadResponse` — `id`, `filename`, `created_at`, `expires_at`, `media`). For an accepted video:

- `media.media_type` is `"video"`.
- `media.original.content_type` is the video's MIME type as uploaded — **video bytes are stored as-is, never re-encoded** (unlike images, which are normalized).
- `media.original.width`/`height` come from the ffprobe result.
- `media.variants` may contain poster-frame WEBP thumbnails (same `sm`/`md` bucket convention as video output posters). **Best-effort — may be `[]`.** Never block on it.
- `duration_ms` is probed and stored server-side but is not in the upload response — the FE receives it from the preview job (§2, §6), which is the first point the scrubber UI needs it.

---

## 2. New endpoints

All three require `Authorization: Bearer <token>` (standard access-token guard, same as every other `/v1/*` write endpoint). None require `Idempotency-Key`.

### `POST /v1/frames/preview`

Request a low-res, N-frame preview strip for a video (generation output or uploaded video).

```ts
interface FramePreviewRequest {
  source_output_id?: string | null; // UUID — exactly one of these two
  source_upload_id?: string | null; // UUID
  frame_count?: number;             // 2–60, default 12
}
```

Exactly one of `source_output_id` / `source_upload_id` must be set — `400 invalid_source` otherwise. The source must be a video (by stored `content_type`) and must belong to the authenticated user — `400 not_a_video` / `404 not_found` otherwise. Unknown body fields are rejected (`forbid_unknown_fields`).

**Response `202`:**

```ts
interface FrameJobCreatedResponse {
  job_id: string;   // UUID
  status: "queued";
}
```

### `POST /v1/frames/extract`

Request full-resolution frame extraction at specific timestamps.

```ts
interface FrameExtractRequest {
  source_output_id?: string | null;
  source_upload_id?: string | null;
  timestamps_ms: number[]; // 1–50 entries, each >= 0
}
```

Same source validation as `/preview`. The `timestamps_ms` *shape* (count, `>= 0`) is validated at the request layer (standard msgspec `400` on a malformed body); whether each timestamp is within the video's duration is validated **after the job starts running** (the worker probes the file anyway) — an out-of-range timestamp fails the whole job asynchronously with a precise message, e.g. `Timestamps out of range [0, 5120)ms: [5120]`. The valid range is `[0, duration_ms)` — **upper bound exclusive**, since an end-of-stream seek frequently decodes nothing. The bound itself, `duration_ms`, comes from the preview result — see §6 for the clamping rule that makes an out-of-range failure unreachable.

**Response:** same `FrameJobCreatedResponse`, `status: "queued"`.

### `GET /v1/frames/jobs/{job_id}`

Poll a job's status/result. Ownership-checked — `404 not_found` for a job belonging to another user.

```ts
interface FrameJobSource {
  type: "output" | "upload";
  id: string; // UUID
}

interface PreviewFrame {
  index: number;
  timestamp_ms: number;
  url: string; // presigned R2 URL — TTL-bounded, regenerated per poll, never persisted (§4)
}

interface FramePreviewResult {
  frames: PreviewFrame[];
  expires_in_seconds: number; // TTL of every url above, as of this response (default 3600)
  duration_ms: number;        // server-probed (ffprobe) source duration — the authoritative
                              // timestamp bound: valid extract timestamps are [0, duration_ms)
}

interface ExtractedFrame {
  timestamp_ms: number;
  upload_id: string;   // UUID — same kind of id as POST /v1/storage/upload returns
  media: MediaObject;  // standard media envelope — see thumbnails-gallery-contract.md
}

interface FrameExtractResult {
  frames: ExtractedFrame[];
}

interface FrameJobResponse {
  job_id: string;
  kind: "preview" | "extract";
  status: "queued" | "running" | "completed" | "failed";
  created_at: string;            // ISO 8601
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;         // populated only when status === "failed"; safe to display
  source: FrameJobSource;
  preview?: FramePreviewResult | null;   // present only when kind=preview AND status=completed
  extracted?: FrameExtractResult | null; // present only when kind=extract AND status=completed
}
```

---

## 3. Polling model

Both `POST /preview` and `POST /extract` are **fire-and-forget** — `202` immediately, never blocking on ffmpeg. Poll `GET /v1/frames/jobs/{job_id}` until `status` is terminal:

```ts
const POLL_INTERVAL_MS = 1_000;
const POLL_BUDGET_MS = 7 * 60_000; // see stale-sweep note below

async function pollFrameJob(jobId: string): Promise<FrameJobResponse> {
  const deadline = Date.now() + POLL_BUDGET_MS;
  for (;;) {
    const res = await apiFetch(`/v1/frames/jobs/${jobId}`);
    const job: FrameJobResponse = await res.json();
    if (job.status === "completed" || job.status === "failed") return job;
    if (Date.now() > deadline) throw new Error("frame job polling timed out");
    await sleep(POLL_INTERVAL_MS); // jobs typically finish in low single-digit seconds
  }
}
```

- No SSE/push for job progress — jobs are short-lived (seconds), a cheap DB-only polling GET is sufficient. If long uploaded videos become common this may grow an SSE variant; not today.
- **Bounded polling is safe by construction:** a job stuck in `running` (worker died mid-execution) is failed server-side by a stale sweep after `FRAME_EXTRACT_STALE_RUNNING_SECONDS` (default **300 s**). Worst case, a job reaches a terminal state within ~5 minutes plus one sweep interval — a 7-minute client budget covers it with margin. A job never stays non-terminal forever.
- On `status === "failed"`, `error` is a human-readable message (ffmpeg/ffprobe failure, or an out-of-range timestamp for extract jobs) — safe to display.

---

## 4. URL expiry semantics

**Preview frames** (`FramePreviewResult.frames[].url`) are **presigned R2 URLs generated fresh on every `GET /jobs/{id}` call** — never the same URL twice, never persisted server-side. `expires_in_seconds` (default **3600**) is the validity of *that response's* URLs. If a held `FrameJobResponse` goes stale, just re-poll the same `job_id` — you get fresh URLs with a fresh TTL.

**Do not cache preview URLs beyond the current page session.** Unlike `/v1/content/*` proxy paths (stable, cacheable indefinitely — see `thumbnails-gallery-contract.md`), preview frames live at a non-authenticated R2 prefix (`frame-previews/...`) that expires via an R2 lifecycle rule (default **2 days**). No proxy indirection, by design — preview frames are stateless, no DB rows.

**Extracted frames are different:** `ExtractedFrame.media` is the standard `MediaObject` — `original.url` / `variants[].url` are stable `/v1/content/uploads/{id}` proxy paths, cached indefinitely per the existing contract. Once an extract job completes, its frames behave exactly like anything from `POST /v1/storage/upload`: same download semantics, same deletion (`DELETE /v1/content/{id}`), same retention/expiry (`expires_at` stamped at extraction time, sliding retention applies when used as generation input).

---

## 5. Output formats & server defaults

| What | Value | Source of truth |
|---|---|---|
| Preview frame format | WEBP, longest edge ≤ `FRAME_PREVIEW_MAX_EDGE` (default **512 px**), never upscaled | `ffmpeg.py` scale filter |
| Extracted frame format | **PNG**, full source resolution | `_CODEC_BY_FORMAT` |
| Preview URL TTL | `FRAME_PREVIEW_URL_TTL_SECONDS`, default **3600 s** | `config.py` |
| Preview R2 retention | `FRAME_PREVIEW_RETENTION_DAYS`, default **2 days** (R2 lifecycle rule) | `config.py` |
| Max uploaded video duration | `FRAME_EXTRACT_MAX_VIDEO_SECONDS`, default **300 s** | `config.py` |
| Stale `running` job failed after | `FRAME_EXTRACT_STALE_RUNNING_SECONDS`, default **300 s** | `config.py` |
| Preview strip spacing | `timestamp[i] = round(i * duration_ms / frame_count)`, `i = 0..N-1` — first frame at 0, **no frame at the exact end** | `compute_uniform_timestamps` |

Defaults may differ per environment; treat them as illustrative, not constants to hard-code.

---

## 6. Timestamp bounds

`FramePreviewResult.duration_ms` is the **server-authoritative duration** (ffprobe), and the only value the scrubber may use as its upper bound. Valid extract timestamps are exactly `[0, duration_ms)` — **upper bound exclusive**, so the last selectable millisecond is `duration_ms - 1`.

- **Clamp every user selection to `Math.min(ts, duration_ms - 1)`** before calling `POST /extract`. Done that way, an out-of-range failure is impossible by construction.
- **Never use `videoEl.duration` as the bound.** Container metadata read by the browser can disagree with ffprobe by a frame or two; the server value wins. `videoEl.duration` is fine for cosmetic display, nothing else.
- Any `PreviewFrame.timestamp_ms` is a guaranteed-valid extract input as-is (the strip is computed from the same probe and never touches the exact end).
- The out-of-range job failure path (§3, `error: "Timestamps out of range [0, N)ms: […]"`) still exists as a backstop — reachable only if the client skips the clamp.

---

## 7. Lineage & deletion semantics

- Extracted frames are ordinary uploads. Their origin (source video + timestamp) is recoverable via the `FrameJobResponse` that produced them (`source.type`/`source.id` + `ExtractedFrame.timestamp_ms`) — it is **not** embedded in upload/gallery list responses. If product wants "extracted from video" badges in the gallery grid, that is a follow-up, not in this contract.
- Deleting the source video (`DELETE /v1/content/{id}`) does **not** delete frames already extracted from it — they become ordinary, source-less uploads. No special client handling needed.
- Creating a frame job against an uploaded video **extends the source upload's retention** (same sliding-retention `touch_expiry` behavior as using an upload in generation) — a video won't expire out from under an in-flight extraction flow.

---

## 8. Suggested FE flow (scrubber UX)

1. User picks a video: a gallery output (`source_output_id`) or a fresh upload via `POST /v1/storage/upload` (`source_upload_id` = `UploadResponse.id`).
2. `POST /v1/frames/preview` (default 12 frames) → poll → render the strip; each thumbnail is labeled with its `timestamp_ms`.
3. User selects one or more frames, or scrubs freely within `[0, duration_ms)` from the preview result — clamp to `duration_ms - 1` per §6.
4. `POST /v1/frames/extract` with the chosen `timestamps_ms` → poll → each `ExtractedFrame.upload_id` is immediately valid as `input_image_id` for i2i/i2v, and `media` renders it like any other upload.
5. On `failed`, show `error` verbatim and let the user adjust timestamps or retry.
