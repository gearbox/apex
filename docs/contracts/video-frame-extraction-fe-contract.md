# Frontend Contract — Video Frame Extraction (client-side)

> **Audience:** `gearbox/apex-frontend` (SvelteKit 2 / Svelte 5).
> **Backend source:** `gearbox/apex` `master` `0.50.0`. Copy this file verbatim into the frontend repo.
> **Authority:** `gen:api` (OpenAPI) is authoritative for **types**; this document is authoritative for **semantics** — what the browser decodes, which fields to send, the single lineage error, and caching behaviour. Run `gen:api` after the backend is deployed.

---

## 0. Feature summary

Frame extraction happens **entirely in the browser**. The frontend loads a video it already has access to, seeks to the timestamps the user picks, captures each frame to an image, and uploads it. The backend does no decoding and no job queue:

1. **Decode** the video through a hidden `<video crossorigin="use-credentials">` pointed at its stable content URL (§1).
2. **Capture** each frame (canvas → PNG/JPEG/WebP blob).
3. **Upload** the frame with `POST /v1/storage/upload`, adding two optional multipart fields that record *which video, at what time* (§3). The frame is stored as an ordinary upload with lineage back to the source, immediately usable as an `asset_ref` input for i2i/i2v generation.

Frame extraction is **free** — no token charge, no `Idempotency-Key` header. There is no server-side preview/extract job API: `/v1/frames/*` and the presigned-URL storage routes no longer exist (they return `404`). Use the stable `/v1/content/*` URLs from `media.original.url`.

---

## 1. Loading the video (CORS-mode, credentialed)

Video bytes come from the existing content proxy, via the `media.original.url` of the asset (`/v1/content/outputs/{id}` or `/v1/content/uploads/{id}`). These URLs are stable and carry no presigned token — authentication is the first-party `apex_content` cookie (see `GET /v1/content/*` in `BACKEND_API_REFERENCE.md` §9).

To read pixels back out of a `<video>` without tainting the canvas, the element **must** make a CORS-mode credentialed request:

```ts
const video = document.createElement("video");
video.crossOrigin = "use-credentials"; // set BEFORE src
video.muted = true;
video.preload = "metadata"; // seeking fetches the needed ranges on demand
video.playsInline = true;
video.src = toApiOrigin(media.original.url); // resolve the root-relative path against the API origin
```

`media.original.url` is root-relative (`/v1/content/...`); assigned as-is it would resolve against the **frontend** origin. The frontend resolves it with its protected-content URL validator (`toMediaSrc`) and must never request a URL that fails validation.

The backend answers a credentialed request from a registered product origin with the exact `Access-Control-Allow-Origin: <origin>` and `Access-Control-Allow-Credentials: true`, on `200`, `206` (Range) and `304`. Range requests are supported (single range), which is what lets the browser seek without downloading the whole file.

### Why every content response carries `Vary: Origin`

The rest of the app plays the same URLs **without** `crossorigin` — a no-cors request that carries no `Origin` header. Content responses are `Cache-Control: private, max-age=<ttl>, immutable`. Without `Vary: Origin`, the browser would cache the no-cors response, reuse it for the extractor's CORS-mode request, find no `Access-Control-Allow-Origin` on it, fail the CORS check — and, because the entry is `immutable`, **never revalidate**. So a user who watches a video in the library and then opens "Extract frames" would get a permanently broken extractor.

The backend therefore sends `Vary: Origin` on **every** `/v1/content/*` response — `200`, `206`, `304`, `404`, `416`, `502`, with or without a request `Origin`. Consequences for the frontend:

- **Do not** work around this with cache-busting query strings or `cache: "reload"`; the stable URL is correct.
- **Do not** switch all `<video>`/`<img>` playback to CORS mode — only the extractor element needs it.
- Expect the first extractor load of an already-watched video to be a cache miss for that variant (the CORS-mode entry is stored separately); this is intended.

---

## 2. Video duration: `MediaOriginal.duration_ms`

`MediaOriginal` now carries the video's duration, probed at ingest:

```ts
interface MediaOriginal {
  url: string;
  width?: number | null;
  height?: number | null;
  content_type: string;
  size_bytes: number;
  duration_ms: number | null; // NEW. Video duration in ms; null for images and legacy rows
}
```

- Present on **every** surface that serializes a media original: the upload response, library list/detail, group detail (outputs and `source_media`), lineage graph nodes, and job outputs.
- For a video ingested through the current pipeline it is a positive integer. It is `null` for images, and may be `null` for a legacy video row that predates ingest probing.
- `duration_ms` is the **lineage upper bound**: a frame upload is rejected when `source_timestamp_ms > duration_ms` or when the source's `duration_ms` is `null`.
- The decoded `video.duration` may differ by a frame or two. The UI may use it for the timeline, but every uploaded timestamp must be clamped to `[0, duration_ms]`.
- If `duration_ms` is `null`, don't open the extractor; show "Frame extraction isn't available for this video".

User-uploaded videos are rejected at ingest above `MEDIA_VIDEO_MAX_DURATION_SECONDS` (default **300 s**) with `400 validation_error`; the message is safe to show verbatim. Generated outputs are not subject to this cap.

---

## 3. Uploading a captured frame

`POST /v1/storage/upload` (multipart) accepts two **optional** extra fields:

| Field | Type | Rule |
|---|---|---|
| `data` | file | the frame image (PNG/JPEG/WebP; unchanged from a normal image upload, 20 MB cap) |
| `source_asset_ref` | string | `upload:<uuid>` or `output:<uuid>` — the **video** the frame came from (the video's `media.asset_ref`) |
| `source_timestamp_ms` | string | plain decimal integer, `0 ≤ ts ≤ duration_ms` of the source |

- **Both absent** → an ordinary upload; behaviour is exactly as before.
- **Exactly one present** → `400 invalid_frame_lineage`.
- **Both present** → a *frame upload*, validated as below.

Send `source_timestamp_ms` as a bare integer string (`"2500"`): no sign, no decimal point, no exponent, no whitespace, no leading `+`, at most 10 digits and at most `2147483647`. `"1e3"`, `" 12"`, `"-0"`, `"12.0"` and `""` are all rejected. Round the captured time to whole milliseconds client-side. `0` and `duration_ms` itself are both accepted; in practice clamp to `Math.max(0, duration_ms - 1)` because a seek to the exact end frequently decodes nothing.

The response is the normal `UploadResponse` (`id`, `filename`, `created_at`, `expires_at`, `media`). The frame is an image with sm/md variants like any upload, and a new row in the library.

### The single error: `invalid_frame_lineage`

```json
{ "error": "invalid_frame_lineage", "message": "Frame source is not available", "status_code": 400 }
```

Every lineage failure returns this **one** status, code and message, deliberately — the response never reveals *why* (it would be an existence oracle for other users' assets). It covers: malformed or partial fields; a source that doesn't exist, belongs to another user or another product, or is a thumbnail; a source that is not a video; a source with unknown duration; a timestamp past `duration_ms`; or an upload that is itself a video file.

Treat it as "this frame couldn't be linked to that video": show a generic failure, and do not retry the same request. A frame-less retry (plain upload without the two fields) is always possible if the user still wants the image.

### What the backend does for a frame upload

- The frame bytes are untrusted client bytes and go through the normal **image-upload** pipeline (decoded, metadata-stripped, normalized) — the client's `Content-Type` is only an early filter.
- Lineage is stored **with** the row. It appears in the source's library lineage (`GET /v1/library/assets/{asset_ref}/lineage` descendants, with `source_timestamp_ms`).
- Capturing a frame from an **upload** source slides that upload's retention window (and its thumbnails'), same as using it as generation input, so a video can't expire under an in-flight extraction session. An **output** source is not touched.

---

## 4. Suggested flow

1. The user opens a video from the library (an asset with `media.media_type === "video"` and `media.original.duration_ms != null`).
2. Create the hidden `<video crossorigin="use-credentials">` (§1), wait for `loadedmetadata`/`seeked`.
3. The user scrubs within `[0, duration_ms - 1]`; for each selected time: `video.currentTime = t / 1000`, await `seeked`, draw to a canvas at the **decoded** dimensions (`video.videoWidth/Height`), `canvas.toBlob(…, "image/png")`.
4. Upload each blob with the lineage fields (§3). On success, the new `UploadResponse.media.asset_ref` is immediately a valid generation input.
5. On `invalid_frame_lineage`, show a generic "couldn't save this frame" message.

Canvas dimensions: use the decoded `videoWidth`/`videoHeight`, **not** `media.original.width/height`. Stored dimensions are the coded size and do not account for display-matrix rotation, so a portrait clip recorded rotated reports landscape there. (A rotation-corrected `width`/`height` is a known follow-up.)

---

## 5. HDR behaviour (parity with the old server extractor)

No tone mapping happens anywhere. A frame captured from an HDR video is exactly what the browser's canvas produces; the previous server-side extractor only scaled and was equally un-tone-mapped, so this is behaviour parity, not a regression. The backend records whether each stored video is HDR/rotated/non-H.264 in server logs (`media.video_profile`, informational only) so a future conversion feature can be justified with data — it is not exposed in the API.

---

## 6. Videos the browser can't decode — the unsupported state

There is no browser-compatibility derivative: the server stores videos as uploaded (remuxed, never transcoded). A video in a codec/profile the user's browser cannot decode (for example HEVC on a browser without a HEVC decoder) will make the hidden `<video>` fire `error` (`MediaError.code === MEDIA_ERR_SRC_NOT_SUPPORTED` / `MEDIA_ERR_DECODE`) or never reach `loadedmetadata`.

Show an explicit **unsupported state** for it rather than an endless spinner: "Your browser can't read this video, so frames can't be extracted here." Generated outputs are expected to be browser-safe by construction (H.264/yuv420p/MP4), so this state is chiefly reachable for user-uploaded videos.

Treat any of these as the unsupported state:

1. The element fires `error` (`MEDIA_ERR_SRC_NOT_SUPPORTED` / `MEDIA_ERR_DECODE`).
2. `loadedmetadata` never fires.
3. `loadedmetadata` fires but `video.videoWidth === 0 || video.videoHeight === 0`. Some browsers decode only the audio track of a video they can't decode.

A `SecurityError` when reading canvas pixels (`toBlob` / `getImageData`) means the CORS-mode load failed (§1). It is not an auth problem, so don't prompt a re-login.

### "Report this video"

Offer a **Report this video** action in the unsupported state. It opens the frontend's existing feedback dialog with `assetRef = media.asset_ref` and `initialCategory = "bug"`. The **user** writes the message (10–4000 code points), and the dialog submits `POST /v1/feedback` with `asset_ref`. The backend stores the reference, not a copy.

Feedback reports are user-authored by design: don't POST automatically or with canned text, which would flood triage with content-free reports.

See `feedback-contract.md` for validation and the `404 asset_not_found` case.

---

## 7. Everything else is unchanged

- Image upload types, the 20 MB cap, video upload types (`video/mp4`, `video/webm`, `video/quicktime`) and their `400 validation_error` rejections are unchanged.
- Extracted frames are ordinary uploads: same download semantics, same deletion (`DELETE /v1/content/{id}`), same retention. Deleting the source video does **not** delete frames already extracted from it — they become ordinary, source-less uploads.
- Frame extraction needs no billing, no `Idempotency-Key`, and no polling.
