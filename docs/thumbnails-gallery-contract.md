# Frontend Contract — Thumbnails & Gallery Refactor

> **Audience:** `gearbox/apex-frontend` (SvelteKit 2 / Svelte 5).
> **Backend source of truth:** the merged thumbnails/`MediaObject` work + the `cookie-for-content-proxy` change (content cookie auth). Pin to the merge commit; regenerate `types.ts` after it lands.
> **Authority:** `gen:api` (OpenAPI) is authoritative for **types**; this document is authoritative for **semantics, the breaking-change map, media-URL auth, and consumption rules**. The cutover is breaking and types change shape.

---

## 0. ✅ Resolved — media URLs authenticate via a first-party cookie

Media URLs in the new contract are **root-relative API paths** to an auth-gated, byte-streaming proxy:

```
/v1/content/outputs/{id}      # generated output (image or video)
/v1/content/uploads/{id}      # uploaded image (or its thumbnail)
```

These are served by `api.<product>` which shares an eTLD+1 with the app (`api.vex.pics` ↔ `vex.pics`, `api.synthara.app` ↔ `synthara.app`), so the backend issues a **first-party `SameSite=Lax` cookie** (`apex_content`, `HttpOnly; Secure; Domain=<product>; Path=/v1/content`) that the browser sends automatically on `<img>`/`<video>` subresource loads. **No per-image token logic, no URL signing, no expiry handling.**

**What the FE must do — exactly two things:**

1. **Send `credentials: "include"` on the four auth calls** so the browser stores/refreshes/clears the cookie: `POST /v1/auth/register`, `/login`, `/refresh`, `/logout`. (These are the only fetches that need it; the cookie is `Path=/v1/content`, so it isn't sent on your other JSON calls — those keep using the `Authorization: Bearer` header as before.) The content cookie is refreshed automatically on every `/refresh`, so it stays alive for the whole session.
2. **Prefix the relative URLs with the product API origin** when rendering: `src = ${API_ORIGIN}${media.original.url}`. That's the entire transform — no token, no helper beyond the origin prefix.

```ts
const API_ORIGIN = "https://api.vex.pics"; // per product/env
const toMediaSrc = (path: string) => `${API_ORIGIN}${path}`;
```

**CORS:** the API now uses a credentialed CORS allowlist (specific product origins, not `*`). Cross-origin JSON `fetch`es that send custom headers (`Authorization`, `Idempotency-Key`, `X-Product-Id`) are allowlisted; if you add a new custom request header, it must be added to the backend `allow_headers` or the preflight will block it.

**Caching:** responses are `Cache-Control: private, immutable` + `ETag`. URLs are stable and never expire — cache freely; there is no presigned-URL refresh to implement (delete any such logic).

**Video caveat (confirm with backend):** the proxy does not implement HTTP **Range** (`206`/`Accept-Ranges`), so `<video>` seeking/scrubbing on `original.url` may be poor for large files. Progressive playback works; flag if scrubbing is required.

**Local dev note:** in dev mode (`debug=True`) the backend sets no `Domain` attribute on the cookie (host-only) and no `Secure` attribute, so the `apex_content` cookie stores correctly over `http://localhost` without mkcert or HTTPS. Production posture is unchanged (`Domain=<product>`, `Secure`).

---

## 1. Core media types

Every image/video-bearing payload now embeds one shape. This replaces all the old flat `*_url` / `thumbnail_url` / `content_type` / `size_bytes` fields.

```ts
type MediaType = "image" | "video";

interface ImageVariant {
  label: string;   // bucket id: "sm" (150) | "md" (512); set is OPEN — may gain "xs"/"xl"
  width: number;   // ACTUAL pixel width (always non-null; backend skips dimensionless legacy rows)
  height: number;  // ACTUAL pixel height (always non-null)
  url: string;     // root-relative proxy path; always WEBP
}

interface MediaOriginal {
  url: string;            // root-relative proxy path to the full asset (image OR mp4)
  width: number | null;
  height: number | null;
  content_type: string;   // e.g. "image/png", "video/mp4"
  size_bytes: number;
}

interface MediaObject {
  media_type: MediaType;        // discriminator: how to render original.url
  original: MediaOriginal;      // the full-fidelity asset
  variants: ImageVariant[];     // preview rasters; ALWAYS WEBP; always present, MAY be []
}
```

**Guarantees the FE may rely on:**
- `variants` is **always present** (required field in the schema — no `?? []` needed) and **sorted ascending by width**. Index 0 is the smallest. Don't re-sort; iterate in order for `srcset`.
- `label` is a stable string but the **set is open** — treat it as data, not an enum. Don't hardcode "exactly sm and md"; iterate `variants`. Today: `sm`=150, `md`=512.
- Variant URLs are **always WEBP**. `original` may be any format (`content_type` tells you).
- `variants` **may be empty** — generation can fail (non-fatal) and legacy content isn't backfilled yet. Always have an `original` fallback.
- `ImageVariant.width`/`height` are **always non-null integers**. The backend serializer skips any legacy row missing dimensions (logs a warning) rather than emitting null. No `.filter(v => v.width)` guard needed in `srcset` construction.
- `original.width`/`height` remain nullable (video originals and legacy content may have unknown dimensions).
- `width`/`height` on variants are **actual pixels** (aspect-ratio preserved, longest edge = the bucket size), so a portrait `sm` is e.g. `84×150`. Use them as real `Nw` descriptors and for layout aspect ratios.
- URLs never expire (stable, `immutable`). **Delete all presigned-URL refresh/expiry logic** — it's obsolete.

---

## 2. Breaking-change map (hard cutover — no aliases)

Grep each removed field; every call site must move to `media`. Old fields are **gone**, not deprecated.

| DTO | Removed | New |
|---|---|---|
| `JobOutputItem` | `url`, `content_type`, `format`, `size_bytes`, `thumbnail_url`, `is_thumbnail` | `{ id, output_index, media }` |
| `UnifiedJobResponse` | top-level `thumbnail_url` | unchanged except `outputs: JobOutputItem[]` now carries `media`; **no presigned URLs anywhere** |
| `GalleryGridItem` | `cover_url`, `video_url`, top-level `media_type` | `cover: MediaObject` (derive media type from `cover.media_type`) |
| `GalleryOutputItem` | `url`, `thumbnail_url`, `content_type`, `media_type`, `format`, `size_bytes` | `{ id, output_index, created_at, media }` |
| `GalleryGroupDetail` | `input_image_url` | `input_media: MediaObject \| null` (+ new `lineage` — see §4); keeps group-level `media_type` |
| `ImageListItem` (uploads) | `content_type`, `size_bytes` | `{ id, filename, created_at, expires_at, media }` |
| `OutputListItem` | `url`, `thumbnail_url`, `content_type`, `size_bytes` | `{ id, job_id, output_index, created_at, expires_at, media }` |
| `UploadResponse` | `storage_key`, `content_type`, `size_bytes` | `{ id, filename, created_at, expires_at, media }` |

Authoritative current field sets (post-fix), for reference:

- `GalleryGridItem`: `job_id, cover, badge, output_count, generation_type, model?, aspect_ratio?, prompt_snippet, created_at`
- `GalleryGroupDetail`: `job_id, badge, input_media?, prompt, negative_prompt?, outputs, media_type, model?, provider, generation_type, aspect_ratio?, token_cost?, created_at, completed_at?, lineage?`
- `GalleryLineage`: `source_type, source_upload_id?, source_job_id?, source_job_name?, source_output_id?`

---

## 3. Consumption rules

**Build `srcset` from `variants`, `src` from `original`:**
```ts
function imgAttrs(m: MediaObject) {
  // variant width/height are always non-null — no .filter() guard needed
  const srcset = m.variants
    .map(v => `${toMediaSrc(v.url)} ${v.width}w`)
    .join(", ");
  return {
    src: toMediaSrc(m.original.url),               // fallback / largest
    srcset: srcset || undefined,                   // omit when no variants
    width: m.original.width ?? undefined,           // intrinsic size → reserve layout box
    height: m.original.height ?? undefined,
  };
}
```
Set `sizes` per layout (grid vs lightbox) so the browser picks the right variant. Don't manually choose a size unless you're not using a real `<img srcset>` (e.g. a CSS `background-image`, where you pick `variants[0]` for grids, `md` for detail, `original` for full view).

**Fallback ladder** when not using native `srcset` (or when variants is empty): preferred variant → any smaller variant → `original.url`. Never assume `sm` or `md` exists.

**Image vs video** — switch on `media_type`:
- `"image"`: render `<img>` with `imgAttrs(media)`.
- `"video"`: `original.url` is the MP4 → `<video src=toMediaSrc(original.url)>`; use a variant (`sm`/`md`) as the **poster** (`poster=toMediaSrc(pickVariant(media))`). The poster comes from `variants`, not a separate field.

**Layout/CLS:** use `original.width/height` (or a variant's) to reserve the aspect-ratio box and avoid layout shift; they're truthful.

**Caching:** URLs are stable + `immutable`. Cache freely; no expiry handling. The cookie carries auth automatically — no per-request work.

---

## 4. Gallery refactor specifics

- **Grid cover is now the job's OWN output**, never the input/source image. Previously, N generations from one source rendered as N near-identical tiles (all showing the input). Now each tile shows that job's own output. Expect distinct covers per job.
- **Lineage moved off the cover.** "This was image-to-image / video-to-video" is conveyed by `GalleryGridItem.badge` (and, in detail, by `input_media` + the structured `lineage` object). If any FE logic inferred lineage from the cover image, switch to `badge`/`lineage`.
- **Grid media type:** there is no top-level `media_type` on `GalleryGridItem` — read `cover.media_type` (video covers carry the MP4 as `original` + poster `variants`).
- **Detail view:** `GalleryGroupDetail.input_media` is the source asset as a full `MediaObject` (image or video — it's `_media`, not `_image`, because v2v inputs are videos) and may be `null` for prompt-only jobs. `lineage` gives structured provenance (`source_type`, source ids/name) for "derived from" UI. `outputs[]` are `GalleryOutputItem` each with their own `media`.
- **`output_count`** counts non-thumbnail outputs only (thumbnails are not separate gallery items).

---

## 5. Edge cases & guarantees

- **Empty `variants`** → render `original`. Happens for (a) generation-time thumbnail failure (non-fatal) and (b) pre-existing content not yet backfilled.
- **Backfill is gradual.** Until the backfill command runs over old content: legacy image outputs may have only `md`; legacy **uploads may have no variants at all**; legacy **video covers may have no `sm`**. Any dimensionless legacy variant row is silently dropped by the serializer (logged server-side), so `variants` may be shorter than expected but will never contain null dims. Your fallback ladder must handle "fewer/zero variants" gracefully and not assume `sm` is present.
- **Open label set.** Future sizes (`xs`, `xl`) may appear with no FE change required if you iterate `variants`. Hardcoding `media.variants.find(v => v.label === "md")!` is brittle — prefer "smallest ≥ target, else largest available, else original."
- **Deletion:** `DELETE /v1/content/{id}` accepts an output **or** upload id (ownership + product scoped). Deleting a parent removes its variants server-side; the FE just drops the item.
- **`ImageAccessResponse` (legacy):** a separate endpoint still returns `presigned_url`/`storage_key`. This is **not** part of the display contract — do not use it for rendering galleries/outputs. Use `MediaObject` everywhere for display.

---

## 6. FE refactor checklist

1. Wait for the backend merge, then `gen:api` to regenerate `types.ts` (shapes change — expect compile errors at every old `*_url`/`thumbnail_url` site, and `variants` becomes non-optional, `ImageVariant.width`/`height` become non-nullable).
2. Implement the **media-URL transform** (§0): send `credentials: "include"` on the four auth calls, and prefix relative `/v1/content/...` URLs with the product API origin in a one-line `toMediaSrc(path)`. No token/refresh logic. Works over `http://localhost` in dev (host-only cookie, no HTTPS needed).
3. Add shared helpers: `imgAttrs(media)`, `pickVariant(media, target)`, `mediaFallbackSrc(media)`. Route **all** image/video rendering through them. Drop `?? []` guards on `variants` and `.filter(v => v.width)` guards in `srcset` — both are now redundant.
4. Replace every removed field per §2. Galleries, job detail, output lists, upload responses, and the upload flow's immediate preview (`UploadResponse.media`).
5. Gallery: render `cover` (grid), `outputs[].media` (detail), `input_media`/`lineage` (provenance UI); read media type from `cover.media_type`; lineage from `badge`/`lineage`.
6. Video: branch on `media_type`; use a variant as `poster`; confirm whether Range/scrubbing is needed (backend caveat in §0).
7. **Delete** all presigned-URL expiry/refresh/re-fetch logic — URLs are now stable and immutable.
8. Verify with: a multi-image job (distinct covers), a video job (poster + playable original), an upload (immediate `media` with variants), and a **not-yet-backfilled** legacy item (empty/partial variants must still render via `original`).

---

## 7. Open decisions / backend dependencies

- **Media-URL auth:** ✅ resolved — first-party `apex_content` cookie (§0). FE work is `credentials: "include"` on the four auth calls + the API-origin prefix.
- **Dev cookie (`localhost`):** ✅ resolved — backend now sets no `Domain` (host-only) and no `Secure` in dev mode, so the cookie stores over `http://localhost` without mkcert.
- **Content-token revocation (FYI):** the content cookie is a stateless token (default TTL is being shortened to ~1h); logout clears it in the current browser, but it can't be force-revoked on other devices before expiry. Read-only, own-content blast radius. No FE action — noted so support/security expectations are calibrated.
- **Video Range support:** confirm whether the proxy needs `206`/`Accept-Ranges` for `<video>` scrubbing, or whether progressive playback is acceptable for MVP.
- **Backfill timing:** the FE must tolerate partial/empty `variants` regardless; confirm when the backfill runs so legacy galleries get full `sm`/`md` coverage.
