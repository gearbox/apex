# Gallery Endpoint Design — `/v1/gallery` (v2)

> **Status:** Design complete — ready for implementation prompt  
> **Author:** Claude × Miša  
> **Date:** 2026-03-24  
> **Scope:** Apex backend — new gallery controller, service, repository, content proxy, schema migration, enum additions

---

## 1. Problem Statement

The current API exposes generation history through two separate surfaces:

| Endpoint | Returns | Pagination | Grouping |
|---|---|---|---|
| `GET /v1/jobs` | Flat job list with outputs inlined | Cursor + offset | None — one item per job |
| `GET /v1/storage/outputs` | Flat output list (no job context) | Cursor + offset | None — one item per output |
| `GET /v1/storage/gallery` | **Dead TODO** — schemas exist, no route | — | — |

**What the frontend actually needs** is a two-level gallery:

1. **Grid view** — an infinite-scroll list of *generation groups* (jobs), each showing a cover image/video, a badge (`image` or `prompt`), and minimal metadata.
2. **Detail view** — opening a group reveals all outputs, full generation parameters, and remix lineage.

Additionally:
- There is no lineage tracking — remix chains are not captured.
- Presigned URLs leak R2 infrastructure details to the client — the new gallery should use a streaming proxy with stable, auth-gated URLs.

---

## 2. Key Design Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | `GenerationJob` **is** the generation group | One prompt submission = one job = N outputs. No new grouping table needed. |
| D2 | Reuse output IDs as generation input (no re-upload) | Avoids R2 duplication. Add `source_output_id` on `GenerationJob` referencing `generation_outputs`. |
| D3 | Add `OutputMediaType` enum (`image` \| `video`) | Strict-typed filter for gallery grid + detail. Derived from `content_type` / `generation_type`. |
| D4 | Add `GalleryBadge` enum (`image` \| `prompt`) | Describes the *input* source, not the output format. Separate from `OutputMediaType`. |
| D5 | Single-level lineage only | "Remixed from Job Y" — no recursive chain traversal. |
| D6 | Gallery is private — auth required, user-scoped | No public sharing. All queries filter by `user_id` + `product_id`. |
| D7 | Video covers: thumbnail-first, then autoplay | Grid returns both `cover_url` (thumbnail) and `video_url` (full video). Frontend loads thumbnails fast, replaces with autoplay. |
| D8 | **Streaming proxy — no presigned URLs exposed** | API acts as auth-gated proxy. Client gets stable paths. Configurable TTL (default 10800s). Designed for future CF Workers + Signed Cookies migration. |
| D9 | No `COUNT(*)` for gallery pagination | Infinite scroll uses `has_more` + `next_cursor` only. No expensive total count. |
| D10 | Delete dead `src/api/routes/gallery.py` | Replace entirely with the new `/v1/gallery` controller. |
| D11 | Max page size: 25 | Lighter payload for image-heavy grid (25 cover URLs per request). |

---

## 3. Content Delivery Architecture

### 3.1 Current State: Presigned URLs

```
Client → GET /v1/storage/outputs/{id} → { presigned_url: "https://{account}.r2.cloudflarestorage.com/..." }
Client → GET presigned_url (direct to R2)
```

**Problems:** Exposes R2 infrastructure, URL is temporary, client must handle URL rotation.

### 3.2 New: Streaming Proxy (Gallery)

```
Client → GET /v1/content/{output_id} → 200 OK + streamed bytes (Content-Type, Content-Length)
                                        ↑ auth-gated, stable URL
                                        ↓ server-side: presign → stream from R2
```

The gallery endpoints return stable *paths* (not presigned URLs). The client then fetches content through a new `/v1/content/` proxy endpoint that:

1. Validates auth (JWT).
2. Verifies ownership (`user_id` + `product_id`).
3. Generates a presigned R2 URL server-side.
4. Streams the R2 response to the client (chunked transfer).
5. Sets `Cache-Control` headers for browser caching (configurable TTL).

### 3.3 Content URL Strategy

Gallery schemas return **path-based references** instead of presigned URLs:

```python
class GalleryGridItem(msgspec.Struct, kw_only=True):
    cover_url: str  # "/v1/content/outputs/{output_id}"
    video_url: str | None  # "/v1/content/outputs/{output_id}" (for video)
    # ...
```

**Why paths instead of full URLs?** The frontend already knows the API base URL. Paths are:
- Stable (don't expire).
- Auth-gated (protected by the same JWT as all other endpoints).
- CDN-ready (when migrating to CF Workers, the Worker URL becomes the base).

### 3.4 Future Migration Path: CF Workers + Signed Cookies

The proxy architecture is designed so that switching to Cloudflare Workers requires only:

1. Deploy a CF Worker at `cdn.vex-domain.com` / `cdn.synthara-domain.com`.
2. Worker validates session cookie or JWT → proxies/redirects to short-lived presigned URL.
3. Cloudflare caches the response at the edge.
4. Gallery service changes URL prefix from `/v1/content/` to `https://cdn.{product}/content/`.

**No API schema changes needed** — only the URL prefix changes.

### 3.5 Content Proxy Endpoint

```
GET /v1/content/outputs/{output_id}
GET /v1/content/uploads/{image_id}

Auth:     Required (auth_guard + ownership)
Response: Streamed bytes with Content-Type, Content-Length, Cache-Control
Errors:   404 (not found / not owned), 502 (R2 fetch failed)

Headers returned:
  Content-Type: image/jpeg | image/png | image/webp | video/mp4
  Content-Length: <size_bytes from DB>
  Cache-Control: private, max-age=<content_url_ttl>, immutable
  ETag: "<output_id>"
  X-Content-Id: <output_id>
```

**Configurable TTL** — add to `Settings`:

```python
# src/core/config.py — Settings

content_url_ttl: int = Field(
    default=10800,  # 3 hours
    ge=60,
    le=86400,
    description="Cache-Control max-age for content proxy responses (seconds).",
)
```

### 3.6 `ContentProxyService`

```python
# src/api/services/content_proxy.py


class ContentProxyService:
    """Auth-gated streaming proxy for R2 content.

    Designed as a thin layer that:
    1. Looks up the DB record (ownership + product check).
    2. Presigns the R2 URL (server-side only, never exposed).
    3. Streams the response using httpx async streaming.

    This abstraction is transport-agnostic — when migrating to
    CF Workers, this service is replaced by a redirect to the
    Worker URL, and the Worker handles the R2 presigning.
    """

    def __init__(
        self,
        storage: R2StorageService,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._ttl = settings.content_url_ttl

    async def stream_output(
        self,
        output_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        session: AsyncSession,
    ) -> StreamResponse:
        """Stream an output file to the client.

        Returns:
            StreamResponse with async byte iterator, content_type, size, headers.

        Raises:
            ContentNotFoundError: Output not found or not owned.
            ContentFetchError: R2 streaming failed.
        """
        ...

    async def stream_upload(
        self,
        image_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        session: AsyncSession,
    ) -> StreamResponse:
        """Stream an uploaded image to the client."""
        ...
```

**Implementation note:** Litestar supports `Stream` responses:

```python
from litestar.response import Stream

@get("/v1/content/outputs/{output_id:uuid}")
async def proxy_output(...) -> Stream:
    result = await content_proxy.stream_output(output_id, ...)
    return Stream(
        result.byte_iterator,
        media_type=result.content_type,
        headers={
            "Content-Length": str(result.size_bytes),
            "Cache-Control": f"private, max-age={result.cache_ttl}, immutable",
            "ETag": f'"{output_id}"',
        },
    )
```

For the R2 side, `aioboto3`'s `get_object` returns a `StreamingBody` — we wrap it in an async generator:

```python
async def _stream_from_r2(self, storage_key: str) -> AsyncIterator[bytes]:
    """Stream R2 object in chunks without buffering the full file."""
    async with self._storage._get_client() as client:
        response = await client.get_object(
            Bucket=self._storage._settings.bucket_name,
            Key=storage_key,
        )
        async for chunk in response["Body"].iter_chunks(chunk_size=65536):
            yield chunk
```

---

## 4. Data Model Changes

### 4.1 New Enums

```python
# src/core/enums.py


class OutputMediaType(str, Enum):
    """Media type classification for gallery filtering."""

    IMAGE = "image"
    VIDEO = "video"


class GalleryBadge(str, Enum):
    """Badge type for gallery grid items — describes the input source."""

    IMAGE = "image"  # Generation used an image/video as input (i2i, i2v, flf2v, v2v)
    PROMPT = "prompt"  # Generation was text-only (t2i, t2v)


class GallerySourceType(str, Enum):
    """Type of input source for lineage display."""

    UPLOAD = "upload"  # User uploaded an image directly
    GENERATION = "generation"  # Used a previous generation's output
```

**`OutputMediaType` derivation** — no new DB column. Determined at query/service time:

```python
# Job-level: from GenerationType
@staticmethod
def media_type_from_generation_type(gt: GenerationType) -> OutputMediaType:
    return OutputMediaType.VIDEO if gt.is_video else OutputMediaType.IMAGE


# Output-level: from content_type (more precise — a video job's thumbnail is still IMAGE)
@staticmethod
def media_type_from_content_type(content_type: str) -> OutputMediaType:
    return OutputMediaType.VIDEO if content_type.startswith("video/") else OutputMediaType.IMAGE
```

### 4.2 New Columns on `GenerationJob`

```python
# src/db/models/storage.py — GenerationJob

# --- Lineage: remix tracking ---
source_job_id: Mapped[UUID | None] = mapped_column(
    PG_UUID(as_uuid=True),
    ForeignKey("generation_jobs.id", ondelete="SET NULL"),
    nullable=True,
    index=True,
)
"""Parent job this generation was remixed from. NULL for original generations."""

source_output_id: Mapped[UUID | None] = mapped_column(
    PG_UUID(as_uuid=True),
    ForeignKey("generation_outputs.id", ondelete="SET NULL"),
    nullable=True,
)
"""Specific output from the parent job used as input. NULL for uploads or originals."""

# --- Direct input reference ---
input_image_id: Mapped[UUID | None] = mapped_column(
    PG_UUID(as_uuid=True),
    ForeignKey("user_images.id", ondelete="SET NULL"),
    nullable=True,
)
"""Uploaded image used as input. Set when source is a UserImage upload (not a remix)."""
```

**Why both `source_job_id` AND `source_output_id`?**

- `source_job_id`: Powers "Remixed from [job name]" display — single FK join.
- `source_output_id`: Points to the *specific* output used as input — needed for cover resolution and "open source" navigation.

**Why `input_image_id` on `GenerationJob` (moved from `GenerationOutput`)?**

The input image is a property of the *job*, not the output. All outputs of a batch share the same input. Moving it to `GenerationJob` is semantically correct. The existing `GenerationOutput.input_image_id` column stays for backward compat but new writes go to the job level.

### 4.3 New Relationships on `GenerationJob`

```python
# Self-referential for lineage
source_job: Mapped[GenerationJob | None] = relationship(
    "GenerationJob",
    remote_side="GenerationJob.id",
    foreign_keys=[source_job_id],
    uselist=False,
)

source_output: Mapped[GenerationOutput | None] = relationship(
    "GenerationOutput",
    foreign_keys=[source_output_id],
    uselist=False,
)

input_image: Mapped[UserImage | None] = relationship(
    "UserImage",
    foreign_keys=[input_image_id],
    uselist=False,
)
```

### 4.4 New Composite Index

```python
# Gallery grid query optimization
Index(
    "ix_generation_jobs_gallery",
    "user_id",
    "product_id",
    "status",
    "created_at",
)
```

### 4.5 Migration

```python
# alembic/versions/NNN_add_gallery_lineage_columns.py


def upgrade() -> None:
    # New columns
    op.add_column(
        "generation_jobs",
        sa.Column(
            "source_job_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.add_column(
        "generation_jobs",
        sa.Column(
            "source_output_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.add_column(
        "generation_jobs",
        sa.Column(
            "input_image_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )

    # Foreign keys
    op.create_foreign_key(
        "fk_generation_jobs_source_job",
        "generation_jobs",
        "generation_jobs",
        ["source_job_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_generation_jobs_source_output",
        "generation_jobs",
        "generation_outputs",
        ["source_output_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_generation_jobs_input_image",
        "generation_jobs",
        "user_images",
        ["input_image_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Indexes
    op.create_index(
        "ix_generation_jobs_source_job",
        "generation_jobs",
        ["source_job_id"],
    )
    op.create_index(
        "ix_generation_jobs_gallery",
        "generation_jobs",
        ["user_id", "product_id", "status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_generation_jobs_gallery")
    op.drop_index("ix_generation_jobs_source_job")
    op.drop_constraint("fk_generation_jobs_input_image", "generation_jobs")
    op.drop_constraint("fk_generation_jobs_source_output", "generation_jobs")
    op.drop_constraint("fk_generation_jobs_source_job", "generation_jobs")
    op.drop_column("generation_jobs", "input_image_id")
    op.drop_column("generation_jobs", "source_output_id")
    op.drop_column("generation_jobs", "source_job_id")
```

---

## 5. API Design

### 5.1 Gallery Grid — `GET /v1/gallery`

**Purpose:** Infinite-scroll grid of generation groups (completed jobs).

```
Path:     GET /v1/gallery
Auth:     Required (auth_guard)
Scoping:  user_id + product_id

Query Parameters:
  limit?           int     Page size (1–25, default 20)
  cursor?          str     Opaque cursor from previous response
  media_type?      str     Filter: "image" | "video" (OutputMediaType)
  generation_type? str     Filter: "t2i" | "i2i" | "t2v" | "i2v" | "v2v" | "flf2v"
  model?           str     Filter by model identifier

Response: GalleryPage (custom — no total count)
```

#### `GalleryPage` Schema (no `total`)

```python
class GalleryPage(msgspec.Struct, kw_only=True):
    """Cursor-paginated gallery response.

    Unlike PaginatedResponse, this does NOT include a total count.
    Infinite scroll only needs has_more + next_cursor.
    """

    items: list[GalleryGridItem]
    limit: int
    has_more: bool
    next_cursor: str | None = None
```

#### `GalleryGridItem` Schema

```python
class GalleryGridItem(msgspec.Struct, kw_only=True):
    """Single cell in the gallery grid — represents one generation group (job)."""

    job_id: UUID

    cover_url: str
    """Content proxy path for the grid cover.
    
    Resolution logic:
    - Image-input types (i2i, i2v, flf2v, v2v):
      → source output's image OR uploaded input image.
    - Text-only types (t2i): → last generated output.
    - Text-only video (t2v): → video thumbnail (poster frame).
    
    Format: "/v1/content/outputs/{id}" or "/v1/content/uploads/{id}"
    """

    video_url: str | None = None
    """Content proxy path for the full video (autoplay).
    Present only for video generation types.
    Frontend uses cover_url (thumbnail) for fast grid load,
    then replaces with autoplaying video_url.
    
    Format: "/v1/content/outputs/{id}"
    """

    badge: GalleryBadge
    """'image' if input-driven (i2i, i2v, flf2v, v2v), 'prompt' if text-only (t2i, t2v)."""

    media_type: OutputMediaType
    """Output media type: 'image' or 'video'. Derived from generation_type.is_video."""

    output_count: int
    """Number of non-thumbnail outputs in this group."""

    generation_type: GenerationType

    model: str | None = None

    prompt_snippet: str
    """First 100 characters of the prompt for preview/search."""

    created_at: datetime
```

### 5.2 Gallery Group Detail — `GET /v1/gallery/{job_id}`

```
Path:     GET /v1/gallery/{job_id}
Auth:     Required (auth_guard + ownership)
Scoping:  user_id + product_id

Response: GalleryGroupDetail
Errors:   404 (not found / not owned / not completed / hidden)
```

#### `GalleryGroupDetail` Schema

```python
class GalleryGroupDetail(msgspec.Struct, kw_only=True):
    """Full detail view of a generation group."""

    job_id: UUID

    # --- Header ---
    badge: GalleryBadge
    input_image_url: str | None = None
    """Content proxy path for the input image/output. Present when badge == 'image'.
    Format: "/v1/content/outputs/{id}" or "/v1/content/uploads/{id}"
    """

    prompt: str
    negative_prompt: str | None = None

    # --- Outputs grid ---
    outputs: list[GalleryOutputItem]
    """All non-thumbnail outputs, ordered by output_index."""

    # --- Metadata ---
    media_type: OutputMediaType
    model: str | None = None
    provider: str
    generation_type: GenerationType
    aspect_ratio: str | None = None
    token_cost: int | None = None
    created_at: datetime
    completed_at: datetime | None = None

    # --- Lineage ---
    lineage: GalleryLineage | None = None


class GalleryOutputItem(msgspec.Struct, kw_only=True):
    """Single output within a gallery group detail view."""

    id: UUID
    url: str
    """Content proxy path: "/v1/content/outputs/{id}" """

    thumbnail_url: str | None = None
    """Content proxy path for video poster frame (if applicable)."""

    content_type: str
    media_type: OutputMediaType
    format: str
    size_bytes: int
    output_index: int
    created_at: datetime


class GalleryLineage(msgspec.Struct, kw_only=True):
    """Single-level remix lineage."""

    source_type: GallerySourceType

    source_upload_id: UUID | None = None
    """If source was a direct upload."""

    source_job_id: UUID | None = None
    """If source was a previous generation's output."""

    source_job_name: str | None = None
    """Human-readable name of the source job."""

    source_output_id: UUID | None = None
    """Specific output used as input."""
```

---

## 6. Cover Resolution Logic

Priority chain per generation type:

```
Image-input jobs (i2i, i2v, flf2v):
  1. source_output_id  → "/v1/content/outputs/{source_output_id}"
  2. input_image_id    → "/v1/content/uploads/{input_image_id}"
  3. Fallback: last output of this job

Video-input jobs (v2v):
  1. source_output_id  → "/v1/content/outputs/{source_output_id}"
  2. Fallback: video thumbnail of this job

Text-only image (t2i):
  1. Last generated output (highest output_index, non-thumbnail)

Text-only video (t2v):
  1. cover_url = video thumbnail (is_thumbnail=True)
  2. video_url = the actual video output
```

---

## 7. Query Design

### 7.1 Gallery Grid Query (MVP: 2-query approach)

**Query 1: Paginated jobs**

```python
query = select(
    GenerationJob.id,
    GenerationJob.generation_type,
    GenerationJob.model,
    GenerationJob.prompt,
    GenerationJob.created_at,
    GenerationJob.source_job_id,
    GenerationJob.source_output_id,
    GenerationJob.input_image_id,
).where(
    GenerationJob.user_id == user_id,
    GenerationJob.product_id == product_id,
    GenerationJob.status == JobStatus.COMPLETED.value,
    GenerationJob.error_message.is_distinct_from("__hidden__"),
)

# Apply media_type filter
if media_type == OutputMediaType.VIDEO:
    query = query.where(GenerationJob.generation_type.in_(VIDEO_TYPES))
elif media_type == OutputMediaType.IMAGE:
    query = query.where(GenerationJob.generation_type.in_(IMAGE_TYPES))

# Apply generation_type filter
if generation_type is not None:
    query = query.where(GenerationJob.generation_type == generation_type.value)

# Apply model filter
if model is not None:
    query = query.where(GenerationJob.model == model)

# Keyset cursor
if cursor_ts is not None and cursor_id is not None:
    query = query.where(
        sa.tuple_(GenerationJob.created_at, GenerationJob.id) < sa.tuple_(cursor_ts, cursor_id)
    )

# Fetch limit + 1 to determine has_more
query = query.order_by(GenerationJob.created_at.desc(), GenerationJob.id.desc()).limit(limit + 1)
```

**Query 2: Batch cover outputs for returned job IDs**

```python
# Fetch cover candidates: for each job, get thumbnail + last real output
cover_query = (
    select(
        GenerationOutput.job_id,
        GenerationOutput.id,
        GenerationOutput.storage_key,
        GenerationOutput.content_type,
        GenerationOutput.is_thumbnail,
        GenerationOutput.output_index,
    )
    .where(
        GenerationOutput.job_id.in_(job_ids),
    )
    .order_by(
        GenerationOutput.job_id,
        GenerationOutput.is_thumbnail.desc(),
        GenerationOutput.output_index.desc(),
    )
)
```

The service groups results by `job_id` and picks the best cover per the resolution logic in Section 6.

**Output count** — use a single aggregate:

```python
count_query = (
    select(
        GenerationOutput.job_id,
        func.count().label("output_count"),
    )
    .where(
        GenerationOutput.job_id.in_(job_ids),
        GenerationOutput.is_thumbnail == False,  # noqa: E712
    )
    .group_by(GenerationOutput.job_id)
)
```

### 7.2 `has_more` Without `COUNT(*)`

Fetch `limit + 1` rows. If `len(results) > limit`, there are more pages — pop the extra row and set `has_more = True`.

```python
jobs = await repo.list_gallery_jobs(...)
has_more = len(jobs) > limit
if has_more:
    jobs = jobs[:limit]

next_cursor = None
if has_more and jobs:
    last = jobs[-1]
    next_cursor = encode_cursor(last.created_at, last.id)
```

### 7.3 Gallery Detail Query

Standard eager-load:

```python
query = (
    select(GenerationJob)
    .where(
        GenerationJob.id == job_id,
        GenerationJob.user_id == user_id,
        GenerationJob.product_id == product_id,
        GenerationJob.status == JobStatus.COMPLETED.value,
        GenerationJob.error_message.is_distinct_from("__hidden__"),
    )
    .options(
        selectinload(GenerationJob.outputs),
        selectinload(GenerationJob.source_job),  # for lineage name
        selectinload(GenerationJob.input_image),  # for upload cover
    )
)
```

---

## 8. `media_type` Filter Semantics

**Grid level (job-based):** Filters by whether the generation produces image or video output:

```python
VIDEO_TYPES = [gt.value for gt in GenerationType if gt.is_video]
# ["t2v", "i2v", "v2v", "flf2v"]

IMAGE_TYPES = [gt.value for gt in GenerationType if not gt.is_video]
# ["t2i", "i2i"]
```

**Detail level (per-output):** Derived from `content_type`:

```python
media_type = (
    OutputMediaType.VIDEO if output.content_type.startswith("video/") else OutputMediaType.IMAGE
)
```

This distinction matters: a video job's thumbnail output has `media_type=IMAGE` at the output level even though the parent job has `media_type=VIDEO` at the grid level.

---

## 9. Layered Architecture

```
┌──────────────────────────────────────────────────────────┐
│  GalleryController (src/api/routes/gallery.py)           │
│  GET /v1/gallery         → list_gallery()                │
│  GET /v1/gallery/{id}    → get_gallery_detail()          │
├──────────────────────────────────────────────────────────┤
│  ContentProxyController (src/api/routes/content.py)      │
│  GET /v1/content/outputs/{id}  → proxy_output()          │
│  GET /v1/content/uploads/{id}  → proxy_upload()          │
└────────────────┬──────────────────┬──────────────────────┘
                 │                  │
┌────────────────▼──────────┐ ┌────▼──────────────────────┐
│  GalleryService           │ │  ContentProxyService      │
│  - Cover resolution       │ │  - Auth + ownership check │
│  - Badge derivation       │ │  - R2 stream proxy        │
│  - Lineage assembly       │ │  - Cache-Control headers  │
│  - Content URL building   │ │  - Configurable TTL       │
└────────────────┬──────────┘ └────┬──────────────────────┘
                 │                  │
┌────────────────▼──────────┐      │
│  GalleryRepository        │      │ uses R2StorageService
│  - list_gallery_jobs()    │      │
│  - get_gallery_job()      │      │
│  - batch_cover_outputs()  │      │
│  - batch_output_counts()  │      │
└───────────────────────────┘      │
                                   │
                              ┌────▼──────────────────────┐
                              │  R2StorageService          │
                              │  (existing — no changes)   │
                              └────────────────────────────┘
```

**DI registration** in `src/api/dependencies/common.py`:

```python
"gallery_service": Provide(get_gallery_service),
"content_proxy": Provide(get_content_proxy),
```

---

## 10. Generation Flow Changes

### 10.1 `UnifiedGenerationRequest` — New Field

```python
class UnifiedGenerationRequest(msgspec.Struct, ...):
    # ... existing fields ...
    
    source_output_id: UUID | None = None
    """ID of a GenerationOutput to use as input (remix — alternative to input_image_id).
    Mutually exclusive with input_image_id."""
```

**Validation:** `input_image_id` XOR `source_output_id`. If both provided → 400.

### 10.2 Generation Service Changes

When `source_output_id` is provided:

1. Look up `GenerationOutput` by ID + `user_id` ownership check.
2. Resolve its `storage_key` → presign for provider (server-side only).
3. Set `GenerationJob.source_output_id = source_output_id`.
4. Set `GenerationJob.source_job_id = output.job_id`.
5. Do NOT set `GenerationJob.input_image_id`.

When `input_image_id` is provided (existing flow):

1. Existing flow unchanged.
2. Set `GenerationJob.input_image_id = input_image_id`.
3. `source_job_id` and `source_output_id` remain NULL.

---

## 11. Endpoint Consolidation

| Existing Endpoint | Action | Rationale |
|---|---|---|
| `src/api/routes/gallery.py` (dead TODO) | **Delete** | Replaced by new `/v1/gallery` |
| `GET /v1/jobs` | **Keep** | Status polling for in-progress work |
| `GET /v1/jobs/{id}` | **Keep** | Poll-on-read for running jobs |
| `GET /v1/storage/outputs` | **Keep** | Low-level programmatic access |
| `GET /v1/storage/outputs/{id}` | **Keep (but deprioritize)** | Returns presigned URL — gallery uses proxy instead |
| `GET /v1/storage/outputs/{id}/download` | **Candidate for proxy migration** | Already does full download — proxy is similar |
| `GET /v1/storage/jobs/{job_id}/outputs` | **Deprecate** | Subsumed by `GET /v1/gallery/{job_id}` |
| `GET /v1/storage/stats` | **Keep** | Orthogonal concern |

---

## 12. File Manifest

### New Files

| File | Purpose |
|---|---|
| `src/api/routes/gallery.py` | **Rewrite** — `GalleryController` (`GET /v1/gallery`, `GET /v1/gallery/{job_id}`) |
| `src/api/routes/content.py` | `ContentProxyController` (`GET /v1/content/outputs/{id}`, `GET /v1/content/uploads/{id}`) |
| `src/api/services/gallery.py` | `GalleryService` — cover logic, badge, lineage, URL building |
| `src/api/services/content_proxy.py` | `ContentProxyService` — auth-gated R2 streaming proxy |
| `src/db/repositories/gallery.py` | `GalleryRepository` — gallery-specific queries |
| `src/api/schemas/gallery.py` | All gallery DTOs |
| `alembic/versions/NNN_add_gallery_lineage.py` | Migration: new columns + indexes |
| `tests/unit/test_gallery_schemas.py` | Schema validation + msgspec round-trips |
| `tests/unit/test_gallery_service.py` | Service logic: cover resolution, badge, lineage |
| `tests/unit/test_content_proxy.py` | Content proxy: streaming, caching, auth |
| `tests/integration/test_gallery_repository.py` | Repository queries against real DB |

### Modified Files

| File | Changes |
|---|---|
| `src/core/enums.py` | Add `OutputMediaType`, `GalleryBadge`, `GallerySourceType` |
| `src/core/config.py` | Add `content_url_ttl` setting |
| `src/db/models/storage.py` | Add `source_job_id`, `source_output_id`, `input_image_id` to `GenerationJob` + relationships + index |
| `src/api/schemas/unified_generation.py` | Add `source_output_id` field |
| `src/api/services/generation/service.py` | Handle `source_output_id` → resolve output → set lineage FKs |
| `src/api/services/generation/grok_provider.py` | Accept `source_output_id` as alternative input |
| `src/api/dependencies/common.py` | Register `gallery_service`, `content_proxy` DI providers |
| `src/db/repositories/storage.py` | Add `create_job()` params for new FK columns |
| `docs/BACKEND_API_REFERENCE.md` | Document new endpoints + schemas |

---

## 13. Phased Implementation Plan

### Phase 1: Data Model + Enums + Config
1. Add enums: `OutputMediaType`, `GalleryBadge`, `GallerySourceType` to `src/core/enums.py`.
2. Add `content_url_ttl` to `Settings`.
3. Add columns to `GenerationJob`: `source_job_id`, `source_output_id`, `input_image_id`.
4. Add relationships + composite gallery index.
5. Write + run Alembic migration.
6. Update `StorageRepository.create_job()` to accept new optional params.

### Phase 2: Content Proxy
1. Create `ContentProxyService` with R2 stream proxy + cache headers.
2. Add `stream_object()` method to `R2StorageService` (async byte iterator, no full buffering).
3. Create `ContentProxyController` at `/v1/content/`.
4. Register DI.
5. Tests: streaming, cache headers, auth, 404 on wrong owner.

### Phase 3: Gallery Repository + Service
1. Create `GalleryRepository` with `list_gallery_jobs()`, `get_gallery_job()`, `batch_cover_outputs()`, `batch_output_counts()`.
2. Create `GalleryService` with cover resolution, badge logic, lineage assembly, content URL building.
3. Register DI.

### Phase 4: Gallery Controller
1. Create `GalleryController` at `/v1/gallery`.
2. Implement `GET /v1/gallery` (grid with cursor pagination, no total count).
3. Implement `GET /v1/gallery/{job_id}` (detail).
4. Delete dead gallery TODO file.

### Phase 5: Generation Flow — Lineage Tracking
1. Add `source_output_id` to `UnifiedGenerationRequest`.
2. Add XOR validation: `input_image_id` vs `source_output_id`.
3. In `GenerationService.generate()` — resolve `source_output_id` → set lineage FKs.
4. Update Grok/Aisha providers to accept output storage key as input.

### Phase 6: Tests + Documentation
1. Schema validation tests (msgspec round-trips, constraint checks).
2. Service unit tests (cover resolution matrix, badge logic, lineage assembly).
3. Integration tests (repository queries, pagination, filters, media_type filter).
4. Generation flow tests (remix with `source_output_id`, lineage populated).
5. Content proxy integration tests.
6. Update `BACKEND_API_REFERENCE.md`.
7. Mark `GET /v1/storage/jobs/{job_id}/outputs` as deprecated.
