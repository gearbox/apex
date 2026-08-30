# Providers V2 API Design — `GET /v1/providers`

## 1. Summary

Redesign the `/v1/providers` endpoint to return a **provider-grouped, capability-rich** response that eliminates the need for frontend hardcoding. The endpoint becomes **auth-optional**: unauthenticated callers get the full capabilities catalog; authenticated callers additionally receive personalized data (currently `subscription_tier` — extensible to tier-specific limits later).

### Key design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Provider naming | Keep `aisha` everywhere | `comfyui` leaks infra detail; Aisha is the product name |
| `description` field | **Keep** on model objects | Zero cost, useful for tooltips, removing = regression |
| Pricing in response | **Exclude** — stays in `GET /v1/billing/pricing` | SRP: capabilities ≠ billing; avoids stale cache issues |
| Rate limits in response | **Exclude** — enforced server-side only | Global per-model rate limits in `GenerationService`; no frontend exposure needed |
| Auth behaviour | Optional — `Authorization` header triggers personalization | Unauthenticated = public catalog; authenticated = adds `user_context` |
| Capabilities format | `list[str]` of generation type values | Compact, easy to filter (`"t2i" in model.capabilities`), forward-compatible |

---

## 2. Response Schema

### 2.1 Full response example

```jsonc
{
  // Present only when Authorization header supplied
  "user_context": {
    "subscription_tier": "pro"
  },

  "providers": [
    {
      "provider": "grok",
      "name": "xAI Grok",
      "available": true,
      "models": [
        {
          "model_key": "grok-imagine-image",
          "name": "Grok Imagine Image",
          "description": "xAI's flagship image generation model with editing support",
          "capabilities": ["t2i", "i2i"],
          "is_enabled": true,
          "max_images": 10,
          "max_prompt_length": 4096,
          "supports_negative_prompt": false,
          "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2"],
          "image": {
            "min_height": null,
            "max_height": null,
            "default_height": null,
            "output_resolutions": ["1024x1024", "2048x2048"]
          },
          "video": null
        },
        {
          "model_key": "grok-2-image-1212",
          "name": "Grok 2 Image",
          "description": "Photorealistic image generation (legacy model)",
          "capabilities": ["t2i"],
          "is_enabled": true,
          "max_images": 10,
          "max_prompt_length": 4096,
          "supports_negative_prompt": false,
          "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2"],
          "image": {
            "min_height": null,
            "max_height": null,
            "default_height": null,
            "output_resolutions": ["1024x1024"]
          },
          "video": null
        },
        {
          "model_key": "grok-imagine-video",
          "name": "Grok Imagine Video",
          "description": "Video generation from text, images, or video editing (up to 15s)",
          "capabilities": ["t2v", "i2v", "v2v", "flf2v"],
          "is_enabled": true,
          "max_images": 1,
          "max_prompt_length": 4096,
          "supports_negative_prompt": false,
          "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2"],
          "image": null,
          "video": {
            "max_duration": 15,
            "resolutions": ["480p", "720p"]
          }
        }
      ]
    },
    {
      "provider": "aisha",
      "name": "Aisha",
      "available": true,
      "models": [
        {
          "model_key": "aisha-image",
          "name": "Aisha Image",
          "description": "ComfyUI-based image generation via GPU workflows",
          "capabilities": ["t2i", "i2i"],
          "is_enabled": true,
          "max_images": 4,
          "max_prompt_length": 4096,
          "supports_negative_prompt": true,
          "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2"],
          "image": {
            "min_height": 256,
            "max_height": 2048,
            "default_height": 1024,
            "output_resolutions": null
          },
          "video": null
        },
        {
          "model_key": "aisha-video",
          "name": "Aisha Video",
          "description": "ComfyUI-based video generation via GPU workflows",
          "capabilities": ["t2v", "i2v", "v2v", "flf2v"],
          "is_enabled": false,
          "max_images": 1,
          "max_prompt_length": 4096,
          "supports_negative_prompt": true,
          "aspect_ratios": ["1:1", "16:9", "9:16"],
          "image": null,
          "video": {
            "max_duration": 10,
            "resolutions": ["480p", "720p"]
          }
        }
      ]
    }
  ]
}
```

### 2.2 Unauthenticated response

Same structure, but `user_context` key is **omitted entirely** (not `null`).

---

## 3. `msgspec.Struct` definitions

```python
# src/api/schemas/providers.py
"""Provider discovery response schemas (v2)."""

from __future__ import annotations

import msgspec


class VideoConstraints(msgspec.Struct, kw_only=True):
    """Video-specific constraints for a model."""

    max_duration: int
    """Maximum video duration in seconds."""

    resolutions: list[str]
    """Supported video resolutions, e.g. ["480p", "720p"]."""


class ModelInfo(msgspec.Struct, kw_only=True):
    """A single model's capabilities and constraints."""

    model_key: str
    """Unique model identifier (matches ModelType enum value)."""

    name: str
    """Human-readable display name."""

    description: str
    """Short description for UI tooltips."""

    capabilities: list[str]
    """Supported generation types: "t2i", "i2i", "t2v", "i2v", "v2v", "flf2v"."""

    is_enabled: bool
    """Whether this model is currently enabled for use."""

    max_images: int
    """Maximum outputs per request."""

    max_prompt_length: int
    """Maximum prompt length in characters."""

    supports_negative_prompt: bool
    """Whether this model uses negative prompts."""

    aspect_ratios: list[str]
    """Allowed aspect ratios, e.g. ["1:1", "16:9"]."""

    video: VideoConstraints | None = None
    """Video constraints. None for image-only models."""


class ProviderInfo(msgspec.Struct, kw_only=True):
    """A provider with its grouped models."""

    provider: str
    """Provider identifier (matches Provider enum value)."""

    name: str
    """Human-readable provider name."""

    available: bool
    """Whether this provider's backend is reachable and configured."""

    models: list[ModelInfo]
    """Models offered by this provider (only enabled models for public view)."""


class UserContext(msgspec.Struct, kw_only=True):
    """Per-user context included when authenticated."""

    subscription_tier: str
    """Current user's subscription tier."""


class ProvidersResponse(msgspec.Struct, kw_only=True):
    """Top-level GET /v1/providers response."""

    providers: list[ProviderInfo]
    """All registered providers with their models."""

    user_context: UserContext | None = None
    """Present only when the request includes a valid auth token."""
```

---

## 4. Model metadata registry

The per-model metadata that **cannot** be derived from `ModelType` enum properties (prompt length, negative prompt support, aspect ratios, video constraints) needs a single source of truth. Two options:

### Option A — Python-side registry dict (recommended for now)

A `MODEL_METADATA` dict in a new `src/core/model_registry.py` module, keyed by `ModelType`. This is the simplest approach that avoids a DB migration and keeps all model metadata co-located:

```python
# src/core/model_registry.py
"""Canonical model metadata registry.

Single source of truth for per-model constraints that are not derivable
from the ModelType enum (prompt limits, aspect ratios, video params, etc.).

The /v1/providers endpoint and GenerationService validation both read from here.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.enums import AspectRatio, ModelType, VideoResolution


@dataclass(frozen=True, slots=True)
class VideoMeta:
    """Video-specific constraints."""

    max_duration: int
    resolutions: tuple[VideoResolution, ...]


@dataclass(frozen=True, slots=True)
class ModelMeta:
    """Per-model metadata not derivable from the ModelType enum."""

    max_prompt_length: int
    supports_negative_prompt: bool
    aspect_ratios: tuple[AspectRatio, ...]
    video: VideoMeta | None = None


# -------------------------------------------------------------------
# Registry — add an entry for every ModelType member
# -------------------------------------------------------------------

ALL_ASPECT_RATIOS = tuple(AspectRatio)

MODEL_METADATA: dict[ModelType, ModelMeta] = {
    ModelType.GROK_IMAGINE_IMAGE: ModelMeta(
        max_prompt_length=4096,
        supports_negative_prompt=False,
        aspect_ratios=ALL_ASPECT_RATIOS,
    ),
    ModelType.GROK_2_IMAGE: ModelMeta(
        max_prompt_length=4096,
        supports_negative_prompt=False,
        aspect_ratios=ALL_ASPECT_RATIOS,
    ),
    ModelType.GROK_IMAGINE_VIDEO: ModelMeta(
        max_prompt_length=4096,
        supports_negative_prompt=False,
        aspect_ratios=ALL_ASPECT_RATIOS,
        video=VideoMeta(
            max_duration=15,
            resolutions=(VideoResolution.RES_480P, VideoResolution.RES_720P),
        ),
    ),
    ModelType.AISHA: ModelMeta(
        max_prompt_length=4096,
        supports_negative_prompt=True,
        aspect_ratios=ALL_ASPECT_RATIOS,
    ),
}


def get_model_meta(model: ModelType) -> ModelMeta:
    """Get metadata for a model. Raises KeyError if model not registered."""
    return MODEL_METADATA[model]
```

### Option B — DB columns on `generation_models`

Add `max_prompt_length`, `supports_negative_prompt`, `aspect_ratios` (JSON array), and a `video_constraints` (JSONB) column. Better for admin-editable metadata, but overkill right now — defer to a future migration if admin UI model editing is needed.

**Recommendation:** Start with **Option A**. If the admin panel later needs to tweak these values at runtime, migrate to Option B with an Alembic migration that seeds from the Python registry.

---

## 5. Provider availability logic

Current logic hardcodes provider list. The new approach makes it data-driven:

```python
# Inside the controller / service builder

PROVIDER_DISPLAY_NAMES: dict[Provider, str] = {
    Provider.AISHA: "Aisha",
    Provider.GROK: "xAI Grok",
}


def _is_provider_available(provider: Provider, *, grok_configured: bool) -> bool:
    """Determine if a provider's backend is currently reachable."""
    if provider == Provider.GROK:
        return grok_configured
    # Aisha: always available (pull-based workers — job sits until a node picks it up)
    return True
```

---

## 6. Controller implementation sketch

```python
# src/api/routes/providers.py (v2)


class ProvidersController(Controller):
    path = "/v1/providers"
    tags = ["Providers"]

    @get("/")
    async def list_providers(
        self,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
        current_user_id: UUID | None = None,  # injected only if auth header present
    ) -> ProvidersResponse:
        repo = GenerationModelRepository(session)
        db_models = await repo.list_enabled()

        # Group DB records by provider
        provider_models: dict[Provider, list[ModelInfo]] = {}
        for record in db_models:
            try:
                mt = ModelType(record.model_key)
            except ValueError:
                continue

            meta = get_model_meta(mt)
            info = ModelInfo(
                model_key=mt.value,
                name=record.name,
                description=record.description,
                capabilities=[gt.value for gt in GenerationType if mt.supports_generation_type(gt)],
                is_enabled=record.is_enabled,
                max_images=mt.max_concurrent_outputs,
                max_prompt_length=meta.max_prompt_length,
                supports_negative_prompt=meta.supports_negative_prompt,
                aspect_ratios=[ar.value for ar in meta.aspect_ratios],
                video=(
                    VideoConstraints(
                        max_duration=meta.video.max_duration,
                        resolutions=[r.value for r in meta.video.resolutions],
                    )
                    if meta.video
                    else None
                ),
            )

            provider_models.setdefault(mt.provider, []).append(info)

        # Build provider list — include all known providers even if they have 0 enabled models
        grok_configured = grok_job_service is not None
        providers = [
            ProviderInfo(
                provider=p.value,
                name=PROVIDER_DISPLAY_NAMES[p],
                available=_is_provider_available(p, grok_configured=grok_configured),
                models=provider_models.get(p, []),
            )
            for p in Provider
        ]

        # Optional user context
        user_context = None
        if current_user_id is not None:
            user = await UserRepository(session).get_active_user(current_user_id)
            if user is not None:
                tier = user.subscription_tier
                user_context = UserContext(
                    subscription_tier=tier.value if hasattr(tier, "value") else tier,
                )

        return ProvidersResponse(providers=providers, user_context=user_context)
```

---

## 7. Global per-model rate limiting (backend-only)

Rate limits are enforced server-side in `GenerationService`, **not** exposed to the frontend.

### 7.1 Configuration

```python
# src/core/model_registry.py — extend ModelMeta


@dataclass(frozen=True, slots=True)
class RateLimitMeta:
    """Global rate limit for a model (shared across all users)."""

    max_requests: int
    """Maximum requests within the window."""

    window_seconds: int
    """Sliding window duration."""


@dataclass(frozen=True, slots=True)
class ModelMeta:
    max_prompt_length: int
    supports_negative_prompt: bool
    aspect_ratios: tuple[AspectRatio, ...]
    video: VideoMeta | None = None
    rate_limit: RateLimitMeta | None = None  # None = no limit


MODEL_METADATA: dict[ModelType, ModelMeta] = {
    ModelType.GROK_IMAGINE_VIDEO: ModelMeta(
        max_prompt_length=4096,
        supports_negative_prompt=False,
        aspect_ratios=ALL_ASPECT_RATIOS,
        video=VideoMeta(max_duration=15, resolutions=(...)),
        rate_limit=RateLimitMeta(max_requests=10, window_seconds=60),
    ),
    # ... other models
}
```

### 7.2 Enforcement — sliding window via PostgreSQL

Use a lightweight `model_rate_limit_log` table or an in-memory counter (Redis if available). For the MVP without Redis:

```python
# src/api/services/generation/rate_limiter.py


class ModelRateLimiter:
    """Sliding-window rate limiter for global per-model request throttling.

    Uses PostgreSQL for distributed correctness (no Redis required).
    """

    def __init__(self, session_factory: AsyncSession) -> None:
        self._session_factory = session_factory

    async def check_and_record(self, model: ModelType) -> None:
        """Check rate limit and record the request atomically.

        Raises:
            RateLimitExceededError: If the model's global rate limit is exceeded.
        """
        meta = get_model_meta(model)
        if meta.rate_limit is None:
            return  # No limit configured

        # Count recent requests within the window via a CTE + INSERT
        # Single round-trip: count + conditional insert
        ...
```

### 7.3 Integration point in `GenerationService`

Add rate limit check as **step 4.5** — after model-enabled check, before provider delegation:

```python
# In GenerationService.generate()

# 4. Check model is enabled
# ... existing code ...

# 4.5 Global rate limit check
await self._rate_limiter.check_and_record(request.model)

# 5. Resolve provider
# ... existing code ...
```

---

## 8. Auth-optional dependency injection

Litestar doesn't natively support "inject user_id if token present, else None" without a guard. The cleanest approach:

```python
# src/api/dependencies/auth.py


async def get_optional_user_id(request: Request) -> UUID | None:
    """Extract user_id from JWT if Authorization header is present.

    Returns None if no token is provided (public access).
    Raises 401 if a token IS provided but is invalid/expired.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None

    # Reuse existing JWT verification logic
    try:
        return _extract_user_id_from_bearer(auth_header)
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
```

Then in the controller:

```python
dependencies = {"current_user_id": Provide(get_optional_user_id)}
```

No guard needed — the dependency itself handles the optional nature.

---

## 9. Migration plan

### Phase 1 — Non-breaking additions (this PR)

1. Create `src/core/model_registry.py` with `ModelMeta` + `MODEL_METADATA`
2. Create `src/api/schemas/providers.py` with new response structs
3. Rewrite `ProvidersController` with new response shape
4. Add `get_optional_user_id` dependency
5. Add `ModelRateLimiter` (PostgreSQL-based, or simple in-memory dict with TTL)
6. Wire rate limiter into `GenerationService`
7. Update `BACKEND_API_REFERENCE.md`
8. Update `schema.json` export for frontend project knowledge

### Phase 2 — Cleanup (follow-up PR)

1. Remove old `ProviderModelInfo`, `_build_model_info` from `providers.py`
2. Remove `GrokModelInfo`, `GROK_MODELS` from `src/api/schemas/grok.py` (replaced by registry)
3. Use `model_registry.get_model_meta()` in `GenerationService._validate_inputs()` for prompt length validation (currently hardcoded as `max_length=4096` in msgspec annotation)

---

## 10. Frontend migration notes

| Current field | New field | Notes |
|---------------|-----------|-------|
| `providers[]` (flat) | `providers[].models[]` (nested) | Models are now grouped under their provider |
| `models[].model` | `models[].model_key` | Renamed for clarity |
| `models[].supports_t2i` | `models[].capabilities: ["t2i"]` | Compact list instead of N booleans |
| `models[].supports_i2i` | `"i2i" in capabilities` | Same |
| `models[].supports_t2v` | `"t2v" in capabilities` | Same |
| *(not present)* | `models[].max_prompt_length` | **New** — use for client-side validation |
| *(not present)* | `models[].supports_negative_prompt` | **New** — conditionally show negative prompt input |
| *(not present)* | `models[].aspect_ratios` | **New** — populate aspect ratio selector per model |
| *(not present)* | `models[].video` | **New** — `null` for image models, object for video models |
| *(not present)* | `user_context` | **New** — present only when authenticated |

### Frontend capability check example

```typescript
// Before (v1)
const canDoI2I = model.supports_i2i;

// After (v2)
const canDoI2I = model.capabilities.includes('i2i');

// Check if video model
const isVideoModel = model.video !== null;

// Get max duration
const maxDuration = model.video?.max_duration ?? 0;
```

---

## 11. Test plan

### Unit tests (`tests/unit/test_providers_v2.py`)

1. **Schema roundtrip** — encode/decode `ProvidersResponse` with all fields populated
2. **Schema roundtrip with null video** — `video=None` serializes correctly
3. **Schema roundtrip without user_context** — `user_context=None` omits the field
4. **Capabilities derivation** — verify `ModelType.GROK_IMAGINE_VIDEO` produces `["t2v", "i2v", "v2v", "flf2v"]`
5. **Capabilities derivation (image model)** — `ModelType.GROK_IMAGINE_IMAGE` produces `["t2i", "i2i"]`
6. **Model metadata completeness** — every `ModelType` member has a `MODEL_METADATA` entry
7. **Aspect ratios serialization** — all `AspectRatio` enum values appear in output

### Unit tests (`tests/unit/test_model_registry.py`)

1. **get_model_meta returns correct data** for each `ModelType`
2. **get_model_meta raises KeyError** for unregistered model
3. **Rate limit config** — models with `rate_limit` have valid `max_requests > 0` and `window_seconds > 0`

### Integration tests (`tests/integration/test_providers_v2.py`)

1. **Unauthenticated request** — returns `providers` without `user_context`
2. **Authenticated request** — returns `providers` with `user_context.subscription_tier`
3. **Invalid token** — returns `401`
4. **Provider availability** — Grok shows `available: false` when `grok_job_service` is `None`
5. **Disabled model excluded** — disable a model via `set_enabled(key, False)`, verify it's absent
6. **Response shape matches schema** — full decode into `ProvidersResponse`

### Integration tests (`tests/integration/test_rate_limiter.py`)

1. **Under limit** — N requests within window succeed
2. **At limit** — request N+1 raises `RateLimitExceededError`
3. **Window expiry** — after window elapses, requests succeed again
4. **No limit configured** — models without `rate_limit` always pass
