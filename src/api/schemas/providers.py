"""Provider discovery response schemas (v2).

Provider-grouped, capability-rich response for GET /v1/providers.
"""

from __future__ import annotations

import msgspec


class ImageConstraints(msgspec.Struct, kw_only=True):
    """Image-specific constraints for a model."""

    min_height: int | None = None
    """Minimum height in px the user can request. None = not user-controllable."""

    max_height: int | None = None
    """Maximum height in px the user can request. None = not user-controllable."""

    default_height: int | None = None
    """Default height applied when the user omits the parameter."""

    output_resolutions: list[str] | None = None
    """Advertised output resolutions (informational, e.g. ["1024x1024", "2048x2048"]).
    None means the backend determines the size automatically."""

    supported_tiers: list[str] | None = None
    """Supported image quality tiers (draft/standard/high/ultra). None for models with fixed sizing."""

    default_tier: str | None = None
    """Default quality tier. None for models with fixed sizing."""

    tier_megapixels: dict[str, float] | None = None
    """Target megapixel budget per tier. Actual W*H depends on model and aspect ratio."""


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

    requires_age_verification: bool = False
    """Whether users must be age-verified before generating with this model."""

    image: ImageConstraints | None = None
    """Image constraints. None for video-only models."""

    video: VideoConstraints | None = None
    """Video constraints. None for image-only models."""

    session_state: str | None = None
    """Per-user readiness for on-demand models (none/provisioning/active/paused/stale).
    Populated only for authenticated requests on on_demand provider models; null otherwise."""


class ProviderInfo(msgspec.Struct, kw_only=True):
    """A provider with its grouped models."""

    provider: str
    """Provider identifier (matches Provider enum value)."""

    name: str
    """Human-readable provider name."""

    available: bool
    """Whether this provider is configured and able to serve requests (registry membership)."""

    provisioning_mode: str
    """How the provider's compute is made available: always_on (cloud API) or on_demand
    (requires a per-user GPU session via POST /v1/sessions before generation)."""

    models: list[ModelInfo]
    """Models offered by this provider."""


class UserContext(msgspec.Struct, kw_only=True):
    """Per-user context included when authenticated."""

    subscription_tier: str
    """Current user's subscription tier."""


class ProvidersResponse(msgspec.Struct, kw_only=True):
    """Top-level GET /v1/providers response."""

    providers: list[ProviderInfo]
    """All registered providers with their models."""

    user_context: UserContext | None = None
    """Present only when the request includes a valid auth token.
    null for unauthenticated requests."""
