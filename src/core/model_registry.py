"""Canonical model metadata registry.

Single source of truth for per-model constraints that are not derivable
from the ModelType enum (prompt limits, aspect ratios, video params, rate limits).

The /v1/providers endpoint and GenerationService validation both read from here.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.enums import AspectRatio, GenerationType, ModelType, Provider, VideoResolution


@dataclass(frozen=True, slots=True)
class ImageMeta:
    """Image-specific constraints for a model.

    Grok models do not expose a resolution parameter — the output size is
    determined server-side based on the aspect ratio.  We still surface
    the *effective* output resolutions so the frontend can display them.
    For Aisha, the user controls height directly (256-2048).
    """

    supported_types: frozenset[GenerationType]
    """Supported video generation types, e.g. frozenset({GenerationType.T2V, GenerationType.I2V})"""

    min_height: int | None = None
    """Minimum height in pixels the user can request. None = not user-controllable."""

    max_height: int | None = None
    """Maximum height in pixels the user can request. None = not user-controllable."""

    default_height: int | None = None
    """Default height applied when the user omits the parameter."""

    output_resolutions: tuple[str, ...] | None = None
    """Advertised output resolutions (informational, e.g. ("1024x1024", "2048x2048")).
    None means the backend determines the size automatically."""


@dataclass(frozen=True, slots=True)
class VideoMeta:
    """Video-specific constraints."""

    max_duration: int
    """Maximum video duration in seconds."""

    resolutions: tuple[VideoResolution, ...]
    """Supported video resolutions."""

    supported_types: frozenset[GenerationType]
    """Supported video generation types, e.g. frozenset({GenerationType.T2V, GenerationType.I2V})"""


@dataclass(frozen=True, slots=True)
class RateLimitMeta:
    """Global rate limit for a model (shared across all users, keyed by API key)."""

    max_requests: int
    """Maximum requests within the window."""

    window_seconds: int
    """Sliding window duration."""


@dataclass(frozen=True, slots=True)
class ModelMeta:
    """Per-model metadata — single source of truth for all model properties."""

    provider: Provider
    """Authoritative provider association."""

    max_prompt_length: int
    """Maximum prompt length in characters."""

    supports_negative_prompt: bool
    """Whether this model accepts negative prompts."""

    aspect_ratios: tuple[AspectRatio, ...]
    """Allowed aspect ratios for this model."""

    max_concurrent_outputs: int
    """Maximum number of outputs this model can produce per request."""

    image: ImageMeta | None = None
    """Image constraints. None for video-only models."""

    video: VideoMeta | None = None
    """Video constraints. None for image-only models."""

    rate_limit: RateLimitMeta | None = None
    """Global rate limit. None means unlimited."""


# -------------------------------------------------------------------
# Registry — every ModelType member MUST have an entry here
# -------------------------------------------------------------------

ALL_ASPECT_RATIOS = tuple(AspectRatio)

# Grok supports additional aspect ratios beyond our enum.
# We expose only those present in the AspectRatio enum for now.
GROK_ASPECT_RATIOS = ALL_ASPECT_RATIOS  # subset of xAI's full list

MODEL_METADATA: dict[ModelType, ModelMeta] = {
    ModelType.GROK_IMAGINE_IMAGE: ModelMeta(
        provider=Provider.GROK,
        max_prompt_length=4096,
        supports_negative_prompt=False,
        aspect_ratios=GROK_ASPECT_RATIOS,
        max_concurrent_outputs=10,
        image=ImageMeta(
            supported_types=frozenset({
                GenerationType.T2I,
                GenerationType.I2I,
            }),
            # Grok does not expose a user-controllable resolution parameter.
            # Output is determined server-side based on aspect ratio.
            output_resolutions=("1024x1024", "2048x2048"),
        ),
        rate_limit=None,  # No global limit for image models (for now)
    ),
    ModelType.GROK_2_IMAGE: ModelMeta(
        provider=Provider.GROK,
        max_prompt_length=4096,
        supports_negative_prompt=False,
        aspect_ratios=GROK_ASPECT_RATIOS,
        max_concurrent_outputs=10,
        image=ImageMeta(
            supported_types=frozenset({
                GenerationType.T2I,
            }),
            output_resolutions=("1024x1024",),
        ),
        rate_limit=None,
    ),
    ModelType.GROK_IMAGINE_VIDEO: ModelMeta(
        provider=Provider.GROK,
        max_prompt_length=4096,
        supports_negative_prompt=False,
        aspect_ratios=GROK_ASPECT_RATIOS,
        max_concurrent_outputs=1,
        video=VideoMeta(
            max_duration=15,
            resolutions=(VideoResolution.RES_480P, VideoResolution.RES_720P),
            supported_types=frozenset({
                GenerationType.T2V,
                GenerationType.I2V,
                GenerationType.V2V,
                # no FLF2V — Grok doesn't support it
            }),
        ),
        rate_limit=RateLimitMeta(max_requests=10, window_seconds=60),
    ),
    ModelType.AISHA_IMAGE: ModelMeta(
        provider=Provider.AISHA,
        max_prompt_length=4096,
        supports_negative_prompt=True,
        aspect_ratios=ALL_ASPECT_RATIOS,
        max_concurrent_outputs=4,
        image=ImageMeta(
            supported_types=frozenset({
                GenerationType.T2I,
                GenerationType.I2I,
            }),
            min_height=256,
            max_height=2048,
            default_height=1024,
        ),
        rate_limit=None,
    ),
    ModelType.AISHA_VIDEO: ModelMeta(
        provider=Provider.AISHA,
        max_prompt_length=4096,
        supports_negative_prompt=True,
        aspect_ratios=(
            AspectRatio.RATIO_1_1,
            AspectRatio.RATIO_16_9,
            AspectRatio.RATIO_9_16,
        ),
        max_concurrent_outputs=1,
        video=VideoMeta(
            max_duration=10,
            resolutions=(VideoResolution.RES_480P, VideoResolution.RES_720P),
                supported_types=frozenset({
                GenerationType.T2V,
                GenerationType.I2V,
                GenerationType.FLF2V,
                # no V2V for aisha yet
            }),
        ),
        rate_limit=None,
    ),
}


def get_model_meta(model: ModelType) -> ModelMeta:
    """Get metadata for a model.

    Raises:
        KeyError: If model is not registered in MODEL_METADATA.
    """
    return MODEL_METADATA[model]
