"""Canonical model metadata registry.

Single source of truth for per-model constraints that are not derivable
from the ModelType enum (prompt limits, aspect ratios, video params, rate limits).

The /v1/providers endpoint and GenerationService validation both read from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.enums import (
    AspectRatio,
    GenerationType,
    MediaFormat,
    MediaKind,
    MediaSlot,
    ModelType,
    Provider,
    Resolution,
    VideoParameter,
    VideoResolution,
)
from src.core.generation_mode import GenerationModeMeta, SourceMediaConstraints

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class ImageMeta:
    """Image-specific constraints for a model.

    Grok models do not expose a resolution parameter — the output size is
    determined server-side based on the aspect ratio.  We still surface
    the *effective* output resolutions so the frontend can display them.
    For Aisha, the user controls height directly (256-2048).
    """

    edit_aspect_ratios: tuple[AspectRatio, ...]
    """Aspect ratios this model can genuinely RESHAPE TO during image editing (i2i).
    Empty tuple = model cannot reshape on edit: the aspect_ratio parameter MUST be
    omitted from provider calls and the output follows the source image's aspect.
    This is intentionally required with no default: every new image model must make
    a conscious, verified reshape decision (fail-loud at review time, not runtime).
    """

    min_height: int | None = None
    """Minimum height in pixels the user can request. None = not user-controllable."""

    max_height: int | None = None
    """Maximum height in pixels the user can request. None = not user-controllable."""

    default_height: int | None = None
    """Default height applied when the user omits the parameter."""

    output_resolutions: tuple[str, ...] | None = None
    """Advertised output resolutions (informational, e.g. ("1024x1024", "2048x2048")).
    None means the backend determines the size automatically."""

    supported_tiers: tuple[Resolution, ...] | None = None
    """Image quality tiers supported by this model. None for Grok (fixed server-side)."""

    default_tier: Resolution | None = None
    """Default quality tier when the user omits image_resolution. None for Grok."""


@dataclass(frozen=True, slots=True)
class InputVideoConstraints:
    """What a model accepts as a *video source* (V2V), checked before any billing work.

    Every field is required with no default so a new model decides each one.
    """

    formats: frozenset[MediaFormat]
    """Containers the provider decodes; a source in any other format is rejected
    locally (e.g. a library WebM for a model that takes only mp4/mov)."""

    min_duration_ms: int
    """Shortest accepted source clip in milliseconds. 0 = no minimum."""

    max_duration_ms: int | None
    """Longest accepted source clip in milliseconds. None = no model-specific cap."""

    max_bytes: int | None
    """Largest accepted source file in bytes (decimal, never rounded up). None = no cap."""

    def __post_init__(self) -> None:
        """Reject a registry entry that could never accept any source."""
        if not self.formats or not all(fmt.is_video for fmt in self.formats):
            raise ValueError("input video formats must be a non-empty set of video formats")
        if self.min_duration_ms < 0:
            raise ValueError("min_duration_ms must be >= 0")
        if self.max_duration_ms is not None and self.max_duration_ms < self.min_duration_ms:
            raise ValueError("max_duration_ms must be >= min_duration_ms")
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")


@dataclass(frozen=True, slots=True)
class SeedRange:
    """Inclusive bounds a model accepts for ``seed``."""

    min: int
    max: int

    def __post_init__(self) -> None:
        """Reject an empty range."""
        if self.min > self.max:
            raise ValueError("seed range min must be <= max")


@dataclass(frozen=True, slots=True)
class VideoMeta:
    """Video-specific constraints.

    Every field is required with no default: each video model must make a
    conscious decision about each one (fail-loud at review time, not runtime).
    """

    min_duration: int
    """Minimum video duration in seconds."""

    max_duration: int
    """Maximum video duration in seconds."""

    resolutions: tuple[VideoResolution, ...]
    """Supported video resolutions."""

    supported_parameters: frozenset[VideoParameter]
    """Optional request parameters this model honours; sending any other as
    non-``None`` raises ``UnsupportedGenerationParameterError``."""

    input_anchored_modes: frozenset[GenerationType]
    """Modes whose output ratio follows the input media: ``aspect_ratio`` must be
    omitted for them (sending one is a validation error). Modes not listed accept
    ``aspect_ratio`` even when source media is supplied (e.g. reference-to-video,
    whose references are subjects, not frames). Every entry must be one of the
    model's ``generation_modes`` with video output."""

    input_video: InputVideoConstraints | None
    """Video-source (V2V) limits. None = the model accepts no video source; it must be
    non-None exactly when one of the model's generation modes takes ``MediaKind.VIDEO``."""

    max_combined_duration: int | None
    """V2V: input seconds + requested output seconds must not exceed this. None = no rule."""

    bills_input_video_seconds: bool
    """Pricing: whether input video seconds count toward billed_seconds."""


@dataclass(frozen=True, slots=True)
class AudioMeta:
    """Future audio constraints; no current ``GenerationType`` emits audio."""

    max_duration_seconds: int
    sample_rates: tuple[int, ...]


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
    """Allowed aspect ratios for text-to-image generation."""

    max_concurrent_outputs: int
    """Maximum number of outputs this model can produce per request."""

    generation_modes: Mapping[GenerationType, GenerationModeMeta]
    """Per-generation-mode owned-library input contracts."""

    image: ImageMeta | None = None
    """Image constraints. None for video-only models."""

    video: VideoMeta | None = None
    """Video constraints. None for image-only models."""

    seed_range: SeedRange | None = None
    """Accepted ``seed`` bounds. None = no model-specific bound. ``null`` / omitted on the
    wire always means "random"; a value outside the range is rejected before any billing."""

    audio: AudioMeta | None = None
    """Audio constraints. Reserved until an audio generation type exists."""

    rate_limit: RateLimitMeta | None = None
    """Global rate limit. None means unlimited."""

    requires_age_verification: bool = False
    """Whether users must be age-verified before generating with this model."""

    typical_bootstrap_seconds: int | None = None
    """Configured display hint for initial node bootstrap; never a computed ETA."""

    typical_attach_seconds: int | None = None
    """Configured display hint for additive model attachment; never a computed ETA."""


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
        generation_modes={
            GenerationType.T2I: GenerationModeMeta(),
            GenerationType.I2I: GenerationModeMeta(
                SourceMediaConstraints(min=1, max=4, media_types=frozenset({MediaKind.IMAGE}))
            ),
        },
        image=ImageMeta(
            # Verified via direct curl against xAI's edit endpoint: the model
            # accepts aspect_ratio on edits but stretches the source to fit
            # rather than recomposing. xAI's own console hides the aspect
            # control for this model in edit mode.
            edit_aspect_ratios=(),
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
        generation_modes={GenerationType.T2I: GenerationModeMeta()},
        image=ImageMeta(
            # t2i-only model — trivially no edit reshape capability.
            edit_aspect_ratios=(),
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
        generation_modes={
            GenerationType.T2V: GenerationModeMeta(),
            GenerationType.I2V: GenerationModeMeta(
                SourceMediaConstraints(min=1, max=1, media_types=frozenset({MediaKind.IMAGE}))
            ),
            GenerationType.V2V: GenerationModeMeta(
                SourceMediaConstraints(min=1, max=1, media_types=frozenset({MediaKind.VIDEO}))
            ),
        },
        video=VideoMeta(
            min_duration=1,
            max_duration=15,
            resolutions=(VideoResolution.RES_480P, VideoResolution.RES_720P),
            supported_parameters=frozenset(),
            input_anchored_modes=frozenset(),
            # Every library video format, no clip limits: preserves the pre-registry behaviour.
            input_video=InputVideoConstraints(
                formats=frozenset({MediaFormat.MP4, MediaFormat.WEBM, MediaFormat.MOV}),
                min_duration_ms=0,
                max_duration_ms=None,
                max_bytes=None,
            ),
            max_combined_duration=None,
            bills_input_video_seconds=False,
        ),
        rate_limit=RateLimitMeta(max_requests=10, window_seconds=60),
    ),
    ModelType.AISHA_IMAGE: ModelMeta(
        provider=Provider.AISHA,
        max_prompt_length=4096,
        supports_negative_prompt=True,
        aspect_ratios=ALL_ASPECT_RATIOS,
        max_concurrent_outputs=4,
        generation_modes={
            GenerationType.T2I: GenerationModeMeta(),
            # Provider limit is TextEncodeQwenImageEditPlus's image inputs (image1-image3);
            # a bundle narrows it to its declared reference slots via resolve_generation_modes.
            GenerationType.I2I: GenerationModeMeta(
                SourceMediaConstraints(min=1, max=3, media_types=frozenset({MediaKind.IMAGE}))
            ),
        },
        image=ImageMeta(
            # Qwen-Image-Edit recomposes natively onto the target latent —
            # the source is conditioning, not the output canvas.
            edit_aspect_ratios=ALL_ASPECT_RATIOS,
            min_height=256,
            max_height=2048,
            default_height=1024,
            supported_tiers=tuple(Resolution),
            default_tier=Resolution.STANDARD,
        ),
        rate_limit=None,
        requires_age_verification=True,
        # Configured display hints only. They never feed an operation ETA.
        typical_bootstrap_seconds=600,
        typical_attach_seconds=360,
    ),
    ModelType.AISHA_IMAGE_LITE: ModelMeta(
        provider=Provider.AISHA,
        max_prompt_length=4096,
        # zit.cyberrealistic has no negative_prompt role (ConditioningZeroOut
        # supplies the sampler's negative) — matches the bundle, not narrowed.
        supports_negative_prompt=False,
        aspect_ratios=ALL_ASPECT_RATIOS,
        max_concurrent_outputs=4,
        # Bundle declares no media_inputs — no owned-library source media.
        generation_modes={GenerationType.T2I: GenerationModeMeta()},
        image=ImageMeta(
            edit_aspect_ratios=(),
            min_height=256,
            max_height=2048,
            default_height=1024,
            supported_tiers=tuple(Resolution),
            default_tier=Resolution.STANDARD,
        ),
        rate_limit=None,
        requires_age_verification=True,
        # Configured display hints only. They never feed an operation ETA.
        typical_bootstrap_seconds=480,
        typical_attach_seconds=240,
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
        generation_modes={
            GenerationType.T2V: GenerationModeMeta(),
            GenerationType.I2V: GenerationModeMeta(
                SourceMediaConstraints(
                    min=1,
                    max=1,
                    media_types=frozenset({MediaKind.IMAGE}),
                    roles=(MediaSlot.FIRST_FRAME,),
                )
            ),
            GenerationType.FLF2V: GenerationModeMeta(
                SourceMediaConstraints(
                    min=2,
                    max=2,
                    media_types=frozenset({MediaKind.IMAGE}),
                    roles=(MediaSlot.FIRST_FRAME, MediaSlot.LAST_FRAME),
                )
            ),
        },
        video=VideoMeta(
            min_duration=1,
            max_duration=10,
            resolutions=(VideoResolution.RES_480P, VideoResolution.RES_720P),
            supported_parameters=frozenset(),
            input_anchored_modes=frozenset(),
            input_video=None,  # no mode takes a video source
            max_combined_duration=None,
            bills_input_video_seconds=False,
        ),
        rate_limit=None,
        requires_age_verification=True,
        # Configured display hints only. They never feed an operation ETA.
        typical_bootstrap_seconds=900,
        typical_attach_seconds=420,
    ),
    ModelType.WAN_3_VIDEO: ModelMeta(
        provider=Provider.ALIBABA,
        max_prompt_length=4096,  # PromptStr cap; upstream allows 20 000, do not widen here
        supports_negative_prompt=False,  # Wan 3.0 has no negative_prompt
        aspect_ratios=(
            AspectRatio.RATIO_16_9,
            AspectRatio.RATIO_4_3,
            AspectRatio.RATIO_1_1,
            AspectRatio.RATIO_3_4,
            AspectRatio.RATIO_9_16,
        ),
        max_concurrent_outputs=1,
        generation_modes={
            GenerationType.T2V: GenerationModeMeta(),
            GenerationType.I2V: GenerationModeMeta(
                SourceMediaConstraints(
                    min=1,
                    max=1,
                    media_types=frozenset({MediaKind.IMAGE}),
                    roles=(MediaSlot.FIRST_FRAME,),
                )
            ),
            GenerationType.FLF2V: GenerationModeMeta(
                SourceMediaConstraints(
                    min=2,
                    max=2,
                    media_types=frozenset({MediaKind.IMAGE}),
                    roles=(MediaSlot.FIRST_FRAME, MediaSlot.LAST_FRAME),
                )
            ),
            GenerationType.V2V: GenerationModeMeta(
                SourceMediaConstraints(min=1, max=1, media_types=frozenset({MediaKind.VIDEO}))
            ),
            # "Image 1..N" in the prompt refers to source_media in request order.
            GenerationType.R2V: GenerationModeMeta(
                SourceMediaConstraints(min=1, max=8, media_types=frozenset({MediaKind.IMAGE}))
            ),
        },
        video=VideoMeta(
            min_duration=2,
            max_duration=30,
            resolutions=(
                VideoResolution.RES_480P,
                VideoResolution.RES_720P,
                VideoResolution.RES_1080P,
            ),
            supported_parameters=frozenset({VideoParameter.AUDIO, VideoParameter.PROMPT_EXTEND}),
            input_anchored_modes=frozenset(
                {GenerationType.I2V, GenerationType.FLF2V, GenerationType.V2V}
            ),
            # reference_video: mp4/mov, 1-15 s per clip, <= 100 MB (Wan 3.0 API reference).
            # Decimal 100 MB, the conservative reading. Frame rate (>= 16 fps) and side
            # length (240-4096 px) are not checked locally: ResolvedSourceMedia has neither.
            input_video=InputVideoConstraints(
                formats=frozenset({MediaFormat.MP4, MediaFormat.MOV}),
                min_duration_ms=1_000,
                max_duration_ms=15_000,
                max_bytes=100_000_000,
            ),
            max_combined_duration=30,
            bills_input_video_seconds=True,
        ),
        # Upstream: "-1 or [0, 2147483647]". -1 means random, which we spell null/omitted.
        seed_range=SeedRange(0, 2_147_483_647),
        rate_limit=None,  # upstream quotas unknown; Throttling maps to RATE_LIMITED
        requires_age_verification=False,
    ),
}


def get_model_meta(model: ModelType) -> ModelMeta:
    """Get metadata for a model.

    Raises:
        KeyError: If model is not registered in MODEL_METADATA.
    """
    return MODEL_METADATA[model]
