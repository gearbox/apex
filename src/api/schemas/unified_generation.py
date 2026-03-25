"""Unified generation request/response schemas."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

import msgspec

from src.core.enums import AspectRatio, GenerationType, ModelType, VideoResolution

PromptStr = Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
NegativePromptStr = Annotated[str, msgspec.Meta(max_length=2048)]
ImageCount = Annotated[int, msgspec.Meta(ge=1, le=10)]
VideoDuration = Annotated[int, msgspec.Meta(ge=1, le=15)]
Height = Annotated[int, msgspec.Meta(ge=256, le=2048)]
Steps = Annotated[int, msgspec.Meta(ge=1, le=20)]


class UnifiedGenerationRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Provider-agnostic generation request.

    The backend resolves the correct provider from ``model`` and validates
    that the requested ``generation_type`` is compatible.
    """

    prompt: PromptStr
    """Text description of the content to generate."""

    generation_type: GenerationType
    """What kind of generation: t2i, i2i, t2v, i2v, v2v, flf2v."""

    model: ModelType
    """Which model to use. Provider is inferred from this field via ModelType.provider."""

    # --- Optional fields (applicability depends on generation_type) ---

    input_image_id: UUID | None = None
    """Required for i2i, i2v, flf2v. ID of a previously uploaded image (via POST /v1/storage/upload)."""

    source_output_id: UUID | None = None
    """ID of a GenerationOutput to use as input (remix).
    Mutually exclusive with input_image_id."""

    input_video_url: str | None = None
    """Required for v2v. Publicly-accessible URL of the source video."""

    negative_prompt: NegativePromptStr | None = None
    """Negative prompt. Applied by Aisha provider; stored but ignored by Grok."""

    aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1
    """Aspect ratio for the generated content."""

    n: ImageCount = 1
    """Number of outputs to generate (image models only). Max varies by model (see ModelType.max_concurrent_outputs)."""

    name: str | None = None
    """Job display name. Auto-generated from prompt[:50] if omitted."""

    # --- Video-specific ---

    duration: VideoDuration = 5
    """Video duration in seconds (1-15). Ignored for image generation."""

    resolution: VideoResolution = VideoResolution.RES_720P
    """Video resolution. Ignored for image generation."""

    # --- Aisha-specific (ignored by Grok provider) ---

    height: Height | None = None
    """Image height in pixels (256-2048). Aisha only. Default: 1024."""

    seed: int | None = None
    """Reproducibility seed. Aisha only. Auto-generated if omitted."""

    steps: Steps | None = None
    """Inference steps (1-20). Aisha only. Default: 12."""
