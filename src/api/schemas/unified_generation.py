"""Unified generation request/response schemas."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

import msgspec

from src.core.enums import (
    AspectRatio,
    GenerationType,
    ModelType,
    Resolution,
    Sampler,
    Scheduler,
    VideoResolution,
)

PromptStr = Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
NegativePromptStr = Annotated[str, msgspec.Meta(max_length=2048)]
ImageCount = Annotated[int, msgspec.Meta(ge=1, le=10)]
VideoDuration = Annotated[int, msgspec.Meta(ge=1, le=15)]
ImageDim = Annotated[int, msgspec.Meta(ge=256, le=4096)]
Cfg = Annotated[float, msgspec.Meta(ge=0.0, le=30.0)]
StepsOpt = Annotated[int, msgspec.Meta(ge=1, le=150)]  # raise from le=20; bundle clamps per-model
Denoise = Annotated[float, msgspec.Meta(ge=0.0, le=1.0)]


class SourceImageReference(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Backend-owned image input reference.

    The API accepts storage IDs only. Providers resolve these IDs to the
    concrete URLs/bytes they need internally.
    """

    input_image_id: UUID | None = None
    """ID of a previously uploaded image."""

    source_output_id: UUID | None = None
    """ID of a generated output to use as input."""

    def __post_init__(self) -> None:
        if (self.input_image_id is None) == (self.source_output_id is None):
            raise ValueError(
                "Each source_images item must provide exactly one of input_image_id or source_output_id"
            )


SourceImageReferences = Annotated[
    list[SourceImageReference], msgspec.Meta(min_length=1, max_length=4)
]


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
    """Required for i2i, i2v, flf2v unless another supported source field is set.
    ID of a previously uploaded image (via POST /v1/storage/upload)."""

    source_output_id: UUID | None = None
    """ID of a GenerationOutput to use as input (remix).
    Mutually exclusive with input_image_id and source_images."""

    source_images: SourceImageReferences | None = None
    """Additional image input references for multi-reference I2I (1-4 items).
    Mutually exclusive with input_image_id and source_output_id. Lineage is recorded
    from the first item with source_output_id when no top-level source_output_id is set."""

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
    """Video resolution (480p/720p). Ignored for image generation."""

    # --- Aisha image sizing. Tier XOR explicit (width AND height). None => bundle default tier. ---

    image_resolution: Resolution | None = None
    """Image quality tier (draft/standard/high/ultra). Aisha only. Mutually exclusive with width+height."""

    width: ImageDim | None = None
    """Explicit image width in pixels (256-4096). Must be paired with height. Aisha only."""

    height: ImageDim | None = None
    """Explicit image height in pixels (256-4096). Must be paired with width. Aisha only."""

    # --- Aisha sampler overrides. None => per-model bundle default. ---

    seed: int | None = None
    """Reproducibility seed. Aisha only. Auto-generated if omitted."""

    steps: StepsOpt | None = None
    """Inference steps (1-150). Aisha only. Bundle clamps per-model max."""

    cfg: Cfg | None = None
    """CFG scale (0.0-30.0). Aisha only."""

    sampler: Sampler | None = None
    """ComfyUI sampler name. Aisha only."""

    scheduler: Scheduler | None = None
    """ComfyUI scheduler name. Aisha only."""

    denoise: Denoise | None = None
    """Denoise strength (0.0-1.0). Aisha only."""

    def __post_init__(self) -> None:
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must be provided together")
        if self.image_resolution is not None and (
            self.width is not None or self.height is not None
        ):
            raise ValueError("image_resolution and explicit width/height are mutually exclusive")
