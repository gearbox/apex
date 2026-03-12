"""Grok API schemas using msgspec for high-performance serialization."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

import msgspec

from src.core.enums import (
    AspectRatio,
    ModelType,
    VideoResolution,
)

# -----------------------------------------------------------------------------
# Validation constraints
# -----------------------------------------------------------------------------

PromptStr = Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
ImageCount = Annotated[int, msgspec.Meta(ge=1, le=10)]
VideoDuration = Annotated[int, msgspec.Meta(ge=1, le=15)]


# -----------------------------------------------------------------------------
# Image Generation Schemas
# -----------------------------------------------------------------------------


class GrokImageRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request schema for Grok image generation."""

    prompt: PromptStr
    """Text description of the image to generate."""

    model: ModelType = ModelType.GROK_IMAGINE_IMAGE
    """Model to use for generation."""

    n: ImageCount = 1
    """Number of images to generate (1-10)."""

    aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1
    """Aspect ratio for the generated image."""

    name: str | None = None
    """Job name (auto-generated from prompt if not provided)."""

    def __post_init__(self) -> None:
        """Validate model supports image generation."""
        if self.model not in (ModelType.GROK_IMAGINE_IMAGE, ModelType.GROK_2_IMAGE):
            raise ValueError(
                f"Model {self.model.value} does not support image generation. "
                f"Use {ModelType.GROK_IMAGINE_IMAGE.value} or {ModelType.GROK_2_IMAGE.value}."
            )


class GrokImageEditRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request schema for Grok image editing (I2I)."""

    prompt: PromptStr
    """Text description of the edit to perform."""

    input_image_id: UUID
    """ID of the uploaded image to edit."""

    model: ModelType = ModelType.GROK_IMAGINE_IMAGE
    """Model to use (must be grok-imagine-image)."""

    name: str | None = None
    """Job name (auto-generated from prompt if not provided)."""

    def __post_init__(self) -> None:
        """Validate model supports image editing."""
        if self.model != ModelType.GROK_IMAGINE_IMAGE:
            raise ValueError(
                f"Model {self.model.value} does not support image editing. "
                f"Use {ModelType.GROK_IMAGINE_IMAGE.value}."
            )


# -----------------------------------------------------------------------------
# Video Generation Schemas
# -----------------------------------------------------------------------------


class GrokVideoRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request schema for Grok video generation (T2V)."""

    prompt: PromptStr
    """Text description of the video to generate."""

    model: ModelType = ModelType.GROK_IMAGINE_VIDEO
    """Model to use (must be grok-imagine-video)."""

    duration: VideoDuration = 5
    """Video duration in seconds (1-15)."""

    aspect_ratio: AspectRatio = AspectRatio.RATIO_16_9
    """Aspect ratio for the generated video."""

    resolution: VideoResolution = VideoResolution.RES_720P
    """Video resolution (480p or 720p)."""

    name: str | None = None
    """Job name (auto-generated from prompt if not provided)."""

    def __post_init__(self) -> None:
        """Validate model supports video generation."""
        if self.model != ModelType.GROK_IMAGINE_VIDEO:
            raise ValueError(
                f"Model {self.model.value} does not support video generation. "
                f"Use {ModelType.GROK_IMAGINE_VIDEO.value}."
            )


class GrokVideoFromImageRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request schema for Grok video from image (I2V)."""

    prompt: PromptStr
    """Text description of the video to generate."""

    input_image_id: UUID
    """ID of the uploaded image to animate."""

    model: ModelType = ModelType.GROK_IMAGINE_VIDEO
    """Model to use (must be grok-imagine-video)."""

    duration: VideoDuration = 5
    """Video duration in seconds (1-15)."""

    aspect_ratio: AspectRatio = AspectRatio.RATIO_16_9
    """Aspect ratio for the generated video."""

    resolution: VideoResolution = VideoResolution.RES_720P
    """Video resolution (480p or 720p)."""

    name: str | None = None
    """Job name (auto-generated from prompt if not provided)."""


class GrokVideoEditRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request schema for Grok video editing (V2V)."""

    prompt: PromptStr
    """Text description of the edit to perform."""

    input_video_url: str
    """URL of the video to edit (must be publicly accessible)."""

    model: ModelType = ModelType.GROK_IMAGINE_VIDEO
    """Model to use (must be grok-imagine-video)."""

    name: str | None = None
    """Job name (auto-generated from prompt if not provided)."""

    def __post_init__(self) -> None:
        """Validate model supports video editing."""
        if self.model != ModelType.GROK_IMAGINE_VIDEO:
            raise ValueError(
                f"Model {self.model.value} does not support video editing. "
                f"Use {ModelType.GROK_IMAGINE_VIDEO.value}."
            )


# -----------------------------------------------------------------------------
# Provider Info Schemas
# -----------------------------------------------------------------------------


class GrokModelInfo(msgspec.Struct, kw_only=True):
    """Information about a Grok model."""

    model: ModelType
    """Model identifier."""

    name: str
    """Human-readable model name."""

    description: str
    """Model description."""

    supports_t2i: bool = False
    """Supports text-to-image."""

    supports_i2i: bool = False
    """Supports image-to-image (editing)."""

    supports_t2v: bool = False
    """Supports text-to-video."""

    supports_i2v: bool = False
    """Supports image-to-video."""

    supports_v2v: bool = False
    """Supports video-to-video (editing)."""

    max_images: int = 1
    """Maximum images per request."""


# Model info registry
GROK_MODELS: list[GrokModelInfo] = [
    GrokModelInfo(
        model=ModelType.GROK_IMAGINE_IMAGE,
        name="Grok Imagine Image",
        description="xAI's flagship image generation model with editing support",
        supports_t2i=True,
        supports_i2i=True,
        max_images=10,
    ),
    GrokModelInfo(
        model=ModelType.GROK_2_IMAGE,
        name="Grok 2 Image",
        description="Photorealistic image generation (legacy model)",
        supports_t2i=True,
        supports_i2i=False,
        max_images=10,
    ),
    GrokModelInfo(
        model=ModelType.GROK_IMAGINE_VIDEO,
        name="Grok Imagine Video",
        description="Video generation from text, images, or video editing (up to 15 seconds)",
        supports_t2i=False,
        supports_t2v=True,
        supports_i2v=True,
        supports_v2v=True,
        max_images=1,
    ),
]


class GrokProviderInfo(msgspec.Struct, kw_only=True):
    """Information about the Grok provider."""

    provider: str = "grok"
    """Provider identifier."""

    name: str = "xAI Grok"
    """Human-readable provider name."""

    available: bool
    """Whether the provider is configured and available."""

    models: list[GrokModelInfo] = msgspec.field(default_factory=lambda: GROK_MODELS)
    """Available models."""
