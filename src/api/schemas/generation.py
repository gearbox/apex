"""API schemas using msgspec for high-performance serialization."""

from __future__ import annotations

import random
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

import msgspec


class ModelType(str, Enum):
    """Available model types."""

    AISHA = "aisha"
    # Future models:
    # SEEDREAM = "seedream"
    # Z_IMAGE = "z-image"


class GenerationType(str, Enum):
    """Generation type - text-to-image or image-to-image."""

    T2I = "t2i"
    I2I = "i2i"


class AspectRatio(str, Enum):
    """Supported aspect ratios."""

    RATIO_1_1 = "1:1"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_2_3 = "2:3"
    RATIO_3_2 = "3:2"
    RATIO_21_9 = "21:9"

    def calculate_width(self, height: int) -> int:
        """Calculate width from height based on aspect ratio.

        Returns width rounded to nearest multiple of 8 for latent space compatibility.
        """
        ratio_map = {
            "1:1": 1.0,
            "4:3": 4 / 3,
            "3:4": 3 / 4,
            "16:9": 16 / 9,
            "9:16": 9 / 16,
            "2:3": 2 / 3,
            "3:2": 3 / 2,
            "21:9": 21 / 9,
        }
        ratio = ratio_map[self.value]
        width = int(height * ratio)
        # Round to nearest multiple of 8
        return (width + 4) // 8 * 8


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Type aliases for validation constraints (used by Litestar's OpenAPI generation)
PromptStr = Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
NegativePromptStr = Annotated[str, msgspec.Meta(max_length=2048)]
Height = Annotated[int, msgspec.Meta(ge=256, le=2048)]
MaxImages = Annotated[int, msgspec.Meta(ge=1, le=4)]
Steps = Annotated[int, msgspec.Meta(ge=1, le=20)]
Progress = Annotated[float, msgspec.Meta(ge=0.0, le=100.0)]


# Default negative prompt constant
DEFAULT_NEGATIVE_PROMPT = (
    "waxy texture, blurry face, over-sharpening, unrealistic symmetry, "
    "flat lighting, low detail skin, extra fingers, distorted anatomy, deformed"
)


class GenerationRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request schema for image generation."""

    prompt: PromptStr
    name: str | None = None
    negative_prompt: NegativePromptStr = DEFAULT_NEGATIVE_PROMPT
    height: Height = 1024
    aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1
    model_type: ModelType = ModelType.AISHA
    generation_type: GenerationType = GenerationType.T2I
    max_images: MaxImages = 1
    seed: int | None = None
    steps: Steps = 12

    def __post_init__(self) -> None:
        """Generate name from prompt and seed if not provided."""
        # Generate name from prompt if not provided
        if self.name is None:
            truncated = self.prompt[:50].strip()
            if len(self.prompt) > 50:
                truncated += "..."
            self.name = truncated

        # Generate random seed if not provided
        if self.seed is None:
            self.seed = random.randint(0, 2**31 - 1)

    def get_calculated_width(self) -> int:
        """Calculate width from height and aspect ratio."""
        return self.aspect_ratio.calculate_width(self.height)


class JobResponse(msgspec.Struct, kw_only=True):
    """Response schema for job creation."""

    job_id: str
    status: JobStatus
    name: str
    created_at: datetime
    message: str | None = None


class JobStatusResponse(msgspec.Struct, kw_only=True):
    """Response schema for job status query."""

    job_id: str
    status: JobStatus
    name: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: Progress = 0.0
    images: list[str] = msgspec.field(default_factory=list)
    error: str | None = None


class ImageUploadResponse(msgspec.Struct, kw_only=True):
    """Response schema for image upload."""

    filename: str
    subfolder: str = ""
    type: Literal["input", "temp"] = "input"


class HealthResponse(msgspec.Struct, kw_only=True):
    """Health check response."""

    status: Literal["healthy", "unhealthy"]
    comfyui_connected: bool
    version: str = "0.1.0"
