"""Pydantic schemas for generation API."""

import random
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class ModelType(str, Enum):
    """Available model types."""

    AISHA = "aisha"
    # Future models:
    # SEEDREAM = "seedream"
    # Z_IMAGE = "z-image"


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


class GenerationRequest(BaseModel):
    """Request schema for image generation."""

    name: str | None = Field(
        default=None,
        description="Name of the text to image task (auto-generated if not provided)",
        examples=["My Generated Image"],
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Text prompt for image generation",
        examples=["A beautiful sunset over mountains"],
    )
    negative_prompt: str = Field(
        default="waxy texture, blurry face, over-sharpening, unrealistic symmetry, "
        "flat lighting, low detail skin, extra fingers, distorted anatomy, deformed",
        max_length=2048,
        description="Text negative prompt for image generation",
    )
    height: Annotated[int, Field(ge=256, le=2048)] = Field(
        default=1024,
        description="Image height in pixels",
        examples=[1024],
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1,
        description="Aspect ratio for the generated image",
        examples=["1:1", "16:9"],
    )
    model_type: ModelType = Field(
        default=ModelType.AISHA,
        description="Model type to use for generation",
        examples=["aisha"],
    )
    max_images: Annotated[int, Field(ge=1, le=4)] = Field(
        default=1,
        description="Number of images to generate in batch",
        examples=[1],
    )
    seed: int | None = Field(
        default=None,
        description="Seed for reproducible generation (auto-generated if not provided)",
        examples=[74137893],
    )
    steps: Annotated[int, Field(ge=1, le=20)] = Field(
        default=12,
        description="Number of generation steps",
        examples=[4, 12],
    )

    @model_validator(mode="after")
    def generate_name_if_none(self) -> "GenerationRequest":
        """Generate task name from prompt if not provided."""
        if self.name is None:
            # Truncate prompt for name
            truncated = self.prompt[:50].strip()
            if len(self.prompt) > 50:
                truncated += "..."
            self.name = truncated

        # Generate random seed if not provided
        if self.seed is None:
            self.seed = random.randint(0, 2**31 - 1)

        return self

    @property
    def calculated_width(self) -> int:
        """Calculate width from height and aspect ratio."""
        return self.aspect_ratio.calculate_width(self.height)


class JobResponse(BaseModel):
    """Response schema for job creation."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    name: str = Field(..., description="Task name")
    created_at: datetime = Field(..., description="Job creation timestamp")
    message: str | None = Field(default=None, description="Status message")


class JobStatusResponse(BaseModel):
    """Response schema for job status query."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    name: str = Field(..., description="Task name")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: datetime | None = Field(default=None, description="Processing start time")
    completed_at: datetime | None = Field(default=None, description="Completion timestamp")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")
    images: list[str] = Field(default_factory=list, description="Generated image URLs")
    error: str | None = Field(default=None, description="Error message if failed")


class ImageUploadResponse(BaseModel):
    """Response schema for image upload."""

    filename: str = Field(..., description="Uploaded filename on ComfyUI server")
    subfolder: str = Field(default="", description="Subfolder path")
    type: Literal["input", "temp"] = Field(default="input", description="Image type")


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"] = Field(..., description="Service health status")
    comfyui_connected: bool = Field(..., description="ComfyUI connection status")
    version: str = Field(default="0.1.0", description="API version")
