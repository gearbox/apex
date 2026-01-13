"""API routes module."""

from .generation import (
    GenerationController,
    HealthController,
    ImageController,
    JobController,
)

__all__ = [
    "GenerationController",
    "HealthController",
    "ImageController",
    "JobController",
]
