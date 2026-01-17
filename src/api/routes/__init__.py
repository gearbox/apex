"""API routes module."""

from .generation import (
    GenerationController,
    HealthController,
    ImageController,
    JobController,
)
from .storage import StorageController

__all__ = [
    "GenerationController",
    "HealthController",
    "ImageController",
    "JobController",
    "StorageController",
]
