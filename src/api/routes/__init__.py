"""API routes module."""

from .auth import AuthController
from .generation import (
    GenerationController,
    HealthController,
    ImageController,
)
from .storage import StorageController
from .user import UserController

__all__ = [
    "AuthController",
    "GenerationController",
    "HealthController",
    "ImageController",
    "StorageController",
    "UserController",
]
