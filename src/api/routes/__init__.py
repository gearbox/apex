"""API routes module."""

from .auth import AuthController
from .generation import (
    HealthController,
    ImageController,
)
from .storage import StorageController
from .user import UserController

__all__ = [
    "AuthController",
    "HealthController",
    "ImageController",
    "StorageController",
    "UserController",
]
