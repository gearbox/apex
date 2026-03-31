"""API routes module."""

from .auth import AuthController
from .generation import ImageController
from .health import AdminHealthController, HealthController
from .storage import StorageController
from .user import UserController

__all__ = [
    "AdminHealthController",
    "AuthController",
    "HealthController",
    "ImageController",
    "StorageController",
    "UserController",
]
