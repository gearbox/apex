"""Alibaba Model Studio (DashScope) integration: REST client, failure classifier, Wan service."""

from src.api.services.model_studio.client import ModelStudioClient
from src.api.services.model_studio.errors import ModelStudioAPIError, ModelStudioTransientError
from src.api.services.model_studio.failure_classifier import ModelStudioFailureClassifier
from src.api.services.model_studio.types import (
    ModelStudioTask,
    ModelStudioTaskRef,
    ModelStudioTaskStatus,
    ModelStudioVideoUsage,
)

__all__ = [
    "ModelStudioAPIError",
    "ModelStudioClient",
    "ModelStudioFailureClassifier",
    "ModelStudioTask",
    "ModelStudioTaskRef",
    "ModelStudioTaskStatus",
    "ModelStudioTransientError",
    "ModelStudioVideoUsage",
]
