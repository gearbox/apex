"""Dependency injection providers for Litestar."""

from pathlib import Path

from litestar.di import Provide

from src.api.services.comfyui_client import ComfyUIClient
from src.api.services.job_manager import JobManager
from src.api.services.workflow_service import WorkflowService
from src.core.config import Settings, get_settings

# Global singleton instances (created at app startup)
_comfyui_client: ComfyUIClient | None = None
_job_manager: JobManager | None = None
_workflow_service: WorkflowService | None = None


async def get_comfyui_client() -> ComfyUIClient:
    """Provide ComfyUI client instance.

    Returns:
        Singleton ComfyUI client.

    Raises:
        RuntimeError: If client not initialized.
    """
    if _comfyui_client is None:
        raise RuntimeError("ComfyUI client not initialized")
    return _comfyui_client


async def get_job_manager() -> JobManager:
    """Provide job manager instance.

    Returns:
        Singleton job manager.

    Raises:
        RuntimeError: If manager not initialized.
    """
    if _job_manager is None:
        raise RuntimeError("Job manager not initialized")
    return _job_manager


async def get_workflow_service() -> WorkflowService:
    """Provide workflow service instance.

    Returns:
        Singleton workflow service.

    Raises:
        RuntimeError: If service not initialized.
    """
    if _workflow_service is None:
        raise RuntimeError("Workflow service not initialized")
    return _workflow_service


def provide_settings() -> Settings:
    """Provide settings instance.

    Returns:
        Application settings.
    """
    return get_settings()


async def init_services(settings: Settings, base_path: Path | None = None) -> None:
    """Initialize all service singletons.

    Called during application startup.

    Args:
        settings: Application settings.
        base_path: Base path for workflow files.
    """
    global _comfyui_client, _job_manager, _workflow_service

    # Initialize ComfyUI client
    _comfyui_client = ComfyUIClient(settings)
    await _comfyui_client.connect()

    # Initialize job manager with client
    _job_manager = JobManager(_comfyui_client)

    # Initialize workflow service
    _workflow_service = WorkflowService(base_path=base_path)


async def shutdown_services() -> None:
    """Cleanup service resources.

    Called during application shutdown.
    """
    global _comfyui_client, _job_manager, _workflow_service

    if _comfyui_client is not None:
        await _comfyui_client.close()
        _comfyui_client = None

    _job_manager = None
    _workflow_service = None


# Dependency providers for Litestar
dependencies = {
    "comfyui_client": Provide(get_comfyui_client),
    "job_manager": Provide(get_job_manager),
    "workflow_service": Provide(get_workflow_service),
    "settings": Provide(provide_settings, sync_to_thread=False),
}
