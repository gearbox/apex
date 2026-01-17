"""Dependency injection providers for Litestar."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from litestar.di import Provide
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.services.comfyui_client import ComfyUIClient
from src.api.services.job_manager import JobManager
from src.api.services.storage import R2StorageService, R2StorageSettings
from src.api.services.user_content import UserContentService
from src.api.services.workflow_service import WorkflowService
from src.core.config import Settings, get_settings
from src.db import DatabaseManager, init_db

logger = logging.getLogger(__name__)

# Global singleton instances (created at app startup)
_comfyui_client: ComfyUIClient | None = None
_job_manager: JobManager | None = None
_workflow_service: WorkflowService | None = None
_r2_storage: R2StorageService | None = None
_db_manager: DatabaseManager | None = None


# -----------------------------------------------------------------------------
# ComfyUI dependencies
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Storage dependencies
# -----------------------------------------------------------------------------


async def get_r2_storage() -> R2StorageService:
    """Provide R2 storage service instance.

    Returns:
        Singleton R2 storage service.

    Raises:
        RuntimeError: If storage not initialized.
    """
    if _r2_storage is None:
        raise RuntimeError("R2 storage not initialized")
    return _r2_storage


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide database session for request scope.

    Yields:
        Database session that auto-commits on success.
    """
    if _db_manager is None:
        raise RuntimeError("Database not initialized")

    async with _db_manager.session() as session:
        yield session


async def get_user_content(
    r2_storage: R2StorageService,
    session: AsyncSession,
    settings: Settings,
) -> UserContentService:
    """Provide user content service.

    Creates a new instance per request with the injected session.

    Args:
        r2_storage: R2 storage service.
        session: Database session.
        settings: Application settings.

    Returns:
        Configured UserContentService.
    """
    return UserContentService(
        storage=r2_storage,
        session=session,
        retention_days=settings.retention_days,
    )


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------


def provide_settings() -> Settings:
    """Provide settings instance.

    Returns:
        Application settings.
    """
    return get_settings()


# -----------------------------------------------------------------------------
# Lifecycle management
# -----------------------------------------------------------------------------


async def init_services(settings: Settings, base_path: Path | None = None) -> None:
    """Initialize all service singletons.

    Called during application startup.

    Args:
        settings: Application settings.
        base_path: Base path for workflow files.
    """
    global _comfyui_client, _job_manager, _workflow_service, _r2_storage, _db_manager

    # Initialize ComfyUI client
    _comfyui_client = ComfyUIClient(settings)
    await _comfyui_client.connect()
    logger.info(f"Connected to ComfyUI at {settings.comfyui_base_url}")

    # Initialize job manager with client
    _job_manager = JobManager(_comfyui_client)

    # Initialize workflow service
    _workflow_service = WorkflowService(base_path=base_path)

    # Initialize database
    _db_manager = init_db(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        echo=settings.db_echo,
    )
    logger.info("Database connection pool initialized")

    # Initialize R2 storage (if configured)
    if settings.r2_configured:
        r2_settings = R2StorageSettings(
            account_id=settings.r2_account_id,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            bucket_name=settings.r2_bucket_name,
            public_url_base=settings.r2_public_url_base,
            retention_days=settings.retention_days,
        )
        _r2_storage = R2StorageService(r2_settings)
        logger.info(f"R2 storage initialized for bucket: {settings.r2_bucket_name}")
    else:
        logger.warning("R2 storage not configured - storage endpoints will be unavailable")


async def shutdown_services() -> None:
    """Cleanup service resources.

    Called during application shutdown.
    """
    global _comfyui_client, _job_manager, _workflow_service, _r2_storage, _db_manager

    if _comfyui_client is not None:
        await _comfyui_client.close()
        _comfyui_client = None
        logger.info("ComfyUI client closed")

    if _r2_storage is not None:
        await _r2_storage.close()
        _r2_storage = None
        logger.info("R2 storage closed")

    if _db_manager is not None:
        await _db_manager.close()
        _db_manager = None
        logger.info("Database connections closed")

    _job_manager = None
    _workflow_service = None


# Dependency providers for Litestar
dependencies = {
    # Core services
    "comfyui_client": Provide(get_comfyui_client),
    "job_manager": Provide(get_job_manager),
    "workflow_service": Provide(get_workflow_service),
    "settings": Provide(provide_settings, sync_to_thread=False),
    # Storage services
    "r2_storage": Provide(get_r2_storage),
    "session": Provide(get_db_session),
    "user_content": Provide(get_user_content),
}
