"""Database session management for async SQLAlchemy."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class DatabaseManager:
    """Manages async database connections and sessions.

    Provides connection pooling and session factory for async operations.
    Should be initialized once at application startup.
    """

    def __init__(
        self,
        database_url: str,
        *,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 1800,
        echo: bool = False,
    ) -> None:
        """Initialize database manager.

        Args:
            database_url: PostgreSQL connection URL (async format).
                         e.g., "postgresql+asyncpg://user:pass@host/db"
            pool_size: Number of connections to maintain in pool.
            max_overflow: Maximum overflow connections above pool_size.
            pool_timeout: Seconds to wait for available connection.
            pool_recycle: Seconds before recycling connections.
            echo: Whether to log SQL statements.
        """
        self._engine: AsyncEngine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            echo=echo,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        """Get the database engine."""
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with automatic cleanup.

        Yields:
            AsyncSession for database operations.

        Example:
            async with db.session() as session:
                result = await session.execute(select(User))
        """
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Dependency provider for Litestar DI.

        Yields:
            AsyncSession for database operations.
        """
        async with self.session() as session:
            yield session

    async def health_check(self) -> bool:
        """Check if database is accessible.

        Returns:
            True if connection succeeds, False otherwise.
        """
        try:
            async with self._engine.connect() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the database engine and all connections.

        Should be called during application shutdown.
        """
        await self._engine.dispose()


# Global instance (initialized by application)
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance.

    Returns:
        Initialized DatabaseManager.

    Raises:
        RuntimeError: If database not initialized.
    """
    if _db_manager is None:
        raise RuntimeError("Database not initialized")
    return _db_manager


def init_db(
    database_url: str,
    **kwargs,
) -> DatabaseManager:
    """Initialize the global database manager.

    Args:
        database_url: PostgreSQL connection URL.
        **kwargs: Additional arguments for DatabaseManager.

    Returns:
        Initialized DatabaseManager.
    """
    global _db_manager
    _db_manager = DatabaseManager(database_url, **kwargs)
    return _db_manager


async def close_db() -> None:
    """Close the global database manager."""
    global _db_manager
    if _db_manager is not None:
        await _db_manager.close()
        _db_manager = None
