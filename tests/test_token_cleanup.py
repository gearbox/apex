from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.repositories.auth_tokens import AuthTokenRepository
from src.db.repositories.user import UserRepository
from src.db.session import DatabaseManager
from src.workers.token_cleanup import TokenCleanupWorker


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock(spec=AsyncSession)

    # Mock the execute result for delete queries
    class MockScalars:
        def all(self) -> list[Any]:
            return [MagicMock(), MagicMock()]

    class MockResult:
        rowcount = 2

        def scalars(self) -> "MockScalars":
            return MockScalars()

    session.execute.return_value = MockResult()
    return session


@pytest.fixture
def auth_token_repo(mock_session: AsyncMock) -> AuthTokenRepository:
    return AuthTokenRepository(mock_session)


@pytest.fixture
def user_repo(mock_session: AsyncMock) -> UserRepository:
    return UserRepository(mock_session)


@pytest.mark.asyncio
async def test_cleanup_expired_email_verification_tokens_executes_delete(
    auth_token_repo: AuthTokenRepository,
    mock_session: AsyncMock,
) -> None:
    count = await auth_token_repo.cleanup_expired_email_verification_tokens()

    # Verify execute and flush were called
    assert count == 2
    mock_session.execute.assert_called_once()
    assert mock_session.delete.call_count == 2
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_expired_password_reset_tokens_executes_delete(
    auth_token_repo: AuthTokenRepository,
    mock_session: AsyncMock,
) -> None:
    count = await auth_token_repo.cleanup_expired_password_reset_tokens()

    assert count == 2
    mock_session.execute.assert_called_once()
    assert mock_session.delete.call_count == 2
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_refresh_tokens_executes_delete(
    user_repo: UserRepository,
    mock_session: AsyncMock,
) -> None:
    count = await user_repo.cleanup_expired_tokens()

    assert count == 2
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_worker_runs_cleanup_on_interval() -> None:
    db_manager_mock = MagicMock(spec=DatabaseManager)

    # We need an async context manager for db_manager.session()
    class AsyncContextManagerMock:
        async def __aenter__(self) -> AsyncMock:
            return AsyncMock(spec=AsyncSession)
        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    db_manager_mock.session.return_value = AsyncContextManagerMock()

    worker = TokenCleanupWorker(db_manager=db_manager_mock, interval=3600)
    worker._running = True  # Avoid starting actual asyncio task that spins immediately

    with patch("src.workers.token_cleanup.UserRepository") as user_repo_cls, \
         patch("src.workers.token_cleanup.AuthTokenRepository") as auth_token_repo_cls, \
         patch("src.workers.token_cleanup.logger") as logger_mock:

        user_repo_mock = AsyncMock()
        user_repo_mock.cleanup_expired_tokens.return_value = 5
        user_repo_cls.return_value = user_repo_mock

        auth_repo_mock = AsyncMock()
        auth_repo_mock.cleanup_expired_email_verification_tokens.return_value = 3
        auth_repo_mock.cleanup_expired_password_reset_tokens.return_value = 2
        auth_token_repo_cls.return_value = auth_repo_mock

        await worker._run_once()

        # Verify repository methods called
        user_repo_mock.cleanup_expired_tokens.assert_called_once()
        auth_repo_mock.cleanup_expired_email_verification_tokens.assert_called_once()
        auth_repo_mock.cleanup_expired_password_reset_tokens.assert_called_once()

        # Verify logger struct
        logger_mock.info.assert_called_with(
            "token_cleanup",
            event="token_cleanup",
            refresh=5,
            email_verification=3,
            password_reset=2,
            duration_ms=pytest.approx(0, abs=100) # approximate time check
        )


@pytest.mark.asyncio
async def test_worker_continues_on_exception() -> None:
    db_manager_mock = MagicMock(spec=DatabaseManager)

    class AsyncContextManagerMock:
        async def __aenter__(self) -> AsyncMock:
            return AsyncMock(spec=AsyncSession)
        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    db_manager_mock.session.return_value = AsyncContextManagerMock()

    worker = TokenCleanupWorker(db_manager=db_manager_mock, interval=3600)

    with patch("src.workers.token_cleanup.UserRepository") as user_repo_cls, \
         patch("src.workers.token_cleanup.logger") as logger_mock:

        user_repo_mock = AsyncMock()
        user_repo_mock.cleanup_expired_tokens.side_effect = Exception("Database failure")
        user_repo_cls.return_value = user_repo_mock

        # Call it once, it should catch the exception and log it instead of raising
        await worker._run_once()

        logger_mock.exception.assert_called_once()
        args, kwargs = logger_mock.exception.call_args
        assert args[0] == "token_cleanup_worker.error"
        assert kwargs["error"] == "Database failure"
