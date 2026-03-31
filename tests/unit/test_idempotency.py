"""Unit tests for IdempotencyService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.idempotency import (
    IdempotencyConflictError,
    IdempotencyReplayResult,
    IdempotencyService,
)


class TestHashRequest:
    def test_deterministic(self) -> None:
        body = b'{"prompt": "a cat", "model": "grok-imagine-image"}'
        h1 = IdempotencyService.hash_request(body)
        h2 = IdempotencyService.hash_request(body)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_body_different_hash(self) -> None:
        h1 = IdempotencyService.hash_request(b'{"a": 1}')
        h2 = IdempotencyService.hash_request(b'{"a": 2}')
        assert h1 != h2


class TestCheck:
    async def test_new_key_returns_record_id(self) -> None:
        service = IdempotencyService(ttl_hours=24)
        record_id = uuid4()
        mock_record = MagicMock(id=record_id)

        with patch("src.api.services.idempotency.IdempotencyRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.try_acquire = AsyncMock(return_value=mock_record)

            result = await service.check(
                user_id=uuid4(),
                product_id="vex",
                idempotency_key="key-123",
                operation="generation",
                request_hash="abc123",
                session=AsyncMock(),
            )

        assert result == record_id

    async def test_completed_key_returns_replay(self) -> None:
        service = IdempotencyService(ttl_hours=24)
        existing = MagicMock(
            status="completed",
            request_hash="abc123",
            response_status_code=201,
            response_body={"job_id": "xxx"},
            resource_id=uuid4(),
        )

        with patch("src.api.services.idempotency.IdempotencyRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.try_acquire = AsyncMock(return_value=None)
            repo.get_existing = AsyncMock(return_value=existing)

            result = await service.check(
                user_id=uuid4(),
                product_id="vex",
                idempotency_key="key-123",
                operation="generation",
                request_hash="abc123",
                session=AsyncMock(),
            )

        assert isinstance(result, IdempotencyReplayResult)
        assert result.status_code == 201
        assert result.body == {"job_id": "xxx"}

    async def test_processing_key_raises_conflict(self) -> None:
        service = IdempotencyService(ttl_hours=24)
        existing = MagicMock(
            status="processing",
            request_hash="abc123",
        )

        with patch("src.api.services.idempotency.IdempotencyRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.try_acquire = AsyncMock(return_value=None)
            repo.get_existing = AsyncMock(return_value=existing)

            with pytest.raises(IdempotencyConflictError):
                await service.check(
                    user_id=uuid4(),
                    product_id="vex",
                    idempotency_key="key-123",
                    operation="generation",
                    request_hash="abc123",
                    session=AsyncMock(),
                )

    async def test_mismatched_hash_raises_conflict(self) -> None:
        service = IdempotencyService(ttl_hours=24)
        existing = MagicMock(
            status="completed",
            request_hash="different-hash",
        )

        with patch("src.api.services.idempotency.IdempotencyRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.try_acquire = AsyncMock(return_value=None)
            repo.get_existing = AsyncMock(return_value=existing)

            with pytest.raises(
                IdempotencyConflictError,
                match="different request body",
            ):
                await service.check(
                    user_id=uuid4(),
                    product_id="vex",
                    idempotency_key="key-123",
                    operation="generation",
                    request_hash="abc123",
                    session=AsyncMock(),
                )

    async def test_failed_key_allows_retry(self) -> None:
        service = IdempotencyService(ttl_hours=24)
        existing = MagicMock(status="failed", request_hash="abc123")
        new_record = MagicMock(id=uuid4())

        mock_session = AsyncMock()
        with patch("src.api.services.idempotency.IdempotencyRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.try_acquire = AsyncMock(side_effect=[None, new_record])
            repo.get_existing = AsyncMock(return_value=existing)

            result = await service.check(
                user_id=uuid4(),
                product_id="vex",
                idempotency_key="key-123",
                operation="generation",
                request_hash="abc123",
                session=mock_session,
            )

        assert result == new_record.id
        mock_session.delete.assert_called_once_with(existing)

    async def test_expired_key_allows_reacquire(self) -> None:
        """If key expired between insert attempt and lookup, re-acquire succeeds."""
        service = IdempotencyService(ttl_hours=24)
        new_record = MagicMock(id=uuid4())

        with patch("src.api.services.idempotency.IdempotencyRepository") as MockRepo:
            repo = MockRepo.return_value
            # First try_acquire returns None (conflict), get_existing returns None (expired),
            # second try_acquire succeeds
            repo.try_acquire = AsyncMock(side_effect=[None, new_record])
            repo.get_existing = AsyncMock(return_value=None)

            result = await service.check(
                user_id=uuid4(),
                product_id="vex",
                idempotency_key="key-123",
                operation="generation",
                request_hash="abc123",
                session=AsyncMock(),
            )

        assert result == new_record.id


class TestComplete:
    async def test_marks_completed(self) -> None:
        service = IdempotencyService()
        record_id = uuid4()

        with patch("src.api.services.idempotency.IdempotencyRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.mark_completed = AsyncMock()

            await service.complete(
                record_id,
                resource_id=uuid4(),
                response_status_code=201,
                response_body={"job_id": "xxx"},
                session=AsyncMock(),
            )

        repo.mark_completed.assert_called_once()


class TestFail:
    async def test_marks_failed(self) -> None:
        service = IdempotencyService()
        record_id = uuid4()

        with patch("src.api.services.idempotency.IdempotencyRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.mark_failed = AsyncMock()

            await service.fail(record_id, session=AsyncMock())

        repo.mark_failed.assert_called_once_with(record_id)


class TestCleanupExpired:
    async def test_delegates_to_repo(self) -> None:
        service = IdempotencyService()

        with patch("src.api.services.idempotency.IdempotencyRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.cleanup_expired = AsyncMock(return_value=5)

            count = await service.cleanup_expired(session=AsyncMock())

        assert count == 5
        repo.cleanup_expired.assert_called_once()
