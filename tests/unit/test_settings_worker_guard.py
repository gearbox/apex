"""Tests for the asgi_workers > 1 without Redis startup guard."""

from __future__ import annotations

import pytest

from src.core.config import Settings
from src.core.enums import WorkerMode

_JWT_SECRET = "test-secret-key-that-is-definitely-long-enough-32bytes"


class TestMultiWorkerWithoutRedisRejected:
    def test_multi_worker_without_redis_rejected(self) -> None:
        with pytest.raises(ValueError, match="asgi_workers > 1 requires REDIS_URL"):
            Settings(
                jwt_secret_key=_JWT_SECRET,
                asgi_workers=2,
                worker_mode=WorkerMode.all,
                redis_url=None,
            )


class TestMultiWorkerWithRedisAccepted:
    def test_multi_worker_with_redis_accepted(self) -> None:
        settings = Settings(
            jwt_secret_key=_JWT_SECRET,
            asgi_workers=2,
            worker_mode=WorkerMode.all,
            redis_url="redis://localhost:6379/0",
        )
        assert settings.asgi_workers == 2


class TestApiOnlyWithoutRedisAccepted:
    def test_api_only_without_redis_accepted(self) -> None:
        settings = Settings(
            jwt_secret_key=_JWT_SECRET,
            asgi_workers=4,
            worker_mode=WorkerMode.api_only,
            redis_url=None,
        )
        assert settings.worker_mode is WorkerMode.api_only

    def test_single_worker_without_redis_accepted(self) -> None:
        """asgi_workers=1 (the default) never needs Redis — no multi-process race."""
        settings = Settings(
            jwt_secret_key=_JWT_SECRET,
            asgi_workers=1,
            worker_mode=WorkerMode.all,
            redis_url=None,
        )
        assert settings.asgi_workers == 1
