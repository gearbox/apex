"""Tests for the global per-model rate limiter."""

from __future__ import annotations

import pytest
from limits.storage import MemoryStorage

from src.api.middleware.rate_limit import init_rate_limiter
from src.api.services.generation.rate_limiter import (
    ModelRateLimiter,
    RateLimitExceededError,
)
from src.core.config import Settings
from src.core.enums import ModelType


@pytest.fixture(autouse=True)
def _setup_memory_storage() -> None:
    """Ensure MemoryStorage is used and cleared for each test."""
    settings = Settings(redis_url=None)
    init_rate_limiter(settings)
    from src.api.middleware.rate_limit import get_rate_limiter_storage

    storage = get_rate_limiter_storage()
    if isinstance(storage, MemoryStorage):
        storage.storage.clear()


class TestModelRateLimiter:
    def test_no_limit_configured_always_passes(self) -> None:
        """Models without rate_limit in metadata should always pass."""
        limiter = ModelRateLimiter()
        # GROK_IMAGINE_IMAGE has rate_limit=None
        for _ in range(100):
            limiter.check(ModelType.GROK_IMAGINE_IMAGE)  # Should not raise

    def test_under_limit_succeeds(self) -> None:
        """Requests under the limit should succeed."""
        limiter = ModelRateLimiter()
        # GROK_IMAGINE_VIDEO has rate_limit=RateLimitMeta(10, 60)
        for _ in range(10):
            limiter.check(ModelType.GROK_IMAGINE_VIDEO)

    def test_over_limit_raises(self) -> None:
        """Request exceeding the limit should raise RateLimitExceededError."""
        limiter = ModelRateLimiter()
        # Exhaust the 10-request limit
        for _ in range(10):
            limiter.check(ModelType.GROK_IMAGINE_VIDEO)

        with pytest.raises(RateLimitExceededError) as exc_info:
            limiter.check(ModelType.GROK_IMAGINE_VIDEO)

        assert exc_info.value.model == ModelType.GROK_IMAGINE_VIDEO
        assert exc_info.value.retry_after >= 0

    def test_different_models_independent(self) -> None:
        """Rate limits are per-model, not shared."""
        limiter = ModelRateLimiter()
        # Exhaust video limit
        for _ in range(10):
            limiter.check(ModelType.GROK_IMAGINE_VIDEO)

        # Image model (no limit) should still work
        limiter.check(ModelType.GROK_IMAGINE_IMAGE)

    def test_aisha_image_no_limit(self) -> None:
        """Aisha image has no rate limit configured."""
        limiter = ModelRateLimiter()
        for _ in range(50):
            limiter.check(ModelType.AISHA_IMAGE)
