"""Global per-model rate limiter.

Enforces sliding-window rate limits per model across all users.
Uses the same `limits` library backend as the HTTP rate limit middleware.
"""

from __future__ import annotations

import time

import structlog
from limits import parse
from limits.strategies import MovingWindowRateLimiter

from src.api.middleware.rate_limit import get_rate_limiter_storage
from src.core.enums import ModelType
from src.core.model_registry import get_model_meta

logger = structlog.get_logger(__name__)


class RateLimitExceededError(Exception):
    """Raised when a model's global rate limit is exceeded."""

    def __init__(self, model: ModelType, retry_after: int) -> None:
        self.model = model
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for model '{model.value}'. Try again in {retry_after} seconds."
        )


class ModelRateLimiter:
    """Sliding-window rate limiter for global per-model request throttling.

    Each model_key gets a single shared counter (not per-user).
    Uses the same storage backend (Memory or Redis) as the HTTP middleware.
    """

    def check(self, model: ModelType) -> None:
        """Check rate limit and record the request atomically.

        Args:
            model: The model being requested.

        Raises:
            RateLimitExceededError: If the model's global rate limit is exceeded.
        """
        meta = get_model_meta(model)
        if meta.rate_limit is None:
            return  # No limit configured for this model

        storage = get_rate_limiter_storage()
        limiter = MovingWindowRateLimiter(storage)

        # Build limits-compatible rate string: "10/60 second"
        limit_str = f"{meta.rate_limit.max_requests}/{meta.rate_limit.window_seconds} second"
        limit_item = parse(limit_str)

        key = f"model_rate_limit:{model.value}"

        if not limiter.hit(limit_item, key):
            stats = limiter.get_window_stats(limit_item, key)
            retry_after = max(0, int(stats.reset_time) - int(time.time()))
            logger.warning(
                "model_rate_limit.exceeded",
                model=model.value,
                retry_after=retry_after,
            )
            raise RateLimitExceededError(model=model, retry_after=retry_after)

        logger.debug("model_rate_limit.ok", model=model.value)
