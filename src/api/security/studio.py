"""Feature gate for ``/v1/studio``.

The studio is invisible, not forbidden, when it is off: the guard answers 404 (never 403)
and runs **before** ``auth_guard`` so even an unauthenticated caller sees 404.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from litestar.exceptions import NotFoundException

from src.core.config import get_settings

if TYPE_CHECKING:
    from litestar.connection import ASGIConnection
    from litestar.handlers import BaseRouteHandler

    from src.core.product import ProductConfig

__all__ = ["STUDIO_FEATURE", "studio_feature_guard", "studio_runs_guard"]

STUDIO_FEATURE: Final = "studio"


async def studio_feature_guard(
    connection: ASGIConnection[Any, Any, Any, Any], _handler: BaseRouteHandler
) -> None:
    """Raise 404 unless ``STUDIO_ENABLED`` is on and the product has the ``studio`` feature.

    Args:
        connection: ASGI connection (``product_config`` is set by ``ProductMiddleware``).
        _handler: Route handler (unused).

    Raises:
        NotFoundException: The studio is disabled globally or for this product.
        RuntimeError: ``ProductMiddleware`` did not run (a wiring bug, not a client error).
    """
    if not get_settings().studio_enabled:
        raise NotFoundException()
    product_config: ProductConfig | None = connection.state.get("product_config")
    if product_config is None:
        raise RuntimeError("Product scope missing — ProductMiddleware not applied")
    if not product_config.has_feature(STUDIO_FEATURE):
        raise NotFoundException()


async def studio_runs_guard(
    _connection: ASGIConnection[Any, Any, Any, Any], _handler: BaseRouteHandler
) -> None:
    """Raise 404 unless ``STUDIO_RUNS_ENABLED`` is on (the separate kill switch for LLM spend).

    Runs after :func:`studio_feature_guard` and before ``auth_guard``, so a disabled run feature
    is as invisible as a disabled studio.

    Args:
        _connection: ASGI connection (unused).
        _handler: Route handler (unused).

    Raises:
        NotFoundException: Planner runs are disabled.
    """
    if not get_settings().studio_runs_enabled:
        raise NotFoundException()
