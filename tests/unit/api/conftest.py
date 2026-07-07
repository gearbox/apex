"""Fixtures shared by tests/unit/api/ — tests here call create_app()."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _suppress_configure_logging() -> Generator[None]:
    """Prevent create_app()/lifespan() from reconfiguring structlog.

    configure_logging() replaces the global structlog processors list with a
    brand-new list object on every call. structlog's cache_logger_on_first_use
    binds a module-level ``logger = structlog.get_logger(__name__)`` to
    whatever list is live the first time it actually logs something — and
    that binding is permanent for the life of the process, not re-checked
    later. If a logger (e.g. the request-logging middleware, or a background
    worker started by the real app lifespan) gets used for the first time in
    the whole test session while one of *our* transient lists is live, it
    stays bound to that dead list forever — silently breaking
    structlog.testing.capture_logs() in unrelated test modules that run
    later and expect to intercept it. Restoring config afterward (see
    tests/unit/test_logging.py::_reset_structlog) doesn't fix an
    already-cached proxy. The only reliable fix is to never let create_app()
    touch the global config in the first place — we don't test logging
    configuration here, so there's nothing lost by suppressing it.
    """
    with patch("src.api.app.configure_logging"):
        yield
