"""Guard against the reserved-kwarg regression class fixed in fix/library-query-param.

`query` is a Litestar *reserved kwarg*: a handler argument literally named
`query` (or `body`/`headers`/`cookies`/`state`/`scope`/`socket`) receives a
framework-internal object (e.g. the raw query-params `MultiDict`) instead of
being resolved against the declared annotation, and — critically — such
params are silently excluded from the generated OpenAPI schema. This is
exactly what broke ``GET /v1/library`` on master: `query: str | None` bound
the whole `MultiDict`, which failed msgspec validation on every request,
while never appearing in the OpenAPI docs the frontend generates from.

This iterates every route handler actually registered by ``create_app()``
(the real app factory, not a hand-maintained controller list, so newly added
controllers are covered automatically) and asserts none of them declare a
parameter name in Litestar's reserved set. See:
https://docs.litestar.dev/latest/usage/requests.html#reserved-keyword-arguments
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from litestar.routes import HTTPRoute

from src.api.app import create_app

pytestmark = pytest.mark.unit

# Per Litestar's "Reserved keyword arguments" docs. A handler argument with
# one of these names receives a framework-internal object regardless of its
# type annotation, and is dropped from the OpenAPI schema.
_RESERVED_KWARGS: frozenset[str] = frozenset(
    {"query", "body", "headers", "cookies", "state", "scope", "socket"}
)


def _app_defined_handlers() -> list[Any]:
    """Every unique route handler registered by create_app(), excluding
    Litestar's own auto-generated handlers (e.g. the synthesized OPTIONS
    handler), which are framework internals we don't control.
    """
    app = create_app()
    unique_by_id: dict[int, Any] = {}
    for route in app.routes:
        if not isinstance(route, HTTPRoute):
            continue
        for handler in route.route_handlers:
            module = getattr(handler.fn, "__module__", "") or ""
            if module.startswith("litestar"):
                continue
            unique_by_id[id(handler)] = handler
    return list(unique_by_id.values())


class TestNoReservedKwargCollisions:
    def test_no_handler_uses_a_reserved_parameter_name(self) -> None:
        handlers = _app_defined_handlers()
        assert handlers, "expected at least one app-defined route handler"

        violations = []
        for handler in handlers:
            params = set(inspect.signature(handler.fn).parameters.keys())
            hits = _RESERVED_KWARGS & params
            if hits:
                violations.append((getattr(handler.fn, "__qualname__", handler.fn), hits))

        assert not violations, (
            "handler(s) declare a Litestar-reserved parameter name — this "
            "silently drops the param from OpenAPI and passes a framework "
            f"object instead of the declared type: {violations}"
        )
