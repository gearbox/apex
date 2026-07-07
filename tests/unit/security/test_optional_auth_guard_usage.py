"""Usage audit for ``optional_auth_guard``.

``optional_auth_guard`` degrades a product-mismatched token to anonymous
instead of raising (see its docstring in ``src.api.security.guards``). That
is only safe on routes whose response is identical-or-safely-narrower for an
anonymous caller. This test pins the exact set of routes allowed to use the
guard so that attaching it to a new route forces a conscious review — the
allowlist below must be updated deliberately, not silently.
"""

from __future__ import annotations

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-optional-auth-guard-audit-key-32-bytes")

from src.api.app import create_app  # noqa: E402
from src.api.security import optional_auth_guard  # noqa: E402

# Routes explicitly reviewed and confirmed safe for anonymous-degrading auth.
# Discovered via: grep -rn "optional_auth_guard" src/api/routes/
_ALLOWED_OPTIONAL_AUTH_GUARD_ROUTES: frozenset[str] = frozenset(
    {
        "/v1/providers",  # public catalog; auth only adds user_context/session_state
    }
)


def _routes_using_optional_auth_guard() -> frozenset[str]:
    app = create_app()
    found: set[str] = set()
    for route in app.routes:
        for handler in getattr(route, "route_handlers", []):
            if optional_auth_guard in handler.resolve_guards():
                found.add(route.path)
    return frozenset(found)


class TestOptionalAuthGuardUsage:
    def test_usage_matches_reviewed_allowlist(self) -> None:
        actual = _routes_using_optional_auth_guard()
        assert actual == _ALLOWED_OPTIONAL_AUTH_GUARD_ROUTES, (
            "optional_auth_guard usage changed. If you added it to a new route, "
            "you must consciously review the route against the guard's safety "
            "contract (see its docstring) and add it to "
            "_ALLOWED_OPTIONAL_AUTH_GUARD_ROUTES in this test. "
            f"actual={sorted(actual)} allowlist={sorted(_ALLOWED_OPTIONAL_AUTH_GUARD_ROUTES)}"
        )
