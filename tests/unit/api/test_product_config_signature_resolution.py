"""Regression tests for the TC-refactor login outage (commit 2873e8e, PR #111).

`src/core/product.py` briefly moved `from src.core.enums import ModelType,
Product` into a `TYPE_CHECKING` block (via the newly-enabled ruff `TC001`
rule, commit 4d4105c), on the assumption that `from __future__ import
annotations` makes the import unnecessary at runtime. That assumption does
not hold for `ProductConfig`: it is a real dataclass handed to route
handlers via Litestar DI (`product_config: ProductConfig` on
`AuthController`, `BillingController`, etc.). Litestar builds one msgspec
`Struct` "signature model" per route covering every parameter — DI-injected
included — and converts the assembled kwargs through it on *every request*
(`litestar/routes/http.py` -> `SignatureModel.parse_values_from_connection_kwargs`
-> `msgspec.convert(...)`). To convert a value into `ProductConfig`, msgspec
must resolve `ProductConfig`'s own field annotations
(`msgspec/_utils.py:get_dataclass_info` -> `get_class_annotations` ->
`typing._eval_type` -> `ForwardRef._evaluate`), which evaluates the strings
`"Product"` / `"frozenset[ModelType] | None"` against **`ProductConfig`'s
own module globals** — not the caller's. With the import only present
under `TYPE_CHECKING`, that name doesn't exist in the module's real
`__dict__` at runtime, so every request to a route with a `ProductConfig`
parameter raised `NameError: name 'Product' is not defined` and was
reported as an unhandled 500 (confirmed against the staging traceback).

Crucially, this resolution is **lazy and per-value** — building the app,
accessing `route_handler.signature_model`, or generating the OpenAPI schema
does *not* trigger it (verified: none of those raise). Only an actual
`msgspec.convert()` call against real kwargs does, which happens inside a
real request. That is also why every *existing* test missed the bug:
`tests/integration/test_content_cookie_flow.py` (and others) call
`AuthController.login.fn(...)` directly, which invokes the Python function
without ever going through Litestar's signature model — so the forward-ref
resolution motion is skipped entirely regardless of whether the import is
correct. "All tests were green" is expected, not surprising, given that
pattern.

The tests below close that gap two ways:

1. An end-to-end request through the real ASGI app (`create_app()`) against
   a route with a bare `ProductConfig` DI dependency (`GET
   /v1/auth/product-info` — no DB, no auth guard, cheapest such route) and a
   direct msgspec-conversion test mirroring exactly what
   `parse_values_from_connection_kwargs` does.
2. A project-wide sweep that walks every *registered* route handler's
   parameter types (as Litestar itself resolves them) and recursively
   checks that every reachable dataclass/msgspec.Struct field type resolves
   via `typing.get_type_hints()` — the same call Litestar/msgspec performs
   internally. This is scoped to types actually reachable from a route
   handler signature (not "every dataclass in `src/`"): plenty of internal
   dataclasses elsewhere in the codebase legitimately keep annotation-only
   imports under `TYPE_CHECKING` per the team's TC-lint convention, and
   never pass through Litestar/msgspec at all — flagging those would be
   noise unrelated to this bug class. Scoping to the request-handling
   surface keeps the sweep precise: it fails only for the exact category of
   mistake that caused the outage, on any current or future route.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import TYPE_CHECKING

import msgspec
import pytest
from litestar.routes import HTTPRoute
from litestar.status_codes import HTTP_200_OK
from litestar.testing import AsyncTestClient

from src.core.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Generator

pytestmark = pytest.mark.unit

_PRODUCT_HEADERS = {"X-Product-Id": "vex"}


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Generator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _set_minimal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET_KEY", "ci-unit-test-secret-key-32-bytes-long")
    monkeypatch.setenv("AISHA_POLLER_ENABLED", "false")
    get_settings.cache_clear()


class TestProductConfigDependencySurvivesRequest:
    """End-to-end: a route depending on `ProductConfig` must not 500.

    `GET /v1/auth/product-info` is the cheapest route with a bare
    `ProductConfig` DI parameter — no DB session, no auth guard — so it
    isolates the signature-model conversion bug from other failure modes.
    """

    async def test_product_info_does_not_500(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_minimal_env(monkeypatch)
        from src.api.app import create_app

        async with AsyncTestClient(app=create_app()) as client:
            resp = await client.get("/v1/auth/product-info", headers=_PRODUCT_HEADERS)

        assert resp.status_code == HTTP_200_OK
        body = resp.json()
        assert body["product"] == "vex"

    async def test_product_info_body_matches_registry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not just "no crash" — the injected ProductConfig round-trips real values."""
        _set_minimal_env(monkeypatch)
        from src.api.app import create_app
        from src.core.product_registry import resolve_product_by_slug

        async with AsyncTestClient(app=create_app()) as client:
            resp = await client.get("/v1/auth/product-info", headers={"X-Product-Id": "synthara"})

        assert resp.status_code == HTTP_200_OK
        body = resp.json()
        expected = resolve_product_by_slug("synthara")
        assert expected is not None
        assert body["display_name"] == expected.display_name
        assert body["age_gate"] == expected.age_gate.value


class TestProductConfigTypeHintsResolve:
    """Narrow unit-level check pinned directly to the previously-broken module.

    Fails immediately (no ASGI app, no client) if `src/core/product.py`
    ever again imports `ModelType`/`Product` only under `TYPE_CHECKING`
    while a dataclass field references them.
    """

    def test_get_type_hints_resolves_without_nameerror(self) -> None:
        from src.core.enums import ModelType, Product
        from src.core.product import ProductConfig

        hints = typing.get_type_hints(ProductConfig)
        assert hints["product"] is Product
        assert hints["allowed_models"] == frozenset[ModelType] | None

    def test_product_config_survives_msgspec_conversion(self) -> None:
        """Mirrors exactly what Litestar's signature model does per-request:
        msgspec.convert() on a value typed as the dataclass, which forces
        resolution of every field's string annotation.
        """
        from src.core.product import ProductConfig
        from src.core.product_registry import VEX_CONFIG

        converted = msgspec.convert(VEX_CONFIG, ProductConfig, strict=False, from_attributes=True)
        assert converted.slug == VEX_CONFIG.slug


class TestRouteHandlerTypesResolveTypeHints:
    """Project-wide sweep, scoped to the actual blast radius of this bug class.

    Walks every route handler registered on the real app (`create_app()`),
    collects every parameter type Litestar hands to its signature model
    (path/query/body params *and* DI-injected dependencies alike — Litestar
    does not distinguish them here), *and* every DI provider function
    reachable via `handler.resolve_dependencies()` — Litestar builds a
    separate signature model per provider too (`SignatureModel.create(fn=
    provider.dependency, ...)` in `litestar/handlers/base.py`), so a
    dataclass could break this way even if no handler declares it directly,
    only some intermediate `Provide()` function. Recursively resolves
    `typing.get_type_hints()` on any dataclass/msgspec.Struct found — the
    same resolution `msgspec.convert()` performs per-request/per-provider
    call. A `NameError` here means a future request to that route would 500
    exactly like the original incident, regardless of which module or PR
    introduced it, and regardless of whether the broken type sits on the
    handler or several dependency hops away.

    Deliberately does **not** scan every dataclass under `src/` — most
    internal ones never reach Litestar's signature model and legitimately
    keep `TYPE_CHECKING`-only imports per the team's TC-lint convention;
    flagging those would be unrelated noise. Verified empirically: reverting
    the fix in `src/core/product.py` (moving the import back under
    `TYPE_CHECKING`) makes this test fail with the same `NameError: name
    'Product' is not defined` seen in production; the unscoped/whole-repo
    variant of this sweep also flags ~8 additional dataclasses that are
    never touched by Litestar and are not part of this bug class.
    """

    def test_all_route_handler_types_resolve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_minimal_env(monkeypatch)
        from src.api.app import create_app

        app = create_app()

        seen: set[type] = set()
        errors: list[str] = []

        def collect(tp: object, origin: str) -> None:
            for arg in getattr(tp, "__args__", ()):
                collect(arg, origin)
            if not isinstance(tp, type) or tp in seen:
                return
            is_dataclass = dataclasses.is_dataclass(tp)
            is_struct = issubclass(tp, msgspec.Struct)
            if not (is_dataclass or is_struct):
                return
            seen.add(tp)
            try:
                hints = typing.get_type_hints(tp)
            except NameError as e:
                errors.append(f"{origin} -> {tp.__module__}.{tp.__qualname__}: {e}")
                return
            for field_name, field_type in hints.items():
                collect(field_type, f"{origin}.{tp.__qualname__}.{field_name}")

        def collect_from_fn(fn: object, origin: str) -> None:
            try:
                param_hints = typing.get_type_hints(fn)
            except Exception:
                # A function's own module failing to resolve its own
                # signature is a separate, louder failure mode (it breaks
                # *every* request touching it, not just ones reaching a
                # specific nested field) — out of scope here.
                return
            for param_name, param_type in param_hints.items():
                collect(param_type, f"{origin}({param_name})")

        checked_providers: set[int] = set()

        for route in app.routes:
            if not isinstance(route, HTTPRoute):
                continue
            for handler in route.route_handlers:
                fn = getattr(handler, "fn", None)
                if fn is not None:
                    collect_from_fn(fn, handler.handler_name)

                # Litestar builds a *separate* signature model for every DI
                # provider function too (see litestar/handlers/base.py
                # ~L386: `provider.signature_model = SignatureModel.create(
                # fn=provider.dependency, ...)`), and converts its resolved
                # kwargs through it on every call — independently of
                # whether the handler itself declares a matching parameter
                # name/type. A dataclass reachable only as a provider's own
                # parameter (never directly on a handler) would be missed
                # by scanning `handler.fn` alone, so walk the full resolved
                # dependency graph for this handler too.
                for dep_name, provide in handler.resolve_dependencies().items():
                    dependency_fn = getattr(provide, "dependency", None)
                    if dependency_fn is None or id(dependency_fn) in checked_providers:
                        continue
                    checked_providers.add(id(dependency_fn))
                    collect_from_fn(dependency_fn, f"provide[{dep_name}]")

        assert seen, "sweep collected zero dataclass/struct types — route walk is broken"
        assert not errors, (
            "Route handler parameter types failed to resolve "
            "(TYPE_CHECKING-only import on a runtime-reachable type?):\n" + "\n".join(errors)
        )
