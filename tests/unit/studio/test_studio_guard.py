"""studio_feature_guard: flag / product feature matrix, 404 (not 403), before auth."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litestar import Litestar, get
from litestar.exceptions import NotFoundException
from litestar.status_codes import HTTP_404_NOT_FOUND
from litestar.testing import TestClient

from src.api.security import auth_guard
from src.api.security.studio import studio_feature_guard
from src.core.product_registry import SYNTHARA_CONFIG, VEX_CONFIG
from tests.legal_support import vex_product_scope

pytestmark = pytest.mark.unit

_NO_STUDIO = VEX_CONFIG.__class__(
    **{**VEX_CONFIG.__dict__, "features": frozenset({"organizations"})}
)


def _settings(enabled: bool) -> MagicMock:
    return MagicMock(studio_enabled=enabled)


async def _call(*, enabled: bool, config: Any) -> None:
    connection = MagicMock()
    connection.state = {"product_config": config}
    with patch("src.api.security.studio.get_settings", return_value=_settings(enabled)):
        await studio_feature_guard(connection, MagicMock())


class TestMatrix:
    @pytest.mark.parametrize("config", [VEX_CONFIG, SYNTHARA_CONFIG], ids=["vex", "synthara"])
    async def test_enabled_and_product_has_feature(self, config: Any) -> None:
        await _call(enabled=True, config=config)

    @pytest.mark.parametrize("config", [VEX_CONFIG, SYNTHARA_CONFIG], ids=["vex", "synthara"])
    async def test_kill_switch_off_hides_studio_for_every_product(self, config: Any) -> None:
        with pytest.raises(NotFoundException):
            await _call(enabled=False, config=config)

    async def test_product_without_the_feature_is_hidden_even_when_enabled(self) -> None:
        assert not _NO_STUDIO.has_feature("studio")
        with pytest.raises(NotFoundException):
            await _call(enabled=True, config=_NO_STUDIO)

    async def test_missing_product_scope_is_a_wiring_bug_not_a_404(self) -> None:
        connection = MagicMock()
        connection.state = {}
        with (
            patch("src.api.security.studio.get_settings", return_value=_settings(True)),
            pytest.raises(RuntimeError, match="ProductMiddleware"),
        ):
            await studio_feature_guard(connection, MagicMock())

    def test_both_products_declare_the_feature(self) -> None:
        assert VEX_CONFIG.has_feature("studio")
        assert SYNTHARA_CONFIG.has_feature("studio")
        assert SYNTHARA_CONFIG.has_feature("organizations")


class TestOrderingBeforeAuth:
    """The feature guard runs first, so a disabled studio is 404 even without credentials."""

    @staticmethod
    def _app() -> Litestar:
        @get("/probe", guards=[studio_feature_guard, auth_guard])
        async def probe() -> dict[str, str]:
            return {"ok": "yes"}

        return Litestar(route_handlers=[probe], middleware=[vex_product_scope])

    def test_disabled_unauthenticated_request_is_404_not_401(self) -> None:
        with (
            patch("src.api.security.studio.get_settings", return_value=_settings(False)),
            TestClient(app=self._app()) as client,
        ):
            assert client.get("/probe").status_code == HTTP_404_NOT_FOUND

    def test_enabled_unauthenticated_request_proceeds_to_auth(self) -> None:
        with (
            patch("src.api.security.studio.get_settings", return_value=_settings(True)),
            TestClient(app=self._app()) as client,
        ):
            assert client.get("/probe").status_code == 401
