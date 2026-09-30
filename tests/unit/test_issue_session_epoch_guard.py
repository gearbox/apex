"""K6 (unit) — ``AuthService.issue_session`` never mints into a revocation's own second.

Access tokens are rejected when ``iat <= epoch`` (whole seconds), so a token
minted right after an OAuth claim's bulk revocation would be revoked as it is
issued. ``issue_session`` therefore waits out the epoch's second — at most ~2 s
and only when an epoch was just written.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.security import JWTConfig, JWTService, PasswordService
from src.api.services.auth import AuthService
from tests.legal_support import TEST_REQUEST_CONTEXT, make_legal_acceptance_service

pytestmark = pytest.mark.unit

_EPOCH = 1_800_000_000


def _service(epoch: int | None) -> tuple[AuthService, JWTService]:
    jwt_service = JWTService(JWTConfig(secret_key="test_secret_key_for_testing_only_256bits"))
    token_revocation = MagicMock()
    token_revocation.get_current_epoch = AsyncMock(return_value=epoch)
    service = AuthService(
        legal_acceptance_service=make_legal_acceptance_service(),
        repository=AsyncMock(),
        jwt_service=jwt_service,
        password_service=PasswordService(),
        token_revocation_service=token_revocation,
    )
    return service, jwt_service


async def _issue(service: AuthService, *, not_before_epoch: int | None = None) -> None:
    await service.issue_session(
        uuid4(),
        product_id="vex",
        context=TEST_REQUEST_CONTEXT,
        not_before_epoch=not_before_epoch,
    )


class TestSleepBounds:
    async def test_no_epoch_never_sleeps(self) -> None:
        service, _ = _service(None)
        with patch("src.api.services.auth.asyncio.sleep", new=AsyncMock()) as sleep:
            await _issue(service)
        sleep.assert_not_awaited()

    async def test_epoch_in_current_second_sleeps_until_the_next_one(self) -> None:
        service, _ = _service(_EPOCH)
        with (
            patch("src.api.services.auth.time.time", return_value=_EPOCH + 0.2),
            patch("src.api.services.auth.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            await _issue(service)
        sleep.assert_awaited_once()
        (wait,) = sleep.await_args_list[0].args
        assert 0.8 <= wait <= 2.0
        assert _EPOCH + 0.2 + wait >= _EPOCH + 1  # the mint lands strictly after the epoch second

    async def test_epoch_from_an_earlier_second_does_not_sleep(self) -> None:
        service, _ = _service(_EPOCH)
        with (
            patch("src.api.services.auth.time.time", return_value=_EPOCH + 30.0),
            patch("src.api.services.auth.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            await _issue(service)
        sleep.assert_not_awaited()

    async def test_wait_is_capped_when_clocks_disagree_wildly(self) -> None:
        """A Redis clock far ahead of the app clock must not stall the request."""
        service, _ = _service(_EPOCH)
        with (
            patch("src.api.services.auth.time.time", return_value=_EPOCH - 60.0),
            patch("src.api.services.auth.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            await _issue(service)
        sleep.assert_not_awaited()


class TestMintedAfterEpoch:
    async def test_token_minted_with_iat_after_the_epoch(self) -> None:
        """Real clock: an epoch written this very second → the token's iat is later."""
        epoch = int(time.time())
        service, jwt_service = _service(epoch)

        tokens = await service.issue_session(
            uuid4(), product_id="vex", context=TEST_REQUEST_CONTEXT
        )

        payload = jwt_service.decode_access_token(tokens.access_token)
        assert payload is not None
        assert payload.iat > epoch  # the revocation rule is `iat <= epoch`


class TestNotBeforeEpoch:
    """V1 — the claim's own epoch cannot be cancelled by an indeterminate epoch read."""

    async def test_v1_a_known_epoch_waits_even_when_the_read_is_none(self) -> None:
        """Breaker open / read failed ⇒ ``get_current_epoch`` is None; the wait still happens."""
        service, _ = _service(None)
        with (
            patch("src.api.services.auth.time.time", return_value=_EPOCH + 0.2),
            patch("src.api.services.auth.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            await _issue(service, not_before_epoch=_EPOCH)
        sleep.assert_awaited_once()
        (wait,) = sleep.await_args_list[0].args
        assert _EPOCH + 0.2 + wait >= _EPOCH + 1

    async def test_v1_a_minted_iat_is_after_not_before_epoch_with_a_none_read(self) -> None:
        """Real clock: epoch written this second, read indeterminate → iat is strictly later."""
        epoch = int(time.time())
        service, jwt_service = _service(None)

        tokens = await service.issue_session(
            uuid4(),
            product_id="vex",
            context=TEST_REQUEST_CONTEXT,
            not_before_epoch=epoch,
        )

        payload = jwt_service.decode_access_token(tokens.access_token)
        assert payload is not None
        assert payload.iat > epoch

    @pytest.mark.parametrize(("not_before", "read"), [(_EPOCH, _EPOCH - 5), (_EPOCH - 5, _EPOCH)])
    async def test_v1_b_waits_on_the_larger_of_both_in_either_order(
        self, not_before: int, read: int
    ) -> None:
        service, _ = _service(read)
        with (
            patch("src.api.services.auth.time.time", return_value=_EPOCH + 0.2),
            patch("src.api.services.auth.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            await _issue(service, not_before_epoch=not_before)
        sleep.assert_awaited_once()
        (wait,) = sleep.await_args_list[0].args
        assert _EPOCH + 0.2 + wait >= _EPOCH + 1

    async def test_v1_b_earlier_second_and_no_read_does_not_sleep(self) -> None:
        service, _ = _service(None)
        with (
            patch("src.api.services.auth.time.time", return_value=_EPOCH + 30.0),
            patch("src.api.services.auth.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            await _issue(service, not_before_epoch=_EPOCH)
        sleep.assert_not_awaited()

    async def test_v1_b_wait_is_still_capped(self) -> None:
        service, _ = _service(None)
        with (
            patch("src.api.services.auth.time.time", return_value=_EPOCH - 60.0),
            patch("src.api.services.auth.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            await _issue(service, not_before_epoch=_EPOCH)
        sleep.assert_not_awaited()
