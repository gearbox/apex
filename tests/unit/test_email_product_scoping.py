"""W2 — verification / reset emails follow the *user's* product (link + brand).

Uses a recording ``EmailService`` (real template rendering, no network) and the
real ``Settings.app_url_for`` / product registry, so the assertions cover the
production wiring in ``init_services`` rather than a hand-rolled map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.email_verification import EmailVerificationService
from src.core.config import Settings
from src.core.product_registry import get_product_config_by_slug
from tests.email_support import RecordingEmailService
from tests.revocation_support import make_session_termination_factory_noop

if TYPE_CHECKING:
    from src.api.services.email.base import EmailMessage

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_VEX_URL = "https://app.vex.test"
_SYNTHARA_URL = "https://app.synthara.test"


def _settings() -> Settings:
    return Settings(app_url_vex=_VEX_URL, app_url_synthara=_SYNTHARA_URL)


def _service(email: RecordingEmailService, settings: Settings) -> EmailVerificationService:
    """Wired exactly as ``init_services`` wires it."""
    return EmailVerificationService(
        email_service=email,
        app_url_for=settings.app_url_for,
        brand_for=lambda slug: get_product_config_by_slug(slug).display_name,
        session_termination_factory=make_session_termination_factory_noop(),
    )


def _user(product_id: str, *, locale: str = "en") -> MagicMock:
    user = MagicMock()
    user.id = uuid4()
    user.email = f"{product_id}-user@example.com"
    user.display_name = "Alice"
    user.locale = locale
    user.product_id = product_id
    return user


async def _send_verification(svc: EmailVerificationService, user: MagicMock) -> None:
    with (
        patch("src.api.services.email_verification.UserRepository") as user_repo_cls,
        patch("src.api.services.email_verification.AuthTokenRepository") as token_repo_cls,
    ):
        user_repo_cls.return_value.get_active_user = AsyncMock(return_value=user)
        token_repo_cls.return_value.create_verification_token = AsyncMock(return_value="tok-1")
        await svc.send_verification_email(user.id, session=AsyncMock())


async def _send_reset(svc: EmailVerificationService, user: MagicMock, **kwargs: Any) -> None:
    with (
        patch("src.api.services.email_verification.UserRepository") as user_repo_cls,
        patch("src.api.services.email_verification.AuthTokenRepository") as token_repo_cls,
    ):
        user_repo_cls.return_value.get_active_user_by_email = AsyncMock(return_value=user)
        token_repo_cls.return_value.create_reset_token = AsyncMock(return_value="tok-2")
        await svc.send_password_reset_email(
            user.email, product_id=user.product_id, session=AsyncMock(), **kwargs
        )


def _whole_message(message: EmailMessage) -> str:
    return f"{message.subject}\n{message.text_body}\n{message.html_body}"


@pytest.mark.parametrize(
    ("product_id", "base", "brand"),
    [("synthara", _SYNTHARA_URL, "Synthara"), ("vex", _VEX_URL, "vex.pics")],
)
@pytest.mark.parametrize("locale", ["en", "ru", "sr"])
class TestW2aVerificationEmailIsPerProduct:
    async def test_w2_a_link_and_brand_follow_the_users_product(
        self, product_id: str, base: str, brand: str, locale: str
    ) -> None:
        email = RecordingEmailService()
        svc = _service(email, _settings())

        await _send_verification(svc, _user(product_id, locale=locale))

        (message,) = email.sent
        assert f"{base}/verify-email?token=tok-1" in message.text_body
        assert f"{base}/verify-email?token=tok-1" in message.html_body
        assert brand in message.subject
        assert brand in message.text_body
        assert brand in message.html_body
        assert "Apex" not in _whole_message(message)


@pytest.mark.parametrize(
    ("product_id", "base", "brand", "other_base"),
    [
        ("synthara", _SYNTHARA_URL, "Synthara", _VEX_URL),
        ("vex", _VEX_URL, "vex.pics", _SYNTHARA_URL),
    ],
)
@pytest.mark.parametrize("locale", ["en", "ru", "sr"])
class TestW2bResetEmailIsPerProduct:
    async def test_w2_b_link_and_brand_follow_the_users_product(
        self, product_id: str, base: str, brand: str, other_base: str, locale: str
    ) -> None:
        email = RecordingEmailService()
        svc = _service(email, _settings())

        await _send_reset(svc, _user(product_id, locale=locale), ip_address="203.0.113.7")

        (message,) = email.sent
        assert f"{base}/reset-password?token=tok-2" in message.text_body
        assert f"{base}/reset-password?token=tok-2" in message.html_body
        assert other_base not in _whole_message(message)
        assert brand in message.subject
        assert brand in message.text_body
        assert "Apex" not in _whole_message(message)


class TestW2fNoHiddenFallbacks:
    def test_w2_f_service_requires_app_url_for_and_brand_for(self) -> None:
        with pytest.raises(TypeError):
            EmailVerificationService(  # type: ignore[call-arg]
                email_service=RecordingEmailService(),
                session_termination_factory=make_session_termination_factory_noop(),
            )

    def test_w2_f_service_no_longer_accepts_a_global_app_url_or_app_name(self) -> None:
        with pytest.raises(TypeError):
            EmailVerificationService(  # type: ignore[call-arg]
                email_service=RecordingEmailService(),
                app_url="https://global.test",  # pyright: ignore[reportCallIssue]
                app_name="Apex",  # pyright: ignore[reportCallIssue]
                session_termination_factory=make_session_termination_factory_noop(),
            )

    async def test_w2_f_verification_sender_requires_app_name(self) -> None:
        with pytest.raises(TypeError):
            await RecordingEmailService().send_verification_email(  # type: ignore[call-arg]
                to="a@example.com", display_name="A", verification_url="https://x.test/v"
            )

    async def test_w2_f_reset_sender_requires_app_name(self) -> None:
        with pytest.raises(TypeError):
            await RecordingEmailService().send_password_reset_email(  # type: ignore[call-arg]
                to="a@example.com", display_name="A", reset_url="https://x.test/r"
            )

    def test_w2_f_settings_have_no_global_app_url_or_app_name(self) -> None:
        assert "app_url" not in Settings.model_fields
        assert "app_name" not in Settings.model_fields
