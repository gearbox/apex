"""Unit tests for Jinja2 email template rendering and subject lines."""

import pytest

from src.api.services.email.i18n import get_subject, render_template


@pytest.mark.parametrize("locale", ["en", "ru", "sr"])
def test_render_verify_email_returns_non_empty(locale: str) -> None:
    context = {
        "display_name": "Test User",
        "verification_url": "https://example.com/verify?token=abc",
        "expires_hours": 24,
        "app_name": "TestApp",
    }
    html, text = render_template("verify_email", locale, context)
    assert len(html) > 0
    assert len(text) > 0
    assert "https://example.com/verify?token=abc" in text
    assert "https://example.com/verify?token=abc" in html
    assert "TestApp" in text
    assert "TestApp" in html


@pytest.mark.parametrize("locale", ["en", "ru", "sr"])
def test_render_reset_password_returns_non_empty(locale: str) -> None:
    context = {
        "display_name": "Test User",
        "reset_url": "https://example.com/reset?token=xyz",
        "expires_minutes": 30,
        "app_name": "TestApp",
    }
    html, text = render_template("reset_password", locale, context)
    assert len(html) > 0
    assert len(text) > 0
    assert "https://example.com/reset?token=xyz" in text
    assert "TestApp" in text


def test_render_unknown_locale_falls_back_to_en() -> None:
    """Unknown locale falls back to English without raising."""
    context = {
        "display_name": "Test",
        "verification_url": "https://example.com/verify",
        "expires_hours": 24,
        "app_name": "TestApp",
    }
    html, text = render_template("verify_email", "zz", context)
    assert "verify" in text.lower() or "Verify" in html
    assert len(html) > 0


@pytest.mark.parametrize("locale", ["en", "ru", "sr"])
def test_subject_line_per_locale(locale: str) -> None:
    subject = get_subject("verify_email", locale, app_name="TestApp")
    assert len(subject) > 0
    assert "TestApp" in subject


def test_subject_line_unknown_locale_falls_back() -> None:
    subject = get_subject("verify_email", "zz", app_name="TestApp")
    assert subject == "Verify your TestApp account"


def test_app_name_is_not_hardcoded_in_templates() -> None:
    """Verify templates use the provided app_name, not a hardcoded string."""
    context_a = {
        "display_name": "User",
        "verification_url": "https://example.com/v",
        "expires_hours": 24,
        "app_name": "BrandAlpha",
    }
    context_b = {**context_a, "app_name": "BrandBeta"}

    _, text_a = render_template("verify_email", "en", context_a)
    _, text_b = render_template("verify_email", "en", context_b)

    assert "BrandAlpha" in text_a
    assert "BrandBeta" in text_b
    assert "Apex" not in text_a  # must not be hardcoded
    assert "Apex" not in text_b
