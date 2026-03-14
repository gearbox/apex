"""Unit tests for SupportedLocale enum."""

import pytest

from src.core.enums import SupportedLocale


def test_locale_values_are_lowercase_strings() -> None:
    """All SupportedLocale values serialize as plain lowercase strings."""
    for locale in SupportedLocale:
        assert locale.value == locale.value.lower()
        assert isinstance(locale.value, str)


def test_locale_from_valid_value() -> None:
    """SupportedLocale('en') succeeds."""
    assert SupportedLocale("en") == SupportedLocale.EN
    assert SupportedLocale("ru") == SupportedLocale.RU
    assert SupportedLocale("sr") == SupportedLocale.SR


def test_locale_from_invalid_value_raises() -> None:
    """SupportedLocale('xx') raises ValueError."""
    with pytest.raises(ValueError):
        SupportedLocale("xx")
