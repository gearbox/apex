"""Locale-aware email subject lines and template rendering via Jinja2.

Subject lines live here as data (they're short, one-line strings — no need for files).
Email bodies are rendered from Jinja2 templates in src/templates/email/{locale}/.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

from src.core.enums import SupportedLocale

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "templates" / "email"
FALLBACK_LOCALE = SupportedLocale.EN.value

# ---------------------------------------------------------------------------
# Subject lines — keyed by (template_name, locale)
# ---------------------------------------------------------------------------

_SUBJECTS: dict[str, dict[str, str]] = {
    "verify_email": {
        "en": "Verify your {app_name} account",
        "ru": "Подтвердите ваш аккаунт {app_name}",
        "sr": "Potvrdite vaš {app_name} nalog",
    },
    "reset_password": {
        "en": "Reset your {app_name} password",
        "ru": "Сбросить пароль {app_name}",
        "sr": "Resetujte vašu {app_name} lozinku",
    },
}


def get_subject(template_name: str, locale: str, *, app_name: str) -> str:
    """Return the subject line for a template + locale, with app_name interpolated.

    Falls back to English if the locale or template is unknown.
    """
    locale_map = _SUBJECTS.get(template_name, {})
    raw = locale_map.get(locale, locale_map.get(FALLBACK_LOCALE, ""))
    return raw.format(app_name=app_name)


# ---------------------------------------------------------------------------
# Jinja2 template renderer
# ---------------------------------------------------------------------------

# Cache one Environment per locale directory to avoid repeated filesystem scans.
_jinja_envs: dict[str, Environment] = {}


def _get_env(locale: str) -> Environment:
    """Return a Jinja2 Environment rooted at the locale's template directory.

    Falls back to the English directory if the requested locale doesn't exist.
    """
    if locale not in _jinja_envs:
        locale_dir = TEMPLATES_DIR / locale
        if not locale_dir.is_dir():
            locale = FALLBACK_LOCALE
            if locale in _jinja_envs:
                return _jinja_envs[locale]
        _jinja_envs[locale] = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR / locale)),
            autoescape=select_autoescape(["html"]),
            keep_trailing_newline=True,
        )
    return _jinja_envs[locale]


def render_template(
    template_name: str,
    locale: str,
    context: dict[str, str | int],
) -> tuple[str, str]:
    """Render (html_body, text_body) for a given template and locale.

    Args:
        template_name: Base name without extension, e.g. ``"verify_email"``.
        locale: ISO 639-1 code, e.g. ``"en"``, ``"ru"``, ``"sr"``.
        context: Template variables dict.

    Returns:
        Tuple of ``(html_body, text_body)``.

    Raises:
        TemplateNotFound: If neither the requested locale nor the fallback
            has the template file. Should not happen if the acceptance criteria
            are met (every locale has every template).
    """
    env = _get_env(locale)
    try:
        html = env.get_template(f"{template_name}.html").render(**context)
        text = env.get_template(f"{template_name}.txt").render(**context)
    except TemplateNotFound:
        # Last resort: fall back to English
        fallback_env = _get_env(FALLBACK_LOCALE)
        html = fallback_env.get_template(f"{template_name}.html").render(**context)
        text = fallback_env.get_template(f"{template_name}.txt").render(**context)
    return html, text
