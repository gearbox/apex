"""Settings: GitHub App credential — all-or-none config, key validity, repo scope, auth mode."""

from __future__ import annotations

import base64
import itertools

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from pydantic import ValidationError

from src.core.config import Settings, parse_github_repo_url
from src.core.enums import GitHubAuthMode
from tests.unit.helpers import hermetic_settings

_JWT_SECRET = "x" * 48
_APP_VARS = ("GITHUB_APP_ID", "GITHUB_APP_INSTALLATION_ID", "GITHUB_APP_PRIVATE_KEY_B64")


def _pem(key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey, *, encrypted: bool = False) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(b"pw") if encrypted else serialization.NoEncryption(),
    )


@pytest.fixture(scope="module")
def rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def rsa_b64(rsa_key: rsa.RSAPrivateKey) -> str:
    return base64.b64encode(_pem(rsa_key)).decode()


@pytest.fixture(autouse=True)
def _clean_github_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*_APP_VARS, "AI_BUNDLES_GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _settings(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], **overrides: object
) -> Settings:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return hermetic_settings(jwt_secret_key=_JWT_SECRET, **overrides)


def _app_env(key_b64: str) -> dict[str, str]:
    return {
        "GITHUB_APP_ID": "12345",
        "GITHUB_APP_INSTALLATION_ID": "678",
        "GITHUB_APP_PRIVATE_KEY_B64": key_b64,
    }


class TestAllOrNone:
    """A1."""

    def test_all_three_set_loads(
        self, monkeypatch: pytest.MonkeyPatch, rsa_b64: str, rsa_key: rsa.RSAPrivateKey
    ) -> None:
        settings = _settings(monkeypatch, _app_env(rsa_b64))

        assert settings.github_app_id == 12345
        assert settings.github_app_installation_id == 678
        assert settings.github_app_private_key_pem == _pem(rsa_key)
        assert settings.github_auth_mode is GitHubAuthMode.app

    def test_private_key_is_never_rendered(
        self, monkeypatch: pytest.MonkeyPatch, rsa_b64: str
    ) -> None:
        settings = _settings(monkeypatch, _app_env(rsa_b64))

        assert rsa_b64 not in repr(settings)
        assert rsa_b64 not in str(settings)
        assert rsa_b64[:40] not in repr(settings.github_app_private_key_b64)

    @pytest.mark.parametrize(
        "present",
        [subset for size in (1, 2) for subset in itertools.combinations(_APP_VARS, size)],
        ids=lambda subset: "+".join(name.removeprefix("GITHUB_APP_") for name in subset),
    )
    def test_partial_config_fails_at_boot(
        self, monkeypatch: pytest.MonkeyPatch, rsa_b64: str, present: tuple[str, ...]
    ) -> None:
        env = {name: value for name, value in _app_env(rsa_b64).items() if name in present}

        with pytest.raises(ValidationError) as excinfo:
            _settings(monkeypatch, env)

        message = str(excinfo.value)
        assert "GitHub App config is partial" in message
        for missing in set(_APP_VARS) - set(present):
            assert missing in message
        assert rsa_b64 not in message
        assert rsa_b64[:40] not in message

    def test_none_set_is_fine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _settings(monkeypatch, {})

        assert settings.github_app_id is None
        assert settings.github_app_private_key_pem is None
        assert settings.github_auth_mode is GitHubAuthMode.none

    def test_blank_values_read_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Compose passes unset vars through as `${VAR:-}` — an empty string, not absent."""
        settings = _settings(monkeypatch, dict.fromkeys(_APP_VARS, ""))

        assert settings.github_app_id is None
        assert settings.github_auth_mode is GitHubAuthMode.none

    def test_a_blank_member_of_a_set_trio_is_partial(
        self, monkeypatch: pytest.MonkeyPatch, rsa_b64: str
    ) -> None:
        env = _app_env(rsa_b64) | {"GITHUB_APP_INSTALLATION_ID": ""}

        with pytest.raises(ValidationError, match="GITHUB_APP_INSTALLATION_ID"):
            _settings(monkeypatch, env)

    @pytest.mark.parametrize("bad_id", ["0", "-1", "abc"])
    def test_ids_must_be_positive_integers(
        self, monkeypatch: pytest.MonkeyPatch, rsa_b64: str, bad_id: str
    ) -> None:
        with pytest.raises(ValidationError):
            _settings(monkeypatch, _app_env(rsa_b64) | {"GITHUB_APP_ID": bad_id})

    @pytest.mark.parametrize("margin", [59, 1801])
    def test_refresh_margin_bounds(self, monkeypatch: pytest.MonkeyPatch, margin: int) -> None:
        with pytest.raises(ValidationError):
            _settings(monkeypatch, {}, github_app_token_refresh_margin_seconds=margin)


class TestPrivateKeyValidation:
    """A2 — the key must be a base64 RSA PEM; errors never echo key material."""

    SENTINEL = "SENTINEL-KEY-MATERIAL-0123456789"

    @pytest.mark.parametrize(
        ("label", "key_b64", "expected"),
        [
            ("not-base64", f"{SENTINEL}!!not base64!!", "not valid base64"),
            (
                "base64-of-garbage",
                base64.b64encode(f"{SENTINEL} hello".encode()).decode(),
                "unencrypted PEM private key",
            ),
            (
                "truncated-pem",
                base64.b64encode(b"-----BEGIN PRIVATE KEY-----\n" + SENTINEL.encode()).decode(),
                "unencrypted PEM private key",
            ),
            (
                "ec-key",
                base64.b64encode(_pem(ec.generate_private_key(ec.SECP256R1()))).decode(),
                "must be an RSA private key",
            ),
            (
                "encrypted-rsa-key",
                base64.b64encode(
                    _pem(
                        rsa.generate_private_key(public_exponent=65537, key_size=2048),
                        encrypted=True,
                    )
                ).decode(),
                "unencrypted PEM private key",
            ),
        ],
        ids=["not-base64", "base64-of-garbage", "truncated-pem", "ec-key", "encrypted-rsa-key"],
    )
    def test_bad_key_fails_at_boot(
        self, monkeypatch: pytest.MonkeyPatch, label: str, key_b64: str, expected: str
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _settings(monkeypatch, _app_env(key_b64))

        message = str(excinfo.value)
        assert expected in message, label
        assert "input_value" not in message
        assert self.SENTINEL not in message
        assert key_b64 not in message
        assert key_b64[:30] not in message

    def test_whitespace_wrapped_base64_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, rsa_b64: str
    ) -> None:
        """`base64 key.pem` without -w0 wraps lines; an env/compose value can carry them."""
        wrapped = "\n".join(rsa_b64[i : i + 64] for i in range(0, len(rsa_b64), 64)) + "\n"

        settings = _settings(monkeypatch, _app_env(wrapped))

        assert settings.github_auth_mode is GitHubAuthMode.app

    def test_other_settings_errors_do_not_echo_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, rsa_b64: str
    ) -> None:
        """An unrelated validation failure must not print a valid key sitting in the env."""
        env = _app_env(rsa_b64)

        with pytest.raises(ValidationError) as excinfo:
            _settings(monkeypatch, env, gpu_session_credit_critical_minutes=10_000)

        message = str(excinfo.value)
        # pydantic would otherwise append a (truncated) repr of the raw env input — the
        # key, JWT secret, etc. — as `input_value=...`.
        assert "input_value" not in message
        assert rsa_b64[:30] not in message
        assert _JWT_SECRET not in message


class TestAuthModeAndRepositories:
    """A3."""

    def test_mode_pat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _settings(monkeypatch, {"AI_BUNDLES_GITHUB_TOKEN": "ghp_static_pat"})

        assert settings.github_auth_mode is GitHubAuthMode.pat

    def test_mode_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _settings(monkeypatch, {}).github_auth_mode is GitHubAuthMode.none

    def test_app_wins_over_a_configured_pat(
        self, monkeypatch: pytest.MonkeyPatch, rsa_b64: str
    ) -> None:
        settings = _settings(
            monkeypatch, _app_env(rsa_b64) | {"AI_BUNDLES_GITHUB_TOKEN": "ghp_static_pat"}
        )

        assert settings.github_auth_mode is GitHubAuthMode.app

    def test_repositories_derive_from_the_default_urls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _settings(monkeypatch, {})

        assert settings.github_repositories == ("aisha", "ai-bundles")

    @pytest.mark.parametrize(
        ("aisha_url", "bundles_url"),
        [
            ("https://github.com/acme/aisha", "https://github.com/acme/ai-bundles"),
            ("https://github.com/acme/aisha.git", "https://github.com/acme/ai-bundles.git"),
            ("https://github.com/acme/aisha/", "https://github.com/acme/ai-bundles/"),
        ],
    )
    def test_repositories_are_bare_names_for_any_url_spelling(
        self, monkeypatch: pytest.MonkeyPatch, rsa_b64: str, aisha_url: str, bundles_url: str
    ) -> None:
        settings = _settings(
            monkeypatch,
            _app_env(rsa_b64),
            aisha_repo_url=aisha_url,
            ai_bundles_repo_url=bundles_url,
        )

        assert settings.github_repositories == ("aisha", "ai-bundles")

    def test_different_owners_fail_in_app_mode(
        self, monkeypatch: pytest.MonkeyPatch, rsa_b64: str
    ) -> None:
        with pytest.raises(ValidationError, match="share one owner"):
            _settings(
                monkeypatch,
                _app_env(rsa_b64),
                aisha_repo_url="https://github.com/acme/aisha.git",
                ai_bundles_repo_url="https://github.com/other/ai-bundles.git",
            )

    @pytest.mark.parametrize(
        "bad_url",
        [
            "git@github.com:gearbox/aisha.git",
            "https://gitlab.com/gearbox/aisha.git",
            "https://github.com/gearbox",
            "https://github.com/gearbox/aisha/extra",
            "",
        ],
    )
    def test_non_github_urls_fail_in_app_mode(
        self, monkeypatch: pytest.MonkeyPatch, rsa_b64: str, bad_url: str
    ) -> None:
        with pytest.raises(ValidationError, match=r"https://github\.com/\{owner\}/\{repo\}"):
            _settings(monkeypatch, _app_env(rsa_b64), aisha_repo_url=bad_url)

    def test_repo_urls_are_only_constrained_in_app_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PAT/none deployments may use whatever URLs they use today."""
        settings = _settings(
            monkeypatch,
            {"AI_BUNDLES_GITHUB_TOKEN": "ghp_static_pat"},
            aisha_repo_url="https://github.com/acme/aisha.git",
            ai_bundles_repo_url="https://github.com/other/ai-bundles.git",
        )

        assert settings.github_auth_mode is GitHubAuthMode.pat
        assert settings.github_repositories == ("aisha", "ai-bundles")

    def test_repositories_raise_for_unparseable_urls_outside_app_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _settings(monkeypatch, {}, aisha_repo_url="git@github.com:gearbox/aisha.git")

        with pytest.raises(ValueError, match=r"github\.com repo URLs"):
            _ = settings.github_repositories


class TestParseGithubRepoUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://github.com/gearbox/aisha.git", ("gearbox", "aisha")),
            ("https://github.com/gearbox/aisha", ("gearbox", "aisha")),
            ("https://github.com/gearbox/ai-bundles/", ("gearbox", "ai-bundles")),
            ("http://github.com/gearbox/aisha", None),
            ("https://github.com//aisha", None),
            ("https://github.com/gearbox/", None),
            ("https://github.com/a/b/c", None),
        ],
    )
    def test_parse(self, url: str, expected: tuple[str, str] | None) -> None:
        assert parse_github_repo_url(url) == expected
