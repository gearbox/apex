"""Verify .env.staging.example produces a usable Settings.

Catches drift between Settings fields and the documented example file.
Run after every Settings field add/rename to confirm operator-facing
docs stay in sync.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import Settings

_REPO_ROOT = Path(__file__).parent.parent.parent
_ENV_STAGING_EXAMPLE = _REPO_ROOT / ".env.staging.example"


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file the way docker compose / pydantic-settings does."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def test_env_staging_example_loads_into_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every var in .env.staging.example must be a Settings field or an
    explicitly-allowlisted compose-only var."""
    env = _parse_env_file(_ENV_STAGING_EXAMPLE)

    # Replace placeholders with values that pass type/format validation.
    overrides = {
        "JWT_SECRET_KEY": "x" * 48,
        "POSTGRES_PASSWORD": "test_pw",
        "VASTAI_API_KEY": "vastai_test",
        "AISHA_CF_API_TOKEN": "cf_test",
        "AISHA_CF_ACCOUNT_ID": "cf_acct",
        "AISHA_CF_ZONE_ID": "cf_zone",
        "AISHA_CF_TUNNEL_DOMAIN": "test.example",
        "AI_BUNDLES_GITHUB_TOKEN": "ghp_test",
        "APEX_CALLBACK_URL": "https://staging.test.example",
        "R2_ACCOUNT_ID": "r2_test",
        "R2_ACCESS_KEY_ID": "r2_test",
        "R2_SECRET_ACCESS_KEY": "r2_test",
    }
    env |= overrides

    # Vars in the example file that are NOT Settings fields (compose-level only)
    non_settings_vars = {
        "API_IMAGE",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        # Consumed by the redis container's --requirepass and folded into
        # REDIS_URL by compose interpolation, not read directly by apex.
        "REDIS_PASSWORD",
        # Consumed by the cloudflared sidecar (compose service), not by apex
        "CLOUDFLARE_TUNNEL_TOKEN",
    }

    settings_env = {k: v for k, v in env.items() if k not in non_settings_vars}

    for k, v in settings_env.items():
        monkeypatch.setenv(k, v)

    Settings()  # ValidationError fails the test


def test_no_unprefixed_cf_in_source() -> None:
    """Guard against accidental reintroduction of CF_API_TOKEN / cf_api_token etc.

    The four CF-API fields must always be prefixed with AISHA_ to disambiguate
    from CLOUDFLARE_TUNNEL_TOKEN (which is a separate concern).

    NOTE: CF_TUNNEL_TOKEN intentionally excluded from forbidden_env — it is the
    Vast.ai Instance Portal's mandated boot-time env var name for named tunnels,
    set in _env_builder.build_acs_env. Its legitimate, sole location is pinned by
    test_cf_tunnel_token_only_in_env_builder.
    """
    import subprocess

    forbidden = [
        "cf_api_token",
        "cf_account_id",
        "cf_zone_id",
        "cf_tunnel_domain",
        "cf_configured",
    ]
    forbidden_env = ["CF_API_TOKEN", "CF_ACCOUNT_ID", "CF_ZONE_ID", "CF_TUNNEL_DOMAIN"]

    src_root = _REPO_ROOT / "src"
    for needle in forbidden + forbidden_env:
        result = subprocess.run(  # noqa: S603
            [
                "grep",
                "-r",
                "-l",
                "-w",
                needle,
                "--exclude-dir=__pycache__",
                "--include=*.py",
                str(src_root),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        # grep exits 1 on no match — that's what we want.
        # -w ensures 'cf_api_token' does not match 'aisha_cf_api_token' (word boundary).
        assert result.returncode == 1, (
            f"Forbidden identifier {needle!r} reintroduced in source: {result.stdout}"
        )


def test_cf_tunnel_token_only_in_env_builder() -> None:
    """CF_TUNNEL_TOKEN (unprefixed) is allowed ONLY in _env_builder.py, where it is
    injected for the Vast.ai portal. Anywhere else is an accidental convention breach."""
    import subprocess

    src_root = _REPO_ROOT / "src"
    result = subprocess.run(  # noqa: S603
        [
            "grep",
            "-r",
            "-l",
            "-w",
            "CF_TUNNEL_TOKEN",
            "--exclude-dir=__pycache__",
            "--include=*.py",
            str(src_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    matched_files = set(result.stdout.split()) if result.returncode == 0 else set()
    allowed = {str(src_root / "api" / "services" / "gpu_session" / "_env_builder.py")}
    assert matched_files <= allowed, (
        f"CF_TUNNEL_TOKEN found outside the allowed location(s): {matched_files - allowed}"
    )
    assert allowed <= matched_files, "CF_TUNNEL_TOKEN missing from _env_builder.py"
