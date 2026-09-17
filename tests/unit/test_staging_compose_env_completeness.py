"""Every Compose-required staging var must be documented in .env.staging.example.

U5 (round-4 remediation): docker-compose.staging.yml and
docker-compose.portainer-staging.yml each use the required-variable
interpolation form (``${VAR:?error message}``) for secrets/config with no
safe default. Compose fails immediately at interpolation time if such a var
is unset, so a deployment that follows the documented "copy .env.staging.example
and fill in every value" procedure must have every one of those names in the
template — a var missing from the template (or merely commented out, which is
functionally the same as missing for --env-file purposes) halts the
deployment. This is the second config-template completeness defect in two
rounds (PROVISIONING_SCRIPT_REF was the first) — this test is the mechanical
check the round-4 finding asked for so a third round doesn't need to
rediscover the same class of gap by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_ENV_STAGING_EXAMPLE = _REPO_ROOT / ".env.staging.example"
_COMPOSE_FILES = (
    _REPO_ROOT / "docker-compose.staging.yml",
    _REPO_ROOT / "docker-compose.portainer-staging.yml",
)

# Matches the Compose "required variable" interpolation form: ${VAR:?message}.
_REQUIRED_VAR_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*):\?")

# Matches an uncommented top-level assignment line, e.g. "FOO=bar" — a
# commented-out line ("# FOO=") does not count: docker compose's --env-file
# never sees it, so it is functionally undocumented.
_ENV_ASSIGNMENT_PATTERN = re.compile(r"^([A-Z][A-Z0-9_]*)=", re.MULTILINE)


def _required_vars_in(compose_path: Path) -> set[str]:
    return set(_REQUIRED_VAR_PATTERN.findall(compose_path.read_text()))


def _defined_vars_in(env_path: Path) -> set[str]:
    return set(_ENV_ASSIGNMENT_PATTERN.findall(env_path.read_text()))


def test_every_compose_required_var_is_documented_in_staging_example() -> None:
    defined = _defined_vars_in(_ENV_STAGING_EXAMPLE)

    for compose_path in _COMPOSE_FILES:
        required = _required_vars_in(compose_path)
        missing = sorted(required - defined)
        assert not missing, (
            f"{compose_path.name} requires these vars via ${{VAR:?...}} but "
            f".env.staging.example does not define them (commented-out lines "
            f"don't count — docker compose's --env-file never sees them): "
            f"{missing}"
        )


def test_at_least_one_required_var_is_found_in_each_compose_file() -> None:
    """Guards against the pattern silently matching nothing (e.g. after a
    Compose syntax change) and this test passing for the wrong reason."""
    for compose_path in _COMPOSE_FILES:
        assert _required_vars_in(compose_path), (
            f"No ${{VAR:?...}} required vars found in {compose_path.name} — "
            "the required-var pattern may no longer match this file's syntax."
        )
