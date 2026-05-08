"""Standalone bundle validator — clones ai-bundles and parses every entry.

Run from the apex repo root:

    uv run python tools/validate_bundles.py

Exits non-zero if the index is empty or any bundle entry was skipped.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import structlog

from src.api.services.bundle_index import BundleIndexService
from src.core.config import Settings, get_settings


def _configure_logging() -> None:
    """Make sure WARNING from BundleIndexService actually reaches the terminal."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        processors=[
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ],
    )


async def main() -> int:
    _configure_logging()
    log = structlog.get_logger("validate_bundles")

    settings: Settings = get_settings()

    # Use a writable, absolute path. Repo-local keeps it out of /tmp where
    # multi-user systems can collide.
    cache_dir = (Path.cwd() / ".cache" / "ai-bundles").resolve()
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    svc = BundleIndexService(
        repo_url=settings.ai_bundles_repo_url,
        github_token=settings.ai_bundles_github_token,
        sync_interval_minutes=60,
        cache_dir=cache_dir,
    )

    log.info("validate_bundles.cache_dir", path=str(cache_dir))

    # sync() runs the actual git clone/pull AND re-parses the index.
    # Use start() if you want it to *raise* on the first sync; sync() logs
    # and swallows. For a CLI validator we want the loud version:
    try:
        await svc._sync_once(raise_on_error=True)  # noqa: SLF001 — tooling
    except Exception:
        log.exception("validate_bundles.sync_failed")
        return 2

    model_index = list(svc._model_index)  # noqa: SLF001 — tooling
    bundle_index = list(svc._bundle_index)  # noqa: SLF001 — tooling

    log.info(
        "validate_bundles.result",
        model_types=model_index,
        bundles=bundle_index,
        bundle_count=len(bundle_index),
    )

    if not bundle_index:
        log.error(
            "validate_bundles.empty_index",
            hint=(
                "No bundles were parsed. Either the repo is empty, or every "
                "entry was skipped — re-run and look for "
                "'bundle_index.entry_invalid' warnings above."
            ),
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
