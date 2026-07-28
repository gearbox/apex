"""Centralized structlog configuration for the Apex API."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Final

import structlog

if TYPE_CHECKING:
    from src.core.config import Settings

# Pinned unconditionally, independent of settings.log_level — raising the app
# to DEBUG must not also enable verbose wire-level dumps from these libraries
# (e.g. botocore's SigV4 canonical request bodies).
_THIRD_PARTY_LOGGER_LEVELS: Final[dict[str, int]] = {
    "botocore": logging.WARNING,
    "aiobotocore": logging.WARNING,
    "boto3": logging.WARNING,
    "httpcore": logging.WARNING,
    "httpx": logging.WARNING,
    "urllib3": logging.WARNING,
}


def configure_logging(settings: Settings) -> None:
    """Configure structlog and stdlib logging bridge.

    Sets up shared processors, output format (JSON or console), and bridges
    stdlib logging into structlog so all logs appear in a unified format.

    Args:
        settings: Application settings containing LOG_LEVEL and LOG_FORMAT.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    use_json = settings.log_format == "json"

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    if use_json:
        formatter_processors: list[structlog.types.Processor] = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        formatter_processors = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ]

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=formatter_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Suppress noisy third-party loggers, unconditionally (see
    # _THIRD_PARTY_LOGGER_LEVELS docstring above) — this must run after
    # root_logger.setLevel() above, since stdlib logging resolves effective
    # level by walking up to the nearest ancestor with an explicit level set.
    for logger_name, level in _THIRD_PARTY_LOGGER_LEVELS.items():
        logging.getLogger(logger_name).setLevel(level)
    logging.getLogger("xai_sdk").setLevel(logging.INFO)


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A structlog BoundLogger instance.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
