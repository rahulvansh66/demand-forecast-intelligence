"""Structured logging setup for demand forecast intelligence."""

import logging
import sys
from typing import Optional
import structlog


def setup_logging(
    level: str = "INFO",
    format_json: bool = True,
    add_handler: Optional[logging.Handler] = None
) -> None:
    """Set up structured logging with consistent format.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_json: Whether to format logs as JSON
        add_handler: Optional custom handler for testing
    """
    # Configure standard logging first
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid conflicts
    root_logger.handlers.clear()

    if add_handler:
        root_logger.addHandler(add_handler)
    else:
        handler = logging.StreamHandler(sys.stdout)
        root_logger.addHandler(handler)

    root_logger.setLevel(getattr(logging, level.upper()))

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.JSONRenderer() if format_json else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger for the given module.

    Args:
        name: Module or component name for the logger

    Returns:
        Configured structlog logger instance
    """
    return structlog.get_logger(name)


# Default setup for package imports
setup_logging()