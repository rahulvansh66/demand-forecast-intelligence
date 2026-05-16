"""Structured logging setup for demand forecast intelligence."""

import logging
import sys
from typing import Optional
import structlog


# Valid log levels for validation
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Track handlers created by this module to manage them safely
_MANAGED_HANDLERS = []


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

    Raises:
        ValueError: If level is not a valid log level
    """
    # Validate log level
    if level.upper() not in _VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level '{level}'. Must be one of: {', '.join(_VALID_LOG_LEVELS)}"
        )

    # Configure standard logging
    root_logger = logging.getLogger()

    # Only remove handlers we previously created to avoid interfering with other libraries
    for handler in _MANAGED_HANDLERS[:]:  # Use slice copy to avoid modification during iteration
        root_logger.removeHandler(handler)
        handler.close()
        _MANAGED_HANDLERS.remove(handler)

    # Add our handler
    if add_handler:
        handler = add_handler
    else:
        handler = logging.StreamHandler(sys.stdout)

    root_logger.addHandler(handler)
    _MANAGED_HANDLERS.append(handler)

    # Set level
    log_level = getattr(logging, level.upper())
    root_logger.setLevel(log_level)

    # Configure structlog with consistent level filtering
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
        wrapper_class=structlog.stdlib.BoundLogger,
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