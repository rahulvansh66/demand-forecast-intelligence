import logging
import json
import pytest
from io import StringIO

from demand_forecast_intelligence.core.logging.logger import setup_logging, get_logger


def test_setup_logging_creates_structured_logger():
    """Test that setup_logging creates a structured logger."""
    # Capture log output
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)

    setup_logging(level="INFO", add_handler=handler)
    logger = get_logger("test_module")

    logger.info("Test message", extra={"user_id": 123, "action": "test"})

    log_output = log_stream.getvalue()
    assert "Test message" in log_output
    assert "test_module" in log_output


def test_logger_includes_context():
    """Test that logger includes structured context."""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)

    setup_logging(level="DEBUG", add_handler=handler)
    logger = get_logger("data_loader")

    logger.info("Loading dataset",
               dataset_name="walmart_m5",
               file_count=4,
               operation="load")

    log_output = log_stream.getvalue()
    assert "Loading dataset" in log_output
    assert "walmart_m5" in log_output


def test_logger_error_handling():
    """Test logger handles errors properly."""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)

    setup_logging(level="ERROR", add_handler=handler)
    logger = get_logger("error_test")

    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.error("Operation failed",
                    error=str(e),
                    operation="test_operation",
                    exc_info=True)

    log_output = log_stream.getvalue()
    assert "Operation failed" in log_output
    assert "Test error" in log_output


def test_setup_logging_validates_level():
    """Test that setup_logging validates log level input."""
    with pytest.raises(ValueError, match="Invalid log level 'INVALID'"):
        setup_logging(level="INVALID")

    with pytest.raises(ValueError, match="Invalid log level 'trace'"):
        setup_logging(level="trace")


def test_setup_logging_manages_handlers_safely():
    """Test that setup_logging only manages its own handlers."""
    # Create a handler that simulates another library's handler
    external_stream = StringIO()
    external_handler = logging.StreamHandler(external_stream)

    # Add the external handler directly to root logger
    root_logger = logging.getLogger()
    initial_handler_count = len(root_logger.handlers)
    root_logger.addHandler(external_handler)

    # Create our logging setup
    our_stream = StringIO()
    our_handler = logging.StreamHandler(our_stream)
    setup_logging(level="INFO", add_handler=our_handler)

    # Verify the external handler is still there
    assert external_handler in root_logger.handlers
    assert our_handler in root_logger.handlers

    # Clean up
    root_logger.removeHandler(external_handler)
    external_handler.close()


def test_no_automatic_setup_on_import():
    """Test that importing the logger module doesn't automatically configure logging."""
    import importlib
    import logging

    # Clear any existing handlers to start clean
    root_logger = logging.getLogger()
    initial_handler_count = len(root_logger.handlers)

    # Re-import the module (this simulates a fresh import)
    import demand_forecast_intelligence.core.logging.logger as logger_module
    importlib.reload(logger_module)

    # Verify no new handlers were added during import
    final_handler_count = len(root_logger.handlers)
    assert final_handler_count == initial_handler_count, \
        "Import should not add handlers automatically"


def test_multiple_setup_calls_manage_handlers_correctly():
    """Test that multiple setup_logging calls properly manage handlers."""
    log_stream1 = StringIO()
    handler1 = logging.StreamHandler(log_stream1)

    log_stream2 = StringIO()
    handler2 = logging.StreamHandler(log_stream2)

    # First setup
    setup_logging(level="INFO", add_handler=handler1)
    logger = get_logger("multi_test")

    # Log something
    logger.info("First message")
    assert "First message" in log_stream1.getvalue()
    assert log_stream2.getvalue() == ""

    # Second setup should replace the first handler
    setup_logging(level="INFO", add_handler=handler2)
    logger = get_logger("multi_test")

    # Clear stream1 and log again
    log_stream1.truncate(0)
    log_stream1.seek(0)

    logger.info("Second message")

    # Should only appear in stream2 now
    assert log_stream1.getvalue() == ""
    assert "Second message" in log_stream2.getvalue()