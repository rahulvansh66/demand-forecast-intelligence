import logging
import json
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