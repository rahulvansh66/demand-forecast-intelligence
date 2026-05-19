from pathlib import Path
import pytest

from demand_forecast_intelligence.core.constants.paths import ProjectPaths, M5Files
from demand_forecast_intelligence.core.exceptions.data_exceptions import (
    DataFileNotFoundError,
    DataValidationError,
    DataQualityError
)


def test_project_paths_initialization():
    """Test ProjectPaths creates correct directory structure."""
    paths = ProjectPaths()

    assert paths.PROJECT_ROOT.name == "demand_forecast_intelligence"
    assert paths.DATA_RAW.name == "raw"
    assert paths.CONFIGS.name == "configs"


def test_m5_files_paths():
    """Test M5Files creates correct file paths."""
    files = M5Files()

    assert files.SALES_TRAIN_VALIDATION.name == "sales_train_validation.csv"
    assert files.CALENDAR.name == "calendar.csv"
    assert str(files.SALES_TRAIN_VALIDATION).endswith("sales_train_validation.csv")


def test_data_file_not_found_error():
    """Test custom DataFileNotFoundError exception."""
    with pytest.raises(DataFileNotFoundError) as exc_info:
        raise DataFileNotFoundError("test_file.csv", "/path/to/file")

    error = exc_info.value
    assert "test_file.csv" in str(error)
    assert "/path/to/file" in str(error)


def test_data_validation_error():
    """Test DataValidationError with file path."""
    error = DataValidationError("Missing columns", "/path/to/file.csv")
    assert "Missing columns" in str(error)
    assert "/path/to/file.csv" in str(error)
    assert error.file_path.name == "file.csv"


def test_data_quality_error():
    """Test DataQualityError with metrics."""
    error = DataQualityError(
        "Too many missing values",
        quality_metric="missing_percentage",
        threshold=0.05
    )
    assert "Too many missing values" in str(error)
    assert "missing_percentage" in str(error)
    assert "0.05" in str(error)