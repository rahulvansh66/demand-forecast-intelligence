"""Data-related custom exceptions."""

from pathlib import Path
from typing import Union


class DataException(Exception):
    """Base exception for data-related errors."""
    pass


class DataFileNotFoundError(DataException):
    """Raised when a required data file is not found."""

    def __init__(self, filename: str, expected_path: Union[str, Path]):
        self.filename = filename
        self.expected_path = Path(expected_path)
        super().__init__(
            f"Data file '{filename}' not found at expected location: {self.expected_path}"
        )


class DataValidationError(DataException):
    """Raised when data fails validation checks."""

    def __init__(self, message: str, file_path: Union[str, Path] = None):
        self.file_path = Path(file_path) if file_path else None
        if self.file_path:
            message = f"Validation failed for {self.file_path}: {message}"
        super().__init__(message)


class DataQualityError(DataException):
    """Raised when data quality checks fail."""

    def __init__(self, message: str, quality_metric: str, threshold: float = None):
        self.quality_metric = quality_metric
        self.threshold = threshold

        if threshold is not None:
            message = f"{message} (metric: {quality_metric}, threshold: {threshold})"
        super().__init__(message)