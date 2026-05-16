"""Project paths and file constants."""

from pathlib import Path
from typing import Union


class ProjectPaths:
    """Central definition of all project paths."""

    def __init__(self, project_root: Union[str, Path] = None):
        """Initialize project paths.

        Args:
            project_root: Override for project root directory.
                         If None, infers from this file's location.
        """
        if project_root:
            self.PROJECT_ROOT = Path(project_root).resolve()
        else:
            # Go up from src/demand_forecast_intelligence/core/constants/paths.py to project root
            self.PROJECT_ROOT = Path(__file__).parents[4].resolve()

        # Data directories
        self.DATA_ROOT = self.PROJECT_ROOT / "data"
        self.DATA_RAW = self.DATA_ROOT / "raw"
        self.DATA_INTERIM = self.DATA_ROOT / "interim"
        self.DATA_PROCESSED = self.DATA_ROOT / "processed"
        self.DATA_EXTERNAL = self.DATA_ROOT / "external"

        # Source directories
        self.SRC_ROOT = self.PROJECT_ROOT / "src"
        self.PACKAGE_ROOT = self.SRC_ROOT / "demand_forecast_intelligence"

        # Configuration and outputs
        self.CONFIGS = self.PROJECT_ROOT / "configs"
        self.MODELS = self.PROJECT_ROOT / "models"
        self.REPORTS = self.PROJECT_ROOT / "reports"
        self.NOTEBOOKS = self.PROJECT_ROOT / "notebooks"

        # Experiments and tests
        self.EXPERIMENTS = self.PROJECT_ROOT / "experiments"
        self.TESTS = self.PROJECT_ROOT / "tests"


class M5Files:
    """M5 dataset file definitions."""

    def __init__(self, data_root: Union[str, Path] = None):
        """Initialize M5 file paths.

        Args:
            data_root: Override for data root directory
        """
        if data_root:
            self.data_root = Path(data_root)
        else:
            self.data_root = ProjectPaths().DATA_RAW

        # M5 Competition CSV files
        self.SALES_TRAIN_VALIDATION = self.data_root / "sales_train_validation.csv"
        self.SALES_TRAIN_EVALUATION = self.data_root / "sales_train_evaluation.csv"
        self.CALENDAR = self.data_root / "calendar.csv"
        self.SELL_PRICES = self.data_root / "sell_prices.csv"
        self.SAMPLE_SUBMISSION = self.data_root / "sample_submission.csv"

    def all_files(self) -> list[Path]:
        """Return list of all M5 dataset files."""
        return [
            self.SALES_TRAIN_VALIDATION,
            self.SALES_TRAIN_EVALUATION,
            self.CALENDAR,
            self.SELL_PRICES,
            self.SAMPLE_SUBMISSION
        ]

    def required_files(self) -> list[Path]:
        """Return list of files required for basic functionality."""
        return [
            self.SALES_TRAIN_VALIDATION,
            self.CALENDAR,
            self.SELL_PRICES
        ]


# Global instances for easy import
PATHS = ProjectPaths()
M5_FILES = M5Files()