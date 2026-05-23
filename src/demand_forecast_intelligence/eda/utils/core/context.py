"""
EDA Context for shared state management across analysis steps.

This module provides the EDAContext class that manages data loading, results caching,
and output management for all EDA analysis steps.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EDAContext:
    """
    Centralized context for EDA analysis with lazy loading and result caching.

    This class provides:
    - Lazy loading of M5 dataset files (sales, calendar, pricing)
    - Results caching with JSON serialization
    - Output directory structure management
    - Configuration-based initialization
    """

    data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    output_dir: Path = field(default_factory=lambda: Path("data/eda/outputs"))
    plots_dir: Path = field(default_factory=lambda: Path("eda/plots"))
    results: Dict[str, Any] = field(default_factory=dict)

    # Private attributes for lazy loading
    _sales_data: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _calendar_data: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _pricing_data: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize paths and create output directories."""
        # Ensure paths are Path objects
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.plots_dir = Path(self.plots_dir)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"EDAContext initialized with data_dir={self.data_dir}, "
                   f"output_dir={self.output_dir}, plots_dir={self.plots_dir}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EDAContext":
        """
        Create EDAContext from configuration dictionary.

        Args:
            config: Configuration dictionary with optional keys:
                   - data_dir: Data directory path
                   - output_dir: Output directory path
                   - plots_dir: Plots directory path

        Returns:
            Configured EDAContext instance
        """
        return cls(
            data_dir=Path(config.get("data_dir", "data/raw")),
            output_dir=Path(config.get("output_dir", "data/eda/outputs")),
            plots_dir=Path(config.get("plots_dir", "eda/plots"))
        )

    @property
    def sales_data(self) -> pd.DataFrame:
        """Lazily load and return sales training data."""
        if self._sales_data is None:
            sales_path = self.data_dir / "sales_train_validation.csv"
            if sales_path.exists():
                logger.info(f"Loading sales data from {sales_path}")
                self._sales_data = pd.read_csv(sales_path)
            else:
                logger.warning(f"Sales data not found at {sales_path}")
                self._sales_data = pd.DataFrame()
        return self._sales_data

    @property
    def calendar_data(self) -> pd.DataFrame:
        """Lazily load and return calendar data."""
        if self._calendar_data is None:
            calendar_path = self.data_dir / "calendar.csv"
            if calendar_path.exists():
                logger.info(f"Loading calendar data from {calendar_path}")
                self._calendar_data = pd.read_csv(calendar_path)
            else:
                logger.warning(f"Calendar data not found at {calendar_path}")
                self._calendar_data = pd.DataFrame()
        return self._calendar_data

    @property
    def pricing_data(self) -> pd.DataFrame:
        """Lazily load and return pricing data."""
        if self._pricing_data is None:
            pricing_path = self.data_dir / "sell_prices.csv"
            if pricing_path.exists():
                logger.info(f"Loading pricing data from {pricing_path}")
                self._pricing_data = pd.read_csv(pricing_path)
            else:
                logger.warning(f"Pricing data not found at {pricing_path}")
                self._pricing_data = pd.DataFrame()
        return self._pricing_data

    def save_result(self, step_key: str, result: Any) -> None:
        """
        Save analysis result for later retrieval.

        Args:
            step_key: Unique identifier for the analysis step
            result: Analysis result (will be JSON serialized)

        Raises:
            ValueError: If step_key is invalid or unsafe for file paths
        """
        # Input validation for security and safety
        if not step_key or not isinstance(step_key, str):
            raise ValueError(f"Invalid step_key: {step_key}")

        # Sanitize step_key to prevent path traversal and ensure safe filename
        if not step_key.replace('_', '').replace('-', '').replace('.', '').isalnum():
            raise ValueError(f"Invalid step_key: {step_key}")

        # Additional check for path traversal attempts
        if '..' in step_key or '/' in step_key or '\\' in step_key:
            raise ValueError(f"Invalid step_key: {step_key}")

        self.results[step_key] = result

        # Also save to disk for persistence
        result_file = self.output_dir / f"{step_key}_result.json"
        try:
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Saved result for {step_key} to {result_file}")
        except (TypeError, ValueError) as e:
            logger.warning(f"Could not serialize result for {step_key}: {e}")

    def get_result(self, step_key: str) -> Optional[Any]:
        """
        Retrieve previously saved analysis result.

        Args:
            step_key: Unique identifier for the analysis step

        Returns:
            Saved result or None if not found
        """
        # First check in-memory cache
        if step_key in self.results:
            return self.results[step_key]

        # Try loading from disk
        result_file = self.output_dir / f"{step_key}_result.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                self.results[step_key] = result  # Cache in memory
                logger.info(f"Loaded result for {step_key} from {result_file}")
                return result
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load result for {step_key}: {e}")

        return None

    def save_plot(self, filename: str, step_subdir: Optional[str] = None) -> Path:
        """
        Generate standardized plot save path.

        Args:
            filename: Plot filename (should include extension)
            step_subdir: Optional subdirectory for organizing plots by step

        Returns:
            Full path where plot should be saved
        """
        if step_subdir:
            plot_path = self.plots_dir / step_subdir / filename
            plot_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            plot_path = self.plots_dir / filename

        return plot_path

    def clear_cache(self) -> None:
        """Clear all cached data and results."""
        self._sales_data = None
        self._calendar_data = None
        self._pricing_data = None
        self.results.clear()
        logger.info("Cleared EDAContext cache")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the context state.

        Returns:
            Dictionary containing context state summary
        """
        return {
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "plots_dir": str(self.plots_dir),
            "cached_datasets": {
                "sales": self._sales_data is not None,
                "calendar": self._calendar_data is not None,
                "pricing": self._pricing_data is not None
            },
            "cached_results": list(self.results.keys()),
            "data_shapes": {
                "sales": self.sales_data.shape if not self.sales_data.empty else (0, 0),
                "calendar": self.calendar_data.shape if not self.calendar_data.empty else (0, 0),
                "pricing": self.pricing_data.shape if not self.pricing_data.empty else (0, 0)
            }
        }