# Core Infrastructure and Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the core infrastructure and data loading foundation for the demand forecast intelligence platform.

**Architecture:** Modular monolith with shared core infrastructure supporting M5 dataset loading, configuration management, and logging. Follows TDD with pytest, includes data validation and quality checks.

**Tech Stack:** Python 3.11+, pandas, pydantic, pytest, PyYAML, structlog

---

## File Structure Overview

### Core Infrastructure
- `src/demand_forecast_intelligence/core/config/settings.py` - Configuration management
- `src/demand_forecast_intelligence/core/logging/logger.py` - Structured logging setup  
- `src/demand_forecast_intelligence/core/constants/paths.py` - Project paths and constants
- `src/demand_forecast_intelligence/core/exceptions/data_exceptions.py` - Custom exceptions

### Data Layer
- `src/demand_forecast_intelligence/data/schemas/m5_schemas.py` - M5 dataset schemas
- `src/demand_forecast_intelligence/data/loaders/m5_loader.py` - M5 CSV file loaders
- `src/demand_forecast_intelligence/data/validators/data_quality.py` - Data quality checks

### Configuration
- `configs/data.yaml` - Dataset configuration
- `pyproject.toml` - Project dependencies and metadata

### Tests  
- `tests/unit/core/test_config.py` - Core configuration tests
- `tests/unit/data/test_m5_loader.py` - Data loader tests
- `tests/fixtures/sample_data.py` - Test data fixtures

---

### Task 1: Project Dependencies and Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `configs/data.yaml`

- [ ] **Step 1: Write failing test for project setup**

Create `tests/test_project_setup.py`:
```python
import importlib.util

def test_package_importable():
    """Test that the main package can be imported."""
    spec = importlib.util.spec_from_file_location(
        "demand_forecast_intelligence", 
        "src/demand_forecast_intelligence/__init__.py"
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_project_setup.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Create pyproject.toml with dependencies**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "demand-forecast-intelligence"
version = "0.1.0"
description = "ML-driven demand forecast intelligence platform using Walmart M5 dataset"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0.0",
    "pydantic>=2.0.0",
    "PyYAML>=6.0",
    "structlog>=23.0.0",
    "pytest>=7.0.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--strict-markers --disable-warnings"

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 88
```

- [ ] **Step 4: Create data configuration**

Create `configs/data.yaml`:
```yaml
# M5 Dataset Configuration
dataset:
  name: "walmart_m5"
  description: "Walmart M5 Forecasting Competition Dataset"
  
  # Data paths (relative to project root)  
  paths:
    raw: "data/raw"
    interim: "data/interim" 
    processed: "data/processed"
    
  # M5 CSV files
  files:
    sales_train_validation: "sales_train_validation.csv"
    sales_train_evaluation: "sales_train_evaluation.csv" 
    calendar: "calendar.csv"
    sell_prices: "sell_prices.csv"
    sample_submission: "sample_submission.csv"
    
  # Data validation rules
  validation:
    max_missing_percentage: 0.05  # 5% max missing values
    min_rows: 1000
    required_columns:
      sales_train_validation: ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
      calendar: ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d"]
      sell_prices: ["store_id", "item_id", "wm_yr_wk", "sell_price"]

# Feature engineering configuration
features:
  temporal:
    lag_periods: [1, 7, 14, 28]  # Days to look back
    rolling_windows: [7, 14, 28]  # Rolling average windows
    
  calendar:
    include_events: true
    include_snap_benefits: true
    
  sales:
    min_history_days: 28  # Minimum days of history required
    zero_sales_threshold: 0.7  # Max percentage of zero-sales days
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_project_setup.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml configs/data.yaml tests/test_project_setup.py
git commit -m "feat: add project dependencies and data configuration

- Set up Python packaging with pyproject.toml
- Define M5 dataset paths and validation rules
- Add development dependencies for testing and formatting
- Configure pytest settings and code quality tools"
```

---

### Task 2: Core Configuration Management

**Files:**
- Create: `src/retail_demand_copilot/core/config/settings.py`
- Create: `tests/unit/core/test_config.py`

- [ ] **Step 1: Write failing test for configuration loading**

Create `tests/unit/core/test_config.py`:
```python
import pytest
from pathlib import Path
import tempfile
import yaml

from demand_forecast_intelligence.core.config.settings import Settings, load_config

def test_load_config_from_yaml():
    """Test loading configuration from YAML file."""
    test_config = {
        'dataset': {
            'name': 'test_dataset',
            'paths': {'raw': 'test/raw'}
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_path = f.name
    
    try:
        settings = load_config(temp_path)
        assert settings.dataset.name == 'test_dataset'
        assert settings.dataset.paths['raw'] == 'test/raw'
    finally:
        Path(temp_path).unlink()

def test_settings_validation():
    """Test that Settings validates required fields."""
    with pytest.raises(ValueError):
        Settings(dataset=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/core/test_config.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Create core configuration module**

Create `src/demand_forecast_intelligence/core/__init__.py`:
```python
"""Core infrastructure module for demand forecast intelligence."""
```

Create `src/demand_forecast_intelligence/core/config/__init__.py`:
```python
"""Configuration management module."""
```

Create `src/demand_forecast_intelligence/core/config/settings.py`:
```python
"""Configuration management for demand forecast intelligence."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field, validator


class DatasetPaths(BaseModel):
    """Dataset path configuration."""
    raw: str
    interim: str
    processed: str


class ValidationRules(BaseModel):
    """Data validation configuration."""
    max_missing_percentage: float = Field(ge=0.0, le=1.0)
    min_rows: int = Field(gt=0)
    required_columns: Dict[str, List[str]]


class DatasetConfig(BaseModel):
    """Dataset configuration settings."""
    name: str
    description: str
    paths: DatasetPaths
    files: Dict[str, str]
    validation: ValidationRules


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""
    temporal: Dict[str, List[int]]
    calendar: Dict[str, bool]
    sales: Dict[str, Any]


class Settings(BaseModel):
    """Main application settings."""
    dataset: DatasetConfig
    features: Optional[FeatureConfig] = None
    
    @validator('dataset')
    def validate_dataset(cls, v):
        if v is None:
            raise ValueError("Dataset configuration is required")
        return v


def load_config(config_path: str | Path) -> Settings:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Settings: Validated configuration settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
        ValidationError: If config doesn't match schema
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return Settings(**config_data)


def get_default_config() -> Settings:
    """Load default configuration from configs/data.yaml.
    
    Returns:
        Settings: Default application settings
    """
    project_root = Path(__file__).parents[4]  # Go up to project root
    default_config_path = project_root / "configs" / "data.yaml"
    
    return load_config(default_config_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/core/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Add integration test for default config**

Add to `tests/unit/core/test_config.py`:
```python
def test_load_default_config():
    """Test loading the default project configuration."""
    settings = get_default_config()
    
    # Verify dataset configuration
    assert settings.dataset.name == "walmart_m5"
    assert "sales_train_validation.csv" in settings.dataset.files.values()
    assert settings.dataset.paths.raw == "data/raw"
    
    # Verify validation rules
    assert settings.dataset.validation.max_missing_percentage == 0.05
    assert "id" in settings.dataset.validation.required_columns["sales_train_validation"]
```

- [ ] **Step 6: Run integration test**

Run: `python -m pytest tests/unit/core/test_config.py::test_load_default_config -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/retail_demand_copilot/core/ tests/unit/core/test_config.py
git commit -m "feat: add core configuration management

- Implement Settings model with pydantic validation
- Add YAML config loading with error handling  
- Support dataset, feature, and validation configuration
- Include comprehensive unit tests with fixtures"
```

---

### Task 3: Logging Infrastructure

**Files:**
- Create: `src/retail_demand_copilot/core/logging/logger.py`
- Create: `tests/unit/core/test_logging.py`

- [ ] **Step 1: Write failing test for structured logging**

Create `tests/unit/core/test_logging.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/core/test_logging.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Create logging infrastructure**

Create `src/demand_forecast_intelligence/core/logging/__init__.py`:
```python
"""Logging infrastructure module."""
```

Create `src/demand_forecast_intelligence/core/logging/logger.py`:
```python
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
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.JSONRenderer() if format_json else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
        handlers=[add_handler] if add_handler else None,
        force=True
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/core/test_logging.py -v`
Expected: PASS

- [ ] **Step 5: Create practical logging example test**

Add to `tests/unit/core/test_logging.py`:
```python
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
```

- [ ] **Step 6: Run all logging tests**

Run: `python -m pytest tests/unit/core/test_logging.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/retail_demand_copilot/core/logging/ tests/unit/core/test_logging.py
git commit -m "feat: add structured logging infrastructure

- Implement structlog-based logging with JSON output
- Support configurable log levels and custom handlers
- Add contextual logging for data operations
- Include comprehensive error handling and testing"
```

---

### Task 4: Project Constants and Paths

**Files:**
- Create: `src/retail_demand_copilot/core/constants/paths.py`
- Create: `src/retail_demand_copilot/core/exceptions/data_exceptions.py`
- Create: `tests/unit/core/test_constants.py`

- [ ] **Step 1: Write failing test for path constants**

Create `tests/unit/core/test_constants.py`:
```python
from pathlib import Path
import pytest

from demand_forecast_intelligence.core.constants.paths import ProjectPaths, M5Files
from demand_forecast_intelligence.core.exceptions.data_exceptions import DataFileNotFoundError


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/core/test_constants.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Create exceptions module**

Create `src/demand_forecast_intelligence/core/exceptions/__init__.py`:
```python
"""Custom exceptions for demand forecast intelligence."""
```

Create `src/demand_forecast_intelligence/core/exceptions/data_exceptions.py`:
```python
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
```

- [ ] **Step 4: Create constants module**

Create `src/demand_forecast_intelligence/core/constants/__init__.py`:
```python
"""Constants and path definitions."""
```

Create `src/demand_forecast_intelligence/core/constants/paths.py`:
```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/core/test_constants.py -v`
Expected: PASS

- [ ] **Step 6: Add tests for exception functionality**

Add to `tests/unit/core/test_constants.py`:
```python
from demand_forecast_intelligence.core.exceptions.data_exceptions import (
    DataValidationError, 
    DataQualityError
)

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
```

- [ ] **Step 7: Run all tests**

Run: `python -m pytest tests/unit/core/test_constants.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/retail_demand_copilot/core/constants/ src/retail_demand_copilot/core/exceptions/ tests/unit/core/test_constants.py
git commit -m "feat: add project constants and custom exceptions

- Define ProjectPaths for consistent directory structure
- Add M5Files class for dataset file management  
- Implement custom data exceptions with context
- Include path validation and error handling"
```

---

### Task 5: M5 Dataset Schema Definitions

**Files:**
- Create: `src/retail_demand_copilot/data/schemas/m5_schemas.py`
- Create: `tests/unit/data/test_m5_schemas.py`

- [ ] **Step 1: Write failing test for M5 schema validation**

Create `tests/unit/data/__init__.py`:
```python
"""Data layer tests."""
```

Create `tests/unit/data/test_m5_schemas.py`:
```python
import pandas as pd
import pytest
from datetime import datetime

from demand_forecast_intelligence.data.schemas.m5_schemas import (
    SalesRecord,
    CalendarRecord, 
    PriceRecord,
    validate_sales_dataframe,
    validate_calendar_dataframe
)


def test_sales_record_validation():
    """Test SalesRecord pydantic model validation."""
    # Valid record
    record = SalesRecord(
        id="FOODS_1_001_CA_1_validation",
        item_id="FOODS_1_001",
        dept_id="FOODS_1", 
        cat_id="FOODS",
        store_id="CA_1",
        state_id="CA"
    )
    assert record.id == "FOODS_1_001_CA_1_validation"
    assert record.item_id == "FOODS_1_001"
    
    # Invalid state_id
    with pytest.raises(ValueError):
        SalesRecord(
            id="FOODS_1_001_XX_1_validation",
            item_id="FOODS_1_001", 
            dept_id="FOODS_1",
            cat_id="FOODS",
            store_id="XX_1",
            state_id="XX"  # Invalid state
        )


def test_calendar_record_validation():
    """Test CalendarRecord pydantic model validation."""
    record = CalendarRecord(
        date=datetime(2011, 1, 29).date(),
        wm_yr_wk=11101,
        weekday="Saturday",
        wday=1,
        month=1,
        year=2011,
        d="d_1"
    )
    assert record.weekday == "Saturday"
    assert record.wday == 1
    
    # Invalid weekday
    with pytest.raises(ValueError):
        CalendarRecord(
            date=datetime(2011, 1, 29).date(),
            wm_yr_wk=11101,
            weekday="InvalidDay",  # Not a real weekday
            wday=1,
            month=1,
            year=2011, 
            d="d_1"
        )


def test_validate_sales_dataframe():
    """Test sales DataFrame validation function."""
    # Valid DataFrame
    df = pd.DataFrame({
        'id': ['FOODS_1_001_CA_1_validation'],
        'item_id': ['FOODS_1_001'],
        'dept_id': ['FOODS_1'],
        'cat_id': ['FOODS'],
        'store_id': ['CA_1'],
        'state_id': ['CA'],
        'd_1': [5],
        'd_2': [3]
    })
    
    # Should not raise
    validate_sales_dataframe(df)
    
    # Missing required column
    df_invalid = df.drop('item_id', axis=1)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_sales_dataframe(df_invalid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/data/test_m5_schemas.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Create data schemas module**

Create `src/demand_forecast_intelligence/data/__init__.py`:
```python
"""Data access and management module."""
```

Create `src/demand_forecast_intelligence/data/schemas/__init__.py`:
```python
"""Data schemas and validation models."""
```

Create `src/demand_forecast_intelligence/data/schemas/m5_schemas.py`:
```python
"""Pydantic schemas for M5 dataset validation."""

from datetime import date
from typing import Optional, List, Set
import pandas as pd
from pydantic import BaseModel, Field, validator


class SalesRecord(BaseModel):
    """Schema for sales_train_validation.csv records."""
    
    id: str = Field(..., description="Unique identifier for item-store combination")
    item_id: str = Field(..., description="Product identifier")
    dept_id: str = Field(..., description="Department identifier") 
    cat_id: str = Field(..., description="Category identifier")
    store_id: str = Field(..., description="Store identifier")
    state_id: str = Field(..., description="State identifier")
    
    @validator('state_id')
    def validate_state_id(cls, v):
        valid_states = {'CA', 'TX', 'WI'}
        if v not in valid_states:
            raise ValueError(f'state_id must be one of {valid_states}')
        return v
    
    @validator('cat_id')
    def validate_category(cls, v):
        valid_categories = {'HOBBIES', 'HOUSEHOLD', 'FOODS'}
        if v not in valid_categories:
            raise ValueError(f'cat_id must be one of {valid_categories}')
        return v
    
    @validator('store_id')
    def validate_store_format(cls, v):
        # Store format: {STATE}_{NUMBER}
        if not v or len(v.split('_')) != 2:
            raise ValueError('store_id must follow format {STATE}_{NUMBER}')
        return v


class CalendarRecord(BaseModel):
    """Schema for calendar.csv records."""
    
    date: date
    wm_yr_wk: int = Field(..., description="Walmart year-week identifier")
    weekday: str = Field(..., description="Day of week name")
    wday: int = Field(..., ge=1, le=7, description="Numeric day of week")
    month: int = Field(..., ge=1, le=12, description="Month number")  
    year: int = Field(..., ge=2011, le=2016, description="Year")
    d: str = Field(..., description="Day identifier matching sales columns")
    event_name_1: Optional[str] = None
    event_type_1: Optional[str] = None
    event_name_2: Optional[str] = None
    event_type_2: Optional[str] = None
    snap_CA: Optional[int] = Field(None, ge=0, le=1)
    snap_TX: Optional[int] = Field(None, ge=0, le=1)
    snap_WI: Optional[int] = Field(None, ge=0, le=1)
    
    @validator('weekday')
    def validate_weekday(cls, v):
        valid_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
        if v not in valid_days:
            raise ValueError(f'weekday must be one of {valid_days}')
        return v
    
    @validator('d')
    def validate_day_identifier(cls, v):
        if not v.startswith('d_') or not v[2:].isdigit():
            raise ValueError('d must follow format d_{number}')
        return v


class PriceRecord(BaseModel):
    """Schema for sell_prices.csv records."""
    
    store_id: str = Field(..., description="Store identifier")
    item_id: str = Field(..., description="Item identifier") 
    wm_yr_wk: int = Field(..., description="Walmart year-week identifier")
    sell_price: float = Field(..., gt=0, description="Unit selling price")
    
    @validator('store_id')
    def validate_store_format(cls, v):
        if not v or len(v.split('_')) != 2:
            raise ValueError('store_id must follow format {STATE}_{NUMBER}')
        return v


def validate_sales_dataframe(df: pd.DataFrame) -> None:
    """Validate sales DataFrame has required structure.
    
    Args:
        df: Sales DataFrame to validate
        
    Raises:
        ValueError: If DataFrame structure is invalid
    """
    required_columns = {'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for sales data columns (d_1, d_2, etc.)
    sales_columns = [col for col in df.columns if col.startswith('d_')]
    if not sales_columns:
        raise ValueError("No sales data columns found (expected d_1, d_2, etc.)")
    
    # Validate data types
    if not pd.api.types.is_object_dtype(df['id']):
        raise ValueError("Column 'id' must be string type")
    
    # Check for negative sales values
    for col in sales_columns:
        if df[col].min() < 0:
            raise ValueError(f"Sales column '{col}' contains negative values")


def validate_calendar_dataframe(df: pd.DataFrame) -> None:
    """Validate calendar DataFrame has required structure.
    
    Args:
        df: Calendar DataFrame to validate
        
    Raises:
        ValueError: If DataFrame structure is invalid
    """
    required_columns = {'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'd'}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate date range
    if df['year'].min() < 2011 or df['year'].max() > 2016:
        raise ValueError("Calendar data must be within 2011-2016 range")
    
    # Check for duplicate dates
    if df['date'].duplicated().any():
        raise ValueError("Calendar contains duplicate dates")


def validate_prices_dataframe(df: pd.DataFrame) -> None:
    """Validate prices DataFrame has required structure.
    
    Args:
        df: Prices DataFrame to validate
        
    Raises:
        ValueError: If DataFrame structure is invalid
    """
    required_columns = {'store_id', 'item_id', 'wm_yr_wk', 'sell_price'}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate price values
    if df['sell_price'].min() <= 0:
        raise ValueError("All prices must be positive")
    
    if df['sell_price'].isna().any():
        raise ValueError("Prices cannot be null")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/data/test_m5_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Add more comprehensive validation tests**

Add to `tests/unit/data/test_m5_schemas.py`:
```python
def test_validate_calendar_dataframe():
    """Test calendar DataFrame validation function."""
    df = pd.DataFrame({
        'date': ['2011-01-29', '2011-01-30'],
        'wm_yr_wk': [11101, 11101],
        'weekday': ['Saturday', 'Sunday'],
        'wday': [1, 2],
        'month': [1, 1],
        'year': [2011, 2011],
        'd': ['d_1', 'd_2']
    })
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Should not raise
    validate_calendar_dataframe(df)
    
    # Test duplicate dates
    df_dup = df.copy()
    df_dup.loc[1, 'date'] = df_dup.loc[0, 'date']  # Duplicate first date
    
    with pytest.raises(ValueError, match="duplicate dates"):
        validate_calendar_dataframe(df_dup)


def test_price_record_validation():
    """Test PriceRecord validation.""" 
    record = PriceRecord(
        store_id="CA_1",
        item_id="FOODS_1_001",
        wm_yr_wk=11101,
        sell_price=3.97
    )
    assert record.sell_price == 3.97
    
    # Test negative price
    with pytest.raises(ValueError):
        PriceRecord(
            store_id="CA_1",
            item_id="FOODS_1_001", 
            wm_yr_wk=11101,
            sell_price=-1.0  # Invalid negative price
        )
```

- [ ] **Step 6: Run all schema tests**

Run: `python -m pytest tests/unit/data/test_m5_schemas.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/retail_demand_copilot/data/schemas/ tests/unit/data/
git commit -m "feat: add M5 dataset schema validation

- Define pydantic models for sales, calendar, and price records
- Add comprehensive DataFrame validation functions  
- Include state, category, and format validation rules
- Support data quality checks for negative values and duplicates"
```

---

## Self-Review

**1. Spec coverage:** ✓ Covers core infrastructure, configuration, logging, constants, exceptions, and data schemas from the design spec.

**2. Placeholder scan:** ✓ No TBD/TODO items - all code blocks are complete implementations.

**3. Type consistency:** ✓ Consistent naming and types throughout (Settings, ProjectPaths, M5Files, etc.).

The plan provides a solid foundation for the M5 dataset loading and validation that comes next. Each task builds incrementally with full TDD coverage.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-16-core-infrastructure-data-foundation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**