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