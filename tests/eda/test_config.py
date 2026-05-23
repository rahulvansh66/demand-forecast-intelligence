"""
Test cases for M5 EDA configuration system.

These tests verify the M5-specific configuration structure, parameters,
and factory functions used throughout the EDA pipeline.
"""

import pytest
from pathlib import Path
from src.demand_forecast_intelligence.eda.config import (
    M5_CONFIG,
    get_m5_context,
    get_step_config,
    validate_m5_data_availability,
    get_m5_hierarchy_config,
    get_analysis_thresholds
)


def test_m5_config_structure():
    """Test M5 configuration has required structure."""
    assert "data_paths" in M5_CONFIG
    assert "output_paths" in M5_CONFIG
    assert "analysis_params" in M5_CONFIG
    assert "m5_specifics" in M5_CONFIG


def test_m5_data_paths():
    """Test M5 data path configuration."""
    data_paths = M5_CONFIG["data_paths"]

    assert "sales_validation" in data_paths
    assert "sales_evaluation" in data_paths
    assert "calendar" in data_paths
    assert "pricing" in data_paths

    # Check paths are strings
    assert isinstance(data_paths["sales_validation"], str)


def test_m5_analysis_params():
    """Test M5 analysis parameters."""
    params = M5_CONFIG["analysis_params"]

    assert "outlier_threshold" in params
    assert "correlation_threshold" in params
    assert "missing_threshold" in params
    assert "time_split_date" in params

    # Check parameter types and ranges
    assert 0 < params["outlier_threshold"] <= 1
    assert 0 < params["correlation_threshold"] <= 1


def test_get_m5_context():
    """Test M5 context factory function."""
    ctx = get_m5_context()

    assert ctx.data_dir == Path("data/raw")
    assert ctx.output_dir == Path("data/eda/outputs")
    assert ctx.plots_dir == Path("eda/plots")
    assert ctx.config == M5_CONFIG


def test_m5_specifics():
    """Test M5 dataset specific parameters."""
    specifics = M5_CONFIG["m5_specifics"]

    assert specifics["item_store_combinations"] == 30490
    assert "FOODS" in specifics["categories"]
    assert "CA" in specifics["states"]
    assert specifics["training_days"] == "d_1 to d_1913"


def test_get_step_config():
    """Test step-specific configuration retrieval."""
    # Test valid step
    step1_config = get_step_config("step1_overview")
    assert "plots_subdir" in step1_config
    assert "sample_size" in step1_config
    assert step1_config["sample_size"] == 1000

    # Test that analysis params are merged
    assert "outlier_threshold" in step1_config
    assert step1_config["outlier_threshold"] == M5_CONFIG["analysis_params"]["outlier_threshold"]

    # Test invalid step raises error
    with pytest.raises(ValueError, match="Unknown step"):
        get_step_config("invalid_step")


def test_validate_m5_data_availability(tmp_path):
    """Test M5 data availability validation."""
    # Create some test files
    (tmp_path / "sales_train_validation.csv").touch()
    (tmp_path / "calendar.csv").touch()

    availability = validate_m5_data_availability(tmp_path)

    assert availability["sales_validation"] is True
    assert availability["calendar"] is True
    assert availability["sales_evaluation"] is False  # Not created
    assert availability["pricing"] is False  # Not created


def test_get_m5_hierarchy_config():
    """Test M5 hierarchy configuration."""
    hierarchy = get_m5_hierarchy_config()

    assert "levels" in hierarchy
    assert "aggregation_keys" in hierarchy
    assert "hierarchy_mappings" in hierarchy

    # Test hierarchy levels
    expected_levels = ["total", "state", "store", "category", "department", "item"]
    assert hierarchy["levels"] == expected_levels

    # Test state aggregation keys
    assert "CA" in hierarchy["aggregation_keys"]["state"]
    assert "TX" in hierarchy["aggregation_keys"]["state"]
    assert "WI" in hierarchy["aggregation_keys"]["state"]

    # Test store aggregation keys
    stores = hierarchy["aggregation_keys"]["store"]
    assert "CA_1" in stores
    assert "TX_1" in stores
    assert "WI_1" in stores


def test_get_analysis_thresholds():
    """Test analysis thresholds retrieval."""
    thresholds = get_analysis_thresholds()

    # Test required thresholds exist
    required_thresholds = [
        "missing_data", "outlier_detection", "high_correlation",
        "low_variance", "sparse_series", "intermittent_series",
        "high_volume", "price_change", "significance"
    ]

    for threshold in required_thresholds:
        assert threshold in thresholds
        assert isinstance(thresholds[threshold], (int, float))

    # Test threshold value ranges
    assert 0 < thresholds["missing_data"] < 1
    assert 0 < thresholds["significance"] < 1
    assert thresholds["high_volume"] > 0


def test_m5_config_completeness():
    """Test M5 config contains all expected sections and values."""
    # Test top-level sections
    expected_sections = ["data_paths", "output_paths", "analysis_params", "m5_specifics"]
    for section in expected_sections:
        assert section in M5_CONFIG

    # Test data paths completeness
    expected_data_files = ["sales_validation", "sales_evaluation", "calendar", "pricing"]
    for file_type in expected_data_files:
        assert file_type in M5_CONFIG["data_paths"]
        assert M5_CONFIG["data_paths"][file_type].endswith(".csv")

    # Test output paths
    assert "step_subdirs" in M5_CONFIG["output_paths"]
    step_subdirs = M5_CONFIG["output_paths"]["step_subdirs"]
    assert len(step_subdirs) >= 9  # Should have subdirs for all EDA steps

    # Test M5 specifics completeness
    m5_specifics = M5_CONFIG["m5_specifics"]
    required_m5_keys = [
        "item_store_combinations", "total_items", "total_stores",
        "states", "categories", "departments", "stores_per_state"
    ]
    for key in required_m5_keys:
        assert key in m5_specifics


def test_get_m5_context_with_custom_paths(tmp_path):
    """Test M5 context creation with custom paths."""
    custom_data = tmp_path / "data"
    custom_output = tmp_path / "output"
    custom_plots = tmp_path / "plots"

    ctx = get_m5_context(
        data_dir=str(custom_data),
        output_dir=str(custom_output),
        plots_dir=str(custom_plots)
    )

    assert ctx.data_dir == Path(custom_data)
    assert ctx.output_dir == Path(custom_output)
    assert ctx.plots_dir == Path(custom_plots)
    assert ctx.config == M5_CONFIG