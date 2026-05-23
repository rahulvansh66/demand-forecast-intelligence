import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open
from src.demand_forecast_intelligence.eda.utils.core.context import EDAContext


def test_eda_context_initialization():
    """Test EDAContext initializes with correct default paths."""
    ctx = EDAContext()

    assert ctx.data_dir == Path("data/raw")
    assert ctx.output_dir == Path("data/eda/outputs")
    assert ctx.plots_dir == Path("eda/plots")
    assert isinstance(ctx.results, dict)
    assert len(ctx.results) == 0


def test_eda_context_save_and_get_result():
    """Test results caching functionality."""
    ctx = EDAContext()

    test_result = {"analysis": "test_data", "score": 0.85}
    ctx.save_result("step_1", test_result)

    retrieved = ctx.get_result("step_1")
    assert retrieved == test_result


def test_eda_context_get_nonexistent_result():
    """Test getting non-existent result returns None."""
    ctx = EDAContext()

    result = ctx.get_result("step_999")
    assert result is None


def test_eda_context_from_config():
    """Test creating EDAContext from configuration."""
    config = {
        "data_dir": "custom/data",
        "output_dir": "custom/output"
    }

    ctx = EDAContext.from_config(config)
    assert ctx.data_dir == Path("custom/data")
    assert ctx.output_dir == Path("custom/output")


# Security and Input Validation Tests
def test_save_result_invalid_step_key_empty():
    """Test save_result rejects empty step_key."""
    ctx = EDAContext()

    with pytest.raises(ValueError, match="Invalid step_key"):
        ctx.save_result("", {"data": "test"})


def test_save_result_invalid_step_key_path_traversal():
    """Test save_result rejects path traversal attempts."""
    ctx = EDAContext()

    with pytest.raises(ValueError, match="Invalid step_key"):
        ctx.save_result("../../../etc/passwd", {"data": "test"})


def test_save_result_invalid_step_key_special_chars():
    """Test save_result rejects special characters."""
    ctx = EDAContext()

    with pytest.raises(ValueError, match="Invalid step_key"):
        ctx.save_result("step@1", {"data": "test"})


def test_save_result_valid_step_keys():
    """Test save_result accepts valid step_keys."""
    ctx = EDAContext()

    # These should all work
    ctx.save_result("step_1", {"data": "test"})
    ctx.save_result("step-2", {"data": "test"})
    ctx.save_result("step123", {"data": "test"})
    ctx.save_result("step_test_123", {"data": "test"})


# Lazy Loading Tests
@patch('pandas.read_csv')
def test_sales_data_lazy_loading(mock_read_csv):
    """Test sales data is loaded lazily and cached."""
    mock_df = pd.DataFrame({"item_id": ["FOODS_1_001"], "store_id": ["CA_1"]})
    mock_read_csv.return_value = mock_df

    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = EDAContext(data_dir=Path(temp_dir))
        # Create mock file
        (Path(temp_dir) / "sales_train_validation.csv").touch()

        # First access should load
        sales1 = ctx.sales_data
        mock_read_csv.assert_called_once()

        # Second access should use cache
        sales2 = ctx.sales_data
        mock_read_csv.assert_called_once()  # Still only called once

        assert sales1 is sales2  # Same object reference


@patch('pandas.read_csv')
def test_calendar_data_lazy_loading(mock_read_csv):
    """Test calendar data is loaded lazily and cached."""
    mock_df = pd.DataFrame({"date": ["2011-01-29"], "wm_yr_wk": [11101]})
    mock_read_csv.return_value = mock_df

    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = EDAContext(data_dir=Path(temp_dir))
        # Create mock file
        (Path(temp_dir) / "calendar.csv").touch()

        # First access should load
        calendar1 = ctx.calendar_data
        mock_read_csv.assert_called_once()

        # Second access should use cache
        calendar2 = ctx.calendar_data
        mock_read_csv.assert_called_once()  # Still only called once

        assert calendar1 is calendar2  # Same object reference


@patch('pandas.read_csv')
def test_pricing_data_lazy_loading(mock_read_csv):
    """Test pricing data is loaded lazily and cached."""
    mock_df = pd.DataFrame({"store_id": ["CA_1"], "item_id": ["FOODS_1_001"]})
    mock_read_csv.return_value = mock_df

    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = EDAContext(data_dir=Path(temp_dir))
        # Create mock file
        (Path(temp_dir) / "sell_prices.csv").touch()

        # First access should load
        pricing1 = ctx.pricing_data
        mock_read_csv.assert_called_once()

        # Second access should use cache
        pricing2 = ctx.pricing_data
        mock_read_csv.assert_called_once()  # Still only called once

        assert pricing1 is pricing2  # Same object reference


# Error Condition Tests
def test_lazy_loading_file_not_found():
    """Test lazy loading handles missing files gracefully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = EDAContext(data_dir=Path(temp_dir))

        # Files don't exist, should return empty DataFrames
        assert ctx.sales_data.empty
        assert ctx.calendar_data.empty
        assert ctx.pricing_data.empty


@patch('pandas.read_csv', side_effect=Exception("CSV read error"))
def test_lazy_loading_read_error(mock_read_csv):
    """Test lazy loading handles CSV read errors gracefully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = EDAContext(data_dir=Path(temp_dir))
        # Create mock files
        (Path(temp_dir) / "sales_train_validation.csv").touch()

        # Should raise the exception since we're not handling CSV errors yet
        with pytest.raises(Exception, match="CSV read error"):
            _ = ctx.sales_data


# Missing Methods Coverage Tests
def test_save_plot_basic():
    """Test save_plot returns correct path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = EDAContext(plots_dir=Path(temp_dir))

        plot_path = ctx.save_plot("test_plot.png")
        expected_path = Path(temp_dir) / "test_plot.png"

        assert plot_path == expected_path


def test_save_plot_with_subdirectory():
    """Test save_plot with subdirectory creates directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = EDAContext(plots_dir=Path(temp_dir))

        plot_path = ctx.save_plot("analysis.png", "step_1")
        expected_path = Path(temp_dir) / "step_1" / "analysis.png"

        assert plot_path == expected_path
        assert plot_path.parent.exists()


def test_clear_cache():
    """Test clear_cache clears all cached data and results."""
    ctx = EDAContext()

    # Add some data and results
    ctx.save_result("test_step", {"data": "test"})
    ctx._sales_data = pd.DataFrame({"test": [1, 2, 3]})
    ctx._calendar_data = pd.DataFrame({"test": [1, 2, 3]})
    ctx._pricing_data = pd.DataFrame({"test": [1, 2, 3]})

    # Verify data exists
    assert len(ctx.results) > 0
    assert ctx._sales_data is not None
    assert ctx._calendar_data is not None
    assert ctx._pricing_data is not None

    # Clear cache
    ctx.clear_cache()

    # Verify everything is cleared
    assert len(ctx.results) == 0
    assert ctx._sales_data is None
    assert ctx._calendar_data is None
    assert ctx._pricing_data is None


def test_get_summary():
    """Test get_summary returns correct context state."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = EDAContext(
            data_dir=Path(temp_dir) / "data",
            output_dir=Path(temp_dir) / "output",
            plots_dir=Path(temp_dir) / "plots"
        )

        # Add some cached data
        ctx._sales_data = pd.DataFrame({"test": [1, 2, 3]})
        ctx.save_result("test_step", {"data": "test"})

        summary = ctx.get_summary()

        assert summary["data_dir"] == str(ctx.data_dir)
        assert summary["output_dir"] == str(ctx.output_dir)
        assert summary["plots_dir"] == str(ctx.plots_dir)
        assert summary["cached_datasets"]["sales"] == True
        assert summary["cached_datasets"]["calendar"] == False
        assert summary["cached_datasets"]["pricing"] == False
        assert "test_step" in summary["cached_results"]
        assert summary["data_shapes"]["sales"] == (3, 1)  # 3 rows, 1 column


# Edge Cases and Configuration Tests
def test_from_config_partial():
    """Test from_config with partial configuration."""
    config = {"data_dir": "custom/data"}

    ctx = EDAContext.from_config(config)
    assert ctx.data_dir == Path("custom/data")
    assert ctx.output_dir == Path("data/eda/outputs")  # Default
    assert ctx.plots_dir == Path("eda/plots")  # Default


def test_from_config_all_paths():
    """Test from_config with all paths specified."""
    config = {
        "data_dir": "custom/data",
        "output_dir": "custom/output",
        "plots_dir": "custom/plots"
    }

    ctx = EDAContext.from_config(config)
    assert ctx.data_dir == Path("custom/data")
    assert ctx.output_dir == Path("custom/output")
    assert ctx.plots_dir == Path("custom/plots")