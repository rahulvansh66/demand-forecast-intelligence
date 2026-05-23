import pytest
import pandas as pd
from pathlib import Path
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