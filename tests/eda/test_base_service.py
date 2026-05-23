import pytest
from unittest.mock import Mock
from src.demand_forecast_intelligence.eda.utils.core.context import EDAContext
from src.demand_forecast_intelligence.eda.utils.core.base_service import BaseEDAService


class TestEDAService(BaseEDAService):
    """Concrete test implementation of BaseEDAService."""

    def execute(self):
        return {"test_step": "completed", "findings": ["test finding"]}

    def validate_prerequisites(self):
        return []

    def generate_summary(self):
        return "Test service completed successfully."


def test_base_service_initialization():
    """Test BaseEDAService initializes with context."""
    ctx = EDAContext()
    service = TestEDAService(ctx)

    assert service.ctx == ctx


def test_base_service_execute_abstract():
    """Test that BaseEDAService.execute is abstract."""
    ctx = EDAContext()

    with pytest.raises(TypeError):
        BaseEDAService(ctx)


def test_concrete_service_execute():
    """Test concrete service execution."""
    ctx = EDAContext()
    service = TestEDAService(ctx)

    result = service.execute()
    assert result["test_step"] == "completed"
    assert "findings" in result


def test_service_prerequisite_validation():
    """Test prerequisite validation."""
    ctx = EDAContext()
    service = TestEDAService(ctx)

    missing = service.validate_prerequisites()
    assert isinstance(missing, list)
    assert len(missing) == 0


def test_service_summary_generation():
    """Test summary generation."""
    ctx = EDAContext()
    service = TestEDAService(ctx)

    summary = service.generate_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_helper_methods():
    """Test helper methods for result management."""
    ctx = EDAContext()
    service = TestEDAService(ctx)

    # Test save and get step result
    test_result = {"analysis": "completed"}
    service._save_step_result("test_step", test_result)

    retrieved = service._get_step_result("test_step")
    assert retrieved == test_result


def test_plot_path_generation():
    """Test plot path helper method."""
    ctx = EDAContext()
    service = TestEDAService(ctx)

    path = service._create_plot_path("step_05_feature_profiling", "sales_distributions")
    expected = "step_05_feature_profiling/sales_distributions.png"
    assert path == expected