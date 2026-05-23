import pytest
from src.demand_forecast_intelligence.eda.config import get_m5_context
from src.demand_forecast_intelligence.eda.utils.services.data_understanding.business_context import BusinessContextService


def test_business_context_service_initialization():
    """Test BusinessContextService initializes correctly."""
    ctx = get_m5_context()
    service = BusinessContextService(ctx)

    assert service.ctx == ctx


def test_analyze_problem_definition():
    """Test problem definition analysis."""
    ctx = get_m5_context()
    service = BusinessContextService(ctx)

    result = service.analyze_problem_definition()

    assert "forecasting_objective" in result
    assert "segmentation_objective" in result

    forecasting = result["forecasting_objective"]
    assert "target_variable" in forecasting
    assert "problem_type" in forecasting
    assert "evaluation_metrics" in forecasting


def test_identify_available_features():
    """Test available features identification."""
    ctx = get_m5_context()
    service = BusinessContextService(ctx)

    result = service.identify_available_features()

    assert "available_features" in result
    assert "forbidden_features" in result
    assert isinstance(result["available_features"], list)
    assert len(result["available_features"]) > 0


def test_define_leakage_risks():
    """Test leakage risk definition."""
    ctx = get_m5_context()
    service = BusinessContextService(ctx)

    result = service.define_leakage_risks()

    assert "temporal_leakage" in result
    assert "pricing_leakage" in result
    assert "event_leakage" in result


def test_validate_prerequisites():
    """Test prerequisite validation (should have no dependencies)."""
    ctx = get_m5_context()
    service = BusinessContextService(ctx)

    missing = service.validate_prerequisites()
    assert missing == []


def test_execute():
    """Test complete service execution."""
    ctx = get_m5_context()
    service = BusinessContextService(ctx)

    result = service.execute()

    assert "step_1" in result
    step_result = result["step_1"]

    assert "problem_definition" in step_result
    assert "available_features" in step_result
    assert "leakage_risks" in step_result
    assert "key_findings" in step_result


def test_generate_summary():
    """Test summary generation."""
    ctx = get_m5_context()
    service = BusinessContextService(ctx)

    # Execute first to have results
    service.execute()

    summary = service.generate_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "Business Context Analysis" in summary


def test_create_problem_definition_plot():
    """Test plot generation."""
    ctx = get_m5_context()
    service = BusinessContextService(ctx)

    # Should not raise any errors
    service._create_problem_definition_plot()

    # Check that plot was saved to context
    # This is a basic test - the plot file should be created