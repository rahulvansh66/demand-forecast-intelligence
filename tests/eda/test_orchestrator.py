import pytest
from unittest.mock import Mock, patch
from src.demand_forecast_intelligence.eda.utils.core.context import EDAContext
from src.demand_forecast_intelligence.eda.utils.core.orchestrator import EDAOrchestrator


def test_orchestrator_initialization():
    """Test EDAOrchestrator initializes with context."""
    ctx = EDAContext()
    orch = EDAOrchestrator(ctx)

    assert orch.ctx == ctx


def test_get_step_dependencies():
    """Test dependency mapping for EDA steps."""
    ctx = EDAContext()
    orch = EDAOrchestrator(ctx)

    # Step 1 has no dependencies
    assert orch.get_step_dependencies(1) == []

    # Step 3 depends on Step 1
    assert orch.get_step_dependencies(3) == [1]

    # Step 5 depends on Steps 1 and 3
    assert orch.get_step_dependencies(5) == [1, 3]

    # Step 15 depends on Steps 12 and 13
    assert orch.get_step_dependencies(15) == [12, 13]


def test_validate_step_prerequisites():
    """Test step prerequisite validation."""
    ctx = EDAContext()
    orch = EDAOrchestrator(ctx)

    # Step 1 should always be valid (no dependencies)
    missing = orch._validate_step_prerequisites(1)
    assert missing == []

    # Step 5 should be invalid without Steps 1 and 3
    missing = orch._validate_step_prerequisites(5)
    assert 1 in missing or 3 in missing


def test_get_phase_steps():
    """Test mapping phases to step numbers."""
    ctx = EDAContext()
    orch = EDAOrchestrator(ctx)

    phase_1_steps = orch._get_phase_steps(1)
    assert phase_1_steps == [1, 3]

    phase_2_steps = orch._get_phase_steps(2)
    assert phase_2_steps == [5, 6, 7]

    phase_4_steps = orch._get_phase_steps(4)
    assert phase_4_steps == [12, 13, 15]


@patch('src.demand_forecast_intelligence.eda.utils.core.orchestrator.EDAOrchestrator._run_single_step')
def test_run_phase(mock_run_step):
    """Test phase execution."""
    ctx = EDAContext()
    orch = EDAOrchestrator(ctx)

    mock_run_step.return_value = {"step": "completed"}

    result = orch.run_phase(1)

    # Should call run_step for each step in phase 1
    assert mock_run_step.call_count == 2
    assert "phase_1" in result


def test_subgroup_steps_mapping():
    """Test subgroup to steps mapping."""
    ctx = EDAContext()
    orch = EDAOrchestrator(ctx)

    # Test some subgroup mappings
    assert "1A" in orch.SUBGROUP_STEPS
    assert "2B" in orch.SUBGROUP_STEPS
    assert "4C" in orch.SUBGROUP_STEPS

    # Test specific mappings
    assert orch.SUBGROUP_STEPS["1A"] == [1]  # Business Context
    assert orch.SUBGROUP_STEPS["2B"] == [6, 7]  # Relationship Analysis