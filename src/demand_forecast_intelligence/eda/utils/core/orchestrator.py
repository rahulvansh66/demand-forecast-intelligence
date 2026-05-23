"""
EDA Orchestrator for coordinating execution of analysis steps.

This module provides the EDAOrchestrator class that manages the execution
of EDA steps with proper dependency management and granularity control.
"""

from typing import Dict, Any, List
from .context import EDAContext


class EDAOrchestrator:
    """Coordinates execution of EDA analysis steps with dependency management."""

    # Step dependencies mapping as specified in the implementation plan
    STEP_DEPENDENCIES = {
        1: [],
        3: [1],
        5: [1, 3],
        6: [1, 5],
        7: [5, 6],
        8: [1, 3],
        9: [3],
        10: [3, 8],
        11: [6, 7],
        12: [5, 6, 7, 8],
        13: [8],
        15: [12, 13]
    }

    # Phase to steps mapping
    PHASE_STEPS = {
        1: [1, 3],           # Phase 1: Business Context & High-level Trends
        2: [5, 6, 7],        # Phase 2: Feature Analysis & Relationships
        3: [8, 9, 10, 11],   # Phase 3: Advanced Analysis
        4: [12, 13, 15]      # Phase 4: Model Readiness
    }

    # Subgroup to steps mapping
    SUBGROUP_STEPS = {
        "1A": [1],           # Business Context
        "1B": [3],           # High-level Trends
        "2A": [5],           # Feature Profiling
        "2B": [6, 7],        # Relationship Analysis
        "3A": [8, 9],        # Time Series Analysis
        "3B": [10, 11],      # Advanced Pattern Detection
        "4A": [12],          # Data Quality Assessment
        "4B": [13],          # Leakage Detection
        "4C": [15]           # Model Readiness Summary
    }

    def __init__(self, ctx: EDAContext):
        """Initialize orchestrator with EDA context."""
        self.ctx = ctx
        self._completed_steps = set()

    def get_step_dependencies(self, step: int) -> List[int]:
        """Get dependencies for a specific EDA step."""
        return self.STEP_DEPENDENCIES.get(step, [])

    def _validate_step_prerequisites(self, step: int) -> List[int]:
        """Validate step prerequisites and return list of missing dependencies."""
        dependencies = self.get_step_dependencies(step)
        return [dep for dep in dependencies if dep not in self._completed_steps]

    def _get_phase_steps(self, phase: int) -> List[int]:
        """Get list of steps for a specific phase."""
        return self.PHASE_STEPS.get(phase, [])

    def _run_single_step(self, step: int) -> Dict[str, Any]:
        """Run a single EDA step (placeholder implementation)."""
        # This is a placeholder - actual implementation will instantiate step services
        return {"step": "completed"}

    def run_phase(self, phase: int) -> Dict[str, Any]:
        """Run all steps in a specific phase."""
        steps = self._get_phase_steps(phase)
        results = {}

        for step in steps:
            results[f"step_{step}"] = self._run_single_step(step)

        return {f"phase_{phase}": results}

    def run_subgroup(self, subgroup: str) -> Dict[str, Any]:
        """Run all steps in a specific subgroup."""
        if subgroup not in self.SUBGROUP_STEPS:
            raise ValueError(f"Unknown subgroup: {subgroup}")

        steps = self.SUBGROUP_STEPS[subgroup]
        results = {}

        for step in steps:
            results[f"step_{step}"] = self._run_single_step(step)

        return {f"subgroup_{subgroup}": results}

    def run_step(self, step: int) -> Dict[str, Any]:
        """Run a single EDA step with prerequisite validation."""
        missing_deps = self._validate_step_prerequisites(step)
        if missing_deps:
            raise ValueError(f"Step {step} missing prerequisites: {missing_deps}")

        result = self._run_single_step(step)
        self._completed_steps.add(step)
        return {f"step_{step}": result}

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete EDA pipeline in dependency order."""
        results = {}

        # Run all phases in order
        for phase in sorted(self.PHASE_STEPS.keys()):
            phase_result = self.run_phase(phase)
            results.update(phase_result)

        return {"full_pipeline": results}