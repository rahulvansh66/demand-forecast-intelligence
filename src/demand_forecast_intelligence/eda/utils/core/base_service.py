"""
Abstract base service for EDA analysis steps.

This module provides the BaseEDAService abstract class that defines the common
interface and patterns for all EDA service implementations across the 8 subgroups.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pathlib import Path

from .context import EDAContext


class BaseEDAService(ABC):
    """
    Abstract base class for all EDA analysis services.

    This class provides the foundation for the service-based architecture used
    across all 8 EDA subgroups. Each service encapsulates a specific analysis
    step with standardized interfaces for execution, validation, and reporting.

    Design Principles:
    - Single Responsibility: Each service handles one analysis step
    - Dependency Injection: Services receive EDAContext for data and state
    - Template Method: Common patterns implemented in base, specifics in subclasses
    - Result Caching: Helper methods for consistent result storage and retrieval
    """

    def __init__(self, ctx: EDAContext) -> None:
        """
        Initialize service with EDA context.

        Args:
            ctx: EDAContext providing data access and state management
        """
        self.ctx = ctx

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the analysis step.

        This method should contain the main analysis logic for the service.
        Results should be returned as a dictionary with standardized keys.

        Returns:
            Dictionary containing analysis results with keys:
            - step_name: Name/identifier of the analysis step
            - findings: List of key findings from the analysis
            - Additional service-specific keys as needed
        """
        pass

    @abstractmethod
    def validate_prerequisites(self) -> List[str]:
        """
        Validate that all prerequisites for the analysis are met.

        This method should check for required data, dependencies, or prior
        analysis results before executing the main analysis.

        Returns:
            List of missing prerequisites. Empty list means all prerequisites met.
        """
        pass

    @abstractmethod
    def generate_summary(self) -> str:
        """
        Generate a human-readable summary of the analysis results.

        This method should create a concise summary suitable for reporting
        or documentation purposes.

        Returns:
            Summary string describing the analysis and key findings
        """
        pass

    def _save_step_result(self, step_key: str, result: Any) -> None:
        """
        Save analysis result using the context's result caching system.

        Args:
            step_key: Unique identifier for this analysis step
            result: Analysis result to cache
        """
        self.ctx.save_result(step_key, result)

    def _get_step_result(self, step_key: str) -> Any:
        """
        Retrieve previously saved analysis result.

        Args:
            step_key: Unique identifier for the analysis step

        Returns:
            Cached result or None if not found
        """
        return self.ctx.get_result(step_key)

    def _create_plot_path(self, step_subdir: str, plot_name: str) -> str:
        """
        Generate standardized plot path for visualization outputs.

        Args:
            step_subdir: Subdirectory name for organizing plots by step
            plot_name: Base name for the plot (without extension)

        Returns:
            Standardized plot path string relative to plots directory
        """
        return f"{step_subdir}/{plot_name}.png"