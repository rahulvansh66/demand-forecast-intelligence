"""
M5-specific configuration for EDA analysis pipeline.

This module provides comprehensive configuration for the M5 retail forecasting dataset,
including data paths, analysis parameters, thresholds, and dataset specifications.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from utils.core.context import EDAContext


# M5 Dataset Configuration
M5_CONFIG: Dict[str, Any] = {
    # Data file paths for M5 dataset
    "data_paths": {
        "sales_validation": "sales_train_validation.csv",
        "sales_evaluation": "sales_train_evaluation.csv",
        "calendar": "calendar.csv",
        "pricing": "sell_prices.csv"
    },

    # Output and visualization paths
    "output_paths": {
        "base_output": "data/eda/outputs",
        "plots": "eda/plots",
        "step_subdirs": {
            "step1_overview": "step1_data_overview",
            "step2_profiling": "step2_feature_profiling",
            "step3_quality": "step3_data_quality",
            "step5_target": "step5_target_analysis",
            "step6_feature_target": "step6_feature_target",
            "step7_relationships": "step7_feature_relationships",
            "step11_segments": "step11_segment_behavior",
            "step13_drift": "step13_distribution_drift",
            "step14_leakage": "step14_leakage_audit"
        }
    },

    # Analysis parameters and thresholds
    "analysis_params": {
        # Data quality thresholds
        "missing_threshold": 0.05,  # 5% missing data threshold
        "outlier_threshold": 0.95,  # 95th percentile for outlier detection
        "correlation_threshold": 0.8,  # High correlation threshold
        "variance_threshold": 0.01,  # Minimum variance for feature relevance

        # Statistical analysis parameters
        "confidence_level": 0.95,
        "significance_level": 0.05,
        "bootstrap_samples": 1000,

        # Time series specific
        "time_split_date": "2016-04-24",  # d_1914 - validation start
        "seasonality_periods": [7, 30, 365],  # Weekly, monthly, yearly patterns
        "min_obs_per_series": 100,  # Minimum observations for analysis

        # Segmentation parameters
        "max_clusters": 10,
        "min_cluster_size": 100,
        "segment_features": ["intermittency", "variability", "trend", "seasonality"],

        # Distribution analysis
        "hist_bins": 50,
        "kde_bandwidth": "scott",
        "distribution_tests": ["shapiro", "anderson", "kstest"],

        # Visualization parameters
        "figure_size": (12, 8),
        "color_palette": "viridis",
        "max_categories_plot": 20,
        "correlation_annot": True
    },

    # M5 dataset specific parameters
    "m5_specifics": {
        "item_store_combinations": 30490,
        "total_items": 3049,
        "total_stores": 10,
        "training_days": "d_1 to d_1913",
        "validation_days": "d_1914 to d_1941",
        "evaluation_days": "d_1942 to d_1969",
        "total_days": 1969,
        "forecast_horizon": 28,  # Days to forecast

        # Hierarchical structure
        "states": ["CA", "TX", "WI"],
        "categories": ["FOODS", "HOBBIES", "HOUSEHOLD"],
        "departments": {
            "FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"],
            "HOBBIES": ["HOBBIES_1", "HOBBIES_2"],
            "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"]
        },

        # Store information
        "stores_per_state": {
            "CA": 4,  # CA_1, CA_2, CA_3, CA_4
            "TX": 3,  # TX_1, TX_2, TX_3
            "WI": 3   # WI_1, WI_2, WI_3
        },

        # Data characteristics
        "sparse_series_threshold": 0.5,  # Series with >50% zeros considered sparse
        "intermittent_threshold": 0.3,   # Series with >30% zeros considered intermittent
        "high_volume_threshold": 100,    # Sales > 100 units considered high volume

        # Calendar features
        "event_types": ["Cultural", "National", "Religious", "Sporting"],
        "snap_states": ["CA", "TX", "WI"],  # SNAP benefit states

        # Price analysis
        "price_change_threshold": 0.1,  # 10% price change significance
        "promotion_detection": True,
        "price_elasticity_analysis": True
    }
}


def get_m5_context(
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    plots_dir: Optional[str] = None
) -> EDAContext:
    """
    Create EDAContext configured for M5 dataset analysis.

    Args:
        data_dir: Override default data directory
        output_dir: Override default output directory
        plots_dir: Override default plots directory

    Returns:
        Configured EDAContext with M5 settings and config attached
    """
    # Use provided paths or defaults from config
    data_path = Path(data_dir) if data_dir else Path("data/raw")
    output_path = Path(output_dir) if output_dir else Path(M5_CONFIG["output_paths"]["base_output"])
    plots_path = Path(plots_dir) if plots_dir else Path(M5_CONFIG["output_paths"]["plots"])

    # Create context
    context = EDAContext(
        data_dir=data_path,
        output_dir=output_path,
        plots_dir=plots_path
    )

    # Attach M5 configuration
    context.config = M5_CONFIG

    return context


def get_step_config(step_name: str) -> Dict[str, Any]:
    """
    Get configuration specific to an EDA step.

    Args:
        step_name: Name of the EDA step (e.g., 'step1_overview')

    Returns:
        Step-specific configuration dictionary

    Raises:
        ValueError: If step_name is not recognized
    """
    step_configs = {
        "step1_overview": {
            "plots_subdir": M5_CONFIG["output_paths"]["step_subdirs"]["step1_overview"],
            "sample_size": 1000,
            "summary_stats": ["count", "mean", "std", "min", "max", "skew", "kurtosis"],
            "data_types_analysis": True,
            "memory_usage": True
        },

        "step2_profiling": {
            "plots_subdir": M5_CONFIG["output_paths"]["step_subdirs"]["step2_profiling"],
            "profile_numerical": True,
            "profile_categorical": True,
            "correlation_analysis": True,
            "outlier_detection": True,
            "distribution_analysis": True
        },

        "step3_quality": {
            "plots_subdir": M5_CONFIG["output_paths"]["step_subdirs"]["step3_quality"],
            "missing_patterns": True,
            "duplicates_check": True,
            "consistency_checks": True,
            "data_validation": True
        },

        "step5_target": {
            "plots_subdir": M5_CONFIG["output_paths"]["step_subdirs"]["step5_target"],
            "target_column": "sales",
            "temporal_analysis": True,
            "seasonal_decomposition": True,
            "trend_analysis": True,
            "stationarity_tests": True
        },

        "step6_feature_target": {
            "plots_subdir": M5_CONFIG["output_paths"]["step_subdirs"]["step6_feature_target"],
            "feature_importance": True,
            "target_correlation": True,
            "feature_selection": True,
            "interaction_analysis": True
        },

        "step7_relationships": {
            "plots_subdir": M5_CONFIG["output_paths"]["step_subdirs"]["step7_relationships"],
            "correlation_matrix": True,
            "partial_correlations": True,
            "mutual_information": True,
            "feature_clustering": True
        },

        "step11_segments": {
            "plots_subdir": M5_CONFIG["output_paths"]["step_subdirs"]["step11_segments"],
            "demand_patterns": True,
            "intermittency_analysis": True,
            "behavioral_clustering": True,
            "segment_profiling": True
        },

        "step13_drift": {
            "plots_subdir": M5_CONFIG["output_paths"]["step_subdirs"]["step13_drift"],
            "temporal_drift": True,
            "distribution_comparison": True,
            "statistical_tests": True,
            "drift_detection": True
        },

        "step14_leakage": {
            "plots_subdir": M5_CONFIG["output_paths"]["step_subdirs"]["step14_leakage"],
            "temporal_leakage": True,
            "target_leakage": True,
            "feature_leakage": True,
            "validation_strategy": True
        }
    }

    if step_name not in step_configs:
        raise ValueError(f"Unknown step: {step_name}. Available steps: {list(step_configs.keys())}")

    # Merge with global analysis parameters
    step_config = step_configs[step_name].copy()
    step_config.update(M5_CONFIG["analysis_params"])

    return step_config


def validate_m5_data_availability(data_dir: Path) -> Dict[str, bool]:
    """
    Validate availability of required M5 data files.

    Args:
        data_dir: Directory containing M5 data files

    Returns:
        Dictionary mapping file types to availability status
    """
    data_paths = M5_CONFIG["data_paths"]
    availability = {}

    for file_type, filename in data_paths.items():
        file_path = data_dir / filename
        availability[file_type] = file_path.exists()

    return availability


def get_m5_hierarchy_config() -> Dict[str, Any]:
    """
    Get M5 hierarchical structure configuration.

    Returns:
        Dictionary containing M5 hierarchy information
    """
    m5_specifics = M5_CONFIG["m5_specifics"]

    return {
        "levels": ["total", "state", "store", "category", "department", "item"],
        "aggregation_keys": {
            "state": ["CA", "TX", "WI"],
            "store": [f"{state}_{i}" for state in ["CA", "TX", "WI"]
                     for i in range(1, m5_specifics["stores_per_state"][state] + 1)],
            "category": ["FOODS", "HOBBIES", "HOUSEHOLD"],
            "department": [dept for depts in m5_specifics["departments"].values() for dept in depts]
        },
        "hierarchy_mappings": {
            "store_to_state": {store: store.split("_")[0]
                              for store in [f"{state}_{i}" for state in ["CA", "TX", "WI"]
                                           for i in range(1, m5_specifics["stores_per_state"][state] + 1)]},
            "department_to_category": {
                dept: cat for cat, depts in m5_specifics["departments"].items() for dept in depts
            }
        }
    }


def get_analysis_thresholds() -> Dict[str, float]:
    """
    Get standardized analysis thresholds for M5 dataset.

    Returns:
        Dictionary of threshold values for various analyses
    """
    params = M5_CONFIG["analysis_params"]
    specifics = M5_CONFIG["m5_specifics"]

    return {
        "missing_data": params["missing_threshold"],
        "outlier_detection": params["outlier_threshold"],
        "high_correlation": params["correlation_threshold"],
        "low_variance": params["variance_threshold"],
        "sparse_series": specifics["sparse_series_threshold"],
        "intermittent_series": specifics["intermittent_threshold"],
        "high_volume": specifics["high_volume_threshold"],
        "price_change": specifics["price_change_threshold"],
        "significance": params["significance_level"]
    }