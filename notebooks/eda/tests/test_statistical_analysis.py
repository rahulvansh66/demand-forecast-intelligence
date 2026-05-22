import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.statistical_analysis import (
    calculate_distribution_stats,
    compute_variation_metrics,
    analyze_outliers,
    test_normality as check_normality,
    test_independence as check_independence,
    calculate_intermittency_score,
    demand_variability_classification,
)


class TestDistributionStats:
    """Test distribution statistics calculation."""

    def test_calculate_distribution_stats(self):
        """Test calculation of distribution statistics."""
        # Test data with known statistical properties
        data = pd.Series([1, 2, 3, 4, 5, 100])  # Right skewed with outlier
        result = calculate_distribution_stats(data, "test_series")

        assert "mean" in result
        assert "median" in result
        assert "skewness" in result
        assert result["skewness"] > 0  # Right skewed
        assert "interpretation" in result

    def test_calculate_distribution_stats_symmetric(self):
        """Test with symmetric distribution."""
        data = pd.Series([1, 2, 3, 4, 5])
        result = calculate_distribution_stats(data, "symmetric")

        assert abs(result["skewness"]) < 0.5  # Nearly symmetric
        assert result["mean"] == result["median"]

    def test_calculate_distribution_stats_with_nan(self):
        """Test handling of NaN values."""
        data = pd.Series([1, 2, np.nan, 4, 5])
        result = calculate_distribution_stats(data, "with_nan")

        assert result["count"] == 4
        assert not np.isnan(result["mean"])


class TestVariationMetrics:
    """Test variation metrics calculation."""

    def test_compute_variation_metrics(self):
        """Test computation of variation metrics."""
        data = pd.Series([1, 2, 3, 4, 5])
        result = compute_variation_metrics(data, "test_series")

        assert "std_dev" in result
        assert "cv" in result  # Coefficient of variation
        assert "range" in result
        assert "iqr" in result

    def test_compute_variation_metrics_zero_mean(self):
        """Test with zero mean (edge case)."""
        data = pd.Series([0, 0, 0, 0])
        result = compute_variation_metrics(data, "zero_mean")

        assert result["std_dev"] == 0
        assert result["cv"] == 0


class TestOutlierAnalysis:
    """Test outlier detection and analysis."""

    def test_analyze_outliers_iqr_method(self):
        """Test outlier detection using IQR method."""
        data = pd.Series([1, 2, 3, 4, 5, 100])
        result = analyze_outliers(data, method="iqr")

        assert "outliers" in result
        assert "outlier_count" in result
        assert 100 in result["outliers"].values
        assert result["outlier_count"] == 1

    def test_analyze_outliers_zscore_method(self):
        """Test outlier detection using z-score method."""
        data = pd.Series([1, 2, 3, 4, 5, 100])
        result = analyze_outliers(data, method="zscore", threshold=2)

        assert "outliers" in result
        assert result["outlier_count"] > 0


class TestNormalityTests:
    """Test normality testing functions."""

    def test_test_normality_normal_data(self):
        """Test normality test on normally distributed data."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 1000))
        result = check_normality(data, "normal_data")

        assert "shapiro_pvalue" in result
        assert "ks_pvalue" in result
        assert "is_normal" in result

    def test_test_normality_non_normal_data(self):
        """Test normality test on non-normal data."""
        data = pd.Series(np.random.exponential(1, 1000))
        result = check_normality(data, "exponential_data")

        assert "is_normal" in result
        assert result["is_normal"] is False or result["shapiro_pvalue"] < 0.05


class TestIndependenceTests:
    """Test independence testing functions."""

    def test_test_independence_no_correlation(self):
        """Test independence test on independent series."""
        np.random.seed(42)
        series1 = pd.Series(np.random.normal(0, 1, 100))
        series2 = pd.Series(np.random.normal(0, 1, 100))
        result = check_independence(series1, series2, "series1", "series2")

        assert "correlation" in result
        assert "pvalue" in result

    def test_test_independence_correlated(self):
        """Test independence test on correlated series."""
        np.random.seed(42)
        series1 = pd.Series(np.random.normal(0, 1, 100))
        series2 = series1 * 2 + np.random.normal(0, 0.1, 100)
        result = check_independence(series1, series2, "series1", "series2")

        assert result["correlation"] > 0.8


class TestIntermittencyScore:
    """Test intermittency scoring."""

    def test_calculate_intermittency_score_zero_demands(self):
        """Test intermittency score with high zero demands."""
        data = pd.Series([0, 0, 0, 1, 0, 0, 2, 0, 0, 0])
        score = calculate_intermittency_score(data, "test_series")

        assert 0 <= score <= 1
        assert score > 0.7  # High intermittency

    def test_calculate_intermittency_score_consistent(self):
        """Test intermittency score with consistent demands."""
        data = pd.Series([5, 6, 5, 6, 5, 6, 5, 6, 5, 6])
        score = calculate_intermittency_score(data, "test_series")

        assert 0 <= score <= 1
        assert score < 0.3  # Low intermittency

    def test_calculate_intermittency_score_all_zeros(self):
        """Test intermittency score with all zeros."""
        data = pd.Series([0, 0, 0, 0, 0])
        score = calculate_intermittency_score(data, "test_series")

        assert score == 1.0  # Maximum intermittency


class TestDemandVariabilityClassification:
    """Test demand variability classification."""

    def test_demand_variability_classification_smooth(self):
        """Test classification of smooth demand."""
        data = pd.Series([10, 11, 10, 11, 10, 11, 10, 11])
        classification = demand_variability_classification(data, "test_series")

        assert classification in [
            "Smooth",
            "Lumpy",
            "Erratic",
            "Intermittent",
        ]
        assert classification == "Smooth"

    def test_demand_variability_classification_lumpy(self):
        """Test classification of lumpy demand."""
        np.random.seed(42)
        data = pd.Series([0, 0, 5, 0, 0, 8, 0, 0, 3, 0])
        classification = demand_variability_classification(data, "test_series")

        assert classification in [
            "Smooth",
            "Lumpy",
            "Erratic",
            "Intermittent",
        ]

    def test_demand_variability_classification_erratic(self):
        """Test classification of erratic demand."""
        np.random.seed(42)
        data = pd.Series(np.random.exponential(5, 100))
        classification = demand_variability_classification(data, "test_series")

        assert classification in [
            "Smooth",
            "Lumpy",
            "Erratic",
            "Intermittent",
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
