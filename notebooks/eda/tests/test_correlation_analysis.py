"""
Tests for correlation analysis module.

This module contains tests for categorical sales pattern analysis
and correlation analysis functionality.
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.correlation_analysis import analyze_categorical_sales_patterns


class TestAnalyzeCategoricalSalesPatterns:
    """Comprehensive test suite for analyze_categorical_sales_patterns function."""

    def test_basic_functionality(self):
        """Test basic categorical sales pattern analysis functionality."""
        # Create sample data with known statistical properties
        sample_data = pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
            'sales': [100, 150, 120, 200, 250, 220, 50, 75, 60]
        })

        result = analyze_categorical_sales_patterns(sample_data, 'category', 'sales')

        # Verify result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'categories' in result, "Result should contain 'categories' key"
        assert 'summary_stats' in result, "Result should contain 'summary_stats' key"
        assert 'by_category' in result, "Result should contain 'by_category' key"

        # Verify categories are identified correctly
        assert len(result['categories']) == 3, "Should identify 3 categories"
        assert set(result['categories']) == {'A', 'B', 'C'}, "Categories should match input data"

        # Verify complete result structure
        required_summary_keys = ['mean', 'std', 'count', 'min', 'max', 'median']
        for key in required_summary_keys:
            assert key in result['summary_stats'], f"Summary stats should contain '{key}'"

        # Verify by_category structure
        assert len(result['by_category']) == 3, "Should have stats for each category"
        for category in ['A', 'B', 'C']:
            assert category in result['by_category'], f"Should have stats for category {category}"
            for key in required_summary_keys:
                assert key in result['by_category'][category], f"Category {category} should contain '{key}'"

    def test_statistical_accuracy(self):
        """Test statistical accuracy of calculations."""
        # Create data with known statistical properties
        sample_data = pd.DataFrame({
            'category': ['A'] * 4 + ['B'] * 4,
            'sales': [10, 20, 30, 40, 100, 200, 300, 400]  # A: mean=25, B: mean=250
        })

        result = analyze_categorical_sales_patterns(sample_data, 'category', 'sales')

        # Test overall statistics
        expected_overall_mean = (10 + 20 + 30 + 40 + 100 + 200 + 300 + 400) / 8
        assert abs(result['summary_stats']['mean'] - expected_overall_mean) < 1e-10

        # Test category-specific statistics
        assert abs(result['by_category']['A']['mean'] - 25.0) < 1e-10
        assert abs(result['by_category']['B']['mean'] - 250.0) < 1e-10
        assert result['by_category']['A']['count'] == 4
        assert result['by_category']['B']['count'] == 4

        # Test min/max values
        assert result['by_category']['A']['min'] == 10
        assert result['by_category']['A']['max'] == 40
        assert result['by_category']['B']['min'] == 100
        assert result['by_category']['B']['max'] == 400

    def test_error_conditions(self):
        """Test error conditions and input validation."""
        sample_data = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'sales': [100, 200, 300]
        })

        # Test wrong data type
        with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
            analyze_categorical_sales_patterns("not_a_dataframe", 'category', 'sales')

        # Test missing category column
        with pytest.raises(ValueError, match="Column 'missing_category' not found in dataframe"):
            analyze_categorical_sales_patterns(sample_data, 'missing_category', 'sales')

        # Test missing sales column
        with pytest.raises(ValueError, match="Column 'missing_sales' not found in dataframe"):
            analyze_categorical_sales_patterns(sample_data, 'category', 'missing_sales')

    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['category', 'sales'])

        with pytest.raises((ValueError, ZeroDivisionError)):
            analyze_categorical_sales_patterns(empty_df, 'category', 'sales')

    def test_single_category(self):
        """Test with single category data."""
        single_category_data = pd.DataFrame({
            'category': ['A', 'A', 'A', 'A'],
            'sales': [10, 20, 30, 40]
        })

        result = analyze_categorical_sales_patterns(single_category_data, 'category', 'sales')

        assert len(result['categories']) == 1
        assert result['categories'][0] == 'A'
        assert result['by_category']['A']['mean'] == 25.0
        assert result['by_category']['A']['count'] == 4

    def test_single_row_per_category(self):
        """Test with single row per category."""
        single_row_data = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'sales': [100, 200, 300]
        })

        result = analyze_categorical_sales_patterns(single_row_data, 'category', 'sales')

        assert len(result['categories']) == 3
        assert result['by_category']['A']['count'] == 1
        assert result['by_category']['A']['mean'] == 100.0
        assert result['by_category']['A']['std'] == 0.0  # std of single value should be 0

    def test_nan_values_in_sales(self):
        """Test handling of NaN values in sales data."""
        nan_data = pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'sales': [100, np.nan, 120, 200, 250, np.nan]
        })

        result = analyze_categorical_sales_patterns(nan_data, 'category', 'sales')

        # The function should handle NaN values appropriately
        # Specific behavior depends on implementation - should either skip or handle gracefully
        assert isinstance(result, dict)
        assert 'categories' in result
        assert 'summary_stats' in result
        assert 'by_category' in result

    def test_nan_values_in_category(self):
        """Test handling of NaN values in category data."""
        nan_category_data = pd.DataFrame({
            'category': ['A', np.nan, 'B', 'C', np.nan],
            'sales': [100, 150, 200, 250, 300]
        })

        result = analyze_categorical_sales_patterns(nan_category_data, 'category', 'sales')

        # Should handle NaN categories appropriately
        assert isinstance(result, dict)
        assert len(result['categories']) >= 2  # At least A, B, C (NaN handling may vary)

    def test_zero_and_negative_sales(self):
        """Test with zero and negative sales values."""
        mixed_sales_data = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'sales': [0, -10, 100, 0, -50, 200]
        })

        result = analyze_categorical_sales_patterns(mixed_sales_data, 'category', 'sales')

        assert result['by_category']['A']['mean'] == -5.0
        assert result['by_category']['B']['mean'] == 50.0
        assert result['by_category']['C']['mean'] == 75.0

    def test_large_dataset_performance(self):
        """Test with larger dataset to ensure reasonable performance."""
        # Create larger dataset
        large_data = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=1000),
            'sales': np.random.randint(1, 1000, size=1000)
        })

        result = analyze_categorical_sales_patterns(large_data, 'category', 'sales')

        assert isinstance(result, dict)
        assert len(result['categories']) <= 5  # Should not exceed number of unique categories
        assert result['summary_stats']['count'] == 1000


# Legacy test function for backwards compatibility
def test_analyze_categorical_sales_patterns():
    """
    Test basic categorical sales pattern analysis functionality.

    This is kept for backwards compatibility with the original test,
    updated to accommodate the enhanced return structure.
    """
    # Create sample data
    sample_data = pd.DataFrame({
        'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'sales': [100, 150, 120, 200, 250, 220, 50, 75, 60]
    })

    # Call the function
    result = analyze_categorical_sales_patterns(sample_data, 'category', 'sales')

    # Verify result is a dictionary
    assert isinstance(result, dict), "Result should be a dictionary"

    # Verify result contains expected keys (including new data_quality key)
    assert 'categories' in result, "Result should contain 'categories' key"
    assert 'summary_stats' in result, "Result should contain 'summary_stats' key"
    assert 'by_category' in result, "Result should contain 'by_category' key"
    assert 'data_quality' in result, "Result should contain 'data_quality' key"

    # Verify categories are identified
    assert len(result['categories']) == 3, "Should identify 3 categories"

    # Verify summary stats have required fields (including new fields)
    required_fields = ['mean', 'std', 'count', 'min', 'max', 'median']
    for field in required_fields:
        assert field in result['summary_stats'], f"Summary stats should contain '{field}'"


class TestComputeTemporalSalesCorrelations:
    """Test suite for compute_temporal_sales_correlations function."""

    def test_basic_functionality(self):
        """Test basic temporal correlation analysis."""
        from utils.correlation_analysis import compute_temporal_sales_correlations

        # Create sample data
        sales_data = pd.DataFrame({
            'cat_id': ['FOODS'] * 3,
            'd_1': [10, 12, 8],
            'd_2': [15, 18, 12],
            'd_3': [5, 8, 4]
        })

        calendar_data = pd.DataFrame({
            'weekday': ['Saturday', 'Sunday', 'Monday'],
            'date': pd.to_datetime(['2011-01-29', '2011-01-30', '2011-01-31'])
        })

        result = compute_temporal_sales_correlations(sales_data, calendar_data)

        # Verify result structure
        assert isinstance(result, dict)
        assert 'temporal_correlations' in result
        assert 'FOODS' in result['temporal_correlations']
        assert 'weekday_correlation' in result['temporal_correlations']['FOODS']
        assert 'month_correlation' in result['temporal_correlations']['FOODS']

    def test_multiple_categories(self):
        """Test with multiple categories."""
        from utils.correlation_analysis import compute_temporal_sales_correlations

        sales_data = pd.DataFrame({
            'cat_id': ['FOODS', 'HOUSEHOLD', 'HOBBIES'] * 2,
            'd_1': [10, 15, 20, 12, 18, 22],
            'd_2': [15, 18, 25, 18, 20, 28]
        })

        calendar_data = pd.DataFrame({
            'weekday': ['Saturday', 'Sunday'],
            'date': pd.to_datetime(['2011-01-29', '2011-01-30'])
        })

        result = compute_temporal_sales_correlations(sales_data, calendar_data)

        assert 'FOODS' in result['temporal_correlations']
        assert 'HOUSEHOLD' in result['temporal_correlations']
        assert 'HOBBIES' in result['temporal_correlations']

    def test_edge_case_single_row(self):
        """Test with single row of data."""
        from utils.correlation_analysis import compute_temporal_sales_correlations

        sales_data = pd.DataFrame({
            'cat_id': ['FOODS'],
            'd_1': [10],
            'd_2': [15]
        })

        calendar_data = pd.DataFrame({
            'weekday': ['Saturday', 'Sunday']
        })

        result = compute_temporal_sales_correlations(sales_data, calendar_data)

        assert isinstance(result, dict)
        assert 'temporal_correlations' in result


class TestComputeSnapBenefitImpact:
    """Test suite for compute_snap_benefit_impact function."""

    def test_basic_functionality(self):
        """Test basic SNAP benefit impact analysis."""
        from utils.correlation_analysis import compute_snap_benefit_impact

        sales_data = pd.DataFrame({
            'cat_id': ['FOODS'] * 2,
            'd_1': [10, 12],
            'd_2': [15, 8],
            'd_3': [20, 18]
        })

        calendar_data = pd.DataFrame({
            'snap_CA': [0, 1, 0],
            'snap_TX': [1, 0, 1]
        })

        result = compute_snap_benefit_impact(sales_data, calendar_data)

        assert isinstance(result, dict)
        assert 'snap_impact_by_state' in result
        assert 'foods_category_analysis' in result

    def test_no_foods_category(self):
        """Test when no FOODS category exists."""
        from utils.correlation_analysis import compute_snap_benefit_impact

        sales_data = pd.DataFrame({
            'cat_id': ['HOUSEHOLD', 'HOBBIES'],
            'd_1': [10, 20],
            'd_2': [15, 25]
        })

        calendar_data = pd.DataFrame({
            'snap_CA': [0, 1]
        })

        result = compute_snap_benefit_impact(sales_data, calendar_data)

        # Should return error or empty analysis
        assert isinstance(result, dict)
        assert 'error' in result or 'snap_impact_by_state' in result

    def test_multiple_states(self):
        """Test with multiple state SNAP columns."""
        from utils.correlation_analysis import compute_snap_benefit_impact

        sales_data = pd.DataFrame({
            'cat_id': ['FOODS'] * 3,
            'd_1': [100, 120, 90],
            'd_2': [150, 180, 120],
            'd_3': [200, 220, 180]
        })

        calendar_data = pd.DataFrame({
            'snap_CA': [0, 1, 0],
            'snap_TX': [1, 0, 1],
            'snap_WI': [0, 1, 1]
        })

        result = compute_snap_benefit_impact(sales_data, calendar_data)

        assert isinstance(result, dict)
        if 'snap_impact_by_state' in result and result['snap_impact_by_state']:
            assert len(result['snap_impact_by_state']) >= 1
