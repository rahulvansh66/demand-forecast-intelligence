"""
Tests for visualization module.

This module contains comprehensive tests for visualization functions
used in EDA framework steps 6-10.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This import will fail initially (expected for TDD)
try:
    from utils.visualization import plot_categorical_sales_distributions
except ImportError:
    plot_categorical_sales_distributions = None


class TestPlotCategoricalSalesDistributions:
    """Test suite for plot_categorical_sales_distributions function."""

    def test_basic_functionality(self):
        """Test basic categorical sales distribution plotting."""
        pytest.skip("Waiting for implementation") if plot_categorical_sales_distributions is None else None

        # Sample data
        sales_data = pd.DataFrame({
            'cat_id': ['FOODS', 'HOUSEHOLD', 'HOBBIES'] * 10,
            'daily_sales': np.random.randint(0, 20, 30)
        })

        save_path = 'notebooks/eda/plots/step6_feature_target/test_category_distributions.png'

        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Should create plot without errors
        result = plot_categorical_sales_distributions(sales_data, save_path)

        # Verify result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'plot_path' in result, "Result should contain 'plot_path'"
        assert os.path.exists(save_path), "Plot file should exist"

        # Verify result contains statistics
        assert 'category_statistics' in result, "Result should contain 'category_statistics'"
        assert 'total_samples' in result, "Result should contain 'total_samples'"

        # Cleanup
        if os.path.exists(save_path):
            os.remove(save_path)

    def test_custom_column_names(self):
        """Test with custom column names."""
        pytest.skip("Waiting for implementation") if plot_categorical_sales_distributions is None else None

        # Sample data with different column names
        sales_data = pd.DataFrame({
            'product_cat': ['A', 'B', 'C'] * 5,
            'sales_volume': np.random.randint(10, 100, 15)
        })

        save_path = 'notebooks/eda/plots/step6_feature_target/test_custom_cols.png'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        result = plot_categorical_sales_distributions(
            sales_data,
            save_path,
            category_col='product_cat',
            sales_col='sales_volume'
        )

        assert os.path.exists(save_path)
        # Should have categories A, B, C
        stats = result.get('category_statistics', {})
        assert len(stats) == 3
        assert all(k in ['A', 'B', 'C'] for k in stats.keys())

        if os.path.exists(save_path):
            os.remove(save_path)

    def test_missing_columns_error(self):
        """Test error handling for missing columns."""
        pytest.skip("Waiting for implementation") if plot_categorical_sales_distributions is None else None

        sales_data = pd.DataFrame({
            'cat_id': ['A', 'B', 'C'],
            'sales': [10, 20, 30]
        })

        save_path = 'notebooks/eda/plots/step6_feature_target/test_error.png'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Should raise ValueError for missing column
        with pytest.raises((ValueError, KeyError)):
            plot_categorical_sales_distributions(
                sales_data,
                save_path,
                category_col='missing_col',
                sales_col='sales'
            )

    def test_empty_dataframe_error(self):
        """Test error handling for empty dataframe."""
        pytest.skip("Waiting for implementation") if plot_categorical_sales_distributions is None else None

        sales_data = pd.DataFrame(columns=['cat_id', 'daily_sales'])
        save_path = 'notebooks/eda/plots/step6_feature_target/test_empty.png'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with pytest.raises((ValueError, IndexError)):
            plot_categorical_sales_distributions(sales_data, save_path)

    def test_single_category(self):
        """Test with single category data."""
        pytest.skip("Waiting for implementation") if plot_categorical_sales_distributions is None else None

        sales_data = pd.DataFrame({
            'cat_id': ['FOODS'] * 10,
            'daily_sales': np.random.randint(5, 50, 10)
        })

        save_path = 'notebooks/eda/plots/step6_feature_target/test_single_cat.png'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        result = plot_categorical_sales_distributions(sales_data, save_path)

        assert os.path.exists(save_path)
        assert len(result['category_statistics']) == 1

        if os.path.exists(save_path):
            os.remove(save_path)

    def test_many_categories(self):
        """Test with many categories."""
        pytest.skip("Waiting for implementation") if plot_categorical_sales_distributions is None else None

        categories = [f'CAT_{i}' for i in range(20)]
        sales_data = pd.DataFrame({
            'cat_id': categories * 5,  # Each category repeated 5 times
            'daily_sales': np.random.randint(1, 100, 100)
        })

        save_path = 'notebooks/eda/plots/step6_feature_target/test_many_cats.png'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        result = plot_categorical_sales_distributions(sales_data, save_path)

        assert os.path.exists(save_path)
        assert len(result['category_statistics']) == 20

        if os.path.exists(save_path):
            os.remove(save_path)

    def test_zero_and_negative_sales(self):
        """Test with zero and negative sales values."""
        pytest.skip("Waiting for implementation") if plot_categorical_sales_distributions is None else None

        sales_data = pd.DataFrame({
            'cat_id': ['A', 'B', 'C'] * 5,
            'daily_sales': [0, -5, 10, 0, 20, -3, 15, 0, 5, -1, 25, 0, 8, -2, 12]
        })

        save_path = 'notebooks/eda/plots/step6_feature_target/test_zero_neg.png'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Should handle gracefully
        result = plot_categorical_sales_distributions(sales_data, save_path)

        assert os.path.exists(save_path)
        assert 'category_statistics' in result

        if os.path.exists(save_path):
            os.remove(save_path)

    def test_dpi_and_resolution(self):
        """Test that plot is saved with correct DPI."""
        pytest.skip("Waiting for implementation") if plot_categorical_sales_distributions is None else None

        sales_data = pd.DataFrame({
            'cat_id': ['A', 'B', 'C'] * 3,
            'daily_sales': np.random.randint(10, 50, 9)
        })

        save_path = 'notebooks/eda/plots/step6_feature_target/test_dpi.png'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        result = plot_categorical_sales_distributions(sales_data, save_path)

        # Verify plot was created and has reasonable file size (300 DPI should be ~50KB+)
        assert os.path.exists(save_path)
        file_size = os.path.getsize(save_path)
        assert file_size > 1000, "Plot should be properly saved with publication quality"

        if os.path.exists(save_path):
            os.remove(save_path)

    def test_statistical_calculations(self):
        """Test that statistical calculations are correct."""
        pytest.skip("Waiting for implementation") if plot_categorical_sales_distributions is None else None

        # Create data with known statistics
        sales_data = pd.DataFrame({
            'cat_id': ['A'] * 5 + ['B'] * 5,
            'daily_sales': [10, 20, 30, 40, 50, 5, 10, 15, 20, 25]
        })

        save_path = 'notebooks/eda/plots/step6_feature_target/test_stats.png'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        result = plot_categorical_sales_distributions(sales_data, save_path)

        # Verify calculations
        assert result['category_statistics']['A']['mean'] == 30.0  # Mean of [10,20,30,40,50]
        assert result['category_statistics']['B']['mean'] == 15.0  # Mean of [5,10,15,20,25]
        assert result['category_statistics']['A']['count'] == 5
        assert result['category_statistics']['B']['count'] == 5

        if os.path.exists(save_path):
            os.remove(save_path)

    def test_nan_values_handling(self):
        """Test handling of NaN values in sales data."""
        pytest.skip("Waiting for implementation") if plot_categorical_sales_distributions is None else None

        sales_data = pd.DataFrame({
            'cat_id': ['A', 'B', 'A', 'B', 'A', 'B'],
            'daily_sales': [10.0, np.nan, 20.0, 30.0, np.nan, 40.0]
        })

        save_path = 'notebooks/eda/plots/step6_feature_target/test_nan.png'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Should handle NaN gracefully
        result = plot_categorical_sales_distributions(sales_data, save_path)

        assert os.path.exists(save_path)
        # NaN values should be excluded from statistics
        assert result['category_statistics']['A']['count'] == 2
        assert result['category_statistics']['B']['count'] == 2

        if os.path.exists(save_path):
            os.remove(save_path)

    def test_plot_file_types(self):
        """Test saving plots in different formats."""
        pytest.skip("Waiting for implementation") if plot_categorical_sales_distributions is None else None

        sales_data = pd.DataFrame({
            'cat_id': ['A', 'B', 'C'] * 3,
            'daily_sales': np.random.randint(10, 50, 9)
        })

        for ext in ['.png', '.jpg']:
            save_path = f'notebooks/eda/plots/step6_feature_target/test_format{ext}'
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            result = plot_categorical_sales_distributions(sales_data, save_path)

            assert os.path.exists(save_path)

            if os.path.exists(save_path):
                os.remove(save_path)


# Legacy test function for backwards compatibility
def test_plot_categorical_sales_distributions_basic():
    """
    Legacy test for basic functionality (backwards compatibility).
    """
    pytest.skip("Waiting for implementation") if plot_categorical_sales_distributions is None else None

    # Create sample data
    sample_data = pd.DataFrame({
        'cat_id': ['FOODS', 'HOUSEHOLD', 'HOBBIES'] * 5,
        'daily_sales': np.random.randint(0, 100, 15)
    })

    save_path = 'notebooks/eda/plots/step6_feature_target/legacy_test.png'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Call the function
    result = plot_categorical_sales_distributions(sample_data, save_path)

    # Verify result is a dictionary
    assert isinstance(result, dict), "Result should be a dictionary"

    # Verify result contains expected keys
    assert 'plot_path' in result, "Result should contain 'plot_path' key"
    assert 'category_statistics' in result, "Result should contain 'category_statistics' key"

    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)
