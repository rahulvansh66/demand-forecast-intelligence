"""
Tests for segment analysis module.

Tests cover product hierarchy behavioral analysis for demand segmentation:
- Category behavior patterns (FOODS/HOUSEHOLD/HOBBIES)
- Department segment patterns
- Segment performance metrics
- Segment seasonality patterns
- Segment lifecycle stages
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.segment_analysis import (
    analyze_category_behavior_patterns,
    analyze_department_segment_patterns,
    compute_segment_performance_metrics,
    analyze_segment_seasonality_patterns,
    detect_segment_lifecycle_stages,
)


class TestCategoryBehaviorPatterns:
    """Test category behavior pattern analysis."""

    def test_analyze_category_behavior_patterns_basic(self):
        """Test basic category behavior analysis."""
        # Create test data with known patterns
        test_data = pd.DataFrame({
            'cat_id': ['FOODS'] * 100 + ['HOUSEHOLD'] * 100 + ['HOBBIES'] * 100,
            'daily_sales': [5, 4, 6] * 33 + [5] + [2, 1, 3] * 33 + [2] + [8, 0, 12] * 33 + [8],
            'date': pd.date_range('2023-01-01', periods=300)
        })

        result = analyze_category_behavior_patterns(test_data, category_col='cat_id')

        # Verify result structure
        assert 'behavioral_metrics' in result
        assert 'statistical_tests' in result
        assert 'business_interpretation' in result

        # Verify categories are analyzed
        assert 'FOODS' in result['behavioral_metrics']
        assert 'HOUSEHOLD' in result['behavioral_metrics']
        assert 'HOBBIES' in result['behavioral_metrics']

    def test_category_variance_comparison(self):
        """Test that category variance comparison works correctly."""
        # FOODS has low variance, HOBBIES has high variance
        test_data = pd.DataFrame({
            'cat_id': ['FOODS'] * 100 + ['HOBBIES'] * 100,
            'daily_sales': [5, 4, 6] * 33 + [5] + [8, 0, 12] * 33 + [8],
        })

        result = analyze_category_behavior_patterns(test_data, category_col='cat_id')

        # FOODS should have lower coefficient of variation than HOBBIES
        foods_cv = result['behavioral_metrics']['FOODS']['coefficient_of_variation']
        hobbies_cv = result['behavioral_metrics']['HOBBIES']['coefficient_of_variation']

        assert foods_cv < hobbies_cv

    def test_category_behavior_with_empty_category(self):
        """Test handling of edge cases with minimal data."""
        test_data = pd.DataFrame({
            'cat_id': ['FOODS'] * 10 + ['HOUSEHOLD'] * 10,
            'daily_sales': np.random.randint(1, 10, 20),
        })

        result = analyze_category_behavior_patterns(test_data, category_col='cat_id')

        assert 'FOODS' in result['behavioral_metrics']
        assert 'HOUSEHOLD' in result['behavioral_metrics']

    def test_category_behavior_statistical_test_presence(self):
        """Test that statistical tests are included."""
        test_data = pd.DataFrame({
            'cat_id': ['FOODS'] * 50 + ['HOUSEHOLD'] * 50,
            'daily_sales': np.random.randint(1, 10, 100),
        })

        result = analyze_category_behavior_patterns(test_data, category_col='cat_id')

        # Should include Kruskal-Wallis test or similar
        assert 'test_name' in result['statistical_tests']
        assert 'p_value' in result['statistical_tests']
        assert 'interpretation' in result['statistical_tests']


class TestDepartmentSegmentPatterns:
    """Test department segment pattern analysis."""

    def test_analyze_department_segment_patterns_basic(self):
        """Test basic department segment pattern analysis."""
        test_data = pd.DataFrame({
            'dept_id': ['DEPT_1'] * 50 + ['DEPT_2'] * 50 + ['DEPT_3'] * 50,
            'daily_sales': np.random.randint(1, 20, 150),
            'cat_id': ['FOODS'] * 50 + ['HOUSEHOLD'] * 50 + ['HOBBIES'] * 50,
        })

        result = analyze_department_segment_patterns(test_data, department_col='dept_id')

        # Verify result structure
        assert 'department_metrics' in result
        assert 'cross_department_comparison' in result
        assert 'segment_recommendations' in result

    def test_department_metrics_completeness(self):
        """Test that department metrics are complete."""
        test_data = pd.DataFrame({
            'dept_id': ['DEPT_1'] * 30 + ['DEPT_2'] * 30,
            'daily_sales': np.random.randint(1, 10, 60),
        })

        result = analyze_department_segment_patterns(test_data, department_col='dept_id')

        for dept in result['department_metrics']:
            dept_metrics = result['department_metrics'][dept]
            assert 'mean_sales' in dept_metrics
            assert 'std_dev' in dept_metrics
            assert 'coefficient_of_variation' in dept_metrics


class TestSegmentPerformanceMetrics:
    """Test segment performance metrics computation."""

    def test_compute_segment_performance_metrics_basic(self):
        """Test basic segment performance metrics."""
        test_data = pd.DataFrame({
            'segment_id': ['SEG_1'] * 50 + ['SEG_2'] * 50,
            'daily_sales': np.random.randint(1, 20, 100),
            'date': pd.date_range('2023-01-01', periods=100),
        })

        result = compute_segment_performance_metrics(test_data, segment_col='segment_id')

        assert 'performance_metrics' in result
        assert 'segment_ranking' in result
        assert 'business_insights' in result

    def test_segment_performance_includes_roi_metrics(self):
        """Test that ROI-related metrics are included."""
        test_data = pd.DataFrame({
            'segment_id': ['SEG_1'] * 30 + ['SEG_2'] * 30,
            'daily_sales': np.random.randint(5, 50, 60),
            'date': pd.date_range('2023-01-01', periods=60),
        })

        result = compute_segment_performance_metrics(test_data, segment_col='segment_id')

        for seg in result['performance_metrics']:
            metrics = result['performance_metrics'][seg]
            assert 'total_sales_volume' in metrics
            assert 'average_daily_sales' in metrics

    def test_segment_ranking_structure(self):
        """Test that segment ranking is properly structured."""
        test_data = pd.DataFrame({
            'segment_id': ['SEG_A'] * 25 + ['SEG_B'] * 25 + ['SEG_C'] * 25,
            'daily_sales': [10] * 25 + [5] * 25 + [15] * 25,
        })

        result = compute_segment_performance_metrics(test_data, segment_col='segment_id')

        assert len(result['segment_ranking']) > 0
        assert 'segment' in result['segment_ranking'][0]
        assert 'rank' in result['segment_ranking'][0]


class TestSegmentSeasonalityPatterns:
    """Test segment seasonality pattern analysis."""

    def test_analyze_segment_seasonality_patterns_basic(self):
        """Test basic seasonality pattern analysis."""
        # Create data with clear weekly seasonality
        dates = pd.date_range('2023-01-01', periods=365)
        daily_sales = 10 + 5 * np.sin(2 * np.pi * np.arange(365) / 7)

        test_data = pd.DataFrame({
            'segment_id': ['SEGMENT_1'] * 365,
            'daily_sales': daily_sales,
            'date': dates,
        })

        result = analyze_segment_seasonality_patterns(test_data, segment_col='segment_id', date_col='date')

        assert 'seasonality_metrics' in result
        assert 'detected_patterns' in result
        assert 'business_implications' in result

    def test_seasonality_metrics_structure(self):
        """Test that seasonality metrics are properly structured."""
        dates = pd.date_range('2023-01-01', periods=100)
        test_data = pd.DataFrame({
            'segment_id': ['SEG_1'] * 50 + ['SEG_2'] * 50,
            'daily_sales': np.random.randint(1, 20, 100),
            'date': list(dates)[:50] + list(dates)[:50],
        })

        result = analyze_segment_seasonality_patterns(test_data, segment_col='segment_id', date_col='date')

        for seg in result['seasonality_metrics']:
            metrics = result['seasonality_metrics'][seg]
            assert 'seasonal_strength' in metrics or 'periodicity_detected' in metrics


class TestSegmentLifecycleStages:
    """Test segment lifecycle stage detection."""

    def test_detect_segment_lifecycle_stages_basic(self):
        """Test basic lifecycle stage detection."""
        # Create data with growth trend
        dates = pd.date_range('2023-01-01', periods=100)
        sales_growth = np.linspace(1, 20, 100)

        test_data = pd.DataFrame({
            'segment_id': ['NEW_SEGMENT'] * 100,
            'daily_sales': sales_growth,
            'date': dates,
        })

        result = detect_segment_lifecycle_stages(test_data, segment_col='segment_id', date_col='date')

        assert 'lifecycle_stages' in result
        assert 'stage_characteristics' in result
        assert 'strategic_recommendations' in result

    def test_lifecycle_stage_identification(self):
        """Test that lifecycle stages are correctly identified."""
        # Create obvious growth trajectory
        dates = pd.date_range('2023-01-01', periods=200)
        sales_growth = np.concatenate([
            np.full(50, 5),  # Introduction
            np.linspace(5, 20, 50),  # Growth
            np.full(50, 20),  # Maturity
            np.linspace(20, 10, 50),  # Decline
        ])

        test_data = pd.DataFrame({
            'segment_id': ['PRODUCT_1'] * 200,
            'daily_sales': sales_growth,
            'date': dates,
        })

        result = detect_segment_lifecycle_stages(test_data, segment_col='segment_id', date_col='date')

        assert len(result['lifecycle_stages']) > 0
        assert 'stage_name' in result['lifecycle_stages'][0]

    def test_lifecycle_characteristics_presence(self):
        """Test that stage characteristics are present."""
        dates = pd.date_range('2023-01-01', periods=100)
        test_data = pd.DataFrame({
            'segment_id': ['SEG_1'] * 100,
            'daily_sales': np.random.randint(5, 30, 100),
            'date': dates,
        })

        result = detect_segment_lifecycle_stages(test_data, segment_col='segment_id', date_col='date')

        assert 'stage_characteristics' in result
        for stage_char in result['stage_characteristics']:
            assert 'characteristic_name' in stage_char


class TestSegmentAnalysisIntegration:
    """Integration tests for segment analysis functions."""

    def test_all_functions_with_real_like_data(self):
        """Test all functions with realistic M5-like data structure."""
        np.random.seed(42)
        n_records = 300

        test_data = pd.DataFrame({
            'cat_id': np.random.choice(['FOODS', 'HOUSEHOLD', 'HOBBIES'], n_records),
            'dept_id': np.random.choice(['DEPT_1', 'DEPT_2', 'DEPT_3', 'DEPT_4'], n_records),
            'segment_id': np.random.choice(['SEGMENT_A', 'SEGMENT_B', 'SEGMENT_C'], n_records),
            'daily_sales': np.random.randint(1, 50, n_records),
            'date': pd.date_range('2023-01-01', periods=n_records),
        })

        # All functions should work without errors
        result1 = analyze_category_behavior_patterns(test_data, category_col='cat_id')
        assert result1 is not None

        result2 = analyze_department_segment_patterns(test_data, department_col='dept_id')
        assert result2 is not None

        result3 = compute_segment_performance_metrics(test_data, segment_col='segment_id')
        assert result3 is not None

        result4 = analyze_segment_seasonality_patterns(test_data, segment_col='segment_id', date_col='date')
        assert result4 is not None

        result5 = detect_segment_lifecycle_stages(test_data, segment_col='segment_id', date_col='date')
        assert result5 is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        test_data = pd.DataFrame({
            'cat_id': [],
            'daily_sales': [],
        })

        result = analyze_category_behavior_patterns(test_data, category_col='cat_id')
        assert result is not None
        assert 'behavioral_metrics' in result

    def test_single_category(self):
        """Test with only one category."""
        test_data = pd.DataFrame({
            'cat_id': ['FOODS'] * 50,
            'daily_sales': np.random.randint(1, 20, 50),
        })

        result = analyze_category_behavior_patterns(test_data, category_col='cat_id')
        assert 'FOODS' in result['behavioral_metrics']

    def test_nan_handling(self):
        """Test handling of NaN values."""
        test_data = pd.DataFrame({
            'cat_id': ['FOODS'] * 50 + ['HOUSEHOLD'] * 50,
            'daily_sales': np.concatenate([
                np.random.randint(1, 20, 45),
                [np.nan] * 5,
                np.random.randint(1, 20, 50),
            ]),
        })

        result = analyze_category_behavior_patterns(test_data, category_col='cat_id')
        assert result is not None

    def test_single_value_series(self):
        """Test with constant sales values."""
        test_data = pd.DataFrame({
            'segment_id': ['CONST_SEG'] * 50,
            'daily_sales': [10] * 50,
            'date': pd.date_range('2023-01-01', periods=50),
        })

        result = compute_segment_performance_metrics(test_data, segment_col='segment_id')
        assert result is not None
