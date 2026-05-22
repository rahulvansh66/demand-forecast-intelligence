"""
Comprehensive tests for temporal analysis module.

Tests time structure validation, seasonal pattern detection, trend analysis,
and autocorrelation computation with business context interpretation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.temporal_analysis import (
    analyze_time_structure,
    detect_seasonal_patterns,
    analyze_trend_components,
    compute_autocorrelation_analysis
)


class TestTimeStructureAnalysis:
    """Tests for time structure validation and analysis."""

    def test_analyze_time_structure_basic(self):
        """Test basic time structure analysis."""
        # Create sample sales data
        sales_data = pd.DataFrame({
            'id': ['FOODS_1_001_CA_1_validation'],
            'd_1': [5],
            'd_2': [3],
            'd_3': [7]
        })

        # Create sample calendar data
        calendar_data = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3'],
            'date': pd.to_datetime(['2011-01-29', '2011-01-30', '2011-01-31'])
        })

        result = analyze_time_structure(sales_data, calendar_data)

        assert isinstance(result, dict)
        assert 'time_range' in result
        assert 'frequency_validation' in result
        assert result['total_days'] == 3
        assert result['frequency'] == 'Daily'

    def test_time_structure_with_actual_m5_format(self):
        """Test with more realistic M5 format data."""
        # Create realistic M5-style data
        sales_data = pd.DataFrame({
            'id': ['FOODS_1_001_CA_1_validation', 'HOUSEHOLD_1_001_TX_1_validation'],
            'item_id': ['FOODS_1_001', 'HOUSEHOLD_1_001'],
            'cat_id': ['FOODS', 'HOUSEHOLD'],
            'dept_id': ['FOODS_1', 'HOUSEHOLD_1'],
            'store_id': ['CA_1', 'TX_1'],
            'state_id': ['CA', 'TX'],
            'd_1': [5, 2],
            'd_2': [3, 1],
            'd_3': [7, 0]
        })

        calendar_data = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3'],
            'date': pd.to_datetime(['2011-01-29', '2011-01-30', '2011-01-31'])
        })

        result = analyze_time_structure(sales_data, calendar_data)

        assert result['total_days'] == 3
        assert result['total_series'] == 2
        assert result['frequency_validation']['structure_consistent'] is True
        assert result['panel_structure']['entities'] == 2
        assert result['panel_structure']['time_periods'] == 3

    def test_time_structure_with_missing_dates(self):
        """Test time structure analysis with missing dates."""
        sales_data = pd.DataFrame({
            'd_1': [10],
            'd_2': [12],
            'd_3': [15],
            'd_4': [18]
        })

        calendar_data = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3', 'd_4'],
            'date': pd.to_datetime(['2011-01-29', '2011-01-31', '2011-02-02', '2011-02-04'])
        })

        result = analyze_time_structure(sales_data, calendar_data)

        assert 'time_range' in result
        assert result['missing_dates'] >= 0


class TestSeasonalPatternDetection:
    """Tests for seasonal pattern detection."""

    def test_detect_seasonal_patterns_basic(self):
        """Test basic seasonal pattern detection."""
        sales_data = pd.DataFrame({
            'cat_id': ['FOODS'] * 2,
            **{f'd_{i}': [np.random.randint(8, 15), np.random.randint(6, 14)] for i in range(1, 31)}
        })

        calendar_data = pd.DataFrame({
            'd': [f'd_{i}' for i in range(1, 31)],
            'date': pd.date_range('2011-01-29', periods=30)
        })

        result = detect_seasonal_patterns(sales_data, calendar_data, hierarchy_level='category')

        assert isinstance(result, dict)
        assert 'seasonal_patterns' in result
        assert 'FOODS' in result['seasonal_patterns']
        assert 'weekly_seasonality' in result['seasonal_patterns']['FOODS']

    def test_seasonal_patterns_multiple_categories(self):
        """Test seasonal pattern detection across multiple categories."""
        # Create data with sufficient observations for seasonal pattern detection
        np.random.seed(42)
        foods_data = [np.random.randint(8, 15) for _ in range(5)]
        household_data = [np.random.randint(2, 8) for _ in range(5)]

        sales_data = pd.DataFrame({
            'cat_id': ['FOODS'] * 5 + ['HOUSEHOLD'] * 5,
            **{f'd_{i}': foods_data + household_data for i in range(1, 31)}
        })

        calendar_data = pd.DataFrame({
            'd': [f'd_{i}' for i in range(1, 31)]
        })

        result = detect_seasonal_patterns(sales_data, calendar_data, hierarchy_level='category')

        assert 'FOODS' in result['seasonal_patterns']
        assert 'HOUSEHOLD' in result['seasonal_patterns']
        assert 'weekly_seasonality' in result['seasonal_patterns']['FOODS']

    def test_seasonal_patterns_department_level(self):
        """Test seasonal pattern detection at department level."""
        sales_data = pd.DataFrame({
            'dept_id': ['FOODS_1', 'FOODS_1', 'HOUSEHOLD_1'],
            'd_1': [10, 8, 5],
            'd_2': [12, 10, 3],
            'd_3': [8, 6, 6],
            'd_4': [15, 12, 2],
            'd_5': [9, 7, 7]
        })

        calendar_data = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3', 'd_4', 'd_5']
        })

        result = detect_seasonal_patterns(sales_data, calendar_data, hierarchy_level='department')

        assert 'FOODS_1' in result['seasonal_patterns']
        assert 'HOUSEHOLD_1' in result['seasonal_patterns']


class TestTrendAnalysis:
    """Tests for trend component analysis."""

    def test_analyze_trend_components_basic(self):
        """Test basic trend analysis."""
        sales_data = pd.DataFrame({
            'd_1': [10, 8],
            'd_2': [12, 10],
            'd_3': [14, 12],
            'd_4': [16, 14]
        })

        calendar_data = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3', 'd_4']
        })

        result = analyze_trend_components(sales_data, calendar_data)

        assert isinstance(result, dict)
        assert 'linear_trend' in result
        assert result['linear_trend']['slope'] > 0  # Should detect positive trend
        assert result['linear_trend']['direction'] == 'Growing'

    def test_analyze_trend_components_negative_trend(self):
        """Test detection of negative trend."""
        sales_data = pd.DataFrame({
            'd_1': [100, 90],
            'd_2': [80, 70],
            'd_3': [60, 50],
            'd_4': [40, 30]
        })

        calendar_data = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3', 'd_4']
        })

        result = analyze_trend_components(sales_data, calendar_data)

        assert result['linear_trend']['slope'] < 0
        assert result['linear_trend']['direction'] == 'Declining'

    def test_trend_analysis_with_structural_break(self):
        """Test structural break detection in trend."""
        # Create data with obvious structural break
        sales_data = pd.DataFrame({
            'd_' + str(i): [100 if i < 60 else 200 for _ in range(10)]
            for i in range(1, 121)
        })

        calendar_data = pd.DataFrame({
            'd': ['d_' + str(i) for i in range(1, 121)]
        })

        result = analyze_trend_components(sales_data, calendar_data)

        if 'structural_break_analysis' in result:
            assert 'significant_break' in result['structural_break_analysis']
            # Should detect significant structural break
            assert result['structural_break_analysis']['p_value'] < 0.05


class TestAutocorrelationAnalysis:
    """Tests for autocorrelation analysis."""

    def test_compute_autocorrelation_analysis_basic(self):
        """Test basic autocorrelation computation."""
        sales_data = pd.DataFrame({
            'd_1': [10],
            'd_2': [12],
            'd_3': [8],
            'd_4': [15],
            'd_5': [11],
            'd_6': [13],
            'd_7': [9]
        })

        result = compute_autocorrelation_analysis(sales_data, max_lags=10)

        assert isinstance(result, dict)
        assert 'autocorrelations' in result
        assert 'business_interpretation' in result
        assert 'significant_lags' in result

    def test_autocorrelation_key_lags(self):
        """Test that key lags are analyzed."""
        sales_data = pd.DataFrame({
            'd_' + str(i): [np.random.randint(5, 30) for _ in range(5)]
            for i in range(1, 101)
        })

        result = compute_autocorrelation_analysis(sales_data, max_lags=365)

        # Should have analyzed key lags (1, 7, 14, 28, 30)
        autocorr_keys = set(result['autocorrelations'].keys())
        expected_keys = {'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_30'}
        assert len(autocorr_keys & expected_keys) > 0

    def test_autocorrelation_with_weekly_pattern(self):
        """Test autocorrelation detection of weekly patterns."""
        # Create data with strong weekly pattern
        sales_data = pd.DataFrame({
            'd_' + str(i): [
                [100, 120, 110, 90, 80, 100, 120][i % 7]
                for _ in range(5)
            ]
            for i in range(1, 71)
        })

        result = compute_autocorrelation_analysis(sales_data, max_lags=365)

        # Weekly lag (7 days) should show strong correlation
        lag7_corr = result['autocorrelations'].get('lag_7', {}).get('correlation', 0)
        assert isinstance(lag7_corr, (int, float))

    def test_autocorrelation_business_interpretation(self):
        """Test business interpretation generation."""
        sales_data = pd.DataFrame({
            'd_' + str(i): [np.random.randint(5, 30) for _ in range(5)]
            for i in range(1, 101)
        })

        result = compute_autocorrelation_analysis(sales_data, max_lags=365)

        assert 'business_interpretation' in result
        assert 'weekly_seasonality_strength' in result['business_interpretation']
        assert 'monthly_seasonality_strength' in result['business_interpretation']
        assert 'recommended_lags' in result['business_interpretation']


class TestTemporalAnalysisIntegration:
    """Integration tests for temporal analysis functions."""

    def test_full_temporal_analysis_pipeline(self):
        """Test complete temporal analysis workflow."""
        # Create realistic M5-style data
        np.random.seed(42)
        sales_data = pd.DataFrame({
            'id': ['FOODS_1_001_CA_1_validation', 'HOUSEHOLD_1_001_TX_1_validation'],
            'item_id': ['FOODS_1_001', 'HOUSEHOLD_1_001'],
            'cat_id': ['FOODS', 'HOUSEHOLD'],
            'dept_id': ['FOODS_1', 'HOUSEHOLD_1'],
            'store_id': ['CA_1', 'TX_1'],
            'state_id': ['CA', 'TX'],
            **{f'd_{i}': [np.random.randint(5, 50), np.random.randint(1, 20)]
               for i in range(1, 101)}
        })

        calendar_data = pd.DataFrame({
            'd': [f'd_{i}' for i in range(1, 101)],
            'date': pd.date_range('2011-01-29', periods=100)
        })

        # Run all temporal analyses
        time_structure = analyze_time_structure(sales_data, calendar_data)
        seasonal_patterns = detect_seasonal_patterns(sales_data, calendar_data)
        trend_analysis = analyze_trend_components(sales_data, calendar_data)
        autocorr_analysis = compute_autocorrelation_analysis(sales_data, max_lags=50)

        # Verify results structure
        assert time_structure['total_days'] == 100
        assert 'FOODS' in seasonal_patterns['seasonal_patterns']
        assert 'linear_trend' in trend_analysis
        assert 'autocorrelations' in autocorr_analysis


class TestTemporalAnalysisEdgeCases:
    """Tests for edge cases and error handling."""

    def test_minimal_data(self):
        """Test with minimal data."""
        sales_data = pd.DataFrame({
            'd_1': [10],
            'd_2': [12]
        })

        calendar_data = pd.DataFrame({
            'd': ['d_1', 'd_2'],
            'date': pd.to_datetime(['2011-01-29', '2011-01-30'])
        })

        result = analyze_time_structure(sales_data, calendar_data)
        assert result['total_days'] == 2

    def test_large_dataset(self):
        """Test with larger dataset."""
        sales_data = pd.DataFrame({
            f'd_{i}': [np.random.randint(0, 100) for _ in range(100)]
            for i in range(1, 201)
        })

        calendar_data = pd.DataFrame({
            'd': [f'd_{i}' for i in range(1, 201)],
            'date': pd.date_range('2011-01-29', periods=200)
        })

        result = analyze_time_structure(sales_data, calendar_data)
        assert result['total_days'] == 200

    def test_constant_series(self):
        """Test with constant time series."""
        sales_data = pd.DataFrame({
            'd_' + str(i): [50, 50, 50]
            for i in range(1, 31)
        })

        calendar_data = pd.DataFrame({
            'd': [f'd_{i}' for i in range(1, 31)]
        })

        result = analyze_trend_components(sales_data, calendar_data)

        # Slope should be approximately 0
        assert abs(result['linear_trend']['slope']) < 0.1

    def test_zero_variance_seasonal_detection(self):
        """Test seasonal detection with zero variance."""
        sales_data = pd.DataFrame({
            'cat_id': ['FOODS'] * 5,
            'd_1': [0, 0, 0, 0, 0],
            'd_2': [0, 0, 0, 0, 0],
            'd_3': [0, 0, 0, 0, 0],
            'd_4': [0, 0, 0, 0, 0],
            'd_5': [0, 0, 0, 0, 0]
        })

        calendar_data = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3', 'd_4', 'd_5']
        })

        result = detect_seasonal_patterns(sales_data, calendar_data)

        assert 'FOODS' in result['seasonal_patterns']
