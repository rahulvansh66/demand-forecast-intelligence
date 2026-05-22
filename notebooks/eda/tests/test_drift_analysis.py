import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.drift_analysis import (
    compare_temporal_distributions,
    analyze_seasonal_representativeness,
    detect_category_drift,
    compute_drift_severity_scores,
    validate_temporal_split_integrity,
)


class TestCompareTemporalDistributions:
    """Test temporal distribution comparison across train/validation periods."""

    def test_compare_temporal_distributions_with_drift(self):
        """Test detection of distribution drift between train and validation."""
        np.random.seed(42)
        train_data = pd.DataFrame({
            'sales': np.random.poisson(5, 1000),
            'category': ['FOODS'] * 500 + ['HOUSEHOLD'] * 500
        })

        # Validation data with higher mean (drift)
        validation_data = pd.DataFrame({
            'sales': np.random.poisson(7, 200),
            'category': ['FOODS'] * 100 + ['HOUSEHOLD'] * 100
        })

        result = compare_temporal_distributions(train_data, validation_data, ['sales'])

        assert 'ks_tests' in result
        assert 'mannwhitney_tests' in result
        assert 'effect_sizes' in result
        assert 'sales' in result['ks_tests']
        assert 'ks_statistic' in result['ks_tests']['sales']
        assert 'p_value' in result['ks_tests']['sales']
        assert result['ks_tests']['sales']['p_value'] < 0.05  # Should detect difference
        assert result['mannwhitney_tests']['sales']['p_value'] < 0.05

    def test_compare_temporal_distributions_no_drift(self):
        """Test with identical distributions (no drift)."""
        np.random.seed(42)
        data = np.random.normal(10, 2, 1000)
        train_data = pd.DataFrame({'sales': data})
        validation_data = pd.DataFrame({'sales': data})

        result = compare_temporal_distributions(train_data, validation_data, ['sales'])

        assert result['ks_tests']['sales']['p_value'] > 0.05  # Should not detect difference
        assert 'cohens_d' in result['effect_sizes']['sales']

    def test_compare_temporal_distributions_multiple_columns(self):
        """Test with multiple numerical columns."""
        np.random.seed(42)
        train_data = pd.DataFrame({
            'sales': np.random.poisson(5, 1000),
            'price': np.random.normal(100, 10, 1000),
        })

        validation_data = pd.DataFrame({
            'sales': np.random.poisson(7, 200),
            'price': np.random.normal(110, 15, 200),
        })

        result = compare_temporal_distributions(
            train_data, validation_data, ['sales', 'price']
        )

        assert 'sales' in result['ks_tests']
        assert 'price' in result['ks_tests']

    def test_compare_temporal_distributions_with_nan(self):
        """Test handling of NaN values."""
        np.random.seed(42)
        train_data = pd.DataFrame({
            'sales': np.concatenate([np.random.poisson(5, 990), [np.nan] * 10])
        })

        validation_data = pd.DataFrame({
            'sales': np.random.poisson(5, 200)
        })

        result = compare_temporal_distributions(train_data, validation_data, ['sales'])

        assert 'sales' in result['ks_tests']
        assert not np.isnan(result['ks_tests']['sales']['p_value'])

    def test_compare_temporal_distributions_empty_data(self):
        """Test with empty dataframe."""
        train_data = pd.DataFrame({'sales': []})
        validation_data = pd.DataFrame({'sales': []})

        with pytest.raises((ValueError, KeyError)):
            compare_temporal_distributions(train_data, validation_data, ['sales'])


class TestAnalyzeSeasonalRepresentativeness:
    """Test seasonal representativeness of 28-day validation period."""

    def test_analyze_seasonal_representativeness_complete(self):
        """Test seasonal analysis with complete 28-day period."""
        np.random.seed(42)

        # Create data spanning multiple years with seasonal pattern (1941 days total)
        dates = pd.date_range('2014-01-29', periods=1941, freq='D')
        sales = 10 + 5 * np.sin(2 * np.pi * np.arange(1941) / 365.25)
        sales = np.maximum(sales, 0)  # Ensure non-negative

        calendar_data = pd.DataFrame({
            'date': dates,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'day': dates.day,
            'day_index': range(1, 1942)
        })

        sales_data = pd.DataFrame({
            'sales': sales,
            'date': dates
        })

        result = analyze_seasonal_representativeness(
            sales_data, calendar_data, val_start_day=1914, val_end_day=1941
        )

        assert 'validation_period' in result
        assert 'seasonal_alignment' in result
        assert 'day_of_week_distribution' in result
        assert 'monthly_coverage' in result

    def test_analyze_seasonal_representativeness_missing_data(self):
        """Test with missing calendar data."""
        sales_data = pd.DataFrame({'sales': np.random.poisson(5, 100)})

        with pytest.raises((KeyError, ValueError)):
            analyze_seasonal_representativeness(
                sales_data, pd.DataFrame(), val_start_day=80, val_end_day=100
            )


class TestDetectCategoryDrift:
    """Test category-specific drift detection."""

    def test_detect_category_drift_with_drift(self):
        """Test detection of category-specific drift."""
        np.random.seed(42)

        train_data = pd.DataFrame({
            'sales': np.random.poisson(5, 1000),
            'category': ['FOODS'] * 600 + ['HOUSEHOLD'] * 400
        })

        # Different drift for each category
        validation_data = pd.DataFrame({
            'sales': np.concatenate([
                np.random.poisson(7, 100),  # FOODS with higher mean
                np.random.poisson(4, 100),  # HOUSEHOLD with lower mean
            ]),
            'category': ['FOODS'] * 100 + ['HOUSEHOLD'] * 100
        })

        result = detect_category_drift(train_data, validation_data, 'category')

        assert 'category_drift_tests' in result
        assert 'FOODS' in result['category_drift_tests']
        assert 'HOUSEHOLD' in result['category_drift_tests']
        assert 'overall_drift_detected' in result

    def test_detect_category_drift_no_category_column(self):
        """Test with missing category column."""
        train_data = pd.DataFrame({'sales': [1, 2, 3, 4, 5]})
        validation_data = pd.DataFrame({'sales': [1, 2, 3, 4, 5]})

        with pytest.raises((KeyError, ValueError)):
            detect_category_drift(train_data, validation_data, 'category')

    def test_detect_category_drift_single_category(self):
        """Test with single category."""
        np.random.seed(42)

        train_data = pd.DataFrame({
            'sales': np.random.poisson(5, 500),
            'category': ['FOODS'] * 500
        })

        validation_data = pd.DataFrame({
            'sales': np.random.poisson(5, 100),
            'category': ['FOODS'] * 100
        })

        result = detect_category_drift(train_data, validation_data, 'category')

        assert 'category_drift_tests' in result
        assert 'FOODS' in result['category_drift_tests']


class TestComputeDriftSeverityScores:
    """Test drift severity quantification."""

    def test_compute_drift_severity_scores_high_drift(self):
        """Test severity scoring with high drift."""
        np.random.seed(42)

        ks_results = {
            'sales': {'ks_statistic': 0.3, 'p_value': 0.001},
            'price': {'ks_statistic': 0.15, 'p_value': 0.1}
        }

        effect_sizes = {
            'sales': {'cohens_d': 1.5},
            'price': {'cohens_d': 0.3}
        }

        result = compute_drift_severity_scores(ks_results, effect_sizes)

        assert 'severity_scores' in result
        assert 'overall_severity' in result
        assert 'drift_classification' in result
        assert result['drift_classification'] in ['None', 'Low', 'Moderate', 'High', 'Critical']
        assert result['severity_scores']['sales'] > result['severity_scores']['price']

    def test_compute_drift_severity_scores_no_drift(self):
        """Test severity scoring with no drift."""
        ks_results = {
            'sales': {'ks_statistic': 0.01, 'p_value': 0.95}
        }

        effect_sizes = {
            'sales': {'cohens_d': 0.02}
        }

        result = compute_drift_severity_scores(ks_results, effect_sizes)

        assert result['drift_classification'] == 'None'

    def test_compute_drift_severity_scores_empty(self):
        """Test with empty results."""
        with pytest.raises((ValueError, KeyError)):
            compute_drift_severity_scores({}, {})


class TestValidateTemporalSplitIntegrity:
    """Test temporal split boundary validation."""

    def test_validate_temporal_split_integrity_valid(self):
        """Test with valid temporal split."""
        np.random.seed(42)

        # M5 context: d_1 to d_1913 (train), d_1914 to d_1941 (validation)
        calendar_data = pd.DataFrame({
            'date': pd.date_range('2014-01-29', periods=1941, freq='D'),
            'day_of_week': [i % 7 for i in range(1941)],
            'month': [((i // 30) % 12) + 1 for i in range(1941)]
        })

        result = validate_temporal_split_integrity(
            calendar_data, train_days=1913, val_days=28
        )

        assert 'split_valid' in result
        assert 'train_period' in result
        assert 'val_period' in result
        assert 'boundary_check' in result

    def test_validate_temporal_split_integrity_invalid(self):
        """Test with invalid temporal split."""
        calendar_data = pd.DataFrame({
            'date': pd.date_range('2014-01-29', periods=100, freq='D')
        })

        result = validate_temporal_split_integrity(
            calendar_data, train_days=1913, val_days=28
        )

        assert result['split_valid'] is False

    def test_validate_temporal_split_integrity_missing_columns(self):
        """Test with missing date column."""
        calendar_data = pd.DataFrame({'other_col': [1, 2, 3]})

        with pytest.raises((KeyError, ValueError)):
            validate_temporal_split_integrity(
                calendar_data, train_days=100, val_days=10
            )


class TestIntegration:
    """Integration tests for complete drift analysis workflow."""

    def test_complete_drift_analysis_workflow(self):
        """Test full workflow from raw data to severity assessment."""
        np.random.seed(42)

        # Create realistic M5-like data
        dates = pd.date_range('2014-01-29', periods=1941, freq='D')

        train_indices = np.arange(1913)
        val_indices = np.arange(1913, 1941)

        train_data = pd.DataFrame({
            'sales': np.random.poisson(5, 1913),
            'category': np.random.choice(['FOODS', 'HOUSEHOLD'], 1913),
            'date': dates[train_indices]
        })

        validation_data = pd.DataFrame({
            'sales': np.random.poisson(6, 28),  # Slight drift
            'category': np.random.choice(['FOODS', 'HOUSEHOLD'], 28),
            'date': dates[val_indices]
        })

        calendar_data = pd.DataFrame({
            'date': dates,
            'day_of_week': dates.dayofweek,
            'month': dates.month
        })

        # Step 1: Temporal distribution comparison
        dist_result = compare_temporal_distributions(
            train_data, validation_data, ['sales']
        )
        assert 'ks_tests' in dist_result

        # Step 2: Seasonal representativeness
        seasonal_result = analyze_seasonal_representativeness(
            train_data, calendar_data, val_start_day=1914, val_end_day=1941
        )
        assert 'seasonal_alignment' in seasonal_result

        # Step 3: Category drift
        category_result = detect_category_drift(
            train_data, validation_data, 'category'
        )
        assert 'category_drift_tests' in category_result

        # Step 4: Severity scoring
        severity_result = compute_drift_severity_scores(
            dist_result['ks_tests'], dist_result['effect_sizes']
        )
        assert 'drift_classification' in severity_result

        # Step 5: Temporal split validation
        split_result = validate_temporal_split_integrity(
            calendar_data, train_days=1913, val_days=28
        )
        assert 'split_valid' in split_result
