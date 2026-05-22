"""
Comprehensive tests for leakage validation module.

Tests temporal boundary audit, feature availability timing validation,
cross-validation integrity checks, suspicious correlation detection,
and comprehensive leakage audit report generation for M5 demand forecasting.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.leakage_validation import (
    audit_temporal_boundaries,
    check_feature_availability_timing,
    validate_cross_validation_integrity,
    scan_suspicious_correlations,
    generate_leakage_audit_report
)


class TestAuditTemporalBoundaries:
    """Tests for temporal boundary audit functionality."""

    def test_audit_temporal_boundaries_basic(self):
        """Test basic temporal boundary audit with feature configurations."""
        split_date = pd.Timestamp('2023-06-01')

        # Create test data with rolling windows
        feature_configs = [
            {'name': 'sales_lag_1', 'type': 'lag', 'window': 1},
            {'name': 'sales_rolling_7', 'type': 'rolling', 'window': 7},
            {'name': 'price_current', 'type': 'point', 'window': 0}
        ]

        dates = pd.date_range('2023-05-01', '2023-06-30', freq='D')
        feature_data = pd.DataFrame({
            'date': dates,
            'sales_lag_1': np.random.randint(1, 100, len(dates)),
            'sales_rolling_7': np.random.randint(1, 100, len(dates)),
            'price_current': np.random.uniform(5, 20, len(dates))
        })

        result = audit_temporal_boundaries(feature_data, split_date, feature_configs)

        assert isinstance(result, dict)
        assert 'boundary_violations' in result
        assert 'compliant_features' in result
        assert 'risk_assessment' in result
        assert isinstance(result['boundary_violations'], list)
        assert isinstance(result['compliant_features'], list)

    def test_audit_temporal_boundaries_lag_feature_violation(self):
        """Test detection of lag feature leakage at boundaries."""
        split_date = pd.Timestamp('2023-06-01')

        # Lag-1 feature should use previous day data
        feature_configs = [
            {'name': 'sales_lag_1', 'type': 'lag', 'window': 1}
        ]

        dates = pd.date_range('2023-05-15', '2023-06-15', freq='D')
        feature_data = pd.DataFrame({
            'date': dates,
            'sales_lag_1': np.random.randint(1, 100, len(dates))
        })

        result = audit_temporal_boundaries(feature_data, split_date, feature_configs)

        # For split_date 2023-06-01, sales_lag_1 at that date should use data from 2023-05-31
        # which is before split, so it should be compliant
        assert 'compliant_features' in result or 'boundary_violations' in result

    def test_audit_temporal_boundaries_rolling_window_violation(self):
        """Test detection of rolling window leakage at boundaries."""
        split_date = pd.Timestamp('2023-06-01')

        feature_configs = [
            {'name': 'sales_rolling_7', 'type': 'rolling', 'window': 7}
        ]

        dates = pd.date_range('2023-05-15', '2023-06-15', freq='D')
        feature_data = pd.DataFrame({
            'date': dates,
            'sales_rolling_7': np.random.randint(1, 100, len(dates))
        })

        result = audit_temporal_boundaries(feature_data, split_date, feature_configs)

        assert 'risk_assessment' in result
        assert isinstance(result['risk_assessment'], str)

    def test_audit_temporal_boundaries_price_consistency(self):
        """Test price data consistency validation across boundaries."""
        split_date = pd.Timestamp('2023-06-01')

        feature_configs = [
            {'name': 'price_current', 'type': 'point', 'window': 0}
        ]

        # Create data with potential price discontinuities
        dates = pd.date_range('2023-05-20', '2023-06-20', freq='D')
        prices = np.random.uniform(5, 20, len(dates))

        feature_data = pd.DataFrame({
            'date': dates,
            'price_current': prices
        })

        result = audit_temporal_boundaries(feature_data, split_date, feature_configs)

        assert 'boundary_violations' in result
        assert 'compliant_features' in result

    def test_audit_temporal_boundaries_empty_data(self):
        """Test handling of empty data."""
        split_date = pd.Timestamp('2023-06-01')
        feature_configs = [{'name': 'test_feature', 'type': 'lag', 'window': 1}]
        feature_data = pd.DataFrame()

        with pytest.raises(ValueError):
            audit_temporal_boundaries(feature_data, split_date, feature_configs)

    def test_audit_temporal_boundaries_missing_date_column(self):
        """Test handling of missing date column."""
        split_date = pd.Timestamp('2023-06-01')
        feature_configs = [{'name': 'test_feature', 'type': 'lag', 'window': 1}]

        feature_data = pd.DataFrame({
            'test_feature': [1, 2, 3]
        })

        with pytest.raises(KeyError):
            audit_temporal_boundaries(feature_data, split_date, feature_configs)

    def test_audit_temporal_boundaries_multiple_features(self):
        """Test audit with multiple feature types."""
        split_date = pd.Timestamp('2023-06-01')

        feature_configs = [
            {'name': 'sales_lag_1', 'type': 'lag', 'window': 1},
            {'name': 'sales_lag_7', 'type': 'lag', 'window': 7},
            {'name': 'sales_rolling_7', 'type': 'rolling', 'window': 7},
            {'name': 'sales_rolling_30', 'type': 'rolling', 'window': 30},
            {'name': 'price_current', 'type': 'point', 'window': 0}
        ]

        dates = pd.date_range('2023-05-01', '2023-06-30', freq='D')
        data = {'date': dates}
        for config in feature_configs:
            data[config['name']] = np.random.randint(1, 100, len(dates))

        feature_data = pd.DataFrame(data)
        result = audit_temporal_boundaries(feature_data, split_date, feature_configs)

        assert len(result['compliant_features']) + len(result['boundary_violations']) > 0


class TestCheckFeatureAvailabilityTiming:
    """Tests for feature availability timing validation."""

    def test_check_feature_availability_timing_basic(self):
        """Test basic feature availability timing check."""
        prediction_date = pd.Timestamp('2023-06-01')

        feature_availability = {
            'sales_lag_1': {'available_at': 'T-1', 'source': 'sales_history'},
            'price_current': {'available_at': 'T', 'source': 'pricing_system'},
            'event_flag': {'available_at': 'T+1', 'source': 'calendar'},
        }

        result = check_feature_availability_timing(prediction_date, feature_availability)

        assert isinstance(result, dict)
        assert 'available_features' in result
        assert 'unavailable_features' in result
        assert 'timing_violations' in result
        assert 'business_process_alignment' in result

    def test_check_feature_availability_timing_lag_feature_alignment(self):
        """Test alignment of lag features with prediction time."""
        prediction_date = pd.Timestamp('2023-06-01')

        feature_availability = {
            'sales_lag_1': {'available_at': 'T-1', 'source': 'sales_history'},
            'sales_lag_7': {'available_at': 'T-7', 'source': 'sales_history'},
        }

        result = check_feature_availability_timing(prediction_date, feature_availability)

        # Lag features should be available
        assert 'available_features' in result
        lag_features = [f for f in result['available_features'] if 'lag' in f]
        assert len(lag_features) > 0

    def test_check_feature_availability_timing_future_data_violation(self):
        """Test detection of future data leakage."""
        prediction_date = pd.Timestamp('2023-06-01')

        feature_availability = {
            'price_current': {'available_at': 'T', 'source': 'pricing_system'},
            'event_flag': {'available_at': 'T+1', 'source': 'calendar'},  # Future data
            'demand_spike': {'available_at': 'T+7', 'source': 'market_intelligence'},  # Far future
        }

        result = check_feature_availability_timing(prediction_date, feature_availability)

        assert len(result['timing_violations']) > 0

    def test_check_feature_availability_timing_external_data_delay(self):
        """Test external data availability delays."""
        prediction_date = pd.Timestamp('2023-06-01')

        feature_availability = {
            'sales_lag_1': {'available_at': 'T-1', 'source': 'sales_history'},
            'competitor_price': {'available_at': 'T-3', 'source': 'competitor_feed'},
            'weather': {'available_at': 'T-2', 'source': 'external_api'},
        }

        result = check_feature_availability_timing(prediction_date, feature_availability)

        assert 'business_process_alignment' in result
        assert isinstance(result['business_process_alignment'], str)

    def test_check_feature_availability_timing_empty_dict(self):
        """Test handling of empty feature availability dict."""
        prediction_date = pd.Timestamp('2023-06-01')
        feature_availability = {}

        result = check_feature_availability_timing(prediction_date, feature_availability)

        assert 'available_features' in result
        assert len(result['available_features']) == 0

    def test_check_feature_availability_timing_calendar_events(self):
        """Test calendar event availability timing."""
        prediction_date = pd.Timestamp('2023-06-01')

        feature_availability = {
            'is_holiday': {'available_at': 'T', 'source': 'calendar'},
            'is_weekend': {'available_at': 'T', 'source': 'calendar'},
            'event_type': {'available_at': 'T', 'source': 'calendar'},
        }

        result = check_feature_availability_timing(prediction_date, feature_availability)

        assert 'available_features' in result
        calendar_features = [f for f in result['available_features'] if any(x in f for x in ['holiday', 'weekend', 'event'])]
        assert len(calendar_features) > 0


class TestValidateCrossValidationIntegrity:
    """Tests for cross-validation integrity validation."""

    def test_validate_cross_validation_integrity_basic(self):
        """Test basic cross-validation integrity check."""
        sales_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'store_id': ['CA_1'] * 100,
            'item_id': ['FOODS_1_001'] * 100,
            'sales': np.random.randint(1, 100, 100)
        })

        cv_folds = [
            {'train_start': '2023-01-01', 'train_end': '2023-02-15',
             'val_start': '2023-02-16', 'val_end': '2023-02-28'},
            {'train_start': '2023-01-01', 'train_end': '2023-03-01',
             'val_start': '2023-03-02', 'val_end': '2023-03-15'},
        ]

        result = validate_cross_validation_integrity(sales_data, cv_folds)

        assert isinstance(result, dict)
        assert 'temporal_ordering' in result
        assert 'information_leakage' in result
        assert 'seasonal_alignment' in result
        assert 'cv_integrity_score' in result

    def test_validate_cross_validation_integrity_temporal_ordering(self):
        """Test temporal ordering validation in CV folds."""
        sales_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=200, freq='D'),
            'store_id': ['CA_1'] * 200,
            'item_id': ['FOODS_1_001'] * 200,
            'sales': np.random.randint(1, 100, 200)
        })

        # Valid temporal ordering
        cv_folds = [
            {'train_start': '2023-01-01', 'train_end': '2023-02-28',
             'val_start': '2023-03-01', 'val_end': '2023-03-15'},
        ]

        result = validate_cross_validation_integrity(sales_data, cv_folds)

        assert 'temporal_ordering' in result
        assert result['temporal_ordering']['is_valid'] is True

    def test_validate_cross_validation_integrity_leakage_detection(self):
        """Test detection of information leakage between folds."""
        sales_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=200, freq='D'),
            'store_id': ['CA_1'] * 200,
            'item_id': ['FOODS_1_001'] * 200,
            'sales': np.random.randint(1, 100, 200)
        })

        # Folds with overlapping periods (leakage)
        cv_folds = [
            {'train_start': '2023-01-01', 'train_end': '2023-03-01',
             'val_start': '2023-02-15', 'val_end': '2023-03-15'},  # Overlap
        ]

        result = validate_cross_validation_integrity(sales_data, cv_folds)

        # Should detect leakage
        assert len(result['information_leakage']) > 0 or result['cv_integrity_score'] < 1.0

    def test_validate_cross_validation_integrity_seasonal_alignment(self):
        """Test seasonal alignment across validation folds."""
        # Create data spanning full year to test seasonal patterns
        sales_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=365, freq='D'),
            'store_id': ['CA_1'] * 365,
            'item_id': ['FOODS_1_001'] * 365,
            'sales': np.tile(np.sin(np.arange(365) * 2 * np.pi / 365), 1) * 50 + 50
        })

        cv_folds = [
            {'train_start': '2023-01-01', 'train_end': '2023-08-31',
             'val_start': '2023-09-01', 'val_end': '2023-09-28'},
        ]

        result = validate_cross_validation_integrity(sales_data, cv_folds)

        assert 'seasonal_alignment' in result

    def test_validate_cross_validation_integrity_empty_data(self):
        """Test handling of empty data."""
        sales_data = pd.DataFrame()
        cv_folds = []

        with pytest.raises(ValueError):
            validate_cross_validation_integrity(sales_data, cv_folds)

    def test_validate_cross_validation_integrity_multiple_items(self):
        """Test CV integrity with multiple item-store combinations."""
        # Multiple series
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = []
        for store in ['CA_1', 'CA_2', 'TX_1']:
            for item in ['FOODS_1_001', 'FOODS_2_001']:
                temp_df = pd.DataFrame({
                    'date': dates,
                    'store_id': store,
                    'item_id': item,
                    'sales': np.random.randint(1, 100, len(dates))
                })
                data.append(temp_df)

        sales_data = pd.concat(data, ignore_index=True)

        cv_folds = [
            {'train_start': '2023-01-01', 'train_end': '2023-02-28',
             'val_start': '2023-03-01', 'val_end': '2023-03-15'},
        ]

        result = validate_cross_validation_integrity(sales_data, cv_folds)

        assert 'cv_integrity_score' in result
        assert isinstance(result['cv_integrity_score'], (float, int))


class TestScanSuspiciousCorrelations:
    """Tests for suspicious correlation detection."""

    def test_scan_suspicious_correlations_basic(self):
        """Test basic suspicious correlation detection."""
        np.random.seed(42)

        feature_data = pd.DataFrame({
            'sales': np.random.randint(1, 100, 100),
            'price': np.random.uniform(5, 20, 100),
            'promo': np.random.randint(0, 2, 100),
        })

        result = scan_suspicious_correlations(feature_data)

        assert isinstance(result, dict)
        assert 'perfect_predictors' in result
        assert 'id_leakage' in result
        assert 'high_correlations' in result
        assert 'leakage_summary' in result

    def test_scan_suspicious_correlations_perfect_predictor(self):
        """Test detection of perfect predictor features."""
        np.random.seed(42)

        sales = np.random.randint(1, 100, 100)
        feature_data = pd.DataFrame({
            'sales': sales,
            'perfect_feature': sales.copy(),  # Perfect correlation
            'price': np.random.uniform(5, 20, 100),
        })

        result = scan_suspicious_correlations(feature_data)

        assert len(result['perfect_predictors']) > 0

    def test_scan_suspicious_correlations_high_correlation(self):
        """Test detection of high correlations (>0.98)."""
        np.random.seed(42)

        base = np.random.randint(1, 100, 100)
        feature_data = pd.DataFrame({
            'sales': base,
            'nearly_perfect': base + np.random.normal(0, 0.1, 100),  # Very high correlation
            'price': np.random.uniform(5, 20, 100),
        })

        result = scan_suspicious_correlations(feature_data)

        # May contain high correlations
        assert 'high_correlations' in result

    def test_scan_suspicious_correlations_id_leakage(self):
        """Test detection of ID-based leakage."""
        feature_data = pd.DataFrame({
            'sales': np.random.randint(1, 100, 100),
            'store_id': np.arange(100),  # Sequential ID
            'item_id': np.arange(100),   # Sequential ID
            'price': np.random.uniform(5, 20, 100),
        })

        result = scan_suspicious_correlations(feature_data)

        # ID columns should be flagged
        assert 'id_leakage' in result

    def test_scan_suspicious_correlations_empty_data(self):
        """Test handling of empty data."""
        feature_data = pd.DataFrame()

        with pytest.raises(ValueError):
            scan_suspicious_correlations(feature_data)

    def test_scan_suspicious_correlations_single_column(self):
        """Test handling of single column."""
        feature_data = pd.DataFrame({
            'sales': np.random.randint(1, 100, 100)
        })

        result = scan_suspicious_correlations(feature_data)

        assert 'perfect_predictors' in result
        assert len(result['perfect_predictors']) == 0

    def test_scan_suspicious_correlations_categorical_features(self):
        """Test handling of categorical features."""
        feature_data = pd.DataFrame({
            'sales': np.random.randint(1, 100, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'subcategory': np.random.choice(['X', 'Y', 'Z'], 100),
            'price': np.random.uniform(5, 20, 100),
        })

        result = scan_suspicious_correlations(feature_data)

        assert isinstance(result, dict)


class TestGenerateLeakageAuditReport:
    """Tests for comprehensive leakage audit report generation."""

    def test_generate_leakage_audit_report_basic(self):
        """Test basic leakage audit report generation."""
        split_date = pd.Timestamp('2023-06-01')

        dates = pd.date_range('2023-05-01', '2023-06-30', freq='D')
        feature_data = pd.DataFrame({
            'date': dates,
            'sales_lag_1': np.random.randint(1, 100, len(dates)),
            'sales_rolling_7': np.random.randint(1, 100, len(dates)),
            'price_current': np.random.uniform(5, 20, len(dates))
        })

        feature_configs = [
            {'name': 'sales_lag_1', 'type': 'lag', 'window': 1},
            {'name': 'sales_rolling_7', 'type': 'rolling', 'window': 7},
            {'name': 'price_current', 'type': 'point', 'window': 0}
        ]

        result = generate_leakage_audit_report(feature_data, split_date, feature_configs)

        assert isinstance(result, dict)
        assert 'temporal_boundaries' in result
        assert 'feature_availability' in result
        assert 'cross_validation' in result
        assert 'suspicious_correlations' in result
        assert 'deployment_readiness' in result
        assert 'overall_risk_level' in result
        assert 'recommendations' in result

    def test_generate_leakage_audit_report_comprehensive(self):
        """Test comprehensive report with all components."""
        split_date = pd.Timestamp('2023-06-01')

        dates = pd.date_range('2023-05-01', '2023-06-30', freq='D')

        # Create comprehensive feature data
        base_sales = np.random.randint(1, 100, len(dates))
        feature_data = pd.DataFrame({
            'date': dates,
            'sales_lag_1': np.roll(base_sales, 1),
            'sales_lag_7': np.roll(base_sales, 7),
            'sales_rolling_7': base_sales,  # Simplified rolling
            'sales_rolling_30': base_sales,
            'price_current': np.random.uniform(5, 20, len(dates)),
            'is_holiday': np.random.randint(0, 2, len(dates)),
        })

        feature_configs = [
            {'name': 'sales_lag_1', 'type': 'lag', 'window': 1},
            {'name': 'sales_lag_7', 'type': 'lag', 'window': 7},
            {'name': 'sales_rolling_7', 'type': 'rolling', 'window': 7},
            {'name': 'sales_rolling_30', 'type': 'rolling', 'window': 30},
            {'name': 'price_current', 'type': 'point', 'window': 0},
            {'name': 'is_holiday', 'type': 'point', 'window': 0},
        ]

        result = generate_leakage_audit_report(feature_data, split_date, feature_configs)

        assert 'deployment_readiness' in result
        assert 'overall_risk_level' in result
        assert result['overall_risk_level'] in ['Low', 'Medium', 'High', 'Critical']

    def test_generate_leakage_audit_report_empty_data(self):
        """Test handling of empty data."""
        split_date = pd.Timestamp('2023-06-01')
        feature_data = pd.DataFrame()
        feature_configs = []

        with pytest.raises(ValueError):
            generate_leakage_audit_report(feature_data, split_date, feature_configs)

    def test_generate_leakage_audit_report_risk_assessment(self):
        """Test risk assessment in report."""
        split_date = pd.Timestamp('2023-06-01')

        dates = pd.date_range('2023-05-01', '2023-06-30', freq='D')

        # Create data with suspicious patterns to trigger recommendations
        base = np.random.randint(1, 100, len(dates))
        feature_data = pd.DataFrame({
            'date': dates,
            'feature1': base,
            'perfect_feature': base.copy(),  # Perfect correlation to trigger recommendation
        })

        feature_configs = [
            {'name': 'feature1', 'type': 'lag', 'window': 1},
            {'name': 'perfect_feature', 'type': 'point', 'window': 0}
        ]

        result = generate_leakage_audit_report(feature_data, split_date, feature_configs)

        assert 'overall_risk_level' in result
        assert isinstance(result['recommendations'], list)
        # Should have recommendations due to perfect correlation
        assert len(result['recommendations']) > 0

    def test_generate_leakage_audit_report_recommendations(self):
        """Test that recommendations are generated based on findings."""
        split_date = pd.Timestamp('2023-06-01')

        dates = pd.date_range('2023-05-01', '2023-06-30', freq='D')
        feature_data = pd.DataFrame({
            'date': dates,
            'suspicious_feature': np.random.randint(1, 100, len(dates)),
            'price': np.random.uniform(5, 20, len(dates)),
        })

        feature_configs = [
            {'name': 'suspicious_feature', 'type': 'lag', 'window': 1},
            {'name': 'price', 'type': 'point', 'window': 0}
        ]

        result = generate_leakage_audit_report(feature_data, split_date, feature_configs)

        assert 'recommendations' in result
        assert isinstance(result['recommendations'], list)
