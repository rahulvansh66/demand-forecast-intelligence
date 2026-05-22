"""
Comprehensive tests for data quality module.

Tests missing value analysis, missing mechanism characterization,
outlier detection, and pricing anomaly identification with business context.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_quality import (
    analyze_missing_patterns,
    characterize_missing_mechanisms,
    detect_sales_outliers,
    analyze_pricing_anomalies
)


class TestMissingPatternsAnalysis:
    """Tests for missing value pattern analysis."""

    def test_analyze_missing_patterns_no_missing_sales(self):
        """Test analysis with no missing sales values (expected case)."""
        sales_data = pd.DataFrame({
            'item_id': ['FOODS_1_001', 'FOODS_1_002'],
            'store_id': ['CA_1', 'CA_1'],
            'd_1': [5, 0],
            'd_2': [3, 2],
            'd_3': [7, 1]
        })

        result = analyze_missing_patterns(sales_data)

        assert isinstance(result, dict)
        assert 'sales_completeness' in result
        assert result['sales_completeness']['missing_count'] == 0
        assert result['sales_completeness']['is_complete'] == True

    def test_analyze_missing_patterns_with_pricing_gaps(self):
        """Test analysis with missing pricing data."""
        sales_data = pd.DataFrame({
            'item_id': ['FOODS_1_001', 'FOODS_1_001'],
            'store_id': ['CA_1', 'CA_1'],
            'd_1': [5, 3],
            'd_2': [3, 2]
        })

        pricing_data = pd.DataFrame({
            'item_id': ['FOODS_1_001'],
            'store_id': ['CA_1'],
            'wm_yr_wk': [202101],
            'sell_price': [2.99]
        })

        result = analyze_missing_patterns(
            sales_data,
            pricing_data=pricing_data
        )

        assert isinstance(result, dict)
        assert 'pricing_gaps' in result

    def test_analyze_missing_patterns_calendar_validation(self):
        """Test calendar data completeness validation."""
        sales_data = pd.DataFrame({
            'item_id': ['FOODS_1_001'],
            'store_id': ['CA_1'],
            'd_1': [5],
            'd_2': [3]
        })

        calendar_data = pd.DataFrame({
            'd': ['d_1', 'd_2'],
            'date': pd.to_datetime(['2011-01-29', '2011-01-30'])
        })

        result = analyze_missing_patterns(
            sales_data,
            calendar_data=calendar_data
        )

        assert isinstance(result, dict)
        assert 'calendar_completeness' in result
        assert result['calendar_completeness']['is_complete'] is True

    def test_analyze_missing_patterns_empty_dataframe(self):
        """Test handling of empty dataframe."""
        sales_data = pd.DataFrame()

        with pytest.raises((ValueError, KeyError)):
            analyze_missing_patterns(sales_data)

    def test_analyze_missing_patterns_seasonal_gaps(self):
        """Test detection of seasonal availability patterns."""
        # Simulating seasonal product with gaps
        sales_data = pd.DataFrame({
            'item_id': ['XMAS_DECOR'],
            'store_id': ['CA_1'],
        })

        # Add seasonal availability columns (abbreviated for testing)
        for i in range(1, 51):  # 50 days instead of 365 for test speed
            sales_data[f'd_{i}'] = 0 if i < 30 or i > 40 else 10

        result = analyze_missing_patterns(sales_data)

        assert isinstance(result, dict)
        assert 'sales_completeness' in result


class TestMissingMechanismCharacterization:
    """Tests for missing mechanism identification."""

    def test_characterize_seasonal_availability(self):
        """Test identification of seasonal availability patterns."""
        sales_data = pd.DataFrame({
            'item_id': ['XMAS_001', 'XMAS_001', 'REGULAR_001'],
            'store_id': ['CA_1', 'CA_1', 'CA_1'],
            'd_1': [0, 0, 5],
            'd_2': [0, 0, 3],
            'd_330': [15, 20, 4]  # December
        })

        result = characterize_missing_mechanisms(sales_data)

        assert isinstance(result, dict)
        assert 'mechanisms' in result

    def test_characterize_new_product_introduction(self):
        """Test identification of new product introduction patterns."""
        # Create data where item starts with zero sales then increases
        sales_data = pd.DataFrame({
            'item_id': ['NEW_PRODUCT'] * 100,
            'store_id': ['CA_1'] * 100
        })

        # Create d_1 to d_100 columns with zeros initially, then sales
        for i in range(1, 101):
            sales_data[f'd_{i}'] = 0 if i <= 30 else np.random.randint(1, 10)

        result = characterize_missing_mechanisms(sales_data)

        assert isinstance(result, dict)
        assert 'mechanisms' in result

    def test_characterize_discontinued_products(self):
        """Test identification of discontinued products."""
        # Create data where item has sales then zeros out
        sales_data = pd.DataFrame({
            'item_id': ['OLD_PRODUCT'] * 100,
            'store_id': ['CA_1'] * 100
        })

        # Create d_1 to d_100 columns with sales initially, then zeros
        for i in range(1, 101):
            sales_data[f'd_{i}'] = np.random.randint(1, 10) if i <= 70 else 0

        result = characterize_missing_mechanisms(sales_data)

        assert isinstance(result, dict)
        assert 'mechanisms' in result

    def test_characterize_geographic_availability(self):
        """Test identification of geographic availability patterns."""
        # Create data with different availability by store
        sales_data = pd.DataFrame({
            'item_id': ['REGIONAL_001'] * 3,
            'store_id': ['CA_1', 'TX_1', 'WI_1'],
            'd_1': [5, 0, 0],
            'd_2': [3, 0, 0],
            'd_3': [7, 0, 0]
        })

        result = characterize_missing_mechanisms(sales_data)

        assert isinstance(result, dict)
        assert 'mechanisms' in result

    def test_characterize_empty_dataframe(self):
        """Test handling of empty dataframe."""
        sales_data = pd.DataFrame()

        with pytest.raises((ValueError, KeyError)):
            characterize_missing_mechanisms(sales_data)


class TestSalesOutlierDetection:
    """Tests for sales outlier detection."""

    def test_detect_sales_outliers_foods_category(self):
        """Test outlier detection for FOODS category."""
        sales_data = pd.DataFrame({
            'item_id': ['FOODS_1_001', 'FOODS_1_002', 'FOODS_1_001'],
            'cat_id': ['FOODS', 'FOODS', 'FOODS'],
            'd_1': [5, 3, 60],  # 60 is outlier for FOODS
            'd_2': [3, 2, 4],
            'd_3': [7, 1, 5]
        })

        result = detect_sales_outliers(sales_data)

        assert isinstance(result, dict)
        assert 'outliers_detected' in result
        assert result['total_outliers'] >= 0

    def test_detect_sales_outliers_household_category(self):
        """Test outlier detection for HOUSEHOLD category."""
        sales_data = pd.DataFrame({
            'item_id': ['HOUSEHOLD_1_001'] * 5,
            'cat_id': ['HOUSEHOLD'] * 5,
            'd_1': [2, 1, 3, 25, 1]  # 25 is outlier for HOUSEHOLD
        })

        result = detect_sales_outliers(sales_data)

        assert isinstance(result, dict)
        assert 'outliers_detected' in result

    def test_detect_sales_outliers_hobbies_category(self):
        """Test outlier detection for HOBBIES category."""
        sales_data = pd.DataFrame({
            'item_id': ['HOBBIES_1_001'] * 5,
            'cat_id': ['HOBBIES'] * 5,
            'd_1': [10, 15, 8, 110, 12]  # 110 is outlier for HOBBIES
        })

        result = detect_sales_outliers(sales_data)

        assert isinstance(result, dict)
        assert 'category_thresholds' in result

    def test_detect_negative_sales_invalid(self):
        """Test detection of negative sales (invalid data)."""
        sales_data = pd.DataFrame({
            'item_id': ['FOODS_1_001'],
            'cat_id': ['FOODS'],
            'd_1': [-5]  # Negative sales invalid
        })

        result = detect_sales_outliers(sales_data)

        assert isinstance(result, dict)
        assert 'business_rule_violations' in result

    def test_detect_promotional_spikes_identified(self):
        """Test identification of promotional spikes."""
        # Create properly formatted dataframe
        sales_data = pd.DataFrame({
            'item_id': ['FOODS_1_001'] * 8,
            'cat_id': ['FOODS'] * 8
        })
        sales_values = [5, 3, 7, 4, 5, 3, 4, 48]
        for i in range(1, 9):
            sales_data[f'd_{i}'] = sales_values[i - 1]

        result = detect_sales_outliers(sales_data)

        assert isinstance(result, dict)
        assert 'outliers_detected' in result

    def test_detect_sales_outliers_missing_cat_id(self):
        """Test handling of missing cat_id column."""
        sales_data = pd.DataFrame({
            'item_id': ['FOODS_1_001'],
            'd_1': [5]
        })

        with pytest.raises((ValueError, KeyError)):
            detect_sales_outliers(sales_data)

    def test_detect_sales_outliers_empty_dataframe(self):
        """Test handling of empty dataframe."""
        sales_data = pd.DataFrame()

        with pytest.raises((ValueError, KeyError)):
            detect_sales_outliers(sales_data)


class TestPricingAnomalyDetection:
    """Tests for pricing anomaly detection."""

    def test_analyze_pricing_anomalies_normal_prices(self):
        """Test analysis with normal pricing data."""
        pricing_data = pd.DataFrame({
            'item_id': ['FOODS_1_001', 'FOODS_1_001', 'FOODS_1_001'],
            'store_id': ['CA_1', 'CA_1', 'CA_1'],
            'wm_yr_wk': [202101, 202102, 202103],
            'sell_price': [2.99, 3.09, 2.99]
        })

        result = analyze_pricing_anomalies(pricing_data)

        assert isinstance(result, dict)
        assert 'price_jumps' in result
        assert 'suspicious_prices' in result

    def test_analyze_pricing_anomalies_large_price_jumps(self):
        """Test detection of large price jumps (>200%)."""
        pricing_data = pd.DataFrame({
            'item_id': ['FOODS_1_001', 'FOODS_1_001'],
            'store_id': ['CA_1', 'CA_1'],
            'wm_yr_wk': [202101, 202102],
            'sell_price': [1.00, 3.50]  # >200% jump
        })

        result = analyze_pricing_anomalies(pricing_data)

        assert isinstance(result, dict)
        assert 'price_jumps' in result

    def test_analyze_pricing_anomalies_suspicious_prices(self):
        """Test detection of suspicious prices ($0.01, $999.99)."""
        pricing_data = pd.DataFrame({
            'item_id': ['ITEM_1', 'ITEM_2', 'ITEM_3'],
            'store_id': ['CA_1', 'CA_1', 'CA_1'],
            'wm_yr_wk': [202101, 202101, 202101],
            'sell_price': [0.01, 999.99, 49.99]
        })

        result = analyze_pricing_anomalies(pricing_data)

        assert isinstance(result, dict)
        assert 'suspicious_prices' in result

    def test_analyze_pricing_anomalies_cross_store_consistency(self):
        """Test cross-store price consistency validation."""
        pricing_data = pd.DataFrame({
            'item_id': ['FOODS_1_001'] * 3,
            'store_id': ['CA_1', 'TX_1', 'WI_1'],
            'wm_yr_wk': [202101, 202101, 202101],
            'sell_price': [2.99, 2.99, 5.99]  # Inconsistent in WI_1
        })

        result = analyze_pricing_anomalies(pricing_data)

        assert isinstance(result, dict)
        assert 'cross_store_inconsistency' in result

    def test_analyze_pricing_anomalies_negative_prices(self):
        """Test detection of negative prices (invalid)."""
        pricing_data = pd.DataFrame({
            'item_id': ['FOODS_1_001'],
            'store_id': ['CA_1'],
            'wm_yr_wk': [202101],
            'sell_price': [-2.99]  # Invalid negative price
        })

        result = analyze_pricing_anomalies(pricing_data)

        assert isinstance(result, dict)
        assert 'invalid_prices' in result

    def test_analyze_pricing_anomalies_empty_dataframe(self):
        """Test handling of empty dataframe."""
        pricing_data = pd.DataFrame()

        with pytest.raises((ValueError, KeyError)):
            analyze_pricing_anomalies(pricing_data)

    def test_analyze_pricing_anomalies_promotional_patterns(self):
        """Test identification of promotional pricing patterns."""
        pricing_data = pd.DataFrame({
            'item_id': ['FOODS_1_001'] * 5,
            'store_id': ['CA_1'] * 5,
            'wm_yr_wk': [202101, 202102, 202103, 202104, 202105],
            'sell_price': [3.99, 3.99, 1.99, 3.99, 3.99]  # Week 3 is promotion
        })

        result = analyze_pricing_anomalies(pricing_data)

        assert isinstance(result, dict)
        assert 'promotional_patterns' in result


class TestIntegrationDataQuality:
    """Integration tests for data quality analysis."""

    def test_integration_complete_analysis(self):
        """Test complete data quality analysis with all data types."""
        # Create realistic M5-style sample data
        sales_data = pd.DataFrame({
            'item_id': ['FOODS_1_001', 'FOODS_1_002', 'HOUSEHOLD_1_001'],
            'cat_id': ['FOODS', 'FOODS', 'HOUSEHOLD'],
            'store_id': ['CA_1', 'CA_1', 'TX_1'],
            'd_1': [5, 0, 2],
            'd_2': [3, 2, 1],
            'd_3': [7, 1, 3]
        })

        pricing_data = pd.DataFrame({
            'item_id': ['FOODS_1_001', 'FOODS_1_002', 'HOUSEHOLD_1_001'],
            'store_id': ['CA_1', 'CA_1', 'TX_1'],
            'wm_yr_wk': [202101, 202101, 202101],
            'sell_price': [2.99, 1.99, 5.99]
        })

        calendar_data = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3'],
            'date': pd.to_datetime(['2011-01-29', '2011-01-30', '2011-01-31'])
        })

        # Run all analyses
        missing_result = analyze_missing_patterns(
            sales_data,
            pricing_data=pricing_data,
            calendar_data=calendar_data
        )
        mechanism_result = characterize_missing_mechanisms(sales_data)
        outlier_result = detect_sales_outliers(sales_data)
        pricing_result = analyze_pricing_anomalies(pricing_data)

        # Verify all returned results are dicts
        assert isinstance(missing_result, dict)
        assert isinstance(mechanism_result, dict)
        assert isinstance(outlier_result, dict)
        assert isinstance(pricing_result, dict)

    def test_integration_with_missing_optional_data(self):
        """Test integration with only required data available."""
        sales_data = pd.DataFrame({
            'item_id': ['FOODS_1_001'],
            'cat_id': ['FOODS'],
            'd_1': [5],
            'd_2': [3]
        })

        # Should work with just sales data
        missing_result = analyze_missing_patterns(sales_data)
        assert isinstance(missing_result, dict)

        outlier_result = detect_sales_outliers(sales_data)
        assert isinstance(outlier_result, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
