import pytest
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, Any

# Add notebooks/eda to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.feature_profiling import (
    analyze_categorical_distributions,
    analyze_geographic_patterns,
    analyze_temporal_correlations,
    analyze_price_behavior,
    calculate_feature_importance_rankings
)


class TestAnalyzeCategoricalDistributions:

    def test_analyze_categorical_distributions_basic(self):
        """Test categorical analysis with basic M5-style data."""
        # Create sales data with categorical hierarchy
        sales_df = pd.DataFrame({
            'item_id': ['FOODS_1_001', 'FOODS_1_002', 'HOUSEHOLD_1_001', 'HOBBIES_1_001'],
            'cat_id': ['FOODS', 'FOODS', 'HOUSEHOLD', 'HOBBIES'],
            'dept_id': ['FOODS_1', 'FOODS_1', 'HOUSEHOLD_1', 'HOBBIES_1'],
            'state_id': ['CA', 'TX', 'WI', 'CA'],
            'store_id': ['CA_1', 'TX_1', 'WI_1', 'CA_2'],
            'd_1': [10, 15, 5, 20],
            'd_2': [12, 18, 3, 25],
            'd_3': [8, 20, 7, 22]
        })

        calendar_df = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3'],
            'date': ['2011-01-29', '2011-01-30', '2011-01-31']
        })

        result = analyze_categorical_distributions(sales_df, calendar_df)

        # Check return structure
        assert isinstance(result, dict)
        assert 'cat_id' in result
        assert 'dept_id' in result
        assert 'state_id' in result
        assert 'store_id' in result

        # Check category statistics
        cat_stats = result['cat_id']
        assert 'mean' in cat_stats
        assert 'median' in cat_stats
        assert 'std' in cat_stats
        assert 'skewness' in cat_stats
        assert 'ranking' in cat_stats

        # Check HOBBIES category has highest mean (20+25+22)/3=22.33 vs others
        assert cat_stats['ranking']['HOBBIES'] == 1

    def test_analyze_categorical_distributions_empty_data(self):
        """Test categorical analysis with empty dataframes."""
        sales_df = pd.DataFrame()
        calendar_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Empty sales data"):
            analyze_categorical_distributions(sales_df, calendar_df)

    def test_analyze_categorical_distributions_missing_columns(self):
        """Test categorical analysis with missing required columns."""
        sales_df = pd.DataFrame({
            'item_id': ['FOODS_1_001'],
            'cat_id': ['FOODS']
            # Missing dept_id, state_id, store_id, sales columns
        })
        calendar_df = pd.DataFrame({'d': ['d_1']})

        with pytest.raises(ValueError, match="Missing required columns"):
            analyze_categorical_distributions(sales_df, calendar_df)


class TestAnalyzeGeographicPatterns:

    def test_analyze_geographic_patterns_basic(self):
        """Test geographic analysis with state/store variations."""
        sales_df = pd.DataFrame({
            'item_id': ['ITEM_001', 'ITEM_001', 'ITEM_002', 'ITEM_002'],
            'state_id': ['CA', 'TX', 'CA', 'TX'],
            'store_id': ['CA_1', 'TX_1', 'CA_1', 'TX_1'],
            'd_1': [100, 50, 80, 40],
            'd_2': [110, 55, 85, 45]
        })

        sell_prices_df = pd.DataFrame({
            'store_id': ['CA_1', 'TX_1', 'CA_1', 'TX_1'],
            'item_id': ['ITEM_001', 'ITEM_001', 'ITEM_002', 'ITEM_002'],
            'wm_yr_wk': [11101, 11101, 11101, 11101],
            'sell_price': [1.58, 1.26, 2.50, 2.00]
        })

        result = analyze_geographic_patterns(sales_df, sell_prices_df)

        # Check return structure
        assert isinstance(result, dict)
        assert 'state_performance' in result
        assert 'store_performance' in result
        assert 'geographic_cv' in result

        # Check state performance metrics
        state_perf = result['state_performance']
        assert 'CA' in state_perf
        assert 'TX' in state_perf
        assert 'mean_sales' in state_perf['CA']
        assert 'mean_price' in state_perf['CA']

        # CA should have higher mean sales than TX
        assert state_perf['CA']['mean_sales'] > state_perf['TX']['mean_sales']

    def test_analyze_geographic_patterns_single_state(self):
        """Test geographic analysis with single state (no variation)."""
        sales_df = pd.DataFrame({
            'item_id': ['ITEM_001'],
            'state_id': ['CA'],
            'store_id': ['CA_1'],
            'd_1': [100]
        })

        sell_prices_df = pd.DataFrame({
            'store_id': ['CA_1'],
            'item_id': ['ITEM_001'],
            'wm_yr_wk': [11101],
            'sell_price': [1.58]
        })

        result = analyze_geographic_patterns(sales_df, sell_prices_df)

        # Should handle single state gracefully
        assert result['geographic_cv']['state_cv'] == 0.0


class TestAnalyzeTemporalCorrelations:

    def test_analyze_temporal_correlations_basic(self):
        """Test temporal correlation analysis with calendar features."""
        sales_df = pd.DataFrame({
            'item_id': ['ITEM_001', 'ITEM_002', 'ITEM_003'],
            'd_1': [10, 12, 15],  # Day 1 sales
            'd_2': [11, 13, 16],  # Day 2 sales
            'd_3': [12, 14, 17]   # Day 3 sales
        })

        calendar_df = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3'],
            'weekday': [1, 2, 3],  # Monday=1, Tuesday=2, Wednesday=3
            'month': [1, 1, 2],   # Mix of months to avoid NaN
            'year': [2011, 2011, 2011],
            'event_name_1': [None, 'SportingEvent', None],
            'event_type_1': [None, 'Sporting', None],
            'snap_CA': [1, 0, 1]
        })

        result = analyze_temporal_correlations(sales_df, calendar_df)

        # Check return structure
        assert isinstance(result, dict)
        assert 'weekday_correlation' in result
        assert 'month_correlation' in result
        assert 'event_correlation' in result
        assert 'snap_correlation' in result
        assert 'significance_tests' in result

        # Check correlation values are between -1 and 1
        assert -1 <= result['weekday_correlation'] <= 1
        assert -1 <= result['month_correlation'] <= 1

    def test_analyze_temporal_correlations_no_variation(self):
        """Test temporal correlations with constant sales (no correlation)."""
        sales_df = pd.DataFrame({
            'item_id': ['ITEM_001', 'ITEM_002', 'ITEM_003'],
            'd_1': [10, 10, 10],  # Constant sales per day
            'd_2': [10, 10, 10],
            'd_3': [10, 10, 10]
        })

        calendar_df = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3'],
            'weekday': [1, 2, 3],
            'month': [1, 1, 2]
        })

        result = analyze_temporal_correlations(sales_df, calendar_df)

        # Constant sales should result in NaN correlations
        assert np.isnan(result['weekday_correlation'])


class TestAnalyzePriceBehavior:

    def test_analyze_price_behavior_basic(self):
        """Test price behavior analysis with normal price variations."""
        sell_prices_df = pd.DataFrame({
            'store_id': ['CA_1', 'CA_1', 'CA_1', 'CA_1'],
            'item_id': ['ITEM_001', 'ITEM_001', 'ITEM_001', 'ITEM_001'],
            'wm_yr_wk': [11101, 11102, 11103, 11104],
            'sell_price': [1.58, 1.62, 1.65, 1.68]  # Gradual increase
        })

        calendar_df = pd.DataFrame({
            'wm_yr_wk': [11101, 11102, 11103, 11104],
            'd': ['d_1', 'd_8', 'd_15', 'd_22'],
            'month': [1, 1, 2, 2]
        })

        result = analyze_price_behavior(sell_prices_df, calendar_df)

        # Check return structure
        assert isinstance(result, dict)
        assert 'price_stability' in result
        assert 'seasonal_patterns' in result
        assert 'price_anomalies' in result
        assert 'coefficient_of_variation' in result

        # Check coefficient of variation calculation
        cv = result['coefficient_of_variation']
        assert isinstance(cv, dict)
        assert 'ITEM_001' in cv

        # Should not detect anomalies in gradual price increase
        assert len(result['price_anomalies']) == 0

    def test_analyze_price_behavior_detect_jumps(self):
        """Test price behavior analysis detects >200% price jumps."""
        sell_prices_df = pd.DataFrame({
            'store_id': ['CA_1', 'CA_1', 'CA_1'],
            'item_id': ['ITEM_001', 'ITEM_001', 'ITEM_001'],
            'wm_yr_wk': [11101, 11102, 11103],
            'sell_price': [1.00, 1.05, 3.50]  # 233% jump in last price
        })

        calendar_df = pd.DataFrame({
            'wm_yr_wk': [11101, 11102, 11103],
            'd': ['d_1', 'd_8', 'd_15'],
            'month': [1, 1, 2]
        })

        result = analyze_price_behavior(sell_prices_df, calendar_df)

        # Should detect the price jump anomaly
        assert len(result['price_anomalies']) > 0
        assert result['price_anomalies'][0]['item_id'] == 'ITEM_001'
        assert result['price_anomalies'][0]['jump_percentage'] > 200


class TestCalculateFeatureImportanceRankings:

    def test_calculate_feature_importance_rankings_basic(self):
        """Test feature importance ranking with correlation dictionary."""
        correlations_dict = {
            'weekday_correlation': 0.75,
            'month_correlation': -0.45,
            'event_correlation': 0.60,
            'snap_correlation': 0.30,
            'price_correlation': -0.80
        }

        result = calculate_feature_importance_rankings(correlations_dict)

        # Check return structure
        assert isinstance(result, dict)
        assert 'rankings' in result
        assert 'absolute_rankings' in result
        assert 'significance_flags' in result

        # Check rankings are sorted by absolute correlation strength
        rankings = result['absolute_rankings']
        assert rankings[0]['feature'] == 'price_correlation'  # |0.80| = highest
        assert rankings[1]['feature'] == 'weekday_correlation'  # |0.75| = second
        assert rankings[-1]['feature'] == 'snap_correlation'  # |0.30| = lowest

    def test_calculate_feature_importance_rankings_empty_dict(self):
        """Test feature importance ranking with empty correlation dictionary."""
        correlations_dict = {}

        result = calculate_feature_importance_rankings(correlations_dict)

        # Should handle empty input gracefully
        assert result['rankings'] == []
        assert result['absolute_rankings'] == []

    def test_calculate_feature_importance_rankings_nan_values(self):
        """Test feature importance ranking handles NaN correlation values."""
        correlations_dict = {
            'weekday_correlation': 0.75,
            'invalid_correlation': np.nan,
            'month_correlation': -0.45
        }

        result = calculate_feature_importance_rankings(correlations_dict)

        # Should exclude NaN values from rankings
        assert len(result['absolute_rankings']) == 2  # Only valid correlations
        assert all('invalid_correlation' not in item['feature']
                  for item in result['absolute_rankings'])