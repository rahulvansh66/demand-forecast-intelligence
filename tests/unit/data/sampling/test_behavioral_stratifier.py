"""
Tests for behavioral stratification engine.

This module tests the BehavioralStratifier class that creates multi-dimensional
behavioral strata to prevent sampling bias in M5 dataset sampling. The stratification
prevents bias toward high-volume, stable items by ensuring proper representation of
intermittent/sparse demand patterns critical for robust forecasting model validation.

Tests follow TDD methodology - each test is written first to verify the behavioral
stratification logic works correctly before implementation.
"""

import pytest
import pandas as pd
from unittest.mock import Mock

from demand_forecast_intelligence.data.sampling.behavioral_stratifier import BehavioralStratifier


class TestBehavioralStratifier:
    """Test suite for BehavioralStratifier class."""

    @pytest.fixture
    def sample_sales_data(self):
        """
        Create sample sales data for testing.

        Returns mock M5 sales data with various demand patterns:
        - High volume regular items
        - Low volume intermittent items
        - Items with different lifecycle patterns
        - Multiple departments to test within-department volume bucketing

        Uses training period columns only (d_1 to d_1913) to prevent future leakage.
        """
        # Create sample data with different behavioral patterns
        data = {
            'id': ['FOODS_3_090_CA_1', 'FOODS_3_091_CA_1', 'HOUSEHOLD_1_001_CA_1', 'HOBBIES_1_001_CA_1'],
            'item_id': ['FOODS_3_090', 'FOODS_3_091', 'HOUSEHOLD_1_001', 'HOBBIES_1_001'],
            'dept_id': ['FOODS_3', 'FOODS_3', 'HOUSEHOLD_1', 'HOBBIES_1'],
            'cat_id': ['FOODS', 'FOODS', 'HOUSEHOLD', 'HOBBIES'],
            'store_id': ['CA_1', 'CA_1', 'CA_1', 'CA_1'],
            'state_id': ['CA', 'CA', 'CA', 'CA']
        }

        # Add training period sales columns (d_1 to d_1913)
        # Create different patterns:
        # FOODS_3_090: High volume, regular (stable pattern)
        # FOODS_3_091: Low volume, intermittent (sparse pattern)
        # HOUSEHOLD_1_001: Medium volume, seasonal (lifecycle pattern)
        # HOBBIES_1_001: Very low volume, discontinued (lifecycle pattern)

        # High volume regular item - 5-15 units daily, very few zeros
        high_vol_pattern = [10, 12, 8, 15, 9, 11, 13] * 273 + [10, 12, 8]  # ~1913 days

        # Low volume intermittent - mostly zeros with occasional sales
        low_vol_pattern = [0, 0, 0, 1, 0, 0, 0] * 273 + [0, 0, 1]  # ~1913 days, ~20% nonzero

        # Medium volume seasonal - starts low, builds up, then stable
        medium_vol_pattern = ([0, 1] * 100 + [2, 3, 4] * 200 + [5, 6, 7, 8] * 400 +
                            [7, 8, 9] * 213)[:1913]  # Lifecycle progression

        # Very low discontinued - starts with some sales, then drops to zero
        discontinued_pattern = ([1, 2, 1] * 200 + [0] * 1313)[:1913]  # Early activity, then discontinued

        # Ensure we have exactly 1913 training period columns
        for i in range(1, 1914):
            col_name = f'd_{i}'
            if i <= len(high_vol_pattern):
                data[col_name] = [
                    high_vol_pattern[i-1],      # FOODS_3_090
                    low_vol_pattern[i-1],       # FOODS_3_091
                    medium_vol_pattern[i-1],    # HOUSEHOLD_1_001
                    discontinued_pattern[i-1]   # HOBBIES_1_001
                ]
            else:
                # Fill remaining with zeros if needed
                data[col_name] = [0, 0, 0, 0]

        return pd.DataFrame(data)

    @pytest.fixture
    def stratifier_config(self):
        """
        Configuration for BehavioralStratifier.

        Returns config dict with stratification parameters that prevent sampling bias:
        - Volume buckets within departments to prevent category bias
        - Intermittency thresholds based on nonzero day ratio
        - Lifecycle analysis windows for product maturity assessment
        """
        return {
            'volume_buckets': ['low', 'medium', 'high'],
            'volume_percentiles': [33, 67],  # Tertiles within each department
            'intermittency_thresholds': {
                'sparse': 0.2,      # ≤20% nonzero days - challenging forecasting cases
                'intermittent': 0.6, # 20-60% nonzero days - moderate challenge
                'regular': 1.0      # >60% nonzero days - stable patterns
            },
            'lifecycle_windows': {
                'early_period': 365,     # First year for early detection
                'late_period': 365,      # Last year for decline detection
                'min_active_days': 30    # Minimum activity for lifecycle classification
            }
        }

    def test_calculate_item_metrics_extracts_training_period_only(self, sample_sales_data, stratifier_config):
        """
        Test that calculate_item_metrics() uses only training period data (d_1 to d_1913).

        This test verifies the critical requirement that prevents future leakage:
        - Only training period columns (d_1 to d_1913) are used for metric calculation
        - No validation/test period data leaks into sampling decisions
        - Metrics reflect only information available during model training

        This is essential for unbiased sampling - using future data would make
        POC validation overly optimistic and not representative of production performance.
        """
        stratifier = BehavioralStratifier(stratifier_config)

        # Calculate metrics from sample data
        metrics = stratifier.calculate_item_metrics(sample_sales_data)

        # Verify we get one row per item with expected metrics
        assert len(metrics) == 4, "Should have metrics for 4 items"

        # Verify required columns exist
        expected_columns = [
            'id', 'item_id', 'dept_id', 'total_sales', 'avg_daily_sales',
            'nonzero_days', 'total_days', 'nonzero_day_ratio',
            'early_period_sales', 'late_period_sales'
        ]
        for col in expected_columns:
            assert col in metrics.columns, f"Missing column: {col}"

        # Verify metrics calculated from training period only (1913 days)
        assert metrics['total_days'].iloc[0] == 1913, "Should use exactly 1913 training days"

        # Verify behavioral patterns are detected correctly
        # High volume item should have high avg_daily_sales and high nonzero_day_ratio
        foods_090_metrics = metrics[metrics['item_id'] == 'FOODS_3_090'].iloc[0]
        assert foods_090_metrics['avg_daily_sales'] > 5, "High volume item should have high daily average"
        assert foods_090_metrics['nonzero_day_ratio'] > 0.8, "Regular item should have high nonzero ratio"

        # Low volume intermittent item should have low nonzero_day_ratio
        foods_091_metrics = metrics[metrics['item_id'] == 'FOODS_3_091'].iloc[0]
        assert foods_091_metrics['nonzero_day_ratio'] <= 0.2, "Intermittent item should have low nonzero ratio"

    def test_create_behavioral_strata_prevents_sampling_bias(self, sample_sales_data, stratifier_config):
        """
        Test that create_behavioral_strata() creates multi-dimensional strata preventing bias.

        This test verifies the core anti-bias mechanism:
        - Volume buckets calculated WITHIN each department (prevents category bias)
        - Intermittency classification captures sparse/challenging patterns
        - Lifecycle analysis identifies product maturity stages
        - Composite stratum keys enable proper allocation

        This stratification ensures the sample includes the intermittent/sparse patterns
        that cause production model failures, rather than biasing toward "clean" high-volume items.
        """
        stratifier = BehavioralStratifier(stratifier_config)

        # Calculate metrics first
        metrics = stratifier.calculate_item_metrics(sample_sales_data)

        # Create behavioral strata
        strata = stratifier.create_behavioral_strata(metrics)

        # Verify strata structure
        assert 'stratum_key' in strata.columns, "Should have composite stratum key"
        assert 'volume_bucket' in strata.columns, "Should have volume buckets"
        assert 'intermittency_class' in strata.columns, "Should have intermittency classification"
        assert 'lifecycle_stage' in strata.columns, "Should have lifecycle stages"

        # Verify volume bucketing is within departments
        # Items in same department should span different volume buckets
        foods_items = strata[strata['dept_id'] == 'FOODS_3']
        volume_buckets_in_foods = set(foods_items['volume_bucket'].values)
        assert len(volume_buckets_in_foods) > 1, "Should have multiple volume buckets within FOODS_3 department"

        # Verify intermittency classification captures sparse patterns
        intermittency_classes = set(strata['intermittency_class'].values)
        assert 'sparse' in intermittency_classes, "Should identify sparse items (≤20% nonzero days)"
        assert 'regular' in intermittency_classes, "Should identify regular items (>60% nonzero days)"

        # Verify composite stratum keys are unique and meaningful
        stratum_keys = strata['stratum_key'].values
        assert len(set(stratum_keys)) > 1, "Should have multiple unique strata"

        # Each stratum key should combine dept + volume + intermittency + lifecycle
        for key in stratum_keys:
            parts = key.split('|')
            assert len(parts) == 4, f"Stratum key should have 4 parts: {key}"

    def test_volume_bucketing_within_departments_prevents_category_bias(self, sample_sales_data, stratifier_config):
        """
        Test that volume bucketing within departments prevents category bias.

        This test verifies a critical anti-bias mechanism:
        - Volume percentiles calculated separately for each department
        - Prevents bias toward high-volume categories (e.g., FOODS vs HOBBIES)
        - Ensures each department contributes items across volume spectrum

        Without this, sampling would bias toward naturally high-volume categories,
        making validation less representative of the full product mix.
        """
        stratifier = BehavioralStratifier(stratifier_config)

        # Calculate metrics and create strata
        metrics = stratifier.calculate_item_metrics(sample_sales_data)
        strata = stratifier.create_behavioral_strata(metrics)

        # Group by department to verify within-department bucketing
        dept_volume_analysis = strata.groupby('dept_id').agg({
            'volume_bucket': lambda x: set(x),
            'avg_daily_sales': ['min', 'max']
        }).reset_index()

        # Verify each department has volume distribution
        for _, dept_row in dept_volume_analysis.iterrows():
            dept_id = dept_row['dept_id']
            volume_buckets = dept_row[('volume_bucket', '<lambda>')]

            # Each department should contribute to volume distribution
            # (Even single-item departments get assigned to a bucket)
            assert len(volume_buckets) >= 1, f"Department {dept_id} should have volume buckets"

        # Verify volume ranges make sense within departments
        # FOODS_3 department items should have different volume buckets despite different absolute volumes
        foods_strata = strata[strata['dept_id'] == 'FOODS_3']
        if len(foods_strata) > 1:
            foods_volume_buckets = set(foods_strata['volume_bucket'].values)
            assert len(foods_volume_buckets) > 1, "FOODS_3 items should span multiple volume buckets"

    def test_intermittency_classification_captures_sparse_patterns(self, sample_sales_data, stratifier_config):
        """
        Test that intermittency classification captures challenging sparse patterns.

        This test verifies the system identifies sparse/intermittent demand patterns:
        - Sparse items (≤20% nonzero days) - the most challenging forecasting cases
        - Intermittent items (20-60% nonzero days) - moderate forecasting challenges
        - Regular items (>60% nonzero days) - stable, predictable patterns

        This classification ensures the sample includes the problematic sparse patterns
        that cause production model failures, rather than only "clean" regular patterns.
        """
        stratifier = BehavioralStratifier(stratifier_config)

        # Calculate metrics and create strata
        metrics = stratifier.calculate_item_metrics(sample_sales_data)
        strata = stratifier.create_behavioral_strata(metrics)

        # Verify intermittency classification logic
        for _, row in strata.iterrows():
            nonzero_ratio = row['nonzero_day_ratio']
            intermittency_class = row['intermittency_class']

            if nonzero_ratio <= 0.2:
                assert intermittency_class == 'sparse', f"Item with {nonzero_ratio:.3f} ratio should be sparse"
            elif nonzero_ratio <= 0.6:
                assert intermittency_class == 'intermittent', f"Item with {nonzero_ratio:.3f} ratio should be intermittent"
            else:
                assert intermittency_class == 'regular', f"Item with {nonzero_ratio:.3f} ratio should be regular"

        # Verify we capture the challenging patterns in our sample data
        intermittency_counts = strata['intermittency_class'].value_counts()

        # Should have sparse items (the low volume intermittent item)
        assert 'sparse' in intermittency_counts, "Should identify sparse items in sample data"

        # Should have regular items (the high volume regular item)
        assert 'regular' in intermittency_counts, "Should identify regular items in sample data"

        # Verify sparse items have the expected low nonzero ratios
        sparse_items = strata[strata['intermittency_class'] == 'sparse']
        for _, item in sparse_items.iterrows():
            assert item['nonzero_day_ratio'] <= 0.2, "Sparse items should have ≤20% nonzero days"