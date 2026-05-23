"""
Tests for intermittent demand analysis enhancement to Step 11 (analyze_segment_behavior).

Tests the comprehensive zero-inflation and intermittent demand analysis added
to the existing segment behavior analysis function. This enhancement addresses
the intentionally skipped Step 4 by integrating intermittent demand patterns
into segment behavior analysis.

Test Coverage:
- Zero-inflation analysis (zero-sales frequency, geographic patterns)
- Intermittency classification metrics (CV, zero-run statistics, demand intensity)
- Forecast horizon viability assessment
- Enhanced segmentation with intermittent demand considerations
- Integration with existing segment behavior functionality
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eda_analysis import analyze_segment_behavior


class TestZeroInflationAnalysis:
    """Test zero-inflation analysis functionality."""

    def test_zero_inflation_metrics_structure(self):
        """Test that zero-inflation metrics are included in results."""
        # Create test data with M5-like structure and known zero patterns
        test_sales_data = pd.DataFrame({
            'item_id': ['FOODS_3_090'] * 10 + ['HOUSEHOLD_1_118'] * 10,
            'cat_id': ['FOODS'] * 10 + ['HOUSEHOLD'] * 10,
            'store_id': ['CA_1'] * 10 + ['TX_1'] * 10,
            'd_1': [0, 5, 0, 0, 3, 0, 0, 0, 2, 0] + [0, 0, 1, 0, 0, 0, 4, 0, 0, 0],
            'd_2': [0, 4, 0, 0, 2, 0, 0, 0, 1, 0] + [0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
            'd_3': [0, 6, 0, 0, 4, 0, 0, 0, 3, 0] + [0, 0, 1, 0, 0, 0, 5, 0, 0, 0],
        })

        # Create temporary CSV file for testing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            # Save test data
            test_sales_data.to_csv(sales_path, index=False)

            # Create minimal calendar data
            calendar_data = pd.DataFrame({
                'd': ['d_1', 'd_2', 'd_3'],
                'date': ['2011-01-29', '2011-01-30', '2011-01-31'],
                'weekday': ['Saturday', 'Sunday', 'Monday']
            })
            calendar_data.to_csv(calendar_path, index=False)

            # Run enhanced function
            result = analyze_segment_behavior(data_path=temp_dir)

            # Verify zero-inflation metrics are present
            assert 'zero_inflation' in result
            assert 'zero_sales_frequency' in result['zero_inflation']
            assert 'category_zero_patterns' in result['zero_inflation']
            assert 'geographic_zero_patterns' in result['zero_inflation']
            assert 'extreme_zero_inflation_items' in result['zero_inflation']

    def test_zero_sales_frequency_calculation(self):
        """Test zero-sales frequency calculation correctness."""
        # Create data with known zero patterns: 70% zeros for first item, 30% for second
        test_sales_data = pd.DataFrame({
            'item_id': ['ITEM_A', 'ITEM_B'],
            'cat_id': ['FOODS', 'FOODS'],
            'store_id': ['CA_1', 'CA_1'],
            # ITEM_A: 7 out of 10 days are zero (70%)
            'd_1': [0, 1], 'd_2': [0, 2], 'd_3': [0, 3], 'd_4': [0, 0], 'd_5': [0, 4],
            'd_6': [0, 5], 'd_7': [0, 0], 'd_8': [5, 6], 'd_9': [4, 0], 'd_10': [3, 7]
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': [f'd_{i}' for i in range(1, 11)],
                'date': pd.date_range('2011-01-29', periods=10).strftime('%Y-%m-%d'),
                'weekday': ['Monday'] * 10
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            # Verify zero-inflation percentages
            zero_inflation = result['zero_inflation']
            assert 'item_zero_percentages' in zero_inflation

            # Should identify ITEM_A as high zero-inflation (70%)
            # Should identify ITEM_B as moderate zero-inflation (30%)
            extreme_items = zero_inflation.get('extreme_zero_inflation_items', [])
            # ITEM_A should be in extreme list (>80% threshold may need adjustment for test)

    def test_geographic_zero_pattern_analysis(self):
        """Test geographic zero-inflation pattern analysis."""
        # Create data with different zero patterns by state
        test_sales_data = pd.DataFrame({
            'item_id': ['ITEM_1'] * 4,
            'cat_id': ['FOODS'] * 4,
            'store_id': ['CA_1', 'CA_2', 'TX_1', 'TX_2'],
            # CA stores have more zeros than TX stores
            'd_1': [0, 0, 3, 4], 'd_2': [0, 0, 2, 3], 'd_3': [0, 0, 4, 5],
            'd_4': [1, 1, 3, 4], 'd_5': [0, 0, 2, 3]
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': [f'd_{i}' for i in range(1, 6)],
                'date': pd.date_range('2011-01-29', periods=5).strftime('%Y-%m-%d'),
                'weekday': ['Monday'] * 5
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            # Verify geographic patterns are analyzed
            assert 'geographic_zero_patterns' in result['zero_inflation']
            geo_patterns = result['zero_inflation']['geographic_zero_patterns']

            # Should identify different zero patterns by state
            assert 'state_comparisons' in geo_patterns or 'store_comparisons' in geo_patterns


class TestIntermittencyClassification:
    """Test intermittency classification metrics."""

    def test_intermittent_demand_metrics_structure(self):
        """Test that intermittent demand metrics are included."""
        # Create basic test data
        test_sales_data = pd.DataFrame({
            'item_id': ['ITEM_1'],
            'cat_id': ['FOODS'],
            'store_id': ['CA_1'],
            'd_1': [5], 'd_2': [0], 'd_3': [10], 'd_4': [0], 'd_5': [0]
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': [f'd_{i}' for i in range(1, 6)],
                'date': pd.date_range('2011-01-29', periods=5).strftime('%Y-%m-%d'),
                'weekday': ['Monday'] * 5
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            # Verify intermittent demand analysis is present
            assert 'intermittent_demand' in result
            assert 'cv_analysis' in result['intermittent_demand']
            assert 'zero_run_statistics' in result['intermittent_demand']
            assert 'demand_intensity_metrics' in result['intermittent_demand']
            assert 'demand_pattern_classification' in result['intermittent_demand']

    def test_coefficient_of_variation_calculation(self):
        """Test coefficient of variation calculation for demand patterns."""
        # Create data with known CV patterns
        # Regular pattern: low CV
        # Intermittent pattern: high CV
        test_sales_data = pd.DataFrame({
            'item_id': ['REGULAR_ITEM', 'INTERMITTENT_ITEM'],
            'cat_id': ['FOODS', 'HOUSEHOLD'],
            'store_id': ['CA_1', 'CA_1'],
            # Regular: consistent sales (low CV)
            'd_1': [5, 0], 'd_2': [4, 0], 'd_3': [6, 15], 'd_4': [5, 0], 'd_5': [5, 0]
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': [f'd_{i}' for i in range(1, 6)],
                'date': pd.date_range('2011-01-29', periods=5).strftime('%Y-%m-%d'),
                'weekday': ['Monday'] * 5
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            # Verify CV analysis exists and makes sense
            cv_analysis = result['intermittent_demand']['cv_analysis']
            assert 'item_cv_scores' in cv_analysis

            # Regular item should have lower CV than intermittent item
            # This will be verified once implementation is complete

    def test_demand_pattern_classification(self):
        """Test demand pattern classification logic."""
        # Create data representing different demand patterns
        test_sales_data = pd.DataFrame({
            'item_id': ['REGULAR', 'INTERMITTENT', 'SPARSE', 'LUMPY'],
            'cat_id': ['FOODS'] * 4,
            'store_id': ['CA_1'] * 4,
            # Regular: CV < 1.0, zero% < 20%
            'd_1': [5, 0, 0, 0], 'd_2': [4, 0, 0, 0], 'd_3': [6, 8, 0, 0],
            'd_4': [5, 0, 0, 0], 'd_5': [5, 0, 0, 25], 'd_6': [4, 0, 5, 0],
            'd_7': [6, 0, 0, 0], 'd_8': [5, 7, 0, 0], 'd_9': [5, 0, 0, 0],
            'd_10': [4, 0, 0, 0]
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': [f'd_{i}' for i in range(1, 11)],
                'date': pd.date_range('2011-01-29', periods=10).strftime('%Y-%m-%d'),
                'weekday': ['Monday'] * 10
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            # Verify classification exists
            classification = result['intermittent_demand']['demand_pattern_classification']
            assert 'classification_results' in classification
            assert 'pattern_distribution' in classification

            # Should classify patterns as Regular, Intermittent, Sparse, Lumpy
            # Exact assertions depend on implementation thresholds

    def test_zero_run_statistics(self):
        """Test zero-run statistics calculation."""
        # Create data with known zero-run patterns
        test_sales_data = pd.DataFrame({
            'item_id': ['TEST_ITEM'],
            'cat_id': ['FOODS'],
            'store_id': ['CA_1'],
            # Pattern: 0,0,0,5,0,0,3,0,0,0,0 (runs of 3,2,4 zeros)
            'd_1': [0], 'd_2': [0], 'd_3': [0], 'd_4': [5], 'd_5': [0],
            'd_6': [0], 'd_7': [3], 'd_8': [0], 'd_9': [0], 'd_10': [0], 'd_11': [0]
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': [f'd_{i}' for i in range(1, 12)],
                'date': pd.date_range('2011-01-29', periods=11).strftime('%Y-%m-%d'),
                'weekday': ['Monday'] * 11
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            # Verify zero-run statistics
            zero_runs = result['intermittent_demand']['zero_run_statistics']
            assert 'average_zero_run_length' in zero_runs
            assert 'max_zero_run_length' in zero_runs
            assert 'zero_run_distribution' in zero_runs


class TestForecastHorizonViability:
    """Test forecast horizon viability assessment."""

    def test_forecast_viability_metrics_structure(self):
        """Test that forecast viability metrics are included."""
        test_sales_data = pd.DataFrame({
            'item_id': ['ITEM_1'],
            'cat_id': ['FOODS'],
            'store_id': ['CA_1'],
            'd_1': [5], 'd_2': [0], 'd_3': [3]
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': ['d_1', 'd_2', 'd_3'],
                'date': ['2011-01-29', '2011-01-30', '2011-01-31'],
                'weekday': ['Monday'] * 3
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            # Verify forecast viability assessment is present
            assert 'forecast_viability' in result
            assert 'sufficient_data_assessment' in result['forecast_viability']
            assert 'minimum_viable_data_requirements' in result['forecast_viability']
            assert 'forecast_methodology_recommendations' in result['forecast_viability']

    def test_sufficient_nonzero_observations_assessment(self):
        """Test assessment of sufficient non-zero observations for forecasting."""
        # Create data with varying amounts of non-zero observations
        test_sales_data = pd.DataFrame({
            'item_id': ['SUFFICIENT_ITEM', 'INSUFFICIENT_ITEM'],
            'cat_id': ['FOODS', 'HOUSEHOLD'],
            'store_id': ['CA_1', 'CA_1'],
        })

        # Add many days of data for sufficient item (20 non-zero days)
        for i in range(1, 29):  # 28 days
            test_sales_data[f'd_{i}'] = [5 if i <= 20 else 0, 1 if i <= 5 else 0]

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': [f'd_{i}' for i in range(1, 29)],
                'date': pd.date_range('2011-01-29', periods=28).strftime('%Y-%m-%d'),
                'weekday': ['Monday'] * 28
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            # Should assess data sufficiency for 28-day forecasts
            viability = result['forecast_viability']['sufficient_data_assessment']
            assert 'items_with_sufficient_data' in viability
            assert 'items_with_insufficient_data' in viability


class TestEnhancedSegmentation:
    """Test enhanced segmentation with intermittent demand considerations."""

    def test_enhanced_segmentation_structure(self):
        """Test that enhanced segmentation is included."""
        test_sales_data = pd.DataFrame({
            'item_id': ['ITEM_1', 'ITEM_2'],
            'cat_id': ['FOODS', 'HOUSEHOLD'],
            'store_id': ['CA_1', 'CA_1'],
            'd_1': [5, 0], 'd_2': [0, 8], 'd_3': [3, 0]
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': ['d_1', 'd_2', 'd_3'],
                'date': ['2011-01-29', '2011-01-30', '2011-01-31'],
                'weekday': ['Monday'] * 3
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            # Verify enhanced segmentation is present
            assert 'enhanced_segmentation' in result
            assert 'comprehensive_demand_profiles' in result['enhanced_segmentation']
            assert 'business_actionable_segments' in result['enhanced_segmentation']
            assert 'intermittent_demand_considerations' in result['enhanced_segmentation']

    def test_comprehensive_demand_behavior_profiles(self):
        """Test comprehensive demand behavior profile generation."""
        # Create varied demand patterns for profiling
        test_sales_data = pd.DataFrame({
            'item_id': ['REGULAR', 'INTERMITTENT', 'SEASONAL'],
            'cat_id': ['FOODS', 'HOUSEHOLD', 'HOBBIES'],
            'store_id': ['CA_1'] * 3,
            'd_1': [5, 0, 0], 'd_2': [4, 0, 0], 'd_3': [6, 10, 8],
            'd_4': [5, 0, 0], 'd_5': [4, 0, 0], 'd_6': [6, 0, 7],
            'd_7': [5, 12, 0]
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': [f'd_{i}' for i in range(1, 8)],
                'date': pd.date_range('2011-01-29', periods=7).strftime('%Y-%m-%d'),
                'weekday': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            # Verify comprehensive profiles combine multiple dimensions
            profiles = result['enhanced_segmentation']['comprehensive_demand_profiles']
            assert 'segment_characteristics' in profiles

            # Should combine traditional segmentation with intermittency metrics


class TestExistingFunctionalityPreservation:
    """Test that existing Step 11 functionality is preserved."""

    def test_existing_results_structure_maintained(self):
        """Test that all existing result keys are still present."""
        test_sales_data = pd.DataFrame({
            'item_id': ['ITEM_1', 'ITEM_2'],
            'cat_id': ['FOODS', 'HOUSEHOLD'],
            'store_id': ['CA_1', 'CA_1'],
            'd_1': [5, 3], 'd_2': [4, 2], 'd_3': [6, 4]
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': ['d_1', 'd_2', 'd_3'],
                'date': ['2011-01-29', '2011-01-30', '2011-01-31'],
                'weekday': ['Monday'] * 3
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            # Verify all original keys are present
            assert 'category_behavior' in result
            assert 'department_segments' in result
            assert 'performance_metrics' in result
            assert 'seasonality_patterns' in result
            assert 'lifecycle_stages' in result
            assert 'visualizations' in result
            assert 'summary' in result

            # Verify new keys are also present
            assert 'intermittent_demand' in result
            assert 'zero_inflation' in result
            assert 'forecast_viability' in result
            assert 'enhanced_segmentation' in result

    def test_original_summary_structure_preserved(self):
        """Test that original summary structure is preserved."""
        test_sales_data = pd.DataFrame({
            'item_id': ['ITEM_1'],
            'cat_id': ['FOODS'],
            'store_id': ['CA_1'],
            'd_1': [5], 'd_2': [4], 'd_3': [6]
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': ['d_1', 'd_2', 'd_3'],
                'date': ['2011-01-29', '2011-01-30', '2011-01-31'],
                'weekday': ['Monday'] * 3
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            summary = result['summary']
            # Original summary keys should be preserved
            assert 'total_categories' in summary
            assert 'departments_analyzed' in summary
            assert 'top_performers' in summary
            assert 'seasonal_segments' in summary
            assert 'lifecycle_distribution' in summary
            assert 'step_status' in summary


class TestIntegrationAndEdgeCases:
    """Test integration between new and existing functionality and edge cases."""

    def test_empty_data_handling(self):
        """Test handling of empty or minimal data."""
        test_sales_data = pd.DataFrame({
            'item_id': [],
            'cat_id': [],
            'store_id': []
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': [],
                'date': [],
                'weekday': []
            })
            calendar_data.to_csv(calendar_path, index=False)

            # Should handle empty data gracefully
            result = analyze_segment_behavior(data_path=temp_dir)
            assert result is not None
            assert 'intermittent_demand' in result
            assert 'zero_inflation' in result

    def test_all_zero_sales_data(self):
        """Test handling of data with all zero sales."""
        test_sales_data = pd.DataFrame({
            'item_id': ['ZERO_ITEM'] * 2,
            'cat_id': ['FOODS'] * 2,
            'store_id': ['CA_1', 'CA_2'],
            'd_1': [0, 0], 'd_2': [0, 0], 'd_3': [0, 0], 'd_4': [0, 0], 'd_5': [0, 0]
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': [f'd_{i}' for i in range(1, 6)],
                'date': pd.date_range('2011-01-29', periods=5).strftime('%Y-%m-%d'),
                'weekday': ['Monday'] * 5
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            # Should identify extreme zero-inflation correctly
            zero_inflation = result['zero_inflation']
            assert 'extreme_zero_inflation_items' in zero_inflation

            # All items should be classified as having 100% zero sales

    def test_single_item_analysis(self):
        """Test analysis with only one item."""
        test_sales_data = pd.DataFrame({
            'item_id': ['SINGLE_ITEM'],
            'cat_id': ['FOODS'],
            'store_id': ['CA_1'],
            'd_1': [5], 'd_2': [0], 'd_3': [3], 'd_4': [0], 'd_5': [2]
        })

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = os.path.join(temp_dir, "sales_train_validation.csv")
            calendar_path = os.path.join(temp_dir, "calendar.csv")

            test_sales_data.to_csv(sales_path, index=False)

            calendar_data = pd.DataFrame({
                'd': [f'd_{i}' for i in range(1, 6)],
                'date': pd.date_range('2011-01-29', periods=5).strftime('%Y-%m-%d'),
                'weekday': ['Monday'] * 5
            })
            calendar_data.to_csv(calendar_path, index=False)

            result = analyze_segment_behavior(data_path=temp_dir)

            # Should handle single item analysis without errors
            assert result['intermittent_demand'] is not None
            assert result['zero_inflation'] is not None
            assert result['forecast_viability'] is not None


if __name__ == "__main__":
    pytest.main([__file__])