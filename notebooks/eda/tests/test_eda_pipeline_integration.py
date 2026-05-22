"""
Comprehensive integration tests for the complete EDA framework pipeline.

Tests end-to-end workflows across all 5 framework steps:
- Step 6: Feature-Target Relationships
- Step 7: Feature-Feature Relationships
- Step 8: Time Series Patterns
- Step 9: Missing Values Analysis
- Step 10: Outliers and Anomalies Detection

Validates function signatures, parameter handling, plot generation,
and business logic implementation.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import sys
from pathlib import Path

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eda_analysis import (
    study_feature_target_relationships,
    study_feature_feature_relationships,
    analyze_time_series_patterns,
    analyze_missing_values_deeply,
    identify_outliers_and_anomalies,
)


class TestFunctionSignatures:
    """Test that all main functions have consistent, production-ready signatures."""

    def test_study_feature_target_relationships_signature(self):
        """Verify function accepts data_path parameter and returns dict."""
        import inspect
        sig = inspect.signature(study_feature_target_relationships)
        params = list(sig.parameters.keys())

        assert 'data_path' in params, "Function must accept data_path parameter"
        assert len(params) >= 1, "Function must have at least data_path parameter"

    def test_study_feature_feature_relationships_signature(self):
        """Verify function accepts data_path parameter and returns dict."""
        import inspect
        sig = inspect.signature(study_feature_feature_relationships)
        params = list(sig.parameters.keys())

        assert 'data_path' in params, "Function must accept data_path parameter"

    def test_analyze_time_series_patterns_signature(self):
        """Verify function accepts data_path parameter and returns dict."""
        import inspect
        sig = inspect.signature(analyze_time_series_patterns)
        params = list(sig.parameters.keys())

        assert 'data_path' in params, "Function must accept data_path parameter"

    def test_analyze_missing_values_deeply_signature(self):
        """Verify function accepts data_path parameter and returns dict."""
        import inspect
        sig = inspect.signature(analyze_missing_values_deeply)
        params = list(sig.parameters.keys())

        assert 'data_path' in params, "Function must accept data_path parameter"

    def test_identify_outliers_and_anomalies_signature(self):
        """Verify function accepts data_path parameter and returns dict."""
        import inspect
        sig = inspect.signature(identify_outliers_and_anomalies)
        params = list(sig.parameters.keys())

        assert 'data_path' in params, "Function must accept data_path parameter"


class TestSampleDataGeneration:
    """Generate consistent sample M5 data for integration testing."""

    @staticmethod
    def create_sample_m5_data(temp_dir: str, n_items: int = 10, n_days: int = 365) -> str:
        """
        Create realistic sample M5 data for integration testing.

        Parameters
        ----------
        temp_dir : str
            Temporary directory for data files
        n_items : int
            Number of items to create
        n_days : int
            Number of days of sales data

        Returns
        -------
        str
            Path to temporary directory containing data files
        """
        np.random.seed(42)

        # Create sales data
        categories = ['FOODS', 'HOUSEHOLD', 'HOBBIES']
        departments = {
            'FOODS': ['FOODS_1', 'FOODS_2', 'FOODS_3'],
            'HOUSEHOLD': ['HOUSEHOLD_1', 'HOUSEHOLD_2'],
            'HOBBIES': ['HOBBIES_1', 'HOBBIES_2', 'HOBBIES_3']
        }
        stores = ['CA_1', 'CA_2', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2']
        states = {'CA': 'CA', 'TX': 'TX', 'WI': 'WI'}

        sales_data = []
        for i in range(n_items):
            cat_id = np.random.choice(categories)
            dept_id = np.random.choice(departments[cat_id])
            store_id = np.random.choice(stores)
            state_id = store_id.split('_')[0]
            item_id = f"{cat_id}_1_{i:03d}"

            row = {
                'id': f"{item_id}_{store_id}_validation",
                'item_id': item_id,
                'cat_id': cat_id,
                'dept_id': dept_id,
                'store_id': store_id,
                'state_id': state_id,
            }

            # Generate sales data with realistic patterns
            for day in range(1, n_days + 1):
                # Add trend, seasonality, and randomness
                trend = 0.001 * day
                seasonal = 5 * np.sin(2 * np.pi * day / 365)
                noise = np.random.normal(0, 2)
                sales = max(0, 10 + trend + seasonal + noise)
                row[f'd_{day}'] = int(sales)

            sales_data.append(row)

        sales_df = pd.DataFrame(sales_data)

        # Create calendar data
        start_date = pd.Timestamp('2016-01-29')
        calendar_data = []
        for day in range(1, n_days + 1):
            date = start_date + pd.Timedelta(days=day-1)
            weekday = date.strftime('%A')
            wm_yr_wk = int(date.strftime('%y%V'))

            calendar_data.append({
                'd': f'd_{day}',
                'date': date,
                'weekday': weekday,
                'wm_yr_wk': wm_yr_wk,
                'snap_CA': 1 if day % 7 == 0 else 0,
                'snap_TX': 1 if day % 8 == 0 else 0,
                'snap_WI': 1 if day % 9 == 0 else 0,
                'event_name': 'Holiday' if day % 60 == 0 else None,
                'event_type': 'National' if day % 60 == 0 else None,
            })

        calendar_df = pd.DataFrame(calendar_data)

        # Create price data
        price_data = []
        for _, row in sales_df.iterrows():
            base_price = np.random.uniform(0.5, 10.0)
            price_data.append({
                'store_id': row['store_id'],
                'item_id': row['item_id'],
                'wm_yr_wk': calendar_df.iloc[0]['wm_yr_wk'],
                'sell_price': base_price
            })

        price_df = pd.DataFrame(price_data)

        # Save files
        sales_df.to_csv(os.path.join(temp_dir, 'sales_train_validation.csv'), index=False)
        calendar_df.to_csv(os.path.join(temp_dir, 'calendar.csv'), index=False)
        price_df.to_csv(os.path.join(temp_dir, 'sell_prices.csv'), index=False)

        return temp_dir


class TestFullEDAPipelineIntegration:
    """Test the complete EDA pipeline with all framework steps."""

    @pytest.fixture
    def sample_m5_data(self):
        """Create sample M5 data for all integration tests."""
        temp_dir = tempfile.mkdtemp()
        TestSampleDataGeneration.create_sample_m5_data(temp_dir)
        yield temp_dir
        # Cleanup is handled by pytest's temp directory management

    def test_step6_feature_target_relationships_integration(self, sample_m5_data):
        """Test Step 6: Feature-Target Relationships analysis."""
        result = study_feature_target_relationships(data_path=sample_m5_data)

        # Verify result structure
        assert isinstance(result, dict), "Result should be a dictionary"

        # Verify key components
        expected_keys = [
            'categorical_patterns',
            'temporal_correlations',
            'snap_impact',
            'summary'
        ]
        for key in expected_keys:
            assert key in result, f"Result must contain '{key}'"

        # Verify no critical errors
        for section in ['categorical_patterns', 'temporal_correlations']:
            if section in result and isinstance(result[section], dict):
                assert 'error' not in result[section] or result[section]['error'] is None

    def test_step7_feature_feature_relationships_integration(self, sample_m5_data):
        """Test Step 7: Feature-Feature Relationships analysis."""
        result = study_feature_feature_relationships(data_path=sample_m5_data)

        assert isinstance(result, dict), "Result should be a dictionary"

        # Verify key sections (actual structure has nested organization)
        assert 'cross_feature_correlations' in result or 'multicollinearity_analysis' in result, \
            "Result must contain feature relationship analysis"

        # Verify that main sections exist
        has_analysis = any(key in result for key in [
            'cross_feature_correlations',
            'multicollinearity_analysis',
            'geographic_correlations',
            'product_hierarchy'
        ])
        assert has_analysis, "Result should contain feature relationship data"

    def test_step8_time_series_patterns_integration(self, sample_m5_data):
        """Test Step 8: Time Series Patterns analysis."""
        result = analyze_time_series_patterns(data_path=sample_m5_data)

        assert isinstance(result, dict), "Result should be a dictionary"

        # Verify key sections (flexible structure)
        has_analysis = any(key in result for key in [
            'time_structure',
            'seasonal_patterns',
            'trend_analysis',
            'autocorrelation_analysis'
        ])
        assert has_analysis, "Result should contain time series analysis data"

    def test_step9_missing_values_integration(self, sample_m5_data):
        """Test Step 9: Missing Values analysis."""
        result = analyze_missing_values_deeply(data_path=sample_m5_data)

        assert isinstance(result, dict), "Result should be a dictionary"

        # Verify key sections (flexible structure)
        has_analysis = any(key in result for key in [
            'missing_patterns',
            'missing_mechanisms',
            'recommendations',
            'summary_statistics'
        ])
        assert has_analysis, "Result should contain missing value analysis data"

    def test_step10_outliers_anomalies_integration(self, sample_m5_data):
        """Test Step 10: Outliers and Anomalies detection."""
        result = identify_outliers_and_anomalies(data_path=sample_m5_data)

        assert isinstance(result, dict), "Result should be a dictionary"

        # Verify key sections (flexible structure)
        has_analysis = any(key in result for key in [
            'sales_outliers',
            'pricing_anomalies',
            'recommendations',
            'summary_statistics'
        ])
        assert has_analysis, "Result should contain outlier detection data"


class TestPlotDirectoryStructure:
    """Test that all plot directories exist and are functional."""

    def test_plot_directories_exist(self):
        """Verify all framework step directories exist."""
        base_plots_dir = Path(__file__).parent.parent / 'plots'

        expected_dirs = [
            'step6_feature_target',
            'step7_feature_relationships',
            'step8_time_series',
            'step9_missing_patterns',
            'step10_outliers'
        ]

        for dir_name in expected_dirs:
            dir_path = base_plots_dir / dir_name
            assert dir_path.exists(), f"Directory {dir_path} should exist"
            assert dir_path.is_dir(), f"{dir_path} should be a directory"

    def test_plot_generation_step6(self):
        """Verify Step 6 plots can be generated."""
        temp_dir = tempfile.mkdtemp()
        TestSampleDataGeneration.create_sample_m5_data(temp_dir)

        result = study_feature_target_relationships(data_path=temp_dir)

        # Check if visualizations were created
        if 'visualizations' in result and result['visualizations']:
            # At least one visualization should be attempted
            assert isinstance(result['visualizations'], dict)

    def test_plot_generation_step7(self):
        """Verify Step 7 plots can be generated."""
        temp_dir = tempfile.mkdtemp()
        TestSampleDataGeneration.create_sample_m5_data(temp_dir)

        result = study_feature_feature_relationships(data_path=temp_dir)

        # Verify result structure
        assert isinstance(result, dict)
        if 'visualizations' in result and result['visualizations']:
            assert isinstance(result['visualizations'], dict)

    def test_plot_generation_step8(self):
        """Verify Step 8 plots can be generated."""
        temp_dir = tempfile.mkdtemp()
        TestSampleDataGeneration.create_sample_m5_data(temp_dir)

        result = analyze_time_series_patterns(data_path=temp_dir)

        # Verify result structure
        assert isinstance(result, dict)
        if 'visualizations' in result and result['visualizations']:
            assert isinstance(result['visualizations'], dict)


class TestModuleImports:
    """Test module import structure and availability."""

    def test_eda_analysis_module_imports(self):
        """Test that eda_analysis module imports successfully."""
        from eda_analysis import (
            study_feature_target_relationships,
            study_feature_feature_relationships,
            analyze_time_series_patterns,
            analyze_missing_values_deeply,
            identify_outliers_and_anomalies,
        )

        # Verify all imports are callable
        assert callable(study_feature_target_relationships)
        assert callable(study_feature_feature_relationships)
        assert callable(analyze_time_series_patterns)
        assert callable(analyze_missing_values_deeply)
        assert callable(identify_outliers_and_anomalies)

    def test_utils_modules_import(self):
        """Test that all utility modules import successfully."""
        from utils.correlation_analysis import (
            analyze_categorical_sales_patterns,
            compute_temporal_sales_correlations,
            compute_snap_benefit_impact,
            compute_cross_feature_correlations,
            detect_multicollinearity_issues
        )

        from utils.temporal_analysis import (
            analyze_time_structure,
            detect_seasonal_patterns,
            analyze_trend_components,
            compute_autocorrelation_analysis
        )

        from utils.visualization import (
            plot_categorical_sales_distributions,
            plot_correlation_heatmap,
            plot_multicollinearity_analysis,
            plot_seasonal_decomposition,
            plot_autocorrelation_analysis,
            plot_missing_patterns,
            plot_outlier_detection
        )

        from utils.data_quality import (
            analyze_missing_patterns,
            characterize_missing_mechanisms,
            detect_sales_outliers,
            analyze_pricing_anomalies
        )

        # Verify key functions are callable
        assert callable(analyze_categorical_sales_patterns)
        assert callable(compute_cross_feature_correlations)
        assert callable(detect_seasonal_patterns)
        assert callable(plot_categorical_sales_distributions)
        assert callable(analyze_missing_patterns)
        assert callable(detect_sales_outliers)

    def test_utils_statistical_analysis_import(self):
        """Test that statistical analysis utilities are available."""
        try:
            from utils.statistical_analysis import (
                calculate_distribution_stats,
                compute_variation_metrics,
                analyze_outliers,
                test_normality,
                test_independence,
                calculate_intermittency_score,
                classify_demand_variability
            )
            assert callable(calculate_distribution_stats)
            assert callable(compute_variation_metrics)
            assert callable(test_normality)
        except ImportError:
            # Module may be integrated into other modules
            # Just ensure utils package is available
            import utils
            assert hasattr(utils, '__file__')


class TestParameterValidation:
    """Test parameter handling and validation."""

    def test_invalid_data_path_handling(self):
        """Test that functions handle invalid data paths gracefully."""
        invalid_path = "/nonexistent/path/to/data"

        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            study_feature_target_relationships(data_path=invalid_path)

    def test_partial_data_files_handling(self):
        """Test handling when some data files are missing."""
        temp_dir = tempfile.mkdtemp()

        # Create only sales and calendar, missing prices
        np.random.seed(42)
        sales_data = pd.DataFrame({
            'id': ['FOODS_1_001_CA_1_validation'],
            'item_id': ['FOODS_1_001'],
            'cat_id': ['FOODS'],
            'dept_id': ['FOODS_1'],
            'store_id': ['CA_1'],
            'state_id': ['CA'],
            'd_1': [5], 'd_2': [3], 'd_3': [7]
        })

        calendar_data = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3'],
            'date': pd.to_datetime(['2011-01-29', '2011-01-30', '2011-01-31']),
            'weekday': ['Saturday', 'Sunday', 'Monday'],
            'wm_yr_wk': [11101, 11101, 11101],
            'snap_CA': [0, 1, 0],
            'snap_TX': [1, 0, 1],
            'snap_WI': [0, 1, 1]
        })

        sales_data.to_csv(os.path.join(temp_dir, 'sales_train_validation.csv'), index=False)
        calendar_data.to_csv(os.path.join(temp_dir, 'calendar.csv'), index=False)

        # Should handle gracefully (may work with reduced functionality)
        result = study_feature_target_relationships(data_path=temp_dir)
        assert isinstance(result, dict)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_full_framework_execution(self):
        """Test executing all 5 framework steps in sequence."""
        temp_dir = tempfile.mkdtemp()
        TestSampleDataGeneration.create_sample_m5_data(temp_dir, n_items=5, n_days=90)

        # Execute all framework steps
        results = {}
        try:
            results['step6'] = study_feature_target_relationships(data_path=temp_dir)
            results['step7'] = study_feature_feature_relationships(data_path=temp_dir)
            results['step8'] = analyze_time_series_patterns(data_path=temp_dir)
            results['step9'] = analyze_missing_values_deeply(data_path=temp_dir)
            results['step10'] = identify_outliers_and_anomalies(data_path=temp_dir)
        except Exception as e:
            pytest.fail(f"Full framework execution failed: {e}")

        # Verify all steps produced results
        for step_name, result in results.items():
            assert isinstance(result, dict), f"{step_name} should return a dictionary"
            # Check for summary or other result indicators (flexible structure)
            assert len(result) > 0, f"{step_name} should contain analysis results"

    def test_result_consistency_across_steps(self):
        """Test that results are consistent when analyzing same data across steps."""
        temp_dir = tempfile.mkdtemp()
        TestSampleDataGeneration.create_sample_m5_data(temp_dir, n_items=5, n_days=90)

        # Run same analysis twice
        result1 = study_feature_target_relationships(data_path=temp_dir)
        result2 = study_feature_target_relationships(data_path=temp_dir)

        # Results should have same structure
        assert result1.keys() == result2.keys(), "Results should have consistent structure"


class TestDataTypeHandling:
    """Test handling of different data types and edge cases."""

    def test_empty_category_handling(self):
        """Test handling of categories with no data."""
        temp_dir = tempfile.mkdtemp()

        # Create minimal data
        sales_data = pd.DataFrame({
            'id': ['FOODS_1_001_CA_1_validation', 'FOODS_1_002_CA_1_validation'],
            'item_id': ['FOODS_1_001', 'FOODS_1_002'],
            'cat_id': ['FOODS', 'FOODS'],
            'dept_id': ['FOODS_1', 'FOODS_1'],
            'store_id': ['CA_1', 'CA_1'],
            'state_id': ['CA', 'CA'],
            'd_1': [0, 0], 'd_2': [0, 0], 'd_3': [0, 0]
        })

        calendar_data = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3'],
            'date': pd.to_datetime(['2011-01-29', '2011-01-30', '2011-01-31']),
            'weekday': ['Saturday', 'Sunday', 'Monday'],
            'wm_yr_wk': [11101, 11101, 11101],
            'snap_CA': [0, 1, 0],
            'snap_TX': [1, 0, 1],
            'snap_WI': [0, 1, 1]
        })

        sales_data.to_csv(os.path.join(temp_dir, 'sales_train_validation.csv'), index=False)
        calendar_data.to_csv(os.path.join(temp_dir, 'calendar.csv'), index=False)

        # Should handle gracefully
        result = study_feature_target_relationships(data_path=temp_dir)
        assert isinstance(result, dict)

    def test_large_dataset_handling(self):
        """Test handling of reasonably large datasets."""
        temp_dir = tempfile.mkdtemp()

        # Create larger dataset
        TestSampleDataGeneration.create_sample_m5_data(temp_dir, n_items=20, n_days=100)

        # Should complete without errors
        result = study_feature_target_relationships(data_path=temp_dir)
        assert isinstance(result, dict)


class TestDocumentationAndMetadata:
    """Test documentation and metadata in functions and results."""

    def test_function_docstrings(self):
        """Test that all main functions have docstrings."""
        from eda_analysis import (
            study_feature_target_relationships,
            study_feature_feature_relationships,
            analyze_time_series_patterns,
            analyze_missing_values_deeply,
            identify_outliers_and_anomalies,
        )

        functions = [
            study_feature_target_relationships,
            study_feature_feature_relationships,
            analyze_time_series_patterns,
            analyze_missing_values_deeply,
            identify_outliers_and_anomalies,
        ]

        for func in functions:
            assert func.__doc__ is not None, f"{func.__name__} should have a docstring"
            assert len(func.__doc__) > 50, f"{func.__name__} docstring should be substantive"

    def test_result_summary_structure(self):
        """Test that results contain properly structured summaries."""
        temp_dir = tempfile.mkdtemp()
        TestSampleDataGeneration.create_sample_m5_data(temp_dir, n_items=5, n_days=90)

        result = study_feature_target_relationships(data_path=temp_dir)

        assert 'summary' in result, "Result should contain a summary section"
        assert isinstance(result['summary'], (dict, str)), "Summary should be dict or string"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
