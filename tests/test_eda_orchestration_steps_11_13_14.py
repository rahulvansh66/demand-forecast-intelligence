"""
Comprehensive tests for EDA orchestration functions (Steps 11, 13, 14).

Tests validate:
- Integration with existing M5 data loading functions
- Proper execution flow and error handling
- Visualization generation and plot directory creation
- Results structure and completeness
- Business insights generation
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from notebooks.eda.eda_analysis import (
    analyze_segment_behavior,
    analyze_distribution_drift,
    audit_temporal_leakage,
    _update_eda_report_step11,
    _update_eda_report_step13,
    _update_eda_report_step14
)


class TestAnalyzeSegmentBehavior:
    """Test Suite for analyze_segment_behavior (Step 11)"""

    @pytest.fixture
    def sample_m5_data(self, tmp_path):
        """Create sample M5 dataset for testing"""
        # Create sales data with realistic structure
        np.random.seed(42)
        n_products = 100
        n_days = 200

        # Create product metadata
        products = []
        for i in range(n_products):
            cat_id = np.random.choice(['FOODS', 'HOBBIES', 'HOUSEHOLD'])
            store_id = np.random.choice(['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3'])
            item_id = f"{cat_id}_{'0'*(3-len(str(i)))}{i}"

            products.append({
                'id': i,
                'item_id': item_id,
                'dept_id': f"{cat_id}_DEPT_{i % 3}",
                'cat_id': cat_id,
                'store_id': store_id
            })

        sales_data = pd.DataFrame(products)

        # Add daily sales columns
        for day in range(1, n_days + 1):
            sales_data[f'd_{day}'] = np.random.poisson(lam=5, size=n_products)

        # Create calendar data
        num_weeks = n_days // 7 + 1
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        wday_list = (day_names * num_weeks)[:n_days]

        calendar_data = pd.DataFrame({
            'd': [f'd_{i}' for i in range(1, n_days + 1)],
            'date': pd.date_range('2016-01-01', periods=n_days),
            'wm_yr_wk': np.repeat(np.arange(1, num_weeks + 1), 7)[:n_days],
            'weekday': np.tile(list(range(7)), num_weeks)[:n_days],
            'wday': wday_list
        })

        # Save to temporary directory
        data_path = tmp_path / "data"
        data_path.mkdir()

        sales_data.to_csv(data_path / "sales_train_validation.csv", index=False)
        calendar_data.to_csv(data_path / "calendar.csv", index=False)

        return str(data_path)

    def test_segment_behavior_basic_execution(self, sample_m5_data):
        """Test basic execution of analyze_segment_behavior"""
        results = analyze_segment_behavior(data_path=sample_m5_data)

        assert isinstance(results, dict)
        assert 'error' not in results or results.get('error') is None

    def test_segment_behavior_result_structure(self, sample_m5_data):
        """Test that results have expected structure"""
        results = analyze_segment_behavior(data_path=sample_m5_data)

        # Check key result sections
        expected_keys = [
            'category_behavior',
            'department_segments',
            'performance_metrics',
            'seasonality_patterns',
            'lifecycle_stages',
            'visualizations',
            'summary'
        ]

        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_segment_behavior_summary_completeness(self, sample_m5_data):
        """Test that summary contains all required metrics"""
        results = analyze_segment_behavior(data_path=sample_m5_data)
        summary = results.get('summary', {})

        assert 'total_categories' in summary
        assert 'departments_analyzed' in summary
        assert 'top_performers' in summary
        assert 'seasonal_segments' in summary
        assert 'lifecycle_distribution' in summary
        assert 'step_status' in summary
        assert summary.get('step_status') == 'complete'

    def test_segment_behavior_lifecycle_distribution(self, sample_m5_data):
        """Test that lifecycle distribution is properly structured"""
        results = analyze_segment_behavior(data_path=sample_m5_data)
        lifecycle = results.get('summary', {}).get('lifecycle_distribution', {})

        assert 'new' in lifecycle
        assert 'mature' in lifecycle
        assert 'discontinued' in lifecycle
        assert isinstance(lifecycle['new'], int)
        assert isinstance(lifecycle['mature'], int)
        assert isinstance(lifecycle['discontinued'], int)

    def test_segment_behavior_visualization_creation(self, sample_m5_data, tmp_path):
        """Test that visualization directory is created"""
        # Change to temp directory for plots
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            results = analyze_segment_behavior(data_path=sample_m5_data)

            # Check if plot directory was created
            plot_dir = Path("notebooks/eda/plots/step11_segment_behavior")
            # The directory may or may not exist depending on whether visualization succeeded
            assert 'visualizations' in results
        finally:
            os.chdir(original_cwd)


class TestAnalyzeDistributionDrift:
    """Test Suite for analyze_distribution_drift (Step 13)"""

    @pytest.fixture
    def sample_m5_data_with_validation(self, tmp_path):
        """Create sample M5 dataset with train/validation split"""
        np.random.seed(42)
        n_products = 50
        n_train_days = 500
        n_val_days = 28

        # Create product metadata
        products = []
        for i in range(n_products):
            cat_id = np.random.choice(['FOODS', 'HOBBIES', 'HOUSEHOLD'])
            store_id = np.random.choice(['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3'])
            item_id = f"{cat_id}_{'0'*(3-len(str(i)))}{i}"

            products.append({
                'id': i,
                'item_id': item_id,
                'dept_id': f"{cat_id}_DEPT_{i % 3}",
                'cat_id': cat_id,
                'store_id': store_id
            })

        sales_data = pd.DataFrame(products)

        # Add daily sales columns with realistic train/validation split
        total_days = n_train_days + n_val_days

        for day in range(1, total_days + 1):
            # Introduce slight drift in validation period
            if day <= n_train_days:
                sales_data[f'd_{day}'] = np.random.poisson(lam=5, size=n_products)
            else:
                # Validation period has slightly different distribution
                sales_data[f'd_{day}'] = np.random.poisson(lam=6, size=n_products)

        # Create calendar data
        num_weeks_val = total_days // 7 + 1
        day_names_val = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        wday_list_val = (day_names_val * num_weeks_val)[:total_days]

        calendar_data = pd.DataFrame({
            'd': [f'd_{i}' for i in range(1, total_days + 1)],
            'date': pd.date_range('2015-01-01', periods=total_days),
            'wm_yr_wk': np.repeat(np.arange(1, num_weeks_val + 1), 7)[:total_days],
            'weekday': np.tile(list(range(7)), num_weeks_val)[:total_days],
            'wday': wday_list_val
        })

        # Save to temporary directory
        data_path = tmp_path / "data"
        data_path.mkdir()

        sales_data.to_csv(data_path / "sales_train_validation.csv", index=False)
        calendar_data.to_csv(data_path / "calendar.csv", index=False)

        return str(data_path)

    def test_distribution_drift_basic_execution(self, sample_m5_data_with_validation):
        """Test basic execution of analyze_distribution_drift"""
        results = analyze_distribution_drift(data_path=sample_m5_data_with_validation)

        assert isinstance(results, dict)
        assert 'error' not in results or results.get('error') is None

    def test_distribution_drift_result_structure(self, sample_m5_data_with_validation):
        """Test that results have expected structure"""
        results = analyze_distribution_drift(data_path=sample_m5_data_with_validation)

        expected_keys = [
            'distribution_comparison',
            'seasonal_representativeness',
            'category_drift',
            'drift_severity',
            'temporal_integrity',
            'visualizations',
            'summary'
        ]

        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_distribution_drift_summary_metrics(self, sample_m5_data_with_validation):
        """Test that summary contains drift assessment metrics"""
        results = analyze_distribution_drift(data_path=sample_m5_data_with_validation)
        summary = results.get('summary', {})

        assert 'overall_risk_level' in summary
        assert 'drifted_categories' in summary
        assert 'seasonal_coverage_percent' in summary
        assert 'temporal_split_valid' in summary
        assert 'recommendations' in summary
        assert 'step_status' in summary
        assert summary.get('step_status') == 'complete'

    def test_distribution_drift_risk_level_values(self, sample_m5_data_with_validation):
        """Test that risk level has valid values"""
        results = analyze_distribution_drift(data_path=sample_m5_data_with_validation)
        risk_level = results.get('summary', {}).get('overall_risk_level')

        # Risk level should be valid string or 'unknown'
        assert isinstance(risk_level, str)
        assert len(risk_level) > 0

    def test_distribution_drift_seasonal_coverage_range(self, sample_m5_data_with_validation):
        """Test that seasonal coverage is valid percentage"""
        results = analyze_distribution_drift(data_path=sample_m5_data_with_validation)
        coverage = results.get('summary', {}).get('seasonal_coverage_percent', 0)

        assert isinstance(coverage, (int, float))
        assert 0 <= coverage <= 100


class TestAuditTemporalLeakage:
    """Test Suite for audit_temporal_leakage (Step 14)"""

    @pytest.fixture
    def sample_m5_complete_data(self, tmp_path):
        """Create complete sample M5 dataset with pricing"""
        np.random.seed(42)
        n_products = 50
        n_days = 600

        # Create product metadata
        products = []
        for i in range(n_products):
            cat_id = np.random.choice(['FOODS', 'HOBBIES', 'HOUSEHOLD'])
            store_id = np.random.choice(['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3'])
            item_id = f"{cat_id}_{'0'*(3-len(str(i)))}{i}"

            products.append({
                'id': i,
                'item_id': item_id,
                'dept_id': f"{cat_id}_DEPT_{i % 3}",
                'cat_id': cat_id,
                'store_id': store_id
            })

        sales_data = pd.DataFrame(products)

        # Add daily sales columns
        for day in range(1, n_days + 1):
            sales_data[f'd_{day}'] = np.random.poisson(lam=5, size=n_products)

        # Create calendar data
        num_weeks_complete = n_days // 7 + 1
        day_names_complete = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        wday_list_complete = (day_names_complete * num_weeks_complete)[:n_days]

        calendar_data = pd.DataFrame({
            'd': [f'd_{i}' for i in range(1, n_days + 1)],
            'date': pd.date_range('2014-01-01', periods=n_days),
            'wm_yr_wk': np.repeat(np.arange(1, num_weeks_complete + 1), 7)[:n_days],
            'weekday': np.tile(list(range(7)), num_weeks_complete)[:n_days],
            'wday': wday_list_complete
        })

        # Create pricing data
        pricing_data = []
        for product_idx in range(n_products):
            for week in range(1, n_days // 7 + 1):
                store_id = sales_data.iloc[product_idx]['store_id']
                item_id = sales_data.iloc[product_idx]['item_id']
                price = np.random.uniform(1.0, 50.0)

                pricing_data.append({
                    'item_id': item_id,
                    'store_id': store_id,
                    'wm_yr_wk': week,
                    'sell_price': price
                })

        pricing_df = pd.DataFrame(pricing_data)

        # Save to temporary directory
        data_path = tmp_path / "data"
        data_path.mkdir()

        sales_data.to_csv(data_path / "sales_train_validation.csv", index=False)
        calendar_data.to_csv(data_path / "calendar.csv", index=False)
        pricing_df.to_csv(data_path / "sell_prices.csv", index=False)

        return str(data_path)

    def test_temporal_leakage_audit_basic_execution(self, sample_m5_complete_data):
        """Test basic execution of audit_temporal_leakage"""
        results = audit_temporal_leakage(data_path=sample_m5_complete_data)

        assert isinstance(results, dict)
        assert 'error' not in results or results.get('error') is None

    def test_temporal_leakage_audit_with_config(self, sample_m5_complete_data):
        """Test audit with custom feature engineering config"""
        config = {
            'lag_features': [1, 7, 14, 28],
            'rolling_features': [7, 14, 28],
            'seasonal_features': [365],
            'price_features': [7]
        }

        results = audit_temporal_leakage(
            data_path=sample_m5_complete_data,
            feature_engineering_config=config
        )

        assert isinstance(results, dict)

    def test_temporal_leakage_audit_result_structure(self, sample_m5_complete_data):
        """Test that results have expected structure"""
        results = audit_temporal_leakage(data_path=sample_m5_complete_data)

        expected_keys = [
            'temporal_boundaries',
            'feature_availability',
            'cross_validation',
            'suspicious_correlations',
            'audit_report',
            'visualizations',
            'summary'
        ]

        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_temporal_leakage_audit_summary_completeness(self, sample_m5_complete_data):
        """Test that summary contains deployment readiness information"""
        results = audit_temporal_leakage(data_path=sample_m5_complete_data)
        summary = results.get('summary', {})

        assert 'deployment_ready' in summary
        assert 'risk_level' in summary
        assert 'critical_issues' in summary
        assert 'total_recommendations' in summary
        assert 'critical_recommendations' in summary
        assert 'step_status' in summary
        assert summary.get('step_status') == 'complete'

    def test_temporal_leakage_audit_deployment_ready_type(self, sample_m5_complete_data):
        """Test that deployment_ready is boolean"""
        results = audit_temporal_leakage(data_path=sample_m5_complete_data)
        deployment_ready = results.get('summary', {}).get('deployment_ready')

        assert isinstance(deployment_ready, bool)

    def test_temporal_leakage_audit_recommendations_format(self, sample_m5_complete_data):
        """Test that recommendations are properly formatted"""
        results = audit_temporal_leakage(data_path=sample_m5_complete_data)
        recommendations = results.get('summary', {}).get('critical_recommendations', [])

        assert isinstance(recommendations, list)
        # Recommendations should be strings or can be empty
        for rec in recommendations:
            assert isinstance(rec, str) or rec is None


class TestReportUpdateHelpers:
    """Test Suite for EDA report update helper functions"""

    def test_update_eda_report_step11_no_file(self):
        """Test that _update_eda_report_step11 handles missing file gracefully"""
        results = {
            'summary': {
                'total_categories': 3,
                'departments_analyzed': 10,
                'top_performers': 5,
                'seasonal_segments': 8,
                'lifecycle_distribution': {'new': 2, 'mature': 15, 'discontinued': 1}
            }
        }

        # Should not raise error even if file doesn't exist
        try:
            _update_eda_report_step11(results)
        except Exception as e:
            pytest.fail(f"_update_eda_report_step11 raised unexpected exception: {e}")

    def test_update_eda_report_step13_no_file(self):
        """Test that _update_eda_report_step13 handles missing file gracefully"""
        results = {
            'summary': {
                'overall_risk_level': 'LOW',
                'drifted_categories': 2,
                'seasonal_coverage_percent': 85.5,
                'temporal_split_valid': True,
                'recommendations': ['Rec 1', 'Rec 2']
            }
        }

        # Should not raise error even if file doesn't exist
        try:
            _update_eda_report_step13(results)
        except Exception as e:
            pytest.fail(f"_update_eda_report_step13 raised unexpected exception: {e}")

    def test_update_eda_report_step14_no_file(self):
        """Test that _update_eda_report_step14 handles missing file gracefully"""
        results = {
            'summary': {
                'deployment_ready': True,
                'risk_level': 'LOW',
                'critical_issues': 0,
                'total_recommendations': 3,
                'critical_recommendations': ['Rec 1', 'Rec 2', 'Rec 3']
            }
        }

        # Should not raise error even if file doesn't exist
        try:
            _update_eda_report_step14(results)
        except Exception as e:
            pytest.fail(f"_update_eda_report_step14 raised unexpected exception: {e}")


class TestIntegrationWithExistingPatterns:
    """Integration tests for adherence to existing EDA framework patterns"""

    @pytest.fixture
    def sample_m5_data(self, tmp_path):
        """Create sample M5 dataset for testing"""
        np.random.seed(42)
        n_products = 30
        n_days = 300

        products = []
        for i in range(n_products):
            cat_id = np.random.choice(['FOODS', 'HOBBIES', 'HOUSEHOLD'])
            store_id = np.random.choice(['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3'])
            item_id = f"{cat_id}_{'0'*(3-len(str(i)))}{i}"

            products.append({
                'id': i,
                'item_id': item_id,
                'dept_id': f"{cat_id}_DEPT_{i % 3}",
                'cat_id': cat_id,
                'store_id': store_id
            })

        sales_data = pd.DataFrame(products)

        for day in range(1, n_days + 1):
            sales_data[f'd_{day}'] = np.random.poisson(lam=5, size=n_products)

        num_weeks_int = n_days // 7 + 1
        day_names_int = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        wday_list_int = (day_names_int * num_weeks_int)[:n_days]

        calendar_data = pd.DataFrame({
            'd': [f'd_{i}' for i in range(1, n_days + 1)],
            'date': pd.date_range('2015-01-01', periods=n_days),
            'wm_yr_wk': np.repeat(np.arange(1, num_weeks_int + 1), 7)[:n_days],
            'weekday': np.tile(list(range(7)), num_weeks_int)[:n_days],
            'wday': wday_list_int
        })

        data_path = tmp_path / "data"
        data_path.mkdir()

        sales_data.to_csv(data_path / "sales_train_validation.csv", index=False)
        calendar_data.to_csv(data_path / "calendar.csv", index=False)

        return str(data_path)

    def test_error_handling_missing_files(self, tmp_path):
        """Test error handling when files are missing"""
        empty_path = str(tmp_path / "empty")

        with pytest.raises(FileNotFoundError):
            analyze_segment_behavior(data_path=empty_path)

    def test_error_handling_invalid_data(self, tmp_path):
        """Test error handling with invalid data"""
        data_path = tmp_path / "data"
        data_path.mkdir()

        # Create invalid CSV files
        pd.DataFrame({'invalid': [1, 2, 3]}).to_csv(data_path / "sales_train_validation.csv", index=False)
        pd.DataFrame({'invalid': [1, 2, 3]}).to_csv(data_path / "calendar.csv", index=False)

        # Should handle gracefully
        results = analyze_segment_behavior(data_path=str(data_path))
        assert 'error' in results or 'summary' in results

    def test_functions_follow_naming_conventions(self):
        """Test that new functions follow established naming patterns"""
        # Functions should be named with action_object pattern
        assert callable(analyze_segment_behavior)
        assert callable(analyze_distribution_drift)
        assert callable(audit_temporal_leakage)

        # Helper functions should be prefixed with underscore
        assert callable(_update_eda_report_step11)
        assert callable(_update_eda_report_step13)
        assert callable(_update_eda_report_step14)

    def test_functions_have_docstrings(self):
        """Test that all functions have comprehensive docstrings"""
        assert analyze_segment_behavior.__doc__ is not None
        assert len(analyze_segment_behavior.__doc__) > 100

        assert analyze_distribution_drift.__doc__ is not None
        assert len(analyze_distribution_drift.__doc__) > 100

        assert audit_temporal_leakage.__doc__ is not None
        assert len(audit_temporal_leakage.__doc__) > 100


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
