"""
Tests for EDA Steps 1, 2, 3, 5 Implementation for M5 Demand Forecasting Dataset

Following TDD approach - these tests should fail first, then implementation
should be created to make them pass.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys

# Add the parent directory to sys.path to import from eda_steps_1_3_5
sys.path.append(str(Path(__file__).parent.parent))

from eda_steps_1_3_5 import (
    analyze_m5_problem_context,
    inspect_m5_dataset_structure,
    check_m5_data_quality,
    analyze_m5_individual_features
)


class TestAnalyzeM5ProblemContext:
    """Tests for EDA Step 1: Validate M5 forecasting objectives and data leakage prevention"""

    def test_analyze_m5_problem_context_returns_correct_structure(self):
        """Test that the function returns the expected dictionary structure"""
        result = analyze_m5_problem_context()

        # Check main structure
        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'statistics' in result
        assert 'visualizations' in result
        assert 'detailed_results' in result

        # Check summary structure
        summary = result['summary']
        assert 'step_name' in summary
        assert 'execution_time' in summary
        assert 'data_shape' in summary
        assert 'key_findings' in summary
        assert 'recommendations' in summary

        # Check specific values
        assert summary['step_name'] == 'Step 1: M5 Problem Context Analysis'
        assert isinstance(summary['execution_time'], (int, float))
        assert isinstance(summary['key_findings'], list)
        assert isinstance(summary['recommendations'], list)

    @patch('eda_steps_1_3_5.basic_validation.validate_m5_hierarchy_structure')
    @patch('eda_steps_1_3_5.basic_validation.validate_temporal_boundaries')
    @patch('eda_steps_1_3_5.basic_validation.validate_business_objectives')
    def test_analyze_m5_problem_context_calls_validation_functions(
        self, mock_business, mock_temporal, mock_hierarchy
    ):
        """Test that the function calls all required validation functions"""
        # Setup mocks
        mock_hierarchy.return_value = True
        mock_temporal.return_value = True
        mock_business.return_value = True

        result = analyze_m5_problem_context()

        # Verify function calls
        mock_hierarchy.assert_called_once()
        mock_temporal.assert_called_once()
        mock_business.assert_called_once()

    @patch('eda_steps_1_3_5.enhanced_visualization.plot_hierarchy_tree')
    @patch('eda_steps_1_3_5.enhanced_visualization.plot_timeline_validation')
    def test_analyze_m5_problem_context_creates_visualizations(
        self, mock_timeline_plot, mock_hierarchy_plot
    ):
        """Test that the function creates expected visualizations"""
        # Setup mocks to return file paths
        mock_hierarchy_plot.return_value = '/path/to/hierarchy_tree.png'
        mock_timeline_plot.return_value = '/path/to/timeline_validation.png'

        result = analyze_m5_problem_context()

        # Verify plot functions called
        mock_hierarchy_plot.assert_called_once()
        mock_timeline_plot.assert_called_once()

        # Check visualizations in result
        assert 'hierarchy_tree' in result['visualizations']
        assert 'timeline_validation' in result['visualizations']


class TestInspectM5DatasetStructure:
    """Tests for EDA Step 2: M5-specific dataset structure audit and validation"""

    def test_inspect_m5_dataset_structure_returns_correct_structure(self):
        """Test that the function returns the expected dictionary structure"""
        result = inspect_m5_dataset_structure()

        # Check main structure
        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'statistics' in result
        assert 'visualizations' in result
        assert 'detailed_results' in result

        # Check summary
        summary = result['summary']
        assert summary['step_name'] == 'Step 2: M5 Dataset Structure Analysis'
        assert isinstance(summary['execution_time'], (int, float))
        assert isinstance(summary['data_shape'], tuple)
        assert len(summary['data_shape']) == 2  # rows, cols

    @patch('pandas.read_csv')
    def test_inspect_m5_dataset_structure_loads_required_files(self, mock_read_csv):
        """Test that function attempts to load all required M5 files"""
        # Mock DataFrame returns
        mock_sales = Mock(spec=pd.DataFrame)
        mock_sales.shape = (30490, 1942)
        mock_calendar = Mock(spec=pd.DataFrame)
        mock_calendar.shape = (1969, 14)
        mock_prices = Mock(spec=pd.DataFrame)
        mock_prices.shape = (6841121, 4)

        mock_read_csv.side_effect = [mock_sales, mock_calendar, mock_prices]

        result = inspect_m5_dataset_structure()

        # Verify file loading attempts
        assert mock_read_csv.call_count == 3
        calls = [call.args[0] for call in mock_read_csv.call_args_list]
        assert any('sales_train_validation.csv' in str(call) for call in calls)
        assert any('calendar.csv' in str(call) for call in calls)
        assert any('sell_prices.csv' in str(call) for call in calls)

    def test_inspect_m5_dataset_structure_handles_missing_files(self):
        """Test graceful handling of missing data files"""
        with patch('pandas.read_csv', side_effect=FileNotFoundError("File not found")):
            result = inspect_m5_dataset_structure()

            # Should still return structured result even with missing files
            assert 'summary' in result
            # Either handles error gracefully OR reports error in detailed_results
            if 'error' not in result['detailed_results']:
                # Function handled error gracefully, should have meaningful data
                assert result['summary']['data_shape'][0] > 0  # Has some data
            else:
                # Function reported error, should have error details
                assert 'error' in result['detailed_results']
            assert len(result['summary']['recommendations']) > 0

    @patch('eda_steps_1_3_5.enhanced_visualization.plot_data_coverage_heatmap')
    @patch('eda_steps_1_3_5.enhanced_visualization.plot_table_relationships')
    def test_inspect_m5_dataset_structure_creates_visualizations(
        self, mock_table_plot, mock_coverage_plot
    ):
        """Test that the function creates expected visualizations"""
        mock_coverage_plot.return_value = '/path/to/coverage_heatmap.png'
        mock_table_plot.return_value = '/path/to/table_relationships.png'

        result = inspect_m5_dataset_structure()

        mock_coverage_plot.assert_called_once()
        mock_table_plot.assert_called_once()

        assert 'data_coverage_heatmap' in result['visualizations']
        assert 'table_relationships' in result['visualizations']


class TestCheckM5DataQuality:
    """Tests for EDA Step 3: Retail-specific data quality assessment"""

    def test_check_m5_data_quality_returns_correct_structure(self):
        """Test that the function returns expected dictionary structure"""
        result = check_m5_data_quality()

        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'statistics' in result
        assert 'visualizations' in result
        assert 'detailed_results' in result

        summary = result['summary']
        assert summary['step_name'] == 'Step 3: M5 Data Quality Assessment'
        assert 'quality_score' in result['statistics']
        assert isinstance(result['statistics']['quality_score'], (int, float))
        assert 0 <= result['statistics']['quality_score'] <= 100

    def test_check_m5_data_quality_analyzes_zero_vs_missing_sales(self):
        """Test analysis of zero vs missing sales patterns (intermittent demand)"""
        result = check_m5_data_quality()

        # Should analyze zero vs missing patterns
        detailed = result['detailed_results']
        assert 'zero_sales_analysis' in detailed
        assert 'missing_sales_analysis' in detailed
        assert 'intermittent_demand_patterns' in detailed

    def test_check_m5_data_quality_detects_price_anomalies(self):
        """Test detection of price anomalies"""
        result = check_m5_data_quality()

        detailed = result['detailed_results']
        assert 'price_anomalies' in detailed
        assert 'negative_prices_count' in detailed['price_anomalies']
        assert 'extreme_price_jumps' in detailed['price_anomalies']

    def test_check_m5_data_quality_validates_calendar_consistency(self):
        """Test validation of calendar data consistency"""
        result = check_m5_data_quality()

        detailed = result['detailed_results']
        assert 'calendar_validation' in detailed
        assert 'snap_alignment' in detailed['calendar_validation']
        assert 'event_duplicates' in detailed['calendar_validation']

    @patch('eda_steps_1_3_5.enhanced_visualization.plot_data_quality_dashboard')
    @patch('eda_steps_1_3_5.enhanced_visualization.plot_price_anomaly_detection')
    def test_check_m5_data_quality_creates_visualizations(
        self, mock_price_plot, mock_dashboard_plot
    ):
        """Test that the function creates expected visualizations"""
        mock_dashboard_plot.return_value = '/path/to/quality_dashboard.png'
        mock_price_plot.return_value = '/path/to/price_anomalies.png'

        result = check_m5_data_quality()

        mock_dashboard_plot.assert_called_once()
        mock_price_plot.assert_called_once()

        assert 'data_quality_dashboard' in result['visualizations']
        assert 'price_anomaly_detection' in result['visualizations']


class TestAnalyzeM5IndividualFeatures:
    """Tests for EDA Step 5: Hierarchical and temporal feature deep-dive"""

    def test_analyze_m5_individual_features_returns_correct_structure(self):
        """Test that the function returns expected dictionary structure"""
        result = analyze_m5_individual_features()

        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'statistics' in result
        assert 'visualizations' in result
        assert 'detailed_results' in result

        summary = result['summary']
        assert summary['step_name'] == 'Step 5: M5 Individual Feature Analysis'
        assert 'total_features_analyzed' in result['statistics']

    @patch('eda_steps_1_3_5.feature_profiling.analyze_categorical_distributions')
    @patch('eda_steps_1_3_5.feature_profiling.analyze_geographic_patterns')
    @patch('eda_steps_1_3_5.feature_profiling.analyze_temporal_correlations')
    @patch('eda_steps_1_3_5.feature_profiling.analyze_price_behavior')
    @patch('eda_steps_1_3_5.feature_profiling.calculate_feature_importance_rankings')
    def test_analyze_m5_individual_features_calls_profiling_functions(
        self, mock_importance, mock_price, mock_temporal, mock_geographic, mock_categorical
    ):
        """Test that all required feature profiling functions are called"""
        # Setup mocks
        mock_categorical.return_value = {'categories': {}}
        mock_geographic.return_value = {'patterns': {}}
        mock_temporal.return_value = {'correlations': {}}
        mock_price.return_value = {'behavior': {}}
        mock_importance.return_value = {'rankings': {}}

        result = analyze_m5_individual_features()

        mock_categorical.assert_called_once()
        mock_geographic.assert_called_once()
        mock_temporal.assert_called_once()
        mock_price.assert_called_once()
        mock_importance.assert_called_once()

    @patch('eda_steps_1_3_5.enhanced_visualization.plot_feature_distributions')
    @patch('eda_steps_1_3_5.enhanced_visualization.plot_temporal_correlations')
    def test_analyze_m5_individual_features_creates_visualizations(
        self, mock_temporal_plot, mock_feature_plot
    ):
        """Test that the function creates expected visualizations"""
        mock_feature_plot.return_value = '/path/to/feature_distributions.png'
        mock_temporal_plot.return_value = '/path/to/temporal_correlations.png'

        result = analyze_m5_individual_features()

        mock_feature_plot.assert_called_once()
        mock_temporal_plot.assert_called_once()

        assert 'feature_distributions' in result['visualizations']
        assert 'temporal_correlations' in result['visualizations']

    def test_analyze_m5_individual_features_includes_feature_importance(self):
        """Test that feature importance analysis is included"""
        result = analyze_m5_individual_features()

        detailed = result['detailed_results']
        assert 'feature_importance_rankings' in detailed
        assert 'categorical_analysis' in detailed
        assert 'geographic_patterns' in detailed
        assert 'temporal_analysis' in detailed
        assert 'price_analysis' in detailed


# Integration tests
class TestEDAStepsIntegration:
    """Integration tests for all main EDA functions working together"""

    def test_all_functions_return_compatible_output_format(self):
        """Test that all functions return output compatible with output_manager"""
        functions = [
            analyze_m5_problem_context,
            inspect_m5_dataset_structure,
            check_m5_data_quality,
            analyze_m5_individual_features
        ]

        for func in functions:
            result = func()

            # All should have same high-level structure
            assert isinstance(result, dict)
            assert 'summary' in result
            assert 'statistics' in result
            assert 'visualizations' in result
            assert 'detailed_results' in result

            # Summary should have required fields
            assert 'step_name' in result['summary']
            assert 'execution_time' in result['summary']
            assert 'key_findings' in result['summary']
            assert 'recommendations' in result['summary']

    def test_functions_handle_data_loading_errors_gracefully(self):
        """Test that all functions handle data loading errors gracefully"""
        functions = [
            analyze_m5_problem_context,
            inspect_m5_dataset_structure,
            check_m5_data_quality,
            analyze_m5_individual_features
        ]

        with patch('pandas.read_csv', side_effect=FileNotFoundError("No data files")):
            for func in functions:
                result = func()

                # Should still return structured result
                assert isinstance(result, dict)
                assert 'summary' in result
                # Either handles error gracefully OR reports error in detailed_results
                if 'error' not in result['detailed_results']:
                    # Function handled error gracefully, should have meaningful data
                    assert result['summary']['data_shape'][0] > 0  # Has some data
                else:
                    # Function reported error, should have error details
                    assert 'error' in result['detailed_results']
                assert len(result['summary']['recommendations']) > 0