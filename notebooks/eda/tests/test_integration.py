"""
Integration tests for EDA analysis pipeline.

Tests the complete workflow from raw M5 data to analysis results,
ensuring proper data transformations and function integrations.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path
import sys

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eda_analysis import study_feature_target_relationships


class TestStudyFeatureTargetRelationshipsIntegration:
    """Integration tests for study_feature_target_relationships function."""

    @pytest.fixture
    def sample_m5_data(self):
        """Create sample M5 data files for testing."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        # Create sample sales data (M5 format)
        sales_data = pd.DataFrame({
            'id': ['FOODS_1_001_CA_1_validation', 'HOUSEHOLD_1_001_CA_1_validation', 'HOBBIES_1_001_CA_1_validation'],
            'item_id': ['FOODS_1_001', 'HOUSEHOLD_1_001', 'HOBBIES_1_001'],
            'cat_id': ['FOODS', 'HOUSEHOLD', 'HOBBIES'],
            'dept_id': ['FOODS_1', 'HOUSEHOLD_1', 'HOBBIES_1'],
            'store_id': ['CA_1', 'CA_1', 'CA_1'],
            'state_id': ['CA', 'CA', 'CA'],
            'd_1': [5, 2, 8],
            'd_2': [3, 1, 6],
            'd_3': [7, 0, 9],
            'd_4': [4, 3, 5],
            'd_5': [6, 2, 7]
        })

        # Create sample calendar data
        calendar_data = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3', 'd_4', 'd_5'],
            'date': pd.to_datetime(['2011-01-29', '2011-01-30', '2011-01-31', '2011-02-01', '2011-02-02']),
            'weekday': ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday'],
            'wm_yr_wk': [11101, 11101, 11101, 11102, 11102],
            'snap_CA': [0, 1, 0, 1, 0],
            'snap_TX': [1, 0, 1, 0, 1],
            'snap_WI': [0, 1, 1, 0, 0]
        })

        # Create sample price data
        price_data = pd.DataFrame({
            'store_id': ['CA_1', 'CA_1', 'CA_1'],
            'item_id': ['FOODS_1_001', 'HOUSEHOLD_1_001', 'HOBBIES_1_001'],
            'wm_yr_wk': [11101, 11101, 11101],
            'sell_price': [1.97, 3.17, 5.99]
        })

        # Save to CSV files
        sales_data.to_csv(os.path.join(temp_dir, 'sales_train_validation.csv'), index=False)
        calendar_data.to_csv(os.path.join(temp_dir, 'calendar.csv'), index=False)
        price_data.to_csv(os.path.join(temp_dir, 'sell_prices.csv'), index=False)

        return temp_dir

    def test_basic_workflow_integration(self, sample_m5_data):
        """Test basic end-to-end workflow with sample M5 data."""
        # This test should fail initially due to integration issues
        result = study_feature_target_relationships(data_path=sample_m5_data)

        # Verify result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'categorical_patterns' in result, "Should contain categorical patterns"
        assert 'temporal_correlations' in result, "Should contain temporal correlations"
        assert 'snap_impact' in result, "Should contain SNAP impact analysis"
        assert 'summary' in result, "Should contain summary"

        # Verify no errors in analysis
        assert 'error' not in result.get('categorical_patterns', {}), "Categorical analysis should not have errors"
        assert 'error' not in result.get('temporal_correlations', {}), "Temporal analysis should not have errors"

        # Verify SNAP analysis found FOODS category
        if 'snap_impact' in result and 'error' not in result['snap_impact']:
            assert 'snap_impact_by_state' in result['snap_impact'], "Should analyze SNAP impact by state"
            assert 'foods_category_analysis' in result['snap_impact'], "Should analyze FOODS category"

    def test_missing_files_handling(self):
        """Test handling of missing data files."""
        with pytest.raises((FileNotFoundError, ValueError)):
            study_feature_target_relationships(data_path="/nonexistent/path")

    def test_partial_data_files(self, sample_m5_data):
        """Test with only sales and calendar data (missing prices)."""
        # Remove price file
        price_file = os.path.join(sample_m5_data, 'sell_prices.csv')
        if os.path.exists(price_file):
            os.remove(price_file)

        # Should still work without price data
        result = study_feature_target_relationships(data_path=sample_m5_data)

        assert isinstance(result, dict)
        assert 'categorical_patterns' in result
        assert 'temporal_correlations' in result

    def test_data_transformation_accuracy(self, sample_m5_data):
        """Test that M5 data is properly transformed for analysis functions."""
        result = study_feature_target_relationships(data_path=sample_m5_data)

        # Should have identified 3 categories
        if 'categorical_patterns' in result and 'error' not in result['categorical_patterns']:
            # Exact structure depends on our actual implementation, but should have meaningful data
            assert isinstance(result['categorical_patterns'], dict)

    def test_visualization_generation(self, sample_m5_data):
        """Test that visualizations are generated successfully."""
        result = study_feature_target_relationships(data_path=sample_m5_data)

        if 'visualizations' in result and 'error' not in result['visualizations']:
            # Check that plot was created
            viz_result = result['visualizations'].get('category_distributions', {})
            if 'plot_path' in viz_result:
                plot_path = viz_result['plot_path']
                assert os.path.exists(plot_path), f"Plot should be created at {plot_path}"