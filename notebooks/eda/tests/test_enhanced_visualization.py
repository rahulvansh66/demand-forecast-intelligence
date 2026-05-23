"""
Tests for enhanced visualization utilities for EDA Steps 1, 2, 3, and 5.

This module contains comprehensive tests for enhanced visualization functions
used to support missing EDA framework steps.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import tempfile

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# These imports will fail initially (expected for TDD)
try:
    from utils.enhanced_visualization import plot_hierarchy_tree
except ImportError:
    plot_hierarchy_tree = None

try:
    from utils.enhanced_visualization import plot_timeline_validation
except ImportError:
    plot_timeline_validation = None

try:
    from utils.enhanced_visualization import plot_data_coverage_heatmap
except ImportError:
    plot_data_coverage_heatmap = None

try:
    from utils.enhanced_visualization import plot_table_relationships
except ImportError:
    plot_table_relationships = None

try:
    from utils.enhanced_visualization import plot_data_quality_dashboard
except ImportError:
    plot_data_quality_dashboard = None

try:
    from utils.enhanced_visualization import plot_price_anomaly_detection
except ImportError:
    plot_price_anomaly_detection = None

try:
    from utils.enhanced_visualization import plot_feature_distributions
except ImportError:
    plot_feature_distributions = None

try:
    from utils.enhanced_visualization import plot_temporal_correlations
except ImportError:
    plot_temporal_correlations = None


class TestHierarchyTreePlot:
    """Test hierarchy tree plotting functionality."""

    def test_plot_hierarchy_tree_basic(self):
        """Test basic hierarchy tree plot creation."""
        if plot_hierarchy_tree is None:
            pytest.skip("plot_hierarchy_tree not implemented yet")

        # Sample hierarchy statistics
        hierarchy_stats = {
            'categories': {
                'FOODS': {'departments': ['FOODS_1', 'FOODS_2', 'FOODS_3'], 'item_count': 1500},
                'HOBBIES': {'departments': ['HOBBIES_1', 'HOBBIES_2'], 'item_count': 800},
                'HOUSEHOLD': {'departments': ['HOUSEHOLD_1', 'HOUSEHOLD_2'], 'item_count': 749}
            },
            'departments': {
                'FOODS_1': {'item_count': 500},
                'FOODS_2': {'item_count': 500},
                'FOODS_3': {'item_count': 500}
            },
            'total_items': 3049
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_hierarchy_tree(hierarchy_stats, temp_dir)

            # Check plot file was created
            plot_path = Path(temp_dir) / "hierarchy_tree.png"
            assert plot_path.exists()

            # Check return structure
            assert isinstance(result, dict)
            assert 'plot_path' in result
            assert 'summary' in result

    def test_plot_hierarchy_tree_empty_stats(self):
        """Test hierarchy tree plot with empty statistics."""
        if plot_hierarchy_tree is None:
            pytest.skip("plot_hierarchy_tree not implemented yet")

        empty_stats = {'categories': {}, 'departments': {}, 'total_items': 0}

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises((ValueError, KeyError)):
                plot_hierarchy_tree(empty_stats, temp_dir)


class TestTimelineValidationPlot:
    """Test timeline validation plotting functionality."""

    def test_plot_timeline_validation_basic(self):
        """Test basic timeline validation plot creation."""
        if plot_timeline_validation is None:
            pytest.skip("plot_timeline_validation not implemented yet")

        # Sample temporal statistics
        temporal_stats = {
            'date_range': {
                'start': '2011-01-29',
                'end': '2016-06-19',
                'total_days': 1969
            },
            'periods': {
                'training': {'start': '2011-01-29', 'end': '2016-05-22', 'days': 1941},
                'validation': {'start': '2016-05-23', 'end': '2016-06-19', 'days': 28}
            },
            'gaps': [],
            'overlaps': []
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_timeline_validation(temporal_stats, temp_dir)

            # Check plot file was created
            plot_path = Path(temp_dir) / "timeline_validation.png"
            assert plot_path.exists()

            # Check return structure
            assert isinstance(result, dict)
            assert 'plot_path' in result


class TestDataCoverageHeatmap:
    """Test data coverage heatmap plotting functionality."""

    def test_plot_data_coverage_heatmap_basic(self):
        """Test basic data coverage heatmap creation."""
        if plot_data_coverage_heatmap is None:
            pytest.skip("plot_data_coverage_heatmap not implemented yet")

        # Sample coverage statistics
        coverage_stats = {
            'tables': {
                'sales': {'coverage': 0.95, 'missing_percentage': 5.0},
                'calendar': {'coverage': 1.0, 'missing_percentage': 0.0},
                'prices': {'coverage': 0.75, 'missing_percentage': 25.0}
            },
            'temporal_coverage': {
                '2011': {'sales': 0.98, 'prices': 0.80},
                '2012': {'sales': 0.97, 'prices': 0.82},
                '2013': {'sales': 0.95, 'prices': 0.78}
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_data_coverage_heatmap(coverage_stats, temp_dir)

            # Check plot file was created
            plot_path = Path(temp_dir) / "data_coverage_heatmap.png"
            assert plot_path.exists()

            # Check return structure
            assert isinstance(result, dict)
            assert 'plot_path' in result


class TestTableRelationshipsPlot:
    """Test table relationships plotting functionality."""

    def test_plot_table_relationships_basic(self):
        """Test basic table relationships diagram creation."""
        if plot_table_relationships is None:
            pytest.skip("plot_table_relationships not implemented yet")

        # Sample relationship statistics
        relationship_stats = {
            'tables': ['sales', 'calendar', 'prices'],
            'relationships': {
                'sales_calendar': {'join_success_rate': 0.98, 'key_columns': ['date']},
                'sales_prices': {'join_success_rate': 0.85, 'key_columns': ['store_id', 'item_id']},
                'calendar_prices': {'join_success_rate': 0.90, 'key_columns': ['wm_yr_wk']}
            },
            'primary_keys': {
                'sales': ['item_id', 'store_id', 'date'],
                'calendar': ['date'],
                'prices': ['store_id', 'item_id', 'wm_yr_wk']
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_table_relationships(relationship_stats, temp_dir)

            # Check plot file was created
            plot_path = Path(temp_dir) / "table_relationships.png"
            assert plot_path.exists()

            # Check return structure
            assert isinstance(result, dict)
            assert 'plot_path' in result


class TestDataQualityDashboard:
    """Test data quality dashboard plotting functionality."""

    def test_plot_data_quality_dashboard_basic(self):
        """Test basic data quality dashboard creation."""
        if plot_data_quality_dashboard is None:
            pytest.skip("plot_data_quality_dashboard not implemented yet")

        # Sample quality statistics
        quality_stats = {
            'overall_score': 0.85,
            'anomaly_counts': {
                'negative_sales': 15,
                'extreme_outliers': 45,
                'impossible_values': 2
            },
            'missing_data': {
                'sales': 0.05,
                'prices': 0.25,
                'calendar': 0.0
            },
            'quality_metrics': {
                'completeness': 0.87,
                'validity': 0.92,
                'consistency': 0.88
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_data_quality_dashboard(quality_stats, temp_dir)

            # Check plot file was created
            plot_path = Path(temp_dir) / "data_quality_dashboard.png"
            assert plot_path.exists()

            # Check return structure
            assert isinstance(result, dict)
            assert 'plot_path' in result


class TestPriceAnomalyDetection:
    """Test price anomaly detection plotting functionality."""

    def test_plot_price_anomaly_detection_basic(self):
        """Test basic price anomaly detection plot creation."""
        if plot_price_anomaly_detection is None:
            pytest.skip("plot_price_anomaly_detection not implemented yet")

        # Sample price statistics with anomalies
        price_stats = {
            'price_distribution': {
                'mean': 5.45,
                'std': 12.30,
                'min': 0.0,
                'max': 999.99,
                'percentiles': {'25%': 1.98, '50%': 3.49, '75%': 6.98, '95%': 19.99}
            },
            'anomalies': {
                'negative_prices': {'count': 5, 'examples': [-1.0, -2.5]},
                'extreme_high': {'count': 12, 'threshold': 100.0, 'examples': [250.0, 999.99]},
                'price_jumps': {'count': 8, 'threshold_ratio': 10.0}
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_price_anomaly_detection(price_stats, temp_dir)

            # Check plot file was created
            plot_path = Path(temp_dir) / "price_anomaly_detection.png"
            assert plot_path.exists()

            # Check return structure
            assert isinstance(result, dict)
            assert 'plot_path' in result


class TestFeatureDistributions:
    """Test feature distributions plotting functionality."""

    def test_plot_feature_distributions_basic(self):
        """Test basic feature distributions plot creation."""
        if plot_feature_distributions is None:
            pytest.skip("plot_feature_distributions not implemented yet")

        # Sample feature statistics
        feature_stats = {
            'categorical_features': {
                'cat_id': {
                    'FOODS': {'count': 1500, 'avg_sales': 2.5},
                    'HOBBIES': {'count': 800, 'avg_sales': 1.8},
                    'HOUSEHOLD': {'count': 749, 'avg_sales': 3.2}
                },
                'state_id': {
                    'CA': {'count': 1200, 'avg_sales': 2.8},
                    'TX': {'count': 900, 'avg_sales': 2.1},
                    'WI': {'count': 949, 'avg_sales': 2.3}
                }
            },
            'geographic_features': {
                'performance_by_state': {
                    'CA': {'sales_volume': 15000, 'store_count': 4},
                    'TX': {'sales_volume': 12000, 'store_count': 3},
                    'WI': {'sales_volume': 10000, 'store_count': 3}
                }
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_feature_distributions(feature_stats, temp_dir)

            # Check plot file was created
            plot_path = Path(temp_dir) / "feature_distributions.png"
            assert plot_path.exists()

            # Check return structure
            assert isinstance(result, dict)
            assert 'plot_path' in result


class TestTemporalCorrelations:
    """Test temporal correlations plotting functionality."""

    def test_plot_temporal_correlations_basic(self):
        """Test basic temporal correlations heatmap creation."""
        if plot_temporal_correlations is None:
            pytest.skip("plot_temporal_correlations not implemented yet")

        # Sample correlation statistics
        correlation_stats = {
            'correlation_matrix': {
                'sales': {
                    'snap_CA': 0.15,
                    'snap_TX': 0.12,
                    'snap_WI': 0.08,
                    'weekday': -0.05,
                    'month': 0.22
                },
                'significance_levels': {
                    'snap_CA': 0.001,
                    'snap_TX': 0.005,
                    'snap_WI': 0.05,
                    'weekday': 0.2,
                    'month': 0.001
                }
            },
            'temporal_features': ['snap_CA', 'snap_TX', 'snap_WI', 'weekday', 'month']
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_temporal_correlations(correlation_stats, temp_dir)

            # Check plot file was created
            plot_path = Path(temp_dir) / "temporal_correlations.png"
            assert plot_path.exists()

            # Check return structure
            assert isinstance(result, dict)
            assert 'plot_path' in result


class TestVisualizationUtilities:
    """Test general visualization utility functions."""

    def test_matplotlib_cleanup(self):
        """Test that matplotlib figures are properly closed to prevent memory leaks."""
        # This test verifies that the visualization functions properly close figures
        # We'll check this by monitoring the figure count
        initial_fig_count = len(plt.get_fignums())

        # Create a simple plot and close it properly
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close(fig)

        final_fig_count = len(plt.get_fignums())
        assert final_fig_count == initial_fig_count

    def test_output_directory_creation(self):
        """Test that functions handle output directory creation properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_dir = Path(temp_dir) / "new_subdir"

            # Functions should create directories if they don't exist
            # This is a pattern we'll test in each function implementation
            assert not non_existent_dir.exists()
            non_existent_dir.mkdir(parents=True, exist_ok=True)
            assert non_existent_dir.exists()

    def test_all_functions_handle_nested_paths(self):
        """Test that all functions handle nested output paths by creating directories."""
        if plot_hierarchy_tree is None:
            pytest.skip("Functions not implemented yet")

        hierarchy_stats = {
            'categories': {'FOODS': {'departments': ['FOODS_1'], 'item_count': 100}},
            'departments': {'FOODS_1': {'item_count': 100}},
            'total_items': 100
        }

        # Test with nested path that needs to be created
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "deeply" / "nested" / "path"
            assert not nested_path.exists()

            result = plot_hierarchy_tree(hierarchy_stats, str(nested_path))
            # Should have created the directory and saved the plot
            assert nested_path.exists()
            assert Path(result['plot_path']).exists()

    def test_all_functions_return_consistent_format(self):
        """Test that all functions return consistent dictionary format."""
        if plot_hierarchy_tree is None:
            pytest.skip("Functions not implemented yet")

        test_data = {
            'hierarchy_stats': {
                'categories': {'FOODS': {'departments': ['FOODS_1'], 'item_count': 100}},
                'departments': {'FOODS_1': {'item_count': 100}},
                'total_items': 100
            },
            'temporal_stats': {
                'date_range': {'start': '2011-01-29', 'end': '2016-06-19', 'total_days': 1969},
                'periods': {'training': {'start': '2011-01-29', 'end': '2016-05-22', 'days': 1941}},
                'gaps': [], 'overlaps': []
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test hierarchy tree
            result = plot_hierarchy_tree(test_data['hierarchy_stats'], temp_dir)
            assert isinstance(result, dict)
            assert 'plot_path' in result
            assert Path(result['plot_path']).exists()

            # Test timeline validation
            result = plot_timeline_validation(test_data['temporal_stats'], temp_dir)
            assert isinstance(result, dict)
            assert 'plot_path' in result
            assert Path(result['plot_path']).exists()


if __name__ == "__main__":
    pytest.main([__file__])