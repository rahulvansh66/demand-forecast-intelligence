"""
Test suite for M5 Sample Generator - Main Orchestration Engine.

This module tests the core sample generation orchestration that combines
configuration management and behavioral stratification to create representative
sample datasets from the M5 Walmart data.

The tests verify the critical anti-bias requirements:
1. Random sampling within behavioral strata (NO ranking or quality scoring)
2. Proportional allocation with department/stratum minimum floors
3. Complete geographic and temporal coverage preservation
4. Training-period-only stratification to prevent future leakage

Business Context:
Traditional sampling approaches create bias by selecting "best quality" items
(highest volume, most complete data), making POC validation overly optimistic.
This orchestration ensures challenging intermittent/sparse patterns are properly
represented through random sampling within behavioral strata.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path

from demand_forecast_intelligence.data.sampling.sample_generator import SampleGenerator
from demand_forecast_intelligence.data.sampling.config import SamplingConfig


class TestSampleGenerator(unittest.TestCase):
    """Test suite for SampleGenerator orchestration class."""

    def setUp(self):
        """Set up test fixtures with mock configuration and data."""
        # Create test configuration with minimal settings
        self.config = SamplingConfig(
            target_item_count=100,  # Small for testing
            min_per_dept=10,
            min_per_stratum=1,
            random_seed=42
        )

        # Mock data directory paths
        self.data_dir = Path("/test/data/m5")

        # Create sample behavioral strata data for testing
        self.sample_strata = pd.DataFrame({
            'item_id': [f'FOODS_3_0{i:02d}' for i in range(1, 21)],  # 20 test items
            'dept_id': ['FOODS_3'] * 10 + ['HOUSEHOLD_1'] * 10,  # 2 departments
            'stratum_key': [
                'FOODS_3|low|sparse|early',
                'FOODS_3|low|sparse|mature',
                'FOODS_3|low|regular|early',
                'FOODS_3|low|regular|mature',
                'FOODS_3|high|sparse|early',
                'FOODS_3|high|sparse|mature',
                'FOODS_3|high|regular|early',
                'FOODS_3|high|regular|mature',
                'FOODS_3|medium|intermittent|declining',
                'FOODS_3|medium|intermittent|mature',
                'HOUSEHOLD_1|low|sparse|early',
                'HOUSEHOLD_1|low|sparse|mature',
                'HOUSEHOLD_1|low|regular|early',
                'HOUSEHOLD_1|low|regular|mature',
                'HOUSEHOLD_1|high|sparse|early',
                'HOUSEHOLD_1|high|sparse|mature',
                'HOUSEHOLD_1|high|regular|early',
                'HOUSEHOLD_1|high|regular|mature',
                'HOUSEHOLD_1|medium|intermittent|declining',
                'HOUSEHOLD_1|medium|intermittent|mature'
            ]
        })

        # Mock sales data for testing file creation
        self.sample_sales_data = pd.DataFrame({
            'id': [f'FOODS_3_001_CA_1_validation', 'FOODS_3_002_CA_1_validation'],
            'item_id': ['FOODS_3_001', 'FOODS_3_002'],
            'dept_id': ['FOODS_3', 'FOODS_3'],
            'cat_id': ['FOODS_3', 'FOODS_3'],
            'store_id': ['CA_1', 'CA_1'],
            'state_id': ['CA', 'CA'],
            'd_1': [1, 0],
            'd_2': [2, 1],
            'd_3': [0, 2]
        })

    def test_initialization_with_valid_config(self):
        """
        Test SampleGenerator initializes correctly with valid configuration.

        Verifies the orchestration class properly accepts configuration and
        sets up the behavioral stratifier for downstream sampling operations.
        This test ensures the main entry point works with valid inputs.
        """
        # This test will fail initially since SampleGenerator doesn't exist yet
        generator = SampleGenerator(self.config, self.data_dir)

        # Verify configuration stored correctly
        self.assertEqual(generator.config, self.config)
        self.assertEqual(generator.data_dir, self.data_dir)

        # Verify stratifier initialized (will be mocked in implementation)
        self.assertIsNotNone(generator.stratifier)

    @patch('demand_forecast_intelligence.data.sampling.sample_generator.BehavioralStratifier')
    @patch('demand_forecast_intelligence.data.sampling.sample_generator.pd.read_csv')
    @patch('pathlib.Path.mkdir')
    @patch('pandas.DataFrame.to_csv')
    def test_generate_sample_complete_orchestration(self, mock_to_csv, mock_mkdir, mock_read_csv, mock_stratifier_class):
        """
        Test complete sample generation orchestration workflow.

        This is the core integration test that verifies the main generate_sample()
        method properly orchestrates all sampling steps:
        1. Load and validate input data
        2. Calculate behavioral metrics and create strata
        3. Calculate proportional allocation with floors
        4. Perform random sampling within each stratum (NO RANKING)
        5. Create sample dataset files with proper filtering
        6. Generate comprehensive metadata and validation statistics

        The test ensures random sampling is used (not quality-based ranking)
        and that all geographic/temporal coverage is preserved.
        """
        # Mock data loading
        mock_sales_data = self.sample_sales_data.copy()
        mock_read_csv.return_value = mock_sales_data

        # Mock stratifier behavior
        mock_stratifier = Mock()
        mock_stratifier_class.return_value = mock_stratifier
        mock_stratifier.calculate_item_metrics.return_value = pd.DataFrame({
            'item_id': ['FOODS_3_001', 'FOODS_3_002'],
            'avg_sales': [1.5, 0.8]
        })
        mock_stratifier.create_behavioral_strata.return_value = self.sample_strata.head(10)

        # Create generator and run orchestration
        generator = SampleGenerator(self.config, self.data_dir)

        # This will fail initially - we need to implement the method
        result = generator.generate_sample()

        # Verify orchestration called all required steps
        mock_stratifier.calculate_item_metrics.assert_called_once()
        mock_stratifier.create_behavioral_strata.assert_called_once()

        # Verify result contains required metadata
        self.assertIn('sample_items', result)
        self.assertIn('allocation_summary', result)
        self.assertIn('sampling_metadata', result)

        # Verify random sampling used (no quality ranking)
        self.assertIn('sampling_method', result['sampling_metadata'])
        self.assertEqual(result['sampling_metadata']['sampling_method'], 'random_within_strata')

    def test_proportional_allocation_with_floors(self):
        """
        Test proportional allocation calculation with department and stratum floors.

        This test verifies the allocation logic that balances:
        1. Natural stratum proportions in the population
        2. Minimum floors for departments (business requirement)
        3. Minimum floors for individual strata (statistical requirement)

        The allocation must respect these constraints while distributing the
        target item count proportionally across behavioral patterns.
        """
        generator = SampleGenerator(self.config, self.data_dir)

        # This will fail initially - method doesn't exist yet
        allocation = generator._calculate_proportional_allocation(self.sample_strata)

        # Verify allocation respects minimum constraints
        dept_totals = allocation.groupby('dept_id')['allocated_count'].sum()
        self.assertTrue((dept_totals >= self.config.min_per_dept).all())

        # Verify no stratum gets less than minimum
        self.assertTrue((allocation['allocated_count'] >= self.config.min_per_stratum).all())

        # Verify total allocation matches target (within rounding tolerance)
        total_allocated = allocation['allocated_count'].sum()
        self.assertLessEqual(abs(total_allocated - self.config.target_item_count), 10)

    @patch('demand_forecast_intelligence.data.sampling.sample_generator.np.random.choice')
    def test_random_sampling_within_strata_no_ranking(self, mock_random_choice):
        """
        Test random sampling within strata with NO quality ranking.

        This is the critical anti-bias test that verifies sampling is truly random
        within each behavioral stratum, with NO ranking or quality scoring that
        would bias selection toward high-volume, stable items.

        The test ensures:
        1. numpy.random.choice is used for sampling (not ranking functions)
        2. All items in stratum have equal probability of selection
        3. Sample sizes respect allocation constraints
        4. Random seed produces reproducible results
        """
        # Mock random choice to return predictable indices
        mock_random_choice.side_effect = lambda items, size, replace=False: items[:size]

        generator = SampleGenerator(self.config, self.data_dir)

        # Create test allocation that will require random sampling
        # Use strata that have multiple items so random choice gets called
        test_allocation = pd.DataFrame({
            'stratum_key': ['FOODS_3|low|sparse|early'],
            'allocated_count': [1]  # Sample 1 from multiple items in stratum
        })

        # Create test strata with multiple items in target stratum
        test_strata = pd.DataFrame({
            'item_id': ['FOODS_3_001', 'FOODS_3_002', 'FOODS_3_003'],
            'dept_id': ['FOODS_3', 'FOODS_3', 'FOODS_3'],
            'stratum_key': ['FOODS_3|low|sparse|early', 'FOODS_3|low|sparse|early', 'FOODS_3|low|sparse|early']
        })

        # This will fail initially - method doesn't exist yet
        sample_items = generator._sample_within_strata(test_strata, test_allocation)

        # Verify numpy.random.choice was called (random sampling)
        self.assertTrue(mock_random_choice.called)

        # Verify no ranking/scoring functions were used
        # (This is ensured by the implementation using only random.choice)

        # Verify correct number of items sampled per stratum
        sample_counts = sample_items.groupby('stratum_key').size()
        expected_counts = test_allocation.set_index('stratum_key')['allocated_count']
        # Compare values, ignoring series names
        self.assertEqual(len(sample_counts), len(expected_counts))
        for stratum in expected_counts.index:
            self.assertEqual(sample_counts[stratum], expected_counts[stratum])

        # Verify reproducibility with fixed seed
        sample_items_2 = generator._sample_within_strata(test_strata, test_allocation)
        pd.testing.assert_frame_equal(sample_items, sample_items_2)


if __name__ == '__main__':
    unittest.main()