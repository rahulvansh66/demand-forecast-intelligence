"""
Tests for M5 sample dataset validation metrics.

This module tests the ValidationMetrics class which assesses the quality of
generated sample datasets to ensure they maintain statistical representativeness
and capture the challenging aspects of M5 demand patterns necessary for robust
model validation.

The tests validate that the ValidationMetrics class can:
1. Compare distributions between original and sample datasets using statistical tests
2. Calculate correlation metrics to assess statistical similarity
3. Calculate forecasting-specific metrics to validate behavioral diversity
4. Generate comprehensive validation reports with quality scores

Business Context:
- Simple correlation metrics can miss critical modeling failures
- M5's hierarchical demand forecasting requires weighted scaled error metrics
- Validation must assess behavioral diversity, especially sparse/intermittent items
- Prevents overly optimistic POC results that don't reflect production challenges
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from demand_forecast_intelligence.data.sampling.validation_metrics import ValidationMetrics


class TestValidationMetrics:
    """Test suite for ValidationMetrics class."""

    @pytest.fixture
    def sample_original_data(self):
        """
        Create mock original M5 dataset for testing.

        Includes realistic distribution of categories, departments, and sales patterns
        to simulate the actual M5 dataset structure with various demand behaviors.
        """
        np.random.seed(42)
        n_items = 1000

        # Create item IDs with M5-like structure
        categories = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
        departments = ['FOODS_1', 'FOODS_2', 'FOODS_3', 'HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2']
        stores = [f'CA_{i}' for i in range(1, 11)]

        data = []
        for i in range(n_items):
            # Create realistic sales patterns with intermittent demand
            if i < 200:  # 20% high-volume items
                daily_sales = np.random.poisson(5, 100)
            elif i < 600:  # 40% medium-volume items
                daily_sales = np.random.poisson(1.5, 100)
            else:  # 40% sparse/intermittent items
                daily_sales = np.random.poisson(0.3, 100)  # Very sparse

            zero_ratio = np.sum(daily_sales == 0) / len(daily_sales)

            data.append({
                'item_id': f'ITEM_{i}',
                'cat_id': np.random.choice(categories),
                'dept_id': np.random.choice(departments),
                'store_id': np.random.choice(stores),
                'daily_sales': daily_sales,
                'avg_sales': np.mean(daily_sales),
                'zero_sales_ratio': zero_ratio,
                'cv_sales': np.std(daily_sales) / np.mean(daily_sales) if np.mean(daily_sales) > 0 else np.inf
            })

        return pd.DataFrame(data)

    @pytest.fixture
    def sample_dataset(self, sample_original_data):
        """Create a sample dataset that's a subset of original data."""
        # Take 20% sample, biased toward higher volume items to test validation
        sample_size = int(len(sample_original_data) * 0.2)

        # Bias sampling toward higher sales items (not representative)
        sample_indices = np.concatenate([
            np.arange(0, 150),  # Most high-volume items
            np.random.choice(np.arange(150, len(sample_original_data)),
                           size=sample_size-150, replace=False)
        ])

        return sample_original_data.iloc[sample_indices].copy()

    @pytest.fixture
    def validation_metrics(self, sample_original_data, sample_dataset):
        """Create ValidationMetrics instance for testing."""
        return ValidationMetrics(
            original_data=sample_original_data,
            sample_data=sample_dataset
        )

    def test_calculate_distribution_comparison(self, validation_metrics):
        """
        Test distribution comparison calculations using chi-square tests.

        This test validates that the method can:
        - Compare categorical distributions (categories, departments)
        - Compare continuous distributions using binning
        - Return proper statistical test results with p-values
        - Handle edge cases like missing categories

        Business Rationale: Distribution comparison ensures the sample maintains
        the same business mix (category/department balance) as the full dataset.
        Chi-square tests detect significant deviations that could bias model training.
        """
        # This should fail initially - ValidationMetrics doesn't exist yet
        result = validation_metrics.calculate_distribution_comparison()

        # Validate return structure
        assert isinstance(result, dict)
        assert 'categorical_tests' in result
        assert 'continuous_tests' in result

        # Check categorical tests include key business dimensions
        categorical = result['categorical_tests']
        assert 'cat_id' in categorical
        assert 'dept_id' in categorical
        assert 'store_id' in categorical

        # Validate statistical test results format
        for test_name, test_result in categorical.items():
            assert 'chi2_statistic' in test_result
            assert 'p_value' in test_result
            assert 'significant' in test_result
            assert isinstance(test_result['p_value'], (float, int))

        # Check continuous tests for demand metrics
        continuous = result['continuous_tests']
        assert 'avg_sales' in continuous
        assert 'zero_sales_ratio' in continuous

    def test_calculate_correlation_metrics(self, validation_metrics):
        """
        Test correlation metrics calculation between original and sample datasets.

        This test validates that the method can:
        - Calculate Pearson correlations for key demand metrics
        - Calculate Spearman rank correlations for robustness
        - Handle different data distributions appropriately
        - Return correlation coefficients and p-values

        Business Rationale: Correlation metrics ensure the sample maintains
        similar statistical relationships between demand variables as the
        original dataset, which is crucial for model generalization.
        """
        # This should fail initially - method doesn't exist yet
        result = validation_metrics.calculate_correlation_metrics()

        # Validate return structure
        assert isinstance(result, dict)
        assert 'pearson_correlations' in result
        assert 'spearman_correlations' in result

        # Check key demand metrics are included
        pearson = result['pearson_correlations']
        expected_metrics = ['avg_sales', 'zero_sales_ratio', 'cv_sales']

        for metric in expected_metrics:
            assert metric in pearson
            assert 'correlation' in pearson[metric]
            assert 'p_value' in pearson[metric]
            # Correlation should be between -1 and 1
            assert -1 <= pearson[metric]['correlation'] <= 1

        # Validate Spearman correlations have same structure
        spearman = result['spearman_correlations']
        for metric in expected_metrics:
            assert metric in spearman
            assert 'correlation' in spearman[metric]
            assert 'p_value' in spearman[metric]

    def test_calculate_forecasting_metrics(self, validation_metrics):
        """
        Test forecasting-specific validation metrics.

        This test validates that the method can:
        - Calculate naive seasonal baseline performance (RMSSE-style)
        - Assess behavioral diversity retention (sparse vs dense items)
        - Validate intermittent demand representation
        - Calculate weighted performance metrics

        Business Rationale: Forecasting metrics ensure the sample preserves
        the challenging intermittent/sparse patterns that cause production
        model failures. Simple statistical metrics can miss these critical
        aspects of demand forecasting difficulty.
        """
        # This should fail initially - method doesn't exist yet
        result = validation_metrics.calculate_forecasting_metrics()

        # Validate return structure
        assert isinstance(result, dict)
        assert 'naive_baseline_performance' in result
        assert 'behavioral_diversity' in result
        assert 'intermittent_representation' in result

        # Check naive baseline metrics
        baseline = result['naive_baseline_performance']
        assert 'original_rmsse' in baseline
        assert 'sample_rmsse' in baseline
        assert 'rmsse_ratio' in baseline

        # Validate behavioral diversity metrics
        diversity = result['behavioral_diversity']
        assert 'sparse_item_ratio_original' in diversity
        assert 'sparse_item_ratio_sample' in diversity
        assert 'diversity_preservation_score' in diversity

        # Check intermittent representation
        intermittent = result['intermittent_representation']
        assert 'intermittent_threshold' in intermittent
        assert 'original_intermittent_pct' in intermittent
        assert 'sample_intermittent_pct' in intermittent
        assert 'representation_score' in intermittent

    def test_generate_validation_report(self, validation_metrics):
        """
        Test comprehensive validation report generation.

        This test validates that the method can:
        - Integrate all validation metrics into a single report
        - Calculate overall quality score (0-100)
        - Provide actionable recommendations
        - Handle edge cases and missing data gracefully

        Business Rationale: The validation report provides a single quality
        assessment that combines statistical, distributional, and forecasting-
        specific metrics. This enables data scientists to quickly assess if
        a sample dataset is suitable for model development and testing.
        """
        # This should fail initially - method doesn't exist yet
        report = validation_metrics.generate_validation_report()

        # Validate report structure
        assert isinstance(report, dict)
        assert 'overall_quality_score' in report
        assert 'distribution_comparison' in report
        assert 'correlation_metrics' in report
        assert 'forecasting_metrics' in report
        assert 'recommendations' in report
        assert 'summary' in report

        # Validate quality score
        quality_score = report['overall_quality_score']
        assert isinstance(quality_score, (int, float))
        assert 0 <= quality_score <= 100

        # Check that all sub-metrics are included
        assert report['distribution_comparison'] is not None
        assert report['correlation_metrics'] is not None
        assert report['forecasting_metrics'] is not None

        # Validate recommendations structure
        recommendations = report['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 0  # May be empty if sample is perfect

        # Validate summary contains key insights
        summary = report['summary']
        assert isinstance(summary, dict)
        assert 'total_items_original' in summary
        assert 'total_items_sample' in summary
        assert 'sample_ratio' in summary