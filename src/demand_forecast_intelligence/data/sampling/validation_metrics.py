"""
M5 Sample Dataset Validation Metrics.

This module provides comprehensive validation metrics for assessing the quality of
generated sample datasets to ensure they maintain statistical representativeness
and capture the challenging aspects of M5 demand patterns necessary for robust
model validation.

Business Context:
The ValidationMetrics class goes beyond simple distribution correlations to include
forecasting-specific metrics. M5's hierarchical demand forecasting requires weighted
scaled error metrics (like RMSSE) that can expose issues missed by correlation analysis.

Critical Design Principle:
The validation must assess behavioral diversity - ensuring sparse/intermittent items
are represented proportionally, not just high-volume items. This prevents overly
optimistic POC results that don't reflect production challenges.

Key Components:
1. Distribution Comparison - Chi-square tests for categorical representativeness
2. Correlation Metrics - Statistical correlation analysis between datasets
3. Forecasting Metrics - Behavioral diversity and RMSSE-style validation
4. Quality Scoring - Overall assessment combining all metrics (0-100 scale)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings
from dataclasses import dataclass

from demand_forecast_intelligence.core.logging.logger import get_logger

logger = get_logger(__name__)


def _chi2_contingency(observed):
    """
    Simple chi-square test implementation without scipy.

    Parameters
    ----------
    observed : array-like
        Contingency table

    Returns
    -------
    tuple
        (chi2_statistic, p_value, dof, expected)
    """
    observed = np.array(observed)

    # Calculate expected frequencies
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()

    expected = np.outer(row_totals, col_totals) / total

    # Avoid division by zero
    expected = np.where(expected == 0, 1e-10, expected)

    # Calculate chi-square statistic
    chi2_stat = np.sum((observed - expected) ** 2 / expected)

    # Degrees of freedom
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)

    # Approximate p-value using chi-square distribution
    # For simplicity, we'll use a rough approximation
    # In practice, you'd use the chi-square CDF
    if dof > 0:
        # Very rough approximation - in real implementation would use proper chi2 CDF
        p_value = np.exp(-chi2_stat / (2 * dof)) if chi2_stat > dof else 0.5
    else:
        p_value = 1.0

    return chi2_stat, p_value, dof, expected


def _pearsonr(x, y):
    """
    Calculate Pearson correlation coefficient without scipy.

    Parameters
    ----------
    x, y : array-like
        Input arrays

    Returns
    -------
    tuple
        (correlation, p_value)
    """
    x = np.array(x)
    y = np.array(y)

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return 0.0, 1.0

    # Calculate correlation
    correlation = np.corrcoef(x, y)[0, 1]

    # Simple p-value approximation
    # In practice, you'd use the t-distribution
    n = len(x)
    if n > 2 and not np.isnan(correlation):
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation ** 2 + 1e-10))
        # Very rough p-value approximation
        p_value = 2 * (1 - np.tanh(abs(t_stat) / 2))
    else:
        p_value = 1.0

    return correlation if not np.isnan(correlation) else 0.0, p_value


def _spearmanr(x, y):
    """
    Calculate Spearman rank correlation without scipy.

    Parameters
    ----------
    x, y : array-like
        Input arrays

    Returns
    -------
    tuple
        (correlation, p_value)
    """
    x = np.array(x)
    y = np.array(y)

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return 0.0, 1.0

    # Convert to ranks
    x_ranks = np.argsort(np.argsort(x))
    y_ranks = np.argsort(np.argsort(y))

    # Calculate Pearson correlation of ranks
    return _pearsonr(x_ranks, y_ranks)


def _ks_2samp(x, y):
    """
    Two-sample Kolmogorov-Smirnov test without scipy.

    Computes the KS statistic as the maximum absolute difference between the
    two empirical CDFs evaluated at every observed value in the combined
    sample, then derives a p-value via the standard asymptotic approximation
    of the Kolmogorov distribution.

    Parameters
    ----------
    x, y : array-like
        The two independent samples to compare. NaN values are dropped.

    Returns
    -------
    tuple
        (ks_statistic, p_value)
        ks_statistic : float in [0, 1] — 0 means identical distributions.
        p_value      : float in [0, 1] — small value means distributions differ.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Drop NaNs
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) == 0 or len(y) == 0:
        return 0.0, 1.0

    x = np.sort(x)
    y = np.sort(y)

    # Evaluate both empirical CDFs at every point in the pooled sample
    all_vals = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, all_vals, side='right') / len(x)
    cdf_y = np.searchsorted(y, all_vals, side='right') / len(y)

    d_stat = float(np.max(np.abs(cdf_x - cdf_y)))

    # Asymptotic p-value: P(D > d) ≈ 2·exp(-2·n_eff·d²)
    # where n_eff = n1*n2 / (n1+n2) is the effective sample size
    n_eff = len(x) * len(y) / (len(x) + len(y))
    p_value = float(np.clip(2.0 * np.exp(-2.0 * n_eff * d_stat ** 2), 0.0, 1.0))

    return d_stat, p_value


@dataclass
class ValidationResult:
    """Container for validation metric results."""
    metric_name: str
    value: float
    threshold: float
    passed: bool
    details: Dict[str, Any]


class ValidationMetrics:
    """
    Comprehensive validation metrics for M5 sample dataset quality assessment.

    This class validates that generated sample datasets maintain statistical
    representativeness and capture the challenging aspects of M5 demand patterns
    necessary for robust model validation.

    The validation process includes:
    1. Distribution comparison using chi-square tests
    2. Correlation analysis between original and sample datasets
    3. Forecasting-specific metrics for behavioral diversity
    4. Overall quality scoring and recommendations

    Business Rationale:
    Simple correlation metrics can miss critical modeling failures. M5's hierarchical
    demand forecasting requires validation that ensures sparse/intermittent items
    are properly represented, as these are often the most challenging to forecast
    and critical for production model performance.

    Parameters
    ----------
    original_data : pd.DataFrame
        The complete original M5 dataset with demand patterns
    sample_data : pd.DataFrame
        The generated sample dataset to validate
    intermittent_threshold : float, default=0.8
        Threshold for classifying items as intermittent (>80% zero sales days)
    significance_level : float, default=0.05
        Statistical significance level for hypothesis tests
    """

    def __init__(
        self,
        original_data: pd.DataFrame,
        sample_data: pd.DataFrame,
        intermittent_threshold: float = 0.8,
        significance_level: float = 0.05
    ):
        """Initialize ValidationMetrics with original and sample datasets."""
        self.original_data = original_data.copy()
        self.sample_data = sample_data.copy()
        self.intermittent_threshold = intermittent_threshold
        self.significance_level = significance_level

        # Validate input data structure
        self._validate_input_data()

        logger.info(
            f"ValidationMetrics initialized - Original: {len(self.original_data)} items, "
            f"Sample: {len(self.sample_data)} items "
            f"({len(self.sample_data)/len(self.original_data)*100:.1f}% sample ratio)"
        )

    def _validate_input_data(self) -> None:
        """
        Validate that input datasets have required columns and structure.

        Raises
        ------
        ValueError
            If required columns are missing or data structure is invalid
        """
        required_columns = ['cat_id', 'dept_id', 'store_id', 'avg_sales', 'zero_sales_ratio']

        for col in required_columns:
            if col not in self.original_data.columns:
                raise ValueError(f"Original data missing required column: {col}")
            if col not in self.sample_data.columns:
                raise ValueError(f"Sample data missing required column: {col}")

        if len(self.original_data) == 0:
            raise ValueError("Original data cannot be empty")
        if len(self.sample_data) == 0:
            raise ValueError("Sample data cannot be empty")

        logger.debug("Input data validation passed")

    def calculate_distribution_comparison(self) -> Dict[str, Any]:
        """
        Compare distributions between original and sample datasets using statistical tests.

        This method performs chi-square tests for categorical variables (categories,
        departments, stores) and binned distributions for continuous variables
        (sales metrics) to assess whether the sample maintains representative
        distributions from the original dataset.

        Business Rationale:
        Distribution comparison ensures the sample maintains the same business mix
        (category/department balance) as the full dataset. Significant deviations
        could bias model training toward certain product types or store formats.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - categorical_tests: Chi-square test results for categorical variables
            - continuous_tests: Binned distribution tests for continuous variables
            - overall_distribution_score: Combined score (0-100)

        Notes
        -----
        Uses chi-square test for independence to compare categorical distributions.
        For continuous variables, the two-sample Kolmogorov-Smirnov test is used
        directly on the raw values — no binning required. The KS test is sensitive
        to any difference in the distribution (location, scale, or shape) and avoids
        the information loss introduced by arbitrary bin boundaries.
        """
        logger.info("Calculating distribution comparison metrics")

        categorical_tests = {}
        continuous_tests = {}

        # Test categorical distributions
        categorical_vars = ['cat_id', 'dept_id', 'store_id']

        for var in categorical_vars:
            try:
                # Create contingency table
                original_counts = self.original_data[var].value_counts()
                sample_counts = self.sample_data[var].value_counts()

                # Align categories (handle missing categories in sample)
                all_categories = original_counts.index.union(sample_counts.index)
                original_aligned = original_counts.reindex(all_categories, fill_value=0)
                sample_aligned = sample_counts.reindex(all_categories, fill_value=0)

                # Create contingency table
                contingency_table = np.array([original_aligned.values, sample_aligned.values])

                # Perform chi-square test
                chi2_stat, p_value, dof, expected = _chi2_contingency(contingency_table)

                categorical_tests[var] = {
                    'chi2_statistic': float(chi2_stat),
                    'p_value': float(p_value),
                    'degrees_of_freedom': int(dof),
                    'significant': p_value < self.significance_level,
                    'categories_original': len(original_counts),
                    'categories_sample': len(sample_counts)
                }

                logger.debug(f"Chi-square test for {var}: χ²={chi2_stat:.3f}, p={p_value:.4f}")

            except Exception as e:
                logger.warning(f"Failed to compute chi-square test for {var}: {e}")
                categorical_tests[var] = {
                    'chi2_statistic': np.nan,
                    'p_value': np.nan,
                    'significant': True,  # Conservative - assume significant difference
                    'error': str(e)
                }

        # Test continuous distributions using two-sample KS test
        # KS test is preferred over binned chi-square because it is non-parametric,
        # does not require binning choices, and is sensitive to differences anywhere
        # in the distribution (location, scale, shape).
        continuous_vars = ['avg_sales', 'zero_sales_ratio']

        for var in continuous_vars:
            if var in self.original_data.columns and var in self.sample_data.columns:
                try:
                    original_values = self.original_data[var].replace([np.inf, -np.inf], np.nan).dropna()
                    sample_values = self.sample_data[var].replace([np.inf, -np.inf], np.nan).dropna()

                    if len(original_values) == 0 or len(sample_values) == 0:
                        continue

                    ks_stat, p_value = _ks_2samp(original_values.values, sample_values.values)

                    continuous_tests[var] = {
                        'ks_statistic': float(ks_stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.significance_level,
                        'original_n': int(len(original_values)),
                        'sample_n': int(len(sample_values)),
                        'original_range': [float(original_values.min()), float(original_values.max())],
                        'sample_range': [float(sample_values.min()), float(sample_values.max())]
                    }

                    logger.debug(f"KS test for {var}: D={ks_stat:.4f}, p={p_value:.4f}")

                except Exception as e:
                    logger.warning(f"Failed to compute KS test for {var}: {e}")
                    continuous_tests[var] = {
                        'ks_statistic': np.nan,
                        'p_value': np.nan,
                        'significant': True,
                        'error': str(e)
                    }

        # Calculate overall distribution score
        # Score based on how many tests pass (non-significant differences are good)
        total_tests = len(categorical_tests) + len(continuous_tests)
        passing_tests = sum([
            not test_result.get('significant', True)
            for test_result in list(categorical_tests.values()) + list(continuous_tests.values())
            if not np.isnan(test_result.get('p_value', np.nan))
        ])

        overall_score = (passing_tests / total_tests * 100) if total_tests > 0 else 0

        return {
            'categorical_tests': categorical_tests,
            'continuous_tests': continuous_tests,
            'overall_distribution_score': float(overall_score),
            'total_tests': total_tests,
            'passing_tests': passing_tests
        }

    def calculate_correlation_metrics(self) -> Dict[str, Any]:
        """
        Calculate correlation metrics between original and sample datasets.

        This method computes both Pearson and Spearman correlations for key demand
        metrics to assess whether the sample maintains similar statistical
        relationships between variables as the original dataset.

        Business Rationale:
        Correlation metrics ensure the sample maintains similar statistical
        relationships between demand variables as the original dataset, which is
        crucial for model generalization. Both Pearson (linear) and Spearman
        (rank-based) correlations are calculated for robustness.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - pearson_correlations: Linear correlation coefficients and p-values
            - spearman_correlations: Rank correlation coefficients and p-values
            - overall_correlation_score: Combined correlation quality score (0-100)

        Notes
        -----
        Correlations are calculated by aggregating sample data to match original
        data structure. Missing values are handled using pairwise deletion.
        """
        logger.info("Calculating correlation metrics")

        # Metrics to compare correlations for
        numeric_columns = ['avg_sales', 'zero_sales_ratio', 'cv_sales']

        # Only use columns that exist in both datasets
        available_columns = [col for col in numeric_columns
                           if col in self.original_data.columns and col in self.sample_data.columns]

        pearson_correlations = {}
        spearman_correlations = {}

        for col in available_columns:
            try:
                # Get clean data (no NaN, no infinite values)
                original_values = self.original_data[col].replace([np.inf, -np.inf], np.nan).dropna()
                sample_values = self.sample_data[col].replace([np.inf, -np.inf], np.nan).dropna()

                if len(original_values) < 3 or len(sample_values) < 3:
                    logger.warning(f"Insufficient data for correlation analysis of {col}")
                    continue

                # For correlation analysis, we need to compare distributions
                # Create matched samples by quantile comparison
                n_points = min(len(original_values), len(sample_values))
                if n_points < 10:
                    continue

                # Sample equal number of points from each dataset
                original_sample = original_values.sample(n=n_points, random_state=42).sort_values()
                sample_sample = sample_values.sample(n=n_points, random_state=42).sort_values()

                # Calculate Pearson correlation
                pearson_corr, pearson_p = _pearsonr(original_sample, sample_sample)
                pearson_correlations[col] = {
                    'correlation': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
                    'p_value': float(pearson_p) if not np.isnan(pearson_p) else 1.0,
                    'significant': pearson_p < self.significance_level if not np.isnan(pearson_p) else False,
                    'sample_size': n_points
                }

                # Calculate Spearman correlation
                spearman_corr, spearman_p = _spearmanr(original_sample, sample_sample)
                spearman_correlations[col] = {
                    'correlation': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
                    'p_value': float(spearman_p) if not np.isnan(spearman_p) else 1.0,
                    'significant': spearman_p < self.significance_level if not np.isnan(spearman_p) else False,
                    'sample_size': n_points
                }

                logger.debug(f"Correlations for {col}: Pearson={pearson_corr:.3f}, Spearman={spearman_corr:.3f}")

            except Exception as e:
                logger.warning(f"Failed to compute correlations for {col}: {e}")
                pearson_correlations[col] = {'correlation': 0.0, 'p_value': 1.0, 'significant': False, 'error': str(e)}
                spearman_correlations[col] = {'correlation': 0.0, 'p_value': 1.0, 'significant': False, 'error': str(e)}

        # Calculate overall correlation score
        # Average of absolute correlations (higher is better)
        all_correlations = []
        for col in available_columns:
            if col in pearson_correlations and 'error' not in pearson_correlations[col]:
                all_correlations.append(abs(pearson_correlations[col]['correlation']))
            if col in spearman_correlations and 'error' not in spearman_correlations[col]:
                all_correlations.append(abs(spearman_correlations[col]['correlation']))

        overall_score = np.mean(all_correlations) * 100 if all_correlations else 0

        return {
            'pearson_correlations': pearson_correlations,
            'spearman_correlations': spearman_correlations,
            'overall_correlation_score': float(overall_score),
            'metrics_analyzed': available_columns
        }

    def calculate_forecasting_metrics(self) -> Dict[str, Any]:
        """
        Calculate forecasting-specific validation metrics.

        This method assesses behavioral diversity retention and calculates
        RMSSE-style metrics to validate that the sample preserves the challenging
        intermittent/sparse patterns that are critical for forecasting model
        validation.

        Business Rationale:
        Forecasting metrics ensure the sample preserves the challenging
        intermittent/sparse patterns that cause production model failures.
        Simple statistical metrics can miss these critical aspects of demand
        forecasting difficulty. This validation uses RMSSE-style weighted
        metrics that mirror actual forecasting evaluation.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - naive_baseline_performance: RMSSE-style baseline comparison
            - behavioral_diversity: Sparse vs dense item representation
            - intermittent_representation: Intermittent demand pattern preservation
            - overall_forecasting_score: Combined forecasting quality score (0-100)

        Notes
        -----
        Uses naive seasonal baseline (weekly patterns) similar to M5 competition.
        Behavioral diversity focuses on sparse items (>intermittent_threshold zero sales).
        RMSSE calculation follows M5 competition methodology where applicable.
        """
        logger.info("Calculating forecasting-specific validation metrics")

        # 1. Naive Baseline Performance (RMSSE-style)
        naive_baseline = self._calculate_naive_baseline_performance()

        # 2. Behavioral Diversity Assessment
        behavioral_diversity = self._calculate_behavioral_diversity()

        # 3. Intermittent Representation
        intermittent_representation = self._calculate_intermittent_representation()

        # 4. Overall forecasting score
        # Combine all forecasting-specific metrics
        forecasting_score = self._calculate_forecasting_score(
            naive_baseline, behavioral_diversity, intermittent_representation
        )

        return {
            'naive_baseline_performance': naive_baseline,
            'behavioral_diversity': behavioral_diversity,
            'intermittent_representation': intermittent_representation,
            'overall_forecasting_score': float(forecasting_score)
        }

    def _calculate_naive_baseline_performance(self) -> Dict[str, Any]:
        """
        Calculate naive seasonal baseline performance comparison.

        Uses simple weekly seasonal patterns to compare forecasting difficulty
        between original and sample datasets. Higher RMSSE indicates more
        challenging forecasting problems.
        """
        try:
            # Calculate naive baseline errors for original data
            original_errors = []
            sample_errors = []

            # For each item with daily sales data, calculate naive forecast error
            for _, row in self.original_data.iterrows():
                if 'daily_sales' in row and isinstance(row['daily_sales'], (list, np.ndarray)):
                    sales = np.array(row['daily_sales'])
                    if len(sales) >= 14:  # Need at least 2 weeks for weekly pattern
                        # Naive forecast: use same weekday from previous week
                        naive_forecast = np.concatenate([sales[:7], sales[:-7]])
                        actual = sales
                        # Calculate scaled error (similar to RMSSE)
                        denominator = np.mean(np.abs(np.diff(sales))) + 1e-8
                        scaled_errors = np.abs(actual - naive_forecast) / denominator
                        original_errors.extend(scaled_errors[7:])  # Skip first week

            # Same for sample data
            for _, row in self.sample_data.iterrows():
                if 'daily_sales' in row and isinstance(row['daily_sales'], (list, np.ndarray)):
                    sales = np.array(row['daily_sales'])
                    if len(sales) >= 14:
                        naive_forecast = np.concatenate([sales[:7], sales[:-7]])
                        actual = sales
                        denominator = np.mean(np.abs(np.diff(sales))) + 1e-8
                        scaled_errors = np.abs(actual - naive_forecast) / denominator
                        sample_errors.extend(scaled_errors[7:])

            if original_errors and sample_errors:
                original_rmsse = np.mean(original_errors)
                sample_rmsse = np.mean(sample_errors)
                rmsse_ratio = sample_rmsse / original_rmsse if original_rmsse > 0 else 1.0
            else:
                # Fallback: use coefficient of variation as proxy for forecasting difficulty
                original_cv = self.original_data['cv_sales'].replace([np.inf, -np.inf], np.nan).dropna()
                sample_cv = self.sample_data['cv_sales'].replace([np.inf, -np.inf], np.nan).dropna()

                original_rmsse = original_cv.mean() if len(original_cv) > 0 else 1.0
                sample_rmsse = sample_cv.mean() if len(sample_cv) > 0 else 1.0
                rmsse_ratio = sample_rmsse / original_rmsse if original_rmsse > 0 else 1.0

            return {
                'original_rmsse': float(original_rmsse),
                'sample_rmsse': float(sample_rmsse),
                'rmsse_ratio': float(rmsse_ratio),
                'baseline_preservation_score': float(100 * min(1.0, 2 - abs(rmsse_ratio - 1)))
            }

        except Exception as e:
            logger.warning(f"Failed to calculate naive baseline performance: {e}")
            return {
                'original_rmsse': np.nan,
                'sample_rmsse': np.nan,
                'rmsse_ratio': np.nan,
                'baseline_preservation_score': 0.0,
                'error': str(e)
            }

    def _calculate_behavioral_diversity(self) -> Dict[str, Any]:
        """
        Calculate behavioral diversity metrics.

        Assesses whether the sample preserves the mix of sparse/dense demand
        patterns that are critical for realistic forecasting validation.
        """
        try:
            # Define sparse items (high zero-sales ratio)
            sparse_threshold = 0.5  # >50% zero sales days

            original_sparse = (self.original_data['zero_sales_ratio'] >= sparse_threshold).sum()
            sample_sparse = (self.sample_data['zero_sales_ratio'] >= sparse_threshold).sum()

            original_sparse_ratio = original_sparse / len(self.original_data)
            sample_sparse_ratio = sample_sparse / len(self.sample_data)

            # Calculate diversity preservation score
            # Perfect preservation = 100, significant deviation = lower score
            ratio_difference = abs(original_sparse_ratio - sample_sparse_ratio)
            diversity_score = 100 * (1 - min(1.0, ratio_difference * 2))

            return {
                'sparse_item_ratio_original': float(original_sparse_ratio),
                'sparse_item_ratio_sample': float(sample_sparse_ratio),
                'sparse_threshold': sparse_threshold,
                'diversity_preservation_score': float(diversity_score),
                'absolute_difference': float(ratio_difference)
            }

        except Exception as e:
            logger.warning(f"Failed to calculate behavioral diversity: {e}")
            return {
                'sparse_item_ratio_original': np.nan,
                'sparse_item_ratio_sample': np.nan,
                'diversity_preservation_score': 0.0,
                'error': str(e)
            }

    def _calculate_intermittent_representation(self) -> Dict[str, Any]:
        """
        Calculate intermittent demand representation metrics.

        Assesses whether intermittent items (very sparse demand patterns)
        are adequately represented in the sample dataset.
        """
        try:
            # Intermittent items: very high zero-sales ratio
            original_intermittent = (self.original_data['zero_sales_ratio'] >= self.intermittent_threshold).sum()
            sample_intermittent = (self.sample_data['zero_sales_ratio'] >= self.intermittent_threshold).sum()

            original_intermittent_pct = original_intermittent / len(self.original_data) * 100
            sample_intermittent_pct = sample_intermittent / len(self.sample_data) * 100

            # Representation score: how well intermittent items are preserved
            if original_intermittent_pct > 0:
                representation_ratio = sample_intermittent_pct / original_intermittent_pct
                representation_score = 100 * min(1.0, representation_ratio)
            else:
                representation_score = 100.0  # No intermittent items to preserve

            return {
                'intermittent_threshold': self.intermittent_threshold,
                'original_intermittent_pct': float(original_intermittent_pct),
                'sample_intermittent_pct': float(sample_intermittent_pct),
                'representation_score': float(representation_score),
                'representation_ratio': float(representation_ratio) if original_intermittent_pct > 0 else 1.0
            }

        except Exception as e:
            logger.warning(f"Failed to calculate intermittent representation: {e}")
            return {
                'original_intermittent_pct': np.nan,
                'sample_intermittent_pct': np.nan,
                'representation_score': 0.0,
                'error': str(e)
            }

    def _calculate_forecasting_score(self, naive_baseline: Dict, behavioral_diversity: Dict,
                                   intermittent_representation: Dict) -> float:
        """Calculate overall forecasting quality score."""
        scores = []

        # Include baseline preservation score
        if 'baseline_preservation_score' in naive_baseline:
            scores.append(naive_baseline['baseline_preservation_score'])

        # Include diversity preservation score
        if 'diversity_preservation_score' in behavioral_diversity:
            scores.append(behavioral_diversity['diversity_preservation_score'])

        # Include intermittent representation score
        if 'representation_score' in intermittent_representation:
            scores.append(intermittent_representation['representation_score'])

        return np.mean(scores) if scores else 0.0

    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.

        Integrates all validation metrics into a single comprehensive report
        that provides an overall quality assessment and actionable recommendations
        for improving sample dataset quality.

        Returns
        -------
        Dict[str, Any]
            Complete validation report containing:
            - overall_quality_score: Combined quality score (0-100)
            - distribution_comparison: Distribution analysis results
            - correlation_metrics: Correlation analysis results
            - forecasting_metrics: Forecasting-specific validation results
            - recommendations: List of actionable improvement suggestions
            - summary: Key metrics and statistics summary

        Business Value:
        The validation report provides a single quality assessment that combines
        statistical, distributional, and forecasting-specific metrics. This
        enables data scientists to quickly assess if a sample dataset is
        suitable for model development and provides guidance for improvement.
        """
        logger.info("Generating comprehensive validation report")

        # Calculate all component metrics
        distribution_results = self.calculate_distribution_comparison()
        correlation_results = self.calculate_correlation_metrics()
        forecasting_results = self.calculate_forecasting_metrics()

        # Calculate overall quality score
        overall_quality_score = self._calculate_quality_score(
            distribution_results, correlation_results, forecasting_results
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            distribution_results, correlation_results, forecasting_results
        )

        # Create summary statistics
        summary = {
            'total_items_original': len(self.original_data),
            'total_items_sample': len(self.sample_data),
            'sample_ratio': len(self.sample_data) / len(self.original_data),
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'intermittent_threshold_used': self.intermittent_threshold,
            'significance_level_used': self.significance_level
        }

        report = {
            'overall_quality_score': float(overall_quality_score),
            'distribution_comparison': distribution_results,
            'correlation_metrics': correlation_results,
            'forecasting_metrics': forecasting_results,
            'recommendations': recommendations,
            'summary': summary
        }

        logger.info(f"Validation report generated - Overall quality score: {overall_quality_score:.1f}/100")

        return report

    def _calculate_quality_score(self, distribution_results: Dict, correlation_results: Dict,
                                forecasting_results: Dict) -> float:
        """
        Calculate overall quality score (0-100).

        Combines all validation metrics into a single quality score using
        weighted averaging. Weights reflect the relative importance of
        different validation aspects for forecasting model development.

        Parameters
        ----------
        distribution_results : Dict
            Results from distribution comparison analysis
        correlation_results : Dict
            Results from correlation analysis
        forecasting_results : Dict
            Results from forecasting-specific analysis

        Returns
        -------
        float
            Overall quality score between 0 and 100
        """
        scores = []
        weights = []

        # Distribution score (weight: 0.3)
        # Important for ensuring representative business mix
        if 'overall_distribution_score' in distribution_results:
            scores.append(distribution_results['overall_distribution_score'])
            weights.append(0.3)

        # Correlation score (weight: 0.3)
        # Important for maintaining statistical relationships
        if 'overall_correlation_score' in correlation_results:
            scores.append(correlation_results['overall_correlation_score'])
            weights.append(0.3)

        # Forecasting score (weight: 0.4)
        # Most important for forecasting model validation
        if 'overall_forecasting_score' in forecasting_results:
            scores.append(forecasting_results['overall_forecasting_score'])
            weights.append(0.4)

        if not scores:
            return 0.0

        # Weighted average
        weighted_score = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)

        return min(100.0, max(0.0, weighted_score))

    def _generate_recommendations(self, distribution_results: Dict, correlation_results: Dict,
                                 forecasting_results: Dict) -> List[str]:
        """
        Generate actionable recommendations based on validation results.

        Analyzes validation metrics to identify specific areas where the sample
        dataset could be improved and provides concrete suggestions for
        addressing quality issues.

        Returns
        -------
        List[str]
            List of actionable recommendations for improving sample quality
        """
        recommendations = []

        # Distribution-based recommendations
        if 'categorical_tests' in distribution_results:
            significant_tests = [
                var for var, result in distribution_results['categorical_tests'].items()
                if result.get('significant', False)
            ]
            if significant_tests:
                recommendations.append(
                    f"Category distributions differ significantly for: {', '.join(significant_tests)}. "
                    f"Consider stratified sampling to maintain proportional representation."
                )

        # Correlation-based recommendations
        if 'overall_correlation_score' in correlation_results:
            correlation_score = correlation_results['overall_correlation_score']
            if correlation_score < 70:
                recommendations.append(
                    f"Low correlation preservation (score: {correlation_score:.1f}). "
                    f"The sample may not maintain key statistical relationships from the original data."
                )

        # Forecasting-specific recommendations
        if 'behavioral_diversity' in forecasting_results:
            diversity = forecasting_results['behavioral_diversity']
            if diversity.get('diversity_preservation_score', 0) < 80:
                recommendations.append(
                    f"Poor behavioral diversity preservation. "
                    f"Ensure adequate representation of sparse/intermittent demand patterns."
                )

        if 'intermittent_representation' in forecasting_results:
            intermittent = forecasting_results['intermittent_representation']
            if intermittent.get('representation_score', 0) < 70:
                recommendations.append(
                    f"Insufficient intermittent item representation. "
                    f"Increase sampling of items with >{self.intermittent_threshold*100:.0f}% zero sales."
                )

        # Overall recommendations
        if not recommendations:
            recommendations.append("Sample quality is good across all validation metrics.")

        return recommendations