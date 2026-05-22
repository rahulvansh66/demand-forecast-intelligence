"""
Drift analysis utilities for M5 demand forecasting EDA framework (Step 13).

Provides functions for rigorous statistical validation of training vs validation
period consistency, ensuring reliable model validation through:
- Distribution comparison with statistical tests
- Seasonal representativeness analysis
- Category-specific drift detection
- Quantitative drift severity assessment
- Temporal split integrity validation

Focus: M5 temporal boundaries (d_1 to d_1913 train, d_1914 to d_1941 validation)
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def compare_temporal_distributions(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    columns: List[str],
    apply_bonferroni: bool = True,
) -> Dict[str, Any]:
    """
    Compare statistical distributions between training and validation periods.

    Performs rigorous statistical testing to detect distribution drift using:
    - Kolmogorov-Smirnov (KS) tests for distribution shape differences
    - Mann-Whitney U tests for central tendency shifts
    - Cohen's d effect sizes for practical significance assessment

    Parameters
    ----------
    train_data : pd.DataFrame
        Training period data (typically d_1 to d_1913 for M5)
    validation_data : pd.DataFrame
        Validation period data (typically d_1914 to d_1941 for M5)
    columns : List[str]
        Numerical columns to analyze for drift
    apply_bonferroni : bool, default=True
        Apply Bonferroni correction for multiple testing
        (adjusted alpha = 0.05 / number_of_tests)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - ks_tests: KS test results per column
            {column: {'ks_statistic': float, 'p_value': float}}
        - mannwhitney_tests: Mann-Whitney U test results
            {column: {'statistic': float, 'p_value': float}}
        - effect_sizes: Cohen's d effect sizes
            {column: {'cohens_d': float}}
        - bonferroni_alpha: Adjusted alpha if applied
        - interpretation: Text summary of findings

    Raises
    ------
    ValueError
        If input dataframes are empty or columns list is empty
    KeyError
        If specified columns not found in dataframes
    """
    if train_data.empty or validation_data.empty:
        raise ValueError("Train and validation dataframes cannot be empty")

    if not columns:
        raise ValueError("Columns list cannot be empty")

    missing_cols = set(columns) - set(train_data.columns) - set(validation_data.columns)
    if missing_cols:
        raise KeyError(f"Columns not found in data: {missing_cols}")

    results = {
        'ks_tests': {},
        'mannwhitney_tests': {},
        'effect_sizes': {},
    }

    # Bonferroni correction
    alpha = 0.05
    if apply_bonferroni:
        alpha = 0.05 / len(columns)
        results['bonferroni_alpha'] = alpha

    for col in columns:
        # Clean data: remove NaN values
        train_clean = train_data[col].dropna().values
        val_clean = validation_data[col].dropna().values

        # Skip if insufficient data
        if len(train_clean) < 2 or len(val_clean) < 2:
            results['ks_tests'][col] = {
                'ks_statistic': np.nan,
                'p_value': np.nan,
                'warning': 'Insufficient data for testing'
            }
            results['mannwhitney_tests'][col] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'warning': 'Insufficient data for testing'
            }
            results['effect_sizes'][col] = {'cohens_d': np.nan}
            continue

        # KS test: tests if distributions are different
        ks_stat, ks_pval = stats.ks_2samp(train_clean, val_clean)
        results['ks_tests'][col] = {
            'ks_statistic': float(ks_stat),
            'p_value': float(ks_pval),
            'significant': bool(ks_pval < alpha)
        }

        # Mann-Whitney U test: tests if central tendencies differ
        mw_stat, mw_pval = stats.mannwhitneyu(train_clean, val_clean, alternative='two-sided')
        results['mannwhitney_tests'][col] = {
            'statistic': float(mw_stat),
            'p_value': float(mw_pval),
            'significant': bool(mw_pval < alpha)
        }

        # Cohen's d effect size
        cohens_d = _calculate_cohens_d(train_clean, val_clean)
        results['effect_sizes'][col] = {'cohens_d': float(cohens_d)}

    # Generate interpretation
    results['interpretation'] = _interpret_distribution_comparison(results, alpha)

    return results


def analyze_seasonal_representativeness(
    sales_data: pd.DataFrame,
    calendar_data: pd.DataFrame,
    val_start_day: int = 1914,
    val_end_day: int = 1941,
) -> Dict[str, Any]:
    """
    Analyze seasonal representativeness of validation period.

    Validates that 28-day validation period (typically d_1914 to d_1941)
    captures representative seasonal patterns from training period.
    Checks day-of-week balance, monthly coverage, and event representation.

    Parameters
    ----------
    sales_data : pd.DataFrame
        Sales data with 'date' column
    calendar_data : pd.DataFrame
        Calendar data with 'date', 'day_of_week', 'month' columns
    val_start_day : int, default=1914
        Starting day index for validation period (M5: d_1914)
    val_end_day : int, default=1941
        Ending day index for validation period (M5: d_1941)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - validation_period: Date range and duration
        - seasonal_alignment: Chi-square test for seasonal pattern alignment
        - day_of_week_distribution: Day-of-week balance analysis
        - monthly_coverage: Monthly representation analysis
        - interpretation: Text summary of seasonal representativeness

    Raises
    ------
    KeyError
        If required columns missing from dataframes
    ValueError
        If dataframes empty or date conversion fails
    """
    if sales_data.empty or calendar_data.empty:
        raise ValueError("Sales and calendar dataframes cannot be empty")

    required_cols = ['date', 'day_of_week', 'month']
    missing_cols = set(required_cols) - set(calendar_data.columns)
    if missing_cols:
        raise KeyError(f"Missing columns in calendar_data: {missing_cols}")

    # Convert dates to datetime
    try:
        cal_df = calendar_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(cal_df['date']):
            cal_df['date'] = pd.to_datetime(cal_df['date'])
    except Exception as e:
        raise ValueError(f"Failed to convert dates: {e}")

    results = {}

    # Add day index if not present
    if 'day_index' not in cal_df.columns:
        cal_df['day_index'] = range(1, len(cal_df) + 1)

    # Extract validation period
    val_period = cal_df[
        (cal_df['day_index'] >= val_start_day) & (cal_df['day_index'] <= val_end_day)
    ].copy()

    if val_period.empty:
        raise ValueError(
            f"Validation period {val_start_day}-{val_end_day} not found in calendar"
        )

    results['validation_period'] = {
        'start_date': str(val_period['date'].min()),
        'end_date': str(val_period['date'].max()),
        'duration_days': len(val_period),
        'expected_days': val_end_day - val_start_day + 1
    }

    # Day-of-week analysis
    dow_val_counts = val_period['day_of_week'].value_counts().reindex(range(7), fill_value=0).values
    dow_expected = np.full(7, len(val_period) / 7)  # Uniform expectation

    # Only perform test if we have valid expected frequencies
    if np.all(dow_expected >= 5):  # Chi-square requires expected freq >= 5
        dow_chi2, dow_pval = stats.chisquare(dow_val_counts, f_exp=dow_expected)
    else:
        # Use Fisher's exact test alternative for small expected frequencies
        dow_chi2, dow_pval = np.nan, np.nan

    results['day_of_week_distribution'] = {
        'chi2_statistic': float(dow_chi2) if not np.isnan(dow_chi2) else np.nan,
        'p_value': float(dow_pval) if not np.isnan(dow_pval) else np.nan,
        'balanced': bool(dow_pval > 0.05) if not np.isnan(dow_pval) else True,
        'validation_dow_counts': dict(val_period['day_of_week'].value_counts())
    }

    # Monthly coverage analysis
    months_in_val = sorted(val_period['month'].unique())
    results['monthly_coverage'] = {
        'months_represented': months_in_val,
        'unique_months': len(months_in_val),
        'interpretation': (
            'Full seasonal coverage' if len(months_in_val) >= 2
            else 'Limited seasonal coverage'
        )
    }

    # Seasonal alignment
    results['seasonal_alignment'] = {
        'day_of_week_aligned': results['day_of_week_distribution']['balanced'],
        'months_covered': len(months_in_val) >= 2,
        'representativeness_score': _calculate_seasonal_score(
            results['day_of_week_distribution']['balanced'],
            len(months_in_val)
        )
    }

    results['interpretation'] = _interpret_seasonal_representativeness(results)

    return results


def detect_category_drift(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    category_column: str,
    test_column: str = 'sales',
) -> Dict[str, Any]:
    """
    Detect category-specific distribution drift.

    Performs category-stratified KS tests to identify if drift is uniform
    across categories or concentrated in specific product/store segments.
    Useful for validating that no category has systematically different behavior.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training period data with category and test columns
    validation_data : pd.DataFrame
        Validation period data with category and test columns
    category_column : str
        Name of column defining categories (e.g., 'category', 'dept_id')
    test_column : str, default='sales'
        Numerical column to test for drift

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - category_drift_tests: KS test results per category
            {category: {'ks_statistic': float, 'p_value': float}}
        - overall_drift_detected: Boolean indicating any significant drift
        - most_drifted_category: Category with highest KS statistic
        - drift_concentration: How drift concentrates across categories

    Raises
    ------
    KeyError
        If category or test columns not found
    ValueError
        If dataframes empty or insufficient category representation
    """
    if train_data.empty or validation_data.empty:
        raise ValueError("Train and validation dataframes cannot be empty")

    if category_column not in train_data.columns:
        raise KeyError(f"Category column '{category_column}' not found in train_data")

    if test_column not in train_data.columns:
        raise KeyError(f"Test column '{test_column}' not found in train_data")

    results = {
        'category_drift_tests': {},
        'overall_drift_detected': False,
        'alpha': 0.05
    }

    # Get categories present in both datasets
    train_cats = set(train_data[category_column].dropna().unique())
    val_cats = set(validation_data[category_column].dropna().unique())
    common_cats = train_cats & val_cats

    if not common_cats:
        raise ValueError("No common categories found between train and validation data")

    drift_pvalues = []
    ks_statistics = []

    for cat in sorted(common_cats):
        train_cat_data = train_data[train_data[category_column] == cat][test_column].dropna().values
        val_cat_data = validation_data[validation_data[category_column] == cat][test_column].dropna().values

        # Skip categories with insufficient data
        if len(train_cat_data) < 2 or len(val_cat_data) < 2:
            results['category_drift_tests'][cat] = {
                'ks_statistic': np.nan,
                'p_value': np.nan,
                'warning': 'Insufficient data'
            }
            continue

        ks_stat, ks_pval = stats.ks_2samp(train_cat_data, val_cat_data)
        results['category_drift_tests'][cat] = {
            'ks_statistic': float(ks_stat),
            'p_value': float(ks_pval),
            'significant': bool(ks_pval < 0.05)
        }

        drift_pvalues.append(ks_pval)
        ks_statistics.append(ks_stat)

    # Apply FDR correction for multiple testing
    if drift_pvalues:
        fdr_threshold = _apply_fdr_correction(drift_pvalues)
        results['fdr_threshold'] = float(fdr_threshold)

        # Check for overall drift
        results['overall_drift_detected'] = any(p < fdr_threshold for p in drift_pvalues)

    # Identify most drifted category
    if ks_statistics:
        max_ks_idx = np.argmax(ks_statistics)
        max_ks_cat = sorted(common_cats)[max_ks_idx]
        results['most_drifted_category'] = max_ks_cat
        results['max_ks_statistic'] = float(ks_statistics[max_ks_idx])

    results['interpretation'] = _interpret_category_drift(results)

    return results


def compute_drift_severity_scores(
    ks_results: Dict[str, Dict[str, float]],
    effect_sizes: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Compute quantitative drift severity scores and classification.

    Combines statistical significance (p-values) and practical significance
    (effect sizes) into an interpretable severity scale: None, Low, Moderate,
    High, Critical. Helps prioritize which drift signals require investigation.

    Parameters
    ----------
    ks_results : Dict[str, Dict[str, float]]
        KS test results from compare_temporal_distributions()
        Format: {column: {'ks_statistic': float, 'p_value': float}}
    effect_sizes : Dict[str, Dict[str, float]]
        Cohen's d effect sizes from compare_temporal_distributions()
        Format: {column: {'cohens_d': float}}

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - severity_scores: Per-column severity score (0-100 scale)
        - overall_severity: Overall severity score across all columns
        - drift_classification: Categorical severity (None/Low/Moderate/High/Critical)
        - components: Breakdown of ks_statistic and effect_size contributions
        - interpretation: Text summary of severity assessment

    Raises
    ------
    ValueError
        If ks_results or effect_sizes empty or mismatched columns
    KeyError
        If required fields missing from input dictionaries
    """
    if not ks_results or not effect_sizes:
        raise ValueError("ks_results and effect_sizes cannot be empty")

    if set(ks_results.keys()) != set(effect_sizes.keys()):
        raise ValueError("ks_results and effect_sizes must have matching columns")

    results = {
        'severity_scores': {},
        'components': {},
    }

    severity_list = []

    for col in ks_results.keys():
        ks_entry = ks_results[col]
        effect_entry = effect_sizes[col]

        # Check for warnings/insufficient data
        if 'warning' in ks_entry or np.isnan(ks_entry.get('ks_statistic', np.nan)):
            results['severity_scores'][col] = 0.0
            results['components'][col] = {
                'ks_component': 0.0,
                'effect_component': 0.0,
                'note': 'Insufficient data'
            }
            continue

        # KS statistic component (0-1 scale, converted to 0-50)
        ks_stat = ks_entry.get('ks_statistic', 0)
        ks_component = min(50, ks_stat * 100)

        # Effect size component (Cohen's d, 0-50 scale)
        cohens_d = abs(effect_entry.get('cohens_d', 0))
        # Small effect: 0.2, Medium: 0.5, Large: 0.8
        effect_component = min(50, (cohens_d / 0.8) * 50)

        severity_score = ks_component + effect_component
        results['severity_scores'][col] = float(severity_score)
        results['components'][col] = {
            'ks_component': float(ks_component),
            'effect_component': float(effect_component),
            'ks_statistic': float(ks_stat),
            'cohens_d': float(cohens_d)
        }

        severity_list.append(severity_score)

    # Overall severity: median of individual scores
    results['overall_severity'] = float(np.median(severity_list)) if severity_list else 0.0

    # Classification thresholds
    overall = results['overall_severity']
    if overall <= 5:
        classification = 'None'
    elif overall < 15:
        classification = 'Low'
    elif overall < 40:
        classification = 'Moderate'
    elif overall < 70:
        classification = 'High'
    else:
        classification = 'Critical'

    results['drift_classification'] = classification
    results['interpretation'] = _interpret_severity(results)

    return results


def validate_temporal_split_integrity(
    calendar_data: pd.DataFrame,
    train_days: int = 1913,
    val_days: int = 28,
) -> Dict[str, Any]:
    """
    Validate temporal split boundaries and integrity.

    Ensures that training and validation periods:
    - Have expected number of days (M5: 1913 train, 28 validation)
    - Are contiguous without gaps
    - Are non-overlapping
    - Cover the specified date range

    Parameters
    ----------
    calendar_data : pd.DataFrame
        Calendar data with 'date' column (and optional day_index)
    train_days : int, default=1913
        Expected number of training days (M5 standard)
    val_days : int, default=28
        Expected number of validation days (M5 standard)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - split_valid: Boolean indicating if split is valid
        - train_period: Training date range and statistics
        - val_period: Validation date range and statistics
        - boundary_check: Details of boundary validation
        - gap_analysis: Check for continuity
        - interpretation: Text summary of integrity assessment

    Raises
    ------
    KeyError
        If 'date' column not found in calendar_data
    ValueError
        If calendar_data empty or dates cannot be parsed
    """
    if calendar_data.empty:
        raise ValueError("Calendar data cannot be empty")

    if 'date' not in calendar_data.columns:
        raise KeyError("'date' column not found in calendar_data")

    # Convert dates
    try:
        cal_df = calendar_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(cal_df['date']):
            cal_df['date'] = pd.to_datetime(cal_df['date'])
    except Exception as e:
        raise ValueError(f"Failed to convert dates: {e}")

    results = {
        'split_valid': False,
        'train_period': {},
        'val_period': {},
        'boundary_check': {},
        'gap_analysis': {}
    }

    # Add day index if not present
    if 'day_index' not in cal_df.columns:
        cal_df['day_index'] = range(1, len(cal_df) + 1)

    total_available = len(cal_df)
    expected_total = train_days + val_days

    # Check if we have enough data
    if total_available < expected_total:
        results['boundary_check']['sufficient_data'] = False
        results['boundary_check']['note'] = (
            f"Insufficient data: {total_available} available, {expected_total} required"
        )
        return results

    # Extract periods
    train_df = cal_df[cal_df['day_index'] <= train_days]
    val_df = cal_df[
        (cal_df['day_index'] > train_days) &
        (cal_df['day_index'] <= train_days + val_days)
    ]

    # Validate train period
    results['train_period'] = {
        'start_date': str(train_df['date'].min()),
        'end_date': str(train_df['date'].max()),
        'actual_days': len(train_df),
        'expected_days': train_days,
        'complete': len(train_df) == train_days
    }

    # Validate validation period
    results['val_period'] = {
        'start_date': str(val_df['date'].min()) if not val_df.empty else 'N/A',
        'end_date': str(val_df['date'].max()) if not val_df.empty else 'N/A',
        'actual_days': len(val_df),
        'expected_days': val_days,
        'complete': len(val_df) == val_days
    }

    # Check continuity
    train_dates = pd.to_datetime(train_df['date']).sort_values()
    expected_delta = pd.Timedelta(days=1)
    date_diffs = train_dates.diff()[1:]

    has_gaps = (date_diffs != expected_delta).any()

    results['gap_analysis'] = {
        'train_has_gaps': bool(has_gaps),
        'consecutive': not has_gaps
    }

    # Overall validity
    results['boundary_check'] = {
        'train_complete': results['train_period']['complete'],
        'val_complete': results['val_period']['complete'],
        'continuous': results['gap_analysis']['consecutive'],
        'non_overlapping': True  # By construction
    }

    results['split_valid'] = (
        results['train_period']['complete'] and
        results['val_period']['complete'] and
        results['gap_analysis']['consecutive']
    )

    results['interpretation'] = _interpret_temporal_integrity(results)

    return results


# ============================================================================
# Helper Functions
# ============================================================================

def _calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    mean_diff = np.mean(group1) - np.mean(group2)
    return mean_diff / pooled_std


def _apply_fdr_correction(pvalues: List[float], alpha: float = 0.05) -> float:
    """Apply Benjamini-Hochberg FDR correction and return threshold."""
    if not pvalues:
        return alpha

    sorted_pvalues = sorted(pvalues)
    n = len(sorted_pvalues)

    for i, pval in enumerate(sorted_pvalues):
        threshold = (i + 1) / n * alpha
        if pval > threshold:
            return threshold

    return alpha


def _calculate_seasonal_score(dow_aligned: bool, months_covered: int) -> float:
    """Calculate seasonal representativeness score (0-1)."""
    score = 0.0

    # Day-of-week balance: 0.5 points
    if dow_aligned:
        score += 0.5

    # Monthly coverage: 0.5 points
    if months_covered >= 2:
        score += min(0.5, (months_covered / 12) * 0.5)

    return min(1.0, score)


def _interpret_distribution_comparison(results: Dict[str, Any], alpha: float) -> str:
    """Generate interpretation text for distribution comparison."""
    significant_cols = [
        col for col, test in results['ks_tests'].items()
        if test.get('significant', False)
    ]

    if not significant_cols:
        return "No significant distribution drift detected (all KS tests p > alpha)"

    effect_summary = []
    for col in significant_cols:
        d = abs(results['effect_sizes'][col]['cohens_d'])
        if d > 0.8:
            effect_summary.append(f"{col} (large effect)")
        elif d > 0.5:
            effect_summary.append(f"{col} (medium effect)")
        else:
            effect_summary.append(f"{col} (small effect)")

    return f"Significant drift detected in: {', '.join(effect_summary)}"


def _interpret_seasonal_representativeness(results: Dict[str, Any]) -> str:
    """Generate interpretation for seasonal representativeness."""
    alignment = results['seasonal_alignment']
    dow_balanced = alignment['day_of_week_aligned']
    months_covered = results['monthly_coverage']['unique_months']

    if dow_balanced and months_covered >= 2:
        return "Validation period has good seasonal representativeness (balanced DoW, adequate month coverage)"
    elif dow_balanced or months_covered >= 2:
        return "Validation period has partial seasonal representativeness (limited month or day-of-week coverage)"
    else:
        return "Validation period has limited seasonal representativeness (consider impact on model evaluation)"


def _interpret_category_drift(results: Dict[str, Any]) -> str:
    """Generate interpretation for category drift analysis."""
    if not results['category_drift_tests']:
        return "Insufficient data for category drift analysis"

    overall_drift = results['overall_drift_detected']
    most_drifted = results.get('most_drifted_category', 'Unknown')

    if overall_drift:
        return f"Significant drift detected. Most affected category: {most_drifted}"
    else:
        return "No significant category-specific drift detected"


def _interpret_severity(results: Dict[str, Any]) -> str:
    """Generate interpretation for severity scores."""
    classification = results['drift_classification']

    interpretations = {
        'None': 'No meaningful drift detected. Training and validation periods are comparable.',
        'Low': 'Minor drift detected. Unlikely to significantly impact model evaluation.',
        'Moderate': 'Moderate drift detected. Model performance may vary between train and validation.',
        'High': 'Significant drift detected. Recommend investigation and potential adjustments.',
        'Critical': 'Severe drift detected. Model validation may be unreliable. Recommend deeper analysis.',
    }

    return interpretations.get(classification, 'Unknown classification')


def _interpret_temporal_integrity(results: Dict[str, Any]) -> str:
    """Generate interpretation for temporal split integrity."""
    if results['split_valid']:
        return "Temporal split is valid and maintains integrity (all checks passed)"

    checks = results['boundary_check']
    issues = []

    if not checks.get('train_complete', False):
        issues.append("incomplete training period")
    if not checks.get('val_complete', False):
        issues.append("incomplete validation period")
    if not checks.get('continuous', False):
        issues.append("date continuity gaps")

    return f"Temporal split integrity issues: {', '.join(issues)}"
