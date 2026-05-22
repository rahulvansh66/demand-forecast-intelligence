"""
Temporal analysis utilities for EDA framework step 8.

Provides functions for:
- Time structure validation
- Seasonality detection and decomposition
- Trend analysis and structural breaks
- Autocorrelation analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")


def analyze_time_structure(
    sales_data: pd.DataFrame,
    calendar_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyze time structure of the M5 dataset.

    Validates daily frequency completeness, panel structure, and calendar alignment.
    Checks for missing dates and time series regularity.

    Parameters
    ----------
    sales_data : pd.DataFrame
        Sales data with daily columns (d_1, d_2, etc.)
    calendar_data : pd.DataFrame
        Calendar data with date mappings

    Returns
    -------
    Dict[str, Any]
        Time structure analysis results including:
        - total_days: Number of daily periods
        - total_series: Number of product-store combinations
        - frequency: Temporal frequency (Daily)
        - time_range: Start, end, and completeness analysis
        - frequency_validation: Consistency checks
        - panel_structure: Panel data structure metrics
    """
    results = {}

    # Get sales columns
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]

    # Basic time structure
    results['total_days'] = len(sales_cols)
    results['total_series'] = len(sales_data)
    results['frequency'] = 'Daily'

    # Date range analysis
    if 'date' in calendar_data.columns:
        dates = pd.to_datetime(calendar_data['date'])
        results['time_range'] = {
            'start_date': str(dates.min()),
            'end_date': str(dates.max()),
            'total_calendar_days': len(dates)
        }

        # Check for missing dates
        expected_days = (dates.max() - dates.min()).days + 1
        actual_days = len(dates)
        results['missing_dates'] = expected_days - actual_days
    else:
        results['time_range'] = {
            'start_date': 'Unknown',
            'end_date': 'Unknown',
            'total_calendar_days': len(calendar_data)
        }
        results['missing_dates'] = 0

    # Frequency validation
    results['frequency_validation'] = {
        'sales_columns': len(sales_cols),
        'calendar_rows': len(calendar_data),
        'structure_consistent': len(sales_cols) == len(calendar_data)
    }

    # Panel structure validation
    results['panel_structure'] = {
        'entities': len(sales_data),
        'time_periods': len(sales_cols),
        'total_observations': len(sales_data) * len(sales_cols)
    }

    return results


def detect_seasonal_patterns(
    sales_data: pd.DataFrame,
    calendar_data: pd.DataFrame,
    hierarchy_level: str = 'category'
) -> Dict[str, Any]:
    """
    Detect seasonal patterns using variance-based seasonality strength metric.

    Performs STL-style decomposition at specified hierarchy level, computing
    weekly and monthly seasonality strength using variance ratios with business
    interpretation for retail demand patterns.

    Parameters
    ----------
    sales_data : pd.DataFrame
        Sales data with hierarchical information (cat_id, dept_id, store_id)
    calendar_data : pd.DataFrame
        Calendar data with temporal features
    hierarchy_level : str, default 'category'
        Level for aggregation ('category', 'department', 'store')

    Returns
    -------
    Dict[str, Any]
        Seasonal pattern analysis results with:
        - seasonal_patterns: Dictionary with analysis by group
        - hierarchy_level: Level used for aggregation
        Each group contains:
        - weekly_seasonality: Strength and pattern (if >= 14 days data)
        - monthly_seasonality: Strength and pattern (if >= 60 days data)
        - trend: Linear trend direction and significance
    """
    results = {}

    # Get sales columns
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]

    # Aggregate by hierarchy level
    if hierarchy_level == 'category':
        group_col = 'cat_id'
    elif hierarchy_level == 'department':
        group_col = 'dept_id'
    else:
        group_col = 'store_id'

    seasonal_analysis = {}

    for group in sales_data[group_col].unique():
        group_data = sales_data[sales_data[group_col] == group]

        # Sum daily sales across all items in group
        daily_sales = []
        for col in sales_cols:
            daily_total = group_data[col].sum()
            daily_sales.append(daily_total)

        # Convert to time series
        ts = pd.Series(daily_sales)

        # Basic seasonality metrics
        group_analysis = {}

        # Weekly seasonality (if we have enough data)
        if len(ts) >= 14:  # At least 2 weeks
            weekly_pattern = []
            for i in range(7):
                week_days = ts[i::7]  # Every 7th day starting from day i
                weekly_pattern.append(week_days.mean())

            # Calculate weekly seasonality strength
            weekly_var = np.var(weekly_pattern)
            total_var = np.var(ts)
            weekly_strength = weekly_var / total_var if total_var > 0 else 0

            group_analysis['weekly_seasonality'] = {
                'strength': float(weekly_strength),
                'pattern': [float(x) for x in weekly_pattern],
                'interpretation': 'Strong' if weekly_strength > 0.1 else 'Weak'
            }

        # Monthly seasonality (if we have enough data)
        if len(ts) >= 60:  # At least 2 months
            # Simple monthly aggregation
            monthly_sales = []
            for i in range(0, len(ts), 30):  # Approximate monthly chunks
                month_sum = ts[i:i+30].sum()
                monthly_sales.append(month_sum)

            if len(monthly_sales) > 1:
                monthly_var = np.var(monthly_sales)
                monthly_strength = monthly_var / np.var(ts) if np.var(ts) > 0 else 0

                group_analysis['monthly_seasonality'] = {
                    'strength': float(monthly_strength),
                    'monthly_totals': [float(x) for x in monthly_sales],
                    'interpretation': 'Strong' if monthly_strength > 0.15 else 'Weak'
                }

        # Trend analysis
        if len(ts) > 1:
            # Simple linear trend
            x = np.arange(len(ts))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts)

            group_analysis['trend'] = {
                'slope': float(slope),
                'direction': 'Growing' if slope > 0 else 'Declining' if slope < 0 else 'Stable',
                'r_squared': float(r_value ** 2),
                'significance': 'Significant' if p_value < 0.05 else 'Not significant'
            }

        seasonal_analysis[group] = group_analysis

    results['seasonal_patterns'] = seasonal_analysis
    results['hierarchy_level'] = hierarchy_level

    return results


def analyze_trend_components(
    sales_data: pd.DataFrame,
    calendar_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyze trend components with structural break detection.

    Performs linear regression on aggregate daily sales, computes growth rates,
    and identifies structural breaks using t-tests between periods.

    Parameters
    ----------
    sales_data : pd.DataFrame
        Sales data for trend analysis
    calendar_data : pd.DataFrame
        Calendar data for temporal context

    Returns
    -------
    Dict[str, Any]
        Trend analysis results with:
        - linear_trend: Slope, intercept, R², p-value, and interpretation
        - growth_analysis: Growth rate over time period (if >= 365 days)
        - structural_break_analysis: Midpoint comparison with statistical test
    """
    results = {}

    # Get sales columns
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]

    # Calculate total daily sales across all products
    daily_totals = []
    for col in sales_cols:
        daily_total = sales_data[col].sum()
        daily_totals.append(daily_total)

    ts = pd.Series(daily_totals)

    # Linear trend analysis
    x = np.arange(len(ts))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts)

    results['linear_trend'] = {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value ** 2),
        'p_value': float(p_value),
        'direction': 'Growing' if slope > 0 else 'Declining' if slope < 0 else 'Stable',
        'interpretation': f"{'Growing' if slope > 0 else 'Declining'} trend, R² = {r_value**2:.3f}"
    }

    # Growth rate calculation (year-over-year if data available)
    if len(ts) >= 365:
        early_period = ts[:365].mean()
        late_period = ts[-365:].mean()
        growth_rate = (late_period / early_period - 1) * 100 if early_period > 0 else 0

        results['growth_analysis'] = {
            'early_period_avg': float(early_period),
            'late_period_avg': float(late_period),
            'growth_rate_percent': float(growth_rate),
            'interpretation': f"{'Growth' if growth_rate > 0 else 'Decline'} of {abs(growth_rate):.1f}% over time period"
        }

    # Basic structural break detection (simplified)
    if len(ts) > 100:
        mid_point = len(ts) // 2
        first_half_mean = ts[:mid_point].mean()
        second_half_mean = ts[mid_point:].mean()

        # t-test for significant difference
        t_stat, p_val = stats.ttest_ind(ts[:mid_point], ts[mid_point:])

        results['structural_break_analysis'] = {
            'first_half_mean': float(first_half_mean),
            'second_half_mean': float(second_half_mean),
            'mean_difference': float(second_half_mean - first_half_mean),
            'p_value': float(p_val),
            'significant_break': bool(p_val < 0.05),
            'interpretation': f"{'Significant' if p_val < 0.05 else 'No significant'} structural break detected"
        }

    return results


def compute_autocorrelation_analysis(
    sales_data: pd.DataFrame,
    max_lags: int = 365
) -> Dict[str, Any]:
    """
    Compute autocorrelation analysis for lag structure identification.

    Calculates autocorrelations at key lags (1, 7, 14, 28, 30 days) representing
    daily, weekly, bi-weekly, 4-week, and monthly periodicities. Provides business
    interpretation for forecasting model specification.

    Parameters
    ----------
    sales_data : pd.DataFrame
        Sales data for autocorrelation analysis
    max_lags : int, default 365
        Maximum number of lags to analyze

    Returns
    -------
    Dict[str, Any]
        Autocorrelation analysis results with:
        - autocorrelations: Dictionary with lag-specific correlations and strength
        - significant_lags: List of lags exceeding threshold
        - business_interpretation: Seasonality strength and recommended lag structure
        - max_lags_analyzed: Actual maximum lag used
    """
    results = {}

    # Get sales columns
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]

    # Calculate total daily sales
    daily_totals = []
    for col in sales_cols:
        daily_total = sales_data[col].sum()
        daily_totals.append(daily_total)

    ts = pd.Series(daily_totals)

    # Limit max_lags to reasonable value
    max_lags = min(max_lags, len(ts) // 4)

    # Calculate autocorrelations for key lags
    key_lags = [1, 7, 14, 28, 30]  # Daily, weekly, bi-weekly, 4-week, monthly
    key_lags = [lag for lag in key_lags if lag < len(ts)]

    autocorrelations = {}
    for lag in key_lags:
        if lag < len(ts):
            corr = ts.autocorr(lag=lag)
            autocorrelations[f'lag_{lag}'] = {
                'correlation': float(corr),
                'interpretation': 'Strong' if abs(corr) > 0.3 else 'Moderate' if abs(corr) > 0.1 else 'Weak'
            }

    results['autocorrelations'] = autocorrelations

    # Identify significant lags
    significant_lags = []
    for lag_name, lag_data in autocorrelations.items():
        if abs(lag_data['correlation']) > 0.2:  # Arbitrary threshold
            significant_lags.append({
                'lag': lag_name,
                'correlation': float(lag_data['correlation'])
            })

    results['significant_lags'] = significant_lags
    results['max_lags_analyzed'] = max_lags

    # Business interpretation
    weekly_corr = autocorrelations.get('lag_7', {}).get('correlation', 0)
    monthly_corr = autocorrelations.get('lag_30', {}).get('correlation', 0)

    results['business_interpretation'] = {
        'weekly_seasonality_strength': float(abs(weekly_corr)),
        'monthly_seasonality_strength': float(abs(monthly_corr)),
        'recommended_lags': [lag['lag'] for lag in significant_lags],
        'summary': f"Weekly pattern: {'Strong' if abs(weekly_corr) > 0.3 else 'Weak'}, Monthly pattern: {'Strong' if abs(monthly_corr) > 0.3 else 'Weak'}"
    }

    return results
