"""
Correlation and categorical pattern analysis module.

This module provides functionality for analyzing relationships between
features and target variables, with a focus on categorical sales patterns
and business-meaningful statistical analysis.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


def analyze_categorical_sales_patterns(
    data: pd.DataFrame,
    category_col: str,
    sales_col: str
) -> Dict[str, Any]:
    """
    Analyze sales patterns by categorical variable.

    This function performs statistical analysis of sales patterns grouped by
    a categorical variable, providing insights into how different categories
    influence sales behavior. It handles NaN values robustly and provides
    meaningful statistics even for edge cases.

    Args:
        data (pd.DataFrame): Input dataframe containing categorical and sales data.
        category_col (str): Name of the categorical column to group by.
        sales_col (str): Name of the sales column to analyze.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'categories': Unique categories found (excluding NaN)
            - 'summary_stats': Overall statistical summary (mean, std, count)
            - 'by_category': Detailed statistics for each category
            - 'data_quality': Information about NaN values and data completeness

    Raises:
        ValueError: If specified columns don't exist in dataframe or no valid data.
        TypeError: If data is not a pandas DataFrame.

    Example:
        >>> df = pd.DataFrame({
        ...     'category': ['A', 'A', 'B', 'B'],
        ...     'sales': [100, 120, 200, 250]
        ... })
        >>> result = analyze_categorical_sales_patterns(df, 'category', 'sales')
        >>> result['categories']
        ['A', 'B']
    """
    # Type checking
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    # Column validation
    if category_col not in data.columns:
        raise ValueError(f"Column '{category_col}' not found in dataframe")
    if sales_col not in data.columns:
        raise ValueError(f"Column '{sales_col}' not found in dataframe")

    # Edge case: empty dataframe
    if len(data) == 0:
        raise ValueError("DataFrame is empty")

    # Data quality analysis
    original_count = len(data)
    category_nan_count = data[category_col].isna().sum()
    sales_nan_count = data[sales_col].isna().sum()

    # Clean data: remove rows where either category or sales is NaN
    clean_data = data.dropna(subset=[category_col, sales_col])

    # Edge case: no valid data after cleaning
    if len(clean_data) == 0:
        raise ValueError("No valid data found after removing NaN values")

    # Extract unique categories (NaN values are already removed)
    categories = clean_data[category_col].unique()

    # Handle categories that might be float NaN (from numpy)
    categories = [cat for cat in categories if pd.notna(cat)]

    if len(categories) == 0:
        raise ValueError("No valid categories found")

    # Convert to list for consistent return type
    categories = list(categories)

    # Calculate overall summary statistics using clean data
    sales_data = clean_data[sales_col]

    def safe_float_conversion(value):
        """Safely convert to float, handling NaN."""
        if pd.isna(value):
            return None
        return float(value)

    summary_stats = {
        'mean': safe_float_conversion(sales_data.mean()),
        'std': safe_float_conversion(sales_data.std()),
        'count': len(clean_data),
        'min': safe_float_conversion(sales_data.min()),
        'max': safe_float_conversion(sales_data.max()),
        'median': safe_float_conversion(sales_data.median())
    }

    # Handle special case where std might be NaN (single value or all values are the same)
    if pd.isna(summary_stats['std']):
        summary_stats['std'] = 0.0

    # Calculate per-category statistics
    by_category = {}
    grouped = clean_data.groupby(category_col)[sales_col]

    for category in categories:
        try:
            category_data = grouped.get_group(category)

            # Calculate statistics for this category
            cat_stats = {
                'mean': safe_float_conversion(category_data.mean()),
                'std': safe_float_conversion(category_data.std()),
                'count': len(category_data),
                'min': safe_float_conversion(category_data.min()),
                'max': safe_float_conversion(category_data.max()),
                'median': safe_float_conversion(category_data.median())
            }

            # Handle single-value categories (std would be NaN)
            if pd.isna(cat_stats['std']) or len(category_data) == 1:
                cat_stats['std'] = 0.0

            by_category[category] = cat_stats

        except KeyError:
            # Category not found in grouped data - this shouldn't happen but handle gracefully
            continue

    # Data quality summary
    data_quality = {
        'original_count': original_count,
        'valid_count': len(clean_data),
        'category_nan_count': int(category_nan_count),
        'sales_nan_count': int(sales_nan_count),
        'data_completeness_ratio': len(clean_data) / original_count if original_count > 0 else 0.0
    }

    return {
        'categories': categories,
        'summary_stats': summary_stats,
        'by_category': by_category,
        'data_quality': data_quality
    }


def compute_temporal_sales_correlations(
    sales_data: pd.DataFrame,
    calendar_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute correlations between temporal features and sales patterns.

    Analyzes how different temporal features (weekday, month) relate to
    sales patterns across categories. Provides business-focused interpretation
    of temporal effects on demand.

    Parameters
    ----------
    sales_data : pd.DataFrame
        Sales data with daily sales columns (d_1, d_2, etc.) and category information
    calendar_data : pd.DataFrame
        Calendar data with temporal features including weekday and date

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - temporal_correlations: Correlation results by category
        - interpretation: Summary of temporal pattern strength

    Example
    -------
    >>> result = compute_temporal_sales_correlations(sales_data, calendar_data)
    >>> print(result['temporal_correlations']['FOODS'])
    {'weekday_correlation': 0.15, 'month_correlation': 0.05, ...}
    """
    results = {}

    # Get sales columns
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]

    if len(sales_cols) == 0:
        raise ValueError("No sales columns (d_*) found in sales_data")

    # Create daily sales series aligned with calendar
    daily_sales_by_category = {}

    for category in sales_data['cat_id'].unique():
        cat_data = sales_data[sales_data['cat_id'] == category]

        # Sum sales across all items in category for each day
        category_daily_sales = []
        for col in sales_cols:
            daily_total = cat_data[col].sum()
            category_daily_sales.append(daily_total)

        daily_sales_by_category[category] = category_daily_sales

    # Create correlation matrix with calendar features
    correlation_results = {}

    for category, daily_sales in daily_sales_by_category.items():
        # Align with calendar data
        num_days = min(len(daily_sales), len(calendar_data))
        daily_sales_aligned = daily_sales[:num_days]

        df = pd.DataFrame({
            'sales': daily_sales_aligned,
        })

        # Add weekday encoding if available
        if 'weekday' in calendar_data.columns:
            weekday_map = {
                'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
                'Friday': 5, 'Saturday': 6, 'Sunday': 7
            }
            df['weekday'] = calendar_data['weekday'].head(num_days).map(weekday_map)
            weekday_corr = df['sales'].corr(df['weekday'])
        else:
            weekday_corr = 0.0

        # Add month encoding if available
        if 'date' in calendar_data.columns:
            df['month'] = pd.to_datetime(calendar_data['date'].head(num_days)).dt.month
            month_corr = df['sales'].corr(df['month'])
        else:
            month_corr = 0.0

        correlation_results[category] = {
            'weekday_correlation': float(weekday_corr) if not np.isnan(weekday_corr) else 0.0,
            'month_correlation': float(month_corr) if not np.isnan(month_corr) else 0.0,
            'weekday_effect_strength': float(abs(weekday_corr)) if not np.isnan(weekday_corr) else 0.0,
            'seasonal_effect_strength': float(abs(month_corr)) if not np.isnan(month_corr) else 0.0
        }

    results['temporal_correlations'] = correlation_results
    results['interpretation'] = 'Temporal pattern strength analysis by category'

    return results


def compute_snap_benefit_impact(
    sales_data: pd.DataFrame,
    calendar_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyze SNAP benefit impact on FOODS category sales.

    Compares average sales on SNAP benefit days vs. non-SNAP days to
    quantify the impact of government assistance programs on retail demand.

    Parameters
    ----------
    sales_data : pd.DataFrame
        Sales data with item/store information and daily sales columns
    calendar_data : pd.DataFrame
        Calendar data with SNAP benefit indicators (snap_CA, snap_TX, snap_WI columns)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - snap_impact_by_state: SNAP benefit impact analysis per state
        - foods_category_analysis: Overall FOODS category statistics
        - error: Error message if FOODS category not found

    Example
    -------
    >>> result = compute_snap_benefit_impact(sales_data, calendar_data)
    >>> print(result['snap_impact_by_state']['CA'])
    {'snap_day_avg_sales': 1500, 'non_snap_day_avg_sales': 1200, 'lift_percentage': 25}
    """
    results = {}

    # Focus on FOODS category
    foods_data = sales_data[sales_data['cat_id'] == 'FOODS']

    if len(foods_data) == 0:
        return {'error': 'No FOODS category data found'}

    # Get sales columns
    sales_cols = [col for col in foods_data.columns if col.startswith('d_')]

    if len(sales_cols) == 0:
        return {'error': 'No sales columns found'}

    # Calculate daily FOODS sales
    daily_foods_sales = []
    for col in sales_cols:
        daily_total = foods_data[col].sum()
        daily_foods_sales.append(daily_total)

    # Align with SNAP data if available
    snap_cols = [col for col in calendar_data.columns if col.startswith('snap_')]

    snap_analysis = {}
    for snap_col in snap_cols:
        state = snap_col.split('_')[1]  # Extract state (CA, TX, WI)

        # Get SNAP days vs non-SNAP days
        snap_days = calendar_data[calendar_data[snap_col] == 1].index.tolist()
        non_snap_days = calendar_data[calendar_data[snap_col] == 0].index.tolist()

        # Calculate sales on SNAP vs non-SNAP days
        snap_sales = [daily_foods_sales[i] for i in snap_days if i < len(daily_foods_sales)]
        non_snap_sales = [daily_foods_sales[i] for i in non_snap_days if i < len(daily_foods_sales)]

        if len(snap_sales) > 0 and len(non_snap_sales) > 0:
            snap_day_avg = np.mean(snap_sales)
            non_snap_day_avg = np.mean(non_snap_sales)

            # Calculate lift percentage
            if non_snap_day_avg > 0:
                lift_pct = (snap_day_avg / non_snap_day_avg - 1) * 100
            else:
                lift_pct = 0.0

            snap_analysis[state] = {
                'snap_day_avg_sales': float(snap_day_avg),
                'non_snap_day_avg_sales': float(non_snap_day_avg),
                'lift_percentage': float(lift_pct),
                'snap_day_count': len(snap_days),
                'non_snap_day_count': len(non_snap_days),
                'statistical_significance': 'Not calculated - requires t-test'
            }

    results['snap_impact_by_state'] = snap_analysis
    results['foods_category_analysis'] = {
        'total_items': len(foods_data),
        'avg_daily_volume': float(np.mean(daily_foods_sales)),
        'min_daily_volume': float(np.min(daily_foods_sales)),
        'max_daily_volume': float(np.max(daily_foods_sales))
    }

    return results
