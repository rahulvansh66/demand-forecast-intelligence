"""
Feature profiling utilities for M5 dataset EDA Step 5.

Provides functions for analyzing individual features in the M5 dataset,
including categorical distributions, geographic patterns, temporal correlations,
price behavior, and feature importance rankings.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats


def analyze_categorical_distributions(sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate distribution statistics for categorical features.

    Analyzes cat_id, dept_id, state_id, store_id performance variations.
    Returns mean, median, std, skewness for sales by category and rankings.

    Parameters
    ----------
    sales_df : pd.DataFrame
        M5 sales data with categorical hierarchy columns
    calendar_df : pd.DataFrame
        M5 calendar data with date mappings

    Returns
    -------
    Dict[str, Any]
        Distribution statistics and rankings for each categorical feature

    Raises
    ------
    ValueError
        If sales_df is empty or missing required columns
    """
    if sales_df.empty:
        raise ValueError("Empty sales data")

    # Check required columns
    required_cols = ['cat_id', 'dept_id', 'state_id', 'store_id']
    missing_cols = [col for col in required_cols if col not in sales_df.columns]

    # Check for sales columns
    sales_cols = [col for col in sales_df.columns if col.startswith('d_')]
    if not sales_cols:
        missing_cols.append('sales columns (d_1, d_2, etc.)')

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    result = {}

    def _calculate_category_statistics(cat_col: str) -> Dict[str, Any]:
        """Helper function to calculate statistics for a categorical column."""
        category_stats = {}

        for category in sales_df[cat_col].unique():
            category_data = sales_df[sales_df[cat_col] == category]

            # Extract sales values across all days
            sales_values = []
            for sales_col in sales_cols:
                sales_values.extend(category_data[sales_col].values)

            sales_values = np.array(sales_values)

            category_stats[category] = {
                'mean': float(np.mean(sales_values)),
                'median': float(np.median(sales_values)),
                'std': float(np.std(sales_values)),
                'skewness': float(pd.Series(sales_values).skew())
            }

        # Create ranking by mean sales
        sorted_categories = sorted(category_stats.items(),
                                 key=lambda x: x[1]['mean'],
                                 reverse=True)

        ranking = {cat: rank + 1 for rank, (cat, stats) in enumerate(sorted_categories)}

        # Return organized statistics and rankings
        return {
            'mean': {cat: stats['mean'] for cat, stats in category_stats.items()},
            'median': {cat: stats['median'] for cat, stats in category_stats.items()},
            'std': {cat: stats['std'] for cat, stats in category_stats.items()},
            'skewness': {cat: stats['skewness'] for cat, stats in category_stats.items()},
            'ranking': ranking
        }

    # Analyze each categorical feature
    for cat_col in required_cols:
        result[cat_col] = _calculate_category_statistics(cat_col)

    return result


def analyze_geographic_patterns(sales_df: pd.DataFrame, sell_prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze state-level and store-level performance variations.

    Calculates geographic coefficients of variation and performance metrics
    across different states and stores.

    Parameters
    ----------
    sales_df : pd.DataFrame
        M5 sales data with geographic identifiers
    sell_prices_df : pd.DataFrame
        M5 price data by store and item

    Returns
    -------
    Dict[str, Any]
        Geographic performance metrics and coefficients of variation
    """
    result = {}

    # Get sales columns
    sales_cols = [col for col in sales_df.columns if col.startswith('d_')]

    # Analyze state-level performance
    state_performance = {}
    for state in sales_df['state_id'].unique():
        state_data = sales_df[sales_df['state_id'] == state]

        # Calculate mean sales for this state
        sales_values = []
        for sales_col in sales_cols:
            sales_values.extend(state_data[sales_col].values)

        # Calculate mean price for this state
        state_stores = state_data['store_id'].unique()
        state_price_data = sell_prices_df[sell_prices_df['store_id'].isin(state_stores)]

        state_performance[state] = {
            'mean_sales': float(np.mean(sales_values)),
            'mean_price': float(state_price_data['sell_price'].mean()) if not state_price_data.empty else 0.0
        }

    # Analyze store-level performance
    store_performance = {}
    for store in sales_df['store_id'].unique():
        store_data = sales_df[sales_df['store_id'] == store]

        # Calculate mean sales for this store
        sales_values = []
        for sales_col in sales_cols:
            sales_values.extend(store_data[sales_col].values)

        # Calculate mean price for this store
        store_price_data = sell_prices_df[sell_prices_df['store_id'] == store]

        store_performance[store] = {
            'mean_sales': float(np.mean(sales_values)),
            'mean_price': float(store_price_data['sell_price'].mean()) if not store_price_data.empty else 0.0
        }

    # Calculate geographic coefficients of variation
    state_sales_means = [perf['mean_sales'] for perf in state_performance.values()]
    store_sales_means = [perf['mean_sales'] for perf in store_performance.values()]

    state_cv = float(np.std(state_sales_means) / np.mean(state_sales_means)) if len(state_sales_means) > 1 and np.mean(state_sales_means) > 0 else 0.0
    store_cv = float(np.std(store_sales_means) / np.mean(store_sales_means)) if len(store_sales_means) > 1 and np.mean(store_sales_means) > 0 else 0.0

    result['state_performance'] = state_performance
    result['store_performance'] = store_performance
    result['geographic_cv'] = {
        'state_cv': state_cv,
        'store_cv': store_cv
    }

    return result


def analyze_temporal_correlations(sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate correlations between temporal features and sales.

    Includes weekday effects, month effects, holiday proximity, and
    statistical significance tests.

    Parameters
    ----------
    sales_df : pd.DataFrame
        M5 sales data with daily sales columns
    calendar_df : pd.DataFrame
        M5 calendar data with temporal features

    Returns
    -------
    Dict[str, Any]
        Correlation coefficients and significance tests for temporal features
    """
    result = {}

    # Get sales columns and prepare data
    sales_cols = [col for col in sales_df.columns if col.startswith('d_')]

    # Create mapping from day to calendar features
    calendar_map = {}
    for _, row in calendar_df.iterrows():
        calendar_map[row['d']] = {
            'weekday': row.get('weekday', None),
            'month': row.get('month', None),
            'event_name_1': row.get('event_name_1', None),
            'event_type_1': row.get('event_type_1', None),
            'snap_CA': row.get('snap_CA', None)
        }

    # Aggregate sales data across all items and days
    daily_sales = []
    weekdays = []
    months = []
    events = []
    snaps = []

    for sales_col in sales_cols:
        if sales_col in calendar_map:
            # Sum sales across all items for this day
            total_sales = sales_df[sales_col].sum()
            daily_sales.append(total_sales)

            # Get calendar features
            cal_features = calendar_map[sales_col]
            weekdays.append(cal_features.get('weekday', np.nan))
            months.append(cal_features.get('month', np.nan))
            events.append(1 if cal_features.get('event_name_1') is not None else 0)
            snaps.append(cal_features.get('snap_CA', 0))

    daily_sales = np.array(daily_sales)

    # Calculate correlations for numeric features
    result['weekday_correlation'] = float(np.corrcoef(weekdays, daily_sales)[0, 1]) if len(set(weekdays)) > 1 else np.nan
    result['month_correlation'] = float(np.corrcoef(months, daily_sales)[0, 1]) if len(set(months)) > 1 else np.nan

    # Calculate correlations for event features
    result['event_correlation'] = float(np.corrcoef(events, daily_sales)[0, 1]) if len(set(events)) > 1 else 0.0
    result['snap_correlation'] = float(np.corrcoef(snaps, daily_sales)[0, 1]) if len(set(snaps)) > 1 else 0.0

    # Calculate significance tests (simplified)
    significance_tests = {}
    if not np.isnan(result['weekday_correlation']) and len(daily_sales) > 10:
        _, p_value = stats.pearsonr(weekdays, daily_sales)
        significance_tests['weekday_p_value'] = float(p_value)

    if not np.isnan(result['month_correlation']) and len(daily_sales) > 10:
        _, p_value = stats.pearsonr(months, daily_sales)
        significance_tests['month_p_value'] = float(p_value)

    result['significance_tests'] = significance_tests

    return result


def analyze_price_behavior(sell_prices_df: pd.DataFrame, calendar_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze price stability, seasonal patterns, and anomalies.

    Calculates coefficient of variation over time, detects price jumps >200%,
    and returns price behavior statistics.

    Parameters
    ----------
    sell_prices_df : pd.DataFrame
        M5 price data with temporal information
    calendar_df : pd.DataFrame
        M5 calendar data for temporal alignment

    Returns
    -------
    Dict[str, Any]
        Price behavior statistics including stability, patterns, and anomalies
    """
    result = {}

    # Calculate coefficient of variation for each item
    coefficient_of_variation = {}

    for item_id in sell_prices_df['item_id'].unique():
        item_prices = sell_prices_df[sell_prices_df['item_id'] == item_id]['sell_price']

        if len(item_prices) > 1 and item_prices.mean() > 0:
            cv = float(item_prices.std() / item_prices.mean())
            coefficient_of_variation[item_id] = cv

    result['coefficient_of_variation'] = coefficient_of_variation

    # Analyze price stability (average CV)
    if coefficient_of_variation:
        avg_cv = np.mean(list(coefficient_of_variation.values()))
        result['price_stability'] = {
            'average_cv': float(avg_cv),
            'stability_rating': 'stable' if avg_cv < 0.2 else 'moderate' if avg_cv < 0.5 else 'volatile'
        }
    else:
        result['price_stability'] = {'average_cv': 0.0, 'stability_rating': 'unknown'}

    # Analyze seasonal patterns by aligning with calendar
    seasonal_patterns = {}
    if 'wm_yr_wk' in sell_prices_df.columns and 'wm_yr_wk' in calendar_df.columns:
        # Merge price data with calendar month information
        price_calendar = sell_prices_df.merge(calendar_df, on='wm_yr_wk', how='inner')

        if 'month' in price_calendar.columns:
            monthly_prices = price_calendar.groupby('month')['sell_price'].mean()
            seasonal_patterns['monthly_avg_prices'] = monthly_prices.to_dict()

    result['seasonal_patterns'] = seasonal_patterns

    # Detect price anomalies (jumps >200%)
    price_anomalies = []

    for item_id in sell_prices_df['item_id'].unique():
        item_data = sell_prices_df[sell_prices_df['item_id'] == item_id].sort_values('wm_yr_wk')

        if len(item_data) > 1:
            prices = item_data['sell_price'].values
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    percent_change = ((prices[i] - prices[i-1]) / prices[i-1]) * 100

                    if abs(percent_change) > 200:
                        price_anomalies.append({
                            'item_id': item_id,
                            'week': item_data.iloc[i]['wm_yr_wk'],
                            'previous_price': float(prices[i-1]),
                            'current_price': float(prices[i]),
                            'jump_percentage': float(abs(percent_change))
                        })

    result['price_anomalies'] = price_anomalies

    return result


def calculate_feature_importance_rankings(correlations_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Rank features by correlation strength with sales.

    Returns sorted importance rankings with statistical significance flags.

    Parameters
    ----------
    correlations_dict : Dict[str, float]
        Dictionary mapping feature names to correlation coefficients

    Returns
    -------
    Dict[str, Any]
        Sorted importance rankings and significance flags
    """
    result = {'rankings': [], 'absolute_rankings': [], 'significance_flags': {}}

    if not correlations_dict:
        return result

    # Filter out NaN values
    valid_correlations = {
        feature: corr for feature, corr in correlations_dict.items()
        if not np.isnan(corr)
    }

    if not valid_correlations:
        return result

    # Create rankings sorted by actual correlation (preserving direction)
    sorted_correlations = sorted(valid_correlations.items(),
                                key=lambda x: x[1],
                                reverse=True)

    rankings = []
    for rank, (feature, correlation) in enumerate(sorted_correlations, 1):
        rankings.append({
            'rank': rank,
            'feature': feature,
            'correlation': float(correlation)
        })

    result['rankings'] = rankings

    # Create rankings sorted by absolute correlation strength
    sorted_absolute = sorted(valid_correlations.items(),
                           key=lambda x: abs(x[1]),
                           reverse=True)

    absolute_rankings = []
    for rank, (feature, correlation) in enumerate(sorted_absolute, 1):
        absolute_rankings.append({
            'rank': rank,
            'feature': feature,
            'correlation': float(correlation),
            'absolute_correlation': float(abs(correlation))
        })

    result['absolute_rankings'] = absolute_rankings

    # Statistical significance flags (simplified heuristic)
    significance_flags = {}
    for feature, correlation in valid_correlations.items():
        # Simple heuristic: correlations with |r| > 0.3 are considered "significant"
        # In real analysis, this would use proper statistical tests with p-values
        significance_flags[feature] = {
            'is_significant': abs(correlation) > 0.3,
            'strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'
        }

    result['significance_flags'] = significance_flags

    return result