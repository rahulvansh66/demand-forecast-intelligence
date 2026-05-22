"""
Correlation and categorical pattern analysis module.

This module provides functionality for analyzing relationships between
features and target variables, with a focus on categorical sales patterns
and business-meaningful statistical analysis.
"""

from typing import Dict, Any, List, Set
import pandas as pd
import numpy as np

# Constants for multicollinearity analysis
VIF_MEDIUM_THRESHOLD = 5.0
VIF_HIGH_THRESHOLD = 10.0
VIF_EPSILON = 0.01  # Small value to prevent division by zero


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


def compute_cross_feature_correlations(
    sales_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyze cross-feature correlations for business-meaningful patterns.

    Examines relationships between:
    - Product hierarchy (category-department relationships)
    - Geographic patterns (stores within states)
    - Price coordination across locations

    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales data with item_id, store_id, cat_id, dept_id columns
        and daily sales columns (d_*)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - product_hierarchy_correlations: Category-department relationships
        - geographic_correlations: State-level store correlations
        - price_coordination: Price patterns if available
        - notes: Interpretation and business context

    Example
    -------
    >>> result = compute_cross_feature_correlations(sales_data)
    >>> print(result['product_hierarchy_correlations'])
    """
    results = {}

    # Get sales columns
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]

    if len(sales_cols) == 0:
        raise ValueError("No sales columns (d_*) found in sales_data")

    # 1. Product hierarchy correlations
    hierarchy_corr = {}

    if 'cat_id' in sales_data.columns and 'dept_id' in sales_data.columns:
        # Calculate correlation between categories and departments
        category_totals = {}
        dept_totals = {}

        for _, row in sales_data.iterrows():
            cat = row.get('cat_id')
            dept = row.get('dept_id')

            if pd.isna(cat) or pd.isna(dept):
                continue

            # Sum daily sales for this item
            item_sales = row[sales_cols].sum()

            if cat not in category_totals:
                category_totals[cat] = 0
            if dept not in dept_totals:
                dept_totals[dept] = 0

            category_totals[cat] += item_sales
            dept_totals[dept] += item_sales

        # Analyze department-category mapping
        dept_cat_mapping: Dict[str, Set[str]] = {}
        for _, row in sales_data.iterrows():
            cat = row.get('cat_id')
            dept = row.get('dept_id')

            if pd.isna(cat) or pd.isna(dept):
                continue

            if dept not in dept_cat_mapping:
                dept_cat_mapping[dept] = set()

            dept_cat_mapping[dept].add(cat)

        # Calculate correlation strength
        hierarchy_corr = {
            'departments_count': len(dept_cat_mapping),
            'categories_count': len(category_totals),
            'department_category_mapping': {
                dept: list(cats) for dept, cats in dept_cat_mapping.items()
            },
            'category_totals': {cat: float(val) for cat, val in category_totals.items()},
            'dept_totals': {dept: float(val) for dept, val in dept_totals.items()}
        }

    results['product_hierarchy_correlations'] = hierarchy_corr

    # 2. Geographic correlations
    geographic_corr = {}

    if 'store_id' in sales_data.columns:
        # Extract state from store_id (e.g., CA_1 -> CA)
        state_stores: Dict[str, List[str]] = {}
        store_totals: Dict[str, float] = {}

        for _, row in sales_data.iterrows():
            store_id = row.get('store_id')

            if pd.isna(store_id):
                continue

            # Extract state code
            try:
                state = str(store_id).split('_')[0]
            except (IndexError, AttributeError):
                continue

            item_sales = row[sales_cols].sum()

            if state not in state_stores:
                state_stores[state] = []
            if store_id not in store_totals:
                store_totals[store_id] = 0

            state_stores[state].append(store_id)
            store_totals[store_id] += item_sales

        # Calculate correlation between stores within states
        state_correlations = {}

        for state, stores in state_stores.items():
            stores = list(set(stores))  # Unique stores

            if len(stores) < 2:
                state_correlations[state] = {
                    'stores': stores,
                    'store_count': len(stores),
                    'avg_correlation': None
                }
            else:
                # Calculate correlation between store pairs within state
                correlations = []

                for i, store1 in enumerate(stores):
                    for store2 in stores[i + 1:]:
                        # Get sales for each store from data
                        store1_data = sales_data[
                            sales_data['store_id'] == store1
                        ][sales_cols].sum(axis=0).values

                        store2_data = sales_data[
                            sales_data['store_id'] == store2
                        ][sales_cols].sum(axis=0).values

                        if len(store1_data) > 0 and len(store2_data) > 0:
                            corr = np.corrcoef(store1_data, store2_data)[0, 1]
                            if not np.isnan(corr):
                                correlations.append(float(corr))

                avg_corr = float(np.mean(correlations)) if correlations else None

                state_correlations[state] = {
                    'stores': stores,
                    'store_count': len(stores),
                    'avg_correlation': avg_corr,
                    'store_pair_correlations': correlations[:5]  # Show first 5 pairs
                }

        geographic_corr = {
            'states_count': len(state_correlations),
            'state_correlations': state_correlations
        }

    results['geographic_correlations'] = geographic_corr

    # 3. Notes and business context
    results['notes'] = (
        "Product hierarchy shows how categories map to departments. "
        "Geographic correlations reveal demand pattern similarities across stores in same state. "
        "High state-level correlations may indicate regional demand patterns."
    )

    return results


def detect_multicollinearity_issues(
    sales_data: pd.DataFrame,
    threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Detect multicollinearity issues in feature data.

    Identifies highly correlated features that may cause problems in
    forecasting models. Provides VIF-style analysis with business
    interpretation of acceptable vs concerning correlations.

    Parameters
    ----------
    sales_data : pd.DataFrame
        Feature data with numeric columns for correlation analysis
    threshold : float, default=0.8
        Correlation threshold above which pairs are flagged as high correlation

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - high_correlation_pairs: List of feature pairs with high correlation
        - vif_analysis: Variance inflation factors or related metrics
        - business_implications: Interpretation for forecasting
        - recommendations: Actions to address multicollinearity

    Example
    -------
    >>> result = detect_multicollinearity_issues(sales_data, threshold=0.8)
    >>> print(result['high_correlation_pairs'])
    """
    results = {}

    # Select numeric columns only
    numeric_data = sales_data.select_dtypes(include=[np.number])

    if numeric_data.empty:
        return {
            'high_correlation_pairs': [],
            'vif_analysis': {},
            'business_implications': 'No numeric features found',
            'recommendations': []
        }

    if len(numeric_data.columns) < 2:
        return {
            'high_correlation_pairs': [],
            'vif_analysis': {},
            'business_implications': 'Insufficient features for multicollinearity analysis',
            'recommendations': []
        }

    # Drop rows with NaN values for correlation calculation
    clean_data = numeric_data.dropna()

    if len(clean_data) == 0:
        return {
            'high_correlation_pairs': [],
            'vif_analysis': {},
            'business_implications': 'No valid data after removing NaN values',
            'recommendations': []
        }

    # Calculate correlation matrix
    correlation_matrix = clean_data.corr()

    # Find high correlation pairs (excluding self-correlations)
    high_correlation_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            feat1 = correlation_matrix.columns[i]
            feat2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]

            if abs(corr_value) >= threshold:
                high_correlation_pairs.append({
                    'feature_1': str(feat1),
                    'feature_2': str(feat2),
                    'correlation': float(corr_value),
                    'abs_correlation': float(abs(corr_value))
                })

    # Sort by absolute correlation value (highest first)
    high_correlation_pairs = sorted(
        high_correlation_pairs,
        key=lambda x: float(x['abs_correlation']),
        reverse=True
    )

    # Simple VIF-style analysis
    vif_analysis = {}

    for col in clean_data.columns:
        # Calculate correlation with all other features
        col_correlations = correlation_matrix[col].drop(col)
        avg_abs_corr = float(abs(col_correlations).mean())

        # VIF approximation: higher average correlation suggests higher VIF
        vif_score = 1 / (1 - avg_abs_corr + VIF_EPSILON) if avg_abs_corr < 0.99 else 100.0

        vif_analysis[str(col)] = {
            'average_abs_correlation': avg_abs_corr,
            'vif_approximation': float(vif_score),
            'concern_level': 'high' if vif_score > VIF_HIGH_THRESHOLD else 'medium' if vif_score > VIF_MEDIUM_THRESHOLD else 'low'
        }

    # Business implications and recommendations
    if len(high_correlation_pairs) > 0:
        business_implications = (
            f"Found {len(high_correlation_pairs)} highly correlated feature pairs. "
            "High multicollinearity may inflate model uncertainty and reduce interpretability."
        )

        recommendations = [
            "Consider removing one feature from each highly correlated pair",
            "Use domain knowledge to decide which feature is more meaningful",
            "Consider PCA or other dimensionality reduction techniques",
            "Monitor feature importance in forecasting models"
        ]
    else:
        business_implications = "No high correlations detected at current threshold."
        recommendations = ["Current feature set appears suitable for modeling"]

    results['high_correlation_pairs'] = high_correlation_pairs
    results['vif_analysis'] = vif_analysis
    results['business_implications'] = business_implications
    results['recommendations'] = recommendations
    results['threshold_used'] = threshold
    results['correlation_matrix'] = correlation_matrix.to_dict()

    return results
