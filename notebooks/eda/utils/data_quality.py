"""
Data quality analysis module for M5 demand forecasting dataset.

Provides functions for:
- Missing value pattern analysis
- Missing mechanism characterization
- Sales outlier detection with category-specific thresholds
- Pricing anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import warnings

warnings.filterwarnings("ignore")


def analyze_missing_patterns(
    sales_data: pd.DataFrame,
    pricing_data: Optional[pd.DataFrame] = None,
    calendar_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Analyze missing value patterns in sales, pricing, and calendar data.

    Validates sales data completeness (should have no missing values),
    identifies pricing gaps by item-store-week combinations,
    and verifies calendar data completeness.

    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales data with d_1, d_2, ... columns
    pricing_data : pd.DataFrame, optional
        Pricing data with columns: item_id, store_id, wm_yr_wk, sell_price
    calendar_data : pd.DataFrame, optional
        Calendar data with columns: d, date, and other temporal features

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - sales_completeness: Sales data missing value statistics
        - pricing_gaps: Analysis of missing pricing combinations
        - calendar_completeness: Calendar data validation
        - missing_correlations: Business logic correlations with missingness

    Raises
    ------
    ValueError
        If input dataframe is empty or missing required columns
    KeyError
        If required columns are not found in dataframe
    """
    if sales_data.empty:
        raise ValueError("Sales dataframe is empty")

    if 'item_id' not in sales_data.columns:
        raise KeyError("'item_id' column not found in sales_data")

    # Initialize results dictionary
    result = {}

    # 1. Analyze sales data completeness
    result['sales_completeness'] = _analyze_sales_completeness(sales_data)

    # 2. Analyze pricing gaps if provided
    if pricing_data is not None and not pricing_data.empty:
        result['pricing_gaps'] = _analyze_pricing_gaps(sales_data, pricing_data)

    # 3. Analyze calendar completeness if provided
    if calendar_data is not None and not calendar_data.empty:
        result['calendar_completeness'] = _analyze_calendar_completeness(calendar_data)

    # 4. Analyze missing correlations with business logic
    result['missing_correlations'] = _analyze_missing_correlations(sales_data)

    return result


def _analyze_sales_completeness(sales_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze sales data for missing values and zeros."""
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]

    if len(sales_cols) == 0:
        raise ValueError("No daily sales columns (d_*) found in sales data")

    # Count missing values across all sales columns
    sales_matrix = sales_data[sales_cols]
    missing_count = sales_matrix.isna().sum().sum()
    total_values = sales_matrix.size
    missing_pct = (missing_count / total_values * 100) if total_values > 0 else 0

    # Identify series with high zero percentages (may indicate availability issues)
    zero_pcts = (sales_matrix == 0).sum(axis=1) / len(sales_cols) * 100

    return {
        'total_days': len(sales_cols),
        'total_series': len(sales_data),
        'missing_count': int(missing_count),
        'missing_percent': round(missing_pct, 4),
        'is_complete': bool(missing_count == 0),
        'zero_heavy_series': {
            'count': int((zero_pcts >= 80).sum()),
            'percent': round((zero_pcts >= 80).sum() / len(sales_data) * 100, 2)
        },
        'series_zero_stats': {
            'mean': round(zero_pcts.mean(), 2),
            'median': round(zero_pcts.median(), 2),
            'max': round(zero_pcts.max(), 2),
            'min': round(zero_pcts.min(), 2)
        }
    }


def _analyze_pricing_gaps(sales_data: pd.DataFrame, pricing_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze pricing data gaps for item-store-week combinations."""
    # Get unique item-store combinations in sales
    sales_combinations = set(
        zip(sales_data['item_id'].astype(str), sales_data.get('store_id', '').astype(str))
    )

    # Get unique item-store combinations in pricing
    pricing_combinations = set(
        zip(pricing_data['item_id'].astype(str), pricing_data['store_id'].astype(str))
    )

    # Find gaps
    missing_in_pricing = sales_combinations - pricing_combinations
    extra_in_pricing = pricing_combinations - sales_combinations

    return {
        'total_sales_combinations': len(sales_combinations),
        'total_pricing_combinations': len(pricing_combinations),
        'missing_in_pricing': len(missing_in_pricing),
        'missing_percent': round(len(missing_in_pricing) / len(sales_combinations) * 100, 2) if sales_combinations else 0,
        'extra_in_pricing': len(extra_in_pricing),
        'coverage_percent': round(len(pricing_combinations & sales_combinations) / len(sales_combinations) * 100, 2) if sales_combinations else 0
    }


def _analyze_calendar_completeness(calendar_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze calendar data completeness and date continuity."""
    if 'd' not in calendar_data.columns or 'date' not in calendar_data.columns:
        raise KeyError("Calendar data must have 'd' and 'date' columns")

    calendar_data = calendar_data.copy()
    calendar_data['date'] = pd.to_datetime(calendar_data['date'])

    # Check for missing date continuity
    date_range = pd.date_range(
        start=calendar_data['date'].min(),
        end=calendar_data['date'].max(),
        freq='D'
    )

    missing_dates = set(date_range) - set(calendar_data['date'])

    return {
        'total_days': len(calendar_data),
        'date_range': {
            'start': str(calendar_data['date'].min().date()),
            'end': str(calendar_data['date'].max().date()),
            'span_days': (calendar_data['date'].max() - calendar_data['date'].min()).days + 1
        },
        'missing_dates_count': len(missing_dates),
        'is_complete': len(missing_dates) == 0,
        'duplicate_dates': int(calendar_data['date'].duplicated().sum())
    }


def _analyze_missing_correlations(sales_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze correlations between missing patterns and business logic."""
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]

    if len(sales_cols) == 0:
        return {}

    sales_matrix = sales_data[sales_cols]

    # Check if missing values correlate with specific categories
    correlations = {}

    if 'cat_id' in sales_data.columns:
        for cat in sales_data['cat_id'].unique():
            cat_mask = sales_data['cat_id'] == cat
            cat_sales = sales_matrix[cat_mask]
            cat_missing_pct = (cat_sales.isna().sum().sum() / cat_sales.size * 100) if cat_sales.size > 0 else 0
            correlations[f'category_{cat}_missing_pct'] = round(cat_missing_pct, 4)

    # Check if missing values correlate with specific stores
    if 'store_id' in sales_data.columns:
        store_correlations = {}
        for store in sales_data['store_id'].unique():
            store_mask = sales_data['store_id'] == store
            store_sales = sales_matrix[store_mask]
            store_missing_pct = (store_sales.isna().sum().sum() / store_sales.size * 100) if store_sales.size > 0 else 0
            store_correlations[str(store)] = round(store_missing_pct, 4)
        correlations['by_store'] = store_correlations

    return correlations


def characterize_missing_mechanisms(
    sales_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Identify types of missing mechanisms in the data.

    Characterizes patterns like:
    - Seasonal availability (products only sold during certain periods)
    - Geographic availability (products sold in certain stores/regions)
    - New product introduction (sales start mid-timeline)
    - Discontinued products (sales end before timeline end)

    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales data with d_1, d_2, ... columns

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - mechanisms: List of identified missing mechanisms
        - seasonal_items: Items with seasonal availability patterns
        - new_products: Items with introduction patterns
        - discontinued_items: Items with discontinuation patterns
        - geographic_restrictions: Items with geographic availability patterns

    Raises
    ------
    ValueError
        If input dataframe is empty or missing required columns
    """
    if sales_data.empty:
        raise ValueError("Sales dataframe is empty")

    if 'item_id' not in sales_data.columns:
        raise KeyError("'item_id' column not found in sales_data")

    result = {}
    mechanisms = []

    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]
    if len(sales_cols) == 0:
        raise ValueError("No daily sales columns (d_*) found in sales data")

    sales_matrix = sales_data[sales_cols].fillna(0)

    # 1. Detect seasonal availability patterns
    seasonal_items = _detect_seasonal_patterns(sales_data, sales_matrix)
    if seasonal_items:
        mechanisms.append('seasonal_availability')
        result['seasonal_items'] = seasonal_items

    # 2. Detect new product introductions
    new_products = _detect_new_products(sales_data, sales_matrix)
    if new_products:
        mechanisms.append('new_product_introduction')
        result['new_products'] = new_products

    # 3. Detect discontinued products
    discontinued = _detect_discontinued_products(sales_data, sales_matrix)
    if discontinued:
        mechanisms.append('product_discontinuation')
        result['discontinued_items'] = discontinued

    # 4. Detect geographic restrictions
    if 'store_id' in sales_data.columns:
        geographic = _detect_geographic_restrictions(sales_data, sales_matrix)
        if geographic:
            mechanisms.append('geographic_availability')
            result['geographic_restrictions'] = geographic

    result['mechanisms'] = mechanisms
    return result


def _detect_seasonal_patterns(
    sales_data: pd.DataFrame,
    sales_matrix: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Detect items with seasonal availability patterns."""
    seasonal_items = []

    for idx, (item_id, row) in enumerate(sales_data[['item_id']].iterrows()):
        sales_values = sales_matrix.iloc[idx].values
        non_zero_idx = np.where(sales_values > 0)[0]

        if len(non_zero_idx) == 0:
            continue

        # Check if sales are concentrated in a specific period
        span = non_zero_idx[-1] - non_zero_idx[0] + 1
        concentration = len(non_zero_idx) / len(sales_values)

        # If sales are concentrated in <50% of timeline, likely seasonal
        if 0.05 < concentration < 0.5:
            seasonal_items.append({
                'item_id': item_id,
                'sales_concentration': round(concentration, 4),
                'active_period_start': int(non_zero_idx[0]) + 1,
                'active_period_end': int(non_zero_idx[-1]) + 1,
                'active_days': int(len(non_zero_idx))
            })

    return seasonal_items


def _detect_new_products(
    sales_data: pd.DataFrame,
    sales_matrix: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Detect items with new product introduction patterns."""
    new_products = []

    for idx, (item_id, row) in enumerate(sales_data[['item_id']].iterrows()):
        sales_values = sales_matrix.iloc[idx].values
        non_zero_idx = np.where(sales_values > 0)[0]

        if len(non_zero_idx) == 0:
            continue

        first_sale_idx = non_zero_idx[0]

        # If first sale is in second half of timeline, likely new product
        if first_sale_idx > len(sales_values) * 0.5:
            new_products.append({
                'item_id': item_id,
                'introduction_day': int(first_sale_idx) + 1,
                'introduction_percent': round((first_sale_idx / len(sales_values)) * 100, 2),
                'sales_since_introduction': int(len(non_zero_idx))
            })

    return new_products


def _detect_discontinued_products(
    sales_data: pd.DataFrame,
    sales_matrix: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Detect items with product discontinuation patterns."""
    discontinued = []

    for idx, (item_id, row) in enumerate(sales_data[['item_id']].iterrows()):
        sales_values = sales_matrix.iloc[idx].values
        non_zero_idx = np.where(sales_values > 0)[0]

        if len(non_zero_idx) == 0:
            continue

        last_sale_idx = non_zero_idx[-1]
        timeline_length = len(sales_values)

        # If last sale is in first half of timeline, likely discontinued
        if last_sale_idx < timeline_length * 0.5:
            discontinued.append({
                'item_id': item_id,
                'last_sale_day': int(last_sale_idx) + 1,
                'discontinuation_percent': round((last_sale_idx / timeline_length) * 100, 2),
                'days_since_discontinuation': int(timeline_length - last_sale_idx - 1)
            })

    return discontinued


def _detect_geographic_restrictions(
    sales_data: pd.DataFrame,
    sales_matrix: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Detect items with geographic availability restrictions."""
    geographic = []

    if 'item_id' not in sales_data.columns or 'store_id' not in sales_data.columns:
        return geographic

    for item_id in sales_data['item_id'].unique():
        item_mask = sales_data['item_id'] == item_id
        item_sales = sales_matrix[item_mask]
        item_stores = sales_data[item_mask]['store_id'].unique()

        # Calculate availability percentage per store
        store_availability = {}
        for store_idx, store in enumerate(item_stores):
            store_mask = (sales_data['item_id'] == item_id) & (sales_data['store_id'] == store)
            store_sales = sales_matrix[store_mask].iloc[0].values if (store_mask).any() else np.array([0])
            availability = (store_sales > 0).sum() / len(store_sales) if len(store_sales) > 0 else 0
            store_availability[str(store)] = round(availability * 100, 2)

        # If significant variation across stores, geographic restriction exists
        availabilities = list(store_availability.values())
        if availabilities and (max(availabilities) - min(availabilities)) > 20:
            geographic.append({
                'item_id': item_id,
                'store_availability_variation': round(max(availabilities) - min(availabilities), 2),
                'store_availability': store_availability,
                'stores_with_item': len(item_stores)
            })

    return geographic


def detect_sales_outliers(
    sales_data: pd.DataFrame,
    method: str = 'business_rules'
) -> Dict[str, Any]:
    """
    Detect outliers in sales data using category-specific thresholds.

    Uses retail-specific business rules:
    - FOODS: >50 units/day is suspicious
    - HOUSEHOLD: >20 units/day is suspicious
    - HOBBIES: >100 units/day is suspicious

    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales data with d_1, d_2, ... columns and cat_id column
    method : str, default='business_rules'
        Outlier detection method ('business_rules' or 'statistical')

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - outliers_detected: Number of outlier values
        - category_thresholds: Thresholds used for each category
        - outlier_distribution: Breakdown by category
        - business_rule_violations: Invalid values (negative sales, etc.)
        - promotional_spikes: Potential promotional sales identified

    Raises
    ------
    ValueError
        If input dataframe is empty or missing required columns
    KeyError
        If cat_id column is not found
    """
    if sales_data.empty:
        raise ValueError("Sales dataframe is empty")

    if 'cat_id' not in sales_data.columns:
        raise KeyError("'cat_id' column not found in sales_data")

    result = {}

    # Define category-specific thresholds (business rules for retail)
    category_thresholds = {
        'FOODS': 50,
        'HOUSEHOLD': 20,
        'HOBBIES': 100
    }

    result['category_thresholds'] = category_thresholds

    # Get all sales columns
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]
    if len(sales_cols) == 0:
        raise ValueError("No daily sales columns (d_*) found in sales data")

    # Check for business rule violations
    violations = _check_business_rule_violations(sales_data, sales_cols)
    result['business_rule_violations'] = violations

    # Detect outliers by category
    outlier_data = []
    outlier_distribution = {}

    for cat in sales_data['cat_id'].unique():
        if cat not in category_thresholds:
            threshold = 50  # Default threshold
        else:
            threshold = category_thresholds[cat]

        cat_mask = sales_data['cat_id'] == cat
        cat_sales = sales_data[cat_mask][sales_cols].values.flatten()

        # Remove NaN values
        cat_sales = cat_sales[~np.isnan(cat_sales)]

        # Find outliers
        outliers = cat_sales[cat_sales > threshold]
        outlier_distribution[cat] = {
            'count': int(len(outliers)),
            'percent': round((len(outliers) / len(cat_sales) * 100) if len(cat_sales) > 0 else 0, 4),
            'threshold': threshold,
            'max_value': int(np.max(cat_sales)) if len(cat_sales) > 0 else 0,
            'mean_value': round(np.mean(cat_sales), 2) if len(cat_sales) > 0 else 0
        }

        outlier_data.extend(outliers)

    result['outliers_detected'] = len(outlier_data)
    result['total_outliers'] = len(outlier_data)
    result['outlier_distribution'] = outlier_distribution

    # Detect promotional spikes
    promotional_spikes = _detect_promotional_spikes(sales_data, sales_cols, category_thresholds)
    result['promotional_spikes'] = promotional_spikes

    return result


def _check_business_rule_violations(
    sales_data: pd.DataFrame,
    sales_cols: List[str]
) -> Dict[str, Any]:
    """Check for business rule violations (negative sales, etc.)."""
    violations = {}

    # Check for negative sales
    negative_count = 0
    for col in sales_cols:
        if col in sales_data.columns:
            neg = (sales_data[col] < 0).sum()
            negative_count += neg

    violations['negative_sales'] = {
        'count': int(negative_count),
        'is_violation': negative_count > 0
    }

    # Check for unrealistic values (>10000 units/day)
    unrealistic_count = 0
    for col in sales_cols:
        if col in sales_data.columns:
            unreal = (sales_data[col] > 10000).sum()
            unrealistic_count += unreal

    violations['unrealistic_values'] = {
        'count': int(unrealistic_count),
        'threshold': 10000,
        'is_violation': unrealistic_count > 0
    }

    return violations


def _detect_promotional_spikes(
    sales_data: pd.DataFrame,
    sales_cols: List[str],
    category_thresholds: Dict[str, int]
) -> Dict[str, Any]:
    """Detect potential promotional spikes in sales."""
    spikes = {}

    for cat in sales_data['cat_id'].unique():
        cat_mask = sales_data['cat_id'] == cat
        cat_sales = sales_data[cat_mask][sales_cols]

        spike_count = 0
        for idx, row in cat_sales.iterrows():
            sales_vals = row.dropna().values

            if len(sales_vals) < 2:
                continue

            # Check if any value is >2x the mean
            mean_sales = np.mean(sales_vals)
            max_sales = np.max(sales_vals)

            if mean_sales > 0 and max_sales > mean_sales * 2:
                spike_count += 1

        spikes[cat] = {
            'potential_spikes': int(spike_count),
            'series_with_spikes_percent': round((spike_count / len(cat_sales) * 100) if len(cat_sales) > 0 else 0, 2)
        }

    return spikes


def analyze_pricing_anomalies(
    pricing_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Detect pricing anomalies and inconsistencies.

    Identifies:
    - Price jumps (>200% week-over-week changes)
    - Suspicious prices ($0.01, $999.99)
    - Cross-store price inconsistencies
    - Negative prices (invalid)
    - Promotional pricing patterns

    Parameters
    ----------
    pricing_data : pd.DataFrame
        Pricing data with columns: item_id, store_id, wm_yr_wk, sell_price

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - price_jumps: Week-over-week price change anomalies
        - suspicious_prices: Extreme or suspicious price values
        - cross_store_inconsistency: Price variation across stores
        - invalid_prices: Negative or zero prices
        - promotional_patterns: Potential promotional pricing

    Raises
    ------
    ValueError
        If input dataframe is empty or missing required columns
    KeyError
        If required columns are not found
    """
    if pricing_data.empty:
        raise ValueError("Pricing dataframe is empty")

    required_cols = ['item_id', 'store_id', 'wm_yr_wk', 'sell_price']
    for col in required_cols:
        if col not in pricing_data.columns:
            raise KeyError(f"'{col}' column not found in pricing_data")

    result = {}

    # 1. Detect price jumps
    result['price_jumps'] = _detect_price_jumps(pricing_data)

    # 2. Detect suspicious prices
    result['suspicious_prices'] = _detect_suspicious_prices(pricing_data)

    # 3. Check cross-store inconsistency
    result['cross_store_inconsistency'] = _check_cross_store_inconsistency(pricing_data)

    # 4. Check for invalid prices
    result['invalid_prices'] = _check_invalid_prices(pricing_data)

    # 5. Detect promotional patterns
    result['promotional_patterns'] = _detect_promotional_patterns(pricing_data)

    return result


def _detect_price_jumps(pricing_data: pd.DataFrame) -> Dict[str, Any]:
    """Detect large week-over-week price changes (>200%)."""
    jumps = {
        'count': 0,
        'items_with_jumps': []
    }

    # Group by item and store, sort by week
    for (item_id, store_id), group in pricing_data.groupby(['item_id', 'store_id']):
        group = group.sort_values('wm_yr_wk')
        prices = group['sell_price'].values

        if len(prices) < 2:
            continue

        # Calculate week-over-week changes
        for i in range(1, len(prices)):
            prev_price = prices[i - 1]
            curr_price = prices[i]

            if prev_price > 0:
                pct_change = abs((curr_price - prev_price) / prev_price * 100)

                if pct_change > 200:
                    jumps['count'] += 1
                    jumps['items_with_jumps'].append({
                        'item_id': item_id,
                        'store_id': store_id,
                        'price_change_percent': round(pct_change, 2),
                        'from_price': round(prev_price, 2),
                        'to_price': round(curr_price, 2)
                    })

    return jumps


def _detect_suspicious_prices(pricing_data: pd.DataFrame) -> Dict[str, Any]:
    """Detect suspicious prices like $0.01 or $999.99."""
    suspicious = {
        'penny_prices': [],
        'extreme_prices': [],
        'count': 0
    }

    # Check for penny prices ($0.01)
    penny_mask = pricing_data['sell_price'] == 0.01
    if penny_mask.any():
        penny_items = pricing_data[penny_mask][['item_id', 'store_id', 'wm_yr_wk']].to_dict('records')
        suspicious['penny_prices'] = [
            {'item_id': r['item_id'], 'store_id': r['store_id'], 'week': r['wm_yr_wk']}
            for r in penny_items
        ]
        suspicious['count'] += len(penny_items)

    # Check for extreme prices (>$100)
    extreme_mask = pricing_data['sell_price'] > 100
    if extreme_mask.any():
        extreme_items = pricing_data[extreme_mask][['item_id', 'store_id', 'sell_price']].to_dict('records')
        suspicious['extreme_prices'] = [
            {'item_id': r['item_id'], 'store_id': r['store_id'], 'price': round(r['sell_price'], 2)}
            for r in extreme_items
        ]
        suspicious['count'] += len(extreme_items)

    return suspicious


def _check_cross_store_inconsistency(pricing_data: pd.DataFrame) -> Dict[str, Any]:
    """Check for cross-store price inconsistencies."""
    inconsistencies = {
        'items_with_inconsistency': [],
        'count': 0
    }

    # Group by item and week to find same item sold at different prices
    for (item_id, week), group in pricing_data.groupby(['item_id', 'wm_yr_wk']):
        prices = group['sell_price'].values

        if len(prices) > 1:
            price_variation = (np.max(prices) - np.min(prices)) / np.mean(prices) * 100

            # If variation > 20%, note it
            if price_variation > 20:
                inconsistencies['items_with_inconsistency'].append({
                    'item_id': item_id,
                    'week': int(week),
                    'price_variation_percent': round(price_variation, 2),
                    'stores': len(group),
                    'min_price': round(np.min(prices), 2),
                    'max_price': round(np.max(prices), 2)
                })
                inconsistencies['count'] += 1

    return inconsistencies


def _check_invalid_prices(pricing_data: pd.DataFrame) -> Dict[str, Any]:
    """Check for invalid prices (negative or zero)."""
    invalid = {
        'negative_prices': [],
        'zero_prices': [],
        'count': 0
    }

    # Check for negative prices
    neg_mask = pricing_data['sell_price'] < 0
    if neg_mask.any():
        neg_items = pricing_data[neg_mask][['item_id', 'store_id', 'sell_price']].to_dict('records')
        invalid['negative_prices'] = [
            {'item_id': r['item_id'], 'store_id': r['store_id'], 'price': r['sell_price']}
            for r in neg_items
        ]
        invalid['count'] += len(neg_items)

    # Check for zero prices
    zero_mask = pricing_data['sell_price'] == 0
    if zero_mask.any():
        zero_items = pricing_data[zero_mask][['item_id', 'store_id']].to_dict('records')
        invalid['zero_prices'] = [
            {'item_id': r['item_id'], 'store_id': r['store_id']}
            for r in zero_items
        ]
        invalid['count'] += len(zero_items)

    return invalid


def _detect_promotional_patterns(pricing_data: pd.DataFrame) -> Dict[str, Any]:
    """Detect potential promotional pricing patterns."""
    patterns = {
        'items_with_promotions': [],
        'count': 0
    }

    # For each item-store, identify promotional periods
    for (item_id, store_id), group in pricing_data.groupby(['item_id', 'store_id']):
        group = group.sort_values('wm_yr_wk')
        prices = group['sell_price'].values

        if len(prices) < 2:
            continue

        # Find periods where price is below average (likely promotional)
        avg_price = np.mean(prices)
        promo_threshold = avg_price * 0.8  # 20% discount

        for i, price in enumerate(prices):
            if price < promo_threshold and price > 0:
                patterns['items_with_promotions'].append({
                    'item_id': item_id,
                    'store_id': store_id,
                    'week': int(group['wm_yr_wk'].iloc[i]),
                    'promotional_price': round(price, 2),
                    'average_price': round(avg_price, 2),
                    'discount_percent': round((1 - price / avg_price) * 100, 2)
                })
                patterns['count'] += 1

    return patterns
