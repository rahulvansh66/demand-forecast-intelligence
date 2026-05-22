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
