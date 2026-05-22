"""
Correlation and categorical pattern analysis module.

This module provides functionality for analyzing relationships between
features and target variables, with a focus on categorical sales patterns
and business-meaningful statistical analysis.
"""

from typing import Dict, Any, Tuple
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
    influence sales behavior.

    Args:
        data (pd.DataFrame): Input dataframe containing categorical and sales data.
        category_col (str): Name of the categorical column to group by.
        sales_col (str): Name of the sales column to analyze.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'categories': Unique categories found
            - 'summary_stats': Overall statistical summary (mean, std, count)
            - 'by_category': Detailed statistics for each category

    Raises:
        ValueError: If specified columns don't exist in dataframe.
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

    # Extract unique categories
    categories = data[category_col].unique().tolist()

    # Calculate overall summary statistics
    summary_stats = {
        'mean': float(data[sales_col].mean()),
        'std': float(data[sales_col].std()),
        'count': len(data),
        'min': float(data[sales_col].min()),
        'max': float(data[sales_col].max()),
        'median': float(data[sales_col].median())
    }

    # Calculate per-category statistics
    by_category = {}
    grouped = data.groupby(category_col)[sales_col]
    for category in categories:
        category_data = grouped.get_group(category)
        by_category[category] = {
            'mean': float(category_data.mean()),
            'std': float(category_data.std()),
            'count': len(category_data),
            'min': float(category_data.min()),
            'max': float(category_data.max()),
            'median': float(category_data.median())
        }

    return {
        'categories': categories,
        'summary_stats': summary_stats,
        'by_category': by_category
    }
