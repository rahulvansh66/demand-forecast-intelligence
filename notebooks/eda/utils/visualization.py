"""
Visualization utilities for EDA framework steps 6-10.

Provides static plotting functions for:
- Feature-target relationship plots
- Correlation matrices and heatmaps
- Time series analysis plots
- Missing data visualizations
- Outlier detection plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")


def plot_categorical_sales_distributions(
    sales_data: pd.DataFrame,
    save_path: str,
    category_col: str = 'cat_id',
    sales_col: str = 'daily_sales'
) -> Dict[str, Any]:
    """
    Create box plot distributions for sales by categorical features.

    This function generates a publication-quality box plot showing the
    distribution of sales across different categories. Statistical summaries
    are computed for each category with proper handling of edge cases.

    Parameters
    ----------
    sales_data : pd.DataFrame
        Data containing categorical and sales columns
    save_path : str
        Path to save the plot (e.g., 'path/to/plot.png')
    category_col : str, default='cat_id'
        Name of the categorical column to group by
    sales_col : str, default='daily_sales'
        Name of the sales column to analyze

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - plot_path: Path where plot was saved
        - category_statistics: Statistics for each category (mean, median, std, count)
        - total_samples: Total number of data points

    Raises
    ------
    ValueError
        If dataframe is empty or specified columns don't exist
    KeyError
        If specified columns are not found in dataframe

    Example
    --------
    >>> sales_data = pd.DataFrame({
    ...     'cat_id': ['FOODS', 'HOUSEHOLD'] * 50,
    ...     'daily_sales': np.random.randint(0, 100, 100)
    ... })
    >>> result = plot_categorical_sales_distributions(
    ...     sales_data,
    ...     'plots/category_distributions.png'
    ... )
    >>> print(result['category_statistics'])
    """
    # Input validation
    if sales_data.empty:
        raise ValueError("DataFrame is empty")

    if category_col not in sales_data.columns:
        raise KeyError(f"Column '{category_col}' not found in dataframe. Available columns: {list(sales_data.columns)}")

    if sales_col not in sales_data.columns:
        raise KeyError(f"Column '{sales_col}' not found in dataframe. Available columns: {list(sales_data.columns)}")

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Create figure with appropriate size
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Create box plot
    try:
        sns.boxplot(data=sales_data, x=category_col, y=sales_col, ax=ax)
    except Exception as e:
        plt.close(fig)
        raise ValueError(f"Failed to create box plot: {str(e)}")

    # Customize plot
    ax.set_title('Daily Sales Distribution by Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Product Category', fontsize=12)
    ax.set_ylabel('Daily Sales Volume', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Add sample size annotations
    categories = sales_data[category_col].unique()

    # Handle NaN categories
    categories = [cat for cat in categories if pd.notna(cat)]

    if len(categories) == 0:
        plt.close(fig)
        raise ValueError("No valid categories found after removing NaN values")

    for i, category in enumerate(sorted(categories)):
        category_mask = sales_data[category_col] == category
        n_samples = category_mask.sum()

        # Add annotation if there's space
        if ax.get_ylim()[1] > 0:
            ax.text(i, ax.get_ylim()[1] * 0.95, f'n={n_samples}',
                    ha='center', va='top', fontsize=10)

    plt.tight_layout()

    # Save plot with publication quality (300 DPI)
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        plt.close(fig)
        raise IOError(f"Failed to save plot to {save_path}: {str(e)}")
    finally:
        plt.close(fig)

    # Calculate statistics for each category
    stats_by_category = {}

    for category in sorted(categories):
        cat_mask = sales_data[category_col] == category
        cat_data = sales_data.loc[cat_mask, sales_col].dropna()

        if len(cat_data) > 0:
            stats_by_category[category] = {
                'mean': float(cat_data.mean()),
                'median': float(cat_data.median()),
                'std': float(cat_data.std()),
                'min': float(cat_data.min()),
                'max': float(cat_data.max()),
                'count': int(len(cat_data))
            }
        else:
            stats_by_category[category] = {
                'mean': None,
                'median': None,
                'std': None,
                'min': None,
                'max': None,
                'count': 0
            }

    return {
        'plot_path': save_path,
        'category_statistics': stats_by_category,
        'total_samples': int(len(sales_data)),
        'categories_count': len(categories)
    }
