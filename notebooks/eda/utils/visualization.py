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


def plot_correlation_heatmap(
    data: pd.DataFrame,
    save_path: str,
    title: str = 'Feature Correlation Heatmap',
    cluster: bool = True,
    figsize: tuple = (12, 10)
) -> Dict[str, Any]:
    """
    Create a publication-quality correlation heatmap with hierarchical clustering.

    Generates a correlation matrix visualization with hierarchical clustering
    to identify feature relationships and potential multicollinearity issues.
    Suitable for publication with 300 DPI output.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing numeric features for correlation analysis
    save_path : str
        Path to save the heatmap plot
    title : str, default='Feature Correlation Heatmap'
        Title for the plot
    cluster : bool, default=True
        Whether to apply hierarchical clustering to reorder features
    figsize : tuple, default=(12, 10)
        Figure size as (width, height)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - plot_path: Path where plot was saved
        - correlation_matrix: Full correlation matrix as dict
        - heatmap_metrics: Summary statistics about correlations
        - high_correlations: Pairs with correlation > 0.7

    Raises
    ------
    ValueError
        If dataframe is empty or has no numeric columns
    IOError
        If plot cannot be saved to specified path

    Example
    --------
    >>> result = plot_correlation_heatmap(sales_data, 'plots/correlation.png')
    >>> print(result['high_correlations'])
    """
    # Input validation
    if data.empty:
        raise ValueError("DataFrame is empty")

    # Select numeric columns only
    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.empty:
        raise ValueError("No numeric columns found in dataframe")

    # Remove rows with all NaN values, but keep some rows
    clean_data = numeric_data.dropna(axis=0, how='all')

    if len(clean_data) == 0:
        raise ValueError("No valid data after removing NaN rows")

    # For correlation, drop rows with any NaN in the selected columns
    clean_data = clean_data.dropna()

    if len(clean_data) < 2:
        raise ValueError("Insufficient valid data points for correlation analysis")

    # Calculate correlation matrix
    correlation_matrix = clean_data.corr()

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    try:
        # Create heatmap with clustering
        if cluster and len(correlation_matrix.columns) > 1:
            sns.clustermap(
                correlation_matrix,
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'},
                figsize=figsize
            )

            # Save the clustermap
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close('all')
        else:
            # Regular heatmap without clustering
            sns.heatmap(
                correlation_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation Coefficient'},
                ax=ax,
                square=True
            )

            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    except Exception as e:
        plt.close('all')
        raise IOError(f"Failed to save heatmap to {save_path}: {str(e)}")

    # Identify high correlations (excluding diagonal)
    high_correlations = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            feat1 = correlation_matrix.columns[i]
            feat2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]

            if abs(corr_value) >= 0.7:
                high_correlations.append({
                    'feature_1': str(feat1),
                    'feature_2': str(feat2),
                    'correlation': float(corr_value)
                })

    # Calculate summary metrics
    corr_values = correlation_matrix.values[np.triu_indices_from(
        correlation_matrix.values, k=1
    )]

    heatmap_metrics = {
        'avg_correlation': float(np.mean(np.abs(corr_values))),
        'max_correlation': float(np.max(np.abs(corr_values))),
        'min_correlation': float(np.min(np.abs(corr_values))),
        'high_correlation_count': len(high_correlations),
        'total_feature_pairs': len(corr_values)
    }

    return {
        'plot_path': save_path,
        'correlation_matrix': correlation_matrix.to_dict(),
        'heatmap_metrics': heatmap_metrics,
        'high_correlations': high_correlations,
        'features_analyzed': int(len(correlation_matrix.columns))
    }


def plot_multicollinearity_analysis(
    data: pd.DataFrame,
    save_path: str,
    threshold: float = 0.8,
    figsize: tuple = (14, 6)
) -> Dict[str, Any]:
    """
    Create visualization of multicollinearity analysis with VIF-style metrics.

    Generates a multi-panel figure showing:
    - Correlation heatmap of high-correlation feature pairs
    - VIF (Variance Inflation Factor) approximations
    - Feature recommendations

    Parameters
    ----------
    data : pd.DataFrame
        Feature data with numeric columns
    save_path : str
        Path to save the analysis plot
    threshold : float, default=0.8
        Correlation threshold for flagging high correlations
    figsize : tuple, default=(14, 6)
        Figure size as (width, height)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - plot_path: Path where plot was saved
        - vif_results: VIF scores for each feature
        - high_correlation_pairs: Pairs flagged as problematic
        - business_implications: Interpretation for modeling
        - recommendations: Actions to address multicollinearity

    Raises
    ------
    ValueError
        If dataframe is empty or insufficient numeric features
    IOError
        If plot cannot be saved

    Example
    --------
    >>> result = plot_multicollinearity_analysis(sales_data, 'plots/multicollinearity.png')
    >>> print(result['recommendations'])
    """
    # Input validation
    if data.empty:
        raise ValueError("DataFrame is empty")

    # Select numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.empty:
        raise ValueError("No numeric columns found in dataframe")

    if len(numeric_data.columns) < 2:
        raise ValueError("At least 2 numeric features required for multicollinearity analysis")

    # Clean data
    clean_data = numeric_data.dropna()

    if len(clean_data) < 2:
        raise ValueError("Insufficient valid data after removing NaN values")

    # Calculate correlation matrix
    correlation_matrix = clean_data.corr()

    # Identify high correlation pairs
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
                    'correlation': float(corr_value)
                })

    # Calculate VIF approximations
    vif_results = {}

    for col in clean_data.columns:
        col_correlations = correlation_matrix[col].drop(col)
        avg_abs_corr = float(abs(col_correlations).mean())

        # VIF approximation
        vif_score = 1 / (1 - avg_abs_corr + 0.01) if avg_abs_corr < 0.99 else 100.0

        vif_results[str(col)] = {
            'vif_score': float(vif_score),
            'concern': 'high' if vif_score > 10 else 'medium' if vif_score > 5 else 'low'
        }

    # Create visualization
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    try:
        # Left panel: Correlation heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            cbar_kws={'label': 'Correlation'},
            ax=axes[0],
            square=True
        )
        axes[0].set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

        # Right panel: VIF scores bar plot
        features = list(vif_results.keys())
        vif_scores = [vif_results[feat]['vif_score'] for feat in features]

        # Color bars based on concern level
        colors = [
            'red' if vif_results[feat]['concern'] == 'high' else
            'orange' if vif_results[feat]['concern'] == 'medium' else
            'green'
            for feat in features
        ]

        axes[1].barh(features, vif_scores, color=colors, alpha=0.7)
        axes[1].axvline(x=5, color='orange', linestyle='--', label='Medium Concern (VIF=5)')
        axes[1].axvline(x=10, color='red', linestyle='--', label='High Concern (VIF=10)')
        axes[1].set_xlabel('VIF Approximation Score', fontsize=11)
        axes[1].set_title('Variance Inflation Factor (VIF) Analysis', fontsize=12, fontweight='bold')
        axes[1].legend(loc='lower right', fontsize=9)
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        plt.close('all')
        raise IOError(f"Failed to save multicollinearity plot to {save_path}: {str(e)}")

    # Generate business implications and recommendations
    if len(high_correlation_pairs) > 0:
        business_implications = (
            f"Detected {len(high_correlation_pairs)} highly correlated feature pairs (correlation > {threshold}). "
            "High multicollinearity can inflate model coefficients and reduce interpretability."
        )

        recommendations = [
            "Remove one feature from each highly correlated pair based on domain knowledge",
            f"Features with VIF > 10 show very high correlation and should be prioritized for removal",
            "Consider feature engineering or PCA for dimensionality reduction",
            "Re-evaluate feature correlations after removing problematic features"
        ]
    else:
        business_implications = f"No feature pairs exceed correlation threshold of {threshold}."
        recommendations = ["Current feature set appears suitable for modeling"]

    return {
        'plot_path': save_path,
        'vif_results': vif_results,
        'high_correlation_pairs': high_correlation_pairs,
        'business_implications': business_implications,
        'recommendations': recommendations,
        'threshold_used': threshold,
        'correlation_matrix': correlation_matrix.to_dict()
    }
