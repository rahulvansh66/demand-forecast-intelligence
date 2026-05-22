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
from typing import Dict, Any
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Constants for VIF analysis visualization
VIF_MEDIUM_THRESHOLD = 5.0
VIF_HIGH_THRESHOLD = 10.0
VIF_EPSILON = 0.01  # Small value to prevent division by zero

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
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
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
        vif_score = 1 / (1 - avg_abs_corr + VIF_EPSILON) if avg_abs_corr < 0.99 else 100.0

        vif_results[str(col)] = {
            'vif_score': float(vif_score),
            'concern': 'high' if vif_score > VIF_HIGH_THRESHOLD else 'medium' if vif_score > VIF_MEDIUM_THRESHOLD else 'low'
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
        axes[1].axvline(x=VIF_MEDIUM_THRESHOLD, color='orange', linestyle='--', label=f'Medium Concern (VIF={VIF_MEDIUM_THRESHOLD})')
        axes[1].axvline(x=VIF_HIGH_THRESHOLD, color='red', linestyle='--', label=f'High Concern (VIF={VIF_HIGH_THRESHOLD})')
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
            f"Features with VIF > {VIF_HIGH_THRESHOLD} show very high correlation and should be prioritized for removal",
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


def plot_seasonal_decomposition(
    time_series_data: pd.Series,
    save_path: str,
    title: str = "Seasonal Decomposition Analysis"
) -> Dict[str, Any]:
    """
    Create seasonal decomposition plots for time series data.

    Generates publication-ready 4-panel decomposition showing original time series,
    trend component (moving average), seasonal component (weekly pattern), and residuals.
    Includes statistical metrics for trend and seasonality strength.

    Parameters
    ----------
    time_series_data : pd.Series
        Time series data to decompose
    save_path : str
        Path to save the plot
    title : str, default "Seasonal Decomposition Analysis"
        Plot title

    Returns
    -------
    Dict[str, Any]
        Plot metadata and decomposition statistics
    """
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Create subplots for decomposition components
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Original time series
    axes[0].plot(time_series_data, color='blue', linewidth=1)
    axes[0].set_title('Original Time Series', fontsize=12)
    axes[0].set_ylabel('Sales Volume')

    # Simple trend (moving average)
    window = min(30, len(time_series_data) // 4)  # 30-day or 1/4 of data
    if window > 1:
        trend = time_series_data.rolling(window=window, center=True).mean()
        axes[1].plot(trend, color='red', linewidth=2)
        axes[1].set_title(f'Trend Component ({window}-day moving average)', fontsize=12)
        axes[1].set_ylabel('Trend')
    else:
        axes[1].text(0.5, 0.5, 'Insufficient data for trend calculation',
                     ha='center', va='center', transform=axes[1].transAxes)

    # Seasonal component (weekly pattern if enough data)
    if len(time_series_data) >= 14:
        seasonal = np.tile(np.arange(7), len(time_series_data) // 7 + 1)[:len(time_series_data)]
        weekly_means = []
        for day in range(7):
            day_values = time_series_data[day::7]
            weekly_means.append(day_values.mean())

        seasonal_component = [weekly_means[day % 7] for day in range(len(time_series_data))]
        axes[2].plot(seasonal_component, color='green', linewidth=1)
        axes[2].set_title('Seasonal Component (Weekly Pattern)', fontsize=12)
        axes[2].set_ylabel('Seasonal')
    else:
        axes[2].text(0.5, 0.5, 'Insufficient data for seasonal pattern',
                     ha='center', va='center', transform=axes[2].transAxes)

    # Residual (simplified)
    if window > 1 and len(time_series_data) >= 14:
        residual = time_series_data - trend.fillna(time_series_data.mean())
        axes[3].plot(residual, color='orange', linewidth=1)
        axes[3].set_title('Residual Component', fontsize=12)
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Time Period')
    else:
        axes[3].text(0.5, 0.5, 'Insufficient data for residual calculation',
                     ha='center', va='center', transform=axes[3].transAxes)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate decomposition statistics
    trend_strength = 0.0
    seasonal_strength = 0.0

    if window > 1:
        trend_strength = float(np.var(trend.dropna()) / np.var(time_series_data))

    if len(time_series_data) >= 14:
        seasonal_strength = float(np.var(seasonal_component) / np.var(time_series_data))

    return {
        'plot_path': save_path,
        'trend_strength': trend_strength,
        'seasonal_strength': seasonal_strength,
        'series_length': len(time_series_data),
        'decomposition_method': 'Simple moving average and weekly seasonality'
    }


def plot_autocorrelation_analysis(
    autocorr_results: Dict[str, Any],
    save_path: str
) -> Dict[str, Any]:
    """
    Plot autocorrelation analysis results.

    Creates bar plot of autocorrelations at key lags (1, 7, 14, 28, 30 days)
    with color-coding for correlation strength and reference lines for
    statistical significance thresholds.

    Parameters
    ----------
    autocorr_results : Dict[str, Any]
        Autocorrelation analysis results from compute_autocorrelation_analysis
    save_path : str
        Path to save the plot

    Returns
    -------
    Dict[str, Any]
        Plot metadata
    """
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Extract lag data
    autocorrs = autocorr_results['autocorrelations']
    lags = [int(lag.split('_')[1]) for lag in autocorrs.keys()]
    correlations = [autocorrs[f'lag_{lag}']['correlation'] for lag in lags]

    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bars = ax.bar(lags, correlations, alpha=0.7)

    # Color bars based on strength
    for bar, corr in zip(bars, correlations):
        if abs(corr) > 0.3:
            bar.set_color('red')
        elif abs(corr) > 0.1:
            bar.set_color('orange')
        else:
            bar.set_color('lightblue')

    ax.set_title('Autocorrelation Analysis - Key Lags', fontsize=14, fontweight='bold')
    ax.set_xlabel('Lag (days)', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Strong correlation threshold')
    ax.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, corr in zip(bars, correlations):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.01 if corr >= 0 else -0.03),
                f'{corr:.3f}', ha='center', va='bottom' if corr >= 0 else 'top', fontsize=10)

    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'plot_path': save_path,
        'lags_analyzed': lags,
        'strong_correlations': len([c for c in correlations if abs(c) > 0.3])
    }


def plot_missing_patterns(
    missing_analysis: Dict[str, Any],
    save_path: str,
    title: str = "Missing Data Patterns Analysis"
) -> Dict[str, Any]:
    """
    Visualize missing data patterns with heatmaps and statistics.

    Creates publication-ready visualization showing:
    - Zero-inflation distribution by category
    - Missing data heatmaps
    - Business logic correlations with missingness

    Parameters
    ----------
    missing_analysis : Dict[str, Any]
        Results from analyze_missing_patterns function
    save_path : str
        Path to save the plot
    title : str, default="Missing Data Patterns Analysis"
        Title for the plot

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - plot_path: Path where plot was saved
        - summary_statistics: Key statistics from analysis

    Raises
    ------
    ValueError
        If input dictionary is empty or missing required keys
    """
    if not missing_analysis or 'sales_completeness' not in missing_analysis:
        raise ValueError("missing_analysis must contain 'sales_completeness' key")

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    sales_complete = missing_analysis.get('sales_completeness', {})

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 1. Zero-heavy series distribution
    zero_stats = sales_complete.get('series_zero_stats', {})
    if zero_stats:
        ax = axes[0, 0]
        stats = [zero_stats.get('min', 0), zero_stats.get('mean', 0),
                 zero_stats.get('median', 0), zero_stats.get('max', 0)]
        labels = ['Min', 'Mean', 'Median', 'Max']
        colors = ['lightblue', 'skyblue', 'steelblue', 'darkblue']

        bars = ax.bar(labels, stats, color=colors, alpha=0.7)
        ax.set_title('Zero-Sales Percentage Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_ylim(0, 100)

        for bar, stat in zip(bars, stats):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{stat:.1f}%', ha='center', va='bottom', fontsize=10)

    # 2. Series completeness summary
    ax = axes[0, 1]
    completeness_data = [
        sales_complete.get('total_series', 0),
        sales_complete.get('zero_heavy_series', {}).get('count', 0)
    ]
    labels = ['Total Series', 'Zero-Heavy (≥80%)']
    colors = ['green', 'red']

    bars = ax.bar(labels, completeness_data, color=colors, alpha=0.7)
    ax.set_title('Data Completeness Overview', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Series', fontsize=11)

    for bar, val in zip(bars, completeness_data):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{int(val)}', ha='center', va='bottom', fontsize=10)

    # 3. Missing percentage by category (if available)
    ax = axes[1, 0]
    missing_corr = missing_analysis.get('missing_correlations', {})

    if missing_corr and 'by_store' in missing_corr:
        stores = list(missing_corr['by_store'].keys())[:10]  # Top 10 stores
        missing_pcts = [missing_corr['by_store'].get(s, 0) for s in stores]

        bars = ax.barh(stores, missing_pcts, color='coral', alpha=0.7)
        ax.set_title('Missing Data % by Store (Top 10)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Missing %', fontsize=11)

        for bar, pct in zip(bars, missing_pcts):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                   f'{pct:.2f}%', ha='left', va='center', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No store-level missing data available',
               ha='center', va='center', transform=ax.transAxes)

    # 4. Pricing gaps analysis
    ax = axes[1, 1]
    pricing_gaps = missing_analysis.get('pricing_gaps', {})

    if pricing_gaps:
        gap_data = [
            pricing_gaps.get('total_pricing_combinations', 0),
            pricing_gaps.get('missing_in_pricing', 0)
        ]
        labels = ['Has Pricing', 'Missing Pricing']
        colors = ['green', 'red']

        wedges, texts, autotexts = ax.pie(gap_data, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        ax.set_title('Pricing Coverage', fontsize=12, fontweight='bold')

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
    else:
        ax.text(0.5, 0.5, 'No pricing gap data available',
               ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'plot_path': save_path,
        'total_series': sales_complete.get('total_series', 0),
        'zero_heavy_percent': round(
            (sales_complete.get('zero_heavy_series', {}).get('count', 0) /
             max(1, sales_complete.get('total_series', 1))) * 100, 2
        )
    }


def plot_outlier_detection(
    outlier_analysis: Dict[str, Any],
    save_path: str,
    title: str = "Sales Outlier Detection Results"
) -> Dict[str, Any]:
    """
    Visualize outlier detection results by category and distribution.

    Creates publication-ready visualization showing:
    - Outliers by product category
    - Category-specific thresholds
    - Business rule violations

    Parameters
    ----------
    outlier_analysis : Dict[str, Any]
        Results from detect_sales_outliers function
    save_path : str
        Path to save the plot
    title : str, default="Sales Outlier Detection Results"
        Title for the plot

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - plot_path: Path where plot was saved
        - total_outliers_found: Number of outliers detected
        - categories_analyzed: List of categories

    Raises
    ------
    ValueError
        If input dictionary is empty or missing required keys
    """
    if not outlier_analysis or 'outlier_distribution' not in outlier_analysis:
        raise ValueError("outlier_analysis must contain 'outlier_distribution' key")

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    outlier_dist = outlier_analysis.get('outlier_distribution', {})
    category_thresholds = outlier_analysis.get('category_thresholds', {})

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 1. Outliers by category
    ax = axes[0, 0]
    if outlier_dist:
        categories = list(outlier_dist.keys())
        outlier_counts = [outlier_dist[cat].get('count', 0) for cat in categories]
        colors = ['red', 'orange', 'blue'][:len(categories)]

        bars = ax.bar(categories, outlier_counts, color=colors, alpha=0.7)
        ax.set_title('Outlier Count by Product Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Outliers', fontsize=11)

        for bar, count in zip(bars, outlier_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{int(count)}', ha='center', va='bottom', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No outlier distribution data available',
               ha='center', va='center', transform=ax.transAxes)

    # 2. Outlier percentage by category
    ax = axes[0, 1]
    if outlier_dist:
        categories = list(outlier_dist.keys())
        outlier_pcts = [outlier_dist[cat].get('percent', 0) for cat in categories]

        bars = ax.barh(categories, outlier_pcts, color=['red', 'orange', 'blue'][:len(categories)], alpha=0.7)
        ax.set_title('Outlier % of Total Sales by Category', fontsize=12, fontweight='bold')
        ax.set_xlabel('Percentage (%)', fontsize=11)

        for bar, pct in zip(bars, outlier_pcts):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{pct:.2f}%', ha='left', va='center', fontsize=10)

    # 3. Category thresholds and max values
    ax = axes[1, 0]
    if outlier_dist and category_thresholds:
        categories = list(outlier_dist.keys())
        thresholds = [category_thresholds.get(cat, 50) for cat in categories]
        max_values = [outlier_dist[cat].get('max_value', 0) for cat in categories]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax.bar(x - width/2, thresholds, width, label='Threshold', color='steelblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, max_values, width, label='Max Value', color='darkred', alpha=0.7)

        ax.set_title('Outlier Thresholds vs Actual Max Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Units', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

    # 4. Business rule violations
    ax = axes[1, 1]
    violations = outlier_analysis.get('business_rule_violations', {})

    if violations:
        violation_types = []
        violation_counts = []

        if violations.get('negative_sales', {}).get('count', 0) > 0:
            violation_types.append('Negative Sales')
            violation_counts.append(violations['negative_sales']['count'])

        if violations.get('unrealistic_values', {}).get('count', 0) > 0:
            violation_types.append('Unrealistic (>10K)')
            violation_counts.append(violations['unrealistic_values']['count'])

        if violation_types:
            bars = ax.bar(violation_types, violation_counts, color=['red', 'darkred'], alpha=0.7)
            ax.set_title('Business Rule Violations', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11)

            for bar, count in zip(bars, violation_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{int(count)}', ha='center', va='bottom', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No business rule violations detected',
                   ha='center', va='center', transform=ax.transAxes, color='green', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No violation data available',
               ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    total_outliers = outlier_analysis.get('total_outliers', 0)
    categories = list(outlier_dist.keys()) if outlier_dist else []

    return {
        'plot_path': save_path,
        'total_outliers_found': total_outliers,
        'categories_analyzed': categories
    }


def plot_segment_behavior_comparison(
    segment_data: Dict[str, Dict[str, float]],
    save_path: str,
    figsize: tuple = (15, 10)
) -> Dict[str, Any]:
    """
    Create business-focused comparison charts for segment behavioral patterns.

    Generates publication-quality multi-panel visualization showing:
    - Sales distribution by segment (box plots)
    - Seasonality strength comparison
    - Intermittency rate visualization
    - Coefficient of variation (demand volatility) comparison

    Useful for inventory planning decisions across product segments.

    Parameters
    ----------
    segment_data : Dict[str, Dict[str, float]]
        Dictionary with segment names as keys and behavioral metrics as values.
        Expected structure:
        {
            'SEGMENT_NAME': {
                'mean_sales': float,
                'cv': float (coefficient of variation),
                'intermittency': float (0-1, proportion of zero sales),
                'seasonality_strength': float (optional),
                'count': int (optional, number of items)
            },
            ...
        }
    save_path : str
        Path to save the plot (e.g., 'path/to/plot.png')
    figsize : tuple, default=(15, 10)
        Figure size as (width, height)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - plot_path: Path where plot was saved
        - segments_analyzed: List of segment names
        - summary_statistics: Key metrics per segment
        - business_insights: Actionable insights for inventory planning

    Raises
    ------
    ValueError
        If segment_data is empty
    KeyError
        If required metrics are missing from segment data
    IOError
        If plot cannot be saved

    Example
    --------
    >>> segment_data = {
    ...     'FOODS': {'mean_sales': 5.2, 'cv': 0.3, 'intermittency': 0.1},
    ...     'HOUSEHOLD': {'mean_sales': 2.8, 'cv': 0.8, 'intermittency': 0.4}
    ... }
    >>> result = plot_segment_behavior_comparison(segment_data, 'plots/segments.png')
    >>> print(result['business_insights'])
    """
    # Input validation
    if not segment_data:
        raise ValueError("segment_data cannot be empty")

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Extract segment names and metrics
    segments = sorted(segment_data.keys())
    mean_sales = [segment_data[s].get('mean_sales', 0) for s in segments]
    cv_values = [segment_data[s].get('cv', 0) for s in segments]
    intermittency = [segment_data[s].get('intermittency', 0) for s in segments]
    seasonality = [segment_data[s].get('seasonality_strength', 0.5) for s in segments]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Segment Behavioral Pattern Analysis for Inventory Planning',
                 fontsize=14, fontweight='bold')

    try:
        # 1. Mean Sales by Segment
        ax = axes[0, 0]
        colors_sales = ['#2ecc71' if val > np.mean(mean_sales) else '#e74c3c'
                       for val in mean_sales]
        bars = ax.bar(segments, mean_sales, color=colors_sales, alpha=0.7, edgecolor='black')
        ax.set_title('Average Daily Sales by Segment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Sales (units)', fontsize=11)
        ax.set_xlabel('Product Segment', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, mean_sales):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        # 2. Demand Volatility (Coefficient of Variation)
        ax = axes[0, 1]
        colors_cv = ['#e74c3c' if val > 0.8 else '#f39c12' if val > 0.5 else '#2ecc71'
                    for val in cv_values]
        bars = ax.bar(segments, cv_values, color=colors_cv, alpha=0.7, edgecolor='black')
        ax.set_title('Demand Volatility (Coefficient of Variation)', fontsize=12, fontweight='bold')
        ax.set_ylabel('CV (higher = more volatile)', fontsize=11)
        ax.set_xlabel('Product Segment', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium volatility')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='High volatility')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, cv_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        # 3. Intermittency Rate (Zero Sales Proportion)
        ax = axes[1, 0]
        colors_inter = ['#e74c3c' if val > 0.5 else '#f39c12' if val > 0.25 else '#2ecc71'
                       for val in intermittency]
        bars = ax.bar(segments, intermittency, color=colors_inter, alpha=0.7, edgecolor='black')
        ax.set_title('Intermittency Rate (Zero Sales %)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Intermittency (0-1)', fontsize=11)
        ax.set_xlabel('Product Segment', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Moderate')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='High')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, intermittency):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2%}', ha='center', va='bottom', fontsize=10)

        # 4. Seasonality Strength
        ax = axes[1, 1]
        colors_season = ['#2ecc71' if val > 0.6 else '#f39c12' if val > 0.3 else '#e74c3c'
                        for val in seasonality]
        bars = ax.bar(segments, seasonality, color=colors_season, alpha=0.7, edgecolor='black')
        ax.set_title('Seasonality Strength (Planning Signal)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Seasonality Strength (0-1)', fontsize=11)
        ax.set_xlabel('Product Segment', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Weak')
        ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Strong')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, seasonality):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        plt.close('all')
        raise IOError(f"Failed to save segment comparison plot to {save_path}: {str(e)}")

    # Generate business insights
    high_volatility_segments = [seg for seg, cv in zip(segments, cv_values) if cv > 0.8]
    high_intermittency_segments = [seg for seg, inter in zip(segments, intermittency) if inter > 0.5]
    strong_seasonality_segments = [seg for seg, season in zip(segments, seasonality) if season > 0.6]

    business_insights = []
    if high_volatility_segments:
        business_insights.append(
            f"HIGH VOLATILITY: {', '.join(high_volatility_segments)} segments show volatile demand. "
            "Recommend safety stock increase and more frequent replenishment."
        )
    if high_intermittency_segments:
        business_insights.append(
            f"SPARSE DEMAND: {', '.join(high_intermittency_segments)} segments have frequent stockouts. "
            "Consider demand-driven replenishment and lower minimum order quantities."
        )
    if strong_seasonality_segments:
        business_insights.append(
            f"SEASONAL PATTERNS: {', '.join(strong_seasonality_segments)} segments show clear seasonality. "
            "Align inventory with seasonal forecasts and adjust storage capacity accordingly."
        )

    return {
        'plot_path': save_path,
        'segments_analyzed': segments,
        'summary_statistics': {seg: segment_data[seg] for seg in segments},
        'business_insights': business_insights if business_insights else ["Segments show balanced demand patterns."],
        'high_volatility_segments': high_volatility_segments,
        'high_intermittency_segments': high_intermittency_segments,
        'strong_seasonality_segments': strong_seasonality_segments
    }


def plot_distribution_drift_analysis(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    drift_results: Dict[str, Any],
    save_path: str,
    figsize: tuple = (15, 10)
) -> Dict[str, Any]:
    """
    Visualize distribution drift analysis comparing train and validation periods.

    Creates publication-quality multi-panel visualization showing:
    - Before/after distribution comparisons (histograms)
    - Statistical test results (KS test, effect sizes)
    - Seasonal alignment validation
    - Drift magnitude with confidence indicators

    Parameters
    ----------
    train_data : pd.DataFrame
        Training period data with numeric features
    validation_data : pd.DataFrame
        Validation period data with same features as train_data
    drift_results : Dict[str, Any]
        Drift analysis results containing:
        - ks_tests: KS test statistics and p-values
        - effect_sizes: Cohen's d and magnitude classifications
    save_path : str
        Path to save the plot
    figsize : tuple, default=(15, 10)
        Figure size as (width, height)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - plot_path: Path where plot was saved
        - features_analyzed: List of features analyzed
        - drift_detected: Boolean indicating if drift was found
        - high_drift_features: Features with significant drift

    Raises
    ------
    ValueError
        If dataframes are empty or inconsistent
    IOError
        If plot cannot be saved

    Example
    --------
    >>> train_data = pd.DataFrame({'sales': np.random.poisson(5, 1000)})
    >>> validation_data = pd.DataFrame({'sales': np.random.poisson(7, 200)})
    >>> drift_results = {
    ...     'ks_tests': {'sales': {'statistic': 0.15, 'p_value': 0.02}},
    ...     'effect_sizes': {'sales': {'cohens_d': 0.6, 'magnitude': 'medium'}}
    ... }
    >>> result = plot_distribution_drift_analysis(train_data, validation_data, drift_results, 'plots/drift.png')
    """
    # Input validation
    if train_data.empty:
        raise ValueError("train_data cannot be empty")
    if validation_data.empty:
        raise ValueError("validation_data cannot be empty")
    if not drift_results:
        raise ValueError("drift_results cannot be empty")

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Get numeric columns common to both datasets
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col in validation_data.columns]

    if not numeric_cols:
        raise ValueError("No common numeric columns found in both datasets")

    # Limit to first 4 features for visualization clarity
    features_to_plot = numeric_cols[:4]

    # Create figure with subplots (2 rows x 2 cols for up to 4 features)
    n_features = len(features_to_plot)
    n_cols = min(2, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    fig.suptitle('Distribution Drift Analysis: Training vs Validation Period',
                 fontsize=14, fontweight='bold')

    try:
        high_drift_features = []

        for idx, feature in enumerate(features_to_plot):
            ax = axes[idx]

            # Get feature data
            train_vals = train_data[feature].dropna()
            val_vals = validation_data[feature].dropna()

            if len(train_vals) == 0 or len(val_vals) == 0:
                ax.text(0.5, 0.5, f'{feature}: Insufficient data',
                       ha='center', va='center', transform=ax.transAxes)
                continue

            # Plot overlapping histograms
            ax.hist(train_vals, bins=30, alpha=0.6, label='Training', color='blue', edgecolor='black')
            ax.hist(val_vals, bins=30, alpha=0.6, label='Validation', color='red', edgecolor='black')

            # Get drift results for this feature
            ks_test = drift_results.get('ks_tests', {}).get(feature, {})
            effect_size = drift_results.get('effect_sizes', {}).get(feature, {})

            ks_stat = ks_test.get('statistic', 0)
            p_value = ks_test.get('p_value', 1.0)
            cohens_d = effect_size.get('cohens_d', 0)
            magnitude = effect_size.get('magnitude', 'unknown')

            # Determine drift severity
            if p_value < 0.05 and cohens_d > 0.5:
                high_drift_features.append(feature)
                drift_indicator = 'SIGNIFICANT DRIFT DETECTED'
                color = 'red'
            elif p_value < 0.05:
                drift_indicator = 'Moderate drift detected'
                color = 'orange'
            else:
                drift_indicator = 'No significant drift'
                color = 'green'

            # Add title with statistical results
            title = (f'{feature}\n'
                    f'KS stat={ks_stat:.3f}, p={p_value:.4f}, Cohen\'s d={cohens_d:.2f} ({magnitude})\n'
                    f'{drift_indicator}')
            ax.set_title(title, fontsize=11, fontweight='bold', color=color)

            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        # Hide unused subplots
        for idx in range(len(features_to_plot), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        plt.close('all')
        raise IOError(f"Failed to save drift analysis plot to {save_path}: {str(e)}")

    return {
        'plot_path': save_path,
        'features_analyzed': features_to_plot,
        'drift_detected': len(high_drift_features) > 0,
        'high_drift_features': high_drift_features,
        'total_features': len(features_to_plot),
        'train_samples': len(train_data),
        'validation_samples': len(validation_data)
    }


def plot_leakage_validation_summary(
    audit_results: Dict[str, Any],
    save_path: str,
    figsize: tuple = (15, 10)
) -> Dict[str, Any]:
    """
    Visualize temporal feature leakage validation results.

    Creates publication-quality dashboard showing:
    - Temporal boundary violations (bar chart of violations by type)
    - Feature availability timeline
    - Leakage risk assessment summary
    - Compliant features inventory

    Parameters
    ----------
    audit_results : Dict[str, Any]
        Leakage validation audit results containing:
        - boundary_violations: List of dicts with feature, violation, risk
        - compliant_features: List of dicts with feature, type, window
        - risk_assessment: Dict with risk_level and violation_rate
    save_path : str
        Path to save the plot
    figsize : tuple, default=(15, 10)
        Figure size as (width, height)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - plot_path: Path where plot was saved
        - risk_level: Overall risk assessment level
        - violations_found: Number of violations
        - compliant_features_count: Number of compliant features
        - recommendations: List of remediation actions

    Raises
    ------
    ValueError
        If audit_results is empty
    IOError
        If plot cannot be saved

    Example
    --------
    >>> audit_results = {
    ...     'boundary_violations': [
    ...         {'feature': 'bad_feature', 'violation': 'future_data', 'risk': 'high'}
    ...     ],
    ...     'compliant_features': [
    ...         {'feature': 'good_feature', 'type': 'lag', 'window': 7}
    ...     ],
    ...     'risk_assessment': {'risk_level': 'medium', 'violation_rate': 0.2}
    ... }
    >>> result = plot_leakage_validation_summary(audit_results, 'plots/leakage.png')
    """
    # Input validation
    if not audit_results:
        raise ValueError("audit_results cannot be empty")

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Extract data
    violations = audit_results.get('boundary_violations', [])
    compliant = audit_results.get('compliant_features', [])
    risk_info = audit_results.get('risk_assessment', {})

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Temporal Leakage Validation Audit Summary',
                 fontsize=14, fontweight='bold')

    try:
        # 1. Violation Types Distribution
        ax = axes[0, 0]
        if violations:
            violation_types = {}
            risk_colors = {}

            for v in violations:
                vtype = v.get('violation', 'unknown')
                risk = v.get('risk', 'low')
                violation_types[vtype] = violation_types.get(vtype, 0) + 1

                if risk == 'high':
                    risk_colors[vtype] = '#e74c3c'
                elif risk == 'medium':
                    risk_colors[vtype] = '#f39c12'
                else:
                    risk_colors[vtype] = '#f1c40f'

            violation_names = list(violation_types.keys())
            violation_counts = list(violation_types.values())
            colors = [risk_colors.get(v, '#95a5a6') for v in violation_names]

            bars = ax.bar(violation_names, violation_counts, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title('Temporal Boundary Violations by Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)

            for bar, count in zip(bars, violation_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{int(count)}', ha='center', va='bottom', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No violations detected\n(Clean feature set)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=11,
                   color='green', fontweight='bold')
            ax.set_title('Temporal Boundary Violations', fontsize=12, fontweight='bold')

        # 2. Compliant Features by Type
        ax = axes[0, 1]
        if compliant:
            feature_types = {}

            for c in compliant:
                ftype = c.get('type', 'unknown')
                feature_types[ftype] = feature_types.get(ftype, 0) + 1

            feature_type_names = list(feature_types.keys())
            feature_type_counts = list(feature_types.values())

            colors_compliant = ['#2ecc71', '#3498db', '#9b59b6', '#1abc9c'][:len(feature_type_names)]
            bars = ax.bar(feature_type_names, feature_type_counts, color=colors_compliant,
                         alpha=0.7, edgecolor='black')
            ax.set_title('Compliant Features by Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)

            for bar, count in zip(bars, feature_type_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{int(count)}', ha='center', va='bottom', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No compliant features',
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title('Compliant Features by Type', fontsize=12, fontweight='bold')

        # 3. Risk Assessment Dashboard
        ax = axes[1, 0]
        risk_level = risk_info.get('risk_level', 'unknown').upper()
        violation_rate = risk_info.get('violation_rate', 0)
        total_features = len(violations) + len(compliant)

        # Color code based on risk
        if risk_level == 'CRITICAL' or risk_level == 'HIGH':
            risk_color = '#e74c3c'
        elif risk_level == 'MEDIUM':
            risk_color = '#f39c12'
        elif risk_level == 'LOW':
            risk_color = '#2ecc71'
        else:
            risk_color = '#95a5a6'

        # Create risk gauge (simplified)
        ax.text(0.5, 0.75, f'Overall Risk Level', ha='center', va='center',
               transform=ax.transAxes, fontsize=11, fontweight='bold')
        ax.text(0.5, 0.55, risk_level, ha='center', va='center',
               transform=ax.transAxes, fontsize=24, fontweight='bold', color=risk_color)

        ax.text(0.5, 0.35, f'Violation Rate: {violation_rate:.1%}', ha='center', va='center',
               transform=ax.transAxes, fontsize=11)
        ax.text(0.5, 0.15, f'Total Features: {total_features}', ha='center', va='center',
               transform=ax.transAxes, fontsize=11)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # 4. Feature Summary Table (as text)
        ax = axes[1, 1]
        summary_text = 'FEATURE LEAKAGE AUDIT SUMMARY\n' + '=' * 40 + '\n\n'
        summary_text += f'Total Violations: {len(violations)}\n'
        summary_text += f'Compliant Features: {len(compliant)}\n'
        summary_text += f'Risk Level: {risk_level}\n'
        summary_text += f'Violation Rate: {violation_rate:.1%}\n\n'

        if violations:
            summary_text += 'VIOLATIONS FOUND:\n' + '-' * 40 + '\n'
            for i, v in enumerate(violations[:5], 1):  # Show first 5
                vtype = v.get('violation', 'unknown')
                risk = v.get('risk', 'unknown')
                feature = v.get('feature', 'unknown')
                summary_text += f'{i}. {feature} ({vtype}, risk={risk})\n'
            if len(violations) > 5:
                summary_text += f'... and {len(violations) - 5} more\n'
        else:
            summary_text += 'NO VIOLATIONS DETECTED\n' + '-' * 40 + '\n'

        ax.text(0.05, 0.95, summary_text, ha='left', va='top', transform=ax.transAxes,
               fontsize=9, family='monospace', verticalalignment='top')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        plt.close('all')
        raise IOError(f"Failed to save leakage validation plot to {save_path}: {str(e)}")

    # Generate recommendations
    recommendations = []

    if len(violations) > 0:
        recommendations.append(f"CRITICAL: Address {len(violations)} feature(s) with temporal boundary violations")

        high_risk_violations = [v for v in violations if v.get('risk') == 'high']
        if high_risk_violations:
            features = ', '.join([v.get('feature', 'unknown') for v in high_risk_violations])
            recommendations.append(f"Remove or transform these high-risk features: {features}")

    if violation_rate > 0.5:
        recommendations.append("Over 50% of features show leakage risk. Consider feature engineering review.")

    if len(compliant) > 0 and len(violations) == 0:
        recommendations.append("All features pass temporal leakage validation. Feature set is ready for modeling.")

    return {
        'plot_path': save_path,
        'risk_level': risk_level,
        'violations_found': len(violations),
        'compliant_features_count': len(compliant),
        'violation_rate': violation_rate,
        'total_features_audited': len(violations) + len(compliant),
        'recommendations': recommendations if recommendations else ["No action required"]
    }
