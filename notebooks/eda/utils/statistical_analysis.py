"""
Statistical analysis utilities for demand forecasting EDA.

Provides functions for:
- Distribution statistics and characterization
- Variation metrics calculation
- Outlier detection and analysis
- Normality and independence testing
- Business-specific demand metrics (intermittency, variability classification)
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def calculate_distribution_stats(
    data: pd.Series, series_name: str = "series"
) -> Dict[str, Any]:
    """
    Calculate comprehensive distribution statistics.

    Parameters
    ----------
    data : pd.Series
        Input time series data
    series_name : str
        Name of the series for reference

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - count: Number of non-null observations
        - mean: Arithmetic mean
        - median: Median value
        - std_dev: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - q25: 25th percentile
        - q75: 75th percentile
        - skewness: Skewness coefficient (right/left tail indicator)
        - kurtosis: Kurtosis coefficient (tail heaviness)
        - interpretation: Text interpretation of distribution shape
    """
    # Remove NaN values
    clean_data = data.dropna()

    if len(clean_data) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std_dev": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "interpretation": "Empty series",
        }

    # Calculate statistics
    mean = float(clean_data.mean())
    median = float(clean_data.median())
    std_dev = float(clean_data.std())
    min_val = float(clean_data.min())
    max_val = float(clean_data.max())
    q25 = float(clean_data.quantile(0.25))
    q75 = float(clean_data.quantile(0.75))
    skewness = float(stats.skew(clean_data))
    kurtosis = float(stats.kurtosis(clean_data))

    # Generate interpretation
    if skewness > 0.5:
        skew_interpretation = "right-skewed"
    elif skewness < -0.5:
        skew_interpretation = "left-skewed"
    else:
        skew_interpretation = "symmetric"

    if kurtosis > 1:
        kurt_interpretation = "heavy-tailed"
    elif kurtosis < -1:
        kurt_interpretation = "light-tailed"
    else:
        kurt_interpretation = "normal-tailed"

    interpretation = f"{skew_interpretation}, {kurt_interpretation}"

    return {
        "count": len(clean_data),
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "min": min_val,
        "max": max_val,
        "q25": q25,
        "q75": q75,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "interpretation": interpretation,
    }


def compute_variation_metrics(
    data: pd.Series, series_name: str = "series"
) -> Dict[str, float]:
    """
    Compute variation metrics for demand data.

    Parameters
    ----------
    data : pd.Series
        Input time series data
    series_name : str
        Name of the series for reference

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - std_dev: Standard deviation
        - cv: Coefficient of variation (std_dev / mean), handles zero mean
        - range: Max - Min
        - iqr: Interquartile range (Q75 - Q25)
        - mad: Mean absolute deviation
        - variance: Variance
    """
    clean_data = data.dropna()

    if len(clean_data) == 0:
        return {
            "std_dev": np.nan,
            "cv": np.nan,
            "range": np.nan,
            "iqr": np.nan,
            "mad": np.nan,
            "variance": np.nan,
        }

    std_dev = float(clean_data.std())
    mean = float(clean_data.mean())

    # Coefficient of variation - handle zero mean
    if abs(mean) < 1e-10:
        cv = 0.0 if std_dev == 0 else np.inf
    else:
        cv = float(std_dev / abs(mean))

    range_val = float(clean_data.max() - clean_data.min())
    q25 = float(clean_data.quantile(0.25))
    q75 = float(clean_data.quantile(0.75))
    iqr = float(q75 - q25)
    mad = float(np.mean(np.abs(clean_data - mean)))
    variance = float(clean_data.var())

    return {
        "std_dev": std_dev,
        "cv": cv,
        "range": range_val,
        "iqr": iqr,
        "mad": mad,
        "variance": variance,
    }


def analyze_outliers(
    data: pd.Series, method: str = "iqr", threshold: float = 2.0
) -> Dict[str, Any]:
    """
    Detect and analyze outliers using specified method.

    Parameters
    ----------
    data : pd.Series
        Input time series data
    method : str
        Detection method: 'iqr' or 'zscore'
    threshold : float
        Threshold for z-score method (default 2.0 for 95% confidence)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - outliers: Series of detected outlier values
        - outlier_count: Number of outliers detected
        - outlier_indices: Indices of outliers
        - method: Method used for detection
        - percentage: Percentage of data points that are outliers
    """
    clean_data = data.dropna()

    if len(clean_data) == 0:
        return {
            "outliers": pd.Series(dtype=float),
            "outlier_count": 0,
            "outlier_indices": [],
            "method": method,
            "percentage": 0.0,
        }

    if method == "iqr":
        q25 = clean_data.quantile(0.25)
        q75 = clean_data.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outlier_mask = (clean_data < lower_bound) | (clean_data > upper_bound)

    elif method == "zscore":
        mean = clean_data.mean()
        std = clean_data.std()
        if std == 0:
            outlier_mask = pd.Series(False, index=clean_data.index)
        else:
            z_scores = np.abs((clean_data - mean) / std)
            outlier_mask = z_scores > threshold

    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")

    outliers = clean_data[outlier_mask]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(clean_data)) * 100

    return {
        "outliers": outliers,
        "outlier_count": outlier_count,
        "outlier_indices": outliers.index.tolist(),
        "method": method,
        "percentage": outlier_percentage,
    }


def test_normality(data: pd.Series, series_name: str = "series") -> Dict[str, Any]:
    """
    Test if data follows normal distribution.

    Parameters
    ----------
    data : pd.Series
        Input time series data
    series_name : str
        Name of the series for reference

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - shapiro_statistic: Shapiro-Wilk test statistic
        - shapiro_pvalue: Shapiro-Wilk test p-value
        - ks_statistic: Kolmogorov-Smirnov test statistic
        - ks_pvalue: Kolmogorov-Smirnov test p-value
        - is_normal: Boolean indicating normality (p-value > 0.05)
        - test_note: Note about which test was used
    """
    clean_data = data.dropna()

    if len(clean_data) < 3:
        return {
            "shapiro_statistic": np.nan,
            "shapiro_pvalue": np.nan,
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "is_normal": None,
            "test_note": "Insufficient data for normality test",
        }

    # Shapiro-Wilk test (works for n <= 5000)
    if len(clean_data) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(clean_data)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan

    # Kolmogorov-Smirnov test
    standardized = (clean_data - clean_data.mean()) / clean_data.std()
    ks_stat, ks_p = stats.kstest(standardized, "norm")

    # Determine normality (use p-value > 0.05)
    if not np.isnan(shapiro_p):
        is_normal = shapiro_p > 0.05
        test_used = "Shapiro-Wilk"
    else:
        is_normal = ks_p > 0.05
        test_used = "Kolmogorov-Smirnov"

    return {
        "shapiro_statistic": float(shapiro_stat) if not np.isnan(shapiro_stat) else np.nan,
        "shapiro_pvalue": float(shapiro_p) if not np.isnan(shapiro_p) else np.nan,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "is_normal": is_normal,
        "test_note": f"Normality test (primary: {test_used})",
    }


def test_independence(
    series1: pd.Series,
    series2: pd.Series,
    name1: str = "series1",
    name2: str = "series2",
) -> Dict[str, Any]:
    """
    Test independence between two time series.

    Parameters
    ----------
    series1 : pd.Series
        First time series
    series2 : pd.Series
        Second time series
    name1 : str
        Name of first series
    name2 : str
        Name of second series

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - correlation: Pearson correlation coefficient
        - pvalue: P-value of correlation test
        - is_independent: Boolean indicating statistical independence
        - interpretation: Text interpretation of relationship
    """
    # Align series
    common_index = series1.index.intersection(series2.index)
    s1_aligned = series1.loc[common_index].dropna()
    s2_aligned = series2.loc[common_index].dropna()

    # Ensure same length
    min_len = min(len(s1_aligned), len(s2_aligned))
    s1_aligned = s1_aligned.iloc[:min_len]
    s2_aligned = s2_aligned.iloc[:min_len]

    if len(s1_aligned) < 3:
        return {
            "correlation": np.nan,
            "pvalue": np.nan,
            "is_independent": None,
            "interpretation": "Insufficient data for independence test",
        }

    # Pearson correlation test
    correlation, pvalue = stats.pearsonr(s1_aligned, s2_aligned)

    is_independent = pvalue > 0.05

    # Interpretation
    if abs(correlation) < 0.3:
        corr_strength = "weak"
    elif abs(correlation) < 0.7:
        corr_strength = "moderate"
    else:
        corr_strength = "strong"

    if correlation > 0:
        direction = "positive"
    elif correlation < 0:
        direction = "negative"
    else:
        direction = "no"

    interpretation = (
        f"{direction} {corr_strength} correlation "
        f"(p={'significant' if not is_independent else 'not significant'})"
    )

    return {
        "correlation": float(correlation),
        "pvalue": float(pvalue),
        "is_independent": is_independent,
        "interpretation": interpretation,
    }


def calculate_intermittency_score(
    data: pd.Series, series_name: str = "series"
) -> float:
    """
    Calculate intermittency score based on proportion of zero/near-zero demands.

    Intermittency score measures how often demand is absent or near-zero.
    Used to classify demand patterns for inventory management strategies.

    Parameters
    ----------
    data : pd.Series
        Input time series data (demand units)
    series_name : str
        Name of the series for reference

    Returns
    -------
    float
        Intermittency score between 0 (no intermittency) and 1 (all zero)
    """
    clean_data = data.dropna()

    if len(clean_data) == 0:
        return np.nan

    # Count zero and near-zero demands (within 1% of mean or just zero if mean is small)
    mean_val = clean_data.mean()

    if mean_val < 1e-10:
        # If mean is essentially zero, just count actual zeros
        zero_count = (clean_data == 0).sum()
    else:
        # Count zero and very small values (< 1% of mean)
        threshold = mean_val * 0.01
        zero_count = (clean_data <= threshold).sum()

    intermittency = float(zero_count / len(clean_data))

    return intermittency


def demand_variability_classification(
    data: pd.Series, series_name: str = "series"
) -> str:
    """
    Classify demand variability pattern for inventory strategy.

    Classification based on:
    - Intermittency: proportion of zero/near-zero demands
    - Coefficient of variation: relative variability

    Classes:
    - Smooth: Low intermittency, low variability
    - Lumpy: Occasional large demands with many zeros
    - Erratic: High variability even with non-zero demands
    - Intermittent: High intermittency, low variability

    Parameters
    ----------
    data : pd.Series
        Input time series data (demand units)
    series_name : str
        Name of the series for reference

    Returns
    -------
    str
        Classification: 'Smooth', 'Lumpy', 'Erratic', or 'Intermittent'
    """
    clean_data = data.dropna()

    if len(clean_data) == 0:
        return "Unknown"

    # Calculate metrics
    intermittency = calculate_intermittency_score(clean_data, series_name)
    variation_metrics = compute_variation_metrics(clean_data, series_name)
    cv = variation_metrics["cv"]

    # Handle infinite CV
    if np.isinf(cv):
        cv = 999  # Large value for classification purposes

    # Classification logic (based on Croston's method and demand classification)
    if intermittency < 0.25:
        # Low intermittency - smooth or erratic
        if cv < 0.5:
            return "Smooth"
        else:
            return "Erratic"
    else:
        # High intermittency - lumpy or intermittent
        if cv < 0.5:
            return "Intermittent"
        else:
            return "Lumpy"
