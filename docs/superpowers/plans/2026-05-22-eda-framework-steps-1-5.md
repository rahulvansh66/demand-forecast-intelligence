# EDA Framework Steps 1-5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement comprehensive EDA analysis for M5 dataset covering framework steps 1-5 with statistical testing, visualization, and detailed reporting.

**Architecture:** Hybrid approach with reusable utility modules (statistical_analysis, visualization, data_validation, time_series_analysis, reporting) orchestrated by main functions in eda_analysis.py that map to framework steps.

**Tech Stack:** Python, pandas, numpy, scipy, matplotlib, seaborn, plotly, jupyter

---

### Task 1: Project Setup and Dependencies

**Files:**
- Modify: `notebooks/eda/utils/__init__.py`
- Create: `notebooks/eda/tests/__init__.py`
- Create: `notebooks/eda/requirements.txt`

- [ ] **Step 1: Update utils package init**

```python
# EDA Utilities Package
# Modular utilities for comprehensive demand forecasting EDA

from .statistical_analysis import *
from .data_validation import *  
from .visualization import *
from .time_series_analysis import *
from .reporting import *

__version__ = "1.0.0"
```

- [ ] **Step 2: Create test directory**

```python
# Test package for EDA utilities
```

- [ ] **Step 3: Create requirements file**

```text
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
jupyter>=1.0.0
pytest>=7.0.0
```

- [ ] **Step 4: Commit setup**

```bash
git add notebooks/eda/utils/__init__.py notebooks/eda/tests/__init__.py notebooks/eda/requirements.txt
git commit -m "feat: setup EDA framework project structure and dependencies"
```

### Task 2: Statistical Analysis Module

**Files:**
- Create: `notebooks/eda/utils/statistical_analysis.py`
- Create: `notebooks/eda/tests/test_statistical_analysis.py`

- [ ] **Step 1: Write failing test for distribution stats**

```python
import pytest
import pandas as pd
import numpy as np
from notebooks.eda.utils.statistical_analysis import calculate_distribution_stats

def test_calculate_distribution_stats():
    # Test data with known statistical properties
    data = pd.Series([1, 2, 3, 4, 5, 100])  # Right skewed with outlier
    result = calculate_distribution_stats(data, "test_series")
    
    assert "mean" in result
    assert "median" in result  
    assert "skewness" in result
    assert result["skewness"] > 0  # Right skewed
    assert "interpretation" in result
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd notebooks/eda && python -m pytest tests/test_statistical_analysis.py::test_calculate_distribution_stats -v`
Expected: ImportError or module not found

- [ ] **Step 3: Create statistical analysis module foundation**

```python
"""
Statistical Analysis Utilities for M5 EDA Framework
Provides hypothesis testing and descriptive statistics with business context
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple
import warnings

def calculate_distribution_stats(data: pd.Series, series_name: str) -> Dict[str, Any]:
    """
    Calculate comprehensive distribution statistics with business interpretation.
    
    Args:
        data: Pandas series to analyze
        series_name: Name for reporting context
        
    Returns:
        Dictionary with statistical measures and business interpretation
    """
    stats_dict = {
        "series_name": series_name,
        "count": len(data),
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),
        "min": data.min(),
        "max": data.max(),
        "q25": data.quantile(0.25),
        "q75": data.quantile(0.75),
        "skewness": data.skew(),
        "kurtosis": data.kurtosis(),
        "cv": data.std() / data.mean() if data.mean() != 0 else np.inf
    }
    
    # Business interpretation
    if abs(stats_dict["skewness"]) < 0.5:
        skew_interp = "approximately symmetric"
    elif stats_dict["skewness"] > 0.5:
        skew_interp = "right-skewed (tail extends toward higher values)"
    else:
        skew_interp = "left-skewed (tail extends toward lower values)"
    
    stats_dict["interpretation"] = f"Distribution is {skew_interp}"
    
    return stats_dict
```

- [ ] **Step 4: Run test to verify pass**

Run: `cd notebooks/eda && python -m pytest tests/test_statistical_analysis.py::test_calculate_distribution_stats -v`
Expected: PASS

- [ ] **Step 5: Add variation metrics function**

```python
def compute_variation_metrics(data: pd.Series) -> Dict[str, float]:
    """
    Compute variation metrics for demand volatility assessment.
    
    Args:
        data: Sales or demand data series
        
    Returns:
        Dictionary with variation metrics
    """
    return {
        "coefficient_variation": data.std() / data.mean() if data.mean() != 0 else np.inf,
        "mad": stats.median_abs_deviation(data),
        "iqr": data.quantile(0.75) - data.quantile(0.25),
        "range": data.max() - data.min(),
        "variance": data.var()
    }

def analyze_outliers(data: pd.Series, method: str = "iqr") -> Dict[str, Any]:
    """
    Detect outliers using specified method with retail context.
    
    Args:
        data: Series to analyze for outliers
        method: Method to use ("iqr", "zscore")
        
    Returns:
        Dictionary with outlier analysis results
    """
    if method == "iqr":
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
    elif method == "zscore":
        z_scores = np.abs(stats.zscore(data))
        outliers = data[z_scores > 3]
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return {
        "method": method,
        "outlier_count": len(outliers),
        "outlier_percentage": len(outliers) / len(data) * 100,
        "outlier_indices": outliers.index.tolist(),
        "outlier_values": outliers.tolist()
    }
```

- [ ] **Step 6: Add normality and independence tests**

```python
def test_normality(data: pd.Series) -> Dict[str, Any]:
    """
    Test for normality using Shapiro-Wilk test.
    
    Args:
        data: Series to test for normality
        
    Returns:
        Dictionary with test results and interpretation
    """
    # Use sample for large datasets (Shapiro-Wilk limitation)
    test_data = data.sample(min(5000, len(data))) if len(data) > 5000 else data
    
    statistic, p_value = stats.shapiro(test_data)
    
    return {
        "test": "Shapiro-Wilk",
        "statistic": statistic,
        "p_value": p_value,
        "is_normal": p_value > 0.05,
        "sample_size": len(test_data),
        "interpretation": "Normal distribution" if p_value > 0.05 else "Non-normal distribution"
    }

def test_independence(data1: pd.Series, data2: pd.Series) -> Dict[str, Any]:
    """
    Test independence between two categorical series using Chi-square test.
    
    Args:
        data1: First categorical series
        data2: Second categorical series
        
    Returns:
        Dictionary with test results
    """
    contingency_table = pd.crosstab(data1, data2)
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Calculate Cramér's V
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * min(contingency_table.shape) - 1))
    
    return {
        "test": "Chi-square independence",
        "chi2_statistic": chi2,
        "p_value": p_value,
        "degrees_freedom": dof,
        "cramers_v": cramers_v,
        "are_independent": p_value > 0.05,
        "interpretation": "Independent" if p_value > 0.05 else "Dependent"
    }
```

- [ ] **Step 7: Add business-specific metrics**

```python
def calculate_intermittency_score(sales_data: pd.Series) -> Dict[str, float]:
    """
    Calculate intermittency metrics for demand patterns.
    
    Args:
        sales_data: Daily sales data series
        
    Returns:
        Dictionary with intermittency metrics
    """
    zero_days = (sales_data == 0).sum()
    total_days = len(sales_data)
    non_zero_days = total_days - zero_days
    
    return {
        "zero_percentage": zero_days / total_days * 100,
        "non_zero_percentage": non_zero_days / total_days * 100,
        "average_demand_intensity": sales_data[sales_data > 0].mean() if non_zero_days > 0 else 0,
        "demand_frequency": non_zero_days / total_days,
        "intermittency_classification": "High" if zero_days / total_days > 0.7 else "Medium" if zero_days / total_days > 0.3 else "Low"
    }

def demand_variability_classification(sales_data: pd.Series) -> str:
    """
    Classify demand variability pattern for segmentation.
    
    Args:
        sales_data: Sales time series
        
    Returns:
        Classification string
    """
    cv = sales_data.std() / sales_data.mean() if sales_data.mean() > 0 else np.inf
    zero_pct = (sales_data == 0).sum() / len(sales_data)
    mean_sales = sales_data.mean()
    
    if zero_pct > 0.7:
        return "Low Demand Intermittent"
    elif mean_sales > 5 and cv < 1.0:
        return "High Demand Stable" 
    elif mean_sales > 5 and cv >= 1.0:
        return "High Demand Volatile"
    elif sales_data.tail(90).mean() < sales_data.head(90).mean() * 0.7:
        return "Declining"
    else:
        return "Seasonal"
```

- [ ] **Step 8: Commit statistical analysis module**

```bash
git add notebooks/eda/utils/statistical_analysis.py notebooks/eda/tests/test_statistical_analysis.py
git commit -m "feat: add comprehensive statistical analysis module with business metrics"
```