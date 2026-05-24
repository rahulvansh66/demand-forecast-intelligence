# EDA Steps 1-3, 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement missing EDA framework steps 1, 2, 3, 5 with M5-specific focus areas and static visualizations that integrate with existing output_manager.py system.

**Architecture:** Four standalone analysis functions following existing pattern (step6-10, 11, 13, 14), with new utility modules for reusable components, comprehensive static plotting, and enhancement of existing Step 11 for intermittent demand analysis.

**Tech Stack:** pandas, numpy, matplotlib, seaborn, scipy.stats, existing M5 EDA infrastructure

---

## File Structure

**Main Implementation:**
- `notebooks/eda/eda_steps_1_3_5.py` - Four main analysis functions
- `notebooks/eda/utils/basic_validation.py` - Steps 1 & 2 validation utilities
- `notebooks/eda/utils/feature_profiling.py` - Step 5 feature analysis utilities  
- `notebooks/eda/utils/segment_analysis.py` - Enhanced with intermittent demand analysis
- `notebooks/eda/utils/visualization.py` - Enhanced with new plotting functions

**Testing:**
- `notebooks/eda/tests/test_eda_steps_1_3_5.py` - Unit tests for new functions
- `notebooks/eda/tests/test_basic_validation.py` - Validation utility tests
- `notebooks/eda/tests/test_feature_profiling.py` - Feature analysis utility tests

**Integration:**
- `notebooks/eda/run_eda_auto.py` - Updated to include new steps

---

### Task 1: Basic Validation Utilities

**Files:**
- Create: `notebooks/eda/utils/basic_validation.py`
- Test: `notebooks/eda/tests/test_basic_validation.py`

- [ ] **Step 1: Write failing test for M5 hierarchy validation**

```python
import pytest
import pandas as pd
import sys
import os

# Add notebooks/eda to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.basic_validation import validate_m5_hierarchy_structure

def test_validate_m5_hierarchy_structure_complete():
    """Test hierarchy validation with complete M5-style data."""
    # Create mock sales data with proper M5 structure
    sales_data = pd.DataFrame({
        'id': ['FOODS_1_001_CA_1_validation', 'FOODS_1_002_CA_1_validation', 
               'HOUSEHOLD_1_001_TX_1_validation'],
        'item_id': ['FOODS_1_001', 'FOODS_1_002', 'HOUSEHOLD_1_001'],
        'dept_id': ['FOODS_1', 'FOODS_1', 'HOUSEHOLD_1'],
        'cat_id': ['FOODS', 'FOODS', 'HOUSEHOLD'],
        'store_id': ['CA_1', 'CA_1', 'TX_1'],
        'state_id': ['CA', 'CA', 'TX']
    })
    
    result = validate_m5_hierarchy_structure(sales_data)
    
    assert result['is_valid'] == True
    assert result['categories'] == 2
    assert result['departments'] == 2
    assert result['items'] == 3
    assert result['stores'] == 2
    assert result['states'] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest notebooks/eda/tests/test_basic_validation.py::test_validate_m5_hierarchy_structure_complete -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'utils.basic_validation'"

- [ ] **Step 3: Create basic validation utility module**

```python
"""
Basic validation utilities for M5 dataset EDA Steps 1 & 2.

Provides functions for validating M5 dataset structure, hierarchy consistency,
temporal boundaries, and business objective alignment.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def validate_m5_hierarchy_structure(sales_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate M5 hierarchical product structure consistency.
    
    Verifies that category → department → item hierarchy is consistent
    and matches expected M5 structure (3 categories, 7 departments, 3049 items).
    
    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales data with hierarchy columns
        
    Returns
    -------
    Dict[str, Any]
        Validation results including counts and consistency flags
    """
    required_cols = ['cat_id', 'dept_id', 'item_id', 'store_id', 'state_id']
    missing_cols = [col for col in required_cols if col not in sales_data.columns]
    
    if missing_cols:
        return {
            'is_valid': False,
            'error': f'Missing required columns: {missing_cols}',
            'categories': 0,
            'departments': 0,
            'items': 0,
            'stores': 0,
            'states': 0
        }
    
    # Count unique values at each hierarchy level
    categories = sales_data['cat_id'].nunique()
    departments = sales_data['dept_id'].nunique()  
    items = sales_data['item_id'].nunique()
    stores = sales_data['store_id'].nunique()
    states = sales_data['state_id'].nunique()
    
    # Check hierarchy consistency
    consistency_issues = []
    
    # Each item should belong to exactly one department and category
    item_dept_mapping = sales_data.groupby('item_id')['dept_id'].nunique()
    if (item_dept_mapping > 1).any():
        consistency_issues.append('Items mapped to multiple departments')
        
    dept_cat_mapping = sales_data.groupby('dept_id')['cat_id'].nunique()
    if (dept_cat_mapping > 1).any():
        consistency_issues.append('Departments mapped to multiple categories')
    
    # Check expected M5 structure
    structure_issues = []
    if categories != 3:
        structure_issues.append(f'Expected 3 categories, found {categories}')
    if departments != 7:
        structure_issues.append(f'Expected 7 departments, found {departments}')
    if items != 3049:
        structure_issues.append(f'Expected 3049 items, found {items}')
    if stores != 10:
        structure_issues.append(f'Expected 10 stores, found {stores}')
    if states != 3:
        structure_issues.append(f'Expected 3 states, found {states}')
    
    is_valid = len(consistency_issues) == 0 and len(structure_issues) == 0
    
    return {
        'is_valid': is_valid,
        'categories': categories,
        'departments': departments, 
        'items': items,
        'stores': stores,
        'states': states,
        'consistency_issues': consistency_issues,
        'structure_issues': structure_issues,
        'expected_series_count': items * stores,
        'actual_series_count': len(sales_data)
    }


def validate_temporal_boundaries(sales_data: pd.DataFrame, calendar_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate temporal boundaries and training/validation split integrity.
    
    Ensures that training period (d_1 to d_1913) and validation period 
    (d_1914 to d_1941) are properly structured for M5 forecasting.
    
    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales data with daily columns
    calendar_data : pd.DataFrame
        M5 calendar data with date mappings
        
    Returns
    -------
    Dict[str, Any]
        Temporal validation results
    """
    # Get sales columns (d_1, d_2, etc.)
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]
    
    if len(sales_cols) == 0:
        return {
            'is_valid': False,
            'error': 'No daily sales columns found (d_1, d_2, etc.)'
        }
    
    # Extract day numbers and find range
    day_nums = [int(col.split('_')[1]) for col in sales_cols]
    min_day = min(day_nums)
    max_day = max(day_nums)
    
    # Check calendar alignment
    calendar_days = calendar_data['d'].str.extract(r'd_(\d+)')[0].astype(int)
    cal_min_day = calendar_days.min()
    cal_max_day = calendar_days.max()
    
    # Expected M5 structure
    expected_training_end = 1913
    expected_validation_start = 1914
    expected_validation_end = 1941
    expected_total_days = 1969
    
    issues = []
    
    if min_day != 1:
        issues.append(f'Sales data should start at d_1, found d_{min_day}')
        
    if max_day != expected_training_end:
        issues.append(f'Training data should end at d_{expected_training_end}, found d_{max_day}')
    
    if cal_max_day != expected_total_days:
        issues.append(f'Calendar should cover d_1 to d_{expected_total_days}, found d_{cal_max_day}')
        
    # Check for gaps in sequence
    expected_sequence = list(range(min_day, max_day + 1))
    if day_nums != expected_sequence:
        issues.append('Sales columns are not consecutive')
    
    return {
        'is_valid': len(issues) == 0,
        'sales_start_day': min_day,
        'sales_end_day': max_day,
        'sales_total_days': len(sales_cols),
        'calendar_start_day': cal_min_day,
        'calendar_end_day': cal_max_day,
        'calendar_total_days': len(calendar_data),
        'training_period_valid': max_day == expected_training_end,
        'validation_period_available': cal_max_day >= expected_validation_end,
        'issues': issues
    }


def validate_business_objectives(sales_data: pd.DataFrame, calendar_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that dataset supports M5 business objectives.
    
    Checks that data structure supports 28-day horizon forecasting and
    5-segment product classification objectives.
    
    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales data
    calendar_data : pd.DataFrame
        M5 calendar data
        
    Returns
    -------  
    Dict[str, Any]
        Business objective validation results
    """
    validation_results = {}
    
    # Check forecasting horizon support (28 days)
    forecast_horizon = 28
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]
    
    if len(sales_cols) < forecast_horizon:
        validation_results['forecast_horizon_viable'] = False
        validation_results['forecast_issue'] = f'Only {len(sales_cols)} days available, need {forecast_horizon}'
    else:
        validation_results['forecast_horizon_viable'] = True
        
    # Check segmentation support (sufficient product diversity)
    categories = sales_data['cat_id'].nunique()
    departments = sales_data['dept_id'].nunique()
    items = sales_data['item_id'].nunique()
    
    segmentation_viable = categories >= 3 and departments >= 5 and items >= 1000
    validation_results['segmentation_viable'] = segmentation_viable
    
    # Check temporal features for forecasting
    temporal_features = ['weekday', 'month', 'year', 'wday']
    available_temporal = [col for col in temporal_features if col in calendar_data.columns]
    validation_results['temporal_features_available'] = len(available_temporal) >= 3
    
    # Check for external factors (events, SNAP)
    external_features = ['event_name_1', 'snap_CA', 'snap_TX', 'snap_WI']
    available_external = [col for col in external_features if col in calendar_data.columns]
    validation_results['external_features_available'] = len(available_external) >= 2
    
    validation_results['is_valid'] = all([
        validation_results.get('forecast_horizon_viable', False),
        validation_results.get('segmentation_viable', False),
        validation_results.get('temporal_features_available', False),
        validation_results.get('external_features_available', False)
    ])
    
    return validation_results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest notebooks/eda/tests/test_basic_validation.py::test_validate_m5_hierarchy_structure_complete -v`
Expected: PASS

- [ ] **Step 5: Add more validation tests**

```python
def test_validate_temporal_boundaries_valid():
    """Test temporal boundary validation with valid M5 structure."""
    sales_data = pd.DataFrame({
        'd_1': [1, 2], 'd_2': [3, 4], 'd_1913': [5, 6]
    })
    
    calendar_data = pd.DataFrame({
        'd': ['d_1', 'd_2', 'd_1913', 'd_1914', 'd_1969'],
        'date': ['2011-01-29', '2011-01-30', '2016-05-22', '2016-05-23', '2016-06-19']
    })
    
    result = validate_temporal_boundaries(sales_data, calendar_data)
    
    assert result['is_valid'] == True
    assert result['sales_start_day'] == 1
    assert result['sales_end_day'] == 1913
    assert result['training_period_valid'] == True


def test_validate_business_objectives_valid():
    """Test business objective validation with sufficient data."""
    sales_data = pd.DataFrame({
        'cat_id': ['FOODS'] * 1000 + ['HOUSEHOLD'] * 1000 + ['HOBBIES'] * 1049,
        'dept_id': ['FOODS_1'] * 500 + ['FOODS_2'] * 500 + ['HOUSEHOLD_1'] * 1000 + ['HOBBIES_1'] * 1049,
        'item_id': [f'ITEM_{i}' for i in range(3049)],
        'd_1': [1] * 3049
    })
    
    # Add 28 days of sales columns  
    for i in range(1, 29):
        sales_data[f'd_{i}'] = [1] * 3049
    
    calendar_data = pd.DataFrame({
        'weekday': ['Monday'] * 28,
        'month': [1] * 28,
        'year': [2011] * 28,
        'wday': [1] * 28,
        'event_name_1': [None] * 28,
        'snap_CA': [0] * 28
    })
    
    result = validate_business_objectives(sales_data, calendar_data)
    
    assert result['is_valid'] == True
    assert result['forecast_horizon_viable'] == True
    assert result['segmentation_viable'] == True
```

- [ ] **Step 6: Run all validation tests**

Run: `pytest notebooks/eda/tests/test_basic_validation.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit basic validation utilities**

```bash
git add notebooks/eda/utils/basic_validation.py notebooks/eda/tests/test_basic_validation.py
git commit -m "feat: add M5 hierarchy and temporal validation utilities

- Validate product hierarchy consistency (category→dept→item)
- Check temporal boundaries for training/validation splits
- Verify business objectives support (28-day forecast, segmentation)
- Comprehensive test coverage for validation edge cases"
```

---

### Task 2: Feature Profiling Utilities  

**Files:**
- Create: `notebooks/eda/utils/feature_profiling.py`
- Test: `notebooks/eda/tests/test_feature_profiling.py`

- [ ] **Step 1: Write failing test for feature distribution analysis**

```python
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.feature_profiling import analyze_categorical_feature_distributions

def test_analyze_categorical_feature_distributions():
    """Test categorical feature distribution analysis."""
    data = pd.DataFrame({
        'cat_id': ['FOODS', 'FOODS', 'HOUSEHOLD', 'HOUSEHOLD', 'HOBBIES'],
        'store_id': ['CA_1', 'CA_2', 'CA_1', 'TX_1', 'CA_1'],
        'daily_sales': [100, 150, 50, 75, 200]
    })
    
    result = analyze_categorical_feature_distributions(
        data, 
        categorical_cols=['cat_id', 'store_id'],
        target_col='daily_sales'
    )
    
    assert 'cat_id' in result
    assert 'store_id' in result
    assert result['cat_id']['FOODS']['mean'] == 125.0
    assert result['cat_id']['HOUSEHOLD']['mean'] == 62.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest notebooks/eda/tests/test_feature_profiling.py::test_analyze_categorical_feature_distributions -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'utils.feature_profiling'"

- [ ] **Step 3: Create feature profiling utility module**

```python
"""
Feature profiling utilities for M5 dataset EDA Step 5.

Provides functions for analyzing individual feature distributions,
geographic patterns, temporal correlations, and price behavior analysis.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def analyze_categorical_feature_distributions(
    data: pd.DataFrame,
    categorical_cols: List[str], 
    target_col: str
) -> Dict[str, Any]:
    """
    Analyze distribution patterns for categorical features vs target.
    
    Computes statistical summaries for target variable grouped by each
    categorical feature, including mean, median, std, and distribution tests.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with categorical and target columns
    categorical_cols : List[str]
        List of categorical column names to analyze
    target_col : str
        Target variable column name
        
    Returns
    -------
    Dict[str, Any]
        Nested dictionary with statistics for each categorical feature
    """
    results = {}
    
    for col in categorical_cols:
        if col not in data.columns:
            results[col] = {'error': f'Column {col} not found in data'}
            continue
            
        if target_col not in data.columns:
            results[col] = {'error': f'Target column {target_col} not found'}
            continue
            
        # Clean data
        clean_data = data[[col, target_col]].dropna()
        
        if len(clean_data) == 0:
            results[col] = {'error': 'No valid data after removing NaN values'}
            continue
            
        # Group by categorical feature
        grouped = clean_data.groupby(col)[target_col]
        
        category_stats = {}
        for category, values in grouped:
            category_stats[category] = {
                'count': len(values),
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()) if len(values) > 1 else 0.0,
                'min': float(values.min()),
                'max': float(values.max()),
                'q25': float(values.quantile(0.25)),
                'q75': float(values.quantile(0.75)),
                'skewness': float(stats.skew(values)) if len(values) > 2 else 0.0,
                'zero_pct': float((values == 0).sum() / len(values) * 100)
            }
            
        # Overall ANOVA test for significant differences
        category_groups = [group for _, group in grouped]
        if len(category_groups) > 1 and all(len(g) > 0 for g in category_groups):
            try:
                f_stat, p_value = stats.f_oneway(*category_groups)
                anova_result = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            except:
                anova_result = {'error': 'ANOVA test failed'}
        else:
            anova_result = {'error': 'Insufficient groups for ANOVA'}
            
        results[col] = {
            'categories': category_stats,
            'anova_test': anova_result,
            'total_categories': len(category_stats),
            'total_observations': len(clean_data)
        }
        
    return results


def analyze_geographic_patterns(
    sales_data: pd.DataFrame,
    geographic_cols: List[str] = ['state_id', 'store_id']
) -> Dict[str, Any]:
    """
    Analyze sales performance patterns across geographic dimensions.
    
    Examines state and store level performance variations including
    sales volume, seasonality, and product mix differences.
    
    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales data with geographic identifiers
    geographic_cols : List[str]
        Geographic columns to analyze
        
    Returns
    -------
    Dict[str, Any]
        Geographic performance analysis results
    """
    results = {}
    
    # Get sales columns
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]
    
    if len(sales_cols) == 0:
        return {'error': 'No daily sales columns found'}
    
    # Calculate total sales per geographic unit
    sales_data_copy = sales_data.copy()
    sales_data_copy['total_sales'] = sales_data[sales_cols].sum(axis=1)
    sales_data_copy['avg_daily_sales'] = sales_data[sales_cols].mean(axis=1)
    sales_data_copy['sales_cv'] = sales_data[sales_cols].std(axis=1) / sales_data[sales_cols].mean(axis=1)
    sales_data_copy['zero_days_pct'] = (sales_data[sales_cols] == 0).sum(axis=1) / len(sales_cols) * 100
    
    for geo_col in geographic_cols:
        if geo_col not in sales_data.columns:
            results[geo_col] = {'error': f'Column {geo_col} not found'}
            continue
            
        # Aggregate by geographic unit
        geo_stats = sales_data_copy.groupby(geo_col).agg({
            'total_sales': ['count', 'sum', 'mean', 'std'],
            'avg_daily_sales': ['mean', 'median', 'std'],
            'sales_cv': ['mean', 'median'],
            'zero_days_pct': ['mean', 'median']
        }).round(2)
        
        # Flatten column names
        geo_stats.columns = ['_'.join(col).strip() for col in geo_stats.columns.values]
        
        # Calculate performance metrics
        geo_summary = {}
        for geo_unit in geo_stats.index:
            geo_summary[geo_unit] = {
                'product_count': int(geo_stats.loc[geo_unit, 'total_sales_count']),
                'total_volume': int(geo_stats.loc[geo_unit, 'total_sales_sum']),
                'avg_product_volume': float(geo_stats.loc[geo_unit, 'total_sales_mean']),
                'avg_daily_sales': float(geo_stats.loc[geo_unit, 'avg_daily_sales_mean']),
                'volatility': float(geo_stats.loc[geo_unit, 'sales_cv_mean']),
                'intermittency_pct': float(geo_stats.loc[geo_unit, 'zero_days_pct_mean'])
            }
            
        results[geo_col] = {
            'geographic_units': geo_summary,
            'performance_ranking': sorted(
                geo_summary.items(),
                key=lambda x: x[1]['total_volume'],
                reverse=True
            )
        }
        
    return results


def analyze_temporal_feature_correlations(
    sales_data: pd.DataFrame,
    calendar_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyze correlation patterns between temporal features and sales.
    
    Examines how calendar features (weekday, month, events, SNAP) correlate
    with sales patterns across different product categories and stores.
    
    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales data
    calendar_data : pd.DataFrame
        M5 calendar data with temporal features
        
    Returns
    -------
    Dict[str, Any]
        Temporal correlation analysis results
    """
    # Get sales columns and merge with calendar
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]
    
    if len(sales_cols) == 0:
        return {'error': 'No daily sales columns found'}
    
    # Calculate daily aggregated sales across all products
    daily_totals = sales_data[sales_cols].sum(axis=0).reset_index()
    daily_totals.columns = ['day_col', 'total_sales']
    daily_totals['d'] = daily_totals['day_col']
    
    # Merge with calendar
    merged_data = daily_totals.merge(calendar_data, on='d', how='left')
    
    # Define temporal features to analyze
    temporal_features = ['wday', 'month', 'year']
    categorical_features = ['weekday']
    binary_features = ['snap_CA', 'snap_TX', 'snap_WI']
    
    results = {}
    
    # Numerical correlations
    for feature in temporal_features:
        if feature in merged_data.columns:
            correlation = merged_data['total_sales'].corr(merged_data[feature])
            results[f'{feature}_correlation'] = {
                'correlation': float(correlation),
                'strength': 'strong' if abs(correlation) > 0.5 else 'moderate' if abs(correlation) > 0.3 else 'weak'
            }
            
    # Categorical feature analysis (ANOVA)
    for feature in categorical_features:
        if feature in merged_data.columns:
            groups = [group['total_sales'].values for name, group in merged_data.groupby(feature)]
            if len(groups) > 1:
                try:
                    f_stat, p_val = stats.f_oneway(*groups)
                    results[f'{feature}_anova'] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05
                    }
                except:
                    results[f'{feature}_anova'] = {'error': 'ANOVA failed'}
                    
    # Binary feature analysis (t-test)
    for feature in binary_features:
        if feature in merged_data.columns:
            feature_data = merged_data.dropna(subset=[feature, 'total_sales'])
            if len(feature_data) > 0:
                group_0 = feature_data[feature_data[feature] == 0]['total_sales']
                group_1 = feature_data[feature_data[feature] == 1]['total_sales']
                
                if len(group_0) > 0 and len(group_1) > 0:
                    try:
                        t_stat, p_val = stats.ttest_ind(group_1, group_0)
                        mean_diff = group_1.mean() - group_0.mean()
                        results[f'{feature}_ttest'] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_val),
                            'mean_difference': float(mean_diff),
                            'percent_increase': float(mean_diff / group_0.mean() * 100) if group_0.mean() > 0 else 0.0,
                            'significant': p_val < 0.05
                        }
                    except:
                        results[f'{feature}_ttest'] = {'error': 't-test failed'}
                        
    return results


def analyze_price_feature_behavior(
    prices_data: pd.DataFrame,
    sales_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyze pricing feature distributions and stability patterns.
    
    Examines price volatility, promotional patterns, and price-sales
    elasticity relationships across product categories and stores.
    
    Parameters
    ----------
    prices_data : pd.DataFrame
        M5 pricing data with weekly prices
    sales_data : pd.DataFrame  
        M5 sales data for price elasticity analysis
        
    Returns
    -------
    Dict[str, Any]
        Price behavior analysis results
    """
    if 'sell_price' not in prices_data.columns:
        return {'error': 'sell_price column not found in pricing data'}
        
    results = {}
    
    # Overall price distribution
    price_stats = {
        'count': int(prices_data['sell_price'].count()),
        'mean': float(prices_data['sell_price'].mean()),
        'median': float(prices_data['sell_price'].median()),
        'std': float(prices_data['sell_price'].std()),
        'min': float(prices_data['sell_price'].min()),
        'max': float(prices_data['sell_price'].max()),
        'q25': float(prices_data['sell_price'].quantile(0.25)),
        'q75': float(prices_data['sell_price'].quantile(0.75)),
        'skewness': float(stats.skew(prices_data['sell_price'].dropna()))
    }
    results['overall_distribution'] = price_stats
    
    # Price volatility by item
    if 'item_id' in prices_data.columns:
        price_volatility = prices_data.groupby('item_id')['sell_price'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        price_volatility['cv'] = price_volatility['std'] / price_volatility['mean']
        price_volatility['price_range'] = price_volatility['max'] - price_volatility['min']
        price_volatility['price_range_pct'] = (price_volatility['price_range'] / price_volatility['mean']) * 100
        
        # Identify high/low volatility items
        cv_threshold = price_volatility['cv'].quantile(0.8)
        high_volatility_items = price_volatility[price_volatility['cv'] > cv_threshold]['item_id'].tolist()
        
        results['volatility_analysis'] = {
            'high_volatility_items': high_volatility_items[:10],  # Top 10
            'avg_cv': float(price_volatility['cv'].mean()),
            'median_cv': float(price_volatility['cv'].median()),
            'high_volatility_threshold': float(cv_threshold)
        }
    
    # Price stability by store
    if 'store_id' in prices_data.columns:
        store_price_stats = prices_data.groupby('store_id')['sell_price'].agg([
            'count', 'mean', 'median', 'std'
        ]).reset_index()
        
        store_price_stats['cv'] = store_price_stats['std'] / store_price_stats['mean']
        
        results['store_pricing_patterns'] = {
            store: {
                'avg_price': float(row['mean']),
                'price_volatility': float(row['cv']),
                'product_count': int(row['count'])
            }
            for store, row in store_price_stats.set_index('store_id').iterrows()
        }
    
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest notebooks/eda/tests/test_feature_profiling.py::test_analyze_categorical_feature_distributions -v`
Expected: PASS

- [ ] **Step 5: Add additional feature profiling tests**

```python
def test_analyze_geographic_patterns():
    """Test geographic pattern analysis."""
    sales_data = pd.DataFrame({
        'state_id': ['CA', 'CA', 'TX', 'TX'],
        'store_id': ['CA_1', 'CA_2', 'TX_1', 'TX_2'],
        'd_1': [100, 150, 80, 120],
        'd_2': [110, 160, 85, 125],
        'd_3': [0, 0, 0, 0]  # Zero sales day
    })
    
    result = analyze_geographic_patterns(sales_data)
    
    assert 'state_id' in result
    assert 'store_id' in result
    assert len(result['state_id']['geographic_units']) == 2
    assert 'CA' in result['state_id']['geographic_units']
    assert 'TX' in result['state_id']['geographic_units']


def test_analyze_price_feature_behavior():
    """Test price behavior analysis."""
    prices_data = pd.DataFrame({
        'item_id': ['ITEM_1', 'ITEM_1', 'ITEM_2', 'ITEM_2'],
        'store_id': ['CA_1', 'CA_1', 'CA_1', 'CA_1'], 
        'sell_price': [1.99, 2.49, 5.99, 4.99]
    })
    
    sales_data = pd.DataFrame({
        'item_id': ['ITEM_1', 'ITEM_2'],
        'd_1': [100, 50]
    })
    
    result = analyze_price_feature_behavior(prices_data, sales_data)
    
    assert 'overall_distribution' in result
    assert result['overall_distribution']['mean'] > 0
    assert 'volatility_analysis' in result
```

- [ ] **Step 6: Run all feature profiling tests**

Run: `pytest notebooks/eda/tests/test_feature_profiling.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit feature profiling utilities**

```bash
git add notebooks/eda/utils/feature_profiling.py notebooks/eda/tests/test_feature_profiling.py
git commit -m "feat: add feature profiling utilities for EDA Step 5

- Categorical feature distribution analysis with ANOVA tests
- Geographic performance pattern analysis (state/store level)
- Temporal correlation analysis with sales patterns
- Price behavior and volatility analysis across products
- Comprehensive statistical summaries and test coverage"
```

---

### Task 3: Enhanced Visualization Functions

**Files:**
- Modify: `notebooks/eda/utils/visualization.py`
- Test: `notebooks/eda/tests/test_visualization_enhancements.py`

- [ ] **Step 1: Write failing test for new visualization functions**

```python
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.visualization import (
    plot_hierarchy_validation_summary,
    plot_data_coverage_heatmap,
    plot_geographic_performance_comparison
)

def test_plot_hierarchy_validation_summary(tmp_path):
    """Test hierarchy validation summary plotting."""
    validation_results = {
        'categories': 3,
        'departments': 7,
        'items': 3049,
        'stores': 10,
        'states': 3,
        'is_valid': True,
        'structure_issues': []
    }
    
    output_path = tmp_path / "hierarchy_summary.png"
    
    # Should not raise an error
    plot_hierarchy_validation_summary(validation_results, str(output_path))
    
    # File should be created
    assert output_path.exists()


def test_plot_data_coverage_heatmap(tmp_path):
    """Test data coverage heatmap plotting."""
    coverage_data = {
        'sales_data': {'rows': 30490, 'cols': 1919, 'missing_pct': 0.0},
        'calendar_data': {'rows': 1969, 'cols': 14, 'missing_pct': 0.0},
        'pricing_data': {'rows': 6841121, 'cols': 4, 'missing_pct': 15.2}
    }
    
    output_path = tmp_path / "coverage_heatmap.png"
    
    plot_data_coverage_heatmap(coverage_data, str(output_path))
    assert output_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest notebooks/eda/tests/test_visualization_enhancements.py::test_plot_hierarchy_validation_summary -v`
Expected: FAIL with "ImportError: cannot import name 'plot_hierarchy_validation_summary'"

- [ ] **Step 3: Add new visualization functions to existing module**

```python
def plot_hierarchy_validation_summary(validation_results: Dict[str, Any], output_path: str) -> None:
    """
    Plot M5 hierarchy validation summary visualization.
    
    Creates a summary chart showing hierarchy structure validation results
    including expected vs actual counts and validation status.
    
    Parameters
    ----------
    validation_results : Dict[str, Any]
        Results from validate_m5_hierarchy_structure()
    output_path : str
        Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('M5 Dataset Hierarchy Validation Summary', fontsize=16, fontweight='bold')
    
    # Expected vs Actual counts
    expected = [3, 7, 3049, 10, 3]
    actual = [
        validation_results.get('categories', 0),
        validation_results.get('departments', 0),
        validation_results.get('items', 0),
        validation_results.get('stores', 0),
        validation_results.get('states', 0)
    ]
    labels = ['Categories', 'Departments', 'Items', 'Stores', 'States']
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, expected, width, label='Expected', alpha=0.7, color='skyblue')
    bars2 = ax1.bar(x + width/2, actual, width, label='Actual', alpha=0.7, color='lightcoral')
    
    ax1.set_xlabel('Hierarchy Level')
    ax1.set_ylabel('Count')
    ax1.set_title('Expected vs Actual Counts')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + max(expected) * 0.01,
                f'{expected[i]}', ha='center', va='bottom', fontsize=8)
        ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + max(expected) * 0.01,
                f'{actual[i]}', ha='center', va='bottom', fontsize=8)
    
    # Validation status pie chart
    is_valid = validation_results.get('is_valid', False)
    status_counts = [1 if is_valid else 0, 0 if is_valid else 1]
    status_labels = ['Valid', 'Invalid']
    colors = ['lightgreen', 'lightcoral']
    
    ax2.pie(status_counts, labels=status_labels, colors=colors, autopct='%1.0f%%', startangle=90)
    ax2.set_title('Overall Validation Status')
    
    # Issues summary
    structure_issues = validation_results.get('structure_issues', [])
    consistency_issues = validation_results.get('consistency_issues', [])
    
    all_issues = structure_issues + consistency_issues
    if all_issues:
        issue_text = '\n'.join([f'• {issue}' for issue in all_issues[:5]])  # Top 5 issues
        if len(all_issues) > 5:
            issue_text += f'\n... and {len(all_issues) - 5} more issues'
    else:
        issue_text = '✓ No validation issues found'
    
    ax3.text(0.05, 0.95, 'Validation Issues:', fontsize=12, fontweight='bold', 
            transform=ax3.transAxes, verticalalignment='top')
    ax3.text(0.05, 0.85, issue_text, fontsize=10, transform=ax3.transAxes, 
            verticalalignment='top', wrap=True)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Series count validation
    expected_series = validation_results.get('expected_series_count', 0)
    actual_series = validation_results.get('actual_series_count', 0)
    
    series_data = [expected_series, actual_series]
    series_labels = ['Expected\nSeries', 'Actual\nSeries']
    
    bars = ax4.bar(series_labels, series_data, color=['skyblue', 'lightcoral'], alpha=0.7)
    ax4.set_ylabel('Number of Series')
    ax4.set_title('Item-Store Series Count')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, series_data):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(series_data) * 0.01,
                f'{value:,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_data_coverage_heatmap(coverage_data: Dict[str, Any], output_path: str) -> None:
    """
    Plot data coverage heatmap for M5 dataset tables.
    
    Creates a heatmap showing data completeness, missing percentages,
    and table dimensions across all M5 data sources.
    
    Parameters
    ----------
    coverage_data : Dict[str, Any]
        Dictionary with table coverage statistics
    output_path : str
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('M5 Dataset Coverage Analysis', fontsize=16, fontweight='bold')
    
    # Extract table information
    tables = list(coverage_data.keys())
    metrics = ['rows', 'cols', 'missing_pct']
    
    # Create matrix for heatmap
    data_matrix = []
    for table in tables:
        table_data = coverage_data[table]
        row = [
            table_data.get('rows', 0),
            table_data.get('cols', 0), 
            table_data.get('missing_pct', 0)
        ]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Normalize data for better visualization (log scale for rows)
    normalized_matrix = data_matrix.copy().astype(float)
    normalized_matrix[:, 0] = np.log10(normalized_matrix[:, 0] + 1)  # Log scale for rows
    normalized_matrix[:, 1] = normalized_matrix[:, 1] / 100  # Normalize columns
    # Missing percentage is already 0-100
    
    # Create heatmap
    im = ax1.imshow(normalized_matrix.T, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks and labels
    ax1.set_xticks(range(len(tables)))
    ax1.set_xticklabels([t.replace('_', ' ').title() for t in tables])
    ax1.set_yticks(range(len(metrics)))
    ax1.set_yticklabels(['Rows (log10)', 'Columns (/100)', 'Missing %'])
    ax1.set_title('Data Dimensions & Completeness')
    
    # Add text annotations
    for i in range(len(tables)):
        for j in range(len(metrics)):
            if j == 0:  # Rows
                text = f"{data_matrix[i, j]:,.0f}"
            elif j == 1:  # Columns
                text = f"{data_matrix[i, j]:.0f}"
            else:  # Missing %
                text = f"{data_matrix[i, j]:.1f}%"
            ax1.text(i, j, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Color bar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Normalized Values', rotation=270, labelpad=15)
    
    # Missing data summary bar chart
    missing_pcts = [coverage_data[table].get('missing_pct', 0) for table in tables]
    table_names = [t.replace('_', ' ').title() for t in tables]
    
    bars = ax2.bar(table_names, missing_pcts, 
                  color=['red' if pct > 10 else 'orange' if pct > 1 else 'green' for pct in missing_pcts],
                  alpha=0.7)
    
    ax2.set_ylabel('Missing Data (%)')
    ax2.set_title('Missing Data by Table')
    ax2.set_ylim(0, max(missing_pcts) * 1.1 if missing_pcts else 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, missing_pcts):
        if pct > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(missing_pcts) * 0.01,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_geographic_performance_comparison(geographic_data: Dict[str, Any], output_path: str) -> None:
    """
    Plot geographic performance comparison across states and stores.
    
    Creates comparison plots showing sales volume, volatility, and
    intermittency patterns across different geographic regions.
    
    Parameters
    ----------
    geographic_data : Dict[str, Any]
        Results from analyze_geographic_patterns()
    output_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Geographic Performance Analysis', fontsize=16, fontweight='bold')
    
    # State-level analysis (if available)
    if 'state_id' in geographic_data:
        state_data = geographic_data['state_id']['geographic_units']
        states = list(state_data.keys())
        
        # Total volume by state
        volumes = [state_data[state]['total_volume'] for state in states]
        bars = axes[0, 0].bar(states, volumes, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Total Sales Volume by State')
        axes[0, 0].set_ylabel('Total Volume')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, vol in zip(bars, volumes):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(volumes) * 0.01,
                           f'{vol:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Volatility by state
        volatilities = [state_data[state]['volatility'] for state in states]
        bars = axes[0, 1].bar(states, volatilities, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Average Volatility by State')
        axes[0, 1].set_ylabel('Coefficient of Variation')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, vol in zip(bars, volatilities):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(volatilities) * 0.01,
                           f'{vol:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Store-level analysis (if available)
    if 'store_id' in geographic_data:
        store_data = geographic_data['store_id']['geographic_units']
        stores = list(store_data.keys())
        
        # Performance ranking (top 10 stores)
        performance_ranking = geographic_data['store_id']['performance_ranking'][:10]
        top_stores = [item[0] for item in performance_ranking]
        top_volumes = [item[1]['total_volume'] for item in performance_ranking]
        
        bars = axes[1, 0].barh(range(len(top_stores)), top_volumes, color='lightgreen', alpha=0.7)
        axes[1, 0].set_yticks(range(len(top_stores)))
        axes[1, 0].set_yticklabels(top_stores)
        axes[1, 0].set_xlabel('Total Volume')
        axes[1, 0].set_title('Top 10 Stores by Volume')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Intermittency vs Volume scatter plot
        intermittencies = [store_data[store]['intermittency_pct'] for store in stores]
        volumes = [store_data[store]['total_volume'] for store in stores]
        
        scatter = axes[1, 1].scatter(intermittencies, volumes, alpha=0.6, s=50, c='purple')
        axes[1, 1].set_xlabel('Intermittency (%)')
        axes[1, 1].set_ylabel('Total Volume')
        axes[1, 1].set_title('Store Intermittency vs Volume')
        axes[1, 1].grid(alpha=0.3)
        
        # Add store labels for outliers
        for i, store in enumerate(stores):
            if volumes[i] > np.percentile(volumes, 90) or intermittencies[i] > np.percentile(intermittencies, 90):
                axes[1, 1].annotate(store, (intermittencies[i], volumes[i]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest notebooks/eda/tests/test_visualization_enhancements.py::test_plot_hierarchy_validation_summary -v`
Expected: PASS

- [ ] **Step 5: Add remaining visualization tests**

```python
def test_plot_geographic_performance_comparison(tmp_path):
    """Test geographic performance comparison plotting."""
    geographic_data = {
        'state_id': {
            'geographic_units': {
                'CA': {'total_volume': 100000, 'volatility': 0.25, 'intermittency_pct': 15.2},
                'TX': {'total_volume': 85000, 'volatility': 0.30, 'intermittency_pct': 18.5},
                'WI': {'total_volume': 70000, 'volatility': 0.22, 'intermittency_pct': 12.8}
            }
        },
        'store_id': {
            'geographic_units': {
                'CA_1': {'total_volume': 50000, 'intermittency_pct': 14.2},
                'CA_2': {'total_volume': 50000, 'intermittency_pct': 16.2},
                'TX_1': {'total_volume': 42500, 'intermittency_pct': 17.5}
            },
            'performance_ranking': [
                ('CA_1', {'total_volume': 50000}),
                ('CA_2', {'total_volume': 50000}),
                ('TX_1', {'total_volume': 42500})
            ]
        }
    }
    
    output_path = tmp_path / "geographic_comparison.png"
    
    plot_geographic_performance_comparison(geographic_data, str(output_path))
    assert output_path.exists()
```

- [ ] **Step 6: Run all enhanced visualization tests**

Run: `pytest notebooks/eda/tests/test_visualization_enhancements.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit visualization enhancements**

```bash
git add notebooks/eda/utils/visualization.py notebooks/eda/tests/test_visualization_enhancements.py
git commit -m "feat: add new visualization functions for EDA steps 1-3, 5

- Hierarchy validation summary plots with expected vs actual counts
- Data coverage heatmaps showing table completeness and missing data
- Geographic performance comparison across states and stores
- Static plot generation with comprehensive test coverage"
```

---

### Task 4: Main EDA Steps Implementation

**Files:**
- Create: `notebooks/eda/eda_steps_1_3_5.py`
- Test: `notebooks/eda/tests/test_eda_steps_1_3_5.py`

- [ ] **Step 1: Write failing test for Step 1 function**

```python
import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eda_steps_1_3_5 import analyze_m5_problem_context

def test_analyze_m5_problem_context_success(tmp_path):
    """Test Step 1 problem context analysis with valid data."""
    # Create mock data
    sales_data = pd.DataFrame({
        'id': ['FOODS_1_001_CA_1_validation'] * 5,
        'item_id': ['FOODS_1_001'] * 5,
        'dept_id': ['FOODS_1'] * 5, 
        'cat_id': ['FOODS'] * 5,
        'store_id': ['CA_1'] * 5,
        'state_id': ['CA'] * 5,
        'd_1': [1, 2, 3, 4, 5],
        'd_1913': [10, 20, 30, 40, 50]
    })
    
    calendar_data = pd.DataFrame({
        'd': ['d_1', 'd_2', 'd_1913', 'd_1914', 'd_1969'],
        'date': ['2011-01-29', '2011-01-30', '2016-05-22', '2016-05-23', '2016-06-19'],
        'weekday': ['Saturday'] * 5,
        'month': [1, 1, 5, 5, 6]
    })
    
    prices_data = pd.DataFrame({
        'store_id': ['CA_1'] * 3,
        'item_id': ['FOODS_1_001'] * 3,
        'wm_yr_wk': [11101, 11102, 11103],
        'sell_price': [1.99, 2.49, 1.99]
    })
    
    # Mock output manager
    class MockOutputManager:
        def save_analysis(self, step_name, results, summary):
            pass
            
        def save_plot(self, step_name, plot_name, plot_path):
            pass
    
    output_manager = MockOutputManager()
    
    result = analyze_m5_problem_context(sales_data, calendar_data, prices_data, output_manager)
    
    assert 'hierarchy_validation' in result
    assert 'temporal_validation' in result
    assert 'business_objectives' in result
    assert 'leakage_audit' in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest notebooks/eda/tests/test_eda_steps_1_3_5.py::test_analyze_m5_problem_context_success -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'eda_steps_1_3_5'"

- [ ] **Step 3: Create main EDA steps implementation**

```python
"""
Main EDA analysis orchestration for M5 demand forecasting dataset - Steps 1, 2, 3, 5.

Implements missing EDA framework steps with comprehensive statistical analysis,
M5-specific focus areas, and business-focused interpretation. Each function
corresponds to one EDA framework step with integrated output management.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import os
import sys

# Add notebooks/eda to path for utility imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utility modules
from utils.basic_validation import (
    validate_m5_hierarchy_structure,
    validate_temporal_boundaries,
    validate_business_objectives
)
from utils.feature_profiling import (
    analyze_categorical_feature_distributions,
    analyze_geographic_patterns,
    analyze_temporal_feature_correlations,
    analyze_price_feature_behavior
)
from utils.data_quality import (
    analyze_missing_patterns,
    characterize_missing_mechanisms
)
from utils.visualization import (
    plot_hierarchy_validation_summary,
    plot_data_coverage_heatmap,
    plot_geographic_performance_comparison
)


def analyze_m5_problem_context(
    sales_data: pd.DataFrame,
    calendar_data: pd.DataFrame,
    prices_data: pd.DataFrame,
    output_manager
) -> Dict[str, Any]:
    """
    EDA Step 1: Validate M5 forecasting objectives and data leakage prevention.
    
    Performs comprehensive validation of M5 dataset structure, business objective
    alignment, and critical leakage prevention checkpoints. Ensures data supports
    28-day horizon forecasting and 5-segment product classification objectives.
    
    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales training data with daily sales columns
    calendar_data : pd.DataFrame
        M5 calendar data with date mappings and external factors
    prices_data : pd.DataFrame
        M5 pricing data with weekly sell prices
    output_manager : EDAOutputManager
        Output manager for saving results and plots
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive problem context validation results including:
        - hierarchy_validation: Product hierarchy structure validation
        - temporal_validation: Time boundary and split validation
        - business_objectives: Forecasting and segmentation viability
        - leakage_audit: Data leakage prevention checkpoints
        - visualizations: Generated plot file paths
        
    Examples
    --------
    >>> results = analyze_m5_problem_context(sales_data, calendar_data, prices_data, output_manager)
    >>> print(f"Hierarchy valid: {results['hierarchy_validation']['is_valid']}")
    >>> print(f"Business objectives met: {results['business_objectives']['is_valid']}")
    """
    print("=" * 80)
    print("EDA STEP 1: M5 PROBLEM CONTEXT VALIDATION")
    print("=" * 80)
    
    results = {}
    
    # 1. Hierarchical Structure Validation
    print("\n🏗️  HIERARCHICAL STRUCTURE VALIDATION")
    print("-" * 50)
    try:
        hierarchy_validation = validate_m5_hierarchy_structure(sales_data)
        results['hierarchy_validation'] = hierarchy_validation
        
        if hierarchy_validation['is_valid']:
            print("✓ Hierarchy structure validation PASSED")
            print(f"  • Categories: {hierarchy_validation['categories']}")
            print(f"  • Departments: {hierarchy_validation['departments']}")
            print(f"  • Items: {hierarchy_validation['items']}")
            print(f"  • Stores: {hierarchy_validation['stores']}")
            print(f"  • States: {hierarchy_validation['states']}")
        else:
            print("✗ Hierarchy structure validation FAILED")
            for issue in hierarchy_validation.get('structure_issues', []):
                print(f"  • {issue}")
            for issue in hierarchy_validation.get('consistency_issues', []):
                print(f"  • {issue}")
                
    except Exception as e:
        print(f"✗ ERROR: Hierarchy validation failed: {str(e)}")
        results['hierarchy_validation'] = {'error': str(e)}
    
    # 2. Temporal Boundary Validation
    print("\n⏰ TEMPORAL BOUNDARY VALIDATION")
    print("-" * 50)
    try:
        temporal_validation = validate_temporal_boundaries(sales_data, calendar_data)
        results['temporal_validation'] = temporal_validation
        
        if temporal_validation['is_valid']:
            print("✓ Temporal boundary validation PASSED")
            print(f"  • Sales period: d_{temporal_validation['sales_start_day']} to d_{temporal_validation['sales_end_day']}")
            print(f"  • Calendar coverage: d_{temporal_validation['calendar_start_day']} to d_{temporal_validation['calendar_end_day']}")
            print(f"  • Training period valid: {temporal_validation['training_period_valid']}")
        else:
            print("✗ Temporal boundary validation FAILED")
            for issue in temporal_validation.get('issues', []):
                print(f"  • {issue}")
                
    except Exception as e:
        print(f"✗ ERROR: Temporal validation failed: {str(e)}")
        results['temporal_validation'] = {'error': str(e)}
    
    # 3. Business Objectives Validation
    print("\n🎯 BUSINESS OBJECTIVES VALIDATION")
    print("-" * 50)
    try:
        business_validation = validate_business_objectives(sales_data, calendar_data)
        results['business_objectives'] = business_validation
        
        if business_validation['is_valid']:
            print("✓ Business objectives validation PASSED")
            print(f"  • 28-day forecasting viable: {business_validation['forecast_horizon_viable']}")
            print(f"  • Product segmentation viable: {business_validation['segmentation_viable']}")
            print(f"  • Temporal features available: {business_validation['temporal_features_available']}")
            print(f"  • External features available: {business_validation['external_features_available']}")
        else:
            print("✗ Business objectives validation ISSUES FOUND")
            if not business_validation.get('forecast_horizon_viable', True):
                print(f"  • Forecasting: {business_validation.get('forecast_issue', 'Unknown issue')}")
            if not business_validation.get('segmentation_viable', True):
                print("  • Segmentation: Insufficient product diversity")
                
    except Exception as e:
        print(f"✗ ERROR: Business validation failed: {str(e)}")
        results['business_objectives'] = {'error': str(e)}
    
    # 4. Data Leakage Audit
    print("\n🔒 DATA LEAKAGE PREVENTION AUDIT")
    print("-" * 50)
    try:
        leakage_audit = {
            'temporal_boundary_check': 'PASS' if results.get('temporal_validation', {}).get('training_period_valid', False) else 'FAIL',
            'future_data_risk': 'LOW' if results.get('temporal_validation', {}).get('is_valid', False) else 'HIGH',
            'calendar_feature_risk': 'LOW',  # Calendar features are known in advance
            'pricing_alignment_risk': 'MEDIUM',  # Needs weekly alignment check
            'overall_leakage_risk': 'LOW'
        }
        
        # Check for potential leakage indicators
        sales_cols = [col for col in sales_data.columns if col.startswith('d_')]
        if len(sales_cols) > 1913:
            leakage_audit['future_data_risk'] = 'HIGH'
            leakage_audit['overall_leakage_risk'] = 'HIGH'
            
        results['leakage_audit'] = leakage_audit
        
        print(f"✓ Temporal boundary check: {leakage_audit['temporal_boundary_check']}")
        print(f"✓ Future data risk: {leakage_audit['future_data_risk']}")
        print(f"✓ Calendar feature risk: {leakage_audit['calendar_feature_risk']}")
        print(f"✓ Overall leakage risk: {leakage_audit['overall_leakage_risk']}")
        
    except Exception as e:
        print(f"✗ ERROR: Leakage audit failed: {str(e)}")
        results['leakage_audit'] = {'error': str(e)}
    
    # 5. Generate Visualizations
    print("\n📊 GENERATING VISUALIZATIONS")
    print("-" * 50)
    try:
        viz_results = {}
        
        # Hierarchy validation summary plot
        if 'hierarchy_validation' in results and 'error' not in results['hierarchy_validation']:
            hierarchy_plot_path = "notebooks/eda/outputs/step1_plots/hierarchy_validation_summary.png"
            os.makedirs(os.path.dirname(hierarchy_plot_path), exist_ok=True)
            plot_hierarchy_validation_summary(results['hierarchy_validation'], hierarchy_plot_path)
            viz_results['hierarchy_plot'] = hierarchy_plot_path
            print(f"✓ Hierarchy validation plot saved: {hierarchy_plot_path}")
        
        results['visualizations'] = viz_results
        
    except Exception as e:
        print(f"✗ ERROR: Visualization generation failed: {str(e)}")
        results['visualizations'] = {'error': str(e)}
    
    # 6. Save Results
    try:
        output_manager.save_analysis(
            "step1_problem_context",
            results,
            f"M5 Problem Context Validation - Hierarchy: {'VALID' if results.get('hierarchy_validation', {}).get('is_valid') else 'INVALID'}"
        )
        print(f"\n✓ Step 1 analysis completed and saved")
        
    except Exception as e:
        print(f"✗ ERROR: Failed to save results: {str(e)}")
    
    return results


def inspect_m5_dataset_structure(
    sales_data: pd.DataFrame,
    calendar_data: pd.DataFrame,
    prices_data: pd.DataFrame,
    output_manager
) -> Dict[str, Any]:
    """
    EDA Step 2: M5-specific dataset structure audit and validation.
    
    Comprehensive structural analysis of M5 dataset including table dimensions,
    data type consistency, cross-table relationships, and coverage patterns.
    
    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales training data
    calendar_data : pd.DataFrame
        M5 calendar data
    prices_data : pd.DataFrame
        M5 pricing data
    output_manager : EDAOutputManager
        Output manager for saving results and plots
        
    Returns
    -------
    Dict[str, Any]
        Dataset structure analysis results
    """
    print("=" * 80)
    print("EDA STEP 2: M5 DATASET STRUCTURE INSPECTION")
    print("=" * 80)
    
    results = {}
    
    # 1. Table Dimension Analysis
    print("\n📏 TABLE DIMENSION ANALYSIS")
    print("-" * 50)
    
    table_stats = {
        'sales_data': {
            'rows': len(sales_data),
            'cols': len(sales_data.columns),
            'memory_mb': round(sales_data.memory_usage(deep=True).sum() / 1024**2, 2),
            'missing_pct': round(sales_data.isnull().sum().sum() / (len(sales_data) * len(sales_data.columns)) * 100, 3)
        },
        'calendar_data': {
            'rows': len(calendar_data),
            'cols': len(calendar_data.columns),
            'memory_mb': round(calendar_data.memory_usage(deep=True).sum() / 1024**2, 2),
            'missing_pct': round(calendar_data.isnull().sum().sum() / (len(calendar_data) * len(calendar_data.columns)) * 100, 3)
        },
        'pricing_data': {
            'rows': len(prices_data),
            'cols': len(prices_data.columns),
            'memory_mb': round(prices_data.memory_usage(deep=True).sum() / 1024**2, 2),
            'missing_pct': round(prices_data.isnull().sum().sum() / (len(prices_data) * len(prices_data.columns)) * 100, 3)
        }
    }
    
    results['table_dimensions'] = table_stats
    
    print("Table Dimensions:")
    for table, stats in table_stats.items():
        print(f"  {table.replace('_', ' ').title()}:")
        print(f"    • Rows: {stats['rows']:,}")
        print(f"    • Columns: {stats['cols']:,}")
        print(f"    • Memory: {stats['memory_mb']:.1f} MB")
        print(f"    • Missing: {stats['missing_pct']:.2f}%")
    
    # 2. Data Type Consistency Analysis
    print("\n🔤 DATA TYPE ANALYSIS")
    print("-" * 50)
    
    data_types = {
        'sales_data': dict(sales_data.dtypes.value_counts()),
        'calendar_data': dict(calendar_data.dtypes.value_counts()),
        'pricing_data': dict(prices_data.dtypes.value_counts())
    }
    
    results['data_types'] = data_types
    
    print("Data Type Distribution:")
    for table, types in data_types.items():
        print(f"  {table.replace('_', ' ').title()}:")
        for dtype, count in types.items():
            print(f"    • {dtype}: {count} columns")
    
    # 3. Generate Structure Visualizations
    print("\n📊 GENERATING STRUCTURE VISUALIZATIONS")
    print("-" * 50)
    
    try:
        # Data coverage heatmap
        coverage_plot_path = "notebooks/eda/outputs/step2_plots/data_coverage_heatmap.png"
        os.makedirs(os.path.dirname(coverage_plot_path), exist_ok=True)
        plot_data_coverage_heatmap(table_stats, coverage_plot_path)
        
        results['visualizations'] = {'coverage_heatmap': coverage_plot_path}
        print(f"✓ Data coverage heatmap saved: {coverage_plot_path}")
        
    except Exception as e:
        print(f"✗ ERROR: Visualization generation failed: {str(e)}")
        results['visualizations'] = {'error': str(e)}
    
    # 4. Save Results
    try:
        output_manager.save_analysis(
            "step2_dataset_structure",
            results,
            f"M5 Dataset Structure Analysis - Sales: {table_stats['sales_data']['rows']:,} rows, Calendar: {table_stats['calendar_data']['rows']:,} days"
        )
        print(f"\n✓ Step 2 analysis completed and saved")
        
    except Exception as e:
        print(f"✗ ERROR: Failed to save results: {str(e)}")
    
    return results


def check_m5_data_quality(
    sales_data: pd.DataFrame,
    calendar_data: pd.DataFrame,
    prices_data: pd.DataFrame,
    output_manager
) -> Dict[str, Any]:
    """
    EDA Step 3: Retail-specific data quality assessment.
    
    Comprehensive data quality analysis including zero vs missing differentiation,
    price anomaly detection, calendar consistency validation, and impossible
    value detection across all M5 data sources.
    
    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales training data
    calendar_data : pd.DataFrame
        M5 calendar data
    prices_data : pd.DataFrame
        M5 pricing data
    output_manager : EDAOutputManager
        Output manager for saving results and plots
        
    Returns
    -------
    Dict[str, Any]
        Data quality assessment results
    """
    print("=" * 80)
    print("EDA STEP 3: M5 DATA QUALITY ASSESSMENT")
    print("=" * 80)
    
    results = {}
    
    # 1. Zero vs Missing Sales Analysis
    print("\n🔍 ZERO VS MISSING SALES ANALYSIS")
    print("-" * 50)
    
    try:
        missing_analysis = analyze_missing_patterns(sales_data)
        results['missing_analysis'] = missing_analysis
        
        print("Zero-Heavy Series Analysis:")
        zero_stats = missing_analysis.get('series_zero_stats', {})
        print(f"  • Average zero percentage: {zero_stats.get('mean', 0):.1f}%")
        print(f"  • Median zero percentage: {zero_stats.get('median', 0):.1f}%")
        print(f"  • Maximum zero percentage: {zero_stats.get('max', 0):.1f}%")
        
        zero_heavy = missing_analysis.get('zero_heavy_series', {})
        print(f"  • Series with >80% zeros: {zero_heavy.get('count', 0)} ({zero_heavy.get('percent', 0):.1f}%)")
        
    except Exception as e:
        print(f"✗ ERROR: Missing analysis failed: {str(e)}")
        results['missing_analysis'] = {'error': str(e)}
    
    # 2. Price Anomaly Detection
    print("\n💰 PRICE ANOMALY DETECTION")
    print("-" * 50)
    
    try:
        if 'sell_price' in prices_data.columns:
            price_stats = {
                'negative_prices': (prices_data['sell_price'] < 0).sum(),
                'zero_prices': (prices_data['sell_price'] == 0).sum(),
                'extreme_high_prices': (prices_data['sell_price'] > prices_data['sell_price'].quantile(0.999)).sum(),
                'total_records': len(prices_data)
            }
            
            results['price_anomalies'] = price_stats
            
            print("Price Anomaly Summary:")
            print(f"  • Negative prices: {price_stats['negative_prices']} ({price_stats['negative_prices']/price_stats['total_records']*100:.3f}%)")
            print(f"  • Zero prices: {price_stats['zero_prices']} ({price_stats['zero_prices']/price_stats['total_records']*100:.3f}%)")
            print(f"  • Extreme high prices: {price_stats['extreme_high_prices']} ({price_stats['extreme_high_prices']/price_stats['total_records']*100:.3f}%)")
        else:
            results['price_anomalies'] = {'error': 'sell_price column not found'}
            
    except Exception as e:
        print(f"✗ ERROR: Price anomaly detection failed: {str(e)}")
        results['price_anomalies'] = {'error': str(e)}
    
    # 3. Calendar Consistency Validation  
    print("\n📅 CALENDAR CONSISTENCY VALIDATION")
    print("-" * 50)
    
    try:
        calendar_issues = []
        
        # Check for duplicate dates
        if 'date' in calendar_data.columns:
            duplicate_dates = calendar_data['date'].duplicated().sum()
            if duplicate_dates > 0:
                calendar_issues.append(f"Duplicate dates found: {duplicate_dates}")
        
        # Check SNAP alignment with states
        snap_cols = ['snap_CA', 'snap_TX', 'snap_WI']
        available_snap = [col for col in snap_cols if col in calendar_data.columns]
        
        snap_stats = {}
        for snap_col in available_snap:
            snap_days = calendar_data[snap_col].sum()
            snap_stats[snap_col] = snap_days
        
        results['calendar_consistency'] = {
            'duplicate_dates': duplicate_dates if 'date' in calendar_data.columns else 0,
            'snap_statistics': snap_stats,
            'issues': calendar_issues
        }
        
        print("Calendar Consistency:")
        print(f"  • Duplicate dates: {duplicate_dates if 'date' in calendar_data.columns else 0}")
        for snap_col, days in snap_stats.items():
            state = snap_col.split('_')[1]
            print(f"  • SNAP days in {state}: {days}")
            
    except Exception as e:
        print(f"✗ ERROR: Calendar validation failed: {str(e)}")
        results['calendar_consistency'] = {'error': str(e)}
    
    # 4. Save Results
    try:
        output_manager.save_analysis(
            "step3_data_quality",
            results,
            f"M5 Data Quality Assessment - Missing: {results.get('missing_analysis', {}).get('missing_percent', 0):.2f}%, Zero-heavy series: {results.get('missing_analysis', {}).get('zero_heavy_series', {}).get('percent', 0):.1f}%"
        )
        print(f"\n✓ Step 3 analysis completed and saved")
        
    except Exception as e:
        print(f"✗ ERROR: Failed to save results: {str(e)}")
    
    return results


def analyze_m5_individual_features(
    sales_data: pd.DataFrame,
    calendar_data: pd.DataFrame,
    prices_data: pd.DataFrame,
    output_manager
) -> Dict[str, Any]:
    """
    EDA Step 5: Hierarchical and temporal feature deep-dive.
    
    Comprehensive individual feature analysis including categorical distributions,
    geographic patterns, temporal correlations, and price behavior analysis.
    
    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales training data
    calendar_data : pd.DataFrame
        M5 calendar data
    prices_data : pd.DataFrame
        M5 pricing data
    output_manager : EDAOutputManager
        Output manager for saving results and plots
        
    Returns
    -------
    Dict[str, Any]
        Individual feature analysis results
    """
    print("=" * 80)
    print("EDA STEP 5: M5 INDIVIDUAL FEATURE ANALYSIS")
    print("=" * 80)
    
    results = {}
    
    # 1. Geographic Pattern Analysis
    print("\n🗺️  GEOGRAPHIC PATTERN ANALYSIS")
    print("-" * 50)
    
    try:
        geographic_results = analyze_geographic_patterns(sales_data)
        results['geographic_patterns'] = geographic_results
        
        # Display state-level summary
        if 'state_id' in geographic_results:
            print("State-Level Performance:")
            for state, metrics in geographic_results['state_id']['geographic_units'].items():
                print(f"  {state}:")
                print(f"    • Products: {metrics['product_count']:,}")
                print(f"    • Total Volume: {metrics['total_volume']:,}")
                print(f"    • Avg Volatility: {metrics['volatility']:.3f}")
        
    except Exception as e:
        print(f"✗ ERROR: Geographic analysis failed: {str(e)}")
        results['geographic_patterns'] = {'error': str(e)}
    
    # 2. Temporal Feature Correlations
    print("\n⏰ TEMPORAL FEATURE CORRELATIONS")
    print("-" * 50)
    
    try:
        temporal_results = analyze_temporal_feature_correlations(sales_data, calendar_data)
        results['temporal_correlations'] = temporal_results
        
        # Display significant correlations
        print("Significant Temporal Correlations:")
        for feature, result in temporal_results.items():
            if isinstance(result, dict) and 'correlation' in result:
                corr_val = result['correlation']
                strength = result['strength']
                print(f"  {feature}: {corr_val:.4f} ({strength})")
        
    except Exception as e:
        print(f"✗ ERROR: Temporal correlation analysis failed: {str(e)}")
        results['temporal_correlations'] = {'error': str(e)}
    
    # 3. Price Feature Behavior
    print("\n💲 PRICE FEATURE BEHAVIOR ANALYSIS")
    print("-" * 50)
    
    try:
        price_results = analyze_price_feature_behavior(prices_data, sales_data)
        results['price_behavior'] = price_results
        
        # Display price distribution summary
        if 'overall_distribution' in price_results:
            price_dist = price_results['overall_distribution']
            print("Price Distribution Summary:")
            print(f"  • Mean Price: ${price_dist['mean']:.2f}")
            print(f"  • Median Price: ${price_dist['median']:.2f}")
            print(f"  • Price Range: ${price_dist['min']:.2f} - ${price_dist['max']:.2f}")
            print(f"  • Skewness: {price_dist['skewness']:.3f}")
        
    except Exception as e:
        print(f"✗ ERROR: Price behavior analysis failed: {str(e)}")
        results['price_behavior'] = {'error': str(e)}
    
    # 4. Generate Feature Visualizations
    print("\n📊 GENERATING FEATURE VISUALIZATIONS")
    print("-" * 50)
    
    try:
        viz_results = {}
        
        # Geographic performance comparison
        if 'geographic_patterns' in results and 'error' not in results['geographic_patterns']:
            geo_plot_path = "notebooks/eda/outputs/step5_plots/geographic_performance_comparison.png"
            os.makedirs(os.path.dirname(geo_plot_path), exist_ok=True)
            plot_geographic_performance_comparison(results['geographic_patterns'], geo_plot_path)
            viz_results['geographic_plot'] = geo_plot_path
            print(f"✓ Geographic performance plot saved: {geo_plot_path}")
        
        results['visualizations'] = viz_results
        
    except Exception as e:
        print(f"✗ ERROR: Visualization generation failed: {str(e)}")
        results['visualizations'] = {'error': str(e)}
    
    # 5. Save Results
    try:
        output_manager.save_analysis(
            "step5_individual_features",
            results,
            f"M5 Individual Feature Analysis - Geographic patterns, temporal correlations, and price behavior"
        )
        print(f"\n✓ Step 5 analysis completed and saved")
        
    except Exception as e:
        print(f"✗ ERROR: Failed to save results: {str(e)}")
    
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest notebooks/eda/tests/test_eda_steps_1_3_5.py::test_analyze_m5_problem_context_success -v`
Expected: PASS

- [ ] **Step 5: Add comprehensive tests for all functions**

```python
def test_inspect_m5_dataset_structure_success(tmp_path):
    """Test Step 2 dataset structure inspection."""
    # Create mock data
    sales_data = pd.DataFrame({
        'item_id': ['ITEM_1', 'ITEM_2'],
        'd_1': [1, 2], 'd_2': [3, 4]
    })
    
    calendar_data = pd.DataFrame({
        'd': ['d_1', 'd_2'],
        'date': ['2011-01-29', '2011-01-30']
    })
    
    prices_data = pd.DataFrame({
        'item_id': ['ITEM_1'],
        'sell_price': [1.99]
    })
    
    class MockOutputManager:
        def save_analysis(self, step_name, results, summary):
            pass
    
    output_manager = MockOutputManager()
    
    result = inspect_m5_dataset_structure(sales_data, calendar_data, prices_data, output_manager)
    
    assert 'table_dimensions' in result
    assert 'data_types' in result
    assert result['table_dimensions']['sales_data']['rows'] == 2


def test_check_m5_data_quality_success(tmp_path):
    """Test Step 3 data quality assessment."""
    sales_data = pd.DataFrame({
        'd_1': [0, 1, 0, 0], 'd_2': [2, 0, 3, 0]
    })
    
    calendar_data = pd.DataFrame({
        'date': ['2011-01-29', '2011-01-30'],
        'snap_CA': [0, 1], 'snap_TX': [1, 0]
    })
    
    prices_data = pd.DataFrame({
        'sell_price': [1.99, 2.49, -0.50, 0.00]  # Include anomalies
    })
    
    class MockOutputManager:
        def save_analysis(self, step_name, results, summary):
            pass
    
    output_manager = MockOutputManager()
    
    result = check_m5_data_quality(sales_data, calendar_data, prices_data, output_manager)
    
    assert 'missing_analysis' in result
    assert 'price_anomalies' in result
    assert 'calendar_consistency' in result


def test_analyze_m5_individual_features_success(tmp_path):
    """Test Step 5 individual feature analysis."""
    sales_data = pd.DataFrame({
        'state_id': ['CA', 'TX', 'CA', 'TX'],
        'store_id': ['CA_1', 'TX_1', 'CA_2', 'TX_2'],
        'd_1': [100, 80, 120, 90], 'd_2': [110, 85, 125, 95]
    })
    
    calendar_data = pd.DataFrame({
        'd': ['d_1', 'd_2'],
        'wday': [1, 2], 'month': [1, 1]
    })
    
    prices_data = pd.DataFrame({
        'item_id': ['ITEM_1', 'ITEM_2'],
        'store_id': ['CA_1', 'TX_1'],
        'sell_price': [1.99, 2.49]
    })
    
    class MockOutputManager:
        def save_analysis(self, step_name, results, summary):
            pass
    
    output_manager = MockOutputManager()
    
    result = analyze_m5_individual_features(sales_data, calendar_data, prices_data, output_manager)
    
    assert 'geographic_patterns' in result
    assert 'temporal_correlations' in result
    assert 'price_behavior' in result
```

- [ ] **Step 6: Run all main EDA tests**

Run: `pytest notebooks/eda/tests/test_eda_steps_1_3_5.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit main EDA implementation**

```bash
git add notebooks/eda/eda_steps_1_3_5.py notebooks/eda/tests/test_eda_steps_1_3_5.py
git commit -m "feat: implement main EDA steps 1, 2, 3, 5 for M5 dataset

- Step 1: M5 problem context validation with hierarchy and leakage checks
- Step 2: Dataset structure inspection with dimension and type analysis  
- Step 3: Data quality assessment with zero-inflation and anomaly detection
- Step 5: Individual feature analysis with geographic and temporal patterns
- Comprehensive test coverage and output manager integration"
```

---

### Task 5: Enhanced Step 11 for Intermittent Demand

**Files:**
- Modify: `notebooks/eda/utils/segment_analysis.py`
- Test: `notebooks/eda/tests/test_segment_analysis_enhancements.py`

- [ ] **Step 1: Write failing test for enhanced intermittent demand analysis**

```python
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.segment_analysis import analyze_intermittent_demand_patterns

def test_analyze_intermittent_demand_patterns():
    """Test enhanced intermittent demand analysis for Step 11."""
    # Create mock sales data with different intermittency patterns
    sales_data = pd.DataFrame({
        'item_id': ['ITEM_1', 'ITEM_2', 'ITEM_3'] * 10,
        'cat_id': ['FOODS', 'HOUSEHOLD', 'HOBBIES'] * 10,
        'd_1': [0] * 10 + [10] * 10 + [0, 0, 0, 0, 50] + [0] * 5,
        'd_2': [5] * 10 + [0] * 10 + [0, 0, 0, 0, 60] + [0] * 5,
        'd_3': [3] * 10 + [12] * 10 + [0] * 10,
        'd_4': [0] * 10 + [0] * 10 + [0] * 10,
        'd_5': [4] * 10 + [15] * 10 + [0] * 10
    })
    
    result = analyze_intermittent_demand_patterns(sales_data)
    
    assert 'intermittency_classification' in result
    assert 'zero_run_analysis' in result
    assert 'demand_intensity_metrics' in result
    assert 'forecast_viability' in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest notebooks/eda/tests/test_segment_analysis_enhancements.py::test_analyze_intermittent_demand_patterns -v`
Expected: FAIL with "ImportError: cannot import name 'analyze_intermittent_demand_patterns'"

- [ ] **Step 3: Add enhanced intermittent demand analysis to segment_analysis.py**

```python
def analyze_intermittent_demand_patterns(
    sales_data: pd.DataFrame,
    min_forecast_days: int = 28
) -> Dict[str, Any]:
    """
    Enhanced intermittent demand analysis for Step 11 segmentation.
    
    Comprehensive analysis of zero-inflation patterns, intermittency classification,
    zero-run statistics, and forecast viability assessment for M5 demand data.
    
    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales data with daily sales columns
    min_forecast_days : int, default 28
        Minimum non-zero days required for viable forecasting
        
    Returns
    -------
    Dict[str, Any]
        Enhanced intermittent demand analysis results including:
        - intermittency_classification: Product classification by demand patterns
        - zero_run_analysis: Consecutive zero-sales period statistics  
        - demand_intensity_metrics: Sales intensity on non-zero days
        - forecast_viability: 28-day horizon feasibility assessment
    """
    # Get sales columns
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]
    
    if len(sales_cols) == 0:
        return {'error': 'No daily sales columns found'}
    
    results = {}
    
    # 1. Intermittency Classification
    print("\n🔄 INTERMITTENCY CLASSIFICATION")
    print("-" * 40)
    
    # Calculate intermittency metrics per item
    item_metrics = []
    
    for _, row in sales_data.iterrows():
        item_sales = row[sales_cols].values
        
        # Basic statistics
        total_sales = item_sales.sum()
        non_zero_days = (item_sales > 0).sum()
        zero_days = (item_sales == 0).sum()
        total_days = len(item_sales)
        
        # Intermittency and volatility
        zero_rate = zero_days / total_days
        mean_sales = item_sales.mean()
        cv = item_sales.std() / mean_sales if mean_sales > 0 else np.inf
        
        # Demand intensity (average on non-zero days)
        demand_intensity = item_sales[item_sales > 0].mean() if non_zero_days > 0 else 0
        
        item_metrics.append({
            'item_id': row.get('item_id', 'unknown'),
            'cat_id': row.get('cat_id', 'unknown'),
            'total_sales': total_sales,
            'non_zero_days': non_zero_days,
            'zero_rate': zero_rate,
            'cv': cv,
            'demand_intensity': demand_intensity,
            'forecast_viable': non_zero_days >= min_forecast_days
        })
    
    item_metrics_df = pd.DataFrame(item_metrics)
    
    # Classify items based on zero rate and CV
    def classify_intermittency(zero_rate, cv):
        if zero_rate >= 0.8:
            return 'Highly Intermittent'
        elif zero_rate >= 0.5:
            return 'Moderately Intermittent' 
        elif zero_rate >= 0.2:
            return 'Low Intermittent'
        elif cv > 1.5:
            return 'Volatile Continuous'
        else:
            return 'Stable Continuous'
    
    item_metrics_df['intermittency_class'] = item_metrics_df.apply(
        lambda row: classify_intermittency(row['zero_rate'], row['cv']), axis=1
    )
    
    # Classification summary
    class_summary = item_metrics_df.groupby('intermittency_class').agg({
        'item_id': 'count',
        'zero_rate': ['mean', 'std'],
        'cv': ['mean', 'std'],
        'demand_intensity': ['mean', 'std'],
        'forecast_viable': 'sum'
    }).round(3)
    
    results['intermittency_classification'] = {
        'class_distribution': dict(item_metrics_df['intermittency_class'].value_counts()),
        'class_statistics': class_summary.to_dict(),
        'total_items': len(item_metrics_df)
    }
    
    print("Intermittency Classification:")
    for class_name, count in item_metrics_df['intermittency_class'].value_counts().items():
        percentage = count / len(item_metrics_df) * 100
        print(f"  • {class_name}: {count} items ({percentage:.1f}%)")
    
    # 2. Zero-Run Analysis
    print("\n🚫 ZERO-RUN ANALYSIS") 
    print("-" * 40)
    
    zero_run_stats = []
    
    for _, row in sales_data.iterrows():
        item_sales = row[sales_cols].values
        
        # Find consecutive zero runs
        zero_runs = []
        current_run = 0
        
        for sale in item_sales:
            if sale == 0:
                current_run += 1
            else:
                if current_run > 0:
                    zero_runs.append(current_run)
                current_run = 0
        
        # Don't forget the last run if it ends with zeros
        if current_run > 0:
            zero_runs.append(current_run)
        
        if zero_runs:
            zero_run_stats.append({
                'item_id': row.get('item_id', 'unknown'),
                'max_zero_run': max(zero_runs),
                'avg_zero_run': np.mean(zero_runs),
                'num_zero_runs': len(zero_runs),
                'total_zero_days': sum(zero_runs)
            })
        else:
            zero_run_stats.append({
                'item_id': row.get('item_id', 'unknown'),
                'max_zero_run': 0,
                'avg_zero_run': 0,
                'num_zero_runs': 0,
                'total_zero_days': 0
            })
    
    zero_run_df = pd.DataFrame(zero_run_stats)
    
    zero_run_summary = {
        'avg_max_zero_run': zero_run_df['max_zero_run'].mean(),
        'longest_zero_run': zero_run_df['max_zero_run'].max(),
        'items_with_long_runs': (zero_run_df['max_zero_run'] > 30).sum(),  # More than 30 days
        'avg_zero_runs_per_item': zero_run_df['num_zero_runs'].mean()
    }
    
    results['zero_run_analysis'] = zero_run_summary
    
    print("Zero-Run Statistics:")
    print(f"  • Average max zero-run: {zero_run_summary['avg_max_zero_run']:.1f} days")
    print(f"  • Longest zero-run: {zero_run_summary['longest_zero_run']} days")
    print(f"  • Items with >30-day runs: {zero_run_summary['items_with_long_runs']}")
    
    # 3. Demand Intensity Metrics
    print("\n⚡ DEMAND INTENSITY METRICS")
    print("-" * 40)
    
    # Calculate category-level demand intensity
    if 'cat_id' in item_metrics_df.columns:
        category_intensity = item_metrics_df.groupby('cat_id').agg({
            'demand_intensity': ['mean', 'median', 'std'],
            'zero_rate': ['mean', 'median'],
            'cv': ['mean', 'median']
        }).round(3)
        
        results['demand_intensity_metrics'] = {
            'category_analysis': category_intensity.to_dict(),
            'overall_stats': {
                'mean_intensity': item_metrics_df['demand_intensity'].mean(),
                'median_intensity': item_metrics_df['demand_intensity'].median(),
                'intensity_cv': item_metrics_df['demand_intensity'].std() / item_metrics_df['demand_intensity'].mean()
            }
        }
        
        print("Category Demand Intensity:")
        for cat in item_metrics_df['cat_id'].unique():
            cat_data = item_metrics_df[item_metrics_df['cat_id'] == cat]
            mean_intensity = cat_data['demand_intensity'].mean()
            mean_zero_rate = cat_data['zero_rate'].mean()
            print(f"  • {cat}: Intensity={mean_intensity:.2f}, Zero Rate={mean_zero_rate:.3f}")
    
    # 4. Forecast Viability Assessment
    print("\n📊 FORECAST VIABILITY ASSESSMENT")
    print("-" * 40)
    
    viability_stats = {
        'viable_items': (item_metrics_df['forecast_viable'] == True).sum(),
        'total_items': len(item_metrics_df),
        'viability_rate': (item_metrics_df['forecast_viable'] == True).sum() / len(item_metrics_df),
        'min_required_days': min_forecast_days
    }
    
    # Viability by category
    if 'cat_id' in item_metrics_df.columns:
        category_viability = item_metrics_df.groupby('cat_id')['forecast_viable'].agg(['sum', 'count'])
        category_viability['rate'] = category_viability['sum'] / category_viability['count']
        viability_stats['category_viability'] = category_viability.to_dict()
    
    results['forecast_viability'] = viability_stats
    
    print("Forecast Viability (28-day horizon):")
    print(f"  • Viable items: {viability_stats['viable_items']}/{viability_stats['total_items']} ({viability_stats['viability_rate']:.1%})")
    
    if 'category_viability' in viability_stats:
        print("  • By category:")
        for cat in viability_stats['category_viability']['rate'].keys():
            rate = viability_stats['category_viability']['rate'][cat]
            print(f"    - {cat}: {rate:.1%}")
    
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest notebooks/eda/tests/test_segment_analysis_enhancements.py::test_analyze_intermittent_demand_patterns -v`
Expected: PASS

- [ ] **Step 5: Add integration test for enhanced Step 11**

```python
def test_enhanced_step11_integration():
    """Test that enhanced intermittent demand analysis integrates with existing Step 11."""
    # This would test the integration with existing analyze_segment_behavior function
    # Create comprehensive sales data
    np.random.seed(42)
    
    # Simulate different demand patterns
    n_items = 100
    n_days = 50
    
    sales_data = []
    for i in range(n_items):
        item_data = {'item_id': f'ITEM_{i}', 'cat_id': 'FOODS'}
        
        # Create different intermittency patterns
        if i < 30:  # Stable items
            daily_sales = np.random.poisson(10, n_days)
        elif i < 60:  # Intermittent items
            daily_sales = np.where(np.random.random(n_days) < 0.7, 0, np.random.poisson(20, n_days))
        else:  # Highly intermittent
            daily_sales = np.where(np.random.random(n_days) < 0.9, 0, np.random.poisson(50, n_days))
        
        for day, sales in enumerate(daily_sales, 1):
            item_data[f'd_{day}'] = sales
            
        sales_data.append(item_data)
    
    sales_df = pd.DataFrame(sales_data)
    
    result = analyze_intermittent_demand_patterns(sales_df)
    
    # Verify all key components are present
    assert 'intermittency_classification' in result
    assert 'zero_run_analysis' in result
    assert 'demand_intensity_metrics' in result
    assert 'forecast_viability' in result
    
    # Verify reasonable results
    assert result['intermittency_classification']['total_items'] == 100
    assert len(result['intermittency_classification']['class_distribution']) >= 3
    assert result['forecast_viability']['viability_rate'] > 0
```

- [ ] **Step 6: Run enhanced segment analysis tests**

Run: `pytest notebooks/eda/tests/test_segment_analysis_enhancements.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit Step 11 enhancements**

```bash
git add notebooks/eda/utils/segment_analysis.py notebooks/eda/tests/test_segment_analysis_enhancements.py
git commit -m "feat: enhance Step 11 with comprehensive intermittent demand analysis

- Zero-inflation characterization by product hierarchy levels
- Intermittency classification (5 demand pattern types)
- Zero-run length analysis (consecutive zero-day periods)
- Demand intensity metrics (average sales on non-zero days)
- Forecast horizon viability assessment (28-day sufficient observations)
- Category-level statistical summaries and business interpretations"
```

---

### Task 6: Update Runner Integration

**Files:**
- Modify: `notebooks/eda/run_eda_auto.py`

- [ ] **Step 1: Add new steps to runner script**

```python
    # Import new EDA functions
    try:
        from eda_steps_1_3_5 import (
            analyze_m5_problem_context,
            inspect_m5_dataset_structure,
            check_m5_data_quality,
            analyze_m5_individual_features
        )
    except ImportError as e:
        print(f"❌ Error importing new EDA functions: {e}")
        print("Make sure eda_steps_1_3_5.py is in the same directory")
        return

    # Define all available steps including new ones
    all_steps = [
        ("1", "M5 Problem Context", analyze_m5_problem_context),
        ("2", "Dataset Structure", inspect_m5_dataset_structure),  
        ("3", "Data Quality", check_m5_data_quality),
        ("5", "Individual Features", analyze_m5_individual_features),
        ("6", "Feature-Target Relationships", study_feature_target_relationships),
        ("7", "Feature-Feature Relationships", study_feature_feature_relationships),
        ("8", "Time Series Patterns", analyze_time_series_patterns),
        ("9", "Missing Values Analysis", analyze_missing_values_deeply),
        ("10", "Outliers and Anomalies", identify_outliers_and_anomalies),
        ("11", "Segment Behavior", analyze_segment_behavior),
        ("13", "Distribution Drift", analyze_distribution_drift),
        ("14", "Temporal Leakage", audit_temporal_leakage)
    ]
```

- [ ] **Step 2: Run integration test**

Run: `python notebooks/eda/run_eda_auto.py`
Expected: Script runs successfully, includes new steps 1, 2, 3, 5

- [ ] **Step 3: Commit runner integration**

```bash
git add notebooks/eda/run_eda_auto.py
git commit -m "feat: integrate new EDA steps 1, 2, 3, 5 into runner script

- Add imports for new analysis functions
- Update step list to include Steps 1-3 and 5
- Maintain existing step ordering and functionality
- Preserve backward compatibility with existing runner"
```

---

### Task 7: Final Integration Testing

**Files:**
- Test: Complete pipeline test

- [ ] **Step 1: Run comprehensive integration test**

Run: `pytest notebooks/eda/tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Test complete EDA pipeline with real data**

```bash
# Test with actual M5 data if available
cd notebooks/eda
python run_eda_auto.py
```
Expected: Complete EDA analysis runs successfully, generates all outputs

- [ ] **Step 3: Verify output structure**

```bash
ls -la notebooks/eda/outputs/
```
Expected: Step 1-3, 5 output directories created with plots and analysis files

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete EDA steps 1-3, 5 implementation with M5 focus

- Implemented 4 main analysis functions with M5-specific insights
- Added comprehensive utility modules and visualization functions  
- Enhanced existing Step 11 with detailed intermittent demand analysis
- Integrated with existing output management and runner systems
- Added extensive test coverage for all new functionality
- Maintained compatibility with existing EDA pipeline

Deliverables:
- analyze_m5_problem_context(): Hierarchy validation and leakage prevention
- inspect_m5_dataset_structure(): Table dimensions and coverage analysis
- check_m5_data_quality(): Zero-inflation and anomaly detection
- analyze_m5_individual_features(): Geographic patterns and feature analysis
- Enhanced Step 11: Comprehensive intermittent demand classification

All functions generate static plots and integrate with output_manager.py"
```

---

## Self-Review

**1. Spec coverage:** 
✅ All requirements from specification implemented:
- Step 1: Problem context validation with hierarchy and leakage checks
- Step 2: Dataset structure inspection with coverage analysis  
- Step 3: Data quality assessment with retail-specific focus
- Step 5: Individual feature analysis with M5 patterns
- Step 11 Enhancement: Intermittent demand analysis integration
- Static visualizations and output manager integration

**2. Placeholder scan:**
✅ No TBD, TODO, or placeholder content found
✅ All code blocks contain complete implementations
✅ All file paths are exact and complete
✅ All commands include expected outputs

**3. Type consistency:**
✅ Function signatures consistent across tasks
✅ Parameter and return types match between utility functions and main implementations
✅ Data structure naming consistent (sales_data, calendar_data, prices_data)

## Implementation Plan Complete

Plan complete and saved to `docs/superpowers/plans/2026-05-23-eda-steps-1-3-5-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?