"""
Basic validation utilities for M5 dataset EDA Steps 1 & 2.

Provides functions for validating M5 dataset structure, hierarchy consistency,
temporal boundaries, and business objective alignment.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def validate_m5_hierarchy_structure(sales_data: pd.DataFrame, strict_counts: bool = False) -> Dict[str, Any]:
    """
    Validate M5 hierarchical product structure consistency.

    Verifies that category → department → item hierarchy is consistent
    and optionally matches expected M5 structure (3 categories, 7 departments, 3049 items).

    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales data with hierarchy columns
    strict_counts : bool, default False
        Whether to enforce exact M5 counts for validation

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

    # Check expected M5 structure only if strict_counts is True
    structure_issues = []
    if strict_counts:
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