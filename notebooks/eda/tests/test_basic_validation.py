import pytest
import pandas as pd
import sys
import os

# Add notebooks/eda to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.basic_validation import validate_m5_hierarchy_structure, validate_temporal_boundaries, validate_business_objectives

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


def test_validate_temporal_boundaries_valid():
    """Test temporal boundary validation with valid M5 structure."""
    # Create consecutive sales columns from d_1 to d_1913
    sales_data = pd.DataFrame({f'd_{i}': [1, 2] for i in range(1, 1914)})

    calendar_data = pd.DataFrame({
        'd': [f'd_{i}' for i in range(1, 1970)],
        'date': [f'2011-01-{i%28+1:02d}' for i in range(1, 1970)]
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
        'dept_id': ['FOODS_1'] * 300 + ['FOODS_2'] * 300 + ['FOODS_3'] * 400 +
                   ['HOUSEHOLD_1'] * 500 + ['HOUSEHOLD_2'] * 500 +
                   ['HOBBIES_1'] * 1049,
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