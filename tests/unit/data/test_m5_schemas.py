import pandas as pd
import pytest
from datetime import datetime

from demand_forecast_intelligence.data.schemas.m5_schemas import (
    SalesRecord,
    CalendarRecord,
    PriceRecord,
    validate_sales_dataframe,
    validate_calendar_dataframe,
    validate_prices_dataframe
)


def test_sales_record_validation():
    """Test SalesRecord pydantic model validation."""
    # Valid record
    record = SalesRecord(
        id="FOODS_1_001_CA_1_validation",
        item_id="FOODS_1_001",
        dept_id="FOODS_1",
        cat_id="FOODS",
        store_id="CA_1",
        state_id="CA"
    )
    assert record.id == "FOODS_1_001_CA_1_validation"
    assert record.item_id == "FOODS_1_001"

    # Invalid state_id
    with pytest.raises(ValueError):
        SalesRecord(
            id="FOODS_1_001_XX_1_validation",
            item_id="FOODS_1_001",
            dept_id="FOODS_1",
            cat_id="FOODS",
            store_id="XX_1",
            state_id="XX"  # Invalid state
        )


def test_calendar_record_validation():
    """Test CalendarRecord pydantic model validation."""
    record = CalendarRecord(
        date=datetime(2011, 1, 29).date(),
        wm_yr_wk=11101,
        weekday="Saturday",
        wday=1,
        month=1,
        year=2011,
        d="d_1"
    )
    assert record.weekday == "Saturday"
    assert record.wday == 1

    # Invalid weekday
    with pytest.raises(ValueError):
        CalendarRecord(
            date=datetime(2011, 1, 29).date(),
            wm_yr_wk=11101,
            weekday="InvalidDay",  # Not a real weekday
            wday=1,
            month=1,
            year=2011,
            d="d_1"
        )


def test_validate_sales_dataframe():
    """Test sales DataFrame validation function."""
    # Valid DataFrame
    df = pd.DataFrame({
        'id': ['FOODS_1_001_CA_1_validation'],
        'item_id': ['FOODS_1_001'],
        'dept_id': ['FOODS_1'],
        'cat_id': ['FOODS'],
        'store_id': ['CA_1'],
        'state_id': ['CA'],
        'd_1': [5],
        'd_2': [3]
    })

    # Should not raise
    validate_sales_dataframe(df)

    # Missing required column
    df_invalid = df.drop('item_id', axis=1)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_sales_dataframe(df_invalid)


def test_validate_calendar_dataframe():
    """Test calendar DataFrame validation function."""
    df = pd.DataFrame({
        'date': ['2011-01-29', '2011-01-30'],
        'wm_yr_wk': [11101, 11101],
        'weekday': ['Saturday', 'Sunday'],
        'wday': [1, 2],
        'month': [1, 1],
        'year': [2011, 2011],
        'd': ['d_1', 'd_2']
    })
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Should not raise
    validate_calendar_dataframe(df)

    # Test duplicate dates
    df_dup = df.copy()
    df_dup.loc[1, 'date'] = df_dup.loc[0, 'date']  # Duplicate first date

    with pytest.raises(ValueError, match="duplicate dates"):
        validate_calendar_dataframe(df_dup)


def test_price_record_validation():
    """Test PriceRecord validation."""
    record = PriceRecord(
        store_id="CA_1",
        item_id="FOODS_1_001",
        wm_yr_wk=11101,
        sell_price=3.97
    )
    assert record.sell_price == 3.97

    # Test negative price
    with pytest.raises(ValueError):
        PriceRecord(
            store_id="CA_1",
            item_id="FOODS_1_001",
            wm_yr_wk=11101,
            sell_price=-1.0  # Invalid negative price
        )


def test_validate_prices_dataframe():
    """Test prices DataFrame validation function."""
    # Valid DataFrame
    df = pd.DataFrame({
        'store_id': ['CA_1', 'TX_1'],
        'item_id': ['FOODS_1_001', 'FOODS_1_002'],
        'wm_yr_wk': [11101, 11101],
        'sell_price': [3.97, 2.50]
    })

    # Should not raise
    validate_prices_dataframe(df)

    # Missing required column
    df_missing = df.drop('sell_price', axis=1)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_prices_dataframe(df_missing)

    # Test negative prices
    df_negative = df.copy()
    df_negative.loc[0, 'sell_price'] = -1.0
    with pytest.raises(ValueError, match="All prices must be positive"):
        validate_prices_dataframe(df_negative)

    # Test null prices
    df_null = df.copy()
    df_null.loc[0, 'sell_price'] = None
    with pytest.raises(ValueError, match="Prices cannot be null"):
        validate_prices_dataframe(df_null)