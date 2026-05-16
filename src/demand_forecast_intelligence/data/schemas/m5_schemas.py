"""Pydantic schemas for M5 dataset validation."""

from datetime import date
from typing import Optional, List, Set
import pandas as pd
from pydantic import BaseModel, Field, validator


class SalesRecord(BaseModel):
    """Schema for sales_train_validation.csv records."""

    id: str = Field(..., description="Unique identifier for item-store combination")
    item_id: str = Field(..., description="Product identifier")
    dept_id: str = Field(..., description="Department identifier")
    cat_id: str = Field(..., description="Category identifier")
    store_id: str = Field(..., description="Store identifier")
    state_id: str = Field(..., description="State identifier")

    @validator('state_id')
    def validate_state_id(cls, v):
        valid_states = {'CA', 'TX', 'WI'}
        if v not in valid_states:
            raise ValueError(f'state_id must be one of {valid_states}')
        return v

    @validator('cat_id')
    def validate_category(cls, v):
        valid_categories = {'HOBBIES', 'HOUSEHOLD', 'FOODS'}
        if v not in valid_categories:
            raise ValueError(f'cat_id must be one of {valid_categories}')
        return v

    @validator('store_id')
    def validate_store_format(cls, v):
        # Store format: {STATE}_{NUMBER}
        if not v or len(v.split('_')) != 2:
            raise ValueError('store_id must follow format {STATE}_{NUMBER}')
        return v


class CalendarRecord(BaseModel):
    """Schema for calendar.csv records."""

    date: date
    wm_yr_wk: int = Field(..., description="Walmart year-week identifier")
    weekday: str = Field(..., description="Day of week name")
    wday: int = Field(..., ge=1, le=7, description="Numeric day of week")
    month: int = Field(..., ge=1, le=12, description="Month number")
    year: int = Field(..., ge=2011, le=2016, description="Year")
    d: str = Field(..., description="Day identifier matching sales columns")
    event_name_1: Optional[str] = None
    event_type_1: Optional[str] = None
    event_name_2: Optional[str] = None
    event_type_2: Optional[str] = None
    snap_CA: Optional[int] = Field(None, ge=0, le=1)
    snap_TX: Optional[int] = Field(None, ge=0, le=1)
    snap_WI: Optional[int] = Field(None, ge=0, le=1)

    @validator('weekday')
    def validate_weekday(cls, v):
        valid_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
        if v not in valid_days:
            raise ValueError(f'weekday must be one of {valid_days}')
        return v

    @validator('d')
    def validate_day_identifier(cls, v):
        if not v.startswith('d_') or not v[2:].isdigit():
            raise ValueError('d must follow format d_{number}')
        return v


class PriceRecord(BaseModel):
    """Schema for sell_prices.csv records."""

    store_id: str = Field(..., description="Store identifier")
    item_id: str = Field(..., description="Item identifier")
    wm_yr_wk: int = Field(..., description="Walmart year-week identifier")
    sell_price: float = Field(..., gt=0, description="Unit selling price")

    @validator('store_id')
    def validate_store_format(cls, v):
        if not v or len(v.split('_')) != 2:
            raise ValueError('store_id must follow format {STATE}_{NUMBER}')
        return v


def validate_sales_dataframe(df: pd.DataFrame) -> None:
    """Validate sales DataFrame has required structure.

    Args:
        df: Sales DataFrame to validate

    Raises:
        ValueError: If DataFrame structure is invalid
    """
    required_columns = {'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check for sales data columns (d_1, d_2, etc.)
    sales_columns = [col for col in df.columns if col.startswith('d_')]
    if not sales_columns:
        raise ValueError("No sales data columns found (expected d_1, d_2, etc.)")

    # Validate data types - check if id column contains strings
    if not pd.api.types.is_string_dtype(df['id']) and not pd.api.types.is_object_dtype(df['id']):
        raise ValueError("Column 'id' must be string type")

    # Check for negative sales values
    for col in sales_columns:
        if df[col].min() < 0:
            raise ValueError(f"Sales column '{col}' contains negative values")


def validate_calendar_dataframe(df: pd.DataFrame) -> None:
    """Validate calendar DataFrame has required structure.

    Args:
        df: Calendar DataFrame to validate

    Raises:
        ValueError: If DataFrame structure is invalid
    """
    required_columns = {'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'd'}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Validate date range
    if df['year'].min() < 2011 or df['year'].max() > 2016:
        raise ValueError("Calendar data must be within 2011-2016 range")

    # Check for duplicate dates
    if df['date'].duplicated().any():
        raise ValueError("Calendar contains duplicate dates")


def validate_prices_dataframe(df: pd.DataFrame) -> None:
    """Validate prices DataFrame has required structure.

    Args:
        df: Prices DataFrame to validate

    Raises:
        ValueError: If DataFrame structure is invalid
    """
    required_columns = {'store_id', 'item_id', 'wm_yr_wk', 'sell_price'}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Validate price values
    if df['sell_price'].min() <= 0:
        raise ValueError("All prices must be positive")

    if df['sell_price'].isna().any():
        raise ValueError("Prices cannot be null")