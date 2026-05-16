"""Pydantic schemas for M5 dataset validation."""

from datetime import date
from typing import Optional, List, Set
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Constants for M5 dataset business rules
VALID_STATES = {'CA', 'TX', 'WI'}
VALID_CATEGORIES = {'HOBBIES', 'HOUSEHOLD', 'FOODS'}
VALID_WEEKDAYS = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
M5_YEAR_RANGE = (2011, 2016)


def validate_store_id_format(store_id: str) -> str:
    """Validate store ID follows format {STATE}_{NUMBER}.

    Args:
        store_id: Store identifier to validate

    Returns:
        str: Validated store ID

    Raises:
        ValueError: If store ID format is invalid
    """
    if not store_id or len(store_id.split('_')) != 2:
        raise ValueError('store_id must follow format {STATE}_{NUMBER}')
    return store_id


class SalesRecord(BaseModel):
    """Schema for sales_train_validation.csv records."""

    id: str = Field(..., description="Unique identifier for item-store combination")
    item_id: str = Field(..., description="Product identifier")
    dept_id: str = Field(..., description="Department identifier")
    cat_id: str = Field(..., description="Category identifier")
    store_id: str = Field(..., description="Store identifier")
    state_id: str = Field(..., description="State identifier")

    @field_validator('state_id')
    @classmethod
    def validate_state_id(cls, v):
        if v not in VALID_STATES:
            raise ValueError(f'state_id must be one of {VALID_STATES}')
        return v

    @field_validator('cat_id')
    @classmethod
    def validate_category(cls, v):
        if v not in VALID_CATEGORIES:
            raise ValueError(f'cat_id must be one of {VALID_CATEGORIES}')
        return v

    @field_validator('store_id')
    @classmethod
    def validate_store_format(cls, v):
        return validate_store_id_format(v)


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

    @field_validator('weekday')
    @classmethod
    def validate_weekday(cls, v):
        if v not in VALID_WEEKDAYS:
            raise ValueError(f'weekday must be one of {VALID_WEEKDAYS}')
        return v

    @field_validator('d')
    @classmethod
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

    @field_validator('store_id')
    @classmethod
    def validate_store_format(cls, v):
        return validate_store_id_format(v)


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
    min_year, max_year = M5_YEAR_RANGE
    if df['year'].min() < min_year or df['year'].max() > max_year:
        raise ValueError(f"Calendar data must be within {min_year}-{max_year} range")

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