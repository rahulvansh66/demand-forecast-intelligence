# Walmart M5 Dataset Schema Documentation

## Overview

The Walmart M5 dataset is a comprehensive sales and retail dataset containing 5 interconnected CSV files covering daily unit sales, calendar information, pricing data, and evaluation/submission formats for a retail demand forecasting task. The dataset spans from 2011-01-29 to 2016-06-19.

## Dataset Files Summary

| File | Records | Primary Purpose | Time Coverage |
|------|---------|-----------------|----------------|
| `sales_train_validation.csv` | 30,490 | Historical sales data | 1,913 days (2011-2016) |
| `sales_train_evaluation.csv` | 30,490 | Extended sales data | 1,941 days (2011-2016) |
| `calendar.csv` | 1,970 | Date and event information | Full dataset coverage |
| `sell_prices.csv` | 6,841,121 | Weekly pricing data | Variable by store/item |
| `sample_submission.csv` | 60,981 | Submission format | Forecast columns (28 days) |

---

## 1. Sales Training Data: `sales_train_validation.csv`

### Dimensions
- **Rows**: 30,490 (one row per product-store combination)
- **Columns**: 1,919 (6 metadata + 1,913 sales columns)
- **Date Range**: 2011-01-29 to 2016-06-19 (1,913 days)

### Column Structure

#### Metadata Columns (6 columns)
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | String | Unique identifier for product-store combination | `HOBBIES_1_001_CA_1_validation` |
| `item_id` | String | Product identifier | `HOBBIES_1_001` |
| `dept_id` | String | Department identifier (parent category) | `HOBBIES_1` |
| `cat_id` | String | Product category | `HOBBIES` |
| `store_id` | String | Store identifier | `CA_1` |
| `state_id` | String | State code | `CA` |

#### Sales Data Columns (1,913 columns)
- **Format**: `d_1` through `d_1913` representing sequential days
- **Data Type**: Non-negative integers (unit sales)
- **Temporal Mapping**: 
  - `d_1` = 2011-01-29
  - `d_1913` = 2016-06-19
- **Characteristics**:
  - Contains zero values indicating no sales on certain days
  - Some products show sparse data (intermittent sales)
  - Wide range of values from 0 to several units per day

### Data Organization
- Each row represents a unique product-store combination
- Products span multiple departments and categories (HOBBIES, FOODS, HOUSEHOLD)
- Stores distributed across multiple states (CA, TX, WI)
- Same product-store pairs exist in both validation and evaluation files

---

## 2. Sales Training Data (Evaluation): `sales_train_evaluation.csv`

### Dimensions
- **Rows**: 30,490 (matches sales_train_validation.csv)
- **Columns**: 1,947 (6 metadata + 1,941 sales columns)
- **Extended Date Range**: 2011-01-29 to 2016-06-19 (1,941 days)

### Differences from Validation File
- **Extended Timeline**: Contains 28 additional days beyond the validation set
- **Structure**: Identical metadata columns, but with `d_1` through `d_1941`
- **Purpose**: Extended training/evaluation data for longer-horizon forecasting
- **Format**: Same as validation file - one row per product-store combination

### Column Structure
- Metadata columns: Same as `sales_train_validation.csv`
- Sales columns: `d_1` through `d_1941` (1,941 days)

---

## 3. Calendar Information: `calendar.csv`

### Dimensions
- **Rows**: 1,970 (one row per day)
- **Columns**: 13
- **Date Range**: 2011-01-29 to 2016-06-19

### Column Structure

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | String (YYYY-MM-DD) | Calendar date | `2011-01-29` |
| `wm_yr_wk` | Integer | Walmart year-week code | `11101` |
| `weekday` | String | Day of week name | `Saturday` |
| `wday` | Integer | Weekday numeric (1-7) | `1` (Saturday), `3` (Monday) |
| `month` | Integer | Month number (1-12) | `1` |
| `year` | Integer | Year | `2011` |
| `d` | String | Day column identifier | `d_1`, `d_2`, etc. |
| `event_name_1` | String | First event name (if any) | Empty or event name |
| `event_type_1` | String | First event type | `Sporting`, `Cultural`, etc. |
| `event_name_2` | String | Second event name (if any) | Empty or event name |
| `event_type_2` | String | Second event type | `Sporting`, `Cultural`, etc. |
| `snap_CA` | Integer | SNAP benefit active in CA (0/1) | `0` or `1` |
| `snap_TX` | Integer | SNAP benefit active in TX (0/1) | `0` or `1` |
| `snap_WI` | Integer | SNAP benefit active in WI (0/1) | `0` or `1` |

### Key Features
- **Walmart Week Code**: Identifies Walmart's fiscal week (5-digit format: YYWW)
- **Events**: Up to 2 events per day with types (Sporting, Cultural, etc.)
- **SNAP**: Supplemental Nutrition Assistance Program indicators per state
- **Weekday Mapping**: 
  - 1 = Saturday
  - 2 = Sunday
  - 3 = Monday
  - ... (7 = Friday)

### Sample Data Patterns
- Events like "NBAFinalsEnd", "Father's day" appear on specific dates
- SNAP activation varies by state and time period
- Weekday and month columns enable day-of-week and monthly seasonality analysis

---

## 4. Selling Prices: `sell_prices.csv`

### Dimensions
- **Rows**: 6,841,121 (approx 6.8M records)
- **Columns**: 4
- **Coverage**: Store-item-week combinations with pricing data

### Column Structure

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `store_id` | String | Store identifier | `CA_1` |
| `item_id` | String | Product identifier | `HOBBIES_1_001` |
| `wm_yr_wk` | Integer | Walmart year-week code | `11325` |
| `sell_price` | Float | Weekly selling price | `9.58` |

### Key Characteristics
- **Weekly Granularity**: Prices are recorded at the Walmart week level
- **Sparse Coverage**: Not all store-item combinations have prices for all weeks
- **Price Range**: Varies significantly across products and time periods
- **Linking**: Use `wm_yr_wk` to join with calendar.csv; use `store_id` and `item_id` to join with sales data

### Data Quality Notes
- Some products may have missing price data for certain weeks
- Prices represent actual selling prices (not list prices)
- Useful for price elasticity and promotional analysis

---

## 5. Submission Format: `sample_submission.csv`

### Dimensions
- **Rows**: 60,981 (30,490 validation + 30,491 evaluation product-store combinations)
- **Columns**: 29 (1 ID + 28 forecast columns)
- **Forecast Horizon**: 28 days (4 weeks)

### Column Structure

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | String | Product-store identifier | `HOBBIES_1_001_CA_1_validation` |
| `F1` through `F28` | Float | Forecasted unit sales | `0.0` |

### Format Details
- **Row Types**:
  - Rows with suffix `_validation`: Forecasts for validation set (30,490 rows)
  - Rows with suffix `_evaluation`: Forecasts for evaluation set (30,491 rows)
- **Forecast Columns**: `F1` through `F28` representing 28 consecutive days
- **Expected Values**: Non-negative numbers (unit sales predictions)
- **Sample Submission**: Provided template shows all zeros as placeholder

### Forecast Alignment
- `F1` = forecast for day 1914 in validation context
- Aligns with extended evaluation period in sales_train_evaluation.csv

---

## Data Interconnections

### Cross-File References

```
sales_train_validation.csv / sales_train_evaluation.csv
    ├── item_id → sell_prices.csv (item_id)
    ├── store_id → sell_prices.csv (store_id)
    └── (implicit d_1 to d_1913/d_1941 mapping to calendar.csv's d column)

calendar.csv
    ├── d column → Maps to d_N columns in sales files
    ├── wm_yr_wk → Links to sell_prices.csv (wm_yr_wk)
    └── date → Provides temporal grounding (2011-01-29 onwards)

sell_prices.csv
    ├── store_id, item_id → Matches sales file dimensions
    └── wm_yr_wk → Maps to calendar.csv for date context

sample_submission.csv
    ├── id → Matches id field in sales files
    └── F1-F28 → Forecast for 28-day horizon post-training period
```

---

## Statistical Summary

### Product-Store Coverage
- **Total Combinations**: 30,490 unique product-store pairs
- **States**: 3 (CA, TX, WI)
- **Store Count**: 10 stores across 3 states
- **Product Categories**: Multiple (HOBBIES, FOODS, HOUSEHOLD)
- **Department Diversity**: Various departments within each category

### Temporal Coverage
- **Training Period**: 1,913 days (validation) / 1,941 days (evaluation)
- **Calendar Days**: 1,970 days
- **Year Range**: 2011-2016 (5+ years of data)
- **Event Coverage**: Up to 2 events per day across the dataset

### Sales Characteristics
- **Value Range**: 0 to multi-digit unit sales per day
- **Sparsity**: Many product-store combinations show zero sales on certain days
- **Variability**: Products show different sales patterns (stable, seasonal, intermittent)

---

## Notes for Implementation

1. **Time Series Nature**: Days are sequential; maintain temporal ordering during train/test splits
2. **Sparse Data**: Handle zero-sales periods appropriately in models
3. **Product-Store Granularity**: Each combination is independent; can be modeled separately or jointly
4. **Price Data**: Use weekly prices; may need to align with daily sales for feature engineering
5. **Calendar Features**: Day-of-week, events, and SNAP indicators are valuable for seasonality capture
6. **Forecast Submission**: Use 28 non-negative numbers for each product-store combination

---

## File Locations

All dataset files are located in: `/data/full_data/`

```
data/full_data/
├── sales_train_validation.csv (30.5K rows)
├── sales_train_evaluation.csv (30.5K rows)
├── calendar.csv (2K rows)
├── sell_prices.csv (6.8M rows)
└── sample_submission.csv (61K rows)
```

Last Updated: 2026-05-14
