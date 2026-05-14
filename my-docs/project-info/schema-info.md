# Walmart M5 Dataset Schema Documentation

## Introduction

This document provides comprehensive schema documentation for the Walmart M5 dataset used in the retail demand forecasting copilot. The dataset originates from the Kaggle M5 Forecasting Accuracy competition and serves as the foundation for ML-driven demand forecasting, pricing insights, and risk analytics.

The dataset spans from January 29, 2011 to June 19, 2016 (1,969 days total) and contains hierarchical sales data for Walmart stores across California (CA), Texas (TX), and Wisconsin (WI). Each CSV file represents a logical table in our data architecture, with clear relationships enabling comprehensive demand analysis.

**Dataset Scope:** 
- **Time Period:** 1,969 days (d_1 to d_1969)
- **Geographic Coverage:** 3 states, 10 stores total
- **Product Hierarchy:** 3 categories → 7 departments → 3,049 items
- **Processing Unit:** Each `item_id + store_id` combination represents one demand series

## Core Sales Data Tables

### sales_train_validation.csv - Primary Sales Training Data

Core transactional data containing daily unit sales for each item-store combination during the training period. This table forms the foundation for demand forecasting model development and historical pattern analysis.

**Technical Details:** 30,490 rows, covers d_1 to d_1913 (Jan 29, 2011 - May 22, 2016), represents all item-store combinations

**Fields:**

• **id** (String)
  - Unique row identifier combining item_id + store_id + "_validation"
  - Format: "HOBBIES_1_001_CA_1_validation"
  - Primary key for this table
  - Business context: Enables tracking of specific product-location demand series

• **item_id** (String)
  - Product identifier within the Walmart hierarchy
  - Format: "CATEGORY_DEPT_ITEM" (e.g., "HOBBIES_1_001")
  - Links to: sell_prices.csv for pricing data analysis
  - Business context: Represents individual SKUs for demand forecasting

• **dept_id** (String)  
  - Department identifier within product category
  - Format: "CATEGORY_DEPT" (e.g., "HOBBIES_1")
  - Links to: Product hierarchy for aggregation analysis
  - Business context: Mid-level product grouping for category management

• **cat_id** (String)
  - Top-level product category identifier
  - Values: "HOBBIES", "HOUSEHOLD", "FOODS"
  - Business context: Highest-level product segmentation for strategic analysis

• **store_id** (String)
  - Store location identifier
  - Format: "{State}_{Store_Number}" (e.g., "CA_1", "TX_2", "WI_3")
  - Links to: sell_prices.csv for store-specific pricing analysis
  - Business context: Geographic demand variation analysis

• **state_id** (String)
  - State-level geographic identifier
  - Values: "CA", "TX", "WI"
  - Business context: Regional demand pattern analysis and market segmentation

• **d_1 to d_1913** (Integer)
  - Daily unit sales for each calendar day
  - Links to: calendar.csv where d_1 = 2011-01-29, d_1913 = 2016-05-22
  - Zero values indicate no sales on that day
  - Business context: Core metric for demand forecasting and trend analysis
  - Technical note: 1,913 consecutive daily columns representing the training period

### sales_train_evaluation.csv - Extended Sales Data for Model Evaluation

Extended version of the training dataset that includes an additional 28 days of sales data for model evaluation purposes. Maintains identical structure to the validation dataset but covers a longer time horizon.

**Technical Details:** 30,490 rows, covers d_1 to d_1941 (Jan 29, 2011 - June 19, 2016), extends 28 days beyond training data

**Fields:**

• **id** (String)
  - Unique row identifier combining item_id + store_id + "_evaluation" 
  - Format: "HOBBIES_1_001_CA_1_evaluation"
  - Primary key for this table
  - Links to: sales_train_validation.csv via item_id and store_id matching

• **item_id** (String)
  - Product identifier, identical structure to sales_train_validation.csv
  - Links to: sales_train_validation.csv and sell_prices.csv
  - Business context: Enables comparison of training vs evaluation period performance

• **dept_id** (String)
  - Department identifier, maintains hierarchy consistency
  - Links to: Product hierarchy across all sales tables

• **cat_id** (String)
  - Category identifier, consistent with validation dataset
  - Business context: Category-level performance evaluation

• **store_id** (String)
  - Store identifier, identical to validation dataset structure
  - Links to: sell_prices.csv and sales_train_validation.csv

• **state_id** (String)
  - State identifier for geographic consistency
  - Business context: Regional evaluation analysis

• **d_1 to d_1941** (Integer)
  - Daily unit sales including evaluation period
  - Links to: calendar.csv for complete date mapping
  - Business context: Full dataset for backtesting and model validation
  - Technical note: Includes all 1,941 days from training start through evaluation end

## Supporting Dimension Tables

### calendar.csv - Date Dimension and Event Calendar

Comprehensive date dimension table providing calendar context, events, holidays, and SNAP benefit information for the entire dataset timespan. Essential for incorporating temporal patterns and external factors into demand forecasting models.

**Technical Details:** 1,969 rows, covers complete dataset timeline from d_1 to d_1969 (Jan 29, 2011 - June 19, 2016)

**Fields:**

• **date** (Date)
  - Calendar date in YYYY-MM-DD format
  - Links to: All sales tables via d_X mapping (d_1 = 2011-01-29)
  - Business context: Primary temporal dimension for time series analysis

• **wm_yr_wk** (Integer)
  - Walmart fiscal year-week identifier
  - Format: YYYWW (e.g., 11101 = 2011 week 1)
  - Links to: sell_prices.csv for weekly pricing alignment
  - Business context: Walmart's internal calendar system for business reporting

• **weekday** (String)
  - Day of the week name
  - Values: "Monday", "Tuesday", ..., "Sunday"
  - Business context: Day-of-week seasonality patterns in retail demand

• **wday** (Integer)
  - Numeric day of week
  - Values: 1-7 (1=Saturday, 7=Friday in Walmart's calendar)
  - Business context: Enables day-of-week effect modeling

• **month** (Integer)
  - Calendar month number
  - Values: 1-12
  - Business context: Monthly seasonality and trend analysis

• **year** (Integer)
  - Calendar year
  - Values: 2011-2016
  - Business context: Year-over-year growth and trend analysis

• **d** (String)
  - Day identifier matching sales table columns
  - Format: "d_X" where X = 1 to 1969
  - Links to: sales_train_validation.csv and sales_train_evaluation.csv columns
  - Business context: Bridge between calendar dates and sales data structure

• **event_name_1** (String)
  - Primary event or holiday name
  - Nullable field for non-event days
  - Business context: Major events affecting retail demand (Christmas, Thanksgiving, etc.)

• **event_type_1** (String)
  - Classification of primary event
  - Values: "Cultural", "National", "Religious", "Sporting"
  - Business context: Event category for demand impact modeling

• **event_name_2** (String)
  - Secondary event name for days with multiple events
  - Nullable field, less common than primary events
  - Business context: Additional events that may influence demand

• **event_type_2** (String)
  - Classification of secondary event
  - Same categories as event_type_1
  - Business context: Secondary event impact analysis

• **snap_CA** (Integer)
  - SNAP (Supplemental Nutrition Assistance Program) benefit issuance indicator for California
  - Values: 0 (no benefits), 1 (benefits issued)
  - Business context: SNAP benefit timing affects food category demand patterns

• **snap_TX** (Integer)
  - SNAP benefit issuance indicator for Texas
  - Values: 0 (no benefits), 1 (benefits issued)
  - Links to: Texas store sales data for benefit impact analysis

• **snap_WI** (Integer)
  - SNAP benefit issuance indicator for Wisconsin  
  - Values: 0 (no benefits), 1 (benefits issued)
  - Business context: State-specific benefit timing for demand modeling

### sell_prices.csv - Weekly Pricing Data

Comprehensive pricing data providing weekly sell prices for item-store combinations. Critical for price-demand elasticity analysis and understanding the relationship between pricing strategies and sales performance.

**Technical Details:** 6,841,121 rows, weekly granularity aligned with Walmart fiscal calendar, sparse matrix (not all item-store-week combinations present)

**Fields:**

• **store_id** (String)
  - Store location identifier
  - Format: "{State}_{Store_Number}" (e.g., "CA_1")
  - Links to: sales_train_validation.csv and sales_train_evaluation.csv
  - Business context: Store-specific pricing strategies and regional variations

• **item_id** (String)
  - Product identifier matching sales data structure
  - Format: "CATEGORY_DEPT_ITEM"
  - Links to: All sales tables for price-demand analysis
  - Business context: Item-level pricing decisions and elasticity modeling

• **wm_yr_wk** (Integer)
  - Walmart fiscal year-week for price period
  - Format: YYYWW, matches calendar.csv wm_yr_wk
  - Links to: calendar.csv for temporal alignment
  - Business context: Weekly pricing cycles and promotional timing

• **sell_price** (Float)
  - Unit selling price in USD for the specified week
  - Precision: Typically to 2 decimal places
  - Business context: Core input for price elasticity and revenue optimization
  - Technical note: Missing combinations indicate item not sold at that store during that week

## Model Output Tables

### sample_submission.csv - Competition Submission Format

Template defining the expected output format for demand forecasting models. Specifies the structure for 28-day ahead predictions required for both validation and evaluation periods.

**Technical Details:** 60,980 rows, represents forecast requirements for all item-store combinations across two submission types

**Fields:**

• **id** (String)
  - Forecast identifier combining item_id + store_id + submission type
  - Format: "HOBBIES_1_001_CA_1_validation" or "HOBBIES_1_001_CA_1_evaluation"
  - Links to: sales_train_validation.csv and sales_train_evaluation.csv via id matching
  - Business context: Identifies which demand series and evaluation period the forecast targets

• **F1 to F28** (Integer)
  - 28-day ahead unit sales forecasts
  - F1 = 1-day ahead, F2 = 2-day ahead, ..., F28 = 28-day ahead
  - Business context: Standard forecasting horizon for retail inventory planning
  - Links to: Future dates beyond the training data cutoff
  - Technical note: Model output format requiring non-negative integer predictions

**Forecast Periods:**
- **Validation forecasts:** Predict d_1914 to d_1941 (28 days after training data)
- **Evaluation forecasts:** Predict d_1942 to d_1969 (28 days after validation period)
- **Business context:** Supports both model development (validation) and final assessment (evaluation)

## Database Relationships Summary

The Walmart M5 dataset follows a hierarchical structure with clear relationships enabling comprehensive demand analysis across temporal, geographic, and product dimensions.

### Primary Data Flow

**Core Transaction Path:**
- `sales_train_validation.csv` ↔ `sales_train_evaluation.csv`: Same item-store combinations, extended timeline
- Daily sales columns (d_1, d_2, ...) → `calendar.csv`: Date dimension mapping
- (item_id, store_id) → `sell_prices.csv`: Price-demand relationship analysis

### Key Relationships

**Temporal Relationships:**
- **calendar.csv** serves as the master date dimension
  - `d` field maps to sales table column names (d_1 ↔ 2011-01-29)
  - `wm_yr_wk` links to pricing data weekly periods
  - Event and SNAP fields provide external demand drivers

**Geographic Hierarchy:**
- **state_id** (CA, TX, WI) → **store_id** (CA_1, TX_2, etc.)
- SNAP benefit timing varies by state (snap_CA, snap_TX, snap_WI)
- Store-specific pricing strategies in sell_prices.csv

**Product Hierarchy:**
- **cat_id** (HOBBIES, HOUSEHOLD, FOODS) → **dept_id** → **item_id**
- Enables aggregation at category, department, or item levels
- Consistent across all sales and pricing tables

**Forecasting Pipeline:**
- Training data (d_1 to d_1913) → Validation period (d_1914 to d_1941) → Evaluation period (d_1942 to d_1969)
- `sample_submission.csv` defines required output format (28-day horizons)
- Model inputs: historical sales + calendar features + pricing data

### Data Integration Patterns

**For Demand Forecasting:**
1. Join sales data with calendar for temporal features
2. Join with pricing data for price elasticity analysis  
3. Aggregate across product/geographic hierarchies as needed
4. Generate forecasts in sample_submission.csv format

**Primary Keys and Joins:**
- **Sales tables:** id (item_id + store_id + type)
- **Calendar:** d (maps to sales column names)
- **Pricing:** (store_id, item_id, wm_yr_wk)
- **Join pattern:** sales ← calendar (via d_X) ← pricing (via wm_yr_wk)

This relational structure supports the complete retail demand forecasting pipeline from raw transaction data through ML model training to business insight generation.
