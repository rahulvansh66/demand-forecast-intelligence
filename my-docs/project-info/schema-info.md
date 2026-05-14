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

[To be filled with detailed table documentation]

## Model Output Tables

[To be filled with detailed table documentation]

## Database Relationships Summary

[To be filled with relationship overview]
