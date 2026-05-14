# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a retail demand forecasting copilot that provides ML-driven demand forecasting, pricing insights, and risk analytics with an agentic decision copilot using the Walmart M5 dataset. The project focuses on three core components:

1. **Demand Forecasting** - Predict future product demand for store-item combinations
2. **Demand Behavior Profiling** - Generate multi-label profiles (trend, seasonality, variability, movement type) for products
3. **GenAI-based Business Insight Generation** - Convert ML outputs into actionable business insights

## Data Architecture

The project uses the Walmart M5 dataset with this data flow:
- **Primary data**: `sales_train_validation.csv` (daily unit sales per product-store combination)
- **Calendar data**: `calendar.csv` (dates, weekdays, events, holidays)
- **Processing unit**: Each `item_id + store_id` combination is treated as one demand series
- **Static input format**: Store ID, Item ID, Forecast Horizon (e.g., "CA_1", "FOODS_3_090", "7 days")

## ML Pipeline Structure

### Feature Engineering Strategy
- **Demand behavior features**: avg_sales, std_sales, zero_sales_ratio, cv_sales (Coefficient of Variation)
- **Trend analysis**: Compare recent vs historical averages (recent_avg_28 / older_avg)
- **Seasonality detection**: Monthly patterns, weekend effects, event/holiday impacts
- **Variability handling**: Special logic for low-demand products (avg < 1 unit or >70% zero-sales days)

### Demand Segmentation Approach
Products are clustered into business segments:
- **Stable Products** - Regular, predictable sales
- **Fast-Moving Products** - High volume, frequent sales  
- **Intermittent Products** - Irregular, low-frequency sales
- **Seasonal Products** - Event/time-driven demand spikes

Use `RobustScaler()` over `StandardScaler()` due to sales outliers in M5 dataset.

### Model Architecture Considerations
- **Forecasting models** should incorporate lag features, rolling averages, day-of-week patterns, and calendar effects
- **Behavior profiling** uses multi-label classification rather than single clustering
- **Variability calculation**: Handle CV instability when average sales near zero (see `docs/handle_variablity.md`)

## Implementation Sequence

The project follows this development pipeline (see `docs/implementation-sequence.md`):

1. **Data Collection & Sampling** → **Data Sanity Checks** → **EDA** 
2. **Hypothesis Testing** → **Preprocessing Pipeline** → **Training Pipeline**
3. **Inference Pipeline** → **Model Interpretation** → **Drift Detection**
4. **FastAPI Endpoints** → **Streamlit UI**

### Key Implementation Notes
- Always check for missing values and data quality BEFORE descriptive statistics
- Handle low-demand products separately in variability calculations (CV becomes unreliable)
- Use feature engineering focused on business-meaningful demand patterns
- Implement proper train/test splits respecting time series nature

## Development Workflow

When implementing ML components:
- Start with basic demand features before adding complex seasonality detection
- Validate demand segmentation clusters make business sense, not just statistical metrics
- Test variability labeling edge cases (zero sales, very low averages)
- Ensure forecast outputs integrate properly with behavior profiling for GenAI insights

## Business Context

The system generates static insights (no chatbot interface) with this flow:
```
User Input (Store ID, Item ID, Horizon) → 
Feature Engineering → 
Forecasting + Behavior Profiling → 
GenAI Business Insight Generation
```

Outputs should be interpretable for inventory planning decisions at Walmart scale.