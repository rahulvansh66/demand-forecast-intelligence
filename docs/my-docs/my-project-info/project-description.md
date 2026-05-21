Below is a strong project concept you can build from the Walmart M5 dataset: a **Demand Forecasting + Product Segmentation + LLM Business Insight System**.

## Project Title

**Retail Demand Intelligence System using XGBoost/LSTM, Product Segmentation, and LLM-Generated Business Insights**

## Core Objective

Build an end-to-end ML system that:

1. Forecasts future daily unit sales for each `item_id + store_id`.
2. Segments products based on sales behavior, volatility, price sensitivity, seasonality, and demand patterns.
3. Feeds forecasts and segment outputs into an LLM to generate actionable business insights for inventory, pricing, promotions, and replenishment.

The M5 dataset is well suited for this because each demand series is defined at the `item_id + store_id` level, with 30,490 product-store combinations, daily sales from `d_1` onward, calendar events, SNAP indicators, and weekly price data. 

---

# 1. Demand Forecasting Model

## Problem Statement

Predict daily product sales for the next 28 days.

For example:

> Given historical sales up to `d_1941`, predict sales for `d_1942` to `d_1969`.

In the original M5 competition, the forecasting horizon was 28 days, and the schema confirms that `sales_train_evaluation.csv` contains sales from `d_1` to `d_1941`, while the future evaluation period extends to `d_1969`. 

---

## Forecasting Target

Target variable:

```text
units_sold
```

Forecast granularity:

```text
item_id + store_id + date
```

Example:

| item_id         | store_id |       date | predicted_sales |
| --------------- | -------: | ---------: | --------------: |
| FOODS_3_090     |     CA_1 | 2016-06-20 |              23 |
| HOUSEHOLD_1_001 |     TX_2 | 2016-06-20 |               4 |
| HOBBIES_1_045   |     WI_3 | 2016-06-20 |               1 |

---

# 2. Feature Engineering for Forecasting

## A. Sales Lag Features

These are very important for XGBoost.

Examples:

```text
lag_1
lag_7
lag_14
lag_28
lag_56
lag_364
```

Business meaning:

* `lag_7`: same weekday last week
* `lag_28`: same weekday pattern roughly one month ago
* `lag_364`: same seasonal point last year

---

## B. Rolling Sales Features

Examples:

```text
rolling_mean_7
rolling_mean_14
rolling_mean_28
rolling_std_28
rolling_max_28
rolling_min_28
```

Business meaning:

* Recent demand trend
* Product volatility
* Demand stability
* Stock or promotion spikes

---

## C. Calendar Features

From `calendar.csv`, you can use:

```text
weekday
wday
month
year
event_name_1
event_type_1
event_name_2
event_type_2
snap_CA
snap_TX
snap_WI
```

The schema shows that `calendar.csv` maps each `d_x` value to an actual date and includes holidays, events, and SNAP benefit indicators. 

Useful derived features:

```text
is_weekend
is_event_day
is_snap_day
days_to_christmas
days_to_thanksgiving
month_start
month_end
```

---

## D. Price Features

From `sell_prices.csv`:

```text
sell_price
price_lag_1
price_change
price_momentum
price_rolling_mean_4w
discount_flag
relative_price_by_item
relative_price_by_dept
```

The schema confirms that `sell_prices.csv` contains weekly prices by `store_id`, `item_id`, and `wm_yr_wk`, which can be joined to daily sales using the calendar table. 

Business meaning:

* Detect price-sensitive products
* Estimate impact of price changes
* Identify promotion-like behavior
* Support revenue forecasting

---

## E. Hierarchical Features

The dataset has product hierarchy:

```text
cat_id → dept_id → item_id
```

And geographic hierarchy:

```text
state_id → store_id
```

You can create features such as:

```text
category_avg_sales
department_avg_sales
store_avg_sales
state_avg_sales
item_avg_sales
```

These help the model learn demand patterns for sparse or low-selling products.

---

# 3. Forecasting Model Options

## Option A: XGBoost Forecasting Model

This is the most practical and explainable starting point.

### Input format

Convert the wide sales table:

```text
id, item_id, dept_id, cat_id, store_id, state_id, d_1, d_2, ...
```

into long format:

```text
item_id, store_id, date, d, sales
```

Then join:

```text
sales + calendar + sell_prices
```

### XGBoost model design

Train one global model across all products and stores.

```text
Input: lag features + rolling features + calendar features + price features + hierarchy features
Output: daily sales prediction
```

### Why XGBoost is good here

* Handles tabular retail data very well
* Works with lag and rolling features
* Faster than LSTM
* Easier to explain using feature importance or SHAP
* Strong baseline for M5-style forecasting

---

## Option B: LSTM Forecasting Model

LSTM can be used for sequence-based demand prediction.

### Input

For each `item_id + store_id`, use a historical window such as:

```text
past 56 days → predict next 28 days
```

Example:

```text
X = sales from d_1886 to d_1941
y = sales from d_1942 to d_1969
```

### Additional features

You can add calendar and price features as external regressors.

### Why LSTM is useful

* Learns temporal patterns directly
* Useful for products with strong sequential behavior
* Can model multi-step forecasting
* Good for comparing deep learning vs tree-based models

### Practical recommendation

Use **XGBoost as your main production-style model** and **LSTM as an experimental comparison model**.

For a portfolio or capstone project, this comparison is valuable:

| Model   | Strength                                      | Weakness                                         |
| ------- | --------------------------------------------- | ------------------------------------------------ |
| XGBoost | Strong tabular performance, explainable, fast | Requires manual lag features                     |
| LSTM    | Learns sequential patterns                    | Slower, harder to explain, needs careful scaling |
| Hybrid  | XGBoost + LSTM ensemble                       | More complex                                     |

---

# 4. Product Sales Segmentation Model

Segmentation is a very good second ML component. It helps convert raw forecasts into business strategy.

## Goal

Cluster products into groups based on sales behavior.

Instead of only asking:

> “How many units will this product sell?”

You also ask:

> “What type of demand behavior does this product have?”

---

# 5. Segmentation Features

Create one row per `item_id + store_id`.

Example features:

```text
avg_daily_sales
median_daily_sales
sales_std
sales_cv
zero_sales_ratio
max_sales
sales_trend
seasonality_strength
event_sales_lift
snap_sales_lift
avg_price
price_std
price_change_frequency
price_elasticity_proxy
recent_28_day_sales
recent_90_day_sales
forecasted_28_day_sales
forecast_growth_rate
```

Where:

```text
sales_cv = sales_std / avg_daily_sales
zero_sales_ratio = number_of_zero_sales_days / total_days
forecast_growth_rate = forecasted_28_day_sales / previous_28_day_sales - 1
```

---

# 6. KMeans Use Cases

KMeans is useful when you want clear business-friendly product groups.

## Possible Segments

### Segment 1: High Demand, Stable Products

Characteristics:

```text
high avg sales
low volatility
low zero-sales ratio
consistent forecast
```

Business action:

* Keep high inventory availability
* Avoid stockouts
* Use automated replenishment
* Monitor supply chain delays

Example LLM insight:

> “FOODS products in CA stores show stable high-volume demand. These should be prioritized for inventory replenishment because even small stockouts may cause significant lost sales.”

---

### Segment 2: High Demand, Volatile Products

Characteristics:

```text
high avg sales
high sales_std
event-sensitive
promotion-sensitive
```

Business action:

* Use event-aware forecasting
* Increase safety stock before holidays
* Monitor promotion periods
* Use dynamic replenishment

Example LLM insight:

> “This cluster contains products with strong demand spikes around events. Inventory planning should account for calendar holidays and SNAP benefit timing.”

---

### Segment 3: Low Demand, Intermittent Products

Characteristics:

```text
low avg sales
high zero-sales ratio
sporadic demand
```

Business action:

* Reduce reorder frequency
* Use conservative inventory
* Consider bundling or promotions
* Avoid overstocking

Example LLM insight:

> “These items sell intermittently and may create excess inventory risk. A lower replenishment frequency or promotion-based liquidation strategy may be appropriate.”

---

### Segment 4: Declining Products

Characteristics:

```text
negative sales trend
falling recent sales
low forecast growth
```

Business action:

* Reduce future purchase orders
* Investigate product lifecycle
* Consider markdowns
* Replace with alternatives

Example LLM insight:

> “Several HOUSEHOLD items show declining demand across multiple stores. Buyers should evaluate whether this is a seasonal dip or a product lifecycle issue.”

---

### Segment 5: Seasonal Products

Characteristics:

```text
strong yearly/monthly pattern
holiday lift
event-based sales spikes
```

Business action:

* Plan seasonal inventory
* Forecast separately around events
* Align marketing campaigns
* Increase stock before predictable demand peaks

---

# 7. DBSCAN Use Cases

DBSCAN is useful for finding unusual products rather than clean business clusters.

## Strong DBSCAN Use Cases

### A. Detect Anomalous Products

Examples:

```text
sudden sales spike
sudden demand collapse
extreme price sensitivity
unusual zero-sales pattern
```

Business action:

* Investigate possible data issues
* Detect stockouts
* Identify promotion effects
* Flag abnormal demand behavior

---

### B. Identify Niche Demand Products

DBSCAN can detect dense groups and label unusual points as noise.

Products marked as noise may represent:

```text
rare demand products
highly seasonal products
newly introduced products
abnormal store-specific behavior
```

Business action:

* Treat separately from standard replenishment
* Use custom forecasting rules
* Review assortment strategy

---

### C. Store-Specific Outlier Detection

Example:

A product may sell normally in CA and TX but unusually in WI.

DBSCAN can help identify:

```text
regional preference differences
store-level demand anomalies
pricing issues
local promotion effects
```

---

# 8. Recommended Segmentation Strategy

Use both models, but for different business purposes.

| Model  | Best For                       | Output                       |
| ------ | ------------------------------ | ---------------------------- |
| KMeans | Business-friendly segmentation | Product groups               |
| DBSCAN | Outlier and anomaly discovery  | Normal groups + noise points |

Recommended approach:

```text
KMeans → main product segmentation
DBSCAN → anomaly and special-case product detection
```

---

# 9. How Forecasting and Segmentation Work Together

The best project design is not to keep forecasting and clustering separate.

Use forecast outputs as segmentation inputs.

Example segmentation features after forecasting:

```text
forecasted_28_day_sales
forecast_vs_last_28_days
forecast_growth_rate
forecast_volatility
risk_of_stockout
expected_revenue
```

Then generate business insights using both:

```text
forecast result + product segment + product hierarchy + price/calendar context
```

---

# 10. LLM Business Insight Layer

## Input to LLM

For each product, department, store, or category, pass structured model outputs.

Example JSON:

```json
{
  "item_id": "FOODS_3_090",
  "store_id": "CA_1",
  "category": "FOODS",
  "department": "FOODS_3",
  "forecast_next_28_days": 428,
  "previous_28_days_sales": 370,
  "forecast_growth_rate": 0.157,
  "cluster_name": "High Demand - Growing",
  "avg_sell_price": 2.98,
  "event_sensitivity": "high",
  "snap_sensitivity": "medium",
  "risk_level": "stockout risk"
}
```

## LLM Output

The LLM can generate:

```text
business summary
inventory recommendation
pricing recommendation
promotion recommendation
risk explanation
category manager action plan
```

Example:

> “FOODS_3_090 in CA_1 is forecasted to sell 428 units over the next 28 days, a 15.7% increase compared with the previous 28 days. Since this product belongs to the ‘High Demand - Growing’ segment and shows event sensitivity, the store should increase replenishment quantities before upcoming event days. Stockout risk is high if inventory remains based on the previous 28-day average.”

---

# 11. Suggested System Architecture

```text
                Walmart M5 Dataset
                       |
        ---------------------------------
        |               |               |
 sales_train       calendar        sell_prices
        |               |               |
        -------- Data Integration -------
                       |
              Feature Engineering
                       |
        ---------------------------------
        |                               |
 Demand Forecasting              Product Segmentation
 XGBoost / LSTM                  KMeans / DBSCAN
        |                               |
  28-day sales forecast          Product segment labels
        |                               |
        -------- Model Output Store ----
                       |
                 LLM Insight Layer
                       |
        ---------------------------------
        |               |               |
 Inventory Insight  Pricing Insight  Promotion Insight
```

---

# 12. Final Project Modules

## Module 1: Data Pipeline

Tasks:

```text
load sales_train_evaluation.csv
melt wide daily columns into long format
join with calendar.csv
join with sell_prices.csv
handle missing prices
create date-based features
```

---

## Module 2: Forecasting

Models:

```text
XGBoost Regressor
LSTM sequence model
```

Evaluation metrics:

```text
RMSE
MAE
RMSSE
MAPE, with care for zero-sales products
```

Forecasting output:

```text
daily forecast for next 28 days
item-store-level forecast
category/store aggregated forecast
```

---

## Module 3: Segmentation

Models:

```text
KMeans
DBSCAN
```

Evaluation:

```text
silhouette score
Davies-Bouldin score
cluster interpretability
business usefulness
```

Outputs:

```text
cluster_id
cluster_name
cluster_profile
anomaly_flag
```

---

## Module 4: Insight Generation

Use LLM to create:

```text
executive summary
store-level recommendations
category-level recommendations
product-level action plan
risk alerts
```

---

# 13. Strong Use Cases for Your Project

## Use Case 1: Inventory Replenishment Optimization

Use forecasted 28-day demand to recommend stock levels.

Example insight:

> “Increase inventory for high-growth FOODS items in CA stores by 12–18% for the next cycle.”

---

## Use Case 2: Stockout Risk Detection

Flag products where forecasted sales exceed recent sales or assumed inventory thresholds.

Example:

```text
high forecast growth + high demand cluster = stockout risk
```

---

## Use Case 3: Overstock Risk Detection

Identify products with declining forecasts and low-demand clusters.

Example:

```text
declining forecast + intermittent demand cluster = overstock risk
```

---

## Use Case 4: Promotion Candidate Selection

Find products with:

```text
low recent sales
high inventory risk
moderate historical demand
price sensitivity
```

Recommended action:

> Run markdown or bundle promotion.

---

## Use Case 5: Event-Based Demand Planning

Use event fields from `calendar.csv`, such as holidays and sporting events, to detect products that experience event-driven demand spikes. 

Example:

> “This product cluster shows higher sales before national and cultural events. Plan promotion and inventory in advance.”

---

## Use Case 6: SNAP-Sensitive Product Analysis

Because the calendar table includes state-specific SNAP indicators for CA, TX, and WI, you can analyze how food demand changes around SNAP benefit days. 

Example:

> “FOODS products in TX stores show increased sales during SNAP days. Replenishment should be increased during SNAP benefit periods.”

---

## Use Case 7: Store-Product Localization

Cluster products by demand behavior across stores.

Example:

> “The same product behaves as high-demand in CA_1 but intermittent in WI_2. Store-specific replenishment is more appropriate than a national-level strategy.”

---

# 14. Recommended Final Project Scope

For a polished project, I recommend this scope:

## Forecasting

Use:

```text
XGBoost as primary model
LSTM as comparison model
```

Forecast level:

```text
item_id + store_id
```

Forecast horizon:

```text
28 days
```

---

## Segmentation

Use:

```text
KMeans for product demand segments
DBSCAN for anomaly detection
```

Segmentation level:

```text
item_id + store_id
```

---

## LLM Insights

Generate insights at three levels:

```text
store level
category level
product level
```

Example outputs:

```text
Top 10 products at stockout risk
Top 10 products at overstock risk
Products suitable for promotion
Event-sensitive products
SNAP-sensitive food products
High-growth products by store
```

---

# 15. Example Final Deliverables

Your final project can include:

```text
1. Data preprocessing notebook
2. Feature engineering pipeline
3. XGBoost forecasting notebook
4. LSTM forecasting notebook
5. KMeans/DBSCAN segmentation notebook
6. LLM insight generation module
7. Streamlit dashboard
8. Model evaluation report
9. Business recommendation report
```

Dashboard sections:

```text
Forecast Explorer
Product Segment Explorer
Stockout Risk Dashboard
Overstock Risk Dashboard
Promotion Recommendation
LLM Business Insight Summary
```

---

# 16. Best Final Framing

You can present the project as:

> “A retail intelligence system that predicts future product-store sales, segments products by demand behavior, and converts ML outputs into business recommendations using an LLM.”

This is stronger than just saying:

> “I built a sales forecasting model.”

Because your project includes:

```text
forecasting
segmentation
anomaly detection
business insight generation
decision support
LLM integration
```

That makes it suitable for a data science, ML engineering, or GenAI portfolio project.
