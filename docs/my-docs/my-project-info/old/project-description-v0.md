# 🧠 Business Problem: Demand Forecasting and Demand Behavior Profiling using Walmart M5 Dataset

Walmart sells thousands of products across different stores, categories, and regions. Each product behaves differently: some products sell regularly, some sell only during specific seasons, some show increasing demand, and some sell only occasionally.

The business challenge is to:

* Predict future product demand
* Understand how each product behaves over time
* Identify demand patterns such as trend, seasonality, variability, and movement type
* Generate simple business insights to support inventory planning

Using the Walmart M5 dataset, this project focuses on:

1. **Demand Forecasting**
2. **Demand Behavior Profiling**
3. **GenAI-based Business Insight Generation**

---

# 1) Demand Forecasting

## What will sell and how much?

### Simple meaning

Demand forecasting predicts how many units of a product are expected to sell for each day within the forecast horizon (e.g., daily predictions for the next 7 days).

### Implementation

Use time series forcasting model

---

### Example

For a selected store and product:

```text
Store ID = CA_1
Item ID = FOODS_3_090
Forecast Horizon = Next 7 days
```

The model predicts:

```text
Day 1: 62 units
Day 2: 58 units  
Day 3: 65 units
Day 4: 59 units
Day 5: 61 units
Day 6: 57 units
Day 7: 58 units

Total expected demand for 7 days = 420 units
```

---

### What the model uses from M5

* Historical daily sales
* Lag features
* Rolling averages
* Day-of-week patterns
* Month-level patterns
* Holiday and event information
* Store, category, and department information

---

### What it learns

The forecasting model learns patterns such as:

* Demand increases on weekends
* Demand changes around holidays and events
* Some products have repeating seasonal demand
* Some products have stable daily demand
* Some products have irregular demand

---

### Output

```text
Forecast Output:
Item: FOODS_3_090
Store: CA_1
Forecast Horizon: 7 days
Daily Predictions: [62, 58, 65, 59, 61, 57, 58]
Total Predicted Demand: 420 units
```

---

### Why business cares

Demand forecasting helps Walmart:

* Plan inventory in advance
* Reduce overstock
* Reduce understock
* Improve product availability
* Support replenishment decisions

---

# 2) Demand Behavior Profiling

## How does the product behave?

### Simple meaning

Demand behavior profiling describes the sales behavior of a product using multiple labels.

Instead of assigning only one cluster, the system creates a richer profile for each product.

---

### Demand behavior labels

For each product-store combination, the system can generate labels such as:

```text
Trend Label: Increasing / Decreasing / Stable
Seasonality Label: Seasonal / Non-seasonal
Demand Segment (do feature engineering and segmentation model) : Fast-moving / Stable / Intermittent / Low-demand / etc
Variability Label: Low / Medium / High
```

---

## A) Trend Label

### What it means

Trend shows whether recent demand is moving upward, downward, or staying stable.

---

### Example

| Week   | Sales |
| ------ | ----- |
| Week 1 | 120   |
| Week 2 | 145   |
| Week 3 | 170   |
| Week 4 | 210   |

This product has an:

```text
Increasing Trend
```

---

### Possible outputs

```text
Trend Label = Increasing
Trend Label = Decreasing
Trend Label = Stable
```

---

## B) Seasonality Label

### What it means

Seasonality shows whether demand repeats during specific periods, such as weekends, holidays, months, or seasons.

---

### Example

A product sells more during holiday periods or summer months.

Output:

```text
Seasonality Label = Seasonal
```

If no clear repeating seasonal pattern is found:

```text
Seasonality Label = Non-seasonal
```

---

## C) Demand Segment

### What it means

Demand segment describes the overall movement type of the product.

### Implementation

Use segmentation model

---

### Example segments

| Demand Segment | Meaning                                   |
| -------------- | ----------------------------------------- |
| Fast-moving    | High and frequent sales                   |
| Stable         | Regular and predictable sales             |
| Intermittent   | Many zero-sales days and irregular demand |
| Low-demand     | Consistently low sales volume             |

---

### Example output

```text
Demand Segment = Fast-moving
```

---

## D) Variability Label

### What it means

Variability shows how much sales fluctuate over time.

It answers: “Does this product sell almost the same amount every day, or does sales quantity change a lot?”

---

### Example

```text
Low variability = sales are consistent
High variability = sales change a lot day to day
```

Possible outputs:

```text
Variability Label = Low
Variability Label = Medium
Variability Label = High
```

You can get **Variability Label** from historical sales fluctuation.

It answers:

> “Does this product sell almost the same amount every day, or does sales quantity change a lot?”

---

# Simple idea

For each `item_id + store_id`, take its historical daily sales and calculate how much sales vary.

Example:

| Day | Sales |
| --- | ----: |
| d1  |    20 |
| d2  |    21 |
| d3  |    19 |
| d4  |    20 |
| d5  |    22 |

This product has **low variability** because sales stay close to 20.

Another product:

| Day | Sales |
| --- | ----: |
| d1  |     0 |
| d2  |     5 |
| d3  |    40 |
| d4  |     2 |
| d5  |    60 |

This product has **high variability** because sales jump a lot.

---

# Best metric: Coefficient of Variation

Use:

```text
Coefficient of Variation = Standard Deviation of Sales / Average Sales
```

This is better than only using standard deviation because it adjusts for product scale.

Example:

```text
Product A average sales = 100, standard deviation = 10
CV = 10 / 100 = 0.10 → Low variability
```

```text
Product B average sales = 5, standard deviation = 5
CV = 5 / 5 = 1.00 → High variability
```

---

# Suggested label rules

You can define simple thresholds:

|       CV Value | Variability Label |
| -------------: | ----------------- |
|       CV < 0.5 | Low               |
| 0.5 ≤ CV < 1.0 | Medium            |
|       CV ≥ 1.0 | High              |

---

# Example calculation

Suppose recent 28-day sales are:

```text
[20, 22, 19, 21, 20, 23, 18]
```

Average sales:

```text
20.43
```

Standard deviation:

```text
1.62
```

CV:

```text
1.62 / 20.43 = 0.08
```

So:

```text
Variability Label = Low
```

---

# How to use it in your project

For each product-store combination:

```text
Input:
Historical daily sales for item_id + store_id

Step 1:
Calculate average daily sales

Step 2:
Calculate standard deviation of daily sales

Step 3:
Calculate CV = std / mean

Step 4:
Assign label:
Low / Medium / High
```

---

# Important handling for low-demand products

Some products have average sales near zero. In that case, CV can become unstable.

So check `docs/handle_variablity.md` to handle Variability Label in such cases. 

---

# Combined Demand Behavior Profile

For one product-store combination, the output may look like this:

```text
Item: FOODS_3_090
Store: CA_1

Trend Label: Increasing
Seasonality Label: Seasonal
Demand Segment: Fast-moving
Variability Label: Medium
```

This gives a clearer picture than a single segment label.

---

### Why business cares

Different demand profiles need different planning strategies.

| Behavior Profile            | Business Meaning                 |
| --------------------------- | -------------------------------- |
| Fast-moving + Increasing    | Prepare for higher demand        |
| Seasonal + High variability | Plan earlier before peak periods |
| Stable + Low variability    | Maintain regular stock levels    |
| Intermittent + Low-demand   | Avoid excess inventory           |
| Decreasing + Low-demand     | Reduce inventory allocation      |

---

# 3) GenAI-based Business Insight Generation

## What does this mean for the business?

### Simple meaning

The ML system produces numerical and label-based outputs. GenAI converts those outputs into simple business insights.

---

### Input sent to GenAI

```text
Store ID: CA_1
Item ID: FOODS_3_090
Forecast Horizon: 7 days
Daily Predictions: [62, 58, 65, 59, 61, 57, 58]
Total Predicted Demand: 420 units

Trend Label: Increasing
Seasonality Label: Seasonal
Demand Segment: Fast-moving
Variability Label: Medium
```

---

### GenAI output

> FOODS_3_090 in Store CA_1 is expected to sell around 420 units over the next 7 days. The product is fast-moving and shows an increasing demand trend. It also has seasonal behavior, meaning demand may rise during certain recurring periods. Walmart should plan inventory proactively to support higher near-term demand.

---

# Static User Input Format

The user will provide a fixed input format:

```text
Store ID = CA_1
Item ID = FOODS_3_090
Forecast Horizon = 7 days
```

The system will not need a dynamic chatbot flow.

---

# Static System Flow

The system follows the same flow every time.

```text
User enters Store ID, Item ID, and Forecast Horizon
        ↓
Historical sales and calendar data are filtered
        ↓
Feature engineering is performed
        ↓
Demand forecasting model predicts future demand
        ↓
Demand behavior profiling generates labels
        ↓
Forecast output + behavior profile are collected
        ↓
Outputs are sent to GenAI
        ↓
GenAI generates business insight
```

---

# Example End-to-End Flow

## User input

```text
Store ID = CA_1
Item ID = FOODS_3_090
Forecast Horizon = 7 days
```

---

## Forecasting output

```text
Daily Predictions = [62, 58, 65, 59, 61, 57, 58]
Total Predicted Demand = 420 units
```

---

## Demand behavior profile

```text
Trend Label = Increasing
Seasonality Label = Seasonal
Demand Segment = Fast-moving
Variability Label = Medium
```

---

## Final GenAI insight

> FOODS_3_090 in CA_1 is forecasted to sell 420 units over the next 7 days, with daily predictions of [62, 58, 65, 59, 61, 57, 58] units respectively. The product is fast-moving, has an increasing demand trend, and shows seasonal behavior. Since demand is expected to remain strong with consistent daily sales around 60 units, Walmart should prepare inventory in advance and monitor this item closely during upcoming high-demand periods.

---

# Final Project Objective

The objective of this project is to build a demand analysis system using the Walmart M5 dataset that forecasts future product demand, profiles each product’s demand behavior, and generates clear GenAI-based business insights. The system helps Walmart understand product demand patterns and make better inventory planning decisions.
