# 🧠 Business Problem: Demand Forecasting and Demand Behavior Analysis using Walmart M5 Dataset

Walmart sells thousands of products across multiple stores, categories, and regions. Product demand changes over time due to seasonality, holidays, customer buying behavior, and regional patterns.

The business challenge is to:

* Predict future product demand accurately
* Understand how product demand behaves over time
* Identify products with increasing, decreasing, seasonal, stable, or intermittent demand patterns
* Support smarter inventory planning and replenishment decisions

Using the Walmart M5 dataset, this project focuses on:

1. Demand Forecasting
2. Demand Segmentation and Trend Analysis
3. GenAI-powered business explanations and recommendations

---

# 🧠 1) Demand Forecasting

## What will sell?

### 👉 Simple meaning

Predict how many units of a product will be sold in the future.

---

### 🛒 Real-world example

A Walmart store observes:

* Milk sales are usually higher on weekends
* Some grocery items sell more during holidays
* Certain products show seasonal spikes

The business wants to know:

👉 *“How much stock should we prepare for upcoming days or weeks?”*

---

### 📊 What the model uses from M5

* Historical daily sales
* Lag features
* Rolling averages
* Day of week
* Month and seasonal patterns
* Holidays and special events
* Store/category/department information

---

### 🧠 What the model learns

The forecasting model learns patterns such as:

* Weekend demand spikes
* Holiday-driven sales increases
* Seasonal demand behavior
* Recurring demand cycles
* Product-level demand variability

---

### 🎯 Output

```text id="4t0t8q"
FOODS_3_090 (CA_1)

Tomorrow forecast → 120 units
Next 7-day forecast → 420 units
```

---

### 💡 Why business cares

Demand forecasting helps Walmart:

* Reduce overstock and waste
* Avoid product shortages
* Improve replenishment planning
* Optimize inventory allocation
* Improve product availability

---

# 🧩 2) Demand Segmentation and Trend Analysis

## How do products behave over time?

### 👉 Simple meaning

Group products based on similar demand behavior and identify whether demand is increasing, decreasing, stable, seasonal, or intermittent.

Instead of treating all products the same, Walmart can understand how different products behave and plan inventory strategies accordingly.

---

# 📈 Trend Analysis

## What is trend analysis?

Trend analysis identifies whether product demand is:

* Increasing
* Decreasing
* Stable

over recent time periods.

This is derived directly from historical sales data in the M5 dataset.

---

### 🛒 Example

| Week   | Sales |
| ------ | ----- |
| Week 1 | 120   |
| Week 2 | 145   |
| Week 3 | 170   |
| Week 4 | 210   |

This indicates:

```text id="l36jgw"
Demand trend = Increasing
```

---

### 🧠 How trend is derived

Using:

* Rolling averages
* Sales growth rate
* Time-series slope
* Recent sales momentum

Example:

```text id="7kgfq9"
Last 7-day average sales > Last 28-day average sales
→ Increasing trend
```

---

### 🎯 Output

```text id="0pfmke"
SKU FOODS_3_090

Trend: Increasing
Recent sales growth: Positive
Demand momentum: Strong
```

---

# 🧩 Demand Segmentation

## Product behavior groups

### Cluster 1: Stable Products

Products with regular and predictable demand.

Examples:

* Milk
* Bread
* Rice

Business meaning:

```text id="6b2t7w"
Maintain regular replenishment and consistent stock levels.
```

---

### Cluster 2: Seasonal Products

Products whose demand changes during specific seasons or events.

Examples:

* Fans during summer
* Holiday products
* Rain-related items

Business meaning:

```text id="jjlwm0"
Prepare inventory before expected seasonal demand spikes.
```

---

### Cluster 3: Intermittent Products

Products with irregular demand and many zero-sales days.

Examples:

* Specialty items
* Rare-purchase products

Business meaning:

```text id="a6kd31"
Avoid excess inventory because demand is unpredictable.
```

---

### Cluster 4: Fast-Moving Products

Products with consistently high sales volume.

Examples:

* Daily-use groceries
* Frequently purchased household products

Business meaning:

```text id="glnkpz"
Require continuous monitoring and frequent replenishment.
```

---

### 📊 What segmentation uses from M5

* Sales frequency
* Demand variability
* Rolling sales statistics
* Trend behavior
* Seasonal patterns
* Zero-sales frequency
* Product/store/category-level demand patterns

---

### 🎯 Output

```text id="98us6n"
SKU FOODS_3_090
Segment: Fast-moving
Trend: Increasing

SKU HOUSEHOLD_1_210
Segment: Seasonal
Trend: Stable
```

---

### 💡 Why business cares

Different demand behaviors require different inventory strategies.

| Product Type     | Business Strategy                     |
| ---------------- | ------------------------------------- |
| Stable           | Maintain regular stock                |
| Seasonal         | Increase inventory before peak season |
| Intermittent     | Keep limited inventory                |
| Fast-moving      | Replenish frequently                  |
| Increasing trend | Prepare for growing demand            |
| Decreasing trend | Avoid overstocking                    |

---

# 🤖 Where GenAI Comes In

The ML system generates outputs such as:

```text id="jlwmgo"
Forecast = 420 units
Segment = Fast-moving
Trend = Increasing
```

GenAI converts them into business-friendly explanations.

---

### Example GenAI explanation

> “This product belongs to the fast-moving segment and has shown an increasing sales trend over recent weeks. Forecasted demand for next week is significantly higher than average historical demand, indicating the need for proactive replenishment planning.”

---

# 🔗 Overall System Flow

```text id="o9yfjg"
Historical Sales Data
        ↓
Feature Engineering
        ↓
Demand Forecasting
        ↓
Demand Trend Analysis
        ↓
Demand Segmentation
        ↓
RAG-based Retrieval
        ↓
GenAI Business Explanation
        ↓
Business User Recommendation
```

---

# 🧠 Final Business Intuition

This project answers important Walmart business questions.

| Use Case             | Business Question                     |
| -------------------- | ------------------------------------- |
| Demand Forecasting   | What products will sell and how much? |
| Trend Analysis       | Is demand increasing or decreasing?   |
| Demand Segmentation  | Which products behave similarly?      |
| GenAI Recommendation | What business action should we take?  |

---

# ✅ Final Project Objective

The objective of this project is to use the Walmart M5 dataset to build an intelligent demand analysis system that forecasts future product sales, identifies demand trends, segments products based on sales behavior, and generates GenAI-powered business recommendations. The system helps support better inventory planning, replenishment strategy, and demand understanding across Walmart stores and product categories.
