You can implement **Demand Segmentation on the Walmart M5 dataset** by converting each product-store time series into demand-behavior features, then clustering similar products.

The goal is:

> “Group SKUs that behave similarly so Walmart can apply different inventory strategies to each group.”

---

# 1) What data from M5 is needed?

For demand segmentation, you mainly need:

### Main file

```text
sales_train_validation.csv
```

This contains daily unit sales for each product-store combination.

Example structure:

```text
id, item_id, dept_id, cat_id, store_id, state_id, d_1, d_2, ..., d_1913
```

Each row is one product-store time series.

Example:

```text
FOODS_3_001_CA_1_validation
```

means product `FOODS_3_001` sold in store `CA_1`.

---

### Optional file

```text
calendar.csv
```

This helps identify:

* weekdays
* weekends
* events
* months
* seasons

For basic demand segmentation, you can start with only sales data.
For better segmentation, use calendar features too.

---

# 2) What is one “object” to segment?

Each row in `sales_train_validation.csv` can be treated as one object:

```text
SKU + Store combination
```

Example:

```text
FOODS_3_001_CA_1
FOODS_3_001_TX_1
HOUSEHOLD_1_020_CA_3
```

This is useful because the same product may behave differently in different stores.

So your segmentation unit is:

```text
item_id + store_id
```

---

# 3) Convert sales history into useful features

Clustering cannot directly understand 1,913 daily sales columns easily.

So first, convert the sales series into summary features.

For each product-store row, calculate features like:

---

## A) Average demand

```text
avg_sales = average daily sales
```

Business meaning:

> How much does this product usually sell?

High value means fast-moving product.
Low value means slow-moving product.

---

## B) Demand variability

```text
std_sales = standard deviation of daily sales
cv_sales = std_sales / avg_sales
```

Business meaning:

> Is demand stable or unpredictable?

Low variation means stable product.
High variation means volatile product.

---

## C) Zero-sales frequency

```text
zero_sales_ratio = number of zero-sales days / total days
```

Business meaning:

> How often does this product not sell at all?

High value means intermittent product.

Example:

```text
zero_sales_ratio = 0.80
```

means the product had zero sales on 80% of days.

---

## D) Recent demand trend

Compare recent sales with older sales.

Example:

```text
recent_avg_28 = average sales in last 28 days
older_avg = average sales before last 28 days
trend_ratio = recent_avg_28 / older_avg
```

Business meaning:

> Is demand increasing, decreasing, or stable?

---

## E) Weekend effect

Using `calendar.csv`, identify weekends and weekdays.

```text
weekend_avg_sales
weekday_avg_sales
weekend_ratio = weekend_avg_sales / weekday_avg_sales
```

Business meaning:

> Does demand increase on weekends?

---

## F) Event or holiday effect

Using `calendar.csv`, compare event days vs normal days.

```text
event_avg_sales
normal_avg_sales
event_ratio = event_avg_sales / normal_avg_sales
```

Business meaning:

> Does this product sell more during events or holidays?

---

## G) Seasonality behavior

You can calculate monthly average sales.

Example:

```text
jan_avg_sales
feb_avg_sales
...
dec_avg_sales
```

Or calculate a seasonality score:

```text
seasonality_score = max_month_avg / overall_avg
```

Business meaning:

> Does demand spike in certain months?

---

# 4) Final feature table

After feature engineering, your dataset should look like this:

| id                   | avg_sales | cv_sales | zero_sales_ratio | trend_ratio | weekend_ratio | event_ratio | seasonality_score |
| -------------------- | --------: | -------: | ---------------: | ----------: | ------------: | ----------: | ----------------: |
| FOODS_3_001_CA_1     |       3.2 |     0.45 |             0.10 |        1.08 |          1.20 |        1.35 |              1.50 |
| HOBBIES_1_050_TX_2   |       0.2 |     2.80 |             0.92 |        0.70 |          1.00 |        1.10 |              1.20 |
| HOUSEHOLD_2_100_WI_1 |       1.5 |     1.40 |             0.40 |        1.50 |          1.10 |        2.20 |              2.80 |

This table is what you give to the clustering algorithm.

---

# 5) Scale the features

Before clustering, apply scaling.

This is important because features have different ranges.

Example:

```text
avg_sales may range from 0 to 100
zero_sales_ratio ranges from 0 to 1
cv_sales may range from 0 to 10
```

Use:

```python
StandardScaler()
```

or

```python
RobustScaler()
```

For M5, `RobustScaler` can be useful because sales can have outliers.

---

# 6) Apply clustering

You can start with **K-Means clustering**.

Example:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
```

Start with 4 clusters because your business segments may be:

```text
Cluster 0 → Stable products
Cluster 1 → Seasonal products
Cluster 2 → Intermittent products
Cluster 3 → Fast-moving products
```

---

# 7) Choose the right number of clusters

Do not blindly choose 4.

Try multiple values:

```text
k = 3, 4, 5, 6, 7
```

Evaluate using:

```text
Elbow method
Silhouette score
Business interpretability
```

The most important part is not only statistical score.

The clusters should make business sense.

---

# 8) Interpret the clusters

After clustering, calculate average feature values per cluster.

Example:

| Cluster | avg_sales | zero_sales_ratio | cv_sales | seasonality_score | Interpretation |
| ------- | --------: | ---------------: | -------: | ----------------: | -------------- |
| 0       |      High |              Low |      Low |            Medium | Fast-moving    |
| 1       |    Medium |              Low |      Low |               Low | Stable         |
| 2       |       Low |             High |     High |               Low | Intermittent   |
| 3       |    Medium |           Medium |     High |              High | Seasonal       |

Then assign business-friendly labels.

---

# 9) Example business cluster definitions

## Cluster 1: Stable Products

Feature pattern:

```text
avg_sales = medium
zero_sales_ratio = low
cv_sales = low
seasonality_score = low
```

Meaning:

> These products sell regularly and predictably.

Business action:

```text
Maintain regular stock.
Use simple replenishment rules.
```

---

## Cluster 2: Fast-Moving Products

Feature pattern:

```text
avg_sales = high
zero_sales_ratio = very low
recent_avg = high
```

Meaning:

> These products sell frequently and need continuous monitoring.

Business action:

```text
Replenish often.
Keep higher safety stock.
Monitor stockout risk closely.
```

---

## Cluster 3: Intermittent Products

Feature pattern:

```text
avg_sales = low
zero_sales_ratio = high
cv_sales = high
```

Meaning:

> These products sell rarely and irregularly.

Business action:

```text
Avoid overstocking.
Use smaller reorder quantities.
Forecast carefully.
```

---

## Cluster 4: Seasonal Products

Feature pattern:

```text
seasonality_score = high
event_ratio = high
cv_sales = high
```

Meaning:

> These products show demand spikes in specific months or events.

Business action:

```text
Increase stock before seasonal peaks.
Reduce stock after the season ends.
```

---

# 10) Simple Python implementation outline

```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
sales = pd.read_csv("sales_train_validation.csv")
calendar = pd.read_csv("calendar.csv")

# Daily sales columns
day_cols = [col for col in sales.columns if col.startswith("d_")]

# Basic demand features
features = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()

features["avg_sales"] = sales[day_cols].mean(axis=1)
features["std_sales"] = sales[day_cols].std(axis=1)
features["zero_sales_ratio"] = (sales[day_cols] == 0).mean(axis=1)

features["cv_sales"] = features["std_sales"] / (features["avg_sales"] + 1e-6)

# Recent trend feature
last_28_cols = day_cols[-28:]
older_cols = day_cols[:-28]

features["recent_avg_28"] = sales[last_28_cols].mean(axis=1)
features["older_avg"] = sales[older_cols].mean(axis=1)

features["trend_ratio"] = features["recent_avg_28"] / (features["older_avg"] + 1e-6)

# Seasonality score using monthly data
calendar_days = calendar[["d", "month", "event_name_1"]]

sales_long = sales[["id"] + day_cols].melt(
    id_vars="id",
    var_name="d",
    value_name="sales"
)

sales_long = sales_long.merge(calendar_days, on="d", how="left")

monthly_sales = (
    sales_long
    .groupby(["id", "month"])["sales"]
    .mean()
    .reset_index()
)

monthly_pivot = monthly_sales.pivot(
    index="id",
    columns="month",
    values="sales"
).fillna(0)

monthly_pivot["max_month_avg"] = monthly_pivot.max(axis=1)
monthly_pivot["overall_month_avg"] = monthly_pivot.mean(axis=1)

monthly_pivot["seasonality_score"] = (
    monthly_pivot["max_month_avg"] / 
    (monthly_pivot["overall_month_avg"] + 1e-6)
)

features = features.merge(
    monthly_pivot[["seasonality_score"]],
    left_on="id",
    right_index=True,
    how="left"
)

# Select clustering features
cluster_cols = [
    "avg_sales",
    "cv_sales",
    "zero_sales_ratio",
    "trend_ratio",
    "seasonality_score"
]

X = features[cluster_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Try different k values
for k in range(3, 8):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}, silhouette={score:.3f}")

# Final model
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
features["cluster"] = kmeans.fit_predict(X_scaled)

# Interpret clusters
cluster_summary = features.groupby("cluster")[cluster_cols].mean()
print(cluster_summary)
```

---

# 11) Add business labels to clusters

After seeing the cluster summary, manually map cluster numbers to business names.

Example:

```python
cluster_name_map = {
    0: "Stable Products",
    1: "Fast-Moving Products",
    2: "Intermittent Products",
    3: "Seasonal Products"
}

features["segment"] = features["cluster"].map(cluster_name_map)
```

Important: cluster numbers may change every time depending on the data and model.
So do not assume:

```text
Cluster 0 = Stable
```

Always check the cluster summary first.

---

# 12) Final output table

Your final demand segmentation output can look like this:

| id                   | item_id         | store_id | avg_sales | zero_sales_ratio | seasonality_score | segment               |
| -------------------- | --------------- | -------- | --------: | ---------------: | ----------------: | --------------------- |
| FOODS_3_001_CA_1     | FOODS_3_001     | CA_1     |       3.2 |             0.08 |               1.3 | Stable Products       |
| HOBBIES_1_050_TX_2   | HOBBIES_1_050   | TX_2     |       0.1 |             0.95 |               1.1 | Intermittent Products |
| HOUSEHOLD_2_100_WI_1 | HOUSEHOLD_2_100 | WI_1     |       1.8 |             0.30 |               3.2 | Seasonal Products     |

---

# 13) How GenAI can use this output

Once each product has a segment, GenAI can convert it into plain English.

Example input to GenAI:

```text
Product: HOUSEHOLD_2_100
Store: WI_1
Segment: Seasonal Products
Average Sales: 1.8 units/day
Seasonality Score: 3.2
Stockout Risk: High
```

Example GenAI output:

> “This product shows strong seasonal demand and is currently at high risk of stockout. Walmart should increase replenishment before the seasonal peak and monitor inventory closely during the demand period.”

---

# 14) Recommended project flow

```text
Load M5 sales data
        ↓
Create demand behavior features
        ↓
Scale features
        ↓
Apply clustering
        ↓
Analyze cluster summaries
        ↓
Assign business segment labels
        ↓
Connect segment with inventory strategy
        ↓
Use GenAI to explain recommendations
```

---

# Final takeaway

For the M5 dataset, demand segmentation is implemented by treating each `item_id + store_id` as one demand series, creating features such as average sales, zero-sales ratio, demand variability, trend, and seasonality, and then clustering those products into meaningful business groups like:

```text
Stable Products
Fast-Moving Products
Intermittent Products
Seasonal Products
```

This segmentation helps Walmart decide **which products need regular stock, seasonal preparation, limited inventory, or frequent replenishment**.
