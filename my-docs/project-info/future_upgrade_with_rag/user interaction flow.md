# 🔄 Example Flow 1 — Demand Forecasting + Trend Analysis

---

# 👤 User Query

```text id="4j80wo"
How much demand is expected for FOODS_3_090 in Store CA_1 next week?
```

---

# 🧠 Step 1 — Agent Understands Intent

## Input

```text id="0cfrpj"
User question:
How much demand is expected for FOODS_3_090 in Store CA_1 next week?
```

---

## Agent identifies

```text id="vrjlwm"
Intent:
Demand Forecasting + Trend Analysis
```

---

## Output

```text id="u4vs6n"
Required data:
- Historical sales
- Calendar events
- Forecast output
- Recent sales trend
```

---

# 📦 Step 2 — Structured Retrieval (RAG Retrieval Layer)

The system retrieves relevant rows from M5 tables.

---

## Input

```text id="6x5xq8"
item_id = FOODS_3_090
store_id = CA_1
forecast horizon = next 7 days
```

---

## Retrieved data

```text id="8q5h6k"
Past 28-day sales:
[42, 44, 39, 50, 53, 60, ...]

Last 7-day average sales = 58
Last 28-day average sales = 46

Upcoming event:
Weekend + Holiday
```

---

## Output

```text id="kqlgqb"
Recent demand trend = Increasing
Sales momentum = Strong
```

---

# 🤖 Step 3 — Forecasting Model

Forecast model predicts future demand.

---

## Input features

```text id="ujqpm7"
- Lag sales features
- Rolling averages
- Day-of-week
- Event indicators
- Trend features
```

---

## Model output

```text id="p1jmhr"
Next 7-day forecast:
[58, 61, 63, 65, 68, 52, 53]

Total forecast:
420 units
```

---

# 🧩 Step 4 — Demand Segmentation

The segmentation model identifies demand behavior.

---

## Input

```text id="17r6te"
Sales frequency
Demand variability
Trend behavior
Seasonality indicators
```

---

## Output

```text id="25zch6"
Segment = Fast-moving
Trend = Increasing
Demand volatility = Medium
```

---

# ✨ Step 5 — GenAI Explanation Layer

GenAI receives structured outputs.

---

## Input to GenAI

```text id="h5wejlwm"
Forecast = 420 units
Segment = Fast-moving
Trend = Increasing
Upcoming holiday = Yes
```

---

## Final User Response

> FOODS_3_090 in Store CA_1 is expected to sell approximately 420 units next week.
> The product belongs to the fast-moving category and has shown an increasing sales trend over recent weeks. Demand is expected to rise further due to the upcoming holiday and weekend sales pattern.
>
> Recommendation: Prepare additional inventory allocation for the coming week.

---

# 🔄 Full Pipeline View

```text id="7mlz0k"
User Question
      ↓
Agent identifies intent
      ↓
Structured retrieval from M5 tables
      ↓
Forecasting model predicts demand
      ↓
Trend + segmentation analysis
      ↓
GenAI explanation generation
      ↓
Business recommendation
```

---

# 🔄 Example Flow 2 — Demand Segmentation + Business Recommendation

---

# 👤 User Query

```text id="3fh1wf"
Which products show seasonal demand behavior in Store TX_2?
```

---

# 🧠 Step 1 — Agent Understands Intent

---

## Input

```text id="x0w8rt"
User question:
Which products show seasonal demand behavior in Store TX_2?
```

---

## Agent identifies

```text id="6hwwry"
Intent:
Demand Segmentation + Pattern Analysis
```

---

## Output

```text id="c4znlx"
Required data:
- Historical sales patterns
- Seasonality indicators
- Product segmentation output
```

---

# 📦 Step 2 — Structured Retrieval

The system retrieves historical sales behavior.

---

## Input

```text id="5kj3ol"
store_id = TX_2
```

---

## Retrieved data

```text id="jlwmn6"
SKU HOUSEHOLD_1_210:
High sales during summer months

SKU HOBBIES_1_055:
Demand spikes during holidays

SKU FOODS_2_300:
Stable demand throughout year
```

---

## Output

```text id="3y9g90"
Seasonality patterns identified
```

---

# 🧠 Step 3 — Segmentation Model

Model groups products by demand behavior.

---

## Input features

```text id="mjlwmz"
- Monthly sales variation
- Seasonal peaks
- Demand consistency
- Frequency of sales
```

---

## Model output

```text id="nztc1d"
HOUSEHOLD_1_210 → Seasonal
HOBBIES_1_055 → Seasonal
FOODS_2_300 → Stable
```

---

# 📈 Step 4 — Trend Analysis

System analyzes recent movement.

---

## Output

```text id="2z6k4t"
HOUSEHOLD_1_210:
Trend = Increasing

HOBBIES_1_055:
Trend = Stable
```

---

# ✨ Step 5 — GenAI Business Explanation

---

## Input to GenAI

```text id="jlwmwf"
Product = HOUSEHOLD_1_210
Segment = Seasonal
Trend = Increasing
Upcoming season = Summer
```

---

## Final User Response

> HOUSEHOLD_1_210 and HOBBIES_1_055 show strong seasonal demand behavior in Store TX_2.
>
> HOUSEHOLD_1_210 currently shows an increasing demand trend as summer approaches, indicating a likely seasonal demand spike in the coming weeks.
>
> Recommendation: Increase inventory planning before peak seasonal demand periods.

---

# 🔄 Full Pipeline View

```text id="u0jcvn"
User Question
      ↓
Agent identifies intent
      ↓
Retrieve historical sales behavior
      ↓
Segmentation model identifies product type
      ↓
Trend analysis performed
      ↓
GenAI generates explanation
      ↓
Business recommendation
```
