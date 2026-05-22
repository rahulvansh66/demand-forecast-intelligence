Here’s a practical **EDA framework for any ML dataset**, including datasets with **time-series features**. Think of EDA as answering:

> “What is this data, can I trust it, what patterns exist, and what risks could hurt my model?”

---

# 1. Understand the problem first

Before touching the data, clarify the ML objective.

Ask:

* What is the target variable?
* Is it a classification, regression, forecasting, ranking, anomaly detection, or clustering problem?
* What does one row represent?
* At prediction time, what information will actually be available?
* What is the business definition of success?
* What is the evaluation metric?

This step prevents **data leakage**, which is one of the biggest EDA mistakes.

Example:

If you are predicting customer churn, any column created after the churn date should not be used.

## 📋 Application to Demand Forecasting Project

**Our Specific Objectives:**

1. **Demand Forecasting Model:**
   - Target variable: Daily unit sales for next 28 days (d_1914 to d_1941)
   - Problem type: Time-series forecasting (regression)
   - What one row represents: One item-store combination's daily sales history
   - Available at prediction time: Historical sales (d_1 to d_1913), calendar features, pricing data
   - Business success: Accurate inventory planning and reduced stockouts/overstock
   - Evaluation metrics: RMSE, MAE, WRMSSE (Walmart's weighted metric)

2. **Product Segmentation Model:**
   - Target: 5 behavioral segments (High Demand Stable, High Demand Volatile, Low Demand Intermittent, Declining, Seasonal)
   - Problem type: Clustering/unsupervised learning
   - Business success: Actionable product categorization for differentiated inventory strategies
   - Evaluation: Silhouette score, business interpretability, segment stability

**Critical Leakage Prevention:**
- Never use future sales (d_1914+) for forecasting training
- Pricing data must be time-aligned (weekly pricing for corresponding sales periods)
- Event/holiday data must reflect knowledge available at prediction time

---

# 2. Inspect the dataset structure

Start with a high-level data audit.

Check:

* Number of rows and columns
* Column names and meanings
* Data types
* Unique identifiers
* Target column
* Date/time columns
* Numerical columns
* Categorical columns
* Text columns
* Boolean/binary columns
* Duplicate rows
* Duplicate IDs

Your goal is to build a mental map of the dataset.

A useful way to think:

| Question                   | Why it matters                  |
| -------------------------- | ------------------------------- |
| What is one row?           | Defines the unit of prediction  |
| What is the target?        | Defines the ML task             |
| Are there IDs?             | Helps detect duplicates/leakage |
| Are there dates?           | Changes validation strategy     |
| Are columns wrongly typed? | Prevents bad analysis           |

## 📋 Application to Demand Forecasting Project

**Expected Dataset Structure:**

1. **sales_train_validation.csv** (30,490 rows × ~1,919 columns):
   - Row unit: One item_id + store_id combination
   - Target columns: d_1 to d_1913 (daily unit sales, integers)
   - Identifiers: `id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`
   - No missing target values expected (zeros indicate no sales)

2. **calendar.csv** (1,969 rows × 14 columns):
   - Key date mappings: `d` ↔ `date` (d_1 = 2011-01-29)
   - Temporal features: `weekday`, `month`, `year`, `wm_yr_wk`
   - External factors: `event_name_1/2`, `event_type_1/2`, `snap_CA/TX/WI`

3. **sell_prices.csv** (~6.8M rows × 4 columns):
   - Sparse weekly pricing: `store_id`, `item_id`, `wm_yr_wk`, `sell_price`
   - Missing combinations = item not sold at that store-week

**Key Validations to Perform:**
- Verify 30,490 = 3,049 items × 10 stores exactly
- Confirm date sequence d_1 to d_1969 is complete
- Check hierarchical consistency: cat_id → dept_id → item_id
- Validate wm_yr_wk alignment between calendar and pricing tables

---

# 3. Check data quality

This is the “can I trust this dataset?” stage.

Look for:

* Missing values
* Duplicates
* Impossible values
* Inconsistent categories
* Wrong data types
* Outliers
* Constant columns
* High-cardinality categorical columns
* Mixed units
* Invalid dates
* Negative values where impossible
* Future dates where impossible

Examples:

* Age = -5
* Price = 0 when price should be positive
* Gender values: `Male`, `male`, `M`, `MALE`
* Date of purchase after cancellation date
* Same customer ID appearing multiple times unexpectedly

For every issue, decide whether it is:

1. Real signal
2. Data entry error
3. Missingness pattern
4. Leakage
5. Something to clean later

## 📋 Application to Demand Forecasting Project

**Specific Quality Checks for M5 Data:**

**Sales Data Quality:**
- **Negative sales**: Should not exist (unit sales ≥ 0)
- **Missing sales values**: Should not exist in d_1 to d_1913 columns
- **Extreme outliers**: Sales > 1000 units/day may indicate data errors or bulk purchases
- **Constant zero series**: Items never sold (valid but may need special handling)
- **ID consistency**: Each `id` should be unique, format should be “{item_id}_{store_id}_validation”

**Calendar Data Quality:**
- **Date continuity**: No missing dates between 2011-01-29 and 2016-06-19
- **Weekend patterns**: Verify Saturday=1, Sunday=2 in wday encoding
- **Event duplicates**: Same event on multiple days (check if intentional)
- **SNAP alignment**: SNAP benefits should align with known state schedules

**Pricing Data Quality:**
- **Zero prices**: Suspicious unless item is free/promotional
- **Price jumps**: >50% price changes week-over-week may indicate errors
- **Missing price periods**: Expected for seasonal items, concerning for staples
- **Negative prices**: Impossible, indicates data corruption

**Cross-table Consistency:**
- **Item coverage**: All sales table item_ids should appear in pricing table
- **Store coverage**: All sales table store_ids should appear in pricing table
- **Week alignment**: calendar.wm_yr_wk should fully cover sell_prices.wm_yr_wk range

---

# 4. Analyze the target variable

This is one of the most important EDA steps.

For classification:

Check:

* Class balance
* Majority/minority class ratio
* Rare classes
* Target distribution across time
* Target distribution across important groups

Example:

If 95% of users did not churn and 5% churned, accuracy is a misleading metric.

For regression:

Check:

* Distribution of target
* Skewness
* Outliers
* Zero-heavy target
* Negative values
* Long tails
* Need for transformation

Example:

House prices are often right-skewed, so log-transforming the target may help.

For time-series forecasting:

Check:

* Trend
* Seasonality
* Cycles
* Sudden jumps
* Missing time periods
* Level shifts
* Outliers
* Forecast horizon

## 📋 Application to Demand Forecasting Project

**Daily Sales Target Analysis (d_1 to d_1913):**

**Distribution Characteristics:**
- **Zero-inflation**: Expect high proportion of zero sales days (intermittent demand)
- **Right-skewness**: Most days have low sales, few days have high sales
- **Discrete values**: Unit sales are integers (0, 1, 2, 3...)
- **Outliers**: Identify bulk purchase days, promotional spikes, seasonal surges

**Time-Series Patterns by Product Category:**
- **FOODS**: Daily consumption, stable with weekly patterns
- **HOUSEHOLD**: Less frequent, more intermittent purchases  
- **HOBBIES**: Seasonal patterns, weekend/holiday spikes

**Critical Time-Series Checks:**
- **Weekly seasonality**: Saturday/Sunday vs. weekday patterns
- **Monthly patterns**: End-of-month, payday effects
- **Holiday effects**: Christmas, Thanksgiving, Black Friday spikes
- **SNAP benefit cycles**: Monthly food category spikes aligned with benefit issuance
- **Long-term trends**: Growth/decline over 2011-2016 period
- **Structural breaks**: Identify sudden level shifts (store remodels, competition)

**Segmentation Implications:**
- **High Demand Stable**: Low coefficient of variation, consistent non-zero sales
- **High Demand Volatile**: High volume but high CV, difficult to predict spikes
- **Low Demand Intermittent**: Many zeros, sporadic sales events
- **Declining Products**: Negative trend over time, decreasing volume
- **Seasonal Products**: Clear seasonal cycles, concentrated selling periods

**Forecasting Challenges to Identify:**
- Items with <10% non-zero days (pure intermittent demand)
- Items with >90% zero days in final 90 days (declining products)
- Items with clear seasonality but insufficient historical cycles

---

# 5. Analyze individual features

Now examine each feature by itself.

For numerical features:

Check:

* Mean, median, min, max
* Standard deviation
* Skewness
* Outliers
* Distribution shape
* Zero inflation
* Negative values
* Natural bounds

Ask:

* Is this feature normally distributed?
* Is it highly skewed?
* Are there extreme values?
* Does zero have special meaning?
* Should this feature be transformed, capped, bucketed, or scaled?

For categorical features:

Check:

* Number of unique values
* Most frequent categories
* Rare categories
* Missing category
* Inconsistent labels
* Cardinality

Ask:

* Are there too many categories?
* Are some categories very rare?
* Should rare categories be grouped?
* Is this actually an ID column?
* Could this cause overfitting?

For datetime features:

Extract meaning from:

* Year
* Month
* Day
* Day of week
* Weekend/weekday
* Hour
* Quarter
* Holiday indicator
* Time since previous event
* Time until next event
* Customer/account age

But always ask:

> Would this be known at prediction time?

## 📋 Application to Demand Forecasting Project

**Key Features to Analyze:**

**Categorical Features:**
- **item_id** (3,049 unique): Product identifier - high cardinality, potential for embeddings
- **dept_id** (7 unique): Department grouping - medium cardinality, useful for hierarchical models
- **cat_id** (3 unique): Category level - low cardinality (FOODS, HOUSEHOLD, HOBBIES)
- **store_id** (10 unique): Store identifier - low cardinality, geographic effects
- **state_id** (3 unique): State level - very low cardinality (CA, TX, WI)

**Temporal Features (from calendar.csv):**
- **weekday/wday**: Day-of-week effects (retail seasonality)
- **month**: Monthly patterns, seasonal cycles
- **year**: Long-term trends, year-over-year growth
- **event_name_1/2**: Holiday impacts on demand
- **event_type_1/2**: Cultural/National/Religious/Sporting event classifications

**External Factor Features:**
- **snap_CA/TX/WI**: SNAP benefit timing effects (especially for FOODS category)
- **sell_price**: Weekly pricing data - critical for price elasticity analysis

**Engineered Feature Opportunities:**
- **Days since last event**: Time-based event effects
- **Holiday proximity**: Days before/after major holidays
- **Price change indicators**: Week-over-week price changes
- **Seasonal flags**: Quarter indicators, holiday seasons
- **Store-level aggregations**: Store performance metrics
- **Category velocity**: Moving averages by category/department

**Critical Feature Engineering Principles:**
- All temporal features must use information available at prediction time
- Price features should use latest available pricing (not future prices)
- Event features should include lead/lag effects (pre-holiday shopping)
- SNAP timing is known in advance, can be used as future features

---

# 6. Study feature-target relationships

This is where EDA becomes useful for modeling.

For numerical feature vs target:

For classification:

* Compare feature distribution by class
* Check whether feature values separate classes
* Look at median/mean by class

For regression:

* Check relationship between feature and target
* Look for linear/nonlinear patterns
* Check if target changes with feature bins

For categorical feature vs target:

For classification:

* Target rate by category
* Category frequency
* Rare category behavior

For regression:

* Average target by category
* Median target by category
* Variability within category

Ask:

* Which features seem predictive?
* Are some relationships nonlinear?
* Are some features only useful after transformation?
* Are some categories risky because they have very few rows?
* Are there suspiciously perfect predictors?

Suspiciously perfect predictors may indicate leakage.

## 📋 Application to Demand Forecasting Project

**Key Relationships to Investigate:**

**Categorical Feature vs. Sales Patterns:**
- **Category analysis**: Compare average daily sales across FOODS/HOUSEHOLD/HOBBIES
- **Department effects**: Sales distribution by dept_id within each category  
- **Store performance**: Average sales by store_id and state_id
- **Temporal patterns**: Sales by weekday, month, year trends

**External Factor Impact Analysis:**
- **Event effects**: Sales lift during holidays (Christmas, Thanksgiving, Super Bowl)
- **SNAP benefit impact**: Food category sales spikes on SNAP issuance days
- **Price sensitivity**: Sales volume changes with price changes (elasticity analysis)
- **Seasonal product behavior**: Holiday decorations, summer items, back-to-school products

**Critical Predictive Relationships:**
- **Historical sales patterns**: Recent sales trend as predictor of future sales
- **Cross-product cannibalization**: Substitution effects within departments
- **Store catchment effects**: Similar performance across stores in same state
- **Price-volume elasticity**: Different categories have different price sensitivities

**Segmentation-Relevant Patterns:**
- **Stable vs. Volatile identification**: Coefficient of variation analysis
- **Intermittent detection**: Percentage of zero-sales days
- **Seasonal detection**: Autocorrelation at 7-day, 30-day, 365-day lags
- **Declining product identification**: Negative trend slopes over time windows

**Warning Signs to Investigate:**
- Perfect correlation between sales and engineered features (potential leakage)
- Suspiciously high predictive power from single categorical features
- Price effects that seem too strong (data quality issues)
- Event effects that don't match business intuition

---

# 7. Study feature-feature relationships

This step helps detect redundancy, multicollinearity, and hidden structure.

Check:

* Correlation between numerical features
* Highly similar columns
* Feature groups
* Categorical associations
* Repeated information
* Derived columns
* Columns that encode the same thing

Examples:

* `total_amount = item_price × quantity`
* `age` and `date_of_birth`
* `signup_year` and `account_age`
* `city` and `postal_code`

Ask:

* Are some features redundant?
* Are there derived variables?
* Could one column leak another?
* Are highly correlated variables acceptable for my model type?

Tree models can tolerate correlation better than linear models, but correlation can still affect interpretation.

## 📋 Application to Demand Forecasting Project

**Expected Feature Redundancies in M5 Dataset:**

**Hierarchical Relationships (Expected Correlation):**
- **Geographic hierarchy**: state_id → store_id (TX → TX_1, TX_2, TX_3)
- **Product hierarchy**: cat_id → dept_id → item_id (FOODS → FOODS_1 → FOODS_1_001)
- **Temporal relationships**: date ↔ d column ↔ wm_yr_wk (same time information)

**Derived Information to Check:**
- **Calendar features**: weekday, month, year all derived from date
- **Event indicators**: event_name_1 and event_type_1 encode same information differently
- **SNAP state alignment**: snap_CA/TX/WI should align with state_id geographic boundaries

**Potential Multicollinearity Issues:**
- **Store clustering**: Stores in same state may have highly correlated sales patterns
- **Product substitution**: Items in same department may show negative correlation
- **Seasonal products**: Holiday items may have perfect correlation with holiday indicators
- **Price-driven correlation**: Items with coordinated pricing strategies

**Beneficial Correlations (Keep):**
- **Geographic store similarity**: Useful for hierarchical modeling
- **Category seasonality**: Expected and informative for forecasting
- **Cross-product effects**: Valuable for understanding demand substitution

**Problematic Correlations (Investigate):**
- **Perfect price-sales correlation**: May indicate data leakage or quality issues
- **Identical time series**: Suggests duplicate products or data errors
- **Suspiciously high cross-store correlation**: May indicate data copying errors

---

# 8. Special time-series EDA

If your dataset has time, add this layer.

## A. Understand the time structure

Ask:

* What is the time column?
* Is time regular or irregular?
* What is the frequency: hourly, daily, weekly, monthly?
* Are there missing timestamps?
* Are there multiple entities over time?

Examples:

* One global time series: daily sales
* Panel time series: daily sales per store
* Event data: transactions by customer
* Snapshot data: customer status every month

## B. Plot target over time mentally

Look for:

* Trend
* Seasonality
* Cycles
* Sudden drops/spikes
* Structural breaks
* Holidays/events
* Regime changes

Examples:

* Sales increase every December
* Traffic drops on weekends
* Demand changes after a policy update
* Data collection changed after a certain date

## C. Check time-based missingness

Ask:

* Are some dates missing?
* Are weekends absent?
* Are there gaps?
* Are all entities observed at the same frequency?
* Does missingness itself contain signal?

## D. Check leakage through time

This is critical.

Avoid using:

* Future values
* Aggregates calculated using future data
* Labels encoded into features
* Rolling statistics that include the current/future target
* Random train-test split for time-dependent problems

Bad:

> Using future sales to predict current sales.

Good:

> Using past 7-day sales average to predict tomorrow’s sales.

## E. Create time-aware validation thinking

For time-series or time-dependent data, prefer:

* Train on past, validate on future
* Rolling validation
* Expanding window validation
* Group-time split if multiple entities exist

Do not randomly split if time order matters.

## 📋 Application to Demand Forecasting Project

**M5 Time Structure Analysis:**

**A. Time Structure Understanding:**
- **Time column**: d_1, d_2, ..., d_1969 (daily frequency)
- **Time range**: January 29, 2011 to June 19, 2016 (1,969 consecutive days)
- **Entity structure**: Panel time series (30,490 item-store combinations × 1,969 days)
- **No missing timestamps**: Complete daily coverage expected
- **Regular frequency**: Daily observations for all entities

**B. Expected Temporal Patterns:**
- **Weekly seasonality**: Strong patterns for retail (weekend vs. weekday effects)
- **Monthly patterns**: SNAP benefit cycles, end-of-month effects
- **Annual seasonality**: Holiday seasons, back-to-school, summer patterns
- **Long-term trends**: 5+ year growth/decline patterns
- **Event-driven spikes**: Black Friday, Christmas, Valentine's Day
- **Structural breaks**: Economic events (2011 recession recovery), store changes

**C. Time-Based Missingness Patterns:**
- **Pricing gaps**: Items not sold in certain weeks (seasonal discontinuation)
- **New product introductions**: Items appearing mid-timeline
- **Store opening/closing**: Rare but possible mid-timeline changes
- **Holiday patterns**: Some stores closed on certain days

**D. Critical Leakage Prevention:**
- **Training period**: d_1 to d_1913 (Jan 29, 2011 - May 22, 2016)
- **Validation period**: d_1914 to d_1941 (May 23, 2016 - June 19, 2016)
- **Feature engineering**: Only use lags, not future values
- **Price alignment**: Use pricing data up to prediction date only
- **Event features**: Use known future events (holidays), not discovered patterns

**E. Validation Strategy:**
- **Time-based split**: Train on d_1 to d_1913, validate on d_1914 to d_1941
- **Walk-forward validation**: Rolling 28-day prediction windows
- **No random splitting**: Preserve temporal order across all item-store series
- **Group consistency**: Keep all stores for same item together in splits

**Key Time-Series EDA Priorities:**
1. **Seasonality detection**: Weekly, monthly, quarterly patterns by category
2. **Trend analysis**: Growth/decline rates by product segments
3. **Event impact quantification**: Holiday lift factors by category
4. **Intermittency patterns**: Zero-sales frequency and clustering
5. **Cross-series correlation**: Store and product similarity patterns

---

# 9. Analyze missing values deeply

Do not just count missing values. Understand why they are missing.

Ask:

* Is missingness random?
* Is it related to the target?
* Is it related to time?
* Is it related to a group?
* Does missing mean “not applicable”?
* Should missingness itself become a feature?

Types of missingness:

| Type               | Example                                             |
| ------------------ | --------------------------------------------------- |
| Random missing     | Sensor failed randomly                              |
| Systematic missing | Income missing mostly for self-employed users       |
| Not applicable     | `car_model` missing because user does not own a car |
| Future unavailable | Feature missing because event has not happened yet  |

A missing value can be highly predictive.

## 📋 Application to Demand Forecasting Project

**Expected Missing Patterns in M5 Dataset:**

**Sales Data Missingness:**
- **Zero vs. missing**: Sales data should have no missing values, only zeros (which is informative)
- **Structural zeros**: Items not sold at specific stores (geographic/demographic mismatch)
- **Seasonal absence**: Holiday decorations missing sales in off-seasons

**Pricing Data Missingness:**
- **Item-store unavailability**: Missing sell_price combinations indicate item not sold at that store-week
- **Seasonal pricing gaps**: Items with seasonal availability (e.g., holiday items)
- **New product introduction**: Pricing starts mid-timeline for new products
- **Discontinued products**: Pricing ends before timeline end

**Calendar Data Completeness:**
- **No missingness expected**: Complete temporal coverage required
- **Event sparsity**: Most days have no events (null event_name_1/2 is normal)
- **SNAP benefit patterns**: State-specific timing, some states may have gaps

**Informative Missingness Patterns:**
- **Missing pricing as feature**: Indicates product unavailability, useful predictor
- **Event absence**: Non-holiday days are as informative as holiday days
- **Geographic availability**: Missing item-store combinations reveal market strategy

**Missing Value Treatment Strategy:**
- **Sales zeros**: Keep as zero (real information about no demand)
- **Missing prices**: Forward-fill or indicate "not available" as category
- **Missing events**: Explicitly encode "no event" as separate category
- **Missing SNAP**: Default to "no benefits issued" for missing state-days

---

# 10. Identify outliers and anomalies

Outliers are not always bad. They may be the most important part of the problem.

Check:

* Extreme numerical values
* Sudden time-series spikes
* Rare categories
* Unusual combinations
* Impossible records
* Duplicated events
* Abnormal target values

Ask:

* Is this a real rare case?
* Is this data error?
* Should I cap it?
* Should I transform it?
* Should I create an anomaly flag?
* Should I remove it?

For fraud, failure prediction, and anomaly detection, outliers may be the signal.

## 📋 Application to Demand Forecasting Project

**Critical Outlier Analysis for Retail Demand:**

**Sales Volume Outliers:**
- **Promotional spikes**: Black Friday, clearance sales (real signal, keep)
- **Bulk purchases**: Unusually high single-day sales (investigate, may cap)
- **Data entry errors**: Sales > 10,000 units/day (likely error, investigate)
- **Seasonal peaks**: Christmas decorations in December (real signal, keep)

**Pricing Outliers:**
- **Price errors**: $0.01 or $999.99 prices (investigate data quality)
- **Promotional pricing**: Significant temporary discounts (real signal, important)
- **Price jumps**: >200% week-over-week increases (investigate for errors)
- **Markdown patterns**: End-of-season clearance pricing (real signal)

**Time-Series Anomalies:**
- **Sudden demand shifts**: New product launches, viral effects (keep and flag)
- **Supply disruptions**: Sudden drop to zero sales (external factor, flag)
- **Store events**: Remodeling periods, temporary closures (contextual outliers)
- **Competitive responses**: Demand drops due to competitor actions

**Category-Specific Outlier Patterns:**
- **FOODS**: Daily consumption limits suggest outliers > 50 units suspicious
- **HOUSEHOLD**: Infrequent purchase patterns, outliers > 20 units/day rare but possible
- **HOBBIES**: High seasonal variation, holiday spikes can be extreme but valid

**Outlier Treatment Strategy:**
- **Keep promotional outliers**: Black Friday spikes are predictable and valuable
- **Flag supply disruptions**: Create indicator features for zero-demand periods
- **Cap extreme outliers**: Sales > 99.9th percentile may be capped to reduce noise
- **Investigate price anomalies**: Manual review for obvious data errors
- **Preserve seasonality**: Holiday spikes contain important forecasting signals

---

# 11. Check segment-level behavior

Overall patterns can hide subgroup problems.

Analyze key segments:

* Region
* Gender
* Age group
* Device type
* Customer type
* Product category
* Store
* Time period
* Acquisition channel

Ask:

* Does the target behave differently by segment?
* Are some segments underrepresented?
* Are there fairness concerns?
* Are there train-test distribution differences?
* Are some groups too small for reliable conclusions?

Example:

Overall churn may be 10%, but for one region it may be 35%.

## 📋 Application to Demand Forecasting Project

**Critical Segment Analysis for M5 Dataset:**

**Geographic Segments:**
- **State-level patterns**: CA vs. TX vs. WI demand characteristics
- **Store performance**: High/medium/low volume stores within each state
- **Urban vs. regional**: Store location demographics effects on demand
- **SNAP program differences**: State-specific benefit timing and participation rates

**Product Category Segments:**
- **FOODS**: Daily necessity, stable demand, SNAP-sensitive
- **HOUSEHOLD**: Intermittent purchases, promotion-driven, bulk buying
- **HOBBIES**: Seasonal, discretionary spending, event-driven

**Department-Level Analysis:**
- **Within FOODS**: Different consumption patterns (dairy vs. frozen vs. produce)
- **Within HOUSEHOLD**: Different replacement cycles (cleaning vs. paper products)
- **Within HOBBIES**: Different seasonality (toys vs. crafts vs. electronics)

**Temporal Segments:**
- **Seasonal products**: Christmas items, summer goods, back-to-school supplies
- **Lifecycle stages**: New launches vs. mature products vs. declining items
- **Economic sensitivity**: Luxury items vs. necessities during economic stress

**Demand Behavior Segments (Target for Segmentation Model):**
1. **High Demand Stable**: Consistent high volume, low variance
2. **High Demand Volatile**: High volume but unpredictable spikes
3. **Low Demand Intermittent**: Sporadic purchases, many zero days
4. **Declining Products**: Negative trends, phase-out patterns
5. **Seasonal Products**: Clear seasonal cycles, concentrated demand

**Key Segment Questions:**
- Do promotion effects vary by category? (expect stronger for HOUSEHOLD/HOBBIES)
- Are SNAP effects isolated to FOODS category?
- Do store characteristics affect all categories equally?
- Are there regional preference differences (CA vs. TX cultural preferences)?
- Do seasonal patterns vary by geographic region?

---

# 12. Think about feature engineering opportunities

EDA should naturally suggest features.

Common feature ideas:

For numerical data:

* Log transformation
* Binning
* Ratios
* Differences
* Interactions
* Flags for extreme values
* Missing indicators

For categorical data:

* Group rare categories
* Frequency encoding
* Target encoding carefully
* Hierarchical grouping
* Category cleanup

For time-series:

* Lag features
* Rolling averages
* Rolling standard deviation
* Expanding averages
* Time since last event
* Event counts in last N days
* Seasonality features
* Holiday indicators
* Trend features

For customer/event data:

* Recency
* Frequency
* Monetary value
* Tenure
* Activity rate
* Change over time

Always check whether the engineered feature is available at prediction time.

## 📋 Application to Demand Forecasting Project

**Feature Engineering Opportunities from M5 EDA:**

**Lag and Rolling Features:**
- **Sales lags**: Previous 1, 7, 14, 28 days sales for same item-store
- **Rolling averages**: 7-day, 14-day, 28-day moving averages
- **Rolling volatility**: Standard deviation over rolling windows
- **Expanding statistics**: Cumulative averages from timeline start
- **Seasonal lags**: Same weekday previous week, same date previous year

**Price-Related Features:**
- **Price changes**: Week-over-week price change percentages
- **Price relative to category**: Item price vs. category average price
- **Promotional indicators**: Price below historical average flag
- **Price elasticity proxies**: Historical price-volume relationship indicators

**Calendar and Event Features:**
- **Holiday proximity**: Days before/after major holidays (Christmas, Thanksgiving)
- **Event type indicators**: Cultural, National, Religious, Sporting event flags
- **Weekend indicators**: Saturday, Sunday, Friday flags
- **Month/quarter indicators**: One-hot encoded temporal cycles
- **SNAP timing**: Days since last SNAP benefit issuance

**Cross-Product Features:**
- **Category velocity**: Store-level category performance metrics
- **Item ranking**: Sales rank within department/store
- **Store performance**: Store total sales, growth rates
- **Substitution effects**: Correlation-based product similarity scores

**Intermittent Demand Features:**
- **Days since last sale**: Recency of last non-zero sales day
- **Zero-sales run length**: Consecutive days with zero sales
- **Sales frequency**: Percentage of non-zero days in recent history
- **Demand intensity**: Average sales on non-zero days

**Critical Engineering Principles:**
- **No future leakage**: Use only past values for lag/rolling features
- **Consistent time alignment**: Ensure weekly price features align with daily sales
- **Handle start-of-series**: Proper initialization for rolling statistics
- **Seasonal adjustment**: Account for different baseline demand levels

---

# 13. Compare train and test distributions

This step is often ignored but very important.

Check whether train and test differ in:

* Feature distributions
* Target distribution
* Time periods
* Missing values
* Category levels
* New/unseen categories
* Entity overlap
* Data collection rules

Ask:

* Is test data from the future?
* Are there categories in test not seen in train?
* Has the population changed?
* Is there data drift?
* Are train and test split correctly?

If train and test distributions differ a lot, your validation score may not reflect real-world performance.

## 📋 Application to Demand Forecasting Project

**Train-Test Distribution Analysis for M5:**

**Temporal Split Validation:**
- **Training period**: d_1 to d_1913 (Jan 29, 2011 - May 22, 2016)
- **Validation period**: d_1914 to d_1941 (May 23, 2016 - June 19, 2016) 
- **Time gap**: No gap between train and validation (continuous timeline)
- **Seasonal alignment**: Validation period covers late spring/early summer

**Expected Distribution Differences:**
- **Seasonal effects**: Training includes 5+ years of seasonal cycles, validation only covers 28 days
- **Trend continuation**: Economic/demographic trends may continue into validation period
- **New product lifecycle**: Items launched late in training may behave differently in validation
- **Price strategy evolution**: Pricing strategies may change over 5+ year period

**Critical Consistency Checks:**
- **Item-store coverage**: All validation period item-store combinations should exist in training
- **Category balance**: Same product categories available in both periods
- **Store operations**: No store closures/openings between train and validation
- **Data collection**: Same measurement methodology throughout entire period

**Potential Distribution Drift Concerns:**
- **Economic conditions**: 2011 recession recovery vs. 2016 economic conditions
- **Consumer behavior**: Shopping pattern evolution over 5-year span
- **Competitive landscape**: New competitors, store expansions affecting demand
- **Product lifecycle**: Items in different maturity phases between train/validation

**Validation Strategy Implications:**
- **Walk-forward validation**: Use multiple 28-day validation windows from training period
- **Seasonal validation**: Include same-season validation periods from previous years
- **Cross-validation timing**: Respect temporal ordering across all CV folds

---

# 14. Look for leakage

Leakage means your model sees information it would not have in production.

Common leakage examples:

* Future data
* Post-outcome variables
* IDs that encode the target
* Aggregates calculated using full dataset
* Target encoding before train-test split
* Duplicates across train and test
* Time-series random split
* Features created after the event
* Columns with suspiciously high correlation with target

Ask for every strong feature:

> Would I know this value at prediction time?

If the answer is no, remove or redesign it.

## 📋 Application to Demand Forecasting Project

**Critical Leakage Prevention for M5 Dataset:**

**Temporal Leakage (Most Critical):**
- **Future sales values**: Never use d_1914+ for predicting d_1913 and earlier
- **Future pricing**: Only use sell_price up to prediction week, not future weeks
- **Future events**: Only use known scheduled events, not data-discovered patterns
- **Rolling statistics**: Ensure rolling windows only include past values

**Feature Engineering Leakage:**
- **Target encoding**: Calculate encodings only on training data, apply to validation
- **Normalization statistics**: Compute means/stds only on training period
- **Categorical mappings**: Handle new categories in validation that weren't in training
- **Outlier thresholds**: Set capping/winsorization limits using training data only

**Information Leakage Examples to Avoid:**
- **Perfect predictors**: Features with correlation >0.98 with target (suspicious)
- **Future event impact**: Using post-holiday sales patterns to predict pre-holiday demand
- **Global statistics**: Using full-dataset statistics instead of time-aware calculations
- **ID leakage**: Item codes that accidentally encode sales volume information

**Subtle Leakage Risks:**
- **Price strategy knowledge**: Using future promotional calendars not available at prediction time
- **Inventory-based features**: Using stockout indicators that depend on future demand
- **Cross-validation leakage**: Information bleeding across temporal CV folds
- **Evaluation period knowledge**: Using validation period statistics during training

**Leakage Validation Checklist:**
- Can this feature be computed using only data available at prediction time?
- Would a business user have access to this information when making forecasts?
- Does this feature require knowledge of the outcome we're trying to predict?
- Are temporal boundaries strictly respected in all feature engineering steps?

---

# 15. Decide modeling implications

At the end of EDA, you should know how the data affects modeling.

Summarize:

* What type of split to use
* What metric to use
* What preprocessing is needed
* Which features need cleaning
* Which features look useful
* Which features may leak
* Which features need transformation
* Which categories need grouping
* Which missing values need special handling
* Whether time-series validation is needed
* Whether class imbalance is a problem
* Whether outliers should be kept or treated

EDA is not just “making plots.” It should guide model design.

## 📋 Application to Demand Forecasting Project

**Modeling Implications Summary for M5 Demand Forecasting:**

**Split Strategy:**
- **Time-based split**: Train on d_1 to d_1913, validate on d_1914 to d_1941
- **Walk-forward validation**: Multiple 28-day windows within training period
- **No random splits**: Preserve temporal ordering for all item-store series
- **Cross-validation**: Time-series grouped CV with proper temporal boundaries

**Evaluation Metrics:**
- **Primary**: WRMSSE (Weighted Root Mean Squared Scaled Error) - Walmart's competition metric
- **Secondary**: RMSE, MAE for interpretability
- **Segmentation**: Silhouette score, business interpretability assessment
- **Hierarchical**: Aggregate accuracy at category, department, and store levels

**Required Preprocessing:**
- **Sales normalization**: Consider log(1+x) transformation for right-skewed sales
- **Price standardization**: Scale prices within categories due to different product types
- **Calendar encoding**: One-hot encoding for categorical temporal features
- **Missing price handling**: Forward-fill or “not available” category encoding
- **Outlier treatment**: Cap extreme sales values at 99.9th percentile

**Feature Engineering Priorities:**
- **Lag features**: 1, 7, 14, 28-day sales lags
- **Rolling statistics**: 7, 14, 28-day moving averages and standard deviations
- **Price features**: Price changes, relative pricing, promotional indicators
- **Calendar features**: Holiday proximity, event type encoding, SNAP timing
- **Cross-product features**: Category velocity, store performance metrics

**Model Architecture Considerations:**
- **Hierarchical models**: Leverage product and geographic hierarchies
- **Multi-task learning**: Joint forecasting and segmentation objectives
- **Intermittent demand**: Specialized models for zero-inflated time series
- **Ensemble approach**: Combine global and local models for different product segments
- **Feature importance**: Interpretable models for business insight generation

**Critical Success Factors:**
- Proper handling of intermittent demand patterns (many zero-sales days)
- Seasonal pattern detection and extrapolation capabilities
- Price elasticity incorporation for promotional planning
- Scalable architecture for 30,490 item-store combinations
- Business-interpretable segmentation for actionable insights

---

# A simple EDA checklist you can follow every time

Use this order:

1. **Problem understanding**
   What are we predicting and when?

2. **Dataset structure**
   Rows, columns, types, IDs, target, dates.

3. **Data quality**
   Missing values, duplicates, impossible values, inconsistent labels.

4. **Target analysis**
   Class balance, target distribution, skew, outliers, time trend.

5. **Feature analysis**
   Numerical, categorical, datetime, text.

6. **Feature-target relationships**
   Which variables seem predictive?

7. **Feature-feature relationships**
   Correlation, redundancy, multicollinearity.

8. **Time-series analysis**
   Trend, seasonality, lags, gaps, leakage, time-based split.

9. **Missingness analysis**
   Why missing? Is missing predictive?

10. **Outlier analysis**
    Error or signal?

11. **Segment analysis**
    Behavior by groups.

12. **Train-test distribution check**
    Drift, unseen categories, time mismatch.

13. **Leakage audit**
    Would this feature exist at prediction time?

14. **Feature engineering plan**
    Transformations, lags, ratios, flags, encodings.

15. **Modeling plan**
    Split strategy, metric, preprocessing, baseline model.

## 📋 Application to Demand Forecasting Project

**Tailored EDA Checklist for M5 Demand Forecasting:**

1. **Walmart M5 Problem Understanding**
   - Predict daily unit sales for 28-day horizon (d_1914 to d_1941)
   - Segment products into 5 behavioral categories
   - Panel time-series: 30,490 item-store combinations

2. **M5 Dataset Structure Validation**
   - 30,490 rows × 1,919 columns (sales_train_validation.csv)
   - Hierarchical IDs: state_id → store_id, cat_id → dept_id → item_id
   - Time mapping: d_1 to d_1969 ↔ calendar.csv dates

3. **Retail Data Quality Checks**
   - Zero vs. missing sales (zeros are informative)
   - Price data completeness by item-store-week combinations
   - Calendar event consistency and SNAP benefit alignment

4. **Intermittent Demand Target Analysis**
   - Zero-inflation patterns by category (FOODS vs. HOUSEHOLD vs. HOBBIES)
   - Sales distribution skewness and outlier identification
   - Seasonal demand cycles and trend patterns

5. **Retail Feature Analysis**
   - Category/department/store performance hierarchies
   - Calendar effects (weekday, holiday, SNAP timing)
   - Pricing strategy patterns and promotional indicators

6. **Demand Driver Relationships**
   - Price elasticity by product category
   - Event impact quantification (holiday lift factors)
   - Cross-store and cross-product correlation patterns

7. **Walmart Hierarchy Correlations**
   - Geographic clustering (state/store similarity)
   - Product substitution effects within departments
   - Pricing coordination across store networks

8. **Retail Time-Series Analysis**
   - Weekly, monthly, yearly seasonality detection
   - Holiday and SNAP benefit cyclical patterns
   - Trend identification and structural break detection

9. **Sparse Data Missingness**
   - Pricing gaps as product availability indicators
   - Seasonal discontinuation patterns
   - Geographic distribution limitations

10. **Retail Outlier Investigation**
    - Promotional spike identification and validation
    - Supply disruption detection (sudden zero periods)
    - Bulk purchase vs. data error discrimination

11. **Product-Store Segment Behavior**
    - Performance differences across geographic regions
    - Category-specific demand patterns and seasonality
    - Store format and demographic effect analysis

12. **5-Year Time Distribution Analysis**
    - Economic condition changes (2011 recession recovery → 2016)
    - Consumer behavior evolution over dataset timeline
    - Seasonal consistency across multiple years

13. **Temporal Leakage Prevention**
    - Future sales, pricing, and event information audit
    - Rolling statistic boundary validation
    - Cross-validation temporal integrity checks

14. **Retail-Specific Feature Engineering**
    - Sales lags, rolling averages, price change indicators
    - Holiday proximity features, SNAP timing effects
    - Category velocity and store performance metrics

15. **Demand Forecasting Model Design**
    - Time-based validation with 28-day horizons
    - WRMSSE metric alignment with business objectives
    - Hierarchical modeling for scalable predictions

---

# My recommended mental model

For every dataset, ask these five questions:

## 1. What is the unit?

What does one row mean?

Example:

* One customer?
* One transaction?
* One day?
* One product per store per week?

## 2. What is the target?

What exactly are we predicting?

## 3. What is the prediction time?

At what moment would the model make the prediction?

This protects you from leakage.

## 4. What changes over time?

This tells you whether you need time-based EDA and validation.

## 5. What can go wrong?

Missingness, outliers, drift, imbalance, leakage, bias, duplicates.

## 📋 Application to Demand Forecasting Project

**M5 Dataset Mental Model:**

## 1. What is the unit?
- **One row = One item-store demand series** (e.g., FOODS_1_001 sold at CA_1)
- 30,490 unique item-store combinations × 1,969 daily observations
- Each series represents independent forecasting challenge

## 2. What is the target?
- **Daily unit sales** for next 28 days (d_1914 to d_1941)
- **Product behavior segments** (5 categories: Stable, Volatile, Intermittent, Declining, Seasonal)
- Integer values representing units sold per day

## 3. What is the prediction time?
- **End of training period**: After d_1913 (May 22, 2016)
- **Available information**: Historical sales, known calendar events, current pricing
- **Business context**: Inventory planning 28 days ahead

## 4. What changes over time?
- **Seasonal demand cycles**: Weekly, monthly, yearly patterns
- **Economic conditions**: 2011-2016 recession recovery period
- **Product lifecycles**: New launches, maturity, decline phases
- **Consumer behavior**: Shopping pattern evolution over 5+ years
- **Pricing strategies**: Promotional timing and competitive responses

## 5. What can go wrong?
- **Intermittent demand**: Many zero-sales days, sparse patterns
- **Seasonal drift**: Different seasonal patterns in validation vs. training
- **Price leakage**: Using future promotional information
- **Event overfitting**: Holiday patterns that don't generalize
- **Hierarchy collapse**: High-cardinality item IDs causing overfitting
- **Geographic bias**: Unequal representation across states/stores
- **Category imbalance**: Different forecasting difficulty by product type

---

# A good final EDA output should look like this

After EDA, you should be able to write:

> The dataset contains X rows and Y columns. The prediction target is Z. The target is imbalanced/skewed. Important issues include missing values in A, outliers in B, high cardinality in C, and possible leakage in D. Since the data contains time, validation should be time-based rather than random. Useful feature engineering ideas include lags, rolling averages, ratios, and missing indicators. The first baseline model should use cleaned features with an appropriate metric such as AUC/F1/RMSE/MAE depending on the problem.

That is the real purpose of EDA: **to make better modeling decisions**, not just to create charts.

## 📋 Application to Demand Forecasting Project

**Expected EDA Output for M5 Demand Forecasting:**

> The M5 dataset contains 30,490 item-store combinations with 1,969 daily observations each (Jan 2011 - June 2016). The prediction targets are daily unit sales for 28-day horizons and product behavioral segmentation into 5 categories. The target is heavily zero-inflated with right-skewed distribution typical of intermittent retail demand. Important issues include pricing data sparsity (seasonal item availability), extreme promotional outliers requiring investigation, high cardinality in item_id (3,049 products), and critical temporal leakage risks from future pricing/event data. Since this is panel time-series data, validation must be strictly time-based using d_1914+ periods, never random splits. Essential feature engineering includes sales lags (1, 7, 14, 28 days), rolling averages, price change indicators, holiday proximity features, and SNAP benefit timing effects. The baseline model should handle zero-inflation (Hurdle/Zero-Inflated models), incorporate hierarchical product/geographic structure, and use WRMSSE metric aligned with Walmart's business objectives. Critical success factors are proper intermittent demand modeling, seasonal pattern extrapolation, and price elasticity integration for promotional planning insights.

**Key Modeling Decision Summary:**
- **Architecture**: Hierarchical models leveraging product/geographic trees
- **Validation**: Time-based walk-forward with 28-day prediction windows  
- **Features**: Lag-based, price-aware, calendar-enriched with proper temporal boundaries
- **Metrics**: WRMSSE for forecasting accuracy, silhouette score for segmentation quality
- **Business value**: Actionable inventory planning with interpretable product categorization
