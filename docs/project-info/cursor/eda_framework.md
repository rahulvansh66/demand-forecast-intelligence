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

**Project application note:** For this project, the main objective is daily demand forecasting: predict unit sales for each `item_id + store_id` demand series for the next 28 days. A second objective is product sales segmentation into five behavior groups: high demand stable, high demand volatile, low demand intermittent, declining, and seasonal. One row in the raw sales table represents one product-store series, while the daily `d_1` to `d_1913` columns represent the historical target values. At prediction time, the model should only use information available before the forecast horizon, such as past sales, known calendar attributes, and known or planned price information.

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

**Project application note:** In the M5 data, the core structure spans four raw files in `data/raw`: `sales_train_validation.csv`, `sales_train_evaluation.csv`, `calendar.csv`, and `sell_prices.csv`. The primary identifiers are `id`, `item_id`, `store_id`, `dept_id`, `cat_id`, and `state_id`. The target values are stored in wide daily sales columns such as `d_1` through `d_1913` in `sales_train_validation.csv`, which map to real dates through `calendar.csv`. For modeling, EDA should make this wide panel structure explicit and usually reshape it into a long format with one row per item-store-date.

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

**Project application note:** Quality checks should verify that each expected `item_id + store_id` combination appears once, daily sales columns are non-negative integers, store and state identifiers are consistent, and all `d_*` sales columns map to valid rows in `calendar.csv`. In `sell_prices.csv`, missing item-store-week combinations may mean an item was not sold in that store-week, so missing prices should not automatically be treated as random missingness. Event and SNAP fields in `calendar.csv` are intentionally nullable or binary and should be interpreted according to their schema.

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

**Project application note:** The forecasting target is daily unit sales for the next 28 days per `item_id + store_id` series. EDA should examine zero-heavy demand, intermittent sales, long-tailed high-volume items, and differences by product hierarchy and geography. For the segmentation model, the "target" is not a supervised label in the raw data; instead, EDA should derive behavior features such as average demand, volatility, intermittency, trend, and seasonality to support clustering into the five planned product sales segments.

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

**Project application note:** Numerical features in this project include historical sales, `sell_price`, lag values, rolling statistics, and derived demand metrics. Categorical features include `cat_id`, `dept_id`, `item_id`, `store_id`, `state_id`, event names, and event types. Datetime features come from `calendar.csv`, including weekday, month, year, Walmart fiscal week, events, and state-level SNAP indicators. Because `item_id` is high-cardinality and each demand series is defined by `item_id + store_id`, EDA should separate true descriptive hierarchy from identifiers that may overfit if encoded naively.

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

**Project application note:** Useful feature-target checks include sales by `cat_id`, `dept_id`, `store_id`, `state_id`, weekday, event type, SNAP flag, and price bands. For example, FOODS demand may respond differently to SNAP days than HOBBIES demand, and price changes may have store-specific effects. Any feature that appears to predict sales too perfectly should be checked for leakage, especially rolling statistics, aggregates, or evaluation-period values computed across the full timeline.

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

**Project application note:** The dataset contains intentional hierarchy and redundancy: `cat_id` is embedded in `dept_id`, `dept_id` is embedded in `item_id`, and `state_id` is embedded in `store_id`. Calendar fields such as `date`, `d`, `wm_yr_wk`, `weekday`, `wday`, `month`, and `year` are also related. EDA should document these relationships so modeling can avoid double-counting signals, misleading importance scores, or accidental leakage through derived aggregates.

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

**Project application note:** This project is a panel time-series problem: each `item_id + store_id` combination is one daily demand series. `sales_train_validation.csv` covers `d_1` to `d_1913`, while `calendar.csv` maps those day IDs to dates from January 29, 2011 through May 22, 2016 for the validation training period. EDA should confirm every series shares the same daily calendar and decide whether to analyze all 30,490 series together, by hierarchy, or through representative samples.

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

**Project application note:** For M5 demand, time plots should be created at multiple levels: single item-store series, store totals, state totals, department totals, and category totals. These views help distinguish item-level intermittency from broader retail seasonality, event spikes, calendar effects, and category-specific patterns. They also support the segmentation goal by making stable, volatile, declining, intermittent, and seasonal behavior visible.

## C. Check time-based missingness

Ask:

* Are some dates missing?
* Are weekends absent?
* Are there gaps?
* Are all entities observed at the same frequency?
* Does missingness itself contain signal?

**Project application note:** The sales matrices are expected to contain consecutive daily columns, so missing timestamps usually appear as schema or integration issues rather than ordinary gaps. However, price data is weekly and sparse, so EDA must check whether absent prices mean unavailable items, not-yet-launched products, discontinued products, or incomplete data. This distinction affects both feature engineering and how zero sales should be interpreted.

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

**Project application note:** Leakage risk is high when converting the wide sales table into lag and rolling features. A 7-day rolling average for a forecast date must use only sales before that forecast date, not the current or future target day. Validation should also avoid using `sales_train_evaluation.csv` values from the 28-day extension when training or tuning models intended to predict that same horizon.

## E. Create time-aware validation thinking

For time-series or time-dependent data, prefer:

* Train on past, validate on future
* Rolling validation
* Expanding window validation
* Group-time split if multiple entities exist

Do not randomly split if time order matters.

**Project application note:** The natural validation setup is time-based: train on earlier `d_*` columns and validate on later 28-day windows, aligned with the project objective of forecasting the next 28 days. Because there are many item-store series, validation can combine temporal splits with grouping by series, category, store, or state to verify that performance is not strong only for high-volume or easy-to-predict products.

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

**Project application note:** In `calendar.csv`, missing `event_name_1`, `event_type_1`, `event_name_2`, and `event_type_2` usually means there was no event, so these values should be converted or interpreted as "no event" rather than treated as unknown. In `sell_prices.csv`, missing prices can indicate product availability patterns and may help detect launch or discontinuation behavior. In the sales table, zeros are observed sales values, not missing values, and are central to identifying intermittent and sparse demand.

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

**Project application note:** Demand spikes around holidays, SNAP periods, sporting events, or promotions may be real business signal rather than data errors. Extremely high sales, long zero stretches, sudden drops, or abrupt level shifts should be reviewed by item, store, category, and date. These patterns can directly inform segmentation, especially for high-demand volatile, declining, seasonal, and low-demand intermittent products.

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

**Project application note:** Key project segments include `cat_id`, `dept_id`, `item_id`, `store_id`, `state_id`, event type, SNAP state, and time period. EDA should compare behavior across CA, TX, and WI stores, across FOODS, HOUSEHOLD, and HOBBIES, and across departments to avoid building a model that performs well only on dominant groups. Segment-level views are also the bridge between raw demand data and the five planned product sales clusters.

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

**Project application note:** Strong candidate features include sales lags, rolling means, rolling standard deviations, zero-sales streaks, days since last sale, price changes, price relative to item-store history, event indicators, SNAP indicators, weekday/month features, and hierarchy-level aggregates. For segmentation, aggregate each item-store or item into behavior features such as mean demand, coefficient of variation, intermittency ratio, trend slope, and seasonal strength. Every rolling or aggregate feature must be computed using only historical data for the relevant forecast cutoff.

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

**Project application note:** Train-test comparison should focus on time periods, product-store series, category mix, store/state mix, price availability, event frequency, SNAP days, and target distribution. The documented forecasting flow uses training data, a validation horizon, and an evaluation horizon, so EDA should check whether the final 28-day windows differ materially from earlier history. Distribution checks should also ensure that rare intermittent products are represented, not hidden by aggregate performance.

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

**Project application note:** Common leakage traps include computing item-level averages across the full dataset, fitting encoders before the time split, using future sales from `sales_train_evaluation.csv`, joining prices or calendar rows beyond the prediction cutoff incorrectly, or creating rolling windows that include the target day. Since the production input is store ID, item ID, and forecast horizon, every feature should be audited against what would be known for that request at the time the forecast is generated.

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

**Project application note:** For this project, EDA should lead to concrete modeling decisions: use time-based validation, evaluate 28-day forecast accuracy at item-store and aggregate hierarchy levels, preserve zero-heavy and intermittent demand behavior, engineer calendar/price/lag features, and create clustering features that map to the five demand behavior segments. The EDA output should also flag business risks such as poor performance on low-volume products, state-specific SNAP effects, category-level seasonality, or unstable price coverage.

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

**Project application note:** When using this checklist for the M5 project, the recurring unit of analysis should be the `item_id + store_id` demand series, the supervised target should be future daily unit sales over a 28-day horizon, and the unsupervised segmentation input should be behavior features derived from historical sales. The checklist should be applied at both granular and aggregated levels so that EDA captures item-store noise as well as store, state, department, and category-level business patterns.

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

**Project application note:** The modeling unit is one product-store demand series, identified by `item_id + store_id`. During reshaping, one long-format observation typically becomes one item-store-date record with sales, calendar, and price context. Aggregated views by category, department, store, or state are useful for EDA but should not replace the item-store unit needed for forecasting.

## 2. What is the target?

What exactly are we predicting?

**Project application note:** For forecasting, the target is daily unit sales for each future day in the next 28-day horizon. For segmentation, the target is not a raw label; the goal is to cluster products or product-store series using historical behavior so they align with interpretable demand groups such as stable high demand, volatile high demand, intermittent low demand, declining, and seasonal.

## 3. What is the prediction time?

At what moment would the model make the prediction?

This protects you from leakage.

**Project application note:** Prediction time is the cutoff date immediately before the requested forecast horizon. Given a store ID, item ID, and horizon, the model may use historical sales and known calendar/price context up to that cutoff, but not actual sales from the future 28 days. EDA should make this cutoff explicit whenever it evaluates features or validation windows.

## 4. What changes over time?

This tells you whether you need time-based EDA and validation.

**Project application note:** Sales, prices, events, SNAP indicators, seasonality, product availability, and demand levels all change over time in the M5 data. Some items may launch late, disappear, decline, or become seasonal. These changes make time-based EDA and validation mandatory for both forecasting reliability and meaningful segmentation.

## 5. What can go wrong?

Missingness, outliers, drift, imbalance, leakage, bias, duplicates.

**Project application note:** Project-specific risks include sparse and intermittent demand, too much emphasis on high-volume products, leakage from future sales or full-history aggregates, sparse price records, event-driven spikes, state-specific SNAP behavior, and distribution drift between training and evaluation windows. EDA should explicitly document these risks and connect each one to a modeling or preprocessing decision.

---

# A good final EDA output should look like this

After EDA, you should be able to write:

> The dataset contains X rows and Y columns. The prediction target is Z. The target is imbalanced/skewed. Important issues include missing values in A, outliers in B, high cardinality in C, and possible leakage in D. Since the data contains time, validation should be time-based rather than random. Useful feature engineering ideas include lags, rolling averages, ratios, and missing indicators. The first baseline model should use cleaned features with an appropriate metric such as AUC/F1/RMSE/MAE depending on the problem.

**Project application note:** For this project, the final EDA summary should state the number of item-store series, date range, available calendar and price context, target horizon, major demand patterns, and implications for both the 28-day forecasting model and the five-segment product behavior model. It should also identify which product/store groups are hardest to forecast and what business recommendations the eventual GenAI insight layer should treat carefully.

That is the real purpose of EDA: **to make better modeling decisions**, not just to create charts.
