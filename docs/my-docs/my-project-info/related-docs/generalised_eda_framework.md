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

---

# A good final EDA output should look like this

After EDA, you should be able to write:

> The dataset contains X rows and Y columns. The prediction target is Z. The target is imbalanced/skewed. Important issues include missing values in A, outliers in B, high cardinality in C, and possible leakage in D. Since the data contains time, validation should be time-based rather than random. Useful feature engineering ideas include lags, rolling averages, ratios, and missing indicators. The first baseline model should use cleaned features with an appropriate metric such as AUC/F1/RMSE/MAE depending on the problem.

That is the real purpose of EDA: **to make better modeling decisions**, not just to create charts.

======
My notes:

Phase 1: Data Understanding

Subgroup 1A: Business Context (Step 1) - Problem understanding, objectives, leakage prevention
Subgroup 1B: Data Quality Audit (Step 3) - Missing values, duplicates, impossible values, consistency
Phase 2: Feature Analysis

Subgroup 2A: Individual Feature Profiling (Step 5) - Distributions, skewness, cardinality, transformations
Subgroup 2B: Relationship Analysis (Steps 6, 7) - Feature-target and feature-feature relationships
Phase 3: Time-Series & Patterns

Subgroup 3A: Temporal Analysis (Step 8) - Time structure, seasonality, trends, validation strategy
Subgroup 3B: Data Quality Patterns (Steps 9, 10) - Missing value patterns, outlier detection
Subgroup 3C: Segment Behavior (Step 11) - Geographic, category, temporal segment analysis
Phase 4: Model Preparation

Subgroup 4A: Feature Engineering (Step 12) - Transformation opportunities, lag features, encodings
Subgroup 4B: Validation Strategy (Step 13) - Train-test distribution comparison, drift detection
Subgroup 4C: Model Design (Step 15) - Architecture implications, metrics, preprocessing decisions