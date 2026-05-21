# Detailed Exploratory Data Analysis (EDA) Checklist

Use this checklist as a reusable framework for approaching almost any EDA project. It is intentionally detailed so you can adapt it to business analysis, analytics reporting, data science modeling, dashboard preparation, or data quality assessment.

---

## 1. Define the Objective

### 1.1 Clarify the purpose of the analysis

- Identify the main reason for performing EDA.
  - Is the goal to understand the dataset?
  - Is the goal to prepare the data for modeling?
  - Is the goal to support a business decision?
  - Is the goal to investigate a known issue?
  - Is the goal to discover unexpected patterns?
- Write the primary objective in one or two clear sentences.
- Separate exploratory goals from confirmatory goals.
  - Exploratory: “What patterns exist in customer behavior?”
  - Confirmatory: “Does churn increase when support tickets increase?”

### 1.2 Understand the business or domain context

- Identify the domain of the dataset.
  - Sales
  - Marketing
  - Finance
  - Product analytics
  - Customer support
  - Operations
  - Healthcare
  - Risk or fraud
  - Supply chain
- Understand the key business entities.
  - Customers
  - Orders
  - Transactions
  - Sessions
  - Products
  - Employees
  - Accounts
  - Devices
- Understand the business process that generated the data.
  - How was the data collected?
  - Who or what created each record?
  - When is data captured?
  - Are there manual inputs?
  - Are there automated system logs?
  - Are there third-party sources?

### 1.3 Identify stakeholders and decision users

- Determine who will consume the analysis.
  - Leadership
  - Product managers
  - Data science team
  - Marketing team
  - Finance team
  - Operations team
  - Engineering team
  - External clients
- Understand what decisions they may make from the analysis.
  - Strategy decisions
  - Budget decisions
  - Product changes
  - Campaign optimization
  - Risk mitigation
  - Process improvements
  - Modeling choices

### 1.4 Define key questions

- List the major questions the EDA should answer.
  - What is the current state of the data?
  - What are the main patterns?
  - What segments behave differently?
  - What are the biggest drivers of the target outcome?
  - What data quality issues exist?
  - What risks or limitations should be highlighted?
- Prioritize questions by importance.
  - Must answer
  - Nice to answer
  - Out of scope

### 1.5 Define the unit of analysis

- Determine what each row represents.
  - One customer
  - One transaction
  - One order
  - One session
  - One product
  - One account per month
  - One event log
- Confirm whether the row grain is consistent across the dataset.
- Check whether the dataset mixes multiple grains.
  - Customer-level columns combined with transaction-level rows
  - Daily metrics mixed with monthly aggregates
  - Product-level attributes mixed with order-level records
- Note any implications of the data grain.
  - Aggregation choices
  - Duplicate risk
  - Incorrect averages
  - Leakage risk in modeling

### 1.6 Define success criteria

- Decide what a useful EDA output should contain.
  - Clean data quality summary
  - Key distributions
  - Important relationships
  - Segment-level findings
  - Target variable behavior
  - Outlier analysis
  - Time trend summary
  - Recommendations for cleaning, modeling, or reporting
- Define the expected final deliverable.
  - Notebook
  - Dashboard
  - Written report
  - Slide deck
  - Data quality memo
  - Model-readiness summary

---

## 2. Understand Rows, Columns, and Data Types

### 2.1 Inspect dataset shape

- Record the number of rows.
- Record the number of columns.
- Understand whether the dataset size is small, medium, or large for the intended analysis.
- Consider whether sampling may be needed for very large datasets.
- Check whether all expected data is present.
  - Expected time period
  - Expected regions
  - Expected products
  - Expected users or accounts
  - Expected business units

### 2.2 Review column names

- Check whether column names are understandable.
- Identify unclear or cryptic column names.
- Look for inconsistent naming patterns.
  - Mixed casing
  - Spaces in column names
  - Abbreviations
  - Special characters
  - Duplicate or near-duplicate names
- Confirm whether column names follow a standard convention.
  - snake_case
  - camelCase
  - PascalCase
  - Business-defined naming
- Create a data dictionary if one does not exist.

### 2.3 Understand column meanings

- For each column, document:
  - Name
  - Description
  - Data type
  - Example values
  - Allowed values if applicable
  - Business meaning
  - Source system
  - Whether it is raw, derived, or manually entered
- Identify columns that require clarification from domain experts.
- Separate technical fields from business fields.
  - Technical: IDs, timestamps, ingestion metadata
  - Business: revenue, category, status, customer segment

### 2.4 Classify variables by type

- Identify numeric variables.
  - Continuous variables
  - Discrete count variables
  - Monetary variables
  - Percentages or rates
  - Scores or indices
- Identify categorical variables.
  - Nominal categories
  - Ordinal categories
  - Binary flags
  - High-cardinality categories
- Identify date and time variables.
  - Event date
  - Created date
  - Updated date
  - Transaction timestamp
  - Subscription start/end date
- Identify text variables.
  - Names
  - Descriptions
  - Comments
  - Search queries
  - Support tickets
- Identify identifiers.
  - Primary keys
  - Foreign keys
  - Composite keys
  - Natural keys
  - Surrogate keys
- Identify boolean variables.
  - True/false fields
  - Yes/no fields
  - 0/1 indicators

### 2.5 Validate data types

- Check whether each column has the correct data type.
- Identify numeric columns stored as text.
- Identify dates stored as text.
- Identify categorical columns stored as numeric codes.
- Identify boolean columns stored inconsistently.
  - `Y`/`N`
  - `Yes`/`No`
  - `1`/`0`
  - `True`/`False`
- Check whether leading zeros are meaningful.
  - ZIP codes
  - Product codes
  - Account numbers
- Avoid converting identifier-like fields to numeric if they are not mathematically meaningful.

### 2.6 Identify key columns

- Determine the primary identifier column, if any.
- Determine columns that define uniqueness.
- Identify foreign key columns that connect to other datasets.
- Identify target or outcome variables.
- Identify segmentation columns.
  - Region
  - Customer type
  - Product category
  - Channel
  - Plan type
  - Tenure group
- Identify time columns needed for trend analysis.

### 2.7 Check column-level availability

- Determine which columns are fully populated.
- Determine which columns are sparsely populated.
- Identify columns with constant values.
- Identify columns with almost constant values.
- Identify columns that may not be useful due to low variation.
- Identify columns that may be useful despite high missingness.
  - Rare event flags
  - Optional user-entered fields
  - Cancellation reason
  - Complaint details

---

## 3. Check Missing Values, Duplicates, and Invalid Values

### 3.1 Analyze missing values

- Calculate missing value count for each column.
- Calculate missing value percentage for each column.
- Rank columns by missingness.
- Identify rows with many missing fields.
- Identify columns with extreme missingness.
- Decide whether missing values are expected or problematic.

### 3.2 Understand missingness patterns

- Determine whether missingness is random or systematic.
- Check whether missingness is concentrated in specific groups.
  - Region
  - Product
  - Time period
  - Customer segment
  - Source system
- Check whether missingness increased or decreased over time.
- Compare missingness against the target variable, if present.
- Identify whether missing values may themselves carry business meaning.
  - No cancellation date may mean active customer
  - No return date may mean product was not returned
  - No complaint reason may mean no complaint was filed

### 3.3 Distinguish types of missing values

- True missing values.
  - Unknown
  - Not collected
  - Lost during ingestion
- Not applicable values.
  - Cancellation date for active customer
  - Spouse name for unmarried customer
  - Return reason for non-returned product
- Placeholder values.
  - `NA`
  - `N/A`
  - `Unknown`
  - `None`
  - `-999`
  - `0` used incorrectly
  - Blank string
- System-generated missing values.
  - Failed joins
  - Incomplete API response
  - Delayed data feed

### 3.4 Evaluate duplicate rows

- Check exact duplicate rows.
- Check duplicates based on key identifiers.
- Check duplicates based on business logic.
  - Same customer, same transaction date, same amount
  - Same order ID appearing multiple times
  - Same event recorded by multiple systems
- Determine whether duplicates are errors or legitimate records.
  - Multiple line items per order
  - Multiple events per user
  - Multiple status changes per account
- Check whether duplicates affect aggregations.
  - Revenue inflation
  - Customer count inflation
  - Conversion rate distortion

### 3.5 Validate uniqueness rules

- Confirm whether the supposed primary key is unique.
- Check for duplicate IDs.
- Check whether composite keys are needed.
- Identify many-to-one or one-to-many relationships.
- Validate whether each entity appears the expected number of times.
  - One row per user
  - One row per user per month
  - One row per transaction
  - One row per order item

### 3.6 Check invalid values

- Identify impossible numeric values.
  - Negative age
  - Negative quantity sold
  - Revenue less than zero when refunds are not included
  - Discount greater than 100%
  - Probability outside 0 to 1
- Identify impossible dates.
  - Future birth date
  - End date before start date
  - Transaction date before account creation
  - Delivery date before order date
- Identify impossible categories.
  - Status values outside allowed list
  - Country codes that do not exist
  - Product categories not in catalog
- Identify malformed values.
  - Invalid email format
  - Invalid phone number format
  - Invalid ZIP or postal code format
  - Invalid currency format

### 3.7 Check inconsistent category values

- Look for case differences.
  - `active`, `Active`, `ACTIVE`
- Look for spelling differences.
  - `Cancelled`, `Canceled`
- Look for trailing or leading spaces.
- Look for abbreviations and full forms mixed together.
  - `US`, `USA`, `United States`
- Look for inconsistent symbols.
  - `&` vs `and`
- Standardize categories where appropriate.

### 3.8 Identify data entry issues

- Check manually entered text fields for typos.
- Look for free-text values that should be categorical.
- Look for numeric values entered with units.
  - `10 kg`
  - `$50`
  - `30 days`
- Check whether decimal separators are consistent.
  - `1.5` vs `1,5`
- Check whether date formats are consistent.
  - `MM/DD/YYYY`
  - `DD/MM/YYYY`
  - ISO date format

### 3.9 Decide treatment strategies

- Document how each issue will be handled.
  - Keep as-is
  - Correct values
  - Standardize categories
  - Remove records
  - Impute missing values
  - Create missingness indicator
  - Flag for downstream review
- Avoid silently deleting data without documenting the impact.
- Quantify how many rows or columns are affected by each cleaning decision.

---

## 4. Analyze Numeric Columns

### 4.1 Profile numeric variables

- For each numeric column, review:
  - Count of non-missing values
  - Missing value percentage
  - Minimum
  - Maximum
  - Mean
  - Median
  - Standard deviation
  - Variance
  - Percentiles
  - Number of unique values
- Separate continuous variables from count variables.
- Separate monetary values from physical measurements.
- Treat IDs stored as numbers as identifiers, not numeric measures.

### 4.2 Understand central tendency

- Compare mean and median.
- Identify whether the mean is heavily influenced by extreme values.
- Use median when the distribution is skewed.
- Use mean when the distribution is roughly symmetric and outliers are limited.
- Compare central tendency across important segments.

### 4.3 Understand spread and variability

- Review standard deviation and interquartile range.
- Identify variables with very low variability.
- Identify variables with extremely high variability.
- Assess whether variation is meaningful or caused by data errors.
- Compare variability across groups.
  - Some segments may be more stable than others.
  - Some products may have wider price ranges.

### 4.4 Examine distribution shape

- Determine whether each numeric variable is:
  - Symmetric
  - Right-skewed
  - Left-skewed
  - Bimodal
  - Multimodal
  - Uniform
  - Heavy-tailed
- Think about why the distribution has that shape.
  - Natural business behavior
  - Pricing tiers
  - Customer segments mixed together
  - Data collection issue
  - Capped or bounded values

### 4.5 Check scale and units

- Confirm measurement units.
  - Dollars
  - Rupees
  - Kilograms
  - Seconds
  - Days
  - Percent
  - Basis points
- Check whether units are mixed within the same column.
- Confirm whether monetary values are gross or net.
- Confirm whether values include tax, discounts, refunds, or fees.
- Confirm whether percentages are represented as `0.25` or `25`.

### 4.6 Review zero values

- Identify numeric columns with many zeros.
- Determine whether zeros are valid values or placeholders for missing values.
- Check whether zero has special meaning.
  - No purchases
  - No usage
  - Free plan
  - No discount
  - No balance
- Compare rows with zero values against other fields.

### 4.7 Review negative values

- Identify numeric columns with negative values.
- Determine whether negative values are valid.
  - Refunds
  - Returns
  - Account credits
  - Losses
  - Adjustments
- Investigate negative values where only positive values are expected.

### 4.8 Check bounded numeric variables

- Identify variables with natural bounds.
  - Age should usually be positive and reasonable.
  - Percentages should often be between 0 and 100.
  - Probabilities should be between 0 and 1.
  - Ratings may be between 1 and 5.
  - Scores may be between 0 and 100.
- Check whether values fall outside valid bounds.
- Check whether values pile up at boundaries.
  - Many values at 0
  - Many values at 100
  - Many maximum scores

### 4.9 Identify transformations that may help

- Consider log transformation for heavily right-skewed variables.
- Consider binning for interpretability.
- Consider normalization or standardization for modeling.
- Consider winsorization only when justified.
- Consider creating ratios or derived metrics.
  - Revenue per customer
  - Orders per user
  - Cost per acquisition
  - Average order value
  - Conversion rate
  - Usage per day

### 4.10 Summarize numeric insights

- Identify the most important numeric variables.
- Highlight unusual distributions.
- Note extreme values requiring investigation.
- Note variables that may need cleaning or transformation.
- Note variables likely to be strong predictors or important business metrics.

---

## 5. Analyze Categorical Columns

### 5.1 Profile categorical variables

- For each categorical column, review:
  - Number of unique values
  - Most common values
  - Least common values
  - Missing value count
  - Missing value percentage
  - Category frequency distribution
- Distinguish between true categorical variables and identifiers.
- Identify categorical variables encoded as numbers.

### 5.2 Review category frequency

- Check whether a few categories dominate.
- Check whether the distribution is balanced or imbalanced.
- Identify rare categories.
- Identify categories with only one or very few records.
- Consider grouping rare categories if useful.
- Avoid grouping categories when rare values are analytically important.
  - Fraud labels
  - High-value customer types
  - Error codes

### 5.3 Identify high-cardinality variables

- Find categorical columns with many unique values.
  - Customer ID
  - Product ID
  - City
  - Search keyword
  - URL
  - Email domain
- Determine whether the variable is useful for analysis.
- Decide whether to aggregate high-cardinality fields.
  - Top categories vs other
  - Region from city
  - Product family from SKU
  - Domain type from email domain
- Consider whether high-cardinality fields may cause overfitting in modeling.

### 5.4 Check category consistency

- Look for duplicated categories caused by formatting issues.
- Check case sensitivity.
- Check whitespace issues.
- Check spelling variants.
- Check inconsistent abbreviations.
- Check language or localization differences.
- Standardize category labels where appropriate.

### 5.5 Analyze ordinal categories

- Identify categories with a natural order.
  - Low, medium, high
  - Bronze, silver, gold, platinum
  - Small, medium, large
  - Beginner, intermediate, advanced
- Confirm the correct order with business context.
- Avoid treating ordinal categories as nominal if order matters.
- Avoid treating ordinal categories as equally spaced unless justified.

### 5.6 Analyze binary flags

- Confirm that binary fields contain only expected values.
- Check class balance.
- Confirm the positive and negative meanings.
  - `1` may mean active, purchased, churned, or flagged depending on context.
- Check whether missing values should be interpreted as false or unknown.
- Compare binary flags with related variables for consistency.

### 5.7 Compare categorical variables to business expectations

- Check whether all expected categories appear.
- Check whether unexpected categories appear.
- Compare category distribution with known business mix.
  - Expected regional split
  - Expected channel mix
  - Expected plan distribution
  - Expected customer type distribution
- Investigate large deviations from expectations.

### 5.8 Summarize categorical insights

- Identify dominant categories.
- Identify rare but important categories.
- Identify messy columns requiring standardization.
- Identify categorical variables useful for segmentation.
- Identify categorical variables that may need encoding before modeling.

---

## 6. Explore Relationships Between Variables

### 6.1 Analyze numeric vs numeric relationships

- Evaluate relationships between pairs of numeric variables.
- Check whether relationships are linear or non-linear.
- Check whether relationships differ by segment.
- Identify strong positive associations.
- Identify strong negative associations.
- Identify weak or no associations.
- Watch for misleading correlations caused by outliers.
- Watch for spurious correlations caused by time trends or aggregation.

### 6.2 Analyze categorical vs numeric relationships

- Compare numeric metrics across categories.
  - Average revenue by region
  - Median order value by channel
  - Usage by customer segment
  - Support tickets by plan type
- Review both mean and median where distributions are skewed.
- Review sample size for each category.
- Avoid overinterpreting categories with very small counts.
- Identify categories with unusually high or low numeric values.
- Check whether differences are practically meaningful, not just visible.

### 6.3 Analyze categorical vs categorical relationships

- Cross-tabulate key categorical variables.
- Check how category distributions change across groups.
- Identify combinations that are common or rare.
- Identify impossible or unexpected combinations.
  - Inactive status with active subscription
  - Enterprise plan with consumer segment
  - Delivered order with missing shipping date
- Review proportions, not only raw counts.
- Look for segment concentration.

### 6.4 Analyze relationships involving dates

- Compare numeric metrics over time.
- Compare category composition over time.
- Check whether relationships change by month, quarter, or year.
- Identify cohort effects.
- Compare behavior before and after important events.
- Check whether recent data behaves differently from older data.

### 6.5 Identify multivariate patterns

- Explore combinations of more than two variables.
- Compare target or key metric across multiple dimensions.
  - Revenue by region and channel
  - Churn by tenure and plan
  - Conversion by device and traffic source
- Identify interaction effects.
  - A variable may matter only for one segment.
  - A category may perform well in one region but poorly in another.
- Avoid assuming one global pattern applies to all groups.

### 6.6 Check correlation and redundancy

- Identify highly correlated numeric variables.
- Identify variables that appear to measure the same concept.
- Identify duplicate or derived fields.
- Determine whether correlated variables are useful or redundant.
- Consider implications for modeling.
  - Multicollinearity
  - Feature selection
  - Interpretability

### 6.7 Check causality assumptions carefully

- Do not treat correlation as causation.
- Ask whether the relationship could be explained by another variable.
- Check whether the timing supports a causal interpretation.
- Watch for reverse causality.
- Identify confounding variables.
- Frame relationship findings as associations unless causal evidence exists.

### 6.8 Summarize relationship insights

- Highlight the strongest associations.
- Highlight surprising non-relationships.
- Highlight segment-level differences.
- Highlight relationships that require deeper statistical testing.
- Highlight relationships that may guide feature engineering or business action.

---

## 7. Analyze Target Variable, If Present

### 7.1 Identify the target variable

- Confirm which column represents the target outcome.
- Clarify the business definition of the target.
- Confirm whether the target is:
  - Classification target
  - Regression target
  - Ranking target
  - Time-to-event target
  - Multi-class target
  - Multi-label target
- Confirm whether the target was available at the right time for modeling.
- Check for possible target leakage.

### 7.2 Analyze classification targets

- Check class counts.
- Check class percentages.
- Identify class imbalance.
- Determine whether imbalance is expected.
- Check whether minority classes are meaningful.
- Compare class distribution across segments.
- Compare class distribution across time.
- Identify whether certain classes are underrepresented.

### 7.3 Analyze regression targets

- Review target distribution.
- Check mean, median, min, max, and percentiles.
- Check skewness and extreme values.
- Identify whether the target has many zeros.
- Identify whether the target has negative values.
- Determine whether transformations may be needed.
- Check whether the target has natural bounds.

### 7.4 Analyze target by feature groups

- Compare target behavior across categorical variables.
  - Churn by plan
  - Revenue by region
  - Conversion by source
- Compare target behavior across numeric variable bands.
  - Churn by tenure bucket
  - Spend by age group
  - Risk by usage tier
- Identify features that show strong separation in target behavior.
- Identify features with weak or no relationship to the target.

### 7.5 Check target stability over time

- Review target rate or target average over time.
- Check for sudden spikes or drops.
- Identify seasonal variation.
- Identify long-term drift.
- Determine whether training and future scoring periods may differ.
- Check whether changes are caused by business events, product changes, or data pipeline changes.

### 7.6 Validate target quality

- Check missing target values.
- Check invalid target values.
- Check inconsistent target labeling.
- Check whether target labels were delayed.
- Check whether target values were updated after initial capture.
- Confirm whether target definition changed over time.

### 7.7 Evaluate leakage risk

- Identify features created after the target event.
- Identify columns that directly encode the outcome.
  - Cancellation reason when predicting churn
  - Fraud investigation result when predicting fraud
  - Final status when predicting approval
- Identify time windows that may leak future information.
- Remove or flag leakage-prone variables before modeling.

### 7.8 Summarize target insights

- Describe target distribution.
- Describe imbalance or skew.
- Identify important target-related segments.
- Identify target quality concerns.
- Identify modeling implications.

---

## 8. Explore Time Trends, If Dates Exist

### 8.1 Identify time-related columns

- List all date and timestamp fields.
- Determine the meaning of each time field.
  - Created date
  - Updated date
  - Transaction date
  - Event timestamp
  - Signup date
  - Cancellation date
  - Delivery date
- Identify the primary time column for analysis.
- Check whether multiple time columns represent different lifecycle stages.

### 8.2 Validate date and time fields

- Check invalid dates.
- Check missing dates.
- Check future dates where not expected.
- Check dates before plausible business start dates.
- Check inconsistent date formats.
- Check timezone issues.
- Check whether timestamps are stored in local time or UTC.
- Check whether daylight saving changes may matter.

### 8.3 Define the analysis period

- Identify earliest and latest dates.
- Confirm whether the date range matches expectations.
- Check whether the first or last period is incomplete.
- Decide whether to exclude partial periods.
- Identify gaps in data coverage.
- Check whether historical data is backfilled or real-time collected.

### 8.4 Analyze volume over time

- Track row counts over time.
- Track transaction counts over time.
- Track active users or customers over time.
- Identify spikes or drops in volume.
- Determine whether volume changes are real or due to data issues.
- Compare weekday, weekend, monthly, quarterly, and yearly patterns.

### 8.5 Analyze key metrics over time

- Track important numeric metrics over time.
  - Revenue
  - Cost
  - Orders
  - Conversion rate
  - Churn rate
  - Average order value
  - Usage
- Compare short-term fluctuations with long-term trends.
- Use appropriate time aggregation.
  - Hourly
  - Daily
  - Weekly
  - Monthly
  - Quarterly
- Avoid overly granular analysis if the data is noisy.

### 8.6 Analyze seasonality

- Check day-of-week patterns.
- Check week-of-month patterns.
- Check month-of-year patterns.
- Check holiday or event effects.
- Check fiscal calendar patterns if relevant.
- Compare current period to similar prior periods.

### 8.7 Analyze cohorts

- Define cohorts based on start date or first event.
  - Signup month
  - First purchase month
  - Acquisition campaign
  - Product launch period
- Track behavior over cohort age.
  - Retention
  - Repeat purchase
  - Revenue growth
  - Product usage
- Compare cohorts to identify improvements or declines.
- Watch for incomplete recent cohorts.

### 8.8 Analyze before-and-after effects

- Identify major business events.
  - Product launch
  - Pricing change
  - Marketing campaign
  - Policy change
  - Data pipeline change
- Compare metrics before and after the event.
- Check whether changes are immediate or gradual.
- Control for seasonality where possible.
- Avoid claiming causality without proper design.

### 8.9 Check time-based data leakage

- For modeling use cases, confirm that features occur before the prediction point.
- Ensure future information is not included in historical rows.
- Use time-based splits where appropriate.
- Avoid random splits when temporal ordering matters.
- Check whether aggregated features include future periods.

### 8.10 Summarize time-based insights

- Highlight major trends.
- Highlight seasonal patterns.
- Highlight sudden anomalies.
- Highlight incomplete periods.
- Highlight potential data pipeline changes.
- Highlight modeling implications for train-test splitting.

---

## 9. Detect Outliers and Anomalies

### 9.1 Define what counts as an outlier

- Distinguish statistical outliers from business anomalies.
- Determine whether outliers are impossible, unlikely, or simply rare.
- Use business context to judge outliers.
- Avoid removing outliers only because they are extreme.
- Document the logic used to define outliers.

### 9.2 Identify numeric outliers

- Review extreme minimum and maximum values.
- Review percentile ranges.
  - 1st percentile
  - 5th percentile
  - 95th percentile
  - 99th percentile
- Check values far outside the interquartile range.
- Check values many standard deviations from the mean.
- Check whether outliers cluster in specific groups or dates.

### 9.3 Identify categorical anomalies

- Find rare categories.
- Find unexpected category combinations.
- Find categories appearing only in certain time periods.
- Find new categories that appear suddenly.
- Find categories that disappear unexpectedly.
- Investigate whether rare categories are data errors or meaningful edge cases.

### 9.4 Identify time-based anomalies

- Detect sudden spikes or drops in counts.
- Detect sudden changes in metric averages.
- Detect missing time periods.
- Detect repeated timestamp patterns that suggest batch processing.
- Detect unusual activity at odd times.
- Compare anomalies with known business or system events.

### 9.5 Identify entity-level anomalies

- Find customers, accounts, users, or products with unusual behavior.
  - Extremely high spend
  - Extremely frequent transactions
  - Extremely high return rate
  - Extremely short or long tenure
  - Unusual usage patterns
- Compare entity behavior against peer groups.
- Check whether anomalies are caused by duplicates or joins.

### 9.6 Investigate root causes

- Determine whether an outlier is caused by:
  - Data entry error
  - Unit mismatch
  - Duplicate record
  - System bug
  - Fraud or abuse
  - Legitimate rare behavior
  - Business event
  - Data integration issue
- Trace suspicious values back to source fields if possible.
- Ask domain experts about extreme but plausible cases.

### 9.7 Decide how to treat outliers

- Possible treatments include:
  - Keep unchanged
  - Correct known errors
  - Remove invalid records
  - Cap or winsorize values
  - Transform skewed variables
  - Segment outliers separately
  - Create anomaly flags
  - Exclude only from certain calculations
- Choose treatment based on use case.
  - Reporting may need accurate raw values.
  - Modeling may need robust handling.
  - Fraud analysis may focus specifically on anomalies.

### 9.8 Document outlier impact

- Quantify how many records are affected.
- Quantify how outliers affect key metrics.
  - Mean vs median
  - Total revenue
  - Conversion rate
  - Standard deviation
- Compare results with and without outliers.
- Document whether conclusions change after treatment.

### 9.9 Summarize anomaly findings

- List major anomalies found.
- Explain likely causes.
- State recommended treatment.
- Flag unresolved anomalies for follow-up.
- Highlight business opportunities or risks revealed by anomalies.

---

## 10. Summarize Insights and Next Steps

### 10.1 Create an executive summary

- Summarize the most important findings in plain language.
- Focus on what matters to the decision-maker.
- Include only high-impact observations.
- Avoid overwhelming the summary with technical details.
- Structure the summary around key themes.
  - Data quality
  - Business performance
  - Customer behavior
  - Risk areas
  - Modeling readiness

### 10.2 Separate facts, interpretations, and recommendations

- Facts are direct observations from the data.
  - “The dataset contains 1.2 million rows.”
  - “23% of records are missing customer segment.”
- Interpretations explain what those facts may mean.
  - “Missing customer segment is concentrated in older records, suggesting a historical tracking issue.”
- Recommendations suggest what to do next.
  - “Exclude pre-tracking records from segment-level analysis or add an unknown segment flag.”

### 10.3 Highlight data quality issues

- Summarize missing value concerns.
- Summarize duplicate concerns.
- Summarize invalid value concerns.
- Summarize inconsistent category concerns.
- Summarize date and time issues.
- Estimate the business or analytical impact of each issue.
- Recommend cleaning actions.

### 10.4 Highlight key patterns

- Summarize important distributions.
- Summarize important relationships.
- Summarize important segment differences.
- Summarize important time trends.
- Summarize important target variable behavior.
- Use specific numbers where available.
- Avoid vague statements when measurable evidence exists.

### 10.5 Highlight risks and limitations

- Note missing data limitations.
- Note sample bias concerns.
- Note incomplete time periods.
- Note small sample sizes in certain groups.
- Note possible data leakage risks.
- Note assumptions made during analysis.
- Note unresolved questions.

### 10.6 Recommend data cleaning next steps

- Standardize column names.
- Correct data types.
- Handle missing values.
- Remove or resolve duplicates.
- Standardize categorical values.
- Validate date fields.
- Treat invalid values.
- Create flags for important data quality issues.
- Build a reproducible cleaning pipeline.

### 10.7 Recommend feature engineering next steps

- Create time-based features.
  - Month
  - Quarter
  - Day of week
  - Tenure
  - Recency
- Create ratio features.
  - Revenue per order
  - Cost per acquisition
  - Usage per active day
- Create aggregate features.
  - Total spend by customer
  - Number of transactions per account
  - Average usage per week
- Create categorical groupings.
  - Region group
  - Product family
  - Customer tier
- Create missingness indicators where missingness is meaningful.

### 10.8 Recommend modeling next steps, if relevant

- Determine whether the dataset is suitable for modeling.
- Identify likely useful predictors.
- Identify variables to exclude due to leakage.
- Identify variables needing encoding.
- Identify variables needing scaling or transformation.
- Recommend train-test split approach.
  - Random split
  - Stratified split
  - Time-based split
  - Group-based split
- Recommend baseline model approach.
- Recommend evaluation metrics based on the problem type.

### 10.9 Recommend business next steps, if relevant

- Identify segments that need action.
- Identify opportunities for growth.
- Identify operational risks.
- Identify customer experience issues.
- Identify areas requiring deeper investigation.
- Suggest additional data that would improve future analysis.

### 10.10 Prepare final EDA deliverable

- Include the objective.
- Include data description.
- Include data quality summary.
- Include variable-level analysis.
- Include relationship analysis.
- Include target analysis, if present.
- Include time trend analysis, if present.
- Include outlier and anomaly analysis.
- Include key insights.
- Include limitations.
- Include recommendations.
- Include appendix or detailed notes where needed.

---

## Final EDA Review Checklist

Before considering the EDA complete, confirm the following:

- [ ] The objective is clearly defined.
- [ ] The business context is understood.
- [ ] The unit of analysis is documented.
- [ ] Dataset shape is recorded.
- [ ] Column meanings are understood or flagged for clarification.
- [ ] Data types are validated.
- [ ] Missing values are quantified and interpreted.
- [ ] Duplicate records are checked.
- [ ] Invalid values are identified.
- [ ] Numeric variables are profiled.
- [ ] Categorical variables are profiled.
- [ ] Important relationships are explored.
- [ ] Target variable is analyzed, if present.
- [ ] Date and time patterns are analyzed, if applicable.
- [ ] Outliers and anomalies are investigated.
- [ ] Data quality issues are summarized.
- [ ] Key insights are stated clearly.
- [ ] Limitations are documented.
- [ ] Cleaning recommendations are provided.
- [ ] Modeling or reporting recommendations are provided, if relevant.
- [ ] Business next steps are suggested, if relevant.
- [ ] All assumptions are documented.

---

## Suggested EDA Output Structure

A strong EDA report or notebook can follow this structure:

1. Objective and scope
2. Dataset overview
3. Data dictionary or column summary
4. Data quality assessment
5. Numeric variable analysis
6. Categorical variable analysis
7. Relationship analysis
8. Target variable analysis, if applicable
9. Time trend analysis, if applicable
10. Outlier and anomaly analysis
11. Key insights
12. Limitations and assumptions
13. Recommendations and next steps

---

## Guiding Principles for Good EDA

- Start with business questions, not charts.
- Understand the row grain before calculating metrics.
- Always check data quality before drawing conclusions.
- Use both summary statistics and visual thinking.
- Compare groups using both counts and percentages.
- Treat missingness as a signal, not just a problem.
- Do not remove outliers without understanding them.
- Be cautious with correlations and causal claims.
- Segment analysis often reveals patterns hidden in aggregate data.
- Document assumptions and decisions clearly.
- End with actionable insights and next steps.
