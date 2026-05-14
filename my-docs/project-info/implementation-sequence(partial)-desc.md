In **EDA**, you usually **check** nulls/missing values/outliers/data types **before** descriptive statistics and visual analysis, but you usually **handle/fix** them after you understand them.

The reason is simple: if you blindly fill missing values before understanding the data, you may destroy useful patterns. But if you run full descriptive statistics without knowing missing values exist, your summaries may mislead you.

A good industry-style sequence is:

---

# 1. Get the data

This is the raw data collection stage.

Examples:

```text
CSV / Excel / SQL database / APIs / data warehouse / logs
```

At this point, do not start modeling. First understand what you received.

---

# 2. Understand the business problem

Before touching the data deeply, clarify:

```text
What are we predicting?
What is the target variable?
Is it classification, regression, ranking, forecasting, clustering?
What does success mean?
What metric matters?
```

Example:

```text
Predict whether a customer will churn.
Target column = churn
Problem type = classification
Metric = recall / F1 / ROC-AUC
```

This matters because your EDA and cleaning decisions depend on the goal.

---

# 3. Initial data inspection

This happens before full EDA.

You check:

```text
Number of rows and columns
Column names
Data types
Sample rows
Duplicate rows
Target column availability
Basic schema issues
```

Typical Python checks:

```python
df.shape
df.head()
df.info()
df.dtypes
df.columns
df.duplicated().sum()
```

At this stage, you are not doing deep analysis yet. You are just getting a first look.

---

# 4. Data quality checks

This is where nulls, missing values, invalid values, and obvious problems are checked.

You check:

```text
Missing values
Null values
Duplicate records
Incorrect data types
Invalid categories
Impossible values
Outliers
Inconsistent formatting
Target leakage
Class imbalance
```

Examples:

```python
df.isnull().sum()
df.nunique()
df.describe()
df['gender'].value_counts()
df['age'].min(), df['age'].max()
```

Important distinction:

```text
Check first.
Handle later after understanding.
```

For example, seeing missing values is only the first step. You should ask:

```text
Why is it missing?
Is it random?
Is missingness meaningful?
Should I drop it, fill it, or create a missing indicator?
```

---

# 5. Descriptive statistics

Now you summarize the data numerically.

For numerical columns:

```text
mean
median
standard deviation
min
max
quartiles
skewness
kurtosis
```

For categorical columns:

```text
frequency counts
unique values
mode
percentage distribution
cardinality
```

Example:

```python
df.describe()
df.describe(include='object')
```

This helps you understand scale, spread, unusual values, and distributions.

---

# 6. Univariate analysis

Univariate means analyzing **one variable at a time**.

For numerical variables:

```text
histogram
boxplot
density plot
skewness
outlier check
```

For categorical variables:

```text
bar chart
count plot
frequency table
```

Purpose:

```text
Understand individual column behavior.
```

Example questions:

```text
Is age normally distributed?
Is income highly skewed?
Are there rare categories?
Is one class dominating?
```

---

# 7. Bivariate analysis

Bivariate means analyzing **two variables together**.

Usually you compare each feature with the target.

Examples:

Numerical feature vs numerical target:

```text
scatter plot
correlation
regression line
```

Categorical feature vs numerical target:

```text
boxplot
grouped mean
ANOVA-style comparison
```

Categorical feature vs categorical target:

```text
cross-tab
stacked bar chart
chi-square test
```

Numerical feature vs categorical target:

```text
boxplot by class
violin plot
grouped statistics
```

Purpose:

```text
Find relationship between feature and target.
```

Example:

```text
Does higher income reduce churn?
Do customers with monthly contracts churn more?
Is age related to default risk?
```

---

# 8. Multivariate analysis

Multivariate means analyzing **multiple variables together**.

You check:

```text
Correlation between many numerical variables
Multicollinearity
Feature interactions
Group-level patterns
Pairplots
Heatmaps
Segment-wise behavior
```

Examples:

```python
df.corr()
sns.heatmap(df.corr())
```

Purpose:

```text
Understand how features interact with each other and with the target.
```

Example:

```text
Income alone may not predict churn,
but income + contract type + tenure may reveal a pattern.
```

---

# So where does missing value handling happen?

Usually:

```text
Missing value checking happens before descriptive/univariate/bivariate/multivariate analysis.
Missing value handling happens after initial EDA, before model training.
```

But practically, there are two levels of handling.

---

## Light cleaning before EDA

You can do basic fixes before EDA:

```text
Fix column names
Correct obvious data types
Remove exact duplicate rows
Standardize category names
Convert date columns
Fix impossible values if clearly wrong
```

Example:

```text
" Male", "male", "MALE" → "Male"
date column from object → datetime
age = -5 → invalid
```

This makes EDA easier.

---

## Serious cleaning after EDA

Do after understanding the data:

```text
Missing value imputation
Outlier treatment
Rare category grouping
Feature transformation
Encoding decisions
Scaling decisions
Leakage removal
```

Why after EDA?

Because the right treatment depends on what EDA reveals.

Example:

```text
If income is highly skewed, median imputation may be better than mean imputation.
If missing value itself predicts churn, create a missing-value flag.
If outliers are valid high-value customers, do not blindly remove them.
```

---

# Full sequence from raw data to model-ready data

Here is the clean sequence.

---

## Step 1: Define problem and target

Before data preparation, define:

```text
Business goal
ML problem type
Target variable
Prediction time
Success metric
```

Example:

```text
Goal: Predict loan default
Target: default_status
Problem: Binary classification
Metric: Recall and ROC-AUC
```

---

## Step 2: Load data

```python
import pandas as pd

df = pd.read_csv("data.csv")
```

Check:

```python
df.shape
df.head()
```

---

## Step 3: Initial inspection

Check:

```text
Rows and columns
Column names
Data types
Sample records
Target column
Duplicate rows
```

Code:

```python
df.info()
df.dtypes
df.duplicated().sum()
```

---

## Step 4: Basic data quality check

Check:

```text
Missing values
Invalid values
Unique values
Cardinality
Range issues
Wrong data types
Duplicate IDs
Target distribution
```

Code:

```python
df.isnull().sum()
df.nunique()
df.describe()
df['target'].value_counts(normalize=True)
```

Do not rush to impute yet. First understand.

---

## Step 5: Light cleaning before EDA

Do only obvious corrections:

```text
Rename messy columns
Strip spaces
Fix data types
Convert dates
Standardize categories
Remove exact duplicates
Handle obvious invalid records
```

Example:

```python
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
df['date'] = pd.to_datetime(df['date'])
```

---

## Step 6: Descriptive statistics

Analyze:

```text
Central tendency
Spread
Min/max
Quartiles
Category frequencies
```

Code:

```python
df.describe()
df.describe(include='object')
```

---

## Step 7: Univariate analysis

Analyze each column individually.

Questions:

```text
What is the distribution?
Is it skewed?
Are there outliers?
Are categories balanced?
Are there rare labels?
```

Outputs:

```text
Histogram
Boxplot
Countplot
Frequency table
```

---

## Step 8: Bivariate analysis

Analyze features against the target.

Questions:

```text
Which variables affect the target?
Are some groups riskier?
Do numerical features separate target classes?
Is there a strong relationship?
```

Examples:

```text
churn rate by contract type
average income by default status
age distribution by customer segment
```

---

## Step 9: Multivariate analysis

Analyze many variables together.

Questions:

```text
Are features correlated?
Are there redundant columns?
Are there interaction effects?
Is there multicollinearity?
Do combinations of features explain the target better?
```

Outputs:

```text
Correlation heatmap
Pairplot
Grouped summaries
Segment analysis
VIF check for linear models
```

---

# 10. Decide data cleaning strategy

Now, based on EDA, decide how to clean properly.

This includes:

```text
Missing value treatment
Outlier treatment
Duplicate treatment
Invalid value treatment
Rare category handling
Data leakage removal
```

Example decisions:

```text
Fill age with median
Fill city with "Unknown"
Cap income at 99th percentile
Group rare occupations as "Other"
Remove future-looking columns
```

This is where actual handling strongly comes into picture.

---

# 11. Split data into train and test

This is very important.

You should split before applying transformations that learn from data.

```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Why split before preprocessing?

Because you must avoid data leakage.

Bad practice:

```text
Fit imputer/scaler/encoder on full data before split
```

Good practice:

```text
Fit imputer/scaler/encoder only on training data
Apply the same transformation to test data
```

---

# 12. Missing value imputation

Now handle missing values properly.

For numerical columns:

```text
Mean imputation
Median imputation
Model-based imputation
KNN imputation
Missing indicator
```

For categorical columns:

```text
Mode imputation
"Unknown" category
Missing indicator
```

Common industry approach:

```text
Numerical skewed column → median
Numerical normal column → mean
Categorical column → mode or "Unknown"
Missingness meaningful → add missing flag
```

Example:

```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
```

---

# 13. Outlier treatment

Outlier handling depends on model type and business context.

Options:

```text
Keep outliers
Remove outliers
Cap/winsorize outliers
Log-transform skewed columns
Use robust scaler
```

Do not remove outliers blindly.

Example:

```text
Fraud detection: outliers may be very important.
Income prediction: extreme values may distort linear models.
Tree-based models: often less sensitive to outliers.
```

---

# 14. Feature engineering

Feature engineering means creating better input variables for the model.

Examples:

From date columns:

```text
year
month
day
weekday
is_weekend
customer_age_days
```

From transaction data:

```text
total_spend
average_order_value
number_of_transactions
days_since_last_purchase
```

From text/categorical data:

```text
domain from email
city tier
product category group
```

From numerical columns:

```text
income_per_family_member
debt_to_income_ratio
price_per_unit
```

This usually happens after EDA because EDA helps you discover useful patterns.

---

# 15. Encoding categorical variables

Machine learning models need numbers, so categorical features must be encoded.

Common methods:

```text
One-hot encoding
Ordinal encoding
Target encoding
Frequency encoding
Binary encoding
```

Use cases:

```text
Nominal categories → one-hot encoding
Ordered categories → ordinal encoding
High-cardinality categories → target/frequency encoding
```

Example:

```text
city has 500 unique values → one-hot may explode dimensions
education level → ordinal encoding may make sense
```

---

# 16. Feature transformation

This is used to improve distributions or relationships.

Common transformations:

```text
Log transformation
Square root transformation
Box-Cox transformation
Yeo-Johnson transformation
Binning
Polynomial features
Interaction features
```

Example:

```text
income is highly skewed → log(income)
age may have non-linear effect → age groups
```

---

# 17. Scaling / normalization / standardization

This is where normalization comes in.

Scaling is important for models that depend on distance or gradient optimization.

Needs scaling:

```text
Linear regression
Logistic regression
KNN
K-means
SVM
Neural networks
PCA
Gradient descent-based models
```

Usually does not require scaling:

```text
Decision tree
Random forest
XGBoost
LightGBM
CatBoost
```

Common scaling methods:

```text
Standardization: mean 0, standard deviation 1
Normalization: values between 0 and 1
Robust scaling: uses median and IQR
```

Example:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
```

Use:

```text
StandardScaler → general default for many ML models
MinMaxScaler → neural networks, bounded scale needs
RobustScaler → when outliers exist
```

---

# 18. Feature selection

After creating and transforming features, reduce unnecessary features.

Methods:

```text
Remove constant columns
Remove duplicate columns
Remove highly correlated features
Use mutual information
Use model-based importance
Use recursive feature elimination
Use domain knowledge
```

Why?

```text
Reduce overfitting
Improve model speed
Improve interpretability
Remove noise
```

---

# 19. Handle class imbalance

For classification problems, check class distribution.

Example:

```text
95% non-fraud
5% fraud
```

Techniques:

```text
Class weights
Oversampling
Undersampling
SMOTE
Threshold tuning
Proper metrics
```

Important: apply resampling only on the training data, not the test data.

---

# 20. Build preprocessing pipeline

In industry, preprocessing should be reproducible.

Instead of manually transforming train and test separately, use pipelines.

Example structure:

```text
Numerical columns:
missing value imputation → scaling

Categorical columns:
missing value imputation → encoding
```

Using scikit-learn:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_columns),
    ('cat', categorical_pipeline, categorical_columns)
])
```

---

# 21. Prepare final training data

Now the data is ready for model training.

Final prepared data should have:

```text
No unhandled missing values
Correct data types
No leakage columns
Encoded categorical variables
Scaled numerical variables if needed
Useful engineered features
Consistent train/test transformations
Target separated from features
```

Then:

```python
model.fit(X_train_processed, y_train)
```

Or with pipeline:

```python
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

pipeline.fit(X_train, y_train)
```

---

# Final recommended sequence

Here is the clean full order:

```text
1. Define business problem and target
2. Collect/load data
3. Initial data inspection
4. Data quality checks
5. Light cleaning
6. Descriptive statistics
7. Univariate analysis
8. Bivariate analysis
9. Multivariate analysis
10. Decide cleaning strategy
11. Train/test split
12. Missing value imputation
13. Outlier treatment
14. Feature engineering
15. Categorical encoding
16. Feature transformation
17. Scaling / normalization / standardization
18. Feature selection
19. Handle class imbalance
20. Build preprocessing pipeline
21. Train model
22. Evaluate model
23. Tune model
24. Final validation
25. Deploy / monitor
```

---

# Simple way to remember

Think of the process like this:

```text
Understand → Analyze → Decide → Transform → Train
```

More specifically:

```text
Before EDA:
Check structure, quality, nulls, duplicates, data types.

During EDA:
Understand distributions, relationships, patterns, outliers, imbalance.

After EDA:
Clean properly, impute, engineer features, encode, scale, select features.

Before model training:
Split data, build pipeline, prevent leakage, transform train/test consistently.
```

---

# Best practical answer to your main question

Null checking and missing value checking happen **before** descriptive statistics and univariate/bivariate/multivariate analysis.

But missing value **handling** usually happens **after initial EDA**, because EDA tells you the best way to handle it.

So the answer is:

```text
Check missing/null values before EDA.
Understand them during EDA.
Handle them after EDA, before model training.
```

That is the industry-friendly workflow.
