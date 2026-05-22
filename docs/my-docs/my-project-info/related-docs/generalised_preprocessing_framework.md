# Generalised Preprocessing Framework for Any ML Dataset

Preprocessing is the bridge between EDA and modeling.

EDA answers:

> “What is this data, can I trust it, and what patterns or risks exist?”

Preprocessing answers:

> “How do I convert this raw dataset into a clean, leakage-safe, model-ready dataset?”

The goal of preprocessing is **not** to blindly clean everything. The goal is to make deliberate decisions that improve model reliability while preserving real signal.

---

# 1. Start from the EDA summary

Do not begin preprocessing from scratch. Begin from the conclusions of your EDA.

From EDA, collect:

* Target type and distribution
* Unit of prediction
* Prediction time
* Train/test split strategy
* Missing value patterns
* Outliers and anomalies
* Wrong data types
* Duplicate rows or IDs
* High-cardinality categorical columns
* Skewed numerical features
* Leakage-risk columns
* Time-series structure
* Segment-level differences
* Train-test distribution differences

Ask:

* What issues must be fixed before modeling?
* What issues should be preserved because they contain signal?
* What issues are model-dependent?
* What issues are risky because they may cause leakage?

A useful preprocessing plan should come directly from EDA.

Example:

If EDA showed that customer income is missing mostly for self-employed users, preprocessing should probably not just fill income with the mean. It may need a missing indicator because the missingness itself may contain signal.

---

# 2. Define the prediction setup clearly

Before changing the data, lock the modeling setup.

Ask:

* What exactly are we predicting?
* At what moment is the prediction made?
* What information is available at that moment?
* What is one row in the modeling table?
* Is this classification, regression, forecasting, ranking, clustering, or anomaly detection?
* What metric will be used?
* How will train, validation, and test data be split?

This step prevents preprocessing leakage.

Bad:

> Imputing missing values using the full dataset before train-test split.

Good:

> Fit the imputer only on training data, then apply it to validation/test data.

Bad:

> Creating target encoding using both train and test categories.

Good:

> Learn target encoding only from training folds.

Preprocessing should always be designed around the real prediction scenario.

---

# 3. Create a preprocessing decision log

During preprocessing, maintain a simple log.

For each change, record:

| Column / Issue | Decision | Reason | Leakage risk? | Applied before or inside pipeline? |
|---|---|---|---|---|
| `age` has negative values | Set invalid ages to missing | Negative age is impossible | Low | Pipeline |
| `city` has 500 categories | Group rare cities | Reduce noise and overfitting | Medium | Pipeline |
| `last_payment_date` | Remove | Happens after prediction time | High | Before modeling |
| `income` missing | Median impute + missing flag | Missingness may be informative | Low | Pipeline |

This helps you avoid random, inconsistent preprocessing decisions.

A good rule:

> Every preprocessing step should have a reason.

---

# 4. Split the data before learning preprocessing rules

This is one of the most important preprocessing principles.

First decide the split:

* Random split
* Stratified split
* Group split
* Time-based split
* Group-time split
* Rolling validation
* Expanding window validation

Then fit preprocessing only on the training data.

Fit on train only:

* Imputation values
* Scaling parameters
* Encoding mappings
* Rare category thresholds
* Target encoding values
* Feature selection decisions
* PCA or dimensionality reduction
* Text vectorizers
* Time-series normalization statistics

Apply learned transformations to validation/test.

Why?

Because validation/test data should simulate unseen production data.

If preprocessing learns from validation/test, your score becomes over-optimistic.

---

# 5. Remove or isolate obvious leakage columns

Before cleaning, identify leakage.

Leakage columns include:

* Future information
* Post-outcome variables
* Columns created after the target event
* Target-derived variables
* IDs that encode the target
* Aggregates calculated using the full dataset
* Features with suspiciously perfect predictive power
* Time-series rolling features that use current or future values
* Duplicate target information in another column

Ask for every strong feature:

> Would I know this value at prediction time?

If no, then:

* Remove it
* Redesign it using only past information
* Move it to analysis-only features
* Confirm with domain knowledge

Examples:

Bad:

> `days_until_cancellation` when predicting churn before cancellation.

Good:

> `days_since_last_login` at prediction date.

Bad:

> Customer lifetime value calculated after the prediction date.

Good:

> Customer spend up to the prediction date only.

Leakage removal should happen early because leaked columns can distort all later preprocessing and feature selection.

---

# 6. Fix the unit of prediction

Preprocessing often fails because the dataset is not at the correct row level.

Ask:

* Should one row be one customer?
* One transaction?
* One session?
* One product per day?
* One user per month?
* One store per week?
* One event?

If raw data has multiple rows per entity, decide how to convert it into the modeling table.

Common operations:

* Aggregate transactions to customer level
* Aggregate events to time windows
* Convert logs into recency/frequency features
* Create one row per prediction date
* Create one row per entity-time pair
* Remove duplicate records
* Keep repeated rows if they represent real repeated events

Example:

For churn prediction, transaction-level data may need to become customer-level data:

* Number of purchases in last 30 days
* Total spend in last 90 days
* Days since last purchase
* Average order value
* Number of support tickets before prediction date

Always ensure aggregation uses only information available before prediction time.

---

# 7. Correct data types

Wrong data types create wrong preprocessing.

Check and fix:

* Numeric columns stored as strings
* Dates stored as objects
* Boolean columns stored as `Yes/No`, `0/1`, or `True/False`
* Categorical columns stored as numbers
* IDs accidentally treated as numerical features
* Currency symbols or commas in numeric fields
* Percentages stored as strings
* Mixed formats in the same column

Ask:

* Is this column truly numeric or just a code?
* Is this number ordinal or nominal?
* Is this date valid?
* Is this ID useful or should it be removed?
* Should this be treated as category?

Examples:

* ZIP code should usually be categorical, not numerical.
* Product ID is usually categorical or removable, not continuous.
* `20240101` may be a date, not an integer.
* `1`, `2`, `3` may represent categories, not quantities.

---

# 8. Handle duplicates carefully

Duplicates can be either errors or valid repeated observations.

Check:

* Fully duplicated rows
* Duplicate IDs
* Duplicate timestamps
* Duplicate entity-event pairs
* Same entity appearing in both train and test
* Duplicate target rows
* Near-duplicates

Ask:

* Is this a data collection issue?
* Does one row represent a repeated real event?
* Should duplicates be aggregated?
* Should only the latest record be kept?
* Are duplicates causing train-test leakage?

Examples:

For transaction data, repeated customer IDs are expected.

For a customer-level churn dataset, repeated customer IDs may be a problem unless each row represents a different prediction date.

For image/text/entity data, near-duplicates across train and test can inflate validation performance.

---

# 9. Handle missing values intentionally

Do not impute missing values automatically. First decide what missing means.

Ask:

* Is missing random?
* Is missing systematic?
* Does missing mean “not applicable”?
* Is missing related to the target?
* Is missing related to time?
* Is missing related to a segment?
* Will this value be missing in production?
* Should missingness itself become a feature?

Common missing value strategies:

| Situation | Possible treatment |
|---|---|
| Numeric missing randomly | Mean/median imputation |
| Numeric missing with skew/outliers | Median imputation |
| Missingness may be predictive | Add missing indicator |
| Missing means not applicable | Create explicit category or flag |
| Categorical missing | Add `Unknown` / `Missing` category |
| Time-series missing timestamps | Resample, interpolate, forward-fill, or mark gaps |
| Missing target | Usually remove for supervised learning |
| Feature mostly missing | Remove, keep if important, or create availability flag |

For numerical features:

* Mean imputation works when distribution is roughly symmetric.
* Median imputation is safer with skew and outliers.
* Constant imputation can be useful when missing has special meaning.
* Model-based imputation can help but may add complexity.

For categorical features:

* Use `Missing` or `Unknown` when absence is meaningful.
* Avoid filling with the mode blindly if missingness is systematic.
* Handle unseen categories in validation/test/production.

Important:

> Fit imputation values on training data only.

---

# 10. Handle outliers and anomalies

Outliers are not always bad. Some are errors, some are signal.

Ask:

* Is the value physically impossible?
* Is it a data entry error?
* Is it a valid rare case?
* Is it caused by unit mismatch?
* Is it important for the business problem?
* Is the model sensitive to outliers?
* Should I cap, transform, remove, or flag it?

Common strategies:

| Situation | Possible treatment |
|---|---|
| Impossible value | Set to missing or remove row |
| Extreme but valid value | Keep, cap, transform, or flag |
| Heavy-tailed numeric feature | Log / Box-Cox / Yeo-Johnson transform |
| Outlier caused by wrong unit | Standardize units |
| Rare but important event | Keep and possibly flag |
| Fraud/anomaly problem | Do not remove outliers blindly |

Model dependency:

* Linear models, KNN, SVM, PCA, and neural networks can be sensitive to outliers.
* Tree-based models are usually more robust but can still be affected.
* Metrics like RMSE are more sensitive to outliers than MAE.

Good practice:

Create an outlier flag when the extreme value may carry signal.

Example:

Instead of only capping transaction amount, create:

* `transaction_amount_capped`
* `is_high_transaction_amount`

---

# 11. Clean categorical variables

Categorical preprocessing has many decisions.

Check:

* Inconsistent labels
* Case differences
* Extra spaces
* Typos
* Rare categories
* High cardinality
* Ordinal vs nominal categories
* Unseen categories in validation/test
* Categories that are actually IDs

Basic cleanup:

* Trim whitespace
* Standardize case
* Merge spelling variants
* Normalize labels
* Convert missing to `Missing`
* Group rare categories
* Remove pure identifiers if not useful

Examples:

`Male`, `male`, `MALE`, `M` → `Male`

`New York`, `NY`, `N.Y.` may need standardization depending on context.

Ask:

* Is this feature nominal or ordinal?
* Are rare categories meaningful?
* Are categories stable over time?
* Could this category appear in production?
* Does this category encode leakage?

---

# 12. Choose encoding strategy for categorical features

Encoding depends on feature type, cardinality, model type, and leakage risk.

Common encoding methods:

| Encoding | Best for | Watch out |
|---|---|---|
| One-hot encoding | Low/medium-cardinality nominal features | Many columns for high cardinality |
| Ordinal encoding | True ordered categories or tree models | False ordering for linear models |
| Frequency/count encoding | High-cardinality features | May encode train distribution only |
| Target encoding | High-cardinality predictive categories | High leakage risk |
| Binary/hash encoding | Very high cardinality | Less interpretable |
| Embeddings | Deep learning / large categorical spaces | More complex |

Decision guide:

Use one-hot encoding when:

* Categories are not too many
* You use linear/logistic models
* Interpretability matters

Use ordinal encoding when:

* Categories have true order, such as low/medium/high
* You use tree models and can tolerate arbitrary integer codes

Use target encoding when:

* Cardinality is high
* Category has predictive signal
* You can use cross-validation or out-of-fold encoding safely

Use frequency encoding when:

* Category popularity itself may be useful
* You want a simple high-cardinality solution

Important:

> Target encoding must be done inside cross-validation or using training folds only.

---

# 13. Scale numerical features when needed

Not all models require scaling.

Scaling is important for:

* Linear regression with regularization
* Logistic regression with regularization
* KNN
* K-means
* SVM
* PCA
* Neural networks
* Gradient-based optimization
* Distance-based algorithms

Scaling is usually less critical for:

* Decision trees
* Random forests
* Gradient boosting trees
* Rule-based tree models

Common scaling methods:

| Method | Use when | Watch out |
|---|---|---|
| StandardScaler | Roughly symmetric distributions | Sensitive to outliers |
| MinMaxScaler | Need bounded range | Sensitive to outliers |
| RobustScaler | Outliers exist | Uses median/IQR |
| MaxAbsScaler | Sparse data | Does not center |
| Normalizer | Row-vector magnitude matters | Less common for tabular data |

Ask:

* Is the model distance-based or gradient-based?
* Are there outliers?
* Is the feature skewed?
* Is preserving sparsity important?
* Should scaling happen after imputation?

Important:

> Fit scalers on training data only.

---

# 14. Transform skewed numerical features

Skewed variables can hurt some models and make relationships harder to learn.

Check:

* Right skew
* Left skew
* Long tails
* Zero inflation
* Negative values
* Extreme values

Common transformations:

| Situation | Possible transformation |
|---|---|
| Positive right-skewed values | Log transform |
| Values include zero | `log1p` |
| Values include negatives | Yeo-Johnson |
| Heavy-tailed values | Winsorization / capping |
| Nonlinear relationship | Binning, splines, tree model |
| Zero-heavy distribution | Add zero flag + transform positive values |

Ask:

* Does the transformation make the relationship with target clearer?
* Does the model need this transformation?
* Does zero have special meaning?
* Are negative values allowed?
* Will transformation make interpretation harder?

Examples:

* Income, price, revenue, transaction amount often benefit from log transformation.
* Count variables may benefit from `log1p`.
* Highly skewed targets in regression may need target transformation.

---

# 15. Decide whether to transform the target

Target preprocessing depends on the task.

For classification:

* Encode class labels
* Check rare classes
* Decide binary vs multiclass vs multilabel
* Handle class imbalance
* Do not scale the target

For regression:

Consider target transformation when:

* Target is highly skewed
* Target has long tails
* Errors are multiplicative
* Model performs poorly on large values
* Predictions must remain positive

Common target transformations:

* Log transform
* `log1p`
* Box-Cox
* Yeo-Johnson
* Quantile transform

Important:

After target transformation, convert predictions back to original scale before interpretation and business evaluation.

For forecasting:

* Avoid using future target values in feature construction
* Use lagged target values carefully
* Align target horizon clearly
* Use appropriate time-based validation

---

# 16. Handle class imbalance

For classification, preprocessing may include imbalance handling.

First ask:

* Is the target imbalanced?
* Is the minority class important?
* What metric matters?
* Is the imbalance natural in production?
* Are there enough minority examples?
* Does imbalance differ by time or segment?

Possible strategies:

| Strategy | When useful |
|---|---|
| Stratified split | Most imbalanced classification tasks |
| Class weights | Logistic regression, SVM, tree models, neural nets |
| Oversampling | Small minority class |
| Undersampling | Very large majority class |
| SMOTE or synthetic sampling | Numeric feature spaces with enough minority examples |
| Threshold tuning | When probability ranking is good |
| Better metric | F1, recall, precision, PR-AUC, ROC-AUC, balanced accuracy |

Important:

> Apply oversampling only on the training data, never before train-test split.

For time-based problems, be careful with synthetic sampling because it may break time structure.

---

# 17. Create feature engineering rules from EDA

Feature engineering should be guided by patterns found during EDA.

Common numerical features:

* Ratios
* Differences
* Sums
* Percentages
* Interactions
* Bins
* Flags
* Log transforms
* Clipped values
* Deviation from group average

Common categorical features:

* Group rare categories
* Extract hierarchy
* Frequency encoding
* Target encoding
* Category combinations
* Cleaned labels

Common datetime features:

* Year
* Month
* Quarter
* Week
* Day of week
* Weekend flag
* Hour
* Is holiday
* Is month-end
* Is quarter-end
* Tenure
* Recency
* Time since last event

Common customer/event features:

* Recency
* Frequency
* Monetary value
* Activity rate
* Number of events in last N days
* Change compared with previous period
* Time since first event
* Time since last event

Always ask:

> Would this engineered feature be available at prediction time?

---

# 18. Special time-series preprocessing

If the dataset has time, add a time-aware preprocessing layer.

## A. Sort and align time

Before feature creation:

* Sort by entity and timestamp
* Check duplicate timestamps
* Check missing periods
* Align all entities to the same frequency if needed
* Define forecast horizon
* Define prediction cutoff time

## B. Use time-safe features

Useful time-series features:

* Lag values
* Rolling mean
* Rolling standard deviation
* Rolling min/max
* Expanding mean
* Previous period value
* Difference from previous period
* Percent change
* Time since last event
* Count of events in past window
* Seasonality indicators

Bad:

> Rolling average that includes the current target when predicting that target.

Good:

> Rolling average shifted so it only uses past values.

Example:

To predict tomorrow’s sales, use:

* Sales from yesterday
* Average sales over previous 7 days
* Average sales over previous 28 days
* Day-of-week
* Holiday flag

Do not use tomorrow’s actual sales in any feature.

## C. Fill missing time periods carefully

Options:

* Leave gaps
* Add missing rows
* Forward-fill
* Backward-fill
* Interpolate
* Fill with zero
* Add missing-period flag

Decision depends on meaning.

Example:

If no transaction occurred on a day, sales may be zero.

If sensor reading is missing, zero may be wrong.

## D. Use time-based validation

Prefer:

* Train past, validate future
* Rolling window validation
* Expanding window validation
* Group-time split for multiple entities

Avoid random split when time order matters.

---

# 19. Handle text features

If the dataset has text columns, decide whether to use them.

Check:

* Missing text
* Empty strings
* Very short text
* Duplicated text
* Language differences
* Personally identifiable information
* Leakage through text
* Labels embedded in text

Basic preprocessing:

* Lowercasing, if useful
* Removing extra whitespace
* Handling punctuation
* Tokenization
* Stopword removal, if useful
* Lemmatization/stemming, if useful
* TF-IDF vectorization
* Embeddings

Ask:

* Does the text exist at prediction time?
* Is there sensitive information?
* Does the text directly reveal the label?
* Is the dataset large enough for text modeling?
* Should text be modeled separately or combined with tabular features?

For modern NLP embeddings, heavy manual cleaning is often less necessary than for traditional bag-of-words models.

---

# 20. Handle IDs and high-cardinality features

IDs are tricky.

Types of ID-like columns:

* Pure row identifiers
* Entity IDs
* Product IDs
* User IDs
* Store IDs
* ZIP/postal codes
* Device IDs
* Transaction IDs

Ask:

* Is this a unique row identifier?
* Does this ID repeat meaningfully?
* Will this ID appear in production?
* Is there enough history for each ID?
* Could this ID leak the target?
* Should it be removed, encoded, grouped, or converted into aggregate features?

Common decisions:

| ID type | Possible treatment |
|---|---|
| Unique row ID | Remove |
| Customer ID with history | Use for group split or aggregate features |
| Product/store ID | Encode carefully if repeated |
| Transaction ID | Usually remove |
| ZIP/postal code | Treat as categorical or convert to region |
| Device/session ID | Usually remove or aggregate |

High cardinality can cause:

* Overfitting
* Huge feature space
* Unseen categories
* Slow training
* Poor generalization

Possible solutions:

* Remove
* Group rare categories
* Frequency encoding
* Target encoding safely
* Hashing
* Use embeddings
* Convert to aggregate historical features

---

# 21. Remove or reduce unhelpful features

Feature removal is also preprocessing.

Consider removing:

* Constant columns
* Near-constant columns
* Duplicate columns
* Pure identifiers
* Leakage columns
* Columns unavailable at prediction time
* Columns with excessive missingness and no signal
* Extremely high-cardinality columns with no strategy
* Redundant features, if model or interpretation requires it

Ask:

* Does this feature add information?
* Is it stable?
* Is it available in production?
* Is it too noisy?
* Does it duplicate another feature?
* Does it cause leakage?
* Does it make the model harder to maintain?

Model dependency:

* Linear models may need stronger feature selection.
* Tree models can tolerate more irrelevant features but still benefit from removing leakage and noise.
* Interpretability-focused models often need simpler feature sets.

---

# 22. Check train-validation-test consistency

After preprocessing design, verify consistency.

Check:

* Same columns after encoding
* Same feature order
* No unseen category crash
* No missing values left unexpectedly
* No infinite values
* No invalid data types
* Same scaling process
* No target leakage
* No train/test duplicate leakage
* Similar train/validation distributions
* Reasonable number of generated features

Ask:

* Can this exact pipeline process new production data?
* What happens if a category appears that was not in training?
* What happens if a column is missing in production?
* What happens if a numeric value is outside the training range?
* Are preprocessing steps reproducible?

---

# 23. Build preprocessing as a pipeline

Avoid manual preprocessing scattered across notebooks.

Use a pipeline mindset:

1. Raw data in
2. Validation checks
3. Type correction
4. Leakage column removal
5. Train-validation-test split
6. Fit preprocessing on train
7. Apply transformations to validation/test
8. Train model
9. Evaluate
10. Save pipeline and model together

The pipeline should include:

* Imputation
* Encoding
* Scaling
* Transformation
* Feature creation
* Feature selection, if applicable

Why pipelines matter:

* Prevent leakage
* Improve reproducibility
* Reduce train/serving mismatch
* Make experiments easier
* Make production deployment safer

A strong rule:

> Anything learned from data should be inside the training pipeline and fitted only on training data.

---

# 24. Validate preprocessing with a simple baseline model

Do not over-engineer before testing.

After initial preprocessing, train a simple baseline:

For classification:

* Logistic regression
* Decision tree
* Random forest
* Gradient boosting baseline
* Dummy classifier

For regression:

* Linear regression
* Decision tree
* Random forest
* Gradient boosting baseline
* Dummy regressor

For forecasting:

* Naive forecast
* Seasonal naive forecast
* Moving average
* Simple lag model

Use the baseline to check:

* Does the pipeline run end-to-end?
* Are features numeric/model-ready?
* Is validation performance reasonable?
* Are there suspiciously high scores?
* Is there leakage?
* Are errors concentrated in certain segments?
* Is preprocessing helping or hurting?

Suspicious signs:

* Validation score is unrealistically high
* Train score is very high but validation score is poor
* Feature importance is dominated by ID-like variables
* Validation performance collapses on future data
* Test categories are mostly unseen
* Predictions are outside valid ranges

---

# 25. Iterate preprocessing based on model feedback

Preprocessing is not a one-time step.

Use model results to improve decisions.

If model underfits:

* Add useful features
* Try nonlinear transformations
* Try interaction features
* Use better encoding
* Use more flexible models

If model overfits:

* Remove leakage
* Reduce high-cardinality noise
* Group rare categories
* Regularize
* Simplify features
* Use proper validation

If model performs poorly on minority class:

* Use class weights
* Tune threshold
* Improve features for minority class
* Use PR-AUC or recall-focused metrics
* Consider resampling carefully

If model performs poorly on certain segments:

* Add segment-specific features
* Check data quality in those segments
* Check if segments are underrepresented
* Check fairness and bias risks

If model performs poorly over time:

* Check drift
* Use time-based validation
* Add time-aware features
* Retrain more frequently
* Remove unstable features

---

# 26. Production-readiness checks

Before finalizing preprocessing, ask:

* Can this pipeline handle new data?
* What if a new category appears?
* What if a feature is missing?
* What if a numeric feature has extreme values?
* What if date format changes?
* What if data arrives late?
* Are all transformations deterministic?
* Are fitted preprocessing objects saved?
* Are training and inference preprocessing identical?
* Are assumptions documented?

Production-safe preprocessing should:

* Handle unseen categories
* Handle missing values
* Avoid manual notebook-only steps
* Avoid using target information
* Save fitted transformers
* Keep feature names consistent
* Log data quality issues
* Be version-controlled

---

# 27. Special considerations by model type

Different models need different preprocessing.

## Linear / Logistic Regression

Usually needs:

* Missing value imputation
* One-hot encoding
* Scaling
* Outlier treatment
* Multicollinearity checks
* Interaction terms if needed
* Target transformation for skewed regression target

Be careful with:

* High-cardinality categories
* Strong outliers
* Nonlinear relationships
* Redundant features

## Tree-based Models

Usually needs:

* Missing handling depending on implementation
* Categorical encoding
* Less need for scaling
* Less need for monotonic transformations
* Careful leakage removal
* High-cardinality handling

Be careful with:

* IDs
* Rare categories
* Leakage features
* Train-test drift

## Distance-based Models

Examples: KNN, K-means, SVM with distance kernels.

Usually needs:

* Scaling
* Outlier treatment
* Low-dimensional useful features
* Careful encoding
* Feature selection

Be careful with:

* High-dimensional one-hot features
* Unscaled variables
* Noisy features

## Neural Networks

Usually needs:

* Scaling
* Encoding or embeddings
* Missing handling
* Larger datasets
* Careful validation
* Regularization

Be careful with:

* Small datasets
* Unstable features
* Poorly scaled inputs

## Time-series Models

Usually needs:

* Time sorting
* Lag features
* Rolling features
* Seasonality features
* Missing timestamp handling
* Time-based validation

Be careful with:

* Future leakage
* Random split
* Unshifted rolling features
* Data availability delays

---

# 28. Common preprocessing mistakes

Avoid these:

* Imputing before train-test split
* Scaling before train-test split
* Target encoding on full data
* Oversampling before train-test split
* Random split for time-dependent data
* Removing outliers that are actually signal
* Treating IDs as numerical features
* One-hot encoding high-cardinality features blindly
* Filling missing values without understanding why they are missing
* Dropping columns only because they have missing values
* Using future data in engineered features
* Creating rolling features without shifting
* Ignoring unseen categories
* Preprocessing train and test separately
* Forgetting to apply the same transformations in production
* Letting validation/test information influence feature selection
* Keeping columns unavailable at prediction time

---

# 29. A simple preprocessing checklist you can follow every time

Use this order:

1. **Start from EDA conclusions**
   What issues did EDA reveal?

2. **Define prediction setup**
   Target, row unit, prediction time, metric, split strategy.

3. **Create decision log**
   Record what you change and why.

4. **Split data correctly**
   Random, stratified, group, time-based, or group-time split.

5. **Remove leakage**
   Drop or redesign future/post-outcome/target-derived features.

6. **Fix row level**
   Make sure one row matches one prediction unit.

7. **Correct data types**
   Numeric, categorical, datetime, boolean, ID, text.

8. **Handle duplicates**
   Remove, aggregate, or preserve depending on meaning.

9. **Handle missing values**
   Impute, flag, create category, remove, or preserve.

10. **Handle outliers**
    Keep, cap, transform, flag, fix, or remove.

11. **Clean categorical variables**
    Standardize labels, group rare categories, handle missing/unseen categories.

12. **Encode categorical variables**
    One-hot, ordinal, frequency, target encoding, hashing, embeddings.

13. **Scale numerical variables**
    Apply only when the model needs it.

14. **Transform skewed variables**
    Log, `log1p`, power transform, capping, binning.

15. **Handle target**
    Encode classes, transform skewed regression target, align forecast horizon.

16. **Handle imbalance**
    Class weights, sampling, threshold tuning, better metrics.

17. **Engineer features**
    Ratios, interactions, datetime features, lags, rolling stats, flags.

18. **Time-series preprocessing**
    Sort, align, shift lags, handle missing periods, use time-based validation.

19. **Handle text**
    Clean, vectorize, embed, and check leakage.

20. **Handle IDs/high cardinality**
    Remove, aggregate, encode safely, or use group splits.

21. **Remove unhelpful features**
    Constant, duplicate, unavailable, unstable, leakage-prone columns.

22. **Check consistency**
    Same columns, no unexpected missing/infinite values, no unseen-category failures.

23. **Build pipeline**
    Fit transformations on train only and reuse them.

24. **Train baseline**
    Check whether preprocessing produces sensible model behavior.

25. **Iterate**
    Improve preprocessing based on validation errors and segment analysis.

26. **Production checks**
    Ensure the pipeline can process future unseen data safely.

---

# 30. Decision guide by issue type

Use this table when you are confused during preprocessing.

| EDA finding | Preprocessing decision options | Key question |
|---|---|---|
| Missing numeric values | Mean, median, constant, model-based imputation, missing flag | Why is it missing? |
| Missing categorical values | `Missing` category, mode, remove, flag | Is missingness meaningful? |
| Skewed numeric feature | Log, `log1p`, power transform, cap, leave as-is | Does the model need transformation? |
| Extreme outliers | Keep, cap, remove, flag, fix units | Error or signal? |
| Inconsistent categories | Standardize labels, merge categories | Are these truly the same group? |
| Rare categories | Group as `Other`, encode differently, keep | Are rare categories meaningful or noisy? |
| High-cardinality category | Frequency, target encoding, hashing, embeddings, remove | Will this generalize? |
| ID column | Remove, group split, aggregate history, encode carefully | Is it a pure identifier or repeated entity? |
| Duplicate rows | Remove, aggregate, keep | Are they errors or real repeated events? |
| Time column | Extract features, create lags, use time split | What is known at prediction time? |
| Class imbalance | Stratify, class weights, sampling, threshold tuning | Which error type matters most? |
| Leakage column | Remove or redesign | Would this be known at prediction time? |
| Train-test drift | Time split, robust validation, drift-aware features | Is validation realistic? |
| Text column | Ignore, clean, TF-IDF, embeddings | Does text add signal safely? |
| Many columns | Feature selection, regularization, dimensionality reduction | Are features useful and stable? |

---

# 31. Recommended mental model

For every preprocessing decision, ask these five questions:

## 1. Is it valid?

Is the value possible and meaningful?

Example:

Age cannot be negative. A transaction amount may be zero only in some businesses.

## 2. Is it available?

Would this feature exist at prediction time?

Example:

A cancellation reason is not available before predicting churn.

## 3. Is it stable?

Will this feature behave similarly in future data?

Example:

A campaign code may be useful in historical data but absent in future campaigns.

## 4. Is it useful?

Does this transformation help the model learn?

Example:

Log-transforming income may help a linear model but may not matter much for a tree model.

## 5. Is it safe?

Could this step introduce leakage, bias, or production failure?

Example:

Target encoding without cross-validation can leak target information.

---

# 32. A good final preprocessing output should look like this

After preprocessing planning, you should be able to write:

> The dataset will be split using a time-based validation strategy because prediction depends on chronological order. Leakage columns such as A and B will be removed because they are created after the prediction time. Numerical columns will be median-imputed, with missing indicators added for C and D because missingness appears informative. Skewed monetary features will be transformed using `log1p`. Low-cardinality categorical features will be one-hot encoded, while high-cardinality columns such as city/product_id will be grouped by rare levels and frequency encoded. Outliers in transaction amount will be capped at the training-set percentile and flagged. All preprocessing steps will be fitted only on training data and implemented as a reproducible pipeline. The first baseline model will be trained after these transformations to verify that the pipeline is leakage-safe and model-ready.

That is the real purpose of preprocessing:

> To convert EDA insights into a clean, reproducible, leakage-safe, model-ready dataset.

---

# 33. Minimal preprocessing template

Use this template at the start of every project.

```text
Problem type:
Target:
Prediction time:
One row represents:
Validation strategy:
Evaluation metric:

Columns to remove due to leakage:
Columns to remove as IDs:
Columns requiring type correction:
Duplicate handling decision:

Missing value strategy:
Numerical imputation:
Categorical imputation:
Missing indicators:

Outlier strategy:
Categorical cleanup:
Encoding strategy:
Scaling strategy:
Transformation strategy:
Target transformation:

Feature engineering plan:
Time-series preprocessing plan:
Text preprocessing plan:
Class imbalance strategy:

Pipeline steps:
Production risks:
Baseline model:
Next iteration ideas:
```

---

# 34. Practical default choices

When you are unsure, these are reasonable starting defaults.

For general tabular classification/regression:

* Split before preprocessing.
* Use stratified split for imbalanced classification.
* Use time-based split if time matters.
* Remove obvious leakage and pure row IDs.
* Median-impute numerical features.
* Add missing indicators for important columns with meaningful missingness.
* Use `Missing` category for categorical nulls.
* One-hot encode low-cardinality categorical features.
* Group rare categories before encoding.
* Use frequency or carefully cross-validated target encoding for high-cardinality categoricals.
* Scale numerical features for linear, distance-based, or neural models.
* Do not worry much about scaling for tree-based models.
* Cap or flag extreme outliers instead of blindly deleting them.
* Use `log1p` for heavily skewed positive monetary/count features.
* Build everything as a pipeline.
* Validate with a simple baseline before adding complexity.

For time-series:

* Never random split if future prediction is the goal.
* Sort by time and entity.
* Use only past data for lags and rolling features.
* Shift rolling features properly.
* Handle missing timestamps based on business meaning.
* Use rolling or expanding validation.
* Check data availability delay.

For high-cardinality features:

* Remove pure unique IDs.
* Use group split if entity leakage is possible.
* Create historical aggregate features when meaningful.
* Use frequency encoding or target encoding safely.
* Always handle unseen categories.

---

# 35. Final reminder

Preprocessing is not about applying every possible cleaning technique.

It is about making the data:

* Valid
* Leakage-safe
* Model-ready
* Reproducible
* Robust to future data
* Aligned with the prediction problem

A strong preprocessing workflow should always connect back to your EDA.

EDA tells you what is wrong, what is useful, and what is risky.

Preprocessing decides what to do about it.
