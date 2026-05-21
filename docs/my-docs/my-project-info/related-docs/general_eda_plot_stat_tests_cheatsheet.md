# General EDA Plot And Statistical Test Cheatsheet

This cheatsheet maps common EDA questions to useful plots, equivalent statistics, and statistical tests. It is written for general EDA and includes a dedicated time series section for forecasting work.

---

## How To Use This Cheatsheet

For every EDA question, think in this order:

1. **What am I trying to understand?**
   - Distribution, outliers, relationships, group differences, missingness, time patterns, stationarity, seasonality, or forecast readiness.

2. **What plot should I use?**
   - Pick the visual that reveals the pattern most directly.

3. **What statistics support the plot?**
   - Use summary statistics and tests to confirm whether the visual pattern is meaningful.

4. **What assumptions matter?**
   - Check normality, variance equality, independence, sample size, and time dependence before selecting a formal test.

---

## Quick Decision Guide

### If You Want To Understand A Single Numeric Variable

- **Use plots:** histogram, KDE plot, box plot, violin plot, Q-Q plot.
- **Use stats:** mean, median, mode, standard deviation, variance, IQR, min, max, skewness, kurtosis, coefficient of variation.
- **Use tests:** Shapiro-Wilk, Jarque-Bera, D'Agostino K-squared, Anderson-Darling, Kolmogorov-Smirnov.

### If You Want To Find Outliers

- **Use plots:** box plot, scatter plot, histogram, violin plot.
- **Use stats:** IQR bounds, z-score, modified z-score, MAD, percentile thresholds.
- **Use tests:** Grubbs test, Dixon Q test, generalized ESD test.

### If You Want To Compare Two Numeric Groups

- **Use plots:** side-by-side box plot, violin plot, bar chart with confidence interval, strip plot.
- **Use stats:** group means, medians, standard deviations, IQRs, confidence intervals, effect size.
- **Use tests:** independent t-test, Welch's t-test, Mann-Whitney U test, paired t-test, Wilcoxon signed-rank test.

### If You Want To Compare More Than Two Numeric Groups

- **Use plots:** grouped box plot, grouped violin plot, bar chart with confidence intervals, swarm plot.
- **Use stats:** group means, medians, standard deviations, IQRs, within-group and between-group variance.
- **Use tests:** one-way ANOVA, Welch's ANOVA, Kruskal-Wallis test, repeated-measures ANOVA, Friedman test.
- **Use post-hoc tests:** Tukey HSD, Games-Howell, Dunn test with p-value correction.

### If You Want To Study Relationship Between Two Numeric Variables

- **Use plots:** scatter plot, regression plot, hexbin plot, 2D density plot.
- **Use stats:** covariance, Pearson correlation, Spearman correlation, Kendall Tau, regression slope, R-squared.
- **Use tests:** Pearson correlation significance test, Spearman rank correlation test, Kendall Tau test, linear regression t-test for slope.

### If You Want To Study Many Numeric Relationships

- **Use plots:** correlation heatmap, pair plot, scatter matrix, clustered heatmap.
- **Use stats:** correlation matrix, covariance matrix, VIF for multicollinearity.
- **Use tests:** pairwise correlation significance tests, FDR correction, Bonferroni correction, variance inflation factor checks.

### If You Want To Study Categorical Variables

- **Use plots:** count plot, bar chart, stacked bar chart, mosaic plot.
- **Use stats:** frequency counts, proportions, mode, entropy, cardinality.
- **Use tests:** chi-square goodness-of-fit test, chi-square test of independence, Fisher's exact test, G-test.

### If You Want To Study Numeric Vs Categorical Variables

- **Use plots:** box plot by category, violin plot by category, strip plot, grouped bar chart.
- **Use stats:** group mean, group median, group standard deviation, group IQR, effect size.
- **Use tests:** t-test, Welch's t-test, Mann-Whitney U, ANOVA, Welch's ANOVA, Kruskal-Wallis.

### If You Want To Study Missing Values

- **Use plots:** missingness heatmap, missingness bar chart, matrix plot, nullity correlation plot.
- **Use stats:** missing count, missing percentage, row-wise missingness, column-wise missingness.
- **Use tests:** Little's MCAR test, chi-square tests for missingness association, logistic regression missingness model.

---

## Distribution Analysis

### Histogram

**What it shows:**  
Frequency distribution of a numeric variable.

**Use it for:**  
Checking shape, skewness, heavy tails, multi-modality, zero inflation, and unusual spikes.

**Equivalent stats to use:**

- Mean and median to compare center.
- Standard deviation and IQR to compare spread.
- Skewness to measure asymmetry.
- Kurtosis to measure tail heaviness.
- Coefficient of variation for relative variability.

**Tests that can be used:**

- Shapiro-Wilk for normality, especially smaller samples.
- Jarque-Bera for skewness/kurtosis-based normality.
- D'Agostino K-squared for omnibus normality.
- Anderson-Darling for distribution fit, often stronger in tails.
- Kolmogorov-Smirnov for comparing against a reference distribution.

**EDA interpretation:**

- If mean is much greater than median, the distribution is likely right-skewed.
- If there are many zeros, consider zero-inflated or intermittent behavior.
- If tails are heavy, robust statistics may be safer than mean/std.

### KDE Plot

**What it shows:**  
Smoothed estimate of the distribution.

**Use it for:**  
Comparing distribution shapes across groups.

**Equivalent stats to use:**

- Mean, median, IQR, skewness, kurtosis.
- Group-level summary statistics when multiple KDE curves are shown.

**Tests that can be used:**

- Kolmogorov-Smirnov two-sample test for distribution differences.
- Anderson-Darling k-sample test for distribution differences.
- Mann-Whitney U test when comparing location differences between two groups.
- Kruskal-Wallis test when comparing more than two groups.

### Q-Q Plot

**What it shows:**  
Whether a variable follows a theoretical distribution, usually normal.

**Use it for:**  
Checking normality and tail behavior before using parametric tests.

**Equivalent stats to use:**

- Skewness.
- Kurtosis.
- Normality test p-values.

**Tests that can be used:**

- Shapiro-Wilk.
- Jarque-Bera.
- D'Agostino K-squared.
- Anderson-Darling.

**EDA interpretation:**

- Points close to the diagonal suggest approximate normality.
- Curved ends suggest heavy or light tails.
- Strong S-shape suggests skewness.

---

## Outlier Analysis

### Box Plot

**What it shows:**  
Median, quartiles, IQR, and potential outliers.

**Use it for:**  
Detecting spread, skewness, and extreme values in one variable or across groups.

**Equivalent stats to use:**

- Median.
- Q1 and Q3.
- IQR.
- Lower bound: Q1 - 1.5 x IQR.
- Upper bound: Q3 + 1.5 x IQR.
- Percentile thresholds such as P1/P99 or P5/P95.

**Tests that can be used:**

- Grubbs test for one outlier in approximately normal data.
- Dixon Q test for small samples.
- Generalized ESD test for multiple outliers.
- Robust z-score using median and MAD.

**EDA interpretation:**

- Outliers are not automatically errors.
- In business data, outliers may represent events, promotions, holidays, or real demand spikes.

### Violin Plot

**What it shows:**  
Distribution shape and density, often by group.

**Use it for:**  
Comparing group distributions beyond just median and IQR.

**Equivalent stats to use:**

- Median, IQR, mean, standard deviation.
- Group-level skewness and kurtosis.

**Tests that can be used:**

- Mann-Whitney U for two independent non-normal groups.
- Kruskal-Wallis for more than two non-normal groups.
- ANOVA if groups are approximately normal with similar variances.

---

## Relationship Analysis

### Scatter Plot

**What it shows:**  
Relationship between two numeric variables.

**Use it for:**  
Detecting linear relationship, non-linear relationship, clusters, heteroscedasticity, and outliers.

**Equivalent stats to use:**

- Pearson correlation for linear relationships.
- Spearman correlation for monotonic relationships.
- Kendall Tau for rank association.
- Regression slope and intercept.
- R-squared.

**Tests that can be used:**

- Pearson correlation significance test.
- Spearman rank correlation test.
- Kendall Tau test.
- Linear regression slope t-test.
- Breusch-Pagan test for heteroscedasticity.

**EDA interpretation:**

- Pearson can miss non-linear patterns.
- Spearman is safer when relationships are monotonic but not linear.
- Outliers can strongly distort correlation.

### Correlation Heatmap

**What it shows:**  
Pairwise correlations across many numeric variables.

**Use it for:**  
Detecting relationships, redundancy, multicollinearity, and feature groups.

**Equivalent stats to use:**

- Pearson correlation matrix.
- Spearman correlation matrix.
- Kendall correlation matrix.
- Covariance matrix.
- VIF for multicollinearity.

**Tests that can be used:**

- Pearson correlation significance test.
- Spearman rank correlation test.
- Kendall Tau test.
- Benjamini-Hochberg FDR correction for many pairwise tests.
- Bonferroni correction for stricter multiple-testing control.

**EDA interpretation:**

- High correlation can indicate redundant features.
- Low correlation does not mean no relationship; the relationship may be non-linear.
- For many features, always consider multiple-testing correction.

### Pair Plot

**What it shows:**  
Multiple scatter plots and distributions together.

**Use it for:**  
Quickly scanning pairwise relationships among selected numeric features.

**Equivalent stats to use:**

- Pairwise correlations.
- Distribution stats for each variable.
- Group-wise summaries if colored by category.

**Tests that can be used:**

- Pairwise correlation significance tests.
- ANOVA/Kruskal-Wallis if color groups show separation.
- MANOVA when testing multivariate group separation.

---

## Group Comparison Analysis

### Bar Chart With Error Bars

**What it shows:**  
Group-level mean, median, count, or proportion with uncertainty.

**Use it for:**  
Comparing categories, stores, departments, treatments, or cohorts.

**Equivalent stats to use:**

- Group mean or median.
- Standard deviation.
- Standard error.
- Confidence interval.
- Count per group.

**Tests that can be used:**

- Independent t-test for two independent normal groups.
- Welch's t-test for two groups with unequal variances.
- Paired t-test for paired before/after measurements.
- ANOVA for more than two normal groups.
- Welch's ANOVA for more than two groups with unequal variances.
- Mann-Whitney U for two non-normal independent groups.
- Wilcoxon signed-rank for paired non-normal data.
- Kruskal-Wallis for more than two non-normal groups.

**EDA interpretation:**

- Avoid comparing only bar heights without uncertainty.
- Very different group sizes can make visual differences misleading.

### Grouped Box Plot

**What it shows:**  
Distribution of a numeric variable across categories.

**Use it for:**  
Comparing spread, median, skewness, and outliers across groups.

**Equivalent stats to use:**

- Group median.
- Group IQR.
- Group mean and standard deviation.
- Effect size such as Cohen's d, Cliff's delta, or eta-squared.

**Tests that can be used:**

- Levene's test for equal variances.
- Bartlett's test for equal variances if normality is reasonable.
- ANOVA or Welch's ANOVA.
- Kruskal-Wallis.
- Tukey HSD for post-hoc pairwise comparison after ANOVA.
- Dunn test for post-hoc pairwise comparison after Kruskal-Wallis.

---

## Categorical Analysis

### Count Plot

**What it shows:**  
Frequency of each category.

**Use it for:**  
Checking class imbalance, rare categories, and dominant groups.

**Equivalent stats to use:**

- Counts.
- Proportions.
- Mode.
- Cardinality.
- Entropy.

**Tests that can be used:**

- Chi-square goodness-of-fit test.
- Binomial test for two-category proportion checks.
- Proportion z-test for comparing proportions.

### Stacked Bar Chart

**What it shows:**  
Relationship between two categorical variables.

**Use it for:**  
Comparing composition across categories.

**Equivalent stats to use:**

- Cross-tabulation.
- Row percentages.
- Column percentages.
- Odds ratio for 2x2 tables.

**Tests that can be used:**

- Chi-square test of independence.
- Fisher's exact test for small expected counts.
- G-test.
- Cramer's V for association strength.

---

## Missing Value Analysis

### Missingness Heatmap

**What it shows:**  
Where missing values occur across rows and columns.

**Use it for:**  
Detecting missingness patterns, blocks, and systematic data gaps.

**Equivalent stats to use:**

- Missing count by column.
- Missing percentage by column.
- Missing count by row.
- Nullity correlation between columns.

**Tests that can be used:**

- Little's MCAR test.
- Chi-square test for missingness association with categorical variables.
- T-test or Mann-Whitney U test comparing numeric variables by missingness flag.
- Logistic regression missingness model.

**EDA interpretation:**

- Missing completely at random is rare in real business data.
- If missingness is related to a target or key feature, simple imputation can bias the model.

---

## Feature Quality And Modeling Readiness

### Feature Distribution Plot

**What it shows:**  
Whether a feature is usable, skewed, sparse, or dominated by a few values.

**Use it for:**  
Checking whether transformation, scaling, clipping, or binning is needed.

**Equivalent stats to use:**

- Missing percentage.
- Unique count.
- Cardinality.
- Skewness.
- Kurtosis.
- Zero ratio.
- Coefficient of variation.

**Tests that can be used:**

- Normality tests for transformation decisions.
- Variance threshold checks for near-constant features.
- Chi-square tests for categorical feature association.
- Mutual information for feature-target dependency.

### Target Vs Feature Plot

**What it shows:**  
How a feature relates to the target variable.

**Use it for:**  
Feature relevance, leakage checks, monotonicity, and non-linear relationships.

**Equivalent stats to use:**

- Correlation.
- Grouped target mean.
- Information value.
- Mutual information.
- Feature importance from baseline models.

**Tests that can be used:**

- Correlation significance tests.
- ANOVA/Kruskal-Wallis across binned feature groups.
- Chi-square test for categorical feature vs categorical target.
- Permutation importance for predictive signal.

---

## Time Series EDA Cheatsheet

### Time Series Line Plot

**What it shows:**  
Value over time.

**Use it for:**  
Trend, seasonality, regime shifts, spikes, missing periods, and level changes.

**Equivalent stats to use:**

- Rolling mean.
- Rolling median.
- Rolling standard deviation.
- Trend slope.
- Percent change.
- Growth rate.
- Moving average differences.

**Tests that can be used:**

- Linear regression trend test.
- Mann-Kendall trend test.
- Theil-Sen slope estimator.
- Chow test for structural breaks.
- CUSUM test for change detection.

**EDA interpretation:**

- A visible trend can break stationarity.
- Rolling statistics help reveal changing mean or variance.

### Rolling Mean And Rolling Standard Deviation Plot

**What it shows:**  
How the level and volatility change over time.

**Use it for:**  
Checking stationarity and stability.

**Equivalent stats to use:**

- Rolling mean.
- Rolling variance.
- Rolling standard deviation.
- Expanding mean.
- Expanding variance.

**Tests that can be used:**

- Augmented Dickey-Fuller test.
- KPSS test.
- Phillips-Perron test.
- Levene-style variance comparison across time windows.

**EDA interpretation:**

- If rolling mean changes strongly, the series may be non-stationary.
- If rolling variance changes, transformation or volatility modeling may be needed.

### Seasonal Plot

**What it shows:**  
Repeated pattern by season, such as weekday, month, quarter, or hour.

**Use it for:**  
Detecting recurring seasonal structure.

**Equivalent stats to use:**

- Seasonal averages.
- Seasonal medians.
- Seasonal standard deviations.
- Seasonal coefficient of variation.
- Peak-to-trough ratio.

**Tests that can be used:**

- ANOVA for seasonal group differences.
- Welch's ANOVA when seasonal group variances differ.
- Kruskal-Wallis for non-normal seasonal groups.
- Friedman test for repeated seasonal patterns across blocks.
- Seasonal Ljung-Box test at seasonal lags.

**EDA interpretation:**

- Strong weekday/month patterns should become calendar features.
- Business calendars and holidays can create seasonality that simple date features miss.

### ACF Plot

**What it shows:**  
Autocorrelation of a series with its past values.

**Use it for:**  
Detecting persistence, seasonality, and lag dependence.

**Equivalent stats to use:**

- Autocorrelation coefficients.
- Significant lag count.
- Seasonal lag strength.

**Tests that can be used:**

- Ljung-Box test.
- Box-Pierce test.
- Durbin-Watson test for lag-1 autocorrelation in residuals.

**EDA interpretation:**

- Slow ACF decay often suggests trend or non-stationarity.
- Spikes at seasonal lags suggest seasonal behavior.

### PACF Plot

**What it shows:**  
Direct correlation between a series and a lag after controlling for earlier lags.

**Use it for:**  
Identifying autoregressive lag structure.

**Equivalent stats to use:**

- Partial autocorrelation coefficients.
- Significant PACF lags.

**Tests that can be used:**

- Significance bands on PACF.
- Ljung-Box test on residuals after candidate models.

**EDA interpretation:**

- PACF spikes can guide AR terms.
- ACF and PACF together help choose baseline ARIMA-style models.

### Lag Plot

**What it shows:**  
Current value versus lagged value.

**Use it for:**  
Checking serial dependence and non-randomness.

**Equivalent stats to use:**

- Lag correlation.
- Autocorrelation coefficient at specific lags.
- Regression slope between current and lagged values.

**Tests that can be used:**

- Pearson/Spearman correlation test between value and lag.
- Ljung-Box test across multiple lags.
- Runs test for randomness.

### Decomposition Plot

**What it shows:**  
Observed series split into trend, seasonal, and residual components.

**Use it for:**  
Understanding how much of the signal is trend, seasonality, or noise.

**Equivalent stats to use:**

- Trend strength.
- Seasonal strength.
- Residual variance.
- Seasonal amplitude.

**Tests that can be used:**

- ADF/KPSS on residuals.
- Ljung-Box on residuals.
- Seasonal strength heuristics from STL decomposition.

**EDA interpretation:**

- If residuals still show autocorrelation, important time structure remains unexplained.
- If seasonal component is strong, include seasonal features or seasonal models.

### Calendar/Event Impact Plot

**What it shows:**  
Difference between normal periods and event/holiday/promotion periods.

**Use it for:**  
Evaluating event effects and calendar-driven demand changes.

**Equivalent stats to use:**

- Event-day mean.
- Non-event-day mean.
- Event lift percentage.
- Event-day median.
- Confidence intervals.

**Tests that can be used:**

- Welch's t-test for event vs non-event periods.
- Mann-Whitney U test for non-normal event effects.
- Paired t-test if matching event periods to comparable non-event periods.
- Wilcoxon signed-rank test for paired non-normal comparisons.
- Difference-in-differences when a control group exists.

### Forecast Residual Plot

**What it shows:**  
Forecast errors over time.

**Use it for:**  
Checking model bias, remaining autocorrelation, heteroscedasticity, and error spikes.

**Equivalent stats to use:**

- Mean error.
- MAE.
- RMSE.
- MAPE or sMAPE.
- Bias.
- Residual standard deviation.

**Tests that can be used:**

- Ljung-Box test on residuals.
- Shapiro-Wilk or Jarque-Bera on residual normality.
- Breusch-Pagan test for heteroscedasticity.
- Durbin-Watson test for residual autocorrelation.

**EDA interpretation:**

- Residuals should ideally look like noise.
- Autocorrelated residuals mean the model missed time-dependent structure.

---

## Test Selection Notes

### Normal Vs Non-Normal Data

- If data is approximately normal, use parametric tests such as t-test and ANOVA.
- If data is heavily skewed, ordinal, sparse, or has extreme outliers, use non-parametric tests such as Mann-Whitney U, Wilcoxon, and Kruskal-Wallis.
- For large samples, tiny differences can become statistically significant, so also report effect size.

### Equal Vs Unequal Variance

- Use Levene's test to check variance equality across groups.
- Use Bartlett's test only when normality is reasonable.
- If variances differ, prefer Welch's t-test or Welch's ANOVA.

### Independent Vs Paired Data

- Use independent tests when observations are from separate groups.
- Use paired tests when observations are before/after, matched, repeated, or same entity over two periods.

### Two Groups Vs Many Groups

- Two numeric groups: t-test, Welch's t-test, Mann-Whitney U, paired t-test, or Wilcoxon.
- More than two numeric groups: ANOVA, Welch's ANOVA, Kruskal-Wallis, repeated-measures ANOVA, or Friedman.
- After a many-group test, use post-hoc tests to identify which groups differ.

### Time Series Data

- Do not treat time series observations as independent by default.
- Check trend, stationarity, seasonality, and autocorrelation before using standard tests.
- Use residual diagnostics after forecasting models.

### Multiple Testing

- If running many pairwise tests, adjust p-values.
- Use Benjamini-Hochberg FDR when controlling false discoveries.
- Use Bonferroni when you need stricter control, though it can be conservative.

---

## Compact Summary

### Numeric Distribution

- **Plots:** histogram, KDE, Q-Q plot, box plot.
- **Stats:** mean, median, std, IQR, skewness, kurtosis, CV.
- **Tests:** Shapiro-Wilk, Jarque-Bera, D'Agostino, Anderson-Darling, KS test.

### Outliers

- **Plots:** box plot, scatter plot, histogram.
- **Stats:** IQR bounds, z-score, MAD, percentiles.
- **Tests:** Grubbs, Dixon Q, generalized ESD.

### Numeric Relationships

- **Plots:** scatter plot, regression plot, pair plot, heatmap.
- **Stats:** Pearson, Spearman, Kendall, covariance, R-squared.
- **Tests:** correlation significance tests, regression slope test, FDR correction.

### Group Comparisons

- **Plots:** grouped box plot, violin plot, bar chart with CI.
- **Stats:** group mean, median, std, IQR, confidence interval, effect size.
- **Tests:** t-test, Welch's t-test, ANOVA, Welch's ANOVA, Mann-Whitney U, Kruskal-Wallis.

### Categorical Variables

- **Plots:** count plot, bar chart, stacked bar chart, mosaic plot.
- **Stats:** counts, proportions, mode, entropy.
- **Tests:** chi-square goodness-of-fit, chi-square independence, Fisher's exact, G-test.

### Missing Values

- **Plots:** missingness heatmap, missingness bar chart, nullity matrix.
- **Stats:** missing count, missing percentage, row-wise missingness, nullity correlation.
- **Tests:** Little's MCAR, chi-square association, missingness logistic model.

### Time Series

- **Plots:** line plot, rolling mean/std, seasonal plot, ACF, PACF, lag plot, decomposition plot.
- **Stats:** rolling stats, trend slope, autocorrelation, seasonal averages, residual metrics.
- **Tests:** ADF, KPSS, Phillips-Perron, Ljung-Box, Mann-Kendall, Durbin-Watson, Chow test.

### Forecast Residuals

- **Plots:** residual line plot, residual histogram, residual ACF, actual vs forecast.
- **Stats:** ME, MAE, RMSE, MAPE, sMAPE, bias.
- **Tests:** Ljung-Box, Durbin-Watson, Shapiro-Wilk, Jarque-Bera, Breusch-Pagan.
