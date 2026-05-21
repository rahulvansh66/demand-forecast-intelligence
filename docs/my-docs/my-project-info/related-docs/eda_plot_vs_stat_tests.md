### Plot To Statistical Test Mapping

- **Sales distribution histograms, log-sales histogram, zero-sales ratio histogram**
  - Used to inspect skewness, intermittency, sparsity.
  - Equivalent tests used: **Shapiro-Wilk**, **Jarque-Bera**, **D’Agostino normality test**.
  - Also computes skewness, kurtosis, IQR, CV.

- **Box plots**
  - Used for outlier/spread detection, especially daily total sales.
  - Equivalent stats used: descriptive stats, IQR, skewness, kurtosis.
  - No formal outlier test like Grubbs/IQR-rule test is implemented.

- **Q-Q plots**
  - Used to visually check normality.
  - Equivalent tests used: Shapiro-Wilk, Jarque-Bera, D’Agostino.

- **Time series line plots**
  - Used to inspect demand trend and temporal behavior.
  - Equivalent tests used: **linear regression trend test**, **ADF**, **KPSS**, **Ljung-Box**.

- **Correlation heatmaps**
  - Used to inspect relationships between numeric variables or sales periods.
  - Equivalent method used: `DataFrame.corr()` with **Pearson** by default; supports Spearman/Kendall through parameter.
  - No correlation p-value/significance test is implemented.
  - Tests that can be implemented: **Pearson correlation significance test** (`scipy.stats.pearsonr`), **Spearman rank correlation test** (`scipy.stats.spearmanr`), **Kendall Tau test** (`scipy.stats.kendalltau`), and **Benjamini-Hochberg/FDR correction** for many pairwise correlations.

- **Category/store performance bar charts**
  - Used to compare total sales, average sales, volatility across groups.
  - Equivalent statistics used: group totals, means, standard deviations.
  - No ANOVA/Kruskal-Wallis test is currently implemented.

- **Temporal pattern bar charts**
  - Weekday average sales, monthly average sales, event-vs-normal sales.
  - Equivalent stats used: grouped averages and seasonal CV indicators.
  - No formal seasonality/group comparison test is implemented.
  - Tests that can be implemented: **one-way ANOVA** for weekday/month group differences, **Welch's ANOVA** when group variances differ, **Kruskal-Wallis test** for non-normal grouped sales, **Mann-Whitney U test** or **Welch's t-test** for event vs no-event days, and **seasonal Ljung-Box test** at weekly/monthly lags for seasonal autocorrelation.

- **Demand segmentation scatter plot**
  - Average sales vs CV, colored by segment.
  - Equivalent stats used: average sales, CV, zero-sales ratio thresholding.
  - No clustering/statistical significance test is implemented.
  - Tests that can be implemented: **ANOVA** or **Kruskal-Wallis test** to compare avg_sales/CV/zero_ratio across segments, **MANOVA** to test multivariate separation, **permutation test** for cluster separation, and clustering validation metrics such as **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Index**.

- **Demand segment pie chart**
  - Shows proportions of demand segments.
  - Equivalent stats used: value counts/proportions.
  - No chi-square proportion test is implemented.

---

For each plot, you can explain it using these 3 points:

What is the plot?
This means what visual chart is being used and what pattern it helps you see.
Example: histogram, box plot, Q-Q plot, time series line plot, heatmap, scatter plot, bar chart, pie chart.

What are equivalent stats used?
This means what statistical summary or test is already being used in your current EDA code to support that plot.
Example: histogram is supported by mean, median, skewness, kurtosis, Shapiro-Wilk, Jarque-Bera, D’Agostino.

What are stats/tests that can be used?
This means additional statistical tests that are not necessarily implemented yet, but would be appropriate for that plot or question.
Example: for category bar charts, your current code uses group mean/std, but you could also use ANOVA or Kruskal-Wallis to test whether category differences are statistically significant.