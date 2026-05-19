# Sample Dataset Validation Report

**Generated:** 2026-05-19 20:54:30
**Random Seed:** 42
**Sampling Method:** `random_within_strata`

---

## Overall Quality Score: 1.02 / 1.00 — EXCELLENT

| Metric | Score |
|---|---|
| Target Achievement | 1.080 |
| Strata Coverage | 1.000 |
| Dept Coverage | 1.000 |
| Geographic Completeness | 1.000 |

---

## Sample Quality Summary

| Field | Value |
|---|---|
| Items Selected | 1,512 |
| Target Item Count | 1,400 |
| Target Achievement | 108.0% |
| Population Items | 30,490 |
| Population Reduction | 50.4% |
| Row Reduction | 60.2% |
| Strata Covered | 307 / 307 |

---

## Behavioral Diversity

### Volume Distribution

| Bucket | Items | % of Sample |
|---|---|---|
| High | 308 | 20.4% |
| Low | 372 | 24.6% |
| Medium | 700 | 46.3% |
| Very_High | 132 | 8.7% |

### Intermittency Patterns

| Pattern | Items | % of Sample |
|---|---|---|
| Intermittent | 667 | 44.1% |
| Regular | 228 | 15.1% |
| Sparse | 617 | 40.8% |

### Lifecycle Stages

| Stage | Items | % of Sample |
|---|---|---|
| Declining | 186 | 12.3% |
| Discontinued | 68 | 4.5% |
| Early | 788 | 52.1% |
| Mature | 431 | 28.5% |
| Minimal | 39 | 2.6% |

---

## Geographic Coverage

| Dimension | Sample | Population | Status |
|---|---|---|---|
| States | 3 | 3 | COMPLETE |
| Stores | 10 | 10 | COMPLETE |
| Departments | 7 | 7 | COMPLETE |
| Categories | 3 | 3 | COMPLETE |

---

## Department Coverage

| Department | Items | % of Sample |
|---|---|---|
| FOODS_1 | 123 | 8.1% |
| FOODS_2 | 201 | 13.3% |
| FOODS_3 | 380 | 25.1% |
| HOBBIES_1 | 208 | 13.8% |
| HOBBIES_2 | 93 | 6.2% |
| HOUSEHOLD_1 | 256 | 16.9% |
| HOUSEHOLD_2 | 251 | 16.6% |

---

## Anti-Bias Verification

| Check | Status |
|---|---|
| Method | Random selection within behavioral strata |
| No Ranking | True - no quality scores or composite metrics used |
| Equal Probability | True - all items in stratum had equal selection chance |
| Stratum Based | True - sampling respects behavioral diversity |

---

## Configuration Snapshot

| Parameter | Value |
|---|---|
| Min per Department | 30 |
| Min per Stratum | 2 |
| Volume Percentiles | [0, 25, 75, 95, 100] |
| Intermittency Thresholds | [0.2, 0.6] |

---

## Generated Files

| Type | Path | Size |
|---|---|---|
| Sales | `data/full_data/samples/sample_1400items/sales_train_validation_sample.csv` | 45.6 MB |
| Calendar | `data/full_data/samples/sample_1400items/calendar.csv` | 0.1 MB |
| Prices | `data/full_data/samples/sample_1400items/sell_prices_sample.csv` | 77.4 MB |
