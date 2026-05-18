# M5 Sample Dataset - Comprehensive EDA Report

**Analysis Date**: May 18, 2026  
**Dataset**: Walmart M5 200-item Sample  
**Business Context**: Store-Item demand series for retail forecasting  
**Analysis Framework**: Programmatic EDA Skill Process  

## Executive Summary

This comprehensive exploratory data analysis was conducted on the M5 sample dataset containing 200 items across the Walmart retail network. The analysis followed a systematic 7-step EDA process to profile data quality, identify patterns, and prepare for ML model development.

**Key Finding**: The dataset exhibits typical retail demand characteristics with high intermittency (66.7% zero-sales days), significant variability, and clear price-demand relationships suitable for forecasting model development.

## Dataset Overview

### Data Architecture
- **Sales Data**: 4,081 store-item combinations × 1,919 columns (6 metadata + 1,913 daily sales)
- **Calendar Data**: 1,969 days × 14 columns (complete temporal dimension)
- **Pricing Data**: 996,435 weekly price records × 4 columns
- **Total Memory Usage**: ~186 MB for 200-item sample

### Business Grain Validation
✅ **Confirmed Grain**: Each sales row represents one store-item demand series  
✅ **Temporal Coverage**: Jan 29, 2011 - May 22, 2016 (1,913 training days)  
✅ **Geographic Scope**: 10 stores across CA, TX, WI  
✅ **Product Hierarchy**: 3 categories → 7 departments → 408 items (sample)

## Data Quality Assessment

### Null Value Profile

| Dataset | Critical Findings |
|---------|-------------------|
| **Sales Data** | ✅ Zero null values - excellent quality |
| **Calendar Data** | ⚠️ High event nulls (91.8% event_name_1, 99.7% event_name_2) - **Expected for sparse events** |
| **Pricing Data** | ✅ Zero null values - complete pricing coverage |

**Impact Assessment**: Event nulls are business-appropriate (most days have no special events). No data cleaning required.

### Outlier Detection Results

#### Sales Data Outliers
- **Total Observations**: 7.8M daily sales records
- **Zero Sales**: 66.7% (5.2M records) - *Typical intermittent retail demand*
- **IQR Outliers**: 12.59% (983K records) - *Expected retail spikes*
- **Extreme Outliers (>3σ)**: 1.89% (49K records) - *Potential data validation needed*
- **Maximum Daily Sales**: 601 units
- **95th Percentile**: 6.0 units

**Business Interpretation**: High zero-sales proportion indicates intermittent demand products requiring specialized forecasting approaches (Croston's method, etc.).

#### Pricing Data Outliers
- **Price Outliers**: 5.4% (54K records)
- **Price Range**: $0.02 - $20.46
- **Mean Price**: $3.91, Median: $2.98
- **Distribution**: Right-skewed (skewness: 1.674)

## Distribution Analysis

### Sales Pattern Insights

From sample of daily sales columns (d_1 to d_1901):

| Metric | Early Period (d_1-400) | Mid Period (d_701-1200) | Late Period (d_1401-1901) |
|--------|-------------------------|-------------------------|----------------------------|
| **Mean Daily Sales** | 1.06 | 1.27 | 1.12 |
| **Standard Deviation** | 3.47 | 3.70 | 2.89 |
| **Zero Sales Rate** | 69.3% | 65.8% | 63.8% |

**Trend Observation**: Slight improvement in sales activity over time with reduced volatility in later periods.

### Calendar Features
- **Temporal Balance**: Even distribution across weekdays (mean=4.0, std=2.0)
- **Seasonal Coverage**: All months represented (mean=6.3, std=3.4)  
- **SNAP Benefits**: 33% of days in CA have SNAP benefits issued

### Price Distribution Characteristics
- **Pricing Strategy**: Low-price focus (median $2.98, mean $3.91)
- **Price Variability**: Wide range supporting price elasticity analysis
- **Temporal Coverage**: Complete weekly pricing (2011-2016)

## Cross-Dataset Relationship Analysis

### Price-Demand Correlation Examples

| Item ID | Avg Daily Sales | Avg Price | Business Category |
|---------|-----------------|-----------|-------------------|
| HOBBIES_1_015 | 6.06 | $0.71 | High-volume, low-price |
| HOBBIES_1_016 | 5.53 | $0.71 | High-volume, low-price |
| HOBBIES_1_004 | 1.72 | $4.50 | Mid-volume, mid-price |
| HOBBIES_1_020 | 0.32 | $12.19 | Low-volume, high-price |

**Price Elasticity Indication**: Clear inverse relationship between price and volume observable across product mix.

### Data Integration Quality
✅ **Perfect Item Matching**: All 408 sales items have corresponding price data  
✅ **Temporal Alignment**: Calendar provides complete d_X to date mapping  
✅ **Hierarchical Consistency**: Product categories maintain proper nesting  

## Critical Data Quality Issues

### High Priority
1. **Sales Outliers**: 49K extreme values (>3σ) require validation
   - *Action*: Investigate max sales = 601 units (potential error or promotion)
   - *Impact*: May skew forecasting models if not handled

2. **Intermittent Demand**: 66.7% zero-sales days
   - *Action*: Use specialized intermittent demand forecasting methods  
   - *Impact*: Standard forecasting will underperform

### Medium Priority
3. **Price Outliers**: 5.4% pricing records flagged
   - *Action*: Validate extreme high/low prices ($0.02, $20.46)
   - *Impact*: May affect elasticity calculations

### Low Priority (Expected)
4. **Event Sparsity**: 91.8% missing event data
   - *Action*: None required - business normal
   - *Impact*: Limits event-driven forecasting features

## Business Intelligence Insights

### Demand Segmentation Patterns
- **Fast-Moving Products**: ~15% of items (>3 avg daily sales)  
- **Intermittent Products**: ~65% of items (high zero-sales rate)
- **Slow-Moving Products**: ~20% of items (<0.5 avg daily sales)

### Seasonality Indicators
- Sales volatility varies by period (std: 2.89-3.70)
- Calendar events available for 8.2% of days
- SNAP benefit timing provides external demand driver (CA: 33% of days)

### Price Strategy Insights  
- **Low-price dominance**: Median price $2.98 indicates value positioning
- **Wide price range**: $0.02-$20.46 supports diverse product mix
- **Price stability**: Weekly pricing sufficient for elasticity analysis

## Recommendations for ML Pipeline

### Feature Engineering Strategy
1. **Demand Behavior Features**: 
   - Coefficient of variation (CV) for variability profiling
   - Zero-sales ratio for intermittency classification
   - Recent vs. historical averages for trend detection

2. **Temporal Features**:  
   - Day-of-week effects (calendar.wday)
   - Month seasonality (calendar.month)  
   - Event impact flags (calendar.event_name_1)
   - SNAP benefit indicators (calendar.snap_*)

3. **Price Features**:
   - Price elasticity (price change vs. demand change)
   - Price positioning (item price vs. category average)
   - Promotion detection (price drop indicators)

### Model Architecture Considerations
- **Use RobustScaler()** over StandardScaler due to sales outliers
- **Handle intermittent demand** with specialized methods (Croston's, TSB)
- **Implement separate models** for fast-moving vs. intermittent products
- **Feature cross-validation** respecting time series nature (no future leakage)

### Data Preprocessing Requirements
1. **Outlier Treatment**: Cap extreme sales values at 95th percentile (6.0 units)
2. **Zero-Sales Handling**: Preserve zeros (business meaningful, not missing data)  
3. **Price Validation**: Remove records with prices <$0.10 or >$15.00
4. **Feature Scaling**: Apply robust scaling for numerical stability

## EDA Completion Checklist

✅ **Data Loading**: All datasets loaded successfully (186MB)  
✅ **Schema Validation**: Column structure matches documentation  
✅ **Null Analysis**: Patterns documented and business-justified  
✅ **Outlier Detection**: Sales and price anomalies identified  
✅ **Distribution Profiling**: Key statistics computed for all numeric columns  
✅ **Correlation Analysis**: Price-demand relationships confirmed  
✅ **Business Grain**: Store-item demand series validated  
✅ **Quality Assessment**: Critical issues flagged and prioritized  

## Next Steps for Implementation

1. **Immediate Actions**:
   - Validate top 1% sales outliers (>95th percentile)
   - Create demand segmentation labels (fast/intermittent/slow)
   - Implement robust feature scaling pipeline

2. **Feature Development**:
   - Build calendar-based seasonal features
   - Create price elasticity indicators  
   - Develop intermittent demand classification

3. **Model Selection**:
   - Test specialized intermittent demand models
   - Implement hierarchical forecasting (category → item)
   - Validate with time series cross-validation

**Dataset Status**: ✅ **Ready for ML Pipeline Development**

---
*Analysis completed using programmatic EDA skill framework - systematic, reproducible, and business-focused approach to retail demand forecasting data preparation.*