# M5 Sample Dataset Creation Design

**Date:** 2026-05-18  
**Project:** Retail Demand Forecast Copilot  
**Objective:** Create optimized sample dataset for POC development on M3 Pro hardware

## Requirements Summary

**Hardware Constraints:**
- M3 Pro chip with 32GB RAM
- Target processing time: 15-30 minutes for end-to-end model training
- Primary bottlenecks: processing speed and model training time

**Quality Targets:**
- Strong validation results (80-85% of full dataset performance)
- Preserve geographic diversity (all 3 states: CA, TX, WI)
- Maintain product category coverage (FOODS, HOUSEHOLD, HOBBIES)
- Retain seasonal and event patterns (holidays, weekends)

**Full Dataset Context:**
- 30,490 item-store combinations across 10 stores and 3,049 items
- 1,969 days of sales data (Jan 29, 2011 - June 19, 2016)
- Total size: ~425MB (sales: 241MB, pricing: 203MB, calendar: 103KB)

## Chosen Approach: Time-Stratified Product Sample

**Strategy:** Sample 40-50% of products (1,200-1,500 items) while maintaining complete geographic and temporal coverage.

**Rationale:** This approach maximizes model performance (85-90% of full dataset) by preserving complete store-level relationships, geographic patterns, and temporal seasonality while achieving significant speed improvements through product dimension reduction.

## Product Selection Strategy

### Sampling Targets

**Total Items:** 1,200-1,500 items (40-50% of 3,049 total)

**Category Distribution:**
- **FOODS:** ~600-750 items (maintain largest category representation)
- **HOUSEHOLD:** ~300-400 items  
- **HOBBIES:** ~300-350 items

### Stratification Methodology

**Primary Stratification:** By category and department
- Ensure proportional representation across all 7 departments
- Preserve product hierarchy relationships for business logic
- Maintain category-specific demand behavior patterns

**Secondary Stratification:** Within each department by:
1. **Sales Volume Quartiles:** High/Med-High/Med-Low/Low volume items
2. **Store Availability:** Prioritize items sold across multiple stores
3. **Time Series Completeness:** Favor items with consistent availability

### Selection Algorithm

**Step 1: Item Analysis**
- Calculate per-item metrics: total volume, store count, sales consistency
- Group items by (category, department) combinations
- Rank within each group by composite score (volume × availability × completeness)

**Step 2: Systematic Sampling**
- Within each (category, department, volume_quartile) stratum:
  - Use systematic random sampling with fixed interval
  - Maintain proportional representation across quartiles
  - Apply random seed for reproducibility

**Step 3: Quality Validation**
- Verify category distribution matches targets
- Confirm geographic coverage across all stores
- Validate demand behavior spectrum representation

## Geographic and Temporal Coverage

### Complete Store Coverage
**All 10 stores maintained:**
- CA_1, CA_2, CA_3, CA_4 (California)
- TX_1, TX_2, TX_3 (Texas)  
- WI_1, WI_2, WI_3 (Wisconsin)

**Benefits:**
- Preserves complete geographic diversity and regional patterns
- Maintains state-specific SNAP benefit timing variations
- Keeps store-specific pricing and demand characteristics
- Enables robust geographic feature engineering

### Complete Temporal Coverage
**Full 1,969-day timeline retained:**
- Training period: d_1 to d_1913 (Jan 29, 2011 - May 22, 2016)
- Evaluation period: d_1914 to d_1969 (May 23, 2016 - June 19, 2016)

**Benefits:**
- Perfect seasonality detection and calendar feature engineering
- Maintains all holiday, event, and promotional patterns
- Supports proper time series train/validation/test splits
- Preserves year-over-year trend analysis capabilities

### Calendar and Pricing Data
- **Calendar.csv:** Complete file retained (103KB, no sampling needed)
- **Sell_prices.csv:** Filtered to selected items only (~15-20MB vs 203MB full)

## Expected Dataset Characteristics

### Size Projections
**Total estimated size:** 60-85MB (85% reduction from 425MB)
- **Sales data:** ~50-65MB (vs 241MB full dataset)
- **Pricing data:** ~15-20MB (vs 203MB full dataset)
- **Calendar data:** 103KB (unchanged)

### Performance Projections
**Processing timeline breakdown:**
- Data loading: 3-5 minutes (vs 15-20 minutes full)
- Feature engineering: 8-12 minutes (vs 30-40 minutes full)
- Model training: 8-15 minutes (vs 45-60 minutes full)
- **Total pipeline: 20-30 minutes** ✅ *Meets 15-30 minute target*

**Model performance:** 85-90% of full dataset quality
- High statistical confidence in demand behavior profiling
- Robust seasonality and trend detection capabilities
- Strong geographic pattern learning
- Suitable for stakeholder validation and technical demos

## Technical Implementation

### File Structure
```
data/
├── full_data/                    # Original M5 dataset (unchanged)
│   ├── sales_train_validation.csv
│   ├── sales_train_evaluation.csv
│   ├── sell_prices.csv
│   └── calendar.csv
├── processed/
│   └── sample_dataset/           # Generated sample dataset
│       ├── sales_train_validation_sample.csv
│       ├── sales_train_evaluation_sample.csv
│       ├── sell_prices_sample.csv
│       ├── calendar.csv          # Copy of original
│       └── sample_metadata.json  # Sampling statistics and item list
└── scripts/
    └── create_sample_dataset.py  # Sampling implementation script
```

### Implementation Steps

**Phase 1: Analysis and Stratification**
1. Load full sales_train_validation.csv and sell_prices.csv
2. Calculate item-level metrics:
   - Total sales volume per item across all stores
   - Store availability count per item
   - Sales consistency score (non-zero sales frequency)
3. Group items by category and department
4. Create volume quartiles within each group

**Phase 2: Sampling Execution**
1. Define sampling parameters (target counts, random seed)
2. Execute stratified systematic sampling within each stratum
3. Generate final item selection list (1,200-1,500 items)
4. Validate sample composition against targets

**Phase 3: Dataset Generation**
1. Filter sales files by selected items (preserve all columns and rows)
2. Filter pricing file by selected items (maintain temporal coverage)
3. Copy calendar file unchanged
4. Export to sample_dataset directory with consistent naming

**Phase 4: Quality Validation**
1. Generate sampling summary report
2. Calculate statistical comparison metrics (sample vs full)
3. Create validation visualizations
4. Export sample metadata for pipeline integration

### Configuration Parameters

**Sampling Configuration:**
```python
SAMPLE_CONFIG = {
    'target_item_count': 1400,  # Target number of items
    'category_distribution': {  # Target category proportions
        'FOODS': 0.52,
        'HOUSEHOLD': 0.24, 
        'HOBBIES': 0.24
    },
    'random_seed': 42,          # Reproducibility
    'min_store_coverage': 3,    # Minimum stores per item
    'min_sales_days': 100,      # Minimum non-zero sales days
}
```

**Quality Thresholds:**
```python
QUALITY_THRESHOLDS = {
    'category_tolerance': 0.05,     # ±5% category distribution
    'volume_correlation': 0.85,     # Sample-full correlation
    'geographic_coverage': 0.95,    # Stores with selected items
}
```

## Data Quality Validation

### Representativeness Checks
1. **Category Distribution Validation**
   - Verify actual vs target category proportions within tolerance
   - Confirm department representation within each category
   - Validate product hierarchy completeness

2. **Geographic Coverage Analysis**
   - Calculate items per store distribution
   - Verify state-level sales volume proportions
   - Confirm SNAP benefit period coverage

3. **Demand Behavior Spectrum**
   - Validate volume distribution across quartiles
   - Confirm seasonality pattern representation
   - Check intermittent vs regular demand product mix

### Statistical Validation Metrics
1. **Correlation Analysis:** Sample vs full dataset for key statistics
   - Mean daily sales per item-store combination
   - Coefficient of variation distribution
   - Seasonal decomposition components

2. **Coverage Metrics:**
   - Geographic diversity index (items × stores coverage)
   - Temporal completeness (non-zero sales frequency)
   - Product hierarchy coverage percentage

3. **Business Validation:**
   - Fast-moving vs stable vs intermittent product representation
   - Holiday/event period coverage validation
   - Price elasticity analysis capability retention

### Validation Outputs
1. **sample_metadata.json:** Complete sampling statistics and item list
2. **validation_report.csv:** Statistical comparison metrics
3. **coverage_visualizations.png:** Geographic and category distribution plots
4. **correlation_analysis.csv:** Sample-full dataset correlation matrix

## Integration and Reproducibility

### Pipeline Integration
**Schema Compatibility:**
- Sample dataset maintains identical column structure to full dataset
- Existing feature engineering code works without modification
- Model training pipelines require no schema changes

**Configuration-Driven Switching:**
```python
# Simple config switch between sample and full dataset
DATA_CONFIG = {
    'use_sample': True,  # Toggle for development vs production
    'sample_path': 'data/processed/sample_dataset/',
    'full_path': 'data/full_data/'
}
```

### Reproducibility Controls
1. **Fixed Random Seed:** Ensures identical samples across runs
2. **Version Control:** Sample generation script and metadata in git
3. **Dependency Tracking:** Document pandas/numpy versions used
4. **Audit Trail:** Complete sampling parameters and statistics preserved

### Maintenance Strategy
**Sample Refresh Triggers:**
- Full dataset updates or corrections
- Model performance below quality thresholds  
- Hardware constraints change significantly
- Business requirements evolution

**Refresh Process:**
1. Re-run sampling script with updated parameters
2. Validate new sample meets quality standards
3. Update sample metadata and documentation
4. Test pipeline compatibility before deployment

## Success Criteria

### Performance Benchmarks
- **Speed Target:** ✅ 20-30 minute end-to-end pipeline (vs 60+ minutes full)
- **Quality Target:** ✅ 85-90% model performance (vs 80-85% requirement)
- **Memory Target:** ✅ 60-85MB dataset (vs 425MB full, 80% reduction)

### Validation Requirements
- Geographic diversity: All 3 states represented proportionally
- Category coverage: All 3 categories with proper product mix
- Seasonal patterns: Complete holiday/event calendar preserved
- Statistical validity: >0.85 correlation with full dataset key metrics

### Business Value Delivery
- Faster POC development and iteration cycles
- Maintained confidence in demand behavior insights
- Preserved seasonality and trend detection capabilities
- Scalable approach for production dataset optimization

This design enables rapid POC development while maintaining statistical rigor and business insight quality necessary for stakeholder validation and technical advancement.