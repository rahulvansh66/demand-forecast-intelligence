# M5 Sample Dataset Generation

This project includes a sophisticated sampling system for creating representative sample datasets from the M5 Walmart dataset. The sampling approach prevents bias toward high-volume, stable items by using behavioral stratification with random sampling, ensuring challenging intermittent/sparse demand patterns are properly represented for robust POC validation.

## Quick Start

### Basic Usage

```bash
# Generate sample dataset with default settings (1400 items, ~50% reduction)
python scripts/create_sample_dataset.py

# Custom sample size and validation report
python scripts/create_sample_dataset.py --target-items 1000 --validate

# Specify data directories
python scripts/create_sample_dataset.py --data-dir /path/to/m5/data --output-dir /path/to/output
```

### Configuration Options

- `--target-items`: Number of items in sample dataset (default: 1400)
- `--data-dir`: Directory containing M5 CSV files (default: data/full_data)
- `--output-dir`: Directory for sample dataset files (default: data/processed/sample_dataset)
- `--random-seed`: Random seed for reproducible sampling (default: 42)
- `--validate`: Generate detailed validation report with quality metrics
- `--verbose`: Enable detailed logging for debugging

## Sample Strategy

### Anti-Bias Methodology

Traditional sampling methods rank items by "quality" (volume x availability) and select the "best" items, creating severe bias toward high-volume, stable products. This makes POC validation overly optimistic since real-world forecasting challenges come from intermittent, sparse, and irregular demand patterns.

Our approach uses **random sampling within behavioral strata** to ensure:

1. **No quality-based ranking** - Prevents high-volume bias through equal probability selection
2. **Proportional representation** - Maintains natural demand pattern diversity
3. **Adequate challenging patterns** - Ensures sparse/intermittent items get fair representation
4. **Statistical rigor** - Follows stratified sampling principles for valid inference

### Behavioral Stratification Dimensions

**Volume Stratification (within departments)**

- **Low volume**: 0-25th percentile - slow-moving products with inventory challenges
- **Medium volume**: 25-75th percentile - steady sellers with predictable patterns
- **High volume**: 75-95th percentile - fast-moving products with economies of scale
- **Very high volume**: 95-100th percentile - top sellers driving most revenue

**Intermittency Classification**

- **Highly intermittent**: <20% days with sales - sparse demand, forecasting challenges
- **Moderately intermittent**: 20-60% days with sales - irregular but predictable patterns
- **Frequently selling**: >60% days with sales - regular demand with seasonal variations

**Lifecycle Stages**

- **Early phase**: New products with limited history and uncertain demand
- **Mature phase**: Established products with reliable patterns and stable demand
- **Late phase**: Products in decline with changing customer preferences
- **Discontinued**: Products ending their lifecycle with inventory liquidation needs

## Output Structure

### Generated Files

The sampling process creates the following files in the output directory:

**Sample Dataset Files:**

- `sample_sales_train_validation.csv` - Sales data for selected items maintaining all time periods and stores
- `sample_calendar.csv` - Complete calendar data (unchanged, needed for temporal analysis)
- `sample_sell_prices.csv` - Pricing data corresponding to sampled items

**Metadata and Validation:**

- `sample_metadata.json` - Sampling configuration, statistics, and audit trail
- `sample_validation_report.json` - Quality metrics, coverage statistics, and bias verification

### Sample Metadata Structure

```json
{
  "target_item_count": 1400,
  "actual_item_count": 1398,
  "random_seed": 42,
  "sampling_method": "Behavioral Stratification with Random Selection",
  "bias_prevention": {
    "method": "Random selection within behavioral strata",
    "no_ranking": "True - no quality scores used",
    "equal_probability": "True - all items in stratum had equal selection chance",
    "stratum_based": "True - sampling respects behavioral diversity"
  },
  "sample_stats": {
    "total_strata": 48,
    "strata_coverage": 45,
    "dept_distribution": {"FOODS_1": 198, "FOODS_2": 187, "...": "..."}
  },
  "population_stats": {
    "total_items": 3049,
    "total_strata": 48,
    "dept_distribution": {"FOODS_1": 432, "FOODS_2": 398, "...": "..."}
  }
}
```

### Validation Report Metrics

```json
{
  "sample_quality": {
    "target_achievement": 0.998,
    "strata_coverage_ratio": 0.938,
    "geographic_completeness": "COMPLETE"
  },
  "behavioral_diversity": {
    "volume_distribution": {"low": 0.28, "medium": 0.45, "high": 0.27},
    "intermittency_patterns": {"sparse": 0.18, "intermittent": 0.32, "regular": 0.5},
    "lifecycle_stages": {"early": 0.15, "mature": 0.7, "late": 0.15}
  },
  "geographic_coverage": {
    "states_represented": "3/3 (CA, TX, WI)",
    "stores_represented": "10/10 (100%)",
    "coverage_complete": true
  }
}
```

## Business Context

### Why This Approach Matters

- **Realistic POC Validation**: Including challenging patterns prevents overoptimistic model performance estimates
- **Production Readiness**: Models trained on representative samples perform better in real-world deployment
- **Business Insights**: Adequate representation across product categories enables meaningful business recommendations
- **Resource Efficiency**: 50-60% dataset reduction enables faster iteration on resource-constrained hardware

### Performance Targets Achieved

- **Computational Efficiency**: ~50% dataset size reduction for faster training and experimentation
- **Pattern Preservation**: All major demand behavior types maintained in statistically valid proportions
- **Geographic Coverage**: Complete store and state representation for regional analysis
- **Temporal Completeness**: All time periods preserved for seasonality and trend analysis

### Use Cases

1. **POC Development** - Fast iteration with representative data on M3 Pro hardware constraints
2. **Model Comparison** - Consistent sample datasets for fair algorithm evaluation
3. **Feature Engineering** - Diverse patterns for robust feature validation across demand types
4. **Business Analysis** - Adequate representation for category and regional insights

## Integration Testing

The sampling system includes comprehensive integration tests that validate:

- **End-to-end pipeline** - Complete workflow from configuration to file generation
- **Schema preservation** - Data structure compatibility with downstream analysis
- **Behavioral diversity** - Multiple volume buckets and intermittency patterns represented
- **Anti-bias verification** - No future leakage, random sampling within strata documented
- **Geographic coverage** - All stores maintained, pricing coverage preserved
- **Reproducibility** - Identical configurations produce identical results

Run integration tests with:

```bash
python -m pytest tests/integration/test_sample_generation_integration.py -v
```

## Technical Details

### Statistical Foundation

- **Stratified Sampling**: Segments population into behavioral strata for representative selection
- **Proportional Allocation**: Larger strata receive more items while maintaining minimum thresholds
- **Random Selection**: numpy.random.choice with fixed seed ensures reproducible, unbiased sampling
- **Training Period Only**: All stratification uses d_1 to d_1913 data to prevent future leakage

### Implementation Architecture

- `SamplingConfig`: Configuration dataclass with validation and business constraints
- `BehavioralStratifier`: Multi-dimensional classification across volume/intermittency/lifecycle
- `SampleGenerator`: Main orchestration engine coordinating all sampling components
- `create_sample_dataset.py`: CLI script providing user-friendly interface to sampling system

### Quality Assurance

- **Comprehensive Testing**: Unit tests for components, integration tests for end-to-end workflow
- **Schema Validation**: Ensures M5 data structure compatibility throughout pipeline
- **Audit Trail**: Complete metadata and bias prevention documentation for reproducibility
- **Performance Monitoring**: Execution time and memory usage validation for POC development
