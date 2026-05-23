# M5 EDA Framework Implementation Design

**Project:** Demand Forecast Intelligence  
**Date:** 2026-05-24  
**Scope:** Comprehensive EDA system for M5 Walmart dataset analysis

## Overview

This document specifies the implementation of a comprehensive Exploratory Data Analysis (EDA) framework for the M5 Walmart demand forecasting project. The system implements 12 of 15 EDA framework steps (excluding steps 2 and 14) using a hybrid approach that supports both interactive analysis and automated execution.

### Key Requirements

- **Complete framework coverage**: 12 EDA steps organized into 4 phases and 8 subgroups
- **Hybrid execution**: Interactive Jupyter notebooks + automated Python scripts
- **Modular architecture**: Service-based design with shared context and results caching
- **Static visualizations**: PNG plots only, no interactive charts
- **Industry standards**: Production-ready structure suitable for ML operations

## System Architecture

### Design Pattern: Modular Services with Shared Context

The implementation follows **Approach C** - a service-oriented architecture where:

- **EDAContext**: Shared state object containing data references, configuration, and results cache
- **Service Modules**: Focused classes handling specific EDA subgroups (8 services total)
- **EDAOrchestrator**: Main controller for executing phases, subgroups, or individual steps
- **Hybrid Interfaces**: Both Jupyter notebooks and Python scripts use the same underlying services

### Directory Structure

```
eda/
├── utils/                          # Core utilities (importable, testable)
│   ├── core/
│   │   ├── context.py              # EDAContext class
│   │   ├── orchestrator.py         # Main execution controller
│   │   └── base_service.py         # Base class for services
│   ├── services/                   # 8 service modules (one per subgroup)
│   │   ├── data_understanding/
│   │   │   ├── business_context.py      # Step 1: Problem understanding
│   │   │   └── data_quality_audit.py    # Step 3: Quality checks
│   │   ├── feature_analysis/
│   │   │   ├── individual_profiling.py  # Step 5: Feature distributions
│   │   │   └── relationship_analysis.py # Steps 6,7: Correlations
│   │   ├── time_patterns/
│   │   │   ├── temporal_analysis.py     # Step 8: Time-series patterns
│   │   │   ├── data_quality_patterns.py # Steps 9,10: Missing/outliers
│   │   │   └── segment_behavior.py      # Step 11: Segment analysis
│   │   └── model_preparation/
│   │       ├── feature_engineering.py   # Step 12: Engineering opportunities
│   │       ├── validation_strategy.py   # Step 13: Train-test comparison
│   │       └── model_design.py          # Step 15: Model implications
│   └── visualization/               # Plot generation utilities
│       ├── plots.py                # Common plotting functions
│       └── reports.py              # Report generation
├── notebooks/                      # Interactive Jupyter interfaces
│   ├── 01_data_understanding.ipynb
│   ├── 02_feature_analysis.ipynb  
│   ├── 03_time_patterns.ipynb
│   └── 04_model_preparation.ipynb
├── scripts/                        # Automated execution scripts
│   ├── run_full_eda.py            # Complete EDA pipeline
│   ├── run_phase.py               # Individual phase execution
│   └── run_subgroup.py            # Individual subgroup execution
├── plots/                          # Generated visualizations (PNG only)
│   ├── step_01_business_context/
│   ├── step_03_data_quality/
│   ├── step_05_feature_profiling/
│   └── ... (continues for all steps)
├── tests/                          # Minimal test coverage
│   ├── test_context.py
│   ├── test_orchestrator.py
│   └── integration/
└── config.py                       # M5-specific configuration
```

### Storage Locations

- **Code**: `eda/` (version controlled)
- **Plots**: `eda/plots/` (version controlled, static PNG files)
- **Analysis outputs**: `data/eda/outputs/` (local only, not version controlled)

## Core Classes & Interfaces

### EDAContext Class

```python
@dataclass
class EDAContext:
    # Data paths and configuration
    data_dir: Path = Path("data/raw")           # Input M5 datasets
    output_dir: Path = Path("data/eda/outputs") # Analysis results  
    plots_dir: Path = Path("eda/plots")         # Generated plots
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Dataset references (lazy-loaded)
    sales_data: Optional[pd.DataFrame] = None
    calendar_data: Optional[pd.DataFrame] = None  
    pricing_data: Optional[pd.DataFrame] = None
    
    # Analysis results cache (in-memory + persistent)
    results: Dict[str, Any] = field(default_factory=dict)
    
    # Core methods
    def load_dataset(self, name: str) -> pd.DataFrame
    def save_result(self, step_id: str, result: Any)
    def get_result(self, step_id: str) -> Any
    def save_plot(self, filename: str, fig) -> Path
```

### Base Service Interface

```python
class BaseEDAService:
    def __init__(self, context: EDAContext):
        self.ctx = context
        
    def execute(self) -> Dict[str, Any]:
        """Execute all steps in this service"""
        
    def validate_prerequisites(self) -> List[str]:
        """Check if required data/results are available"""
        
    def generate_summary(self) -> str:
        """Generate text summary of findings"""
```

### EDAOrchestrator Interface

```python
class EDAOrchestrator:
    def __init__(self, context: EDAContext):
        self.ctx = context
        
    def run_full_pipeline(self) -> Dict[str, Any]:
    def run_phase(self, phase_num: int) -> Dict[str, Any]:  
    def run_subgroup(self, subgroup_id: str) -> Dict[str, Any]:
    def run_step(self, step_num: int) -> Dict[str, Any]:
    def generate_report(self, scope: str = "full") -> str:
    def get_step_dependencies(self, step_num: int) -> List[int]
```

## EDA Step Organization

### Phase 1: Data Understanding

**Subgroup 1A: Business Context (Step 1)**
- Service: `BusinessContextService`
- Purpose: Define ML objectives, target variables, and leakage prevention rules
- Key outputs: Problem definition, available features at prediction time, temporal boundaries
- Framework correlation: "Understand the problem first" - prevents data leakage

**Subgroup 1B: Data Quality Audit (Step 3)**
- Service: `DataQualityAuditService`  
- Purpose: Validate data integrity, identify missing values, duplicates, impossible values
- Key outputs: Data quality scores, missing patterns, consistency issues
- Framework correlation: "Can I trust this dataset?" validation

### Phase 2: Feature Analysis

**Subgroup 2A: Individual Feature Profiling (Step 5)**
- Service: `IndividualProfilingService`
- Purpose: Analyze distributions, skewness, cardinality for each feature type
- Key outputs: Feature statistics, transformation needs, outlier thresholds
- Framework correlation: "Analyze individual features" - distributions and patterns

**Subgroup 2B: Relationship Analysis (Steps 6, 7)**
- Service: `RelationshipAnalysisService`
- Purpose: Feature-target relationships and feature-feature correlations
- Key outputs: Predictive relationships, multicollinearity detection, redundancy identification
- Framework correlation: "Study feature-target and feature-feature relationships"

### Phase 3: Time-Series & Patterns

**Subgroup 3A: Temporal Analysis (Step 8)**
- Service: `TemporalAnalysisService`
- Purpose: Time structure, seasonality, trends, validation strategy for time-series
- Key outputs: Seasonal patterns, trend analysis, time-based split recommendations
- Framework correlation: "Special time-series EDA" - temporal patterns and leakage prevention

**Subgroup 3B: Data Quality Patterns (Steps 9, 10)**
- Service: `DataQualityPatternsService`
- Purpose: Missing value patterns and outlier detection in temporal context
- Key outputs: Informative missingness, anomaly detection, temporal data quality issues
- Framework correlation: "Analyze missing values deeply" + "Identify outliers and anomalies"

**Subgroup 3C: Segment Behavior (Step 11)**
- Service: `SegmentBehaviorService`
- Purpose: Geographic, category, and temporal segment analysis
- Key outputs: Segment performance differences, behavioral patterns, fairness concerns
- Framework correlation: "Check segment-level behavior" - subgroup analysis

### Phase 4: Model Preparation

**Subgroup 4A: Feature Engineering (Step 12)**
- Service: `FeatureEngineeringService`
- Purpose: Identify transformation opportunities based on EDA findings
- Key outputs: Lag features, price features, calendar features, engineering recommendations
- Framework correlation: "Think about feature engineering opportunities"

**Subgroup 4B: Validation Strategy (Step 13)**
- Service: `ValidationStrategyService`
- Purpose: Compare train-test distributions, detect drift
- Key outputs: Validation approach, distribution comparisons, drift analysis
- Framework correlation: "Compare train and test distributions"

**Subgroup 4C: Model Design (Step 15)**
- Service: `ModelDesignService`
- Purpose: Synthesize EDA findings into modeling implications
- Key outputs: Architecture recommendations, metrics, preprocessing decisions
- Framework correlation: "Decide modeling implications"

## Data Flow & Dependencies

### Service Execution Dependencies

```
Phase 1: Foundation
├── 1A: BusinessContext (Step 1) → No dependencies
└── 1B: DataQualityAudit (Step 3) → Depends on [1]

Phase 2: Feature Understanding  
├── 2A: IndividualProfiling (Step 5) → Depends on [1, 3]
└── 2B: RelationshipAnalysis (Steps 6, 7) → Depends on [1, 5]

Phase 3: Patterns & Context
├── 3A: TemporalAnalysis (Step 8) → Depends on [1, 3]  
├── 3B: DataQualityPatterns (Steps 9, 10) → Depends on [3, 8]
└── 3C: SegmentBehavior (Step 11) → Depends on [6, 7]

Phase 4: Synthesis
├── 4A: FeatureEngineering (Step 12) → Depends on [5, 6, 7, 8]
├── 4B: ValidationStrategy (Step 13) → Depends on [8]
└── 4C: ModelDesign (Step 15) → Depends on [12, 13]
```

### Results Caching System

**In-Memory Cache:**
- Active during execution: `ctx.results[step_id] = analysis_outputs`
- Lost when session ends, used for cross-step dependencies

**Persistent Storage:**
```
data/eda/outputs/
├── step_results/
│   ├── step_01_business_context.json
│   ├── step_03_data_quality.json
│   ├── step_05_feature_profiling.json
│   └── ... (all 12 steps)
├── phase_summaries/
│   ├── phase_1_data_understanding.json
│   ├── phase_2_feature_analysis.json
│   ├── phase_3_time_patterns.json
│   └── phase_4_model_preparation.json
├── reports/
│   ├── full_eda_report.md
│   ├── executive_summary.md
│   └── technical_findings.json
└── cache/
    ├── dataset_stats.pkl
    └── correlation_matrices.pkl
```

## Visualization Strategy

### Static Plot Generation

**Plot Libraries:**
- **matplotlib/seaborn**: Primary plotting for statistical analysis
- **pandas.plot**: Quick exploratory visualizations
- **No interactive libraries**: Only static PNG outputs

**Plot Organization:**
```
eda/plots/
├── step_01_business_context/
│   └── problem_definition_summary.png
├── step_03_data_quality/
│   ├── missing_values_heatmap.png
│   ├── data_types_summary.png
│   └── duplicate_analysis.png
├── step_05_feature_profiling/
│   ├── sales_distributions_by_category.png
│   ├── price_distributions.png
│   └── calendar_feature_analysis.png
├── step_06_feature_target_relationships/
│   ├── category_sales_relationships.png
│   ├── price_elasticity_analysis.png
│   └── temporal_target_patterns.png
└── ... (continues for all 12 steps)
```

### Report Generation

**Output Formats:**
- **JSON**: Structured results for programmatic access
- **Markdown**: Human-readable reports with embedded PNG plots
- **PNG**: All visualizations as static images
- **Text summaries**: Statistical results printed to console/saved to files

**Report Types:**
```python
class ReportGenerator:
    def generate_step_report(self, step_id: str) -> str:
        """Markdown report for individual step with embedded plots"""
        
    def generate_phase_report(self, phase_num: int) -> str:
        """Combined report for entire phase"""
        
    def generate_executive_summary(self) -> str:
        """Business-focused summary for stakeholders"""
        
    def generate_technical_summary(self) -> str:
        """Technical summary for ML practitioners"""
```

## Configuration & M5 Dataset Integration

### M5-Specific Configuration

```python
# eda/config.py
M5_CONFIG = {
    "data_paths": {
        "sales_validation": "data/raw/sales_train_validation.csv",
        "sales_evaluation": "data/raw/sales_train_evaluation.csv", 
        "calendar": "data/raw/calendar.csv",
        "pricing": "data/raw/sell_prices.csv"
    },
    "output_paths": {
        "results": "data/eda/outputs",
        "plots": "eda/plots"
    },
    "analysis_params": {
        "outlier_threshold": 0.999,      # 99.9th percentile capping
        "correlation_threshold": 0.8,     # High correlation detection
        "missing_threshold": 0.1,         # 10% missing value concern
        "time_split_date": "2016-05-22"   # Training/validation boundary
    },
    "m5_specifics": {
        "item_store_combinations": 30490,
        "training_days": "d_1 to d_1913", 
        "validation_days": "d_1914 to d_1941",
        "categories": ["FOODS", "HOUSEHOLD", "HOBBIES"],
        "states": ["CA", "TX", "WI"]
    }
}
```

### Dataset Schema Integration

Each service will leverage the M5 schema documentation (`docs/project-info/schema-info.md`) for:
- Hierarchical relationships (state → store, category → department → item)
- Temporal mappings (d_X ↔ calendar dates)
- Join patterns (sales ← calendar ← pricing via wm_yr_wk)

## Testing Strategy

### Minimal Test Coverage (As Requested)

```
eda/tests/
├── test_context.py              # EDAContext functionality
├── test_orchestrator.py         # Execution flow and dependencies  
├── test_data_loading.py         # M5 dataset loading validation
└── integration/
    └── test_phase_execution.py  # End-to-end phase testing
```

**Key Test Areas:**
- **Data loading**: M5 datasets load correctly with expected schema
- **Service dependencies**: Prerequisite validation works properly
- **Results caching**: Save/load of intermediate results functions
- **Plot generation**: Plots are created and saved to correct locations

## Execution Interfaces

### Script-Based Execution

```bash
# Complete EDA pipeline
python eda/scripts/run_full_eda.py

# Individual phase execution  
python eda/scripts/run_phase.py --phase 1
python eda/scripts/run_phase.py --phase 2

# Subgroup-level execution
python eda/scripts/run_subgroup.py --subgroup "1A"  # Business Context
python eda/scripts/run_subgroup.py --subgroup "2B"  # Relationship Analysis

# Individual step execution
python eda/scripts/run_step.py --step 5  # Feature profiling
```

### Notebook Interface

```python
# Interactive analysis in Jupyter
from eda.utils.core.orchestrator import EDAOrchestrator
from eda.utils.core.context import EDAContext

# Initialize with M5 configuration
ctx = EDAContext.from_config()
orchestrator = EDAOrchestrator(ctx)

# Execute at different granularities
results = orchestrator.run_phase(1)           # Full phase
results = orchestrator.run_subgroup("2A")     # Specific subgroup
results = orchestrator.run_step(5)            # Individual step

# Access cached results
business_context = ctx.get_result("step_1")
quality_audit = ctx.get_result("step_3")
```

## Framework Correlation & Comments

Each service method includes comments linking back to the original EDA framework:

```python
def analyze_business_context(self):
    """
    EDA Framework Step 1: Understand the problem first
    
    From framework: "Before touching the data, clarify the ML objective"
    - What is the target variable?
    - What does one row represent?  
    - At prediction time, what information will be available?
    - Critical for preventing data leakage
    
    M5 Application: Define forecasting objectives, segmentation goals,
    and establish temporal boundaries for valid feature engineering.
    """
```

This ensures direct traceability between implementation and framework guidance.

## Implementation Benefits

### Production Readiness
- **Modular services**: Easy to maintain, test, and extend
- **Dependency management**: Prevents execution order issues
- **Results caching**: Efficient incremental analysis
- **Industry patterns**: Matches ML operations best practices

### Flexibility
- **Multiple interfaces**: Support both research and automation workflows
- **Granular execution**: Run full pipeline, phases, subgroups, or individual steps
- **Configuration-driven**: Easy adaptation to different datasets
- **Framework alignment**: Direct correlation with established EDA methodology

### M5-Specific Value
- **Time-series focus**: Specialized handling of temporal patterns and validation
- **Retail domain**: Category analysis, price elasticity, seasonal patterns
- **Scalability**: Handle 30,490 item-store combinations efficiently
- **Business insights**: Generate actionable recommendations for inventory planning

This design provides a comprehensive, production-ready EDA framework specifically tailored for the M5 demand forecasting project while maintaining flexibility for future enhancements.