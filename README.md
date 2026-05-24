# demand-forecast-intelligence
ML-driven demand forecasting, pricing insights, and risk analytics with an agentic decision copilot.

## Table of Contents
- [Python Environment Setup](#python-environment-setup)
- [Exploratory Data Analysis (EDA) Framework](#exploratory-data-analysis-eda-framework)
  - [EDA Framework Architecture](#eda-framework-architecture)  
  - [Quick Start - EDA Demo](#quick-start---eda-demo)
  - [EDA Usage Examples](#eda-usage-examples)
  - [EDA Configuration](#eda-configuration)
  - [EDA Output Structure](#eda-output-structure)
  - [Troubleshooting EDA](#troubleshooting-eda)
  - [Integration with Model Development](#integration-with-model-development)

## Python Environment Setup

This project uses `pyproject.toml` as the main package definition and `uv.lock` for reproducible dependency resolution. Use a local `.venv` for development; it is intentionally ignored by git.

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create the Virtual Environment

```bash
uv venv
source .venv/bin/activate
```

### Install Dependencies

```bash
# Runtime dependencies only
uv sync

# Runtime + notebook dependencies
uv sync --extra jupyternotebook

# Runtime + linting/formatting/type-checking tools
uv sync --extra lint

# Runtime + test tools
uv sync --extra test

# Full local development environment
uv sync --all-extras
```

### Jupyter Kernel

After installing notebook dependencies, register the local environment as a Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name demand-forecast-intelligence --display-name "Python (demand-forecast-intelligence)"
```

### Common Commands

```bash
uv run pytest
uv run ruff check .
uv run black --check .
uv run isort --check-only .
uv run mypy src
```

## Exploratory Data Analysis (EDA) Framework

This project includes a comprehensive, modular EDA framework specifically designed for the M5 Walmart demand forecasting dataset. The framework provides automated analysis across multiple phases with dependency management and results caching.

### EDA Framework Architecture

The EDA framework is organized into **4 phases** and **8 subgroups** covering 12 analysis steps:

```
Phase 1: Data Understanding
├── Subgroup 1A: Business Context (Step 1)
└── Subgroup 1B: High-level Trends (Step 3)

Phase 2: Feature Analysis  
├── Subgroup 2A: Feature Profiling (Step 5)
└── Subgroup 2B: Relationship Analysis (Steps 6, 7)

Phase 3: Time Patterns
├── Subgroup 3A: Time Series Analysis (Steps 8, 9)  
└── Subgroup 3B: Advanced Pattern Detection (Steps 10, 11)

Phase 4: Model Preparation
├── Subgroup 4A: Data Quality Assessment (Step 12)
├── Subgroup 4B: Leakage Detection (Step 13)
└── Subgroup 4C: Model Readiness Summary (Step 15)
```

### Quick Start - EDA Demo

Run the complete framework demonstration:

```bash
# Navigate to EDA directory
cd eda/

# Run framework demonstration (no dataset required)
python main_eda_demo.py

# View captured output
cat ../main_eda_demo_output.txt
```

This will show the framework capabilities, configuration, and architecture without requiring M5 data files.

### EDA Usage Examples

#### 1. Initialize EDA Context
```python
from eda.config import get_m5_context
from eda.utils.core.orchestrator import EDAOrchestrator

# Create M5-configured context
ctx = get_m5_context(
    data_dir="data/raw",           # M5 CSV files location
    output_dir="data/eda/outputs", # Analysis results
    plots_dir="eda/plots"          # Generated visualizations
)

# Initialize orchestrator
orchestrator = EDAOrchestrator(ctx)
```

#### 2. Run Individual Analysis Steps
```python
# Run specific EDA step
result = orchestrator.run_step(1)  # Business Context Analysis

# Run specific subgroup  
result = orchestrator.run_subgroup('1A')  # Business Context subgroup

# Run complete phase
result = orchestrator.run_phase(1)  # Data Understanding phase
```

#### 3. Execute Complete EDA Pipeline
```python
# Run full 12-step analysis (requires M5 dataset)
results = orchestrator.run_full_pipeline()

# Results automatically saved to:
# - data/eda/outputs/step_results/     (JSON results)
# - data/eda/outputs/reports/          (Text summaries) 
# - eda/plots/step_XX_*/               (Visualizations)
```

#### 4. Access Individual Services
```python
# Use specific analysis services directly
from eda.utils.services.data_understanding.business_context import BusinessContextService

service = BusinessContextService(ctx)
result = service.execute()
summary = service.generate_summary()
```

### EDA Configuration

The framework uses M5-specific configuration in `eda/config.py`:

```python
# Key analysis parameters
M5_CONFIG = {
    "analysis_params": {
        "missing_threshold": 0.05,      # 5% missing data tolerance
        "outlier_threshold": 0.95,      # 95th percentile outlier detection
        "correlation_threshold": 0.8,    # High correlation threshold
        "time_split_date": "2016-04-24", # Temporal cutoff (d_1913)
        "seasonality_periods": [7, 30, 365], # Weekly, monthly, yearly
        "bootstrap_samples": 1000,       # Statistical testing samples
        "confidence_level": 0.95         # Statistical confidence level
    },
    
    "m5_specifics": {
        "forecast_horizon": 28,          # Days to forecast
        "item_store_combinations": 30490, # Total series count
        "sparse_series_threshold": 0.5,  # >50% zeros = sparse
        "intermittent_threshold": 0.3    # >30% zeros = intermittent
    }
}
```

### EDA Output Structure

After running the EDA pipeline, outputs are organized as:

```
data/eda/outputs/
├── step_results/           # JSON analysis results per step
│   ├── step_1.json        # Business context analysis
│   ├── step_3.json        # High-level trends
│   └── ...
├── reports/               # Human-readable summaries
│   ├── step_1_summary.md  # Business context summary
│   └── ...
└── phase_summaries/       # Phase-level consolidated reports
    ├── phase_1_summary.md
    └── ...

eda/plots/                 # Generated visualizations
├── step1_business_context/
│   └── problem_definition_timeline.png
├── step3_trends/
├── step5_target_analysis/
└── ...
```

### EDA Dependencies and Validation

The framework includes automatic dependency validation:

```python
# Step dependencies (automatically managed)
STEP_DEPENDENCIES = {
    1: [],           # Business Context (no dependencies)
    3: [1],          # Trends (requires Business Context)
    5: [1, 3],       # Target Analysis (requires Context + Trends)
    6: [1, 5],       # Feature-Target (requires Context + Target)
    7: [5, 6],       # Relationships (requires Feature analysis)
    # ... (complete dependency chain)
}
```

### Troubleshooting EDA

**Missing M5 Dataset Files:**
```bash
# Check data availability
python -c "
from eda.config import validate_m5_data_availability
from pathlib import Path
availability = validate_m5_data_availability(Path('data/raw'))
print('M5 Data Availability:', availability)
"
```

**Memory Issues with Large Dataset:**
```python
# Use lazy loading (default behavior)
ctx = get_m5_context()
# Dataset loaded only when accessed: ctx.load_dataset()
```

**Clear Cached Results:**
```bash
# Remove cached analysis results
rm -rf data/eda/outputs/step_results/
rm -rf eda/plots/step*/
```

### EDA Testing

The framework includes comprehensive test coverage:

```bash
# Run EDA-specific tests
cd eda/
python -m pytest tests/ -v

# Test coverage report
python -m pytest tests/ --cov=utils --cov-report=html
```

### Integration with Model Development

The EDA results inform preprocessing and modeling decisions documented in:
- [`docs/project-info/model-development-recommendations.md`](docs/project-info/model-development-recommendations.md)

Key EDA insights for modeling:
- **Temporal Boundary**: d_1913 cutoff prevents data leakage
- **Demand Segments**: 5 behavioral patterns requiring specialized models  
- **Zero-Inflation**: 30%+ zeros in intermittent series need Hurdle models
- **Seasonality**: Weekly (7-day) and annual (365-day) patterns detected
- **Target Metric**: WRMSSE < 0.5 for competitive performance

### Advanced EDA Usage

For production scenarios with custom datasets or extended analysis:

```python
# Custom configuration override
custom_config = {
    "analysis_params": {
        "missing_threshold": 0.10,  # More tolerant threshold
        "seasonality_periods": [7, 14, 30]  # Custom seasonal patterns
    }
}

ctx = get_m5_context()
ctx.config.update(custom_config)

# Custom service implementation
class CustomAnalysisService(BaseEDAService):
    # Implement custom analysis following framework patterns
    pass
```

For detailed implementation guidance, see the comprehensive [EDA Framework Design Specification](docs/superpowers/specs/2026-05-24-m5-eda-framework-design.md).
