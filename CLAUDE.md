# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: File Access Restrictions

**NEVER access or read files from `docs/my-docs/` directory.** This directory contains outdated or misleading information that should not be used for project understanding or CLAUDE.md updates. Only use `docs/project-info/` for authoritative project documentati on.

## Project Overview

A retail intelligence system that predicts future product-store sales, segments products by demand behavior, and converts ML outputs into business recommendations using an LLM.

## Data Architecture

The project uses the Walmart M5 dataset with this data flow:
- **Primary data**: `sales_train_validation.csv` (daily unit sales per product-store combination)
- **Calendar data**: `calendar.csv` (dates, weekdays, events, holidays)
- **Processing unit**: Each `item_id + store_id` combination is treated as one demand series
- **Static input format**: Store ID, Item ID, Forecast Horizon (e.g., "CA_1", "FOODS_3_090", "7 days")

## Development Environment

### Git Configuration

Before any git commands, source the environment variables from `.env`:
```bash
# Source git configuration
source .env
git config user.name "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"
```

**Important**: Always use the git user from `.env` and never include "Co-Authored-By" lines in commit messages.

### Setup Commands

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies for different use cases
uv sync                           # Runtime dependencies only
uv sync --extra jupyternotebook  # Add Jupyter notebook dependencies
uv sync --extra lint             # Add linting/formatting tools
uv sync --extra test             # Add testing tools
uv sync --all-extras             # Full development environment

# Register Jupyter kernel
uv run python -m ipykernel install --user --name demand-forecast-intelligence --display-name "Python (demand-forecast-intelligence)"
```

### Code Quality Commands

```bash
# Check code formatting and style
uv run ruff check .              # Linting
uv run black --check .           # Code formatting
uv run isort --check-only .      # Import sorting
uv run mypy src                  # Type checking

# Auto-fix formatting
uv run black .
uv run isort .
uv run ruff check . --fix
```

## Source Code Architecture

### Package Structure (src/demand_forecast_intelligence/)

- **core/**: Shared foundation (config, logging, constants, exceptions, utilities)
- **data/**: Data access layer (loaders, validators, schemas, repositories, sampling)
- **preprocessing/**: Data preparation (cleaners, transformers, features, splitters, pipelines)
- **model_development/**: ML domains
  - `forecasting/`: Demand forecasting model training
  - `segmentation/`: Segmentation model training
- **genai_business_insights/**: LLM integration for business insights
- **pipelines/**: End-to-end orchestration workflows
- **orchestration/**: Airflow integration helpers
- **monitoring/**: Post-deployment monitoring (drift, data quality, performance)
- **api/**: FastAPI backend application
- **ui/**: Streamlit dashboard interface

## Sample Dataset Generation

The project includes an anti-bias sampling system to create representative datasets from M5 data:

```bash
# Generate sample dataset with default settings (1400 items, ~50% reduction)
python scripts/create_sample_dataset.py

# Custom sample size with validation report
python scripts/create_sample_dataset.py --target-items 1000 --validate

# Specify custom directories
python scripts/create_sample_dataset.py --data-dir /path/to/m5/data --output-dir /path/to/output
```

The sampling uses **behavioral stratification with random selection** to prevent bias toward high-volume, stable products while ensuring challenging intermittent/sparse demand patterns are properly represented.

## Business Context

The system generates static insights (no chatbot interface) with this flow:
```
User Input (Store ID, Item ID, Horizon) → 
Feature Engineering → 
Forecasting + Segmentation → 
GenAI Business Insight Generation
```

Outputs should be interpretable for inventory planning decisions at Walmart scale.