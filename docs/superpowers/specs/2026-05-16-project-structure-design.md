---
name: project-structure-design
description: Retail demand forecasting copilot project structure and architecture design
metadata:
  type: design-spec
---

# Retail Demand Forecasting Copilot - Project Structure Design

## Project Overview

This document defines the project structure and architecture for a retail demand forecasting copilot that provides ML-driven demand forecasting, pricing insights, and risk analytics using the Walmart M5 dataset. The system integrates three core components:

1. **Demand Forecasting** - Time series predictions for future product demand
2. **Demand Behavior Profiling** - Multi-label classification for product demand characteristics 
3. **GenAI Business Insight Generation** - Convert ML outputs into actionable business insights

## Architecture Decision

**Selected Architecture:** Modular Monolith with Hybrid Organization

**Why:** This approach provides the optimal balance for the project's requirements:
- **Shared Infrastructure Efficiency** - Both ML models use the same M5 data, feature engineering, and preprocessing
- **Domain Separation** - Forecasting and behavior profiling are distinct business problems with independent development paths
- **AWS Migration Ready** - Clear module boundaries enable future service decomposition
- **Development Simplicity** - Single deployment, unified logging, no network complexity between ML components

## Project Structure

The implementation follows the hybrid approach documented in `docs/project-info/dir-structure.md`:

### Core Infrastructure (`src/retail_demand_copilot/core/`)
- **Shared foundation** used across all modules
- Configuration management, logging, constants, exceptions, utilities
- **Why:** Prevents code duplication and ensures consistent behavior

### Data Layer (`src/retail_demand_copilot/data/`)
- **M5 dataset handling** - loaders, validators, schemas, repositories
- **Single source of truth** for data access patterns
- **Why:** Centralizes data quality and access logic used by both ML domains

### Preprocessing Layer (`src/retail_demand_copilot/preprocessing/`)
- **Shared data preparation** - cleaners, transformers, splitters, pipelines
- **Common preprocessing workflows** used by both forecasting and behavior profiling
- **Why:** Ensures consistent data preparation across models

### Feature Engineering (`src/retail_demand_copilot/features/`)
- **Temporal features** - date, weekday, seasonality patterns
- **Sales features** - lag features, rolling averages, demand metrics  
- **Calendar features** - holidays, events, SNAP benefits
- **Pricing features** - sell price changes and elasticity
- **Why:** Shared feature library prevents duplication and ensures feature consistency

### Business Domains (`src/retail_demand_copilot/domains/`)

#### Forecasting Domain
- **Purpose:** Time series forecasting for daily unit sales predictions
- **Components:** Domain-specific features, models, training, inference, evaluation, explainability
- **Output:** Daily predictions for specified forecast horizons

#### Behavior Profiling Domain  
- **Purpose:** Multi-label classification for demand behavior characterization
- **Components:** Trend analysis, variability calculation, segmentation models, rule-based labeling
- **Output:** Trend, seasonality, variability, and movement type labels

#### Business Explanations Domain
- **Purpose:** GenAI-powered conversion of ML outputs to business insights
- **Components:** LLM client, prompt templates, output formatting, insight processors
- **Output:** Natural language business recommendations

### Orchestration Layer (`src/retail_demand_copilot/pipelines/`)
- **Training pipeline** - Coordinated training across both ML domains
- **Inference pipeline** - Combined forecasting and profiling predictions
- **Insight pipeline** - End-to-end flow from input to business insights

### API Layer (`src/retail_demand_copilot/api/`)
- **FastAPI backend** with endpoints for static user input format
- **Input:** Store ID, Item ID, Forecast Horizon
- **Output:** Forecasts + Behavior Profile + Business Insights

## Data Flow Architecture

```
Static User Input (Store ID, Item ID, Horizon)
    ↓
Historical M5 Data Loading (data layer)
    ↓
Feature Engineering (shared features)
    ↓
┌─────────────────────┬─────────────────────┐
│ Forecasting Domain  │ Behavior Profiling  │
│ - Time series model │ - Multi-label model │
│ - Daily predictions │ - Behavior labels   │
└─────────────────────┴─────────────────────┘
    ↓
Business Explanations Domain (GenAI integration)
    ↓
Business Insight Output
```

## Key Design Principles

### Separation of Concerns
- **Shared Infrastructure:** Common utilities, data access, feature engineering
- **Domain Isolation:** Business logic separated by forecasting vs. behavior profiling
- **Clean Interfaces:** Well-defined contracts between layers

### Scalability Considerations
- **Modular Design:** Clear boundaries for future service decomposition
- **Container Ready:** Structure supports Docker containerization
- **AWS Migration Path:** Domains can become independent services

### ML Pipeline Integration
- **Experiment Tracking:** MLflow integration for model versioning
- **Model Management:** Centralized artifact storage
- **Monitoring Ready:** Data drift, model performance tracking infrastructure

## Development Workflow

1. **Experimentation Phase:** Jupyter notebooks for EDA and model development
2. **Implementation Phase:** Migrate proven approaches to structured modules  
3. **Integration Phase:** Combine models through orchestration pipelines
4. **Deployment Phase:** FastAPI + Streamlit for user interface

## Implementation Sequence Alignment

The structure supports the documented implementation sequence:
1. **Data Collection & EDA** → `notebooks/` and `data/` modules
2. **Preprocessing Pipeline** → `preprocessing/` modules  
3. **Feature Engineering** → `features/` modules
4. **Model Training** → `domains/*/training/` modules
5. **Inference Pipeline** → `pipelines/` and `domains/*/inference/` modules
6. **API Development** → `api/` modules
7. **UI Development** → `ui/streamlit/` modules

## Quality Assurance

- **Testing Strategy:** Unit tests for modules, integration tests for pipelines, API tests for endpoints
- **Configuration Management:** YAML-based configuration with environment-specific overrides
- **Monitoring Infrastructure:** Built-in data quality, drift detection, and performance monitoring

This project structure provides a solid foundation for developing a production-ready retail demand forecasting system that can scale from local development to AWS deployment while maintaining clean architecture and development efficiency.