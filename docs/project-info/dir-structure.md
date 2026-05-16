demand-forecast-intelligence/
├── data/                         # Local dataset storage, not Python package code
│   ├── raw/                      # Original untouched M5 CSV files
│   ├── interim/                  # Temporary intermediate data during processing
│   ├── processed/                # Cleaned and model-ready datasets
│   └── external/                 # External/reference data, if added later
│
├── src/                          # Main application source code
│   └── retail_demand_copilot/    # Installable Python package for the project
│       ├── core/                 # Shared foundation used across all modules
│       │   ├── config/           # Config loading and environment handling
│       │   ├── logging/          # Centralized logging setup
│       │   ├── constants/        # Project-wide constants and fixed values
│       │   ├── exceptions/       # Custom error classes
│       │   └── utils/            # Generic helper functions
│       │
│       ├── data/                 # Code for reading, validating, and accessing data
│       │   ├── loaders/          # Load M5 sales, calendar, and price files
│       │   ├── validators/       # Data quality and schema validation checks
│       │   ├── schemas/          # Data contracts and expected column definitions
│       │   └── repositories/     # Reusable data access/query logic
│       │
│       ├── preprocessing/        # Data cleaning and preparation before modeling
│       │   ├── cleaners/         # Missing values, invalid values, duplicates
│       │   ├── transformers/     # Scaling, encoding, and transformations
│       │   ├── splitters/        # Train/test and time-based split logic
│       │   └── pipelines/        # Reusable preprocessing workflows
│       │
│       ├── features/             # Shared feature engineering logic
│       │   ├── common/           # Features used by multiple models
│       │   ├── temporal/         # Date, weekday, month, and seasonality features
│       │   ├── sales/            # Lag, rolling average, and sales history features
│       │   ├── calendar/         # Holiday, event, and SNAP-related features
│       │   └── pricing/          # Sell price and price-change features
│       │
│       ├── domains/              # Business-specific ML modules
│       │   ├── forecasting/      # Demand forecasting model domain
│       │   │   ├── features/     # Forecasting-specific feature logic
│       │   │   ├── models/       # Forecasting model classes and wrappers
│       │   │   ├── training/     # Forecasting training pipeline
│       │   │   ├── inference/    # Forecast generation logic
│       │   │   ├── evaluation/   # Forecast metrics and backtesting
│       │   │   └── explainability/ # Forecast interpretation methods
│       │   │
│       │   ├── behavior_profiling/ # Demand behavior profiling domain
│       │   │   ├── features/     # Trend, variability, and segmentation features
│       │   │   ├── models/       # Profiling and segmentation models
│       │   │   ├── training/     # Profiling model training logic
│       │   │   ├── inference/    # Behavior label prediction logic
│       │   │   ├── evaluation/   # Profiling metrics and validation
│       │   │   └── rules/        # Rule-based labels such as CV thresholds
│       │   │
│       │   └── business_explanations/         # GenAI business insight generation domain
│       │       ├── llm_client/        # LLM client and integration code
│       │       ├── prompts/      # Prompt templates for insight generation
│       │       ├── templates/    # Business explanation templates
│       │       ├── processors/   # Combine forecast and profile outputs
│       │       └── formatters/   # Format final insight responses
│       │
│       ├── pipelines/            # End-to-end orchestration across domains
│       │   ├── training_pipeline.py # Runs training flow for ML models
│       │   ├── inference_pipeline.py # Runs forecast + profiling inference
│       │   └── insight_pipeline.py # Runs final GenAI insight generation
│       │
│       ├── monitoring/           # Post-model checks and monitoring logic
│       │   ├── drift/            # Data and prediction drift detection
│       │   ├── data_quality/     # Ongoing input data quality monitoring
│       │   └── model_performance/ # Model metric tracking over time
│       │
│       ├── api/                  # FastAPI backend application
│       │   ├── main.py           # FastAPI app entry point
│       │   ├── endpoints/        # API route definitions
│       │   ├── schemas/          # Request and response models
│       │   ├── middleware/       # Auth, logging, and request middleware
│       │   └── dependencies/     # Shared FastAPI dependency providers
│       │
│       └── ui/                   # User-facing interface code
│           └── streamlit/        # Streamlit dashboard/application pages
│
├── notebooks/                    # Exploration and experimentation notebooks
│   ├── 01_data_sanity.ipynb      # Initial loading and data quality checks
│   ├── 02_eda.ipynb              # Exploratory data analysis
│   ├── 03_hypothesis_testing.ipynb # Statistical tests after EDA
│   ├── 04_forecasting_experiments.ipynb # Forecasting model experiments
│   └── 05_behavior_profiling_experiments.ipynb # Profiling experiments
│
├── experiments/                  # Experiment tracking artifacts
│   └── mlflow/                   # Local MLflow runs and metadata
│
├── models/                       # Saved trained model artifacts
│   ├── forecasting/              # Forecasting model files
│   └── behavior_profiling/       # Profiling/segmentation model files
│
├── reports/                      # Generated analysis outputs
│   ├── figures/                  # EDA charts and model plots
│   └── metrics/                  # Evaluation reports and metric files
│
├── tests/                        # Automated test suite
│   ├── unit/                     # Tests for individual functions/classes
│   ├── integration/              # Tests for pipelines and module interaction
│   ├── api/                      # Tests for FastAPI endpoints
│   └── fixtures/                 # Small sample test datasets and mock objects
│
├── configs/                      # YAML configuration files
│   ├── data.yaml                 # Dataset paths and data settings
│   ├── forecasting.yaml          # Forecasting model/training settings
│   ├── behavior_profiling.yaml   # Profiling model/rule settings
│   ├── genai.yaml                # GenAI prompt/model settings
│   └── app.yaml                  # API/UI application settings
│
├── pyproject.toml                # Python project metadata and dependencies
├── README.md                     # Project overview and setup instructions
├── .env.example                  # Example environment variables
└── .gitignore                    # Files/folders Git should ignore