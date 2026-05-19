

```text
demand_forecast_intelligence/
├── data/                         # Local dataset storage, not Python package code
│   ├── raw/                      # Original untouched M5 CSV files
│   ├── interim/                  # Temporary intermediate data during processing
│   ├── processed/                # Cleaned and model-ready datasets
│   ├── samples/                  # Small sampled datasets created from large raw/processed data
│   └── external/                 # External/reference data, if added later
│
├── src/                          # Main application source code
│   └── demand_forecast_intelligence/    # Installable Python package for the project
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
│       │   ├── repositories/     # Reusable data access/query logic
│       │   └── sampling/         # Code to create small representative datasets from large files
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
│       │   ├── segmentation/     # Segmentation model domain
│       │   │   ├── features/     # Segmentation-specific features
│       │   │   ├── models/       # Segmentation model classes and wrappers
│       │   │   ├── training/     # Segmentation model training pipeline
│       │   │   ├── inference/    # Segmentation label prediction logic
│       │   │   ├── evaluation/   # Segmentation metrics and validation
│       │   │   └── rules/        # Rule-based labels
│       │   │
│       │   └── insights/         # GenAI business insight generation domain
│       │       ├── genai/        # LLM client and integration code
│       │       ├── prompts/      # Prompt templates for insight generation
│       │       ├── templates/    # Business explanation templates
│       │       ├── processors/   # Combine forecast and profile outputs
│       │       └── formatters/   # Format final insight responses
│       │
│       ├── pipelines/            # End-to-end orchestration across domains
│       │   ├── data_preparation_pipeline.py # Ingestion, validation, preprocessing, features
│       │   ├── training_pipeline.py         # Runs training flow for ML models
│       │   ├── inference_pipeline.py        # Runs forecast + profiling inference
│       │   ├── monitoring_pipeline.py       # Runs quality, drift, and performance checks
│       │   ├── retraining_pipeline.py       # Re-runs training based on schedule/drift/performance
│       │   └── insight_pipeline.py          # Runs final GenAI insight generation
│       │
│       ├── orchestration/         # Orchestration helpers used by Airflow and other schedulers
│       │   ├── airflow/           # Shared Airflow utilities, operators, sensors, callbacks
│       │   └── tasks/             # Reusable task-level wrappers for DAG steps
│       │
│       ├── monitoring/            # Post-model checks and monitoring logic
│       │   ├── drift/             # Data and prediction drift detection
│       │   ├── data_quality/      # Ongoing input data quality monitoring
│       │   └── model_performance/ # Model metric tracking over time
│       │
│       ├── api/                   # FastAPI backend application
│       │   ├── main.py            # FastAPI app entry point
│       │   ├── endpoints/         # API route definitions
│       │   ├── schemas/           # Request and response models
│       │   ├── middleware/        # Auth, logging, and request middleware
│       │   └── dependencies/      # Shared FastAPI dependency providers
│       │
│       └── ui/                    # User-facing interface code
│           └── streamlit/         # Streamlit dashboard/application pages
│
├── dags/                         # Apache Airflow DAG definitions
│   ├── data_preparation_dag.py   # Ingestion → validation/profiling → preprocessing → features → versioned data
│   ├── training_dag.py           # Load data → train → MLflow tracking → evaluate → validate → register/promote
│   ├── batch_inference_dag.py    # Load model → load input → predict → store outputs → log inference metrics
│   ├── monitoring_dag.py         # Data quality → prediction quality → drift → performance tracking
│   └── retraining_dag.py         # Triggered retraining from schedule, drift, or performance degradation
│
├── docker/                       # Docker-related files
│   ├── Dockerfile.api            # Container image for FastAPI backend
│   ├── Dockerfile.streamlit      # Container image for Streamlit UI
│   ├── Dockerfile.airflow        # Container image/customization for Airflow
│   ├── docker-compose.yml        # Local multi-service setup
│   ├── docker-compose.airflow.yml # Local Airflow setup
│   └── entrypoints/              # Startup scripts for containers
│
├── deployment/                   # Deployment-related infrastructure and scripts
│   └── aws/                      # AWS deployment files
│       ├── ec2/                  # EC2 deployment configuration
│       │   ├── user_data.sh      # EC2 bootstrap script
│       │   ├── setup_ec2.sh      # Install Docker, dependencies, and project services
│       │   ├── deploy.sh         # Pull/build images and deploy application
│       │   ├── nginx.conf        # Optional reverse proxy config
│       │   └── systemd/          # Optional service definitions
│       ├── iam/                  # IAM policy templates or notes
│       ├── s3/                   # S3 bucket setup or sync scripts
│       └── README.md             # AWS deployment instructions
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
│   └── segmentation/             # Segmentation model files
│
├── reports/                      # Generated analysis outputs
│   ├── figures/                  # EDA charts and model plots
│   └── metrics/                  # Evaluation reports and metric files
│
├── tests/                        # Automated test suite
│   ├── unit/                     # Tests for individual functions/classes
│   ├── integration/              # Tests for pipelines and module interaction
│   ├── api/                      # Tests for FastAPI endpoints
│   ├── orchestration/            # Tests for Airflow DAG/task logic
│   └── fixtures/                 # Small sample test datasets and mock objects
│
├── configs/                      # YAML configuration files
│   ├── data.yaml                 # Dataset paths and data settings
│   ├── sampling.yaml             # Sampling size, strategy, seed, and output paths
│   ├── forecasting.yaml          # Forecasting model/training settings
│   ├── behavior_profiling.yaml   # Profiling model/rule settings
│   ├── genai.yaml                # GenAI prompt/model settings
│   ├── airflow.yaml              # DAG schedules, retries, and orchestration settings
│   ├── aws.yaml                  # EC2, S3, region, and deployment settings
│   └── app.yaml                  # API/UI application settings
│
├── scripts/                      # Utility scripts for local/dev/ops workflows
│   ├── sample_dataset.py         # CLI script to create small datasets
│   ├── run_data_preparation.py   # Run data preparation locally
│   ├── run_training.py           # Run training locally
│   ├── run_batch_inference.py    # Run batch inference locally
│   └── run_monitoring.py         # Run monitoring locally
│
├── docs                          # To store docs
├── pyproject.toml                # Python project metadata and dependencies
├── README.md                     # Project overview and setup instructions
├── .env.example                  # Example environment variables
└── .gitignore                    # Files/folders Git should ignore
```
