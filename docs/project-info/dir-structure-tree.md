```text
demand_forecast_intelligence/              # Root project directory for demand forecasting intelligence system
├── data/                                  # Local dataset storage, not Python package code
│   ├── raw/                               # Original untouched M5 CSV files
│   ├── interim/                           # Temporary intermediate data during processing
│   ├── processed/                         # Cleaned and model-ready datasets
│   ├── samples/                           # Small sampled datasets created from large raw/processed data
│   └── external/                          # External/reference datasets, if added later
│
├── src/                                   # Main application source code
│   └── demand_forecast_intelligence/      # Installable Python package for the project
│       ├── core/                          # Shared foundation used across all modules
│       │   ├── config/                    # Configuration loading and environment handling
│       │   ├── logging/                   # Centralized application logging setup
│       │   ├── constants/                 # Project-wide constants and fixed values
│       │   ├── exceptions/                # Custom exception classes
│       │   └── utils/                     # Generic reusable helper functions
│       │
│       ├── data/                          # Code for reading, validating, sampling, and accessing data
│       │   ├── loaders/                   # Load M5 sales, calendar, and price files
│       │   ├── validators/                # Data quality and schema validation checks
│       │   ├── schemas/                   # Data contracts and expected column definitions
│       │   ├── repositories/              # Reusable data access and query logic
│       │   └── sampling/                  # Code to create small representative datasets from large files
│       │
│       ├── preprocessing/                 # Data cleaning and preparation before modeling
│       │   ├── cleaners/                  # Missing value, invalid value, and duplicate handling logic
│       │   ├── transformers/              # Scaling, encoding, and transformation logic
│       │   ├── splitters/                 # Train/test and time-based split logic
│       │   └── pipelines/                 # Reusable preprocessing workflows
│       │
│       ├── features/                      # Shared feature engineering logic
│       │   ├── common/                    # Features used by multiple models
│       │   ├── temporal/                  # Date, weekday, month, and seasonality features
│       │   ├── sales/                     # Lag, rolling average, and sales history features
│       │   ├── calendar/                  # Holiday, event, and SNAP-related features
│       │   └── pricing/                   # Sell price and price-change features
│       │
│       ├── domains/                       # Business-specific ML modules
│       │   ├── forecasting/               # Demand forecasting model domain
│       │   │   ├── features/              # Forecasting-specific feature logic
│       │   │   ├── models/                # Forecasting model classes and wrappers
│       │   │   ├── training/              # Forecasting training pipeline logic
│       │   │   ├── inference/             # Forecast generation logic
│       │   │   ├── evaluation/            # Forecast metrics and backtesting logic
│       │   │   └── explainability/        # Forecast interpretation methods
│       │   │
│       │   ├── segmentation/              # Customer/item behavior segmentation model domain
│       │   │   ├── features/              # Segmentation-specific feature logic
│       │   │   ├── models/                # Segmentation model classes and wrappers
│       │   │   ├── training/              # Segmentation model training pipeline logic
│       │   │   ├── inference/             # Segmentation label prediction logic
│       │   │   ├── evaluation/            # Segmentation metrics and validation logic
│       │   │   └── rules/                 # Rule-based segmentation label logic
│       │   │
│       │   └── genai_insights/                  # GenAI business insight generation domain
│       │       ├── genai/                 # LLM client and integration code
│       │       ├── prompts/               # Prompt templates for insight generation
│       │       ├── templates/             # Business explanation templates
│       │       ├── processors/            # Logic to combine forecast and profile outputs
│       │       └── formatters/            # Format final insight responses
│       │
│       ├── pipelines/                     # End-to-end orchestration across domains
│       │   ├── data_preparation_pipeline.py # Ingestion, validation, preprocessing, and feature generation pipeline
│       │   ├── training_pipeline.py       # Training pipeline for ML models
│       │   ├── inference_pipeline.py      # Forecasting and segmentation inference pipeline
│       │   ├── monitoring_pipeline.py     # Data quality, drift, and model performance monitoring pipeline
│       │   ├── retraining_pipeline.py     # Retraining pipeline triggered by schedule, drift, or performance degradation
│       │   └── insight_pipeline.py        # GenAI business insight generation pipeline
│       │
│       ├── orchestration/                 # Orchestration helpers used by Airflow and other schedulers
│       │   ├── airflow/                   # Shared Airflow utilities, operators, sensors, and callbacks
│       │   └── tasks/                     # Reusable task-level wrappers for DAG steps
│       │
│       ├── monitoring/                    # Post-model checks and monitoring logic
│       │   ├── drift/                     # Data drift and prediction drift detection logic
│       │   ├── data_quality/              # Ongoing input data quality monitoring logic
│       │   └── model_performance/         # Model metric tracking over time
│       │
│       ├── api/                           # FastAPI backend application
│       │   ├── main.py                    # FastAPI app entry point
│       │   ├── endpoints/                 # API route definitions
│       │   ├── schemas/                   # Request and response models
│       │   ├── middleware/                # Auth, logging, and request middleware
│       │   └── dependencies/              # Shared FastAPI dependency providers
│       │
│       └── ui/                            # User-facing interface code
│           └── streamlit/                 # Streamlit dashboard and application pages
│
├── dags/                                  # Apache Airflow DAG definitions
│   ├── data_preparation_dag.py            # Data ingestion → validation/profiling → preprocessing → features → versioned data
│   ├── training_dag.py                    # Load data → train model → track in MLflow → evaluate → validate → register/promote
│   ├── batch_inference_dag.py             # Load production model → load input data → predict → store outputs → log metrics
│   ├── monitoring_dag.py                  # Data quality → prediction quality → drift detection → performance tracking
│   └── retraining_dag.py                  # Scheduled/drift/performance-triggered retraining DAG
│
├── docker/                                # Docker-related files for local and deployment environments
│   ├── Dockerfile.api                     # Container image definition for FastAPI backend
│   ├── Dockerfile.streamlit               # Container image definition for Streamlit UI
│   ├── Dockerfile.airflow                 # Container image definition/customization for Airflow
│   ├── docker-compose.yml                 # Local multi-service setup for app components
│   ├── docker-compose.airflow.yml         # Local Airflow setup with scheduler, webserver, and dependencies
│   └── entrypoints/                       # Startup scripts used by Docker containers
│
├── terraform/                             # Infrastructure as Code for AWS deployment
│   ├── environments/                      # Environment-specific Terraform configurations
│   │   ├── dev/                           # Terraform configuration for development environment
│   │   │   ├── main.tf                    # Main Terraform resources for dev
│   │   │   ├── variables.tf               # Input variables for dev
│   │   │   ├── outputs.tf                 # Output values for dev
│   │   │   └── terraform.tfvars.example   # Example dev variable values
│   │   ├── staging/                       # Terraform configuration for staging environment
│   │   │   ├── main.tf                    # Main Terraform resources for staging
│   │   │   ├── variables.tf               # Input variables for staging
│   │   │   ├── outputs.tf                 # Output values for staging
│   │   │   └── terraform.tfvars.example   # Example staging variable values
│   │   └── prod/                          # Terraform configuration for production environment
│   │       ├── main.tf                    # Main Terraform resources for prod
│   │       ├── variables.tf               # Input variables for prod
│   │       ├── outputs.tf                 # Output values for prod
│   │       └── terraform.tfvars.example   # Example prod variable values
│   │
│   ├── modules/                           # Reusable Terraform modules
│   │   ├── ec2/                           # EC2 instance, security group, and key pair resources
│   │   ├── s3/                            # S3 buckets for datasets, artifacts, and logs
│   │   ├── iam/                           # IAM roles, policies, and instance profiles
│   │   ├── vpc/                           # VPC, subnets, route tables, and networking resources
│   │   ├── ecr/                           # Elastic Container Registry repositories
│   │   ├── cloudwatch/                    # CloudWatch logs, metrics, and alarms
│   │   └── airflow/                       # Optional infrastructure resources for Airflow
│   │
│   ├── backend.tf                         # Remote Terraform state backend configuration
│   ├── providers.tf                       # AWS provider and provider version configuration
│   ├── versions.tf                        # Terraform version requirements
│   └── README.md                          # Terraform usage and provisioning instructions
│
├── deployment/                            # Deployment-related infrastructure scripts and configs
│   └── aws/                               # AWS deployment files
│       ├── ec2/                           # EC2 deployment configuration
│       │   ├── user_data.sh               # EC2 bootstrap script
│       │   ├── setup_ec2.sh               # Script to install Docker, dependencies, and project services
│       │   ├── deploy.sh                  # Script to pull/build images and deploy application
│       │   ├── nginx.conf                 # Optional Nginx reverse proxy configuration
│       │   └── systemd/                   # Optional systemd service definitions
│       ├── iam/                           # IAM policy templates or deployment notes
│       ├── s3/                            # S3 bucket setup or sync scripts
│       └── README.md                      # AWS deployment instructions
│
├── docs/                                  # Empty directory reserved for project documentation
│
├── notebooks/                             # Exploration and experimentation notebooks
│   ├── 01_data_sanity.ipynb               # Initial loading and data quality checks
│   ├── 02_eda.ipynb                       # Exploratory data analysis
│   ├── 03_hypothesis_testing.ipynb        # Statistical tests after EDA
│   ├── 04_forecasting_experiments.ipynb   # Forecasting model experiments
│   └── 05_behavior_profiling_experiments.ipynb # Behavior profiling and segmentation experiments
│
├── experiments/                           # Experiment tracking artifacts
│   └── mlflow/                            # Local MLflow runs and metadata
│
├── models/                                # Saved trained model artifacts
│   ├── forecasting/                       # Forecasting model files
│   └── segmentation/                      # Segmentation model files
│
├── reports/                               # Generated analysis outputs
│   ├── figures/                           # EDA charts and model plots
│   └── metrics/                           # Evaluation reports and metric files
│
├── tests/                                 # Automated test suite
│   ├── unit/                              # Tests for individual functions and classes
│   ├── integration/                       # Tests for pipelines and module interactions
│   ├── api/                               # Tests for FastAPI endpoints
│   ├── orchestration/                     # Tests for Airflow DAG and task logic
│   └── fixtures/                          # Small sample test datasets and mock objects
│
├── configs/                               # YAML configuration files
│   ├── data.yaml                          # Dataset paths and data settings
│   ├── sampling.yaml                      # Sampling size, strategy, seed, and output paths
│   ├── forecasting.yaml                   # Forecasting model and training settings
│   ├── behavior_profiling.yaml            # Segmentation/profiling model and rule settings
│   ├── genai.yaml                         # GenAI prompt, provider, and model settings
│   ├── airflow.yaml                       # DAG schedules, retries, and orchestration settings
│   ├── aws.yaml                           # EC2, S3, region, and deployment settings
│   └── app.yaml                           # API and UI application settings
│
├── scripts/                               # Utility scripts for local, development, and operations workflows
│   ├── sample_dataset.py                  # CLI script to create small datasets
│   ├── run_data_preparation.py            # Script to run data preparation locally
│   ├── run_training.py                    # Script to run training locally
│   ├── run_batch_inference.py             # Script to run batch inference locally
│   └── run_monitoring.py                  # Script to run monitoring locally
│
├── pyproject.toml                         # Python project metadata, dependencies, and build configuration
├── README.md                              # Project overview, setup instructions, and usage guide
├── .env.example                           # Example environment variables
└── .gitignore                             # Files and folders Git should ignore
```
