# Demand Forecast Intelligence - Project Structure

This document describes the proposed directory and file structure for the `demand_forecast_intelligence` project. The structure supports data preparation, forecasting, segmentation, GenAI insights, Airflow orchestration, Docker-based deployment, AWS EC2 deployment, Terraform infrastructure, monitoring, testing, and documentation.

---

## 1. Root Directory

- `demand_forecast_intelligence/`
  - Root project directory for the complete demand forecasting intelligence system.
  - Contains source code, data folders, orchestration files, infrastructure code, deployment scripts, notebooks, tests, configuration files, and project documentation.

---

## 2. Data Directory

- `data/`
  - Local dataset storage.
  - This directory is not meant to be imported as Python package code.
  - Used to organize raw, intermediate, processed, sampled, and external datasets.

  - `raw/`
    - Stores original untouched M5 CSV files.
    - Files placed here should remain unchanged for reproducibility.

  - `interim/`
    - Stores temporary intermediate datasets created during processing.
    - Useful for debugging and staged pipeline outputs.

  - `processed/`
    - Stores cleaned and model-ready datasets.
    - These datasets are typically consumed by training, evaluation, and inference pipelines.

  - `samples/`
    - Stores small sampled datasets created from large raw or processed datasets.
    - Useful for development, testing, local experimentation, and faster debugging.

  - `external/`
    - Stores external or reference datasets if added later.
    - Examples may include holiday calendars, economic indicators, or third-party reference data.

---

## 3. Source Code Directory

- `src/`
  - Main application source code directory.
  - Uses the common `src` layout for Python projects.

  - `demand_forecast_intelligence/`
    - Installable Python package for the project.
    - Contains reusable application modules and domain-specific logic.

    ### 3.1 Core Module

    - `core/`
      - Shared foundation used across all modules.
      - Contains common configuration, logging, constants, exceptions, and utilities.

      - `config/`
        - Handles configuration loading and environment-specific settings.
        - Reads values from YAML files, environment variables, or `.env` files.

      - `logging/`
        - Contains centralized application logging setup.
        - Provides consistent log formatting and log-level management.

      - `constants/`
        - Stores project-wide constants and fixed values.
        - Examples include column names, default date formats, model names, and file naming conventions.

      - `exceptions/`
        - Contains custom exception classes.
        - Helps make error handling more explicit and domain-aware.

      - `utils/`
        - Contains generic reusable helper functions.
        - Used across data processing, modeling, API, pipelines, and monitoring.

    ### 3.2 Data Module

    - `data/`
      - Code for reading, validating, sampling, and accessing data.
      - Separates data access logic from transformation and modeling logic.

      - `loaders/`
        - Loads M5 sales, calendar, and price files.
        - Responsible for reading source files into dataframes or other in-memory formats.

      - `validators/`
        - Performs data quality and schema validation checks.
        - Checks missing values, duplicate rows, invalid values, required columns, and data types.

      - `schemas/`
        - Defines data contracts and expected column definitions.
        - Helps standardize input and output dataset formats.

      - `repositories/`
        - Contains reusable data access and query logic.
        - Provides a clean interface for fetching specific datasets or filtered data slices.

      - `sampling/`
        - Contains code to create small representative datasets from large files.
        - Useful for local development, quick experiments, tests, and demos.

    ### 3.3 Preprocessing Module

    - `preprocessing/`
      - Data cleaning and preparation before modeling.
      - Converts raw or interim data into consistent model-ready structures.

      - `cleaners/`
        - Handles missing values, invalid values, and duplicates.
        - Contains reusable cleaning functions or classes.

      - `transformers/`
        - Handles scaling, encoding, and transformation logic.
        - May include categorical encoding, numerical scaling, and data reshaping.

      - `splitters/`
        - Contains train/test and time-based split logic.
        - Ensures forecasting data is split correctly without time leakage.

      - `pipelines/`
        - Defines reusable preprocessing workflows.
        - Combines cleaners, transformers, and splitters into repeatable steps.

    ### 3.4 Feature Engineering Module

    - `features/`
      - Shared feature engineering logic used by multiple models.
      - Contains common, temporal, sales, calendar, and pricing feature generation.

      - `common/`
        - Features used by multiple modeling domains.
        - Includes generic derived features that are not specific to one model type.

      - `temporal/`
        - Date, weekday, month, quarter, year, and seasonality features.
        - Useful for capturing time-based demand patterns.

      - `sales/`
        - Lag, rolling average, cumulative sales, and sales history features.
        - Important for time-series forecasting models.

      - `calendar/`
        - Holiday, event, and SNAP-related features.
        - Uses calendar information to explain demand changes.

      - `pricing/`
        - Sell price and price-change features.
        - Captures effects of promotions, price movement, and relative price changes.

    ### 3.5 Domain Modules

    - `domains/`
      - Business-specific machine learning modules.
      - Separates forecasting, segmentation, and GenAI insight generation concerns.

      #### 3.5.1 Forecasting Domain

      - `forecasting/`
        - Demand forecasting model domain.
        - Contains forecasting-specific features, models, training, inference, evaluation, and explainability.

        - `features/`
          - Forecasting-specific feature logic.
          - Includes features that are only relevant to forecasting models.

        - `models/`
          - Forecasting model classes and wrappers.
          - May include baseline models, statistical models, machine learning models, and deep learning wrappers.

        - `training/`
          - Forecasting training pipeline logic.
          - Handles model fitting, experiment tracking hooks, and training artifacts.

        - `inference/`
          - Forecast generation logic.
          - Loads trained models and produces future demand predictions.

        - `evaluation/`
          - Forecast metrics and backtesting logic.
          - May include MAE, RMSE, MAPE, WAPE, SMAPE, and time-based validation.

        - `explainability/`
          - Forecast interpretation methods.
          - Explains drivers behind forecasted demand using feature importance or model explanation tools.

      #### 3.5.2 Segmentation Domain

      - `segmentation/`
        - Customer, product, or item behavior segmentation model domain.
        - Uses model-based or rule-based logic to assign behavior profiles.

        - `features/`
          - Segmentation-specific feature logic.
          - Includes features such as demand variability, sales velocity, seasonality strength, and lifecycle indicators.

        - `models/`
          - Segmentation model classes and wrappers.
          - May include clustering models, classification models, or profiling models.

        - `training/`
          - Segmentation model training pipeline logic.
          - Trains models used to assign behavior segments.

        - `inference/`
          - Segmentation label prediction logic.
          - Assigns segment labels to products, stores, or item-store combinations.

        - `evaluation/`
          - Segmentation metrics and validation logic.
          - Evaluates cluster quality, label stability, or business usefulness.

        - `rules/`
          - Rule-based segmentation label logic.
          - Useful for interpretable business rules and fallback labeling.

      #### 3.5.3 GenAI insights Domain

      - `genai_insights/`
        - GenAI business insight generation domain.
        - Combines forecast outputs, segmentation profiles, and business templates into final explanations.

        - `genai/`
          - LLM client and integration code.
          - Handles communication with the selected GenAI provider or local model.

        - `prompts/`
          - Prompt templates for insight generation.
          - Stores reusable prompts for forecast explanations and business summaries.

        - `templates/`
          - Business explanation templates.
          - Provides structured response formats for consistent insight output.

        - `processors/`
          - Logic to combine forecast and profile outputs.
          - Prepares structured context before sending it to the GenAI layer.

        - `formatters/`
          - Formats final insight responses.
          - Converts generated insights into API, UI, report, or dashboard-ready text.

    ### 3.6 Pipeline Module

    - `pipelines/`
      - End-to-end orchestration across domains.
      - Contains Python pipeline entry points used by scripts, Airflow DAGs, APIs, or manual runs.

      - `data_preparation_pipeline.py`
        - Runs ingestion, validation, preprocessing, and feature generation.
        - Produces versioned model-ready datasets and feature outputs.

      - `training_pipeline.py`
        - Runs the training flow for machine learning models.
        - Connects prepared data, model training, evaluation, and MLflow tracking.

      - `inference_pipeline.py`
        - Runs forecasting and segmentation inference.
        - Produces predictions, segment labels, and model outputs.

      - `monitoring_pipeline.py`
        - Runs data quality, drift, and model performance monitoring.
        - Tracks production data and prediction health over time.

      - `retraining_pipeline.py`
        - Runs retraining based on schedule, drift, or performance degradation.
        - Produces new candidate models for validation and promotion.

      - `insight_pipeline.py`
        - Runs final GenAI business insight generation.
        - Combines forecasts, segmentation profiles, and templates into business-readable insights.

    ### 3.7 Orchestration Module

    - `orchestration/`
      - Orchestration helpers used by Airflow and other schedulers.
      - Keeps DAG files clean by moving reusable orchestration logic into the package.

      - `airflow/`
        - Shared Airflow utilities, operators, sensors, and callbacks.
        - Used by DAGs to avoid duplicating orchestration logic.

      - `tasks/`
        - Reusable task-level wrappers for DAG steps.
        - Examples include task wrappers for data validation, training, inference, and monitoring.

    ### 3.8 Monitoring Module

    - `monitoring/`
      - Post-model checks and monitoring logic.
      - Tracks data quality, drift, and model performance after deployment.

      - `drift/`
        - Data drift and prediction drift detection logic.
        - Compares current production data against historical or training distributions.

      - `data_quality/`
        - Ongoing input data quality monitoring logic.
        - Checks missing values, schema changes, invalid values, and anomaly patterns.

      - `model_performance/`
        - Model metric tracking over time.
        - Tracks degradation in forecasting and segmentation model performance.

    ### 3.9 API Module

    - `api/`
      - FastAPI backend application.
      - Exposes model predictions, insights, health checks, and application endpoints.

      - `main.py`
        - FastAPI app entry point.
        - Creates the application instance and registers routers/middleware.

      - `endpoints/`
        - API route definitions.
        - Contains route files for forecasting, segmentation, insights, health checks, and metadata.

      - `schemas/`
        - Request and response models.
        - Defines API contracts using Pydantic models.

      - `middleware/`
        - Auth, logging, and request middleware.
        - Handles cross-cutting API concerns.

      - `dependencies/`
        - Shared FastAPI dependency providers.
        - Provides reusable dependencies for configuration, services, models, and clients.

    ### 3.10 UI Module

    - `ui/`
      - User-facing interface code.
      - Contains frontend or dashboard-related application code.

      - `streamlit/`
        - Streamlit dashboard and application pages.
        - Used to visualize forecasts, segments, metrics, and generated insights.

---

## 4. Airflow DAGs Directory

- `dags/`
  - Apache Airflow DAG definitions.
  - Contains workflow definitions for data preparation, training, inference, monitoring, and retraining.

  - `data_preparation_dag.py`
    - Defines the Data Preparation DAG.
    - Flow:
      - Data ingestion.
      - Data validation and profiling.
      - Preprocessing.
      - Feature generation.
      - Store versioned dataset and features.

  - `training_dag.py`
    - Defines the Training DAG.
    - Flow:
      - Load prepared data.
      - Train model.
      - Track experiment in MLflow.
      - Evaluate model.
      - Validate model.
      - Register or promote model.

  - `batch_inference_dag.py`
    - Defines the Batch Inference DAG.
    - Flow:
      - Load production model.
      - Load input data.
      - Generate predictions.
      - Store outputs.
      - Log inference metrics.

  - `monitoring_dag.py`
    - Defines the Monitoring DAG.
    - Flow:
      - Monitor data quality.
      - Monitor prediction quality.
      - Detect drift.
      - Track model performance.

  - `retraining_dag.py`
    - Defines the Retraining DAG.
    - Triggered by:
      - Schedule.
      - Drift detection.
      - Performance degradation.
    - Re-runs the training pipeline and registers a new candidate model.

---

## 5. Docker Directory

- `docker/`
  - Docker-related files for local and deployment environments.
  - Supports containerized development, testing, Airflow orchestration, API hosting, and UI hosting.

  - `Dockerfile.api`
    - Container image definition for the FastAPI backend.
    - Installs API dependencies and starts the API service.

  - `Dockerfile.streamlit`
    - Container image definition for the Streamlit UI.
    - Installs UI dependencies and starts the dashboard.

  - `Dockerfile.airflow`
    - Container image definition or customization for Airflow.
    - Adds project dependencies and DAG runtime requirements.

  - `docker-compose.yml`
    - Local multi-service setup for app components.
    - Can run API, UI, MLflow, database, or other local services together.

  - `docker-compose.airflow.yml`
    - Local Airflow setup.
    - Runs Airflow webserver, scheduler, metadata database, and related dependencies.

  - `entrypoints/`
    - Startup scripts used by Docker containers.
    - Useful for waiting on services, applying migrations, or bootstrapping runtime commands.

---

## 6. Terraform Directory

- `terraform/`
  - Infrastructure as Code for AWS deployment.
  - Used to provision cloud infrastructure such as EC2, S3, IAM, VPC, ECR, and CloudWatch.

  - `environments/`
    - Environment-specific Terraform configurations.
    - Separates dev, staging, and production infrastructure settings.

    - `dev/`
      - Terraform configuration for the development environment.

      - `main.tf`
        - Main Terraform resources for development.

      - `variables.tf`
        - Input variables for development.

      - `outputs.tf`
        - Output values for development.

      - `terraform.tfvars.example`
        - Example variable values for development.

    - `staging/`
      - Terraform configuration for the staging environment.

      - `main.tf`
        - Main Terraform resources for staging.

      - `variables.tf`
        - Input variables for staging.

      - `outputs.tf`
        - Output values for staging.

      - `terraform.tfvars.example`
        - Example variable values for staging.

    - `prod/`
      - Terraform configuration for the production environment.

      - `main.tf`
        - Main Terraform resources for production.

      - `variables.tf`
        - Input variables for production.

      - `outputs.tf`
        - Output values for production.

      - `terraform.tfvars.example`
        - Example variable values for production.

  - `modules/`
    - Reusable Terraform modules.
    - Provides common infrastructure blocks used across environments.

    - `ec2/`
      - EC2 instance, security group, and key pair resources.
      - Used to host application services on virtual machines.

    - `s3/`
      - S3 buckets for datasets, model artifacts, logs, and pipeline outputs.
      - Supports durable object storage.

    - `iam/`
      - IAM roles, policies, and instance profiles.
      - Controls AWS permissions for EC2, S3, ECR, CloudWatch, and other services.

    - `vpc/`
      - VPC, subnets, route tables, and networking resources.
      - Defines network isolation and routing.

    - `ecr/`
      - Elastic Container Registry repositories.
      - Stores Docker images for API, UI, Airflow, or worker services.

    - `cloudwatch/`
      - CloudWatch logs, metrics, and alarms.
      - Supports operational monitoring and alerting.

    - `airflow/`
      - Optional infrastructure resources for Airflow.
      - Can contain resources needed to run Airflow on EC2 or related AWS services.

  - `backend.tf`
    - Remote Terraform state backend configuration.
    - Defines where Terraform state is stored.

  - `providers.tf`
    - AWS provider and provider version configuration.
    - Defines cloud provider settings.

  - `versions.tf`
    - Terraform version requirements.
    - Ensures consistent Terraform and provider versions.

  - `README.md`
    - Terraform usage and provisioning instructions.
    - Explains how to initialize, plan, apply, and destroy infrastructure.

---

## 7. Deployment Directory

- `deployment/`
  - Deployment-related infrastructure scripts and configs.
  - Contains operational scripts separate from Terraform provisioning.

  - `aws/`
    - AWS deployment files.
    - Contains EC2, IAM, and S3 deployment helpers.

    - `ec2/`
      - EC2 deployment configuration.
      - Used after infrastructure provisioning to configure and deploy application services.

      - `user_data.sh`
        - EC2 bootstrap script.
        - Runs during instance startup to install or configure required software.

      - `setup_ec2.sh`
        - Script to install Docker, dependencies, and project services.
        - Used for manual or automated EC2 setup.

      - `deploy.sh`
        - Script to pull or build images and deploy the application.
        - Can restart services and update running containers.

      - `nginx.conf`
        - Optional Nginx reverse proxy configuration.
        - Routes traffic to API, UI, or Airflow services.

      - `systemd/`
        - Optional systemd service definitions.
        - Used to run project services as managed Linux services.

    - `iam/`
      - IAM policy templates or deployment notes.
      - Documents permissions required by deployed services.

    - `s3/`
      - S3 bucket setup or sync scripts.
      - Helps upload or synchronize datasets, artifacts, and logs.

    - `README.md`
      - AWS deployment instructions.
      - Explains how to deploy and operate the project on AWS.

---

## 8. Documentation Directory

- `docs/`
  - Empty directory reserved for future project documentation.
  - Can later contain architecture notes, API docs, runbooks, deployment guides, and design decisions.

---

## 9. Notebooks Directory

- `notebooks/`
  - Exploration and experimentation notebooks.
  - Used for analysis, EDA, hypothesis testing, forecasting experiments, and behavior profiling experiments.

  - `01_data_sanity.ipynb`
    - Initial loading and data quality checks.
    - Verifies basic structure, columns, nulls, and file readability.

  - `02_eda.ipynb`
    - Exploratory data analysis.
    - Studies demand patterns, seasonality, trends, outliers, and relationships.

  - `03_hypothesis_testing.ipynb`
    - Statistical tests after EDA.
    - Tests assumptions and validates observed patterns.

  - `04_forecasting_experiments.ipynb`
    - Forecasting model experiments.
    - Tests baseline and advanced forecasting approaches.

  - `05_behavior_profiling_experiments.ipynb`
    - Behavior profiling and segmentation experiments.
    - Explores clustering, rules, and profile generation.

---

## 10. Experiments Directory

- `experiments/`
  - Experiment tracking artifacts.
  - Stores metadata and outputs from modeling experiments.

  - `mlflow/`
    - Local MLflow runs and metadata.
    - Tracks parameters, metrics, artifacts, and model versions.

---

## 11. Models Directory

- `models/`
  - Saved trained model artifacts.
  - Stores serialized models and related model files.

  - `forecasting/`
    - Forecasting model files.
    - Contains trained demand forecasting models and related artifacts.

  - `segmentation/`
    - Segmentation model files.
    - Contains trained segmentation or profiling models and related artifacts.

---

## 12. Reports Directory

- `reports/`
  - Generated analysis outputs.
  - Stores charts, metrics, and other generated reporting artifacts.

  - `figures/`
    - EDA charts and model plots.
    - Stores visual outputs such as trend charts, residual plots, and feature importance plots.

  - `metrics/`
    - Evaluation reports and metric files.
    - Stores model evaluation outputs, backtesting summaries, and monitoring reports.

---

## 13. Tests Directory

- `tests/`
  - Automated test suite.
  - Contains unit, integration, API, orchestration, and fixture tests.

  - `unit/`
    - Tests for individual functions and classes.
    - Focuses on small isolated pieces of logic.

  - `integration/`
    - Tests for pipelines and module interactions.
    - Verifies that multiple components work together correctly.

  - `api/`
    - Tests for FastAPI endpoints.
    - Validates request handling, response schemas, and API behavior.

  - `orchestration/`
    - Tests for Airflow DAG and task logic.
    - Ensures DAG imports, dependencies, and task wrappers behave correctly.

  - `fixtures/`
    - Small sample test datasets and mock objects.
    - Provides reusable test inputs.

---

## 14. Configuration Directory

- `configs/`
  - YAML configuration files.
  - Stores project settings separate from source code.

  - `data.yaml`
    - Dataset paths and data settings.
    - Defines where raw, interim, processed, and feature datasets are stored.

  - `sampling.yaml`
    - Sampling size, strategy, seed, and output paths.
    - Controls how small representative datasets are created.

  - `forecasting.yaml`
    - Forecasting model and training settings.
    - Stores forecast horizon, model parameters, validation settings, and metric preferences.

  - `behavior_profiling.yaml`
    - Segmentation or profiling model and rule settings.
    - Defines rules, thresholds, clustering parameters, or profiling logic.

  - `genai.yaml`
    - GenAI prompt, provider, and model settings.
    - Stores model provider configuration, prompt behavior, and generation parameters.

  - `airflow.yaml`
    - DAG schedules, retries, and orchestration settings.
    - Controls runtime behavior for Airflow workflows.

  - `aws.yaml`
    - EC2, S3, region, and deployment settings.
    - Stores AWS-specific runtime configuration.

  - `app.yaml`
    - API and UI application settings.
    - Defines service ports, feature flags, UI options, and app-level configuration.

---

## 15. Scripts Directory

- `scripts/`
  - Utility scripts for local, development, and operations workflows.
  - Provides simple command-line entry points for common tasks.

  - `sample_dataset.py`
    - CLI script to create small datasets.
    - Uses sampling logic from the source package.

  - `run_data_preparation.py`
    - Script to run data preparation locally.
    - Useful before introducing full Airflow orchestration.

  - `run_training.py`
    - Script to run training locally.
    - Useful for experimentation, debugging, and manual model training.

  - `run_batch_inference.py`
    - Script to run batch inference locally.
    - Generates predictions outside Airflow for testing or manual operations.

  - `run_monitoring.py`
    - Script to run monitoring locally.
    - Executes quality, drift, and performance checks.

---

## 16. Root-Level Project Files

- `pyproject.toml`
  - Python project metadata, dependencies, and build configuration.
  - Defines package settings, dependency groups, formatters, linters, and build system.

- `README.md`
  - Project overview, setup instructions, and usage guide.
  - First document users should read to understand and run the project.

- `.env.example`
  - Example environment variables.
  - Shows required environment variables without exposing secrets.

- `.gitignore`
  - Files and folders Git should ignore.
  - Excludes local data, secrets, caches, virtual environments, logs, and generated artifacts.

---

## 17. High-Level Workflow Coverage

- Data Preparation DAG
  - Data ingestion.
  - Data validation and profiling.
  - Preprocessing.
  - Feature generation.
  - Store versioned dataset and features.

- Training DAG
  - Load prepared data.
  - Train model.
  - Track experiment in MLflow.
  - Evaluate model.
  - Validate model.
  - Register or promote model.

- Batch Inference DAG
  - Load production model.
  - Load input data.
  - Generate predictions.
  - Store outputs.
  - Log inference metrics.

- Monitoring DAG
  - Monitor data quality.
  - Monitor prediction quality.
  - Detect drift.
  - Track model performance.

- Retraining DAG
  - Triggered by schedule, drift, or performance degradation.
  - Re-runs training pipeline.
  - Registers a new candidate model.

---

## 18. Notes

- `docs/` is intentionally kept empty in the initial structure.
- `segmentation/` is lowercase to follow Python package naming conventions.
- `terraform/` is used for infrastructure provisioning.
- `deployment/aws/` is used for operational deployment scripts after infrastructure exists.
- `docker/` supports local development and containerized deployment.
- `dags/` contains Airflow workflow definitions, while `src/.../orchestration/` contains reusable orchestration logic.
