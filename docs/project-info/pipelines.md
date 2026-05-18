# MLOps Pipeline Architecture

## Airflow DAG-Based Implementation

### **1. Data Preparation DAG**
**Purpose**: Prepare and version datasets for ML pipeline
**Schedule**: Daily/Weekly based on data refresh requirements

- Data ingestion
  - Decide sampling technique on original data
  - Load raw M5 data (sales, calendar, prices)
- Data validation/profiling
  - Initial data inspection
  - Data quality checks: Missing values, Invalid values, Unique values, Cardinality, Range issues, data types checking, Duplicate IDs, Target distribution
  - Validate complete M5 schemas
- Preprocessing
  - Light cleaning: Rename messy columns, Strip spaces, Fix data types, Convert dates. Standardize categories, Remove exact duplicates, Handle obvious invalid records
  - Decide cleaning strategy for ML pipeline
- Feature generation
  - Basic demand features (avg_sales, std_sales, zero_sales_ratio, cv_sales)
  - Trend analysis features
  - Seasonality and calendar features
- Store versioned dataset/features
  - Version control processed datasets
  - Store feature metadata and schemas

### **2. Training DAG**
**Purpose**: Train and validate ML models with experiment tracking
**Schedule**: Triggered by new data versions or on-demand

- Load prepared data
  - Retrieve versioned datasets from Data Preparation DAG
  - Train/test split (respecting time series nature)
- EDA & Hypothesis testing
  - Descriptive statistics
  - Univariate, Bivariate, Multivariate analysis
  - Time series specific EDA
  - Hypothesis testing for model assumptions
- Advanced preprocessing
  - Missing value imputation
  - Outlier treatment
  - Advanced feature engineering
  - Categorical encoding
  - Feature transformation
  - Scaling/normalization/standardization (use RobustScaler)
  - Feature selection
  - Handle class imbalance
- Train model
  - Model Selection & Validation: Architecture selection, Cross-validation strategy, Baseline establishment
  - Train demand forecasting models
  - Train behavior profiling models (multi-label classification)
- Track in MLflow
  - Log experiments, parameters, metrics
  - Track model artifacts
  - Version model iterations
- Evaluate & Validate
  - Model performance evaluation
  - Business metric validation
  - A/B testing for model comparison
- Register/promote model
  - Register validated models in MLflow Model Registry
  - Promote to production stage

### **3. Batch Inference DAG**
**Purpose**: Generate predictions using production models
**Schedule**: Daily/hourly based on business requirements

- Load production model
  - Retrieve latest production model from MLflow registry
- Load input data
  - Fetch new store-item combinations for forecasting
  - Apply same preprocessing as training pipeline
- Generate predictions
  - Run demand forecasting
  - Generate demand behavior profiles
  - Create multi-label demand classifications
- Store outputs
  - Save predictions to data store
  - Version prediction outputs
- Log inference metrics
  - Track prediction volumes
  - Monitor inference performance

### **4. Monitoring DAG**
**Purpose**: Monitor model and data quality in production
**Schedule**: Continuous/hourly monitoring

- Monitor data quality
  - Input data validation
  - Schema drift detection
  - Data distribution monitoring
- Monitor prediction quality
  - Prediction accuracy tracking
  - Business KPI monitoring
- Drift detection
  - Feature drift detection
  - Concept drift monitoring
  - Model performance degradation alerts
- Performance tracking
  - Model latency monitoring
  - System resource utilization
  - Error rate tracking

### **5. Retraining DAG**
**Purpose**: Automated model retraining pipeline
**Triggers**: Schedule, drift alerts, performance degradation

- Trigger conditions
  - Scheduled retraining (weekly/monthly)
  - Data drift threshold exceeded
  - Model performance below threshold
- Re-run training pipeline
  - Execute Training DAG with latest data
  - Compare new model with current production
- Register new candidate model
  - Evaluate candidate vs production model
  - Stage for A/B testing if performance improves
  - Automatic promotion based on validation criteria

## Supporting Infrastructure Components

### **Model Interpretation & Explainability**
- Integrated into Training DAG for model validation
- Available through FastAPI endpoints for real-time explanations

### **GenAI Business Insight Generation**
- Integrated into Batch Inference DAG output processing
- Converts ML predictions into actionable business insights
- Static insight generation for inventory planning decisions

### **FastAPI Endpoints**
- Real-time inference API
- Model explanation endpoints
- Health check and monitoring endpoints
- Integration with production model registry

### **Streamlit UI**
- User interface for demand forecasting insights
- Model performance dashboards
- Business insight visualization
- Manual forecast request interface


-------
Note: 
- Though I didnt mention in above, we need to have write required test cases too. 
- things are missing but is out of scope of this project is system design, aws deployment