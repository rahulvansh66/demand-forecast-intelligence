Data Collection 
- Decide sampling technique on original data 

Basics of data scanity 
- load data
- Initial data inspection
- Data quality checks: Missing values, Invalid values, Unique values, Cardinality, Range issues, data types checking, Duplicate IDs, Target distribution
- Light cleaning : Rename messy columns, Strip spaces, Fix data types, Convert dates. Standardize categories, Remove exact duplicates, Handle obvious invalid records

EDA

- Descriptive statistics
- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- EDA related to time series 

Hypothesis testing (after eda before Preprocessing pipeline)

setup Airflow

Preprocessing pipeline

- Decide cleaning strategy
- Train/test split
- Missing value imputation
- Outlier treatment
- Feature engineering
- Categorical encoding
- Feature transformation
- Scaling / normalization / standardization
- Feature selection
- Handle class imbalance

Training pipeline

- Model Selection & Validation : Model architecture selection process, Cross-validation strategy, Baseline model establishment, Model comparison framework
- MLflow
- Train and tune model
- Hypothesis testing for model assumptions
- A/B testing for model comparison

Inference pipeline

Model interpretation and explainability

Drift detection and monitoring

GenAI for Sale insight and explanation

Fast API
- Create endpoints 

Streamlit UI


-------
Note: 
- Though I didnt mention in above, we need to have write required test cases too. 
- things are missing but is out of scope of this project is system design, aws deployment