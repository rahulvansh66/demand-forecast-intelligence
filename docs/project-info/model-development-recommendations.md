# Model Development Recommendations for M5 Demand Forecasting

## Executive Summary

Based on the comprehensive EDA analysis of the M5 Walmart dataset, this document provides targeted preprocessing and model development strategies for SARIMA and LightGBM approaches. The analysis reveals a dual-objective problem requiring both 28-day demand forecasting and behavioral segmentation across 30,490 item-store combinations.

## EDA Key Findings Analysis

### Dataset Characteristics
- **Scale**: 30,490 item-store combinations × 1,969 days = ~60M data points
- **Temporal Structure**: Daily sales from 2011-01-29 to 2016-06-19
- **Hierarchical Nature**: 3 states × 10 stores × 3 categories × 7 departments × 3,049 items
- **Target Variable**: Unit sales (highly variable, intermittent, zero-inflated)
- **Evaluation Metric**: WRMSSE (Weighted Root Mean Squared Scaled Error)

### Critical Business Constraints
- **Temporal Cutoff**: d_1913 (2016-04-24) prevents data leakage
- **Forecast Horizon**: 28 days ahead prediction required
- **Performance Target**: WRMSSE < 0.5 for competitive advantage
- **Business Impact**: 10-15% inventory cost reduction potential

### Demand Pattern Segmentation
Five distinct behavioral segments identified:
1. **Smooth Regular**: Consistent patterns with predictable trends
2. **Intermittent Low**: Sporadic demand with long zero periods
3. **Intermittent High**: Irregular high-volume spikes
4. **Lumpy Seasonal**: Seasonal patterns with irregular timing
5. **Erratic Volatile**: Highly unpredictable demand patterns

---

## Preprocessing Strategy

### 1. Data Quality and Preparation

#### Missing Value Handling

**EDA Finding**: Analysis revealed 5% missing data threshold as critical business constraint for maintaining forecast quality.

**Why This Approach**:
```python
# Zero sales vs. missing sales distinction
- EDA Reasoning: BusinessContextService identified that retail context requires 
  distinguishing true zeros (no sales) from missing observations (system errors)
- Business Impact: Misclassifying zeros as missing would inflate demand estimates
- Method Choice: Semantic differentiation prevents demand overestimation

# Forward-fill for calendar features  
- EDA Reasoning: Calendar data (holidays, events) are deterministic and known in advance
- Why Needed: Missing holiday indicators would break seasonal pattern recognition
- Method Choice: Forward-fill preserves temporal sequence integrity vs. mode imputation

# Median imputation for pricing within item-category groups
- EDA Reasoning: M5 config shows price_change_threshold of 10% indicates pricing stability
- Why Needed: Missing prices would break price elasticity feature engineering
- Method Choice: Median more robust than mean for skewed price distributions within categories

# Drop series with >5% missing in critical periods
- EDA Reasoning: Analysis shows min_obs_per_series = 100 for reliable statistical inference
- Why Needed: Insufficient data compromises time series model parameter estimation
- Method Choice: Dropping vs. imputation prevents model degradation on unreliable series
```

#### Outlier Detection and Treatment

**EDA Finding**: Analysis set outlier_threshold at 95th percentile, revealing promotional spikes vs. anomalies.

**Why This Approach**:
```python
# IQR method for sales outliers (Q3 + 1.5×IQR)
- EDA Reasoning: M5 retail context shows legitimate high-volume sales during events
- Why Needed: Raw outlier removal would eliminate crucial promotional patterns
- Method Choice: IQR less sensitive than z-score for skewed retail distributions

# Separate treatment for promotional vs. non-promotional periods  
- EDA Reasoning: EDA config enables promotion_detection = True for event-driven spikes
- Why Needed: Promotional periods have different statistical distributions than regular sales
- Method Choice: Context-aware outlier detection prevents removing legitimate business events

# Cap extreme values rather than removal
- EDA Reasoning: Business context analysis showed sales events are informative, not noise
- Why Needed: Complete removal loses information about demand surge capacity
- Method Choice: Capping preserves directional information while limiting extreme influence

# Log-transform before outlier detection  
- EDA Reasoning: Retail sales exhibit heavy-tailed distributions (high skewness)
- Why Needed: Standard outlier detection fails on non-Gaussian distributions
- Method Choice: Log-transform normalizes distribution shape for robust detection
```

#### Zero Sales Handling

**EDA Finding**: Analysis revealed intermittent_threshold of 30% zeros, with sparse_series_threshold at 50% identifying critical patterns.

**Why This Approach**:
```python
# Preserve zero structure (don't impute zeros)
- EDA Reasoning: BusinessContextService identified 5 demand segments including intermittent patterns
- Why Needed: Zero sales contain information about demand intermittency patterns
- Method Choice: Preserving zeros maintains true demand signal vs. smoothing artifacts

# Create binary indicators for zero/non-zero periods
- EDA Reasoning: Intermittent demand requires separate modeling of occurrence vs. magnitude
- Why Needed: Traditional forecasting assumes continuous demand, fails on sparse series
- Method Choice: Binary indicators enable Hurdle models for two-stage prediction

# Different models for zero probability vs. positive sales magnitude
- EDA Reasoning: EDA segments show intermittent_low vs. intermittent_high have different patterns
- Why Needed: Probability of sale vs. amount when sale occurs are different phenomena
- Method Choice: Two-stage modeling captures both demand occurrence and intensity

# Hurdle/Zero-Inflated models for high-zero series  
- EDA Reasoning: Analysis shows >30% zero threshold identifies series needing special treatment
- Why Needed: Standard models underperform on zero-inflated distributions
- Method Choice: Hurdle models specifically designed for excess-zero retail scenarios
```

### 2. Feature Engineering Pipeline

#### Temporal Features
```python
# Calendar-based features (known in advance)
- Day/week/month/quarter indicators
- Holiday proximity (days to/from major holidays)
- SNAP benefit days by state (CA, TX, WI)
- Seasonal indicators (back-to-school, Christmas, etc.)
- Working days vs. weekends/holidays
```

#### Lag Features (Respecting Temporal Cutoff)
```python
# Sales lags (minimum 1-day lag to prevent leakage)
- sales_lag_1, sales_lag_7, sales_lag_28 (weekly/monthly patterns)
- sales_lag_365 (yearly seasonality)
- Rolling windows: 7-day, 14-day, 28-day averages
- Exponentially weighted moving averages (α=0.1, 0.3, 0.5)
```

#### Price and Promotion Features
```python
# Pricing features (with appropriate lags)
- price_lag_1_to_7 (recent pricing history)
- price_change_indicators (>10% change threshold)
- promotional_period_binary
- price_relative_to_category_average
- Price elasticity proxies (price × demand interactions)
```

#### Hierarchical Aggregation Features
```python
# Cross-series information
- category_total_sales (lag-adjusted)
- store_total_sales (lag-adjusted)
- state_performance_indicators
- department_seasonal_trends
- Item popularity ranking within category
```

---

## SARIMA Model Development Strategy

### 1. Model Architecture

#### Individual Series Modeling
```python
# Segment-specific SARIMA configurations
SARIMA_CONFIGS = {
    'smooth_regular': {
        'order': (2, 1, 2),           # ARIMA(p,d,q)
        'seasonal_order': (1, 1, 1, 7), # Seasonal(P,D,Q,s) - weekly
        'trend': 'c'                   # Constant trend
    },
    'intermittent_low': {
        'order': (1, 0, 1),           # Lower complexity for sparse data
        'seasonal_order': (0, 0, 0, 0), # No seasonal for irregular patterns
        'trend': None
    },
    'lumpy_seasonal': {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 365), # Annual seasonality
        'trend': 'ct'                   # Linear trend
    }
}
```

#### Hierarchical Forecasting Integration
```python
# Bottom-up and top-down reconciliation
- Base forecasts at item-store level
- Hierarchical constraints (store totals, category totals)
- MinT (Minimum Trace) reconciliation for coherent forecasts
- Cross-validation with temporal hierarchy preserved
```

### 2. Preprocessing for SARIMA

#### Stationarity Transformation
```python
# Sequential transformation pipeline
1. Log transformation for positive sales (handle zeros separately)
2. Seasonal differencing (d=1, D=1) based on ACF/PACF analysis
3. Box-Cox transformation parameter optimization per segment
4. Unit root tests (ADF, KPSS) for stationarity verification
```

#### Time Series Decomposition
```python
# STL (Seasonal and Trend decomposition using Loess)
- Separate trend, seasonal, and residual components
- Model residuals with ARIMA
- Reconstruct forecasts with seasonal + trend projections
- Handle multiple seasonalities (weekly + annual)
```

#### Exogenous Variables Integration (SARIMAX)
```python
# External regressors for SARIMAX models
- Holiday indicators (known future values)
- Promotional calendar (planned events)
- Economic indicators (fuel prices, unemployment)
- Weather proxies (regional temperature, precipitation)
```

### 3. Model Selection and Validation

#### Automated Model Selection with Optuna

**EDA-Informed Parameter Space**:
```python
# Optuna-based Bayesian optimization for SARIMA parameters
import optuna
from optuna.integration import MLflowCallback

def objective_sarima(trial, series_data, segment_type):
    """
    EDA-informed parameter space based on segment characteristics
    """
    # EDA Reasoning: Different segments need different complexity levels
    if segment_type == 'smooth_regular':
        p = trial.suggest_int('p', 1, 3)  # Higher complexity for stable patterns
        d = trial.suggest_int('d', 0, 2)  
        q = trial.suggest_int('q', 1, 3)
        seasonal_period = 7  # EDA shows strong weekly patterns
    elif segment_type == 'intermittent_low':
        p = trial.suggest_int('p', 0, 1)  # Lower complexity for sparse data
        d = trial.suggest_int('d', 0, 1)
        q = trial.suggest_int('q', 0, 1)  
        seasonal_period = 0   # EDA shows weak seasonality in intermittent series
    elif segment_type == 'lumpy_seasonal':
        p = trial.suggest_int('p', 1, 2)
        d = trial.suggest_int('d', 1, 2)  # EDA shows trend components
        q = trial.suggest_int('q', 1, 2)
        seasonal_period = 365  # EDA identifies annual patterns
    
    # MLflow tracking integration
    with mlflow.start_run():
        model = SARIMAX(series_data, order=(p,d,q), 
                       seasonal_order=(1,1,1,seasonal_period) if seasonal_period > 0 else None)
        fitted = model.fit(disp=False)
        
        # EDA-aligned validation using WRMSSE-approximation
        cv_scores = time_series_cv_wrmsse(fitted, series_data)
        
        # Log parameters and metrics to MLflow
        mlflow.log_params({
            'p': p, 'd': d, 'q': q, 
            'seasonal_period': seasonal_period,
            'segment_type': segment_type
        })
        mlflow.log_metrics({
            'cv_wrmsse': cv_scores.mean(),
            'aic': fitted.aic,
            'bic': fitted.bic
        })
        
        return cv_scores.mean()

# Run optimization with MLflow callback
mlflow_callback = MLflowCallback(
    tracking_uri="http://localhost:5000",
    metric_name="cv_wrmsse"
)

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
)

study.optimize(objective_sarima, n_trials=100, callbacks=[mlflow_callback])
```

#### Cross-Validation Strategy
```python
# Time-aware validation preserving temporal structure
- Walk-forward validation with 28-day horizons
- Minimum 365 days training window
- 4-fold temporal CV with growing window
- WRMSSE calculation at each fold
```

---

## LightGBM Model Development Strategy

### 1. Model Architecture

#### Gradient Boosting Configuration
```python
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 127,           # 2^7 - 1 for balanced trees
    'learning_rate': 0.05,       # Conservative for stability
    'feature_fraction': 0.8,     # Feature sampling
    'bagging_fraction': 0.8,     # Data sampling
    'bagging_freq': 5,
    'min_data_in_leaf': 100,     # Prevent overfitting sparse series
    'lambda_l1': 0.1,           # L1 regularization
    'lambda_l2': 0.1,           # L2 regularization
    'verbose': -1
}
```

#### Multi-Output Strategy
```python
# Hierarchical forecasting with LightGBM
- Direct multi-step forecasting (28 outputs)
- Recursive forecasting (1-step with lag update)
- Ensemble of both approaches
- Quantile regression for uncertainty estimation
```

### 2. Feature Engineering for LightGBM

#### Advanced Lag Features
```python
# Extensive lag feature set
LAG_FEATURES = {
    'sales_lags': [1, 2, 3, 7, 14, 21, 28, 35, 42, 49, 56],  # Weekly patterns
    'rolling_stats': {
        'windows': [7, 14, 28, 56, 84],
        'stats': ['mean', 'std', 'min', 'max', 'median', 'skew']
    },
    'exponential_smoothing': [0.1, 0.3, 0.5, 0.7, 0.9],  # Different alpha values
    'seasonal_lags': [365, 364, 366]  # Annual patterns ±1 day
}
```

#### Categorical Feature Engineering
```python
# High-cardinality categorical handling
- Target encoding for item_id (3,049 categories)
- Frequency encoding for store-item combinations
- Binary encoding for hierarchical categories
- Interaction features (store × category, state × department)
- Leave-one-out encoding to prevent overfitting
```

#### Time-Based Features
```python
# Comprehensive temporal features
TIME_FEATURES = {
    'calendar': ['dayofweek', 'dayofmonth', 'month', 'quarter'],
    'cyclical': ['dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin', 'dayofyear_cos'],
    'holiday_proximity': ['days_to_christmas', 'days_to_thanksgiving', 'days_to_easter'],
    'event_indicators': ['cultural_events', 'sporting_events', 'national_holidays'],
    'business_calendar': ['is_weekend', 'is_month_end', 'is_quarter_end']
}
```

#### Cross-Series Features
```python
# Leveraging hierarchical structure
CROSS_SERIES_FEATURES = {
    'aggregation_lags': {
        'store_level': ['total_store_sales_lag_1', 'store_category_sales_lag_7'],
        'category_level': ['category_performance_lag_1', 'category_trend_28d'],
        'state_level': ['state_economic_indicators', 'state_seasonal_patterns']
    },
    'relative_performance': [
        'item_share_in_category', 'store_market_share', 'category_growth_rate'
    ]
}
```

### 3. Model Training Strategy

#### Segment-Aware Training
```python
# Different models per demand segment
SEGMENT_MODELS = {
    'smooth_regular': {
        'max_depth': 8,
        'num_leaves': 255,
        'min_data_in_leaf': 50
    },
    'intermittent_low': {
        'max_depth': 4,
        'num_leaves': 15,
        'min_data_in_leaf': 200,  # Higher threshold for sparse data
        'pos_bagging_fraction': 0.9  # Focus on non-zero samples
    },
    'erratic_volatile': {
        'max_depth': 6,
        'num_leaves': 63,
        'lambda_l1': 0.5,  # Higher regularization
        'lambda_l2': 0.5
    }
}
```

#### Hyperparameter Optimization with Optuna & MLflow

**EDA-Informed LightGBM Tuning**:
```python
import optuna
import mlflow
import mlflow.lightgbm
from optuna.integration import MLflowCallback

def objective_lightgbm(trial, X_train, y_train, X_val, y_val, segment_type):
    """
    Segment-specific hyperparameter optimization based on EDA insights
    """
    # EDA Reasoning: Different segments need different model configurations
    if segment_type == 'smooth_regular':
        # Higher complexity for stable patterns with clear trends
        max_depth = trial.suggest_int('max_depth', 6, 12)
        num_leaves = trial.suggest_int('num_leaves', 100, 300)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 20, 100)
    elif segment_type == 'intermittent_low':
        # Lower complexity for sparse data to prevent overfitting
        max_depth = trial.suggest_int('max_depth', 3, 6)  
        num_leaves = trial.suggest_int('num_leaves', 10, 50)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 100, 500)
    elif segment_type == 'erratic_volatile':
        # Higher regularization for noisy patterns
        max_depth = trial.suggest_int('max_depth', 4, 8)
        num_leaves = trial.suggest_int('num_leaves', 30, 150)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 50, 200)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
        'verbosity': -1
    }
    
    # EDA-specific adjustments for intermittent demand
    if segment_type in ['intermittent_low', 'intermittent_high']:
        # Higher regularization for sparse series
        params['lambda_l1'] = trial.suggest_float('lambda_l1', 0.1, 2.0)
        params['lambda_l2'] = trial.suggest_float('lambda_l2', 0.1, 2.0)
        # Focus sampling on non-zero values  
        params['pos_bagging_fraction'] = trial.suggest_float('pos_bagging_fraction', 0.7, 1.0)
    
    # MLflow experiment tracking
    with mlflow.start_run():
        # Train model with early stopping
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Predict and calculate WRMSSE-approximation
        y_pred = model.predict(X_val)
        wrmsse_score = calculate_wrmsse_approximation(y_val, y_pred, segment_type)
        
        # Log comprehensive metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics({
            'wrmsse_score': wrmsse_score,
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'mae': mean_absolute_error(y_val, y_pred),
            'num_boost_round': model.num_trees(),
            'segment_type_encoded': hash(segment_type) % 1000
        })
        
        # Log feature importance
        importance_dict = dict(zip(model.feature_name(), model.feature_importance()))
        mlflow.log_dict(importance_dict, "feature_importance.json")
        
        # Log model artifact
        mlflow.lightgbm.log_model(model, f"model_{segment_type}")
        
        return wrmsse_score

# Multi-objective optimization for WRMSSE and inference speed
def multi_objective_lightgbm(trial, X_train, y_train, X_val, y_val, segment_type):
    """Multi-objective optimization balancing accuracy and speed"""
    
    # Get single objective score
    wrmsse_score = objective_lightgbm(trial, X_train, y_train, X_val, y_val, segment_type)
    
    # Measure inference time (business requirement: <100ms per series)
    start_time = time.time()
    _ = trial.study.best_trials[0].user_attrs.get('model').predict(X_val[:1000])
    inference_time = (time.time() - start_time) / 1000  # per sample
    
    # Return both objectives (Optuna will handle Pareto optimization)
    return wrmsse_score, inference_time

# Set up MLflow experiment
mlflow.set_experiment("M5_LightGBM_Segment_Optimization")

# Configure Optuna study with MLflow integration
mlflow_callback = MLflowCallback(
    tracking_uri="http://localhost:5000",
    metric_name="wrmsse_score"
)

# Separate study per segment (EDA-driven approach)
for segment in ['smooth_regular', 'intermittent_low', 'intermittent_high', 'lumpy_seasonal', 'erratic_volatile']:
    
    study = optuna.create_study(
        study_name=f"lightgbm_{segment}",
        direction='minimize',
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True  # Capture parameter interactions
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=50,  # Minimum boosting rounds
            max_resource=1000,  # Maximum boosting rounds  
            reduction_factor=3
        )
    )
    
    # Add segment-specific constraints based on EDA findings
    if segment in ['intermittent_low', 'intermittent_high']:
        # Constrain complexity for sparse data
        study.enqueue_trial({
            'max_depth': 4, 'num_leaves': 15, 'min_data_in_leaf': 200,
            'lambda_l1': 0.5, 'lambda_l2': 0.5
        })
    
    study.optimize(
        lambda trial: objective_lightgbm(trial, X_train_segment, y_train_segment, 
                                       X_val_segment, y_val_segment, segment),
        n_trials=200,
        callbacks=[mlflow_callback],
        timeout=3600  # 1 hour per segment
    )
    
    # Log best parameters for segment
    with mlflow.start_run(run_name=f"best_{segment}"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_wrmsse", study.best_value)
        mlflow.log_text(str(study.best_trial), "optimization_summary.txt")
```

#### Advanced Training Techniques
```python
# MLflow-tracked ensemble and stacking strategies
1. Time-based cross-validation with purged gaps (logged to MLflow)
2. Stacked generalization with meta-learner performance tracking
3. Quantile regression with uncertainty calibration metrics
4. Early stopping with validation WRMSSE monitoring via MLflow callbacks
5. Feature importance analysis with MLflow artifact logging
6. Hyperparameter importance analysis using Optuna's built-in tools
7. Multi-objective optimization (accuracy vs. inference speed)
```

---

## Hybrid Modeling Approach

### 1. Model Combination Strategy

#### SARIMA + LightGBM Ensemble
```python
ENSEMBLE_WEIGHTS = {
    'smooth_regular': {'sarima': 0.7, 'lgbm': 0.3},
    'intermittent_low': {'sarima': 0.3, 'lgbm': 0.7},  # LightGBM better for sparse
    'lumpy_seasonal': {'sarima': 0.8, 'lgbm': 0.2},    # SARIMA captures seasonality
    'erratic_volatile': {'sarima': 0.2, 'lgbm': 0.8}   # LightGBM handles complexity
}
```

#### Residual Modeling
```python
# Two-stage approach
Stage 1: SARIMA for trend/seasonality capture
Stage 2: LightGBM on SARIMA residuals for complex patterns
Final Forecast: SARIMA + LightGBM_residuals
```

### 2. Model Selection Pipeline
```python
# Automated segment-model matching
def select_model_by_series_characteristics(series_stats):
    """
    Automatically select best model based on series properties
    - CV < 0.5 and Intermittency < 0.3 → SARIMA
    - CV > 1.0 or Intermittency > 0.5 → LightGBM
    - Seasonal strength > 0.6 → SARIMA-dominant ensemble
    - Trend strength > 0.7 → LightGBM-dominant ensemble
    """
    return model_assignment
```

---

## MLOps and Experiment Management

### 1. MLflow Integration Strategy

#### Experiment Organization
```python
# Hierarchical experiment structure based on EDA segments
EXPERIMENT_HIERARCHY = {
    'M5_Demand_Forecasting': {
        'SARIMA_Models': [
            'SARIMA_Smooth_Regular',
            'SARIMA_Intermittent_Low', 
            'SARIMA_Intermittent_High',
            'SARIMA_Lumpy_Seasonal',
            'SARIMA_Erratic_Volatile'
        ],
        'LightGBM_Models': [
            'LightGBM_Smooth_Regular',
            'LightGBM_Intermittent_Low',
            'LightGBM_Intermittent_High', 
            'LightGBM_Lumpy_Seasonal',
            'LightGBM_Erratic_Volatile'
        ],
        'Ensemble_Models': [
            'Hybrid_SARIMA_LightGBM',
            'Stacked_Residual_Models',
            'Multi_Objective_Ensemble'
        ]
    }
}

# Custom MLflow metrics aligned with EDA findings
def log_eda_aligned_metrics(y_true, y_pred, segment_type, model_type):
    """Log business-relevant metrics derived from EDA analysis"""
    
    # Primary metric: WRMSSE approximation
    wrmsse_score = calculate_wrmsse_approximation(y_true, y_pred, segment_type)
    mlflow.log_metric("wrmsse_score", wrmsse_score)
    
    # EDA-specific metrics per segment
    if segment_type in ['intermittent_low', 'intermittent_high']:
        # Zero-inflation accuracy for intermittent demand
        zero_accuracy = calculate_zero_prediction_accuracy(y_true, y_pred)
        mlflow.log_metric("zero_prediction_accuracy", zero_accuracy)
        
        # Intermittency preservation
        true_intermittency = (y_true == 0).mean()
        pred_intermittency = (y_pred <= 0.5).mean()  # Threshold for zero classification
        mlflow.log_metric("intermittency_preservation", 1 - abs(true_intermittency - pred_intermittency))
    
    if segment_type == 'lumpy_seasonal':
        # Seasonal pattern preservation
        seasonal_correlation = calculate_seasonal_correlation(y_true, y_pred, period=365)
        mlflow.log_metric("seasonal_correlation", seasonal_correlation)
    
    # Business impact metrics
    inventory_cost_impact = calculate_inventory_impact(y_true, y_pred)
    stockout_reduction = calculate_stockout_reduction(y_true, y_pred)
    
    mlflow.log_metrics({
        "inventory_cost_impact": inventory_cost_impact,
        "stockout_reduction": stockout_reduction,
        "directional_accuracy": calculate_directional_accuracy(y_true, y_pred)
    })
```

#### Model Registry and Versioning
```python
# EDA-informed model registration strategy
def register_best_model_per_segment():
    """Register best performing model per segment to MLflow Model Registry"""
    
    for segment in ['smooth_regular', 'intermittent_low', 'intermittent_high', 'lumpy_seasonal', 'erratic_volatile']:
        
        # Get best run per segment based on WRMSSE
        experiment = mlflow.get_experiment_by_name(f"M5_LightGBM_{segment}")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        best_run = runs.loc[runs['metrics.wrmsse_score'].idxmin()]
        
        # Register model with segment-specific alias
        model_uri = f"runs:/{best_run.run_id}/model"
        mlflow.register_model(
            model_uri=model_uri,
            name=f"M5_Forecast_{segment.title()}",
            tags={
                "segment_type": segment,
                "eda_characteristics": get_segment_characteristics(segment),
                "business_priority": get_segment_priority(segment),
                "deployment_strategy": "batch" if segment in ['intermittent_low'] else "realtime"
            }
        )
```

### 2. Optuna Integration and Study Management

#### Advanced Study Configuration
```python
# EDA-informed study design with knowledge sharing
def create_multi_segment_study():
    """Create Optuna studies with parameter sharing between similar segments"""
    
    # Define parameter relationships based on EDA insights  
    PARAMETER_SHARING = {
        'regular_patterns': ['smooth_regular', 'lumpy_seasonal'],  # Share trend parameters
        'sparse_patterns': ['intermittent_low', 'intermittent_high'],  # Share sparsity handling
        'volatile_patterns': ['erratic_volatile']  # Isolated optimization
    }
    
    storage = optuna.storages.RDBStorage(
        url="postgresql://optuna:optuna@localhost/optuna_m5",
        heartbeat_interval=60,
        grace_period=120
    )
    
    studies = {}
    for group_name, segments in PARAMETER_SHARING.items():
        
        # Shared study for similar segments
        study = optuna.create_study(
            study_name=f"m5_shared_{group_name}",
            storage=storage,
            direction='minimize',
            sampler=optuna.samplers.TPESampler(
                multivariate=True,
                group=True,  # Enable parameter grouping
                n_startup_trials=20,  # More startup trials for shared learning
                n_ei_candidates=48   # Higher exploration for complex space
            ),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=100,
                max_resource=2000,
                reduction_factor=3
            )
        )
        studies[group_name] = study
    
    return studies

# Optuna callbacks for MLflow integration
class MLflowOptunaCallback:
    def __init__(self, experiment_name, segment_type):
        self.experiment_name = experiment_name  
        self.segment_type = segment_type
        mlflow.set_experiment(experiment_name)
    
    def __call__(self, study, trial):
        with mlflow.start_run(run_name=f"trial_{trial.number}_{self.segment_type}"):
            # Log trial parameters
            mlflow.log_params(trial.params)
            
            # Log trial result
            if trial.value is not None:
                mlflow.log_metric("objective_value", trial.value)
                mlflow.log_metric("trial_number", trial.number)
            
            # Log study statistics
            mlflow.log_metrics({
                "study_trials_complete": len(study.trials),
                "study_best_value": study.best_value if study.best_trial else float('inf'),
                "pruned_trials": len([t for t in study.trials if t.state == optuna.TrialState.PRUNED])
            })
            
            # Log hyperparameter importance (every 50 trials)
            if trial.number % 50 == 0 and len(study.trials) >= 10:
                importance = optuna.importance.get_param_importances(study)
                mlflow.log_dict(importance, "hyperparameter_importance.json")
```

#### Automated Hyperparameter Analysis
```python
# EDA-driven parameter importance analysis
def analyze_hyperparameter_importance():
    """Analyze parameter importance across segments using Optuna's built-in tools"""
    
    results = {}
    for segment in ['smooth_regular', 'intermittent_low', 'intermittent_high', 'lumpy_seasonal', 'erratic_volatile']:
        
        study = optuna.load_study(
            study_name=f"lightgbm_{segment}",
            storage="postgresql://optuna:optuna@localhost/optuna_m5"
        )
        
        # Parameter importance analysis
        param_importance = optuna.importance.get_param_importances(study)
        
        # Hyperparameter effect analysis
        contour_importance = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
        )
        
        results[segment] = {
            'param_importance': param_importance,
            'contour_importance': contour_importance,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
        
        # Log to MLflow for visualization
        with mlflow.start_run(run_name=f"importance_analysis_{segment}"):
            mlflow.log_dict(param_importance, "param_importance.json")
            mlflow.log_dict(contour_importance, "contour_importance.json") 
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_wrmsse", study.best_value)
            
            # Create and log visualization
            fig = optuna.visualization.plot_param_importances(study)
            mlflow.log_figure(fig, f"param_importance_{segment}.html")
    
    return results
```

---

## Evaluation and Validation Framework

### 1. WRMSSE Optimization

#### Custom Loss Function
```python
# WRMSSE-aligned training objective
def wrmsse_objective(y_true, y_pred):
    """
    Custom LightGBM objective function aligned with WRMSSE metric
    Incorporates hierarchical weights and scaling factors
    """
    weights = get_hierarchical_weights()
    scale_factors = calculate_naive_forecast_errors()
    return weighted_scaled_error(y_true, y_pred, weights, scale_factors)
```

#### Hierarchical Validation
```python
# Multi-level evaluation
EVALUATION_LEVELS = {
    'item_level': 30490,      # Individual item-store combinations
    'store_level': 10,        # Store aggregations
    'category_level': 3,      # Category aggregations
    'state_level': 3,         # State aggregations
    'total_level': 1          # Overall aggregation
}
```

### 2. Cross-Validation Strategy

#### Time Series Cross-Validation
```python
# Temporal validation preserving data leakage constraints
CV_STRATEGY = {
    'n_splits': 4,
    'train_size': 365,        # Minimum 1 year training
    'test_size': 28,          # Forecast horizon
    'gap': 0,                 # No gap between train/test
    'purged_buffer': 1        # 1-day purge to prevent leakage
}
```

#### Segment-Stratified Validation
```python
# Ensure each segment represented in CV folds
- Stratify by demand segment distribution
- Balance intermittent vs. regular series in each fold
- Maintain seasonal pattern representation across folds
```

---

## Implementation Roadmap

### Phase 1: Data Infrastructure & MLOps Setup (Weeks 1-2)
1. **Data Pipeline Setup**
   - Implement lazy loading for large M5 dataset
   - Create EDA-informed preprocessing pipelines for both SARIMA and LightGBM
   - Set up feature engineering framework with temporal constraints

2. **MLOps Infrastructure**
   - Deploy MLflow tracking server (http://localhost:5000)
   - Configure Optuna database backend (PostgreSQL/MySQL for persistence)
   - Set up experiment tracking with hierarchical organization by segment
   - Implement WRMSSE calculation with hierarchical weights as custom MLflow metric

3. **Validation Framework**
   - Create time series cross-validation utilities with MLflow integration
   - Build segment classification pipeline with performance logging
   - Implement EDA-driven data quality monitoring

### Phase 2: Baseline Models with Automated Tuning (Weeks 3-4)
1. **SARIMA Implementation with Optuna**
   - Segment-specific parameter space definition based on EDA insights
   - Bayesian optimization for (p,d,q) × (P,D,Q,s) parameter selection
   - MLflow experiment tracking for model comparison across segments
   - Seasonal decomposition and stationarity testing with automated logging

2. **LightGBM Implementation with Hyperparameter Optimization**
   - EDA-informed feature engineering pipeline (200+ features)
   - Multi-objective Optuna optimization (WRMSSE vs. inference speed)
   - Segment-specific hyperparameter tuning with MLflow artifact storage
   - Multi-output forecasting framework with performance benchmarking

### Phase 3: Advanced Modeling & Optimization (Weeks 5-6)
1. **Ensemble Methods with Experiment Tracking**
   - SARIMA + LightGBM combination with MLflow model registry
   - Quantile regression with uncertainty calibration metrics
   - Automated model selection based on EDA-derived series characteristics
   - Cross-validation ensemble weights optimization via Optuna

2. **Advanced Optimization & Analysis**
   - Multi-study Optuna optimization across segments with shared knowledge
   - Feature importance analysis with MLflow artifact storage and visualization  
   - Hyperparameter sensitivity analysis using Optuna's importance evaluation
   - Performance profiling with MLflow metrics (memory, CPU, inference time)
   - WRMSSE-focused loss function integration in LightGBM training

### Phase 4: Production Readiness (Weeks 7-8)
1. **Model Deployment**
   - Batch prediction pipeline for 30,490 series
   - Real-time inference API for single series
   - Model monitoring and drift detection

2. **Business Integration**
   - Inventory planning dashboard integration
   - A/B testing framework for model comparison
   - Business KPI tracking (cost reduction, service levels)

---

## Success Metrics and KPIs

### Technical Metrics
- **Primary**: WRMSSE < 0.5 (competitive threshold)
- **Secondary**: Individual segment RMSE, MAE, MAPE
- **Operational**: Inference time < 100ms per series
- **Scalability**: Handle 30,490 series in <4 hours batch processing

### Business Metrics
- **Cost Reduction**: 10-15% inventory cost optimization
- **Service Level**: Reduced stockouts (<5% improvement)
- **Forecast Accuracy**: 85%+ directional accuracy (up/down/flat)
- **Model Adoption**: >80% forecasts used in inventory decisions

---

## Risk Mitigation

### Data Quality Risks
- **Mitigation**: Comprehensive data validation pipeline
- **Monitoring**: Automated alerts for missing data, distribution shifts
- **Fallback**: Seasonal naive forecasts for corrupted series

### Model Performance Risks
- **Mitigation**: Ensemble approaches with multiple model types
- **Monitoring**: WRMSSE tracking with control charts
- **Fallback**: Segment-specific model switching based on performance

### Computational Risks
- **Mitigation**: Distributed computing for large-scale training
- **Monitoring**: Memory usage and processing time tracking
- **Fallback**: Simplified models for resource-constrained scenarios

---

## Conclusion

The EDA analysis reveals M5 as a complex, hierarchical demand forecasting challenge requiring sophisticated preprocessing and model development strategies. The recommended hybrid SARIMA + LightGBM approach leverages the strengths of both statistical time series methods and modern machine learning techniques.

Key success factors:
1. **Segment-aware modeling** tailored to different demand patterns
2. **Rigorous temporal validation** preventing data leakage
3. **Hierarchical forecasting** ensuring coherent predictions across levels
4. **WRMSSE optimization** aligning models with business evaluation metrics

The implementation roadmap provides a structured 8-week path to production deployment, with clear technical and business success metrics throughout the development cycle.