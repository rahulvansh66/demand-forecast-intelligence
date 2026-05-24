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
```python
# Strategy based on EDA threshold: 5% missing tolerance
- Zero sales vs. missing sales distinction crucial for retail
- Forward-fill approach for calendar features (events, holidays)
- Median imputation for pricing data within item-category groups
- Drop series with >5% missing values in critical periods
```

#### Outlier Detection and Treatment
```python
# Based on 95th percentile threshold from EDA
- IQR method for sales outliers (Q3 + 1.5×IQR)
- Separate treatment for promotional vs. non-promotional periods
- Cap extreme values rather than removal (preserve sales events)
- Log-transform before outlier detection for skewed distributions
```

#### Zero Sales Handling
```python
# Critical for intermittent demand (>30% zeros threshold)
- Preserve zero structure (don't impute zeros)
- Create binary indicators for zero/non-zero periods
- Apply different models for zero probability vs. positive sales magnitude
- Consider Hurdle/Zero-Inflated models for high-zero series
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

#### Automated Model Selection
```python
# AIC/BIC optimization per series
- Grid search over (p,d,q) × (P,D,Q,s) space
- Information criteria minimization
- Out-of-sample validation (time series CV)
- Segment-specific parameter constraints
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

#### Advanced Training Techniques
```python
# Ensemble and stacking strategies
1. Time-based cross-validation with purged gaps
2. Stacked generalization (meta-learner on residuals)
3. Quantile regression for prediction intervals
4. Early stopping with validation WRMSSE monitoring
5. Feature importance analysis for interpretability
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

### Phase 1: Data Infrastructure (Weeks 1-2)
1. **Data Pipeline Setup**
   - Implement lazy loading for large M5 dataset
   - Create preprocessing pipelines for both SARIMA and LightGBM
   - Set up feature engineering framework with temporal constraints

2. **Validation Framework**
   - Implement WRMSSE calculation with hierarchical weights
   - Create time series cross-validation utilities
   - Build segment classification pipeline

### Phase 2: Baseline Models (Weeks 3-4)
1. **SARIMA Implementation**
   - Auto-ARIMA parameter selection per segment
   - Seasonal decomposition and stationarity testing
   - Hierarchical forecasting reconciliation

2. **LightGBM Implementation**
   - Feature engineering pipeline with 200+ features
   - Segment-specific hyperparameter tuning
   - Multi-output forecasting framework

### Phase 3: Advanced Modeling (Weeks 5-6)
1. **Ensemble Methods**
   - SARIMA + LightGBM combination strategies
   - Quantile regression for uncertainty estimation
   - Model selection automation by series characteristics

2. **Optimization**
   - WRMSSE-focused hyperparameter tuning
   - Feature selection and importance analysis
   - Performance profiling and scalability testing

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