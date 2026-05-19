Here is a clean, scalable directory structure for a **multi-model ML project** with:

1. **Forecasting model**
2. **Segmentation model**

```text
demand_forecast_intelligence/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
│
├── configs/
│   ├── base.yaml
│   ├── forecasting.yaml
│   ├── segmentation.yaml
│   └── logging.yaml
│
├── data/
│   ├── raw/
│   │   ├── forecasting/
│   │   └── segmentation/
│   │
│   ├── interim/
│   │   ├── forecasting/
│   │   └── segmentation/
│   │
│   ├── processed/
│   │   ├── forecasting/
│   │   └── segmentation/
│   │
│   └── external/
│
├── notebooks/
│   ├── forecasting/
│   │   ├── 01_eda.ipynb
│   │   ├── 02_feature_analysis.ipynb
│   │   └── 03_model_experiments.ipynb
│   │
│   └── segmentation/
│       ├── 01_eda.ipynb
│       ├── 02_cluster_analysis.ipynb
│       └── 03_model_experiments.ipynb
│
├── src/
│   └── demand_forecast_intelligence/
│       │
│       ├── __init__.py
│       │
│       ├── common/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── logging.py
│       │   ├── constants.py
│       │   ├── utils.py
│       │   └── exceptions.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ingestion.py
│       │   ├── validation.py
│       │   ├── cleaning.py
│       │   └── splitting.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── common_features.py
│       │   ├── forecasting_features.py
│       │   └── segmentation_features.py
│       │
│       ├── forecasting/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── preprocessing.py
│       │   ├── model.py
│       │   ├── train.py
│       │   ├── predict.py
│       │   ├── evaluate.py
│       │   └── inference.py
│       │
│       ├── segmentation/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── preprocessing.py
│       │   ├── model.py
│       │   ├── train.py
│       │   ├── predict.py
│       │   ├── evaluate.py
│       │   └── inference.py
│       │
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── forecasting_pipeline.py
│       │   ├── segmentation_pipeline.py
│       │   └── full_pipeline.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── forecasting_metrics.py
│       │   ├── segmentation_metrics.py
│       │   └── reporting.py
│       │
│       └── serving/
│           ├── __init__.py
│           ├── api.py
│           ├── schemas.py
│           └── router.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_forecasting.py
│   ├── train_segmentation.py
│   ├── batch_predict_forecasting.py
│   ├── batch_predict_segmentation.py
│   └── run_full_pipeline.py
│
├── models/
│   ├── forecasting/
│   │   ├── baseline/
│   │   ├── experiments/
│   │   └── production/
│   │
│   └── segmentation/
│       ├── baseline/
│       ├── experiments/
│       └── production/
│
├── artifacts/
│   ├── forecasting/
│   │   ├── metrics/
│   │   ├── plots/
│   │   ├── predictions/
│   │   └── reports/
│   │
│   └── segmentation/
│       ├── metrics/
│       ├── plots/
│       ├── predictions/
│       └── reports/
│
├── tests/
│   ├── unit/
│   │   ├── test_common.py
│   │   ├── test_forecasting.py
│   │   └── test_segmentation.py
│   │
│   ├── integration/
│   │   ├── test_forecasting_pipeline.py
│   │   └── test_segmentation_pipeline.py
│   │
│   └── fixtures/
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── api/
│   ├── batch/
│   └── k8s/
│
├── monitoring/
│   ├── data_drift/
│   ├── model_drift/
│   ├── performance/
│   └── alerts/
│
└── docs/
    ├── architecture.md
    ├── data_contract.md
    ├── forecasting_model.md
    ├── segmentation_model.md
    ├── training.md
    └── deployment.md

```

## Recommended logic

Use **shared modules** for things both models need:

```text
common/
data/
features/common_features.py
evaluation/reporting.py

```

Use **model-specific modules** for anything that differs:

```text
forecasting/
segmentation/
features/forecasting_features.py
features/segmentation_features.py
evaluation/forecasting_metrics.py
evaluation/segmentation_metrics.py

```

## Example responsibilities

### `forecasting/`

For time-series or demand prediction workflows.

```text
forecasting/
├── dataset.py          # Time-series dataset creation
├── preprocessing.py    # Lag features, rolling windows, date features
├── model.py            # Forecasting model class
├── train.py            # Training logic
├── predict.py          # Batch prediction
├── evaluate.py         # MAE, RMSE, MAPE, WAPE
└── inference.py        # Production inference wrapper

```

### `segmentation/`

For clustering, customer segmentation, or image/object segmentation depending on use case.

```text
segmentation/
├── dataset.py          # Segmentation dataset creation
├── preprocessing.py    # Scaling, encoding, dimensionality reduction
├── model.py            # KMeans, GMM, DBSCAN, U-Net, etc.
├── train.py            # Training or fitting logic
├── predict.py          # Segment assignment
├── evaluate.py         # Silhouette score, Davies-Bouldin, IoU, Dice, etc.
└── inference.py        # Production inference wrapper

```

## Good command pattern

```bash
python scripts/prepare_data.py --config configs/base.yaml

python scripts/train_forecasting.py --config configs/forecasting.yaml

python scripts/train_segmentation.py --config configs/segmentation.yaml

python scripts/run_full_pipeline.py --config configs/base.yaml

```

## Naming convention

Use clear model domains:

```text
forecasting/
segmentation/

```

Avoid vague names like:

```text
model_1/
model_2/

```

## Suggested config split

```yaml
# configs/base.yaml
project:
  name: ml-multi-model-project
  random_seed: 42

paths:
  raw_data: data/raw
  processed_data: data/processed
  models: models
  artifacts: artifacts

```

```yaml
# configs/forecasting.yaml
model:
  name: forecasting_model
  type: xgboost

training:
  target_column: sales
  forecast_horizon: 30
  validation_strategy: time_series_split

features:
  use_lags: true
  use_rolling_stats: true
  use_calendar_features: true

```

```yaml
# configs/segmentation.yaml
model:
  name: segmentation_model
  type: kmeans

training:
  n_clusters: 5
  scaling: standard

features:
  use_pca: true
  pca_components: 10

```

A good rule: **one shared project, separate model domains, shared infrastructure**. This keeps the project simple now but ready for production later. 