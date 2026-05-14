# Walmart M5 Dataset

This directory should contain the Walmart M5 dataset CSV files required for the retail demand forecasting copilot.

## Dataset Download

The dataset can be downloaded from the Kaggle M5 Forecasting Accuracy competition:
- **Source:** https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
- **License:** Competition data (refer to Kaggle terms)

## Required Files

The following CSV files should be placed in the `full_data/` directory:

- `calendar.csv` - Date dimension and event calendar (~101KB)
- `sales_train_validation.csv` - Primary sales training data (~114MB)
- `sales_train_evaluation.csv` - Extended sales data for evaluation (~116MB)
- `sell_prices.csv` - Weekly pricing data (~194MB)
- `sample_submission.csv` - Competition submission format (~5MB)

## Directory Structure

```
data/
├── README.md (this file)
└── full_data/
    ├── calendar.csv
    ├── sales_train_validation.csv
    ├── sales_train_evaluation.csv
    ├── sell_prices.csv
    └── sample_submission.csv
```

## Schema Documentation

For comprehensive field-level documentation of these files, see:
- `my-docs/project-info/schema-info.md`

## Note

These files are excluded from version control due to their large size (>100MB each for sales data). Download them directly from Kaggle and place them in the `full_data/` directory to use the demand forecasting pipeline.