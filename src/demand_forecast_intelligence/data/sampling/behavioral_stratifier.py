"""
Behavioral Stratification Engine for M5 Dataset Sampling.

This module implements the core anti-bias mechanism for M5 dataset sampling through
multi-dimensional behavioral stratification. The approach prevents sampling bias
toward high-volume, stable items by ensuring proper representation of intermittent
and sparse demand patterns critical for robust forecasting model validation.

Business Rationale:
Traditional sampling approaches bias toward "clean" high-volume items, making POC
validation overly optimistic. This behavioral stratification ensures the sample
includes challenging intermittent/sparse demand patterns - the very patterns that
cause production model failures.

Key Innovation:
Uses ONLY training period data (d_1 to d_1913) for all metric calculations to
prevent future leakage. This ensures sampling decisions don't use information
that wouldn't be available during actual model training.

Statistical Rationale for Stratification:
- Volume within departments: Retail sales follow heavy-tailed distributions - need
  representation across the full spectrum
- Intermittency: Sparse items (≤20% nonzero days) are forecasting challenges but
  business-critical
- Lifecycle: Product maturity affects demand patterns - early vs established vs
  declining products behave differently
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class BehavioralStratifier:
    """
    Creates multi-dimensional behavioral strata to prevent sampling bias.

    This class implements the core behavioral stratification engine that segments
    M5 products across three key dimensions:
    1. Volume (within departments to prevent category bias)
    2. Intermittency (sparse/intermittent/regular patterns)
    3. Lifecycle (product maturity stages)

    The stratification prevents bias toward high-volume, stable items and ensures
    challenging intermittent/sparse patterns are properly represented in samples.

    Design Principles:
    - Uses training-period-only data (d_1 to d_1913) to prevent future leakage
    - Volume bucketing within departments prevents category bias
    - Intermittency classification captures forecasting difficulty
    - Lifecycle analysis identifies product maturity stages
    - Composite stratum keys enable proper allocation logic
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BehavioralStratifier with configuration.

        Args:
            config: Dictionary containing stratification parameters:
                - volume_buckets: List of volume bucket names (e.g., ['low', 'medium', 'high', 'very_high'])
                - volume_percentiles: Percentile thresholds for bucketing (e.g., [33, 67] for tertiles)
                - intermittency_thresholds: Dict with sparse/intermittent/regular thresholds
                - lifecycle_windows: Dict with early_period, late_period, min_active_days

        The configuration enables flexible stratification tuning while maintaining
        the core anti-bias behavioral segmentation approach.
        """
        self.config = config
        self.volume_buckets = config['volume_buckets']
        self.volume_percentiles = config['volume_percentiles']
        self.intermittency_thresholds = config['intermittency_thresholds']
        self.lifecycle_windows = config['lifecycle_windows']

        logger.info(f"Initialized BehavioralStratifier with {len(self.volume_buckets)} volume buckets")

    def calculate_item_metrics(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract training-period-only metrics for behavioral stratification.

        This method implements the critical anti-leakage requirement by using ONLY
        training period data (d_1 to d_1913) for all metric calculations. This ensures
        sampling decisions don't use information that wouldn't be available during
        actual model training.

        Args:
            sales_data: M5 sales DataFrame with id, item_id, dept_id columns and
                       daily sales columns (d_1, d_2, ..., d_1969)

        Returns:
            DataFrame with behavioral metrics for each item:
            - Basic metrics: total_sales, avg_daily_sales, total_days
            - Intermittency metrics: nonzero_days, nonzero_day_ratio
            - Lifecycle metrics: early_period_sales, late_period_sales

        The metrics capture the behavioral dimensions needed for stratification:
        - Volume patterns for within-department bucketing
        - Intermittency patterns for forecasting difficulty assessment
        - Lifecycle patterns for product maturity analysis
        """
        logger.info("Calculating item metrics from training period data only (d_1 to d_1913)")

        # Extract training period columns only (d_1 to d_1913)
        # This prevents future leakage - sampling uses only data available during training
        training_columns = [f'd_{i}' for i in range(1, 1914)]
        available_training_columns = [col for col in training_columns if col in sales_data.columns]

        if not available_training_columns:
            raise ValueError("No training period columns (d_1 to d_1913) found in sales data")

        logger.info(f"Found {len(available_training_columns)} training period columns")

        # Create metrics DataFrame with item identifiers
        metrics = sales_data[['id', 'item_id', 'dept_id', 'cat_id', 'store_id']].copy()

        # Extract sales values for training period only
        sales_values = sales_data[available_training_columns].values

        # Calculate basic volume metrics from training period
        # These capture the volume dimension for within-department bucketing
        metrics['total_sales'] = sales_values.sum(axis=1)
        metrics['total_days'] = len(available_training_columns)
        metrics['avg_daily_sales'] = metrics['total_sales'] / metrics['total_days']

        # Calculate intermittency metrics
        # These capture the forecasting difficulty dimension
        # Sparse items (≤20% nonzero days) are the most challenging forecasting cases
        metrics['nonzero_days'] = (sales_values > 0).sum(axis=1)
        metrics['nonzero_day_ratio'] = metrics['nonzero_days'] / metrics['total_days']

        # Calculate lifecycle metrics
        # These capture product maturity patterns for lifecycle stratification
        early_days = self.lifecycle_windows['early_period']
        late_days = self.lifecycle_windows['late_period']

        # Early period sales (first N days) - identifies new/emerging products
        if early_days <= len(available_training_columns):
            early_sales = sales_values[:, :early_days].sum(axis=1)
        else:
            early_sales = sales_values.sum(axis=1)  # Use all available if less than window

        # Late period sales (last N days) - identifies declining/discontinued products
        if late_days <= len(available_training_columns):
            late_sales = sales_values[:, -late_days:].sum(axis=1)
        else:
            late_sales = sales_values.sum(axis=1)  # Use all available if less than window

        metrics['early_period_sales'] = early_sales
        metrics['late_period_sales'] = late_sales

        logger.info(f"Calculated metrics for {len(metrics)} items using training period only")

        return metrics

    def create_behavioral_strata(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Create multi-dimensional behavioral strata to prevent sampling bias.

        This method implements the core anti-bias mechanism through behavioral
        stratification across three key dimensions:
        1. Volume buckets (within departments to prevent category bias)
        2. Intermittency classification (sparse/intermittent/regular patterns)
        3. Lifecycle stages (product maturity analysis)

        Args:
            metrics: DataFrame with item behavioral metrics from calculate_item_metrics()

        Returns:
            DataFrame with original metrics plus stratification columns:
            - volume_bucket: Volume bucket within department (low/medium/high/very_high)
            - intermittency_class: Forecasting difficulty (sparse/intermittent/regular)
            - lifecycle_stage: Product maturity (early/mature/declining/discontinued)
            - stratum_key: Composite key for allocation (dept_volume_intermittency_lifecycle)

        The stratification ensures:
        - Each department contributes across the volume spectrum
        - Challenging sparse/intermittent patterns are captured
        - Different product lifecycle stages are represented
        - Composite keys enable proper sample allocation
        """
        logger.info("Creating behavioral strata to prevent sampling bias")

        strata = metrics.copy()

        # 1. Volume bucketing within departments
        # This prevents category bias - each department contributes across volume spectrum
        # Without this, high-volume categories (FOODS) would dominate samples
        logger.info("Creating volume buckets within departments to prevent category bias")

        volume_buckets = []
        for dept in strata['dept_id'].unique():
            dept_mask = strata['dept_id'] == dept
            dept_data = strata.loc[dept_mask]

            if len(dept_data) == 1:
                # Single item in department gets assigned to the second bucket (medium)
                volume_buckets.extend(['medium'])
            else:
                # Compute threshold values from inner cut points (e.g. [25, 75, 95] after
                # stripping the 0/100 sentinels from config.volume_percentiles [0,25,75,95,100])
                dept_sales = dept_data['avg_daily_sales'].values
                percentile_values = np.percentile(dept_sales, self.volume_percentiles)

                # Count how many thresholds each item's sales exceeds → bucket index
                # e.g. 3 thresholds → indices 0-3 → maps to ['low','medium','high','very_high']
                dept_buckets = [
                    self.volume_buckets[int(np.sum(sales > percentile_values))]
                    for sales in dept_sales
                ]

                volume_buckets.extend(dept_buckets)

        strata['volume_bucket'] = volume_buckets

        # 2. Intermittency classification based on nonzero day ratio
        # This captures forecasting difficulty - sparse items are most challenging
        logger.info("Classifying intermittency patterns for forecasting difficulty")

        def classify_intermittency(nonzero_ratio):
            """
            Classify items by intermittency for forecasting difficulty assessment.

            - Sparse (≤20% nonzero days): Most challenging forecasting cases, but business-critical
            - Intermittent (20-60% nonzero days): Moderate forecasting challenges
            - Regular (>60% nonzero days): Stable, predictable patterns

            This classification ensures challenging patterns are captured in samples.
            """
            if nonzero_ratio <= self.intermittency_thresholds['sparse']:
                return 'sparse'
            elif nonzero_ratio <= self.intermittency_thresholds['intermittent']:
                return 'intermittent'
            else:
                return 'regular'

        strata['intermittency_class'] = strata['nonzero_day_ratio'].apply(classify_intermittency)

        # 3. Lifecycle stage analysis
        # This captures product maturity patterns for comprehensive representation
        logger.info("Analyzing product lifecycle stages for maturity patterns")

        def classify_lifecycle(row):
            """
            Classify product lifecycle stage based on early vs late period activity.

            Uses ratio of late_period to early_period sales to identify:
            - Early: New/emerging products (low early, building momentum)
            - Mature: Established products (consistent activity)
            - Declining: Products losing momentum (high early, declining late)
            - Discontinued: Products with minimal late activity

            This ensures different product maturity stages are represented.
            """
            early_sales = row['early_period_sales']
            late_sales = row['late_period_sales']
            min_active = self.lifecycle_windows['min_active_days']

            # Check if product has sufficient activity for classification
            total_nonzero = row['nonzero_days']
            if total_nonzero < min_active:
                return 'minimal'

            # Avoid division by zero
            if early_sales == 0:
                if late_sales > 0:
                    return 'early'  # Started later in period
                else:
                    return 'minimal'

            # Calculate late-to-early ratio for lifecycle assessment
            late_to_early_ratio = late_sales / early_sales

            if late_to_early_ratio > 1.5:
                return 'early'      # Building momentum
            elif late_to_early_ratio > 0.5:
                return 'mature'     # Stable activity
            elif late_to_early_ratio > 0.1:
                return 'declining'  # Losing momentum
            else:
                return 'discontinued'  # Minimal late activity

        strata['lifecycle_stage'] = strata.apply(classify_lifecycle, axis=1)

        # 4. Create composite stratum keys for allocation logic
        # Format: dept|volume|intermittency|lifecycle (using | to avoid underscore conflicts)
        logger.info("Creating composite stratum keys for allocation logic")

        strata['stratum_key'] = (
            strata['dept_id'] + '|' +
            strata['volume_bucket'] + '|' +
            strata['intermittency_class'] + '|' +
            strata['lifecycle_stage']
        )

        # Log stratification results for monitoring bias prevention
        logger.info("Behavioral stratification summary:")
        logger.info(f"Volume buckets: {strata['volume_bucket'].value_counts().to_dict()}")
        logger.info(f"Intermittency classes: {strata['intermittency_class'].value_counts().to_dict()}")
        logger.info(f"Lifecycle stages: {strata['lifecycle_stage'].value_counts().to_dict()}")
        logger.info(f"Unique strata: {strata['stratum_key'].nunique()}")

        return strata