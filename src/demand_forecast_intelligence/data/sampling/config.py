"""
M5 Dataset Sampling Configuration Module.

This module defines the SamplingConfig dataclass that contains all parameters
for behavioral stratification and sampling of the M5 Walmart dataset.

The configuration enables creating statistically rigorous sample datasets while
maintaining behavioral diversity across three key dimensions:
1. Volume: Sales volume distribution (using percentile-based strata)
2. Intermittency: Frequency of non-zero sales (using ratio thresholds)
3. Lifecycle: Implicit through temporal patterns in training data

Business Context:
The M5 dataset contains 30,490 item-store combinations, which is computationally
intensive for POC development. This configuration supports a sampling strategy
that reduces dataset size by 50-60% while preserving statistical validity and
behavioral diversity needed for reliable demand forecasting models.

Statistical Rationale:
- Behavioral stratification prevents sampling bias by ensuring representation
  across different demand patterns (fast-moving, intermittent, seasonal, etc.)
- Training-period-only data prevents future leakage that could invalidate models
- Minimum constraints ensure adequate sample sizes for statistical significance
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class SamplingConfig:
    """
    Configuration parameters for M5 dataset behavioral stratification and sampling.

    This dataclass defines the complete set of parameters needed to create
    representative sample datasets from the M5 Walmart data using product-stratified
    sampling with behavioral diversity constraints.

    The configuration balances three competing objectives:
    1. Computational efficiency (reduced dataset size for faster training)
    2. Statistical rigor (unbiased representation across behavioral patterns)
    3. Business relevance (adequate representation across departments/categories)

    Attributes:
        target_item_count: Target number of unique items in sample dataset.
                          Default 1400 represents ~46% of 3,049 M5 unique items,
                          chosen to achieve 50-60% overall dataset reduction while
                          maintaining statistical power for demand forecasting.

        training_end_day: Last day of training period in M5 format (d_XXXX).
                         Default "d_1913" represents the end of M5 training data,
                         preventing future leakage that could bias model validation.
                         Critical for unbiased sampling - only uses historical data
                         available at prediction time.

        volume_percentiles: Percentile thresholds for volume-based stratification.
                           Default [0,25,75,95,100] creates 4 volume strata:
                           - Low volume: 0-25th percentile (slow-moving products)
                           - Medium volume: 25-75th percentile (steady sellers)
                           - High volume: 75-95th percentile (fast-moving products)
                           - Very high volume: 95-100th percentile (top sellers)

                           This distribution captures the heavy-tailed nature of
                           retail sales where few items generate most volume.

        intermittency_thresholds: Ratio thresholds for intermittency classification.
                                 Default [0.2, 0.6] creates 3 intermittency levels:
                                 - Highly intermittent: <20% days with sales
                                 - Moderately intermittent: 20-60% days with sales
                                 - Frequently selling: >60% days with sales

                                 These thresholds align with retail demand patterns
                                 where intermittency strongly influences forecasting
                                 approach and accuracy.

        min_per_dept: Minimum number of items required per department.
                     Default 30 ensures adequate representation across M5's
                     7 departments (FOODS_1, FOODS_2, FOODS_3, HOBBIES_1,
                     HOBBIES_2, HOUSEHOLD_1, HOUSEHOLD_2) for meaningful
                     business insights and cross-department model validation.

        min_per_stratum: Minimum number of items required per behavioral stratum.
                        Default 2 prevents empty strata while allowing natural
                        distribution to determine final stratum sizes. Ensures
                        every volume×intermittency combination has some representation
                        for robust statistical inference.

    Raises:
        ValueError: If any parameter fails validation in __post_init__.
                   Validation ensures mathematical consistency (sorted lists,
                   valid ranges) and business logic (positive counts, proper
                   date format) required for reliable sampling.

    Example:
        >>> # Use default configuration for standard M5 sampling
        >>> config = SamplingConfig()
        >>> print(f"Target items: {config.target_item_count}")
        Target items: 1400

        >>> # Customize for smaller sample with different stratification
        >>> config = SamplingConfig(
        ...     target_item_count=1000,
        ...     volume_percentiles=[0, 50, 100],  # 2 volume strata
        ...     intermittency_thresholds=[0.5]    # 2 intermittency levels
        ... )
    """

    # Core sampling parameters - balance efficiency vs statistical power
    target_item_count: int = 1400  # ~46% of 3,049 unique M5 items for 50-60% reduction
    training_end_day: str = "d_1913"  # End of M5 training period, prevents future leakage

    # Behavioral stratification thresholds - capture demand pattern diversity
    volume_percentiles: List[int] = None  # Volume distribution strata
    intermittency_thresholds: List[float] = None  # Sales frequency classification

    # Business constraints - ensure practical applicability
    min_per_dept: int = 30  # Adequate dept representation for business insights
    min_per_stratum: int = 2  # Prevent empty behavioral combinations

    def __post_init__(self):
        """
        Validate configuration parameters after initialization.

        Performs comprehensive validation to ensure mathematical consistency
        and business logic constraints required for reliable sampling:

        1. Positive integer constraints for counts
        2. Proper M5 date format for temporal boundaries
        3. Sorted, bounded lists for stratification thresholds
        4. Non-empty lists for required parameters

        This validation prevents common configuration errors that could lead
        to biased sampling or runtime failures during dataset creation.

        Raises:
            ValueError: If any parameter violates validation rules with
                       specific error message indicating the problem and
                       expected format/range.
        """
        # Set default values for mutable lists (avoided in signature to prevent shared state)
        if self.volume_percentiles is None:
            # Statistical quintiles + extremes for heavy-tailed retail distribution
            self.volume_percentiles = [0, 25, 75, 95, 100]

        if self.intermittency_thresholds is None:
            # Industry-standard intermittency classification for retail demand
            self.intermittency_thresholds = [0.2, 0.6]

        # Validate target_item_count - must be positive for meaningful sampling
        if self.target_item_count <= 0:
            raise ValueError("target_item_count must be positive")

        # Validate training_end_day - must follow M5 format (d_XXXX) for consistency
        if not re.match(r'^d_\d+$', self.training_end_day):
            raise ValueError("training_end_day must be in format 'd_XXXX' where XXXX is numeric")

        # Validate volume_percentiles - must be non-empty, sorted, and within [0,100]
        if not self.volume_percentiles:
            raise ValueError("volume_percentiles cannot be empty")

        if not all(0 <= p <= 100 for p in self.volume_percentiles):
            raise ValueError("volume_percentiles must be between 0 and 100")

        if self.volume_percentiles != sorted(self.volume_percentiles):
            raise ValueError("volume_percentiles must be sorted in ascending order")

        # Validate intermittency_thresholds - must be non-empty, sorted, and within [0,1]
        if not self.intermittency_thresholds:
            raise ValueError("intermittency_thresholds cannot be empty")

        if not all(0 <= t <= 1 for t in self.intermittency_thresholds):
            raise ValueError("intermittency_thresholds must be between 0 and 1")

        if self.intermittency_thresholds != sorted(self.intermittency_thresholds):
            raise ValueError("intermittency_thresholds must be sorted in ascending order")

        # Validate minimum constraints - must be positive for statistical significance
        if self.min_per_dept <= 0:
            raise ValueError("min_per_dept must be positive")

        if self.min_per_stratum <= 0:
            raise ValueError("min_per_stratum must be positive")