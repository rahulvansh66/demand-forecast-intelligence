"""
M5 Dataset Sampling Module.

This module provides functionality for creating statistically rigorous sample datasets
from the M5 Walmart dataset using product-stratified sampling with behavioral diversity.

The sampling approach addresses the challenge of M5's large scale (30,490 item-store
combinations) by creating representative subsets that maintain:
- Statistical rigor through stratified sampling
- Behavioral diversity across volume, intermittency, and lifecycle dimensions
- Prevention of future data leakage by using training-period-only data
- Business relevance through department-aware constraints

Key Components:
- config: Configuration parameters for behavioral stratification thresholds
- stratification: Logic for creating behavioral strata
- sampling: Core sampling algorithms with bias prevention

Business Rationale:
The full M5 dataset requires 60+ minutes for model training on M3 Pro hardware,
making rapid POC development impractical. This sampling system achieves:
- 50-60% dataset size reduction (target: ~1,400 items from 3,049 unique items)
- 20-30 minute training times for faster iteration
- Maintained statistical validity for business insights
- Unbiased representation across product categories and behaviors
"""

from .config import SamplingConfig

__all__ = ["SamplingConfig"]