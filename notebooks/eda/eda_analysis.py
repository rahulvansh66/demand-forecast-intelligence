"""
Main EDA analysis orchestration for M5 demand forecasting dataset.

Implements framework steps 6-10 with hierarchical analysis approach.
Each function corresponds to one EDA framework step with comprehensive
statistical analysis and business-focused interpretation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import os
import sys

# Add notebooks/eda to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utility modules
from utils.statistical_analysis import (
    calculate_distribution_stats,
    compute_variation_metrics,
    analyze_outliers
)
from utils.correlation_analysis import (
    analyze_categorical_sales_patterns,
    compute_temporal_sales_correlations,
    compute_snap_benefit_impact
)
from utils.visualization import plot_categorical_sales_distributions


def study_feature_target_relationships(
    data_path: str = "/Users/rahul.vansh/Documents/Personal/demand_forecast_intelligence/data/raw"
) -> Dict[str, Any]:
    """
    Step 6: Study feature-target relationships.

    Analyze relationships between categorical features (category, store, etc.)
    and sales targets using hierarchical business-focused approach.

    This step focuses on understanding how different product categories,
    stores, and states relate to demand patterns. It provides insights
    into category-level performance and business factors affecting sales.

    Parameters
    ----------
    data_path : str
        Path to directory containing M5 CSV files (sales_train_validation.csv, etc.)

    Returns
    -------
    Dict[str, Any]
        Comprehensive analysis results containing:
        - categorical_patterns: Sales patterns by category/store/state
        - temporal_correlations: Relationships between temporal features and sales
        - snap_impact: SNAP benefit impact on FOODS category
        - visualizations: Generated plot metadata
        - summary: High-level findings

    Raises
    ------
    FileNotFoundError
        If required CSV files are not found
    ValueError
        If data validation fails

    Example
    -------
    >>> results = study_feature_target_relationships()
    >>> print(results['summary'])
    """
    print("Starting Step 6: Feature-Target Relationship Analysis")
    print("-" * 60)

    # Load datasets
    try:
        sales_path = os.path.join(data_path, "sales_train_validation.csv")
        calendar_path = os.path.join(data_path, "calendar.csv")
        price_path = os.path.join(data_path, "sell_prices.csv")

        if not os.path.exists(sales_path):
            raise FileNotFoundError(f"Sales file not found: {sales_path}")
        if not os.path.exists(calendar_path):
            raise FileNotFoundError(f"Calendar file not found: {calendar_path}")

        sales_data = pd.read_csv(sales_path)
        calendar_data = pd.read_csv(calendar_path)

        # Price data is optional
        if os.path.exists(price_path):
            price_data = pd.read_csv(price_path)
        else:
            price_data = pd.DataFrame()

        print(f"Loaded datasets:")
        print(f"  - Sales: {len(sales_data)} rows × {len(sales_data.columns)} cols")
        print(f"  - Calendar: {len(calendar_data)} rows × {len(calendar_data.columns)} cols")
        if not price_data.empty:
            print(f"  - Prices: {len(price_data)} rows × {len(price_data.columns)} cols")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load data files: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

    results = {}

    # 1. Categorical sales pattern analysis
    print("\n1. Analyzing categorical sales patterns...")
    try:
        categorical_results = analyze_categorical_sales_patterns(
            sales_data, calendar_data, price_data
        )
        results['categorical_patterns'] = categorical_results
        print(f"   ✓ Identified {len(categorical_results.get('category_performance', {}))} categories")
    except Exception as e:
        print(f"   ✗ Error in categorical analysis: {str(e)}")
        results['categorical_patterns'] = {'error': str(e)}

    # 2. Temporal correlation analysis
    print("2. Computing temporal sales correlations...")
    try:
        temporal_results = compute_temporal_sales_correlations(sales_data, calendar_data)
        results['temporal_correlations'] = temporal_results
        print(f"   ✓ Analyzed temporal patterns for {len(temporal_results.get('temporal_correlations', {}))} categories")
    except Exception as e:
        print(f"   ✗ Error in temporal analysis: {str(e)}")
        results['temporal_correlations'] = {'error': str(e)}

    # 3. SNAP benefit impact analysis
    print("3. Analyzing SNAP benefit impact...")
    try:
        snap_results = compute_snap_benefit_impact(sales_data, calendar_data)
        results['snap_impact'] = snap_results
        if 'error' not in snap_results:
            snap_states = len(snap_results.get('snap_impact_by_state', {}))
            print(f"   ✓ Analyzed SNAP impact for {snap_states} states")
        else:
            print(f"   ℹ  SNAP analysis: {snap_results['error']}")
    except Exception as e:
        print(f"   ✗ Error in SNAP analysis: {str(e)}")
        results['snap_impact'] = {'error': str(e)}

    # 4. Generate visualizations
    print("4. Generating feature-target relationship plots...")

    # Prepare data for visualization
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]

    # Create daily sales summary by category
    viz_data = []
    for _, row in sales_data.iterrows():
        daily_sales = row[sales_cols].values
        for sales_val in daily_sales:
            if pd.notna(sales_val):
                viz_data.append({
                    'cat_id': row.get('cat_id', 'Unknown'),
                    'store_id': row.get('store_id', 'Unknown'),
                    'daily_sales': sales_val
                })

    if len(viz_data) > 0:
        viz_df = pd.DataFrame(viz_data)

        # Ensure plot directory exists
        plot_dir = "notebooks/eda/plots/step6_feature_target"
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        # Plot categorical distributions
        try:
            plot_path = os.path.join(plot_dir, "category_sales_distributions.png")
            plot_results = plot_categorical_sales_distributions(viz_df, plot_path)
            results['visualizations'] = {'category_distributions': plot_results}
            print(f"   ✓ Generated category distribution plot")
            print(f"     Path: {plot_path}")
        except Exception as e:
            print(f"   ✗ Error generating visualization: {str(e)}")
            results['visualizations'] = {'error': str(e)}
    else:
        print("   ✗ No valid sales data for visualization")

    # 5. Generate summary
    print("\n5. Generating summary...")

    summary = {
        'total_categories': len(categorical_results.get('category_performance', {})) if 'category_performance' in categorical_results else 0,
        'temporal_features_analyzed': 2,  # weekday, month
        'snap_states_analyzed': len(snap_results.get('snap_impact_by_state', {})) if 'snap_impact_by_state' in snap_results else 0,
        'total_observations': len(viz_data) if len(viz_data) > 0 else 0,
        'step_status': 'complete'
    }

    results['summary'] = summary

    print("\nStep 6 analysis complete!")
    print("-" * 60)
    print(f"Summary:")
    print(f"  - Categories analyzed: {summary['total_categories']}")
    print(f"  - Temporal features: {summary['temporal_features_analyzed']}")
    print(f"  - SNAP states: {summary['snap_states_analyzed']}")
    print(f"  - Total observations: {summary['total_observations']}")

    return results
