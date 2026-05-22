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
from utils.correlation_analysis import (
    analyze_categorical_sales_patterns,
    compute_temporal_sales_correlations,
    compute_snap_benefit_impact,
    compute_cross_feature_correlations,
    detect_multicollinearity_issues
)
from utils.temporal_analysis import (
    analyze_time_structure,
    detect_seasonal_patterns,
    analyze_trend_components,
    compute_autocorrelation_analysis
)
from utils.visualization import (
    plot_categorical_sales_distributions,
    plot_correlation_heatmap,
    plot_multicollinearity_analysis,
    plot_seasonal_decomposition,
    plot_autocorrelation_analysis
)


def _transform_m5_to_long_format(sales_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform M5 sales data from wide format to long format for categorical analysis.

    Converts daily sales columns (d_1, d_2, etc.) into individual observations
    with category, store, and daily sales information.

    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales data with d_1, d_2, ... columns

    Returns
    -------
    pd.DataFrame
        Long format DataFrame with columns:
        - cat_id: Product category
        - store_id: Store identifier
        - daily_sales: Daily sales value
        - item_id: Item identifier (for reference)
    """
    # Get sales columns
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]

    if len(sales_cols) == 0:
        raise ValueError("No daily sales columns (d_*) found in sales data")

    # Transform to long format
    long_data = []

    for _, row in sales_data.iterrows():
        # Extract metadata
        item_id = row.get('item_id', 'Unknown')
        cat_id = row.get('cat_id', 'Unknown')
        store_id = row.get('store_id', 'Unknown')

        # Extract daily sales values
        for sales_col in sales_cols:
            sales_value = row[sales_col]

            # Only include valid (non-NaN) sales values
            if pd.notna(sales_value):
                long_data.append({
                    'item_id': item_id,
                    'cat_id': cat_id,
                    'store_id': store_id,
                    'daily_sales': sales_value,
                    'day_col': sales_col
                })

    if len(long_data) == 0:
        raise ValueError("No valid sales data found after transformation")

    return pd.DataFrame(long_data)


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

        print("Loaded datasets:")
        print(f"  - Sales: {len(sales_data)} rows × {len(sales_data.columns)} cols")
        print(f"  - Calendar: {len(calendar_data)} rows × {len(calendar_data.columns)} cols")
        if not price_data.empty:
            print(f"  - Prices: {len(price_data)} rows × {len(price_data.columns)} cols")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load data files: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

    results = {}

    # Transform M5 data to long format for categorical analysis
    print("\nTransforming M5 data for categorical analysis...")
    try:
        transformed_data = _transform_m5_to_long_format(sales_data)
        print(f"   ✓ Transformed {len(sales_data)} products into {len(transformed_data)} daily observations")
    except Exception as e:
        print(f"   ✗ Error transforming data: {str(e)}")
        return {'error': f'Data transformation failed: {str(e)}'}

    # 1. Categorical sales pattern analysis
    print("\n1. Analyzing categorical sales patterns...")
    try:
        # Analyze by category
        categorical_results = analyze_categorical_sales_patterns(
            data=transformed_data,
            category_col='cat_id',
            sales_col='daily_sales'
        )

        # Also analyze by store for additional insights
        store_results = analyze_categorical_sales_patterns(
            data=transformed_data,
            category_col='store_id',
            sales_col='daily_sales'
        )

        # Combine results
        categorical_results['store_analysis'] = store_results
        results['categorical_patterns'] = categorical_results
        print(f"   ✓ Analyzed {len(categorical_results.get('categories', []))} categories")
        print(f"   ✓ Analyzed {len(store_results.get('categories', []))} stores")
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

    # Use the already transformed data for visualization
    if 'transformed_data' in locals() and len(transformed_data) > 0:
        # Ensure plot directory exists
        plot_dir = "notebooks/eda/plots/step6_feature_target"
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        # Plot categorical distributions
        try:
            plot_path = os.path.join(plot_dir, "category_sales_distributions.png")
            plot_results = plot_categorical_sales_distributions(transformed_data, plot_path)
            results['visualizations'] = {'category_distributions': plot_results}
            print("   ✓ Generated category distribution plot")
            print(f"     Path: {plot_path}")
        except Exception as e:
            print(f"   ✗ Error generating visualization: {str(e)}")
            results['visualizations'] = {'error': str(e)}
    else:
        print("   ✗ No valid sales data for visualization")

    # 5. Generate summary
    print("\n5. Generating summary...")

    # Get categories from results
    categories_count = len(categorical_results.get('categories', [])) if 'categories' in categorical_results else 0
    snap_states_count = len(snap_results.get('snap_impact_by_state', {})) if 'snap_impact_by_state' in snap_results else 0
    total_observations = len(transformed_data) if 'transformed_data' in locals() else 0

    summary = {
        'total_categories': categories_count,
        'temporal_features_analyzed': 2,  # weekday, month
        'snap_states_analyzed': snap_states_count,
        'total_observations': total_observations,
        'step_status': 'complete'
    }

    results['summary'] = summary

    print("\nStep 6 analysis complete!")
    print("-" * 60)
    print("Summary:")
    print(f"  - Categories analyzed: {summary['total_categories']}")
    print(f"  - Temporal features: {summary['temporal_features_analyzed']}")
    print(f"  - SNAP states: {summary['snap_states_analyzed']}")
    print(f"  - Total observations: {summary['total_observations']}")

    return results


def study_feature_feature_relationships(
    data_path: str = "/Users/rahul.vansh/Documents/Personal/demand_forecast_intelligence/data/raw"
) -> Dict[str, Any]:
    """
    Step 7: Study feature-feature relationships.

    Analyze relationships between features to understand:
    - Product hierarchy (category-department correlations)
    - Geographic demand pattern similarities
    - Price coordination across stores
    - Multicollinearity issues for forecasting models

    This step focuses on understanding business-meaningful feature relationships
    and identifying potential issues for model development.

    Parameters
    ----------
    data_path : str
        Path to directory containing M5 CSV files

    Returns
    -------
    Dict[str, Any]
        Comprehensive analysis results containing:
        - product_hierarchy: Category-department relationships
        - geographic_patterns: Store correlations within states
        - multicollinearity: Potential modeling issues
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
    >>> results = study_feature_feature_relationships()
    >>> print(results['summary'])
    """
    print("Starting Step 7: Feature-Feature Relationship Analysis")
    print("-" * 60)

    # Load datasets
    try:
        sales_path = os.path.join(data_path, "sales_train_validation.csv")

        if not os.path.exists(sales_path):
            raise FileNotFoundError(f"Sales file not found: {sales_path}")

        sales_data = pd.read_csv(sales_path)

        print("Loaded datasets:")
        print(f"  - Sales: {len(sales_data)} rows × {len(sales_data.columns)} cols")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load data files: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

    results = {}

    # 1. Compute cross-feature correlations
    print("\n1. Computing cross-feature correlations...")
    try:
        cross_feature_results = compute_cross_feature_correlations(sales_data)
        results['cross_feature_correlations'] = cross_feature_results

        if 'product_hierarchy_correlations' in cross_feature_results:
            hierarchy = cross_feature_results['product_hierarchy_correlations']
            print(f"   ✓ Analyzed {hierarchy.get('categories_count', 0)} categories")
            print(f"   ✓ Analyzed {hierarchy.get('departments_count', 0)} departments")

        if 'geographic_correlations' in cross_feature_results:
            geo = cross_feature_results['geographic_correlations']
            print(f"   ✓ Analyzed {geo.get('states_count', 0)} states")

    except Exception as e:
        print(f"   ✗ Error in cross-feature analysis: {str(e)}")
        results['cross_feature_correlations'] = {'error': str(e)}

    # 2. Create feature matrix for multicollinearity analysis
    print("2. Creating feature matrix for multicollinearity detection...")
    try:
        # Extract numeric features from M5 data
        # Focus on category-level aggregates for cleaner analysis
        feature_matrix = _create_feature_matrix_for_multicollinearity(sales_data)

        if len(feature_matrix) > 0:
            print(f"   ✓ Created feature matrix with {len(feature_matrix)} samples")
            print(f"   ✓ Analyzing {len(feature_matrix.columns)} features")

            # 3. Detect multicollinearity issues
            print("3. Detecting multicollinearity issues...")
            try:
                multicollinearity_results = detect_multicollinearity_issues(
                    feature_matrix,
                    threshold=0.8
                )
                results['multicollinearity_analysis'] = multicollinearity_results

                high_pairs = multicollinearity_results.get('high_correlation_pairs', [])
                print(f"   ✓ Identified {len(high_pairs)} high-correlation feature pairs")

                # Show recommendations
                recommendations = multicollinearity_results.get('recommendations', [])
                if recommendations:
                    print(f"   ✓ Recommendations generated ({len(recommendations)} items)")

            except Exception as e:
                print(f"   ✗ Error in multicollinearity analysis: {str(e)}")
                results['multicollinearity_analysis'] = {'error': str(e)}
        else:
            print("   ℹ  Insufficient data for multicollinearity analysis")

    except Exception as e:
        print(f"   ✗ Error creating feature matrix: {str(e)}")

    # 4. Generate visualizations
    print("4. Generating feature-feature relationship plots...")

    # Ensure plot directory exists
    plot_dir = "notebooks/eda/plots/step7_feature_relationships"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    visualizations = {}

    # Correlation heatmap
    if 'feature_matrix' in locals() and len(feature_matrix) > 0:
        try:
            heatmap_path = os.path.join(plot_dir, "correlation_heatmap.png")
            heatmap_results = plot_correlation_heatmap(
                feature_matrix,
                heatmap_path,
                title='Feature Correlation Analysis',
                cluster=True
            )
            visualizations['correlation_heatmap'] = heatmap_results
            print("   ✓ Generated correlation heatmap")
            print(f"     Path: {heatmap_path}")
        except Exception as e:
            print(f"   ✗ Error generating correlation heatmap: {str(e)}")

        # Multicollinearity analysis plot
        try:
            multicollinearity_path = os.path.join(plot_dir, "multicollinearity_analysis.png")
            multicollinearity_plot_results = plot_multicollinearity_analysis(
                feature_matrix,
                multicollinearity_path,
                threshold=0.8
            )
            visualizations['multicollinearity_plot'] = multicollinearity_plot_results
            print("   ✓ Generated multicollinearity analysis plot")
            print(f"     Path: {multicollinearity_path}")
        except Exception as e:
            print(f"   ✗ Error generating multicollinearity plot: {str(e)}")

    results['visualizations'] = visualizations

    # 5. Generate summary
    print("\n5. Generating summary...")

    high_corr_count = len(results.get('multicollinearity_analysis', {}).get('high_correlation_pairs', []))
    features_analyzed = len(feature_matrix.columns) if 'feature_matrix' in locals() else 0

    summary = {
        'features_analyzed': features_analyzed,
        'high_correlation_pairs': high_corr_count,
        'product_hierarchy_analyzed': 'product_hierarchy_correlations' in results.get('cross_feature_correlations', {}),
        'geographic_patterns_analyzed': 'geographic_correlations' in results.get('cross_feature_correlations', {}),
        'step_status': 'complete'
    }

    results['summary'] = summary

    print("\nStep 7 analysis complete!")
    print("-" * 60)
    print("Summary:")
    print(f"  - Features analyzed: {summary['features_analyzed']}")
    print(f"  - High correlation pairs: {summary['high_correlation_pairs']}")
    print(f"  - Product hierarchy: {'Yes' if summary['product_hierarchy_analyzed'] else 'No'}")
    print(f"  - Geographic patterns: {'Yes' if summary['geographic_patterns_analyzed'] else 'No'}")

    return results


def _create_feature_matrix_for_multicollinearity(sales_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a feature matrix for multicollinearity analysis from M5 sales data.

    Aggregates daily sales data to create features suitable for correlation analysis.

    Parameters
    ----------
    sales_data : pd.DataFrame
        M5 sales data with daily sales columns (d_*)

    Returns
    -------
    pd.DataFrame
        Feature matrix with category-level aggregate features
    """
    # Get sales columns
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]

    if len(sales_cols) == 0:
        return pd.DataFrame()

    # Create features by category
    features = []

    if 'cat_id' in sales_data.columns:
        for cat in sales_data['cat_id'].unique():
            if pd.isna(cat):
                continue

            cat_data = sales_data[sales_data['cat_id'] == cat][sales_cols]

            if len(cat_data) > 0:
                daily_totals = cat_data.sum(axis=0)

                # Extract features: mean, std, trend
                mean_sales = float(daily_totals.mean())
                std_sales = float(daily_totals.std())
                min_sales = float(daily_totals.min())
                max_sales = float(daily_totals.max())

                # Simple trend calculation (last week vs first week)
                if len(daily_totals) >= 14:
                    first_week = daily_totals.iloc[:7].mean()
                    last_week = daily_totals.iloc[-7:].mean()
                    trend = float(last_week - first_week) if first_week > 0 else 0
                else:
                    trend = 0

                features.append({
                    'category': cat,
                    'mean_sales': mean_sales,
                    'std_sales': std_sales,
                    'min_sales': min_sales,
                    'max_sales': max_sales,
                    'trend': trend
                })

    if len(features) > 0:
        feature_df = pd.DataFrame(features)
        # Keep only numeric columns
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        return feature_df[numeric_cols]
    else:
        return pd.DataFrame()


def analyze_time_series_patterns(
    data_path: str = "/Users/rahul.vansh/Documents/Personal/demand_forecast_intelligence/data/raw"
) -> Dict[str, Any]:
    """
    Step 8: Special time-series EDA.

    Comprehensive temporal pattern analysis including seasonality detection,
    trend analysis, and autocorrelation structure identification. Provides
    business-focused interpretations for demand forecasting model specification.

    Parameters
    ----------
    data_path : str, default raw data path
        Path to directory containing M5 CSV files (sales_train_validation.csv, calendar.csv)

    Returns
    -------
    Dict[str, Any]
        Comprehensive time series analysis results including:
        - time_structure: Panel data structure validation
        - seasonal_patterns: Category-level seasonality analysis
        - trend_analysis: Linear trend and structural break detection
        - autocorrelation_analysis: Lag structure for forecasting
        - visualizations: Path references to generated plots

    Examples
    --------
    >>> results = analyze_time_series_patterns()
    >>> print(results['trend_analysis']['linear_trend'])
    >>> print(results['autocorrelation_analysis']['business_interpretation'])
    """
    print("Starting Step 8: Time Series Pattern Analysis")

    # Load datasets
    sales_data = pd.read_csv(os.path.join(data_path, "sales_train_validation.csv"))
    calendar_data = pd.read_csv(os.path.join(data_path, "calendar.csv"))

    results = {}

    # 1. Time structure analysis
    print("Analyzing time structure...")
    time_structure = analyze_time_structure(sales_data, calendar_data)
    results['time_structure'] = time_structure

    # 2. Seasonal pattern detection
    print("Detecting seasonal patterns...")
    seasonal_patterns = detect_seasonal_patterns(sales_data, calendar_data, hierarchy_level='category')
    results['seasonal_patterns'] = seasonal_patterns

    # 3. Trend analysis
    print("Analyzing trend components...")
    trend_analysis = analyze_trend_components(sales_data, calendar_data)
    results['trend_analysis'] = trend_analysis

    # 4. Autocorrelation analysis
    print("Computing autocorrelation analysis...")
    autocorr_analysis = compute_autocorrelation_analysis(sales_data, max_lags=365)
    results['autocorrelation_analysis'] = autocorr_analysis

    # 5. Generate visualizations
    print("Generating time series plots...")

    # Prepare total daily sales time series
    sales_cols = [col for col in sales_data.columns if col.startswith('d_')]
    daily_totals = []
    for col in sales_cols:
        daily_total = sales_data[col].sum()
        daily_totals.append(daily_total)

    ts = pd.Series(daily_totals)

    # Seasonal decomposition plot
    decomp_path = "notebooks/eda/plots/step8_time_series/seasonal_decomposition.png"
    Path(decomp_path).parent.mkdir(parents=True, exist_ok=True)
    decomp_results = plot_seasonal_decomposition(
        ts, decomp_path,
        title="M5 Total Sales Seasonal Decomposition"
    )

    # Autocorrelation plot
    autocorr_path = "notebooks/eda/plots/step8_time_series/autocorrelation_analysis.png"
    Path(autocorr_path).parent.mkdir(parents=True, exist_ok=True)
    autocorr_plot_results = plot_autocorrelation_analysis(autocorr_analysis, autocorr_path)

    results['visualizations'] = {
        'seasonal_decomposition': decomp_results,
        'autocorrelation_plot': autocorr_plot_results
    }

    print(f"Step 8 analysis complete. Identified {len(autocorr_analysis['significant_lags'])} significant lag patterns.")

    return results
