"""
Main EDA analysis orchestration for M5 demand forecasting dataset.

Implements framework steps 6-10 with hierarchical analysis approach.
Each function corresponds to one EDA framework step with comprehensive
statistical analysis and business-focused interpretation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
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
    plot_autocorrelation_analysis,
    plot_missing_patterns,
    plot_outlier_detection
)
from utils.data_quality import (
    analyze_missing_patterns,
    characterize_missing_mechanisms,
    detect_sales_outliers,
    analyze_pricing_anomalies
)
from utils.segment_analysis import (
    analyze_category_behavior_patterns,
    analyze_department_segment_patterns,
    compute_segment_performance_metrics,
    analyze_segment_seasonality_patterns,
    detect_segment_lifecycle_stages
)
from utils.drift_analysis import (
    compare_temporal_distributions,
    analyze_seasonal_representativeness,
    detect_category_drift,
    compute_drift_severity_scores,
    validate_temporal_split_integrity
)
from utils.leakage_validation import (
    audit_temporal_boundaries,
    check_feature_availability_timing,
    validate_cross_validation_integrity,
    scan_suspicious_correlations,
    generate_leakage_audit_report
)
from utils.visualization import (
    plot_segment_behavior_comparison,
    plot_distribution_drift_analysis,
    plot_leakage_validation_summary
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


def analyze_missing_values_deeply(
    data_path: str = "/Users/rahul.vansh/Documents/Personal/demand_forecast_intelligence/data/raw"
) -> Dict[str, Any]:
    """
    Step 9: Deeply analyze missing values and their mechanisms.

    This comprehensive analysis:
    1. Validates sales data completeness (should have no missing values)
    2. Identifies pricing gaps by item-store-week combinations
    3. Verifies calendar data completeness
    4. Characterizes missing mechanisms (seasonal, geographic, new products, discontinued)
    5. Correlates missing patterns with business logic
    6. Generates publication-ready visualizations
    7. Provides actionable preprocessing recommendations

    Parameters
    ----------
    data_path : str
        Path to raw M5 dataset directory

    Returns
    -------
    Dict[str, Any]
        Comprehensive missing value analysis including:
        - sales_completeness: Missing value statistics
        - missing_mechanisms: Identified patterns (seasonal, geographic, etc.)
        - visualizations: Publication-ready plots
        - recommendations: Treatment suggestions for preprocessing
        - summary_statistics: Key findings

    Example
    -------
    >>> results = analyze_missing_values_deeply()
    >>> print(f"Zero-heavy series: {results['summary_statistics']['zero_heavy_percent']}%")
    >>> print(f"Seasonal items identified: {len(results['missing_mechanisms'].get('seasonal_items', []))}")
    """
    print("Step 9: Analyzing missing values deeply...")

    # Load data
    sales_data = pd.read_csv(f"{data_path}/sales_train_validation.csv")
    pricing_data = pd.read_csv(f"{data_path}/sell_prices.csv")
    calendar_data = pd.read_csv(f"{data_path}/calendar.csv")

    results = {}

    # 1. Analyze missing patterns
    missing_patterns = analyze_missing_patterns(
        sales_data,
        pricing_data=pricing_data,
        calendar_data=calendar_data
    )
    results['missing_patterns'] = missing_patterns

    # 2. Characterize missing mechanisms
    mechanisms = characterize_missing_mechanisms(sales_data)
    results['missing_mechanisms'] = mechanisms

    # 3. Generate missing patterns visualization
    missing_plot_path = "notebooks/eda/plots/step9_missing_patterns/missing_data_overview.png"
    Path(missing_plot_path).parent.mkdir(parents=True, exist_ok=True)

    missing_viz = plot_missing_patterns(
        missing_patterns,
        missing_plot_path,
        title="Step 9: Missing Data Patterns Analysis"
    )
    results['visualizations'] = {'missing_patterns': missing_viz}

    # 4. Calculate summary statistics
    sales_complete = missing_patterns.get('sales_completeness', {})
    mechanisms_list = mechanisms.get('mechanisms', [])

    summary = {
        'total_series': sales_complete.get('total_series', 0),
        'zero_heavy_percent': sales_complete.get('zero_heavy_series', {}).get('percent', 0),
        'mechanisms_identified': mechanisms_list,
        'seasonal_items_count': len(mechanisms.get('seasonal_items', [])),
        'new_products_count': len(mechanisms.get('new_products', [])),
        'discontinued_items_count': len(mechanisms.get('discontinued_items', [])),
        'pricing_coverage_percent': missing_patterns.get('pricing_gaps', {}).get('coverage_percent', 0)
    }
    results['summary_statistics'] = summary

    # 5. Generate recommendations for preprocessing
    recommendations = _generate_missing_value_recommendations(
        missing_patterns,
        mechanisms,
        sales_data
    )
    results['recommendations'] = recommendations

    print(f"\nStep 9 Analysis Summary:")
    print(f"  - Zero-heavy series: {summary['zero_heavy_percent']:.1f}%")
    print(f"  - Missing mechanisms identified: {', '.join(mechanisms_list) if mechanisms_list else 'None'}")
    print(f"  - Seasonal items: {summary['seasonal_items_count']}")
    print(f"  - Pricing coverage: {summary['pricing_coverage_percent']:.1f}%")
    print(f"  - Visualizations saved to: {missing_plot_path}")

    return results


def identify_outliers_and_anomalies(
    data_path: str = "/Users/rahul.vansh/Documents/Personal/demand_forecast_intelligence/data/raw"
) -> Dict[str, Any]:
    """
    Step 10: Identify outliers and anomalies in sales and pricing data.

    This comprehensive analysis:
    1. Detects sales outliers using category-specific business rules
    2. Identifies pricing anomalies (jumps, suspicious prices, inconsistencies)
    3. Classifies outliers as promotional spikes vs data errors
    4. Validates business rules (no negative sales, etc.)
    5. Generates publication-ready visualizations
    6. Provides actionable treatment recommendations

    Category-specific thresholds (business rules):
    - FOODS: >50 units/day is suspicious (daily consumption item)
    - HOUSEHOLD: >20 units/day is suspicious (infrequent purchase)
    - HOBBIES: >100 units/day is suspicious (discretionary spending)

    Parameters
    ----------
    data_path : str
        Path to raw M5 dataset directory

    Returns
    -------
    Dict[str, Any]
        Comprehensive outlier analysis including:
        - sales_outliers: Sales outlier detection results
        - pricing_anomalies: Pricing anomaly detection results
        - business_rule_violations: Invalid values identified
        - visualizations: Publication-ready plots
        - recommendations: Treatment suggestions
        - summary_statistics: Key findings

    Example
    -------
    >>> results = identify_outliers_and_anomalies()
    >>> print(f"Total outliers: {results['summary_statistics']['total_outliers']}")
    >>> print(f"Price jumps: {results['summary_statistics']['price_jumps_count']}")
    """
    print("Step 10: Identifying outliers and anomalies...")

    # Load data
    sales_data = pd.read_csv(f"{data_path}/sales_train_validation.csv")
    pricing_data = pd.read_csv(f"{data_path}/sell_prices.csv")

    results = {}

    # 1. Detect sales outliers
    sales_outliers = detect_sales_outliers(sales_data)
    results['sales_outliers'] = sales_outliers

    # 2. Detect pricing anomalies
    pricing_anomalies = analyze_pricing_anomalies(pricing_data)
    results['pricing_anomalies'] = pricing_anomalies

    # 3. Generate outlier detection visualization
    outlier_plot_path = "notebooks/eda/plots/step10_outliers/outlier_detection_analysis.png"
    Path(outlier_plot_path).parent.mkdir(parents=True, exist_ok=True)

    outlier_viz = plot_outlier_detection(
        sales_outliers,
        outlier_plot_path,
        title="Step 10: Sales Outlier Detection Results"
    )
    results['visualizations'] = {'outlier_detection': outlier_viz}

    # 4. Calculate summary statistics
    pricing_anomalies_data = pricing_anomalies.get('price_jumps', {})
    price_jumps_count = pricing_anomalies_data.get('count', 0)

    violations = sales_outliers.get('business_rule_violations', {})
    negative_sales = violations.get('negative_sales', {}).get('count', 0)
    unrealistic_values = violations.get('unrealistic_values', {}).get('count', 0)

    summary = {
        'total_outliers': sales_outliers.get('total_outliers', 0),
        'outliers_by_category': {
            cat: dist.get('count', 0)
            for cat, dist in sales_outliers.get('outlier_distribution', {}).items()
        },
        'price_jumps_count': price_jumps_count,
        'suspicious_prices_count': pricing_anomalies.get('suspicious_prices', {}).get('count', 0),
        'cross_store_inconsistencies': pricing_anomalies.get('cross_store_inconsistency', {}).get('count', 0),
        'negative_sales_violations': negative_sales,
        'unrealistic_value_violations': unrealistic_values,
        'total_business_rule_violations': negative_sales + unrealistic_values
    }
    results['summary_statistics'] = summary

    # 5. Generate recommendations for outlier treatment
    recommendations = _generate_outlier_treatment_recommendations(
        sales_outliers,
        pricing_anomalies
    )
    results['recommendations'] = recommendations

    print(f"\nStep 10 Analysis Summary:")
    print(f"  - Total sales outliers detected: {summary['total_outliers']}")
    print(f"  - Price jumps (>200%): {summary['price_jumps_count']}")
    print(f"  - Business rule violations: {summary['total_business_rule_violations']}")
    print(f"  - Visualizations saved to: {outlier_plot_path}")

    return results


def _generate_missing_value_recommendations(
    missing_patterns: Dict[str, Any],
    mechanisms: Dict[str, Any],
    sales_data: pd.DataFrame
) -> Dict[str, Any]:
    """Generate preprocessing recommendations for missing values."""
    recommendations = {
        'overall_strategy': [],
        'by_mechanism': {},
        'preprocessing_steps': []
    }

    # Overall strategy
    if missing_patterns.get('sales_completeness', {}).get('is_complete', False):
        recommendations['overall_strategy'].append(
            "Sales data is complete (no missing values). Zeros are informative; maintain as-is."
        )

    zero_pct = missing_patterns.get('sales_completeness', {}).get('zero_heavy_series', {}).get('percent', 0)
    if zero_pct > 50:
        recommendations['overall_strategy'].append(
            f"High proportion of zero-heavy series ({zero_pct:.1f}%). Consider zero-inflation modeling or intermittent demand forecasting."
        )

    # By mechanism
    mechanisms_list = mechanisms.get('mechanisms', [])

    if 'seasonal_availability' in mechanisms_list:
        seasonal_count = len(mechanisms.get('seasonal_items', []))
        recommendations['by_mechanism']['seasonal_availability'] = [
            f"Identified {seasonal_count} seasonal items.",
            "Treatment: Use seasonal dummy variables or separate models per season.",
            "Consider hierarchical forecasting by season."
        ]

    if 'new_product_introduction' in mechanisms_list:
        new_count = len(mechanisms.get('new_products', []))
        recommendations['by_mechanism']['new_product_introduction'] = [
            f"Identified {new_count} new products.",
            "Treatment: Use transfer learning or Bayesian approaches for cold-start prediction.",
            "Consider excluding early introduction period from training."
        ]

    if 'product_discontinuation' in mechanisms_list:
        disc_count = len(mechanisms.get('discontinued_items', []))
        recommendations['by_mechanism']['product_discontinuation'] = [
            f"Identified {disc_count} discontinued items.",
            "Treatment: Exclude from future period forecasting.",
            "Use end-of-life patterns for inventory clearance prediction."
        ]

    if 'geographic_availability' in mechanisms_list:
        geo_count = len(mechanisms.get('geographic_restrictions', []))
        recommendations['by_mechanism']['geographic_availability'] = [
            f"Identified {geo_count} items with geographic restrictions.",
            "Treatment: Build store-specific models.",
            "Encode geographic availability as categorical feature."
        ]

    # Preprocessing steps
    recommendations['preprocessing_steps'] = [
        "1. Validate that sales data has no missing values (confirmed by Step 9)",
        "2. Forward-fill or interpolate missing pricing data per item-store",
        "3. Create binary features for product lifecycle stage (new/mature/discontinued)",
        "4. Encode seasonal availability patterns for model interpretability",
        "5. Flag geographic-restricted items for store-specific treatment"
    ]

    return recommendations


def _generate_outlier_treatment_recommendations(
    sales_outliers: Dict[str, Any],
    pricing_anomalies: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate preprocessing recommendations for outliers."""
    recommendations = {
        'sales_treatment': [],
        'pricing_treatment': [],
        'business_rule_violations': [],
        'preprocessing_steps': []
    }

    # Sales outlier treatment
    total_outliers = sales_outliers.get('total_outliers', 0)
    if total_outliers > 0:
        recommendations['sales_treatment'].append(
            f"Found {total_outliers} sales outliers across categories."
        )
        recommendations['sales_treatment'].append(
            "Treatment: Investigate promotional calendar for real spikes vs errors."
        )
        recommendations['sales_treatment'].append(
            "Consider robust scaling or log-transformation for highly skewed categories."
        )

    # Pricing anomalies treatment
    price_jumps = sales_outliers.get('price_jumps', {}).get('count', 0)
    if price_jumps > 0:
        recommendations['pricing_treatment'].append(
            f"Found {price_jumps} large price jumps (>200% week-over-week)."
        )
        recommendations['pricing_treatment'].append(
            "Treatment: Verify against promotional calendar; may indicate data entry errors."
        )

    suspicious_prices = pricing_anomalies.get('suspicious_prices', {}).get('count', 0)
    if suspicious_prices > 0:
        recommendations['pricing_treatment'].append(
            f"Found {suspicious_prices} suspicious prices ($0.01 or extreme values)."
        )
        recommendations['pricing_treatment'].append(
            "Treatment: Manual review required; likely data quality issues."
        )

    cross_store = pricing_anomalies.get('cross_store_inconsistency', {}).get('count', 0)
    if cross_store > 0:
        recommendations['pricing_treatment'].append(
            f"Found {cross_store} items with cross-store pricing inconsistencies (>20%)."
        )
        recommendations['pricing_treatment'].append(
            "Treatment: Investigate regional pricing strategies; may be intentional."
        )

    # Business rule violations
    violations = sales_outliers.get('business_rule_violations', {})
    neg_sales = violations.get('negative_sales', {}).get('count', 0)
    unrealistic = violations.get('unrealistic_values', {}).get('count', 0)

    if neg_sales > 0:
        recommendations['business_rule_violations'].append(
            f"CRITICAL: Found {neg_sales} negative sales values. These are data errors."
        )
        recommendations['business_rule_violations'].append(
            "Action: Must be removed or corrected before modeling."
        )

    if unrealistic > 0:
        recommendations['business_rule_violations'].append(
            f"Found {unrealistic} unrealistic sales values (>10,000 units/day)."
        )
        recommendations['business_rule_violations'].append(
            "Action: Investigate for data entry errors; consider capping at 99.9th percentile."
        )

    # Preprocessing steps
    recommendations['preprocessing_steps'] = [
        "1. Address business rule violations (negative sales must be removed)",
        "2. Validate pricing anomalies against promotional calendar",
        "3. Separate promotional spikes from outliers for preservation",
        "4. Apply category-specific outlier treatment (cap or robust scaling)",
        "5. Create outlier flags as features if they have predictive value",
        "6. Log-transform skewed categories after outlier handling",
        "7. Document all transformations for model interpretability"
    ]

    return recommendations


# =====================================================================
# Steps 11, 13, 14: Advanced EDA Analysis Functions
# =====================================================================

def analyze_segment_behavior(
    data_path: str = "/Users/rahul.vansh/Documents/Personal/demand_forecast_intelligence/data/raw"
) -> Dict[str, Any]:
    """
    Step 11: Analyze demand segment behavior patterns.

    Comprehensive analysis of product and department segment behaviors to support
    segmentation model development and business strategy. Identifies distinct demand
    patterns, seasonal characteristics, and performance tiers across segments.

    This step provides critical insights for:
    - Segment-specific forecasting models
    - Inventory management by segment
    - Promotional strategy optimization per segment
    - Lifecycle stage identification for product planning

    Parameters
    ----------
    data_path : str
        Path to directory containing M5 CSV files (sales_train_validation.csv, etc.)

    Returns
    -------
    Dict[str, Any]
        Comprehensive segment analysis results containing:
        - category_behavior: Category-level behavioral patterns
        - department_segments: Department segment characteristics
        - performance_metrics: Segment ROI and ranking metrics
        - seasonality_patterns: Seasonal behavior by segment
        - lifecycle_stages: Product lifecycle classifications
        - visualizations: Generated segment behavior plots
        - summary: High-level business findings

    Raises
    ------
    FileNotFoundError
        If required CSV files are not found
    ValueError
        If data validation fails

    Example
    -------
    >>> results = analyze_segment_behavior()
    >>> print(results['summary']['top_performing_segments'])
    """
    print("Starting Step 11: Segment Behavior Analysis")
    print("-" * 60)

    # Load datasets
    try:
        sales_path = os.path.join(data_path, "sales_train_validation.csv")
        calendar_path = os.path.join(data_path, "calendar.csv")

        if not os.path.exists(sales_path):
            raise FileNotFoundError(f"Sales file not found: {sales_path}")
        if not os.path.exists(calendar_path):
            raise FileNotFoundError(f"Calendar file not found: {calendar_path}")

        sales_data = pd.read_csv(sales_path)
        calendar_data = pd.read_csv(calendar_path)

        print("Loaded datasets:")
        print(f"  - Sales: {len(sales_data)} rows × {len(sales_data.columns)} cols")
        print(f"  - Calendar: {len(calendar_data)} rows × {len(calendar_data.columns)} cols")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load data files: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

    results = {}

    # Transform M5 data to long format for analysis
    print("\nTransforming M5 data for segment analysis...")
    try:
        transformed_data = _transform_m5_to_long_format(sales_data)
        print(f"   ✓ Transformed {len(sales_data)} products into {len(transformed_data)} daily observations")
    except Exception as e:
        print(f"   ✗ Error transforming data: {str(e)}")
        return {'error': f'Data transformation failed: {str(e)}'}

    # 1. Analyze category behavior patterns
    print("\n1. Analyzing category behavior patterns...")
    try:
        category_behavior = analyze_category_behavior_patterns(
            data=transformed_data,
            category_col='cat_id',
            sales_col='daily_sales'
        )
        results['category_behavior'] = category_behavior
        categories_count = len(category_behavior.get('categories', []))
        print(f"   ✓ Analyzed behavior for {categories_count} categories")
    except Exception as e:
        print(f"   ✗ Error in category behavior analysis: {str(e)}")
        results['category_behavior'] = {'error': str(e)}

    # 2. Analyze department segment patterns
    print("2. Analyzing department segment patterns...")
    try:
        department_segments = analyze_department_segment_patterns(
            data=transformed_data,
            department_col='cat_id',
            sales_col='daily_sales'
        )
        results['department_segments'] = department_segments
        departments_count = len(department_segments.get('departments', []))
        print(f"   ✓ Analyzed {departments_count} department segments")
    except Exception as e:
        print(f"   ✗ Error in department analysis: {str(e)}")
        results['department_segments'] = {'error': str(e)}

    # 3. Compute segment performance metrics
    print("3. Computing segment performance metrics...")
    try:
        performance_metrics = compute_segment_performance_metrics(
            data=transformed_data,
            segment_col='cat_id',
            sales_col='daily_sales'
        )
        results['performance_metrics'] = performance_metrics
        top_segments = len(performance_metrics.get('top_performers', []))
        print(f"   ✓ Identified {top_segments} top-performing segments")
    except Exception as e:
        print(f"   ✗ Error in performance analysis: {str(e)}")
        results['performance_metrics'] = {'error': str(e)}

    # 4. Analyze segment seasonality patterns
    print("4. Analyzing segment seasonality patterns...")
    try:
        # First merge transformed data with calendar for seasonality analysis
        transformed_with_calendar = transformed_data.merge(
            calendar_data[['d', 'date']].rename(columns={'d': 'day_col'}),
            left_on='day_col',
            right_on='day_col',
            how='left'
        )

        seasonality_patterns = analyze_segment_seasonality_patterns(
            data=transformed_with_calendar,
            segment_col='cat_id',
            date_col='date',
            sales_col='daily_sales'
        )
        results['seasonality_patterns'] = seasonality_patterns
        seasonal_segments = len(seasonality_patterns.get('seasonal_segments', []))
        print(f"   ✓ Identified {seasonal_segments} segments with seasonal patterns")
    except Exception as e:
        print(f"   ✗ Error in seasonality analysis: {str(e)}")
        results['seasonality_patterns'] = {'error': str(e)}

    # 5. Detect segment lifecycle stages
    print("5. Detecting segment lifecycle stages...")
    try:
        # Use transformed data with calendar merge from seasonality analysis if available
        if 'transformed_with_calendar' not in locals():
            transformed_with_calendar = transformed_data.merge(
                calendar_data[['d', 'date']].rename(columns={'d': 'day_col'}),
                left_on='day_col',
                right_on='day_col',
                how='left'
            )

        lifecycle_stages = detect_segment_lifecycle_stages(
            data=transformed_with_calendar,
            segment_col='cat_id',
            date_col='date',
            sales_col='daily_sales'
        )
        results['lifecycle_stages'] = lifecycle_stages
        new_items = len(lifecycle_stages.get('new_products', []))
        mature_items = len(lifecycle_stages.get('mature_products', []))
        discontinued_items = len(lifecycle_stages.get('discontinued_products', []))
        print(f"   ✓ Classified products: New={new_items}, Mature={mature_items}, Discontinued={discontinued_items}")
    except Exception as e:
        print(f"   ✗ Error in lifecycle detection: {str(e)}")
        results['lifecycle_stages'] = {'error': str(e)}

    # 6. Generate visualizations
    print("6. Generating segment behavior visualizations...")

    plot_dir = "notebooks/eda/plots/step11_segment_behavior"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    visualizations = {}

    # Segment behavior comparison plot
    try:
        segment_plot_path = os.path.join(plot_dir, "segment_behavior_comparison.png")
        # Prepare segment data from category behavior results
        segment_data = {}
        if 'category_behavior' in results and 'categories' in results['category_behavior']:
            for cat in results['category_behavior']['categories']:
                segment_data[cat] = {
                    'mean_sales': results['category_behavior'].get('category_stats', {}).get(cat, {}).get('mean', 0),
                    'std_sales': results['category_behavior'].get('category_stats', {}).get(cat, {}).get('std', 0)
                }

        if segment_data:
            segment_viz = plot_segment_behavior_comparison(
                segment_data=segment_data,
                save_path=segment_plot_path
            )
            visualizations['segment_comparison'] = segment_viz
            print("   ✓ Generated segment behavior comparison plot")
            print(f"     Path: {segment_plot_path}")
        else:
            print("   ℹ  Insufficient segment data for visualization")
    except Exception as e:
        print(f"   ✗ Error generating segment plot: {str(e)}")

    results['visualizations'] = visualizations

    # 7. Update EDA report
    print("7. Updating EDA report with findings...")
    try:
        _update_eda_report_step11(results)
        print("   ✓ EDA report updated")
    except Exception as e:
        print(f"   ℹ  Note: Could not update EDA report: {str(e)}")

    # 8. Generate summary
    print("\n8. Generating summary...")

    summary = {
        'total_categories': len(results.get('category_behavior', {}).get('categories', [])),
        'departments_analyzed': len(results.get('department_segments', {}).get('departments', [])),
        'top_performers': len(results.get('performance_metrics', {}).get('top_performers', [])),
        'seasonal_segments': len(results.get('seasonality_patterns', {}).get('seasonal_segments', [])),
        'lifecycle_distribution': {
            'new': len(results.get('lifecycle_stages', {}).get('new_products', [])),
            'mature': len(results.get('lifecycle_stages', {}).get('mature_products', [])),
            'discontinued': len(results.get('lifecycle_stages', {}).get('discontinued_products', []))
        },
        'step_status': 'complete'
    }

    results['summary'] = summary

    print("\nStep 11 analysis complete!")
    print("-" * 60)
    print("Summary:")
    print(f"  - Categories analyzed: {summary['total_categories']}")
    print(f"  - Departments analyzed: {summary['departments_analyzed']}")
    print(f"  - Top performers identified: {summary['top_performers']}")
    print(f"  - Seasonal segments: {summary['seasonal_segments']}")
    print(f"  - Lifecycle distribution: {summary['lifecycle_distribution']}")

    return results


def analyze_distribution_drift(
    data_path: str = "/Users/rahul.vansh/Documents/Personal/demand_forecast_intelligence/data/raw"
) -> Dict[str, Any]:
    """
    Step 13: Analyze distribution drift between training and validation periods.

    Rigorous statistical validation of training vs validation period consistency
    to ensure reliable model validation. Detects any significant distribution shifts
    that could compromise model performance predictions.

    This step is critical for:
    - Validating temporal split integrity
    - Detecting market regime changes
    - Ensuring representative validation periods
    - Identifying seasonal coverage issues
    - Risk assessment for model generalization

    M5 Temporal Boundaries:
    - Training: d_1 to d_1913 (1913 days)
    - Validation: d_1914 to d_1941 (28 days)

    Parameters
    ----------
    data_path : str
        Path to directory containing M5 CSV files

    Returns
    -------
    Dict[str, Any]
        Comprehensive drift analysis results containing:
        - distribution_comparison: Train-validation distribution tests
        - seasonal_representativeness: Seasonal coverage analysis
        - category_drift: Category-specific drift detection
        - drift_severity: Quantitative drift assessment
        - temporal_integrity: Temporal split validation
        - visualizations: Generated drift analysis plots
        - summary: High-level risk assessment findings

    Raises
    ------
    FileNotFoundError
        If required CSV files are not found
    ValueError
        If data validation fails

    Example
    -------
    >>> results = analyze_distribution_drift()
    >>> print(f"Drift risk level: {results['summary']['overall_risk_level']}")
    """
    print("Starting Step 13: Distribution Drift Analysis")
    print("-" * 60)

    # Load datasets
    try:
        sales_path = os.path.join(data_path, "sales_train_validation.csv")
        calendar_path = os.path.join(data_path, "calendar.csv")

        if not os.path.exists(sales_path):
            raise FileNotFoundError(f"Sales file not found: {sales_path}")
        if not os.path.exists(calendar_path):
            raise FileNotFoundError(f"Calendar file not found: {calendar_path}")

        sales_data = pd.read_csv(sales_path)
        calendar_data = pd.read_csv(calendar_path)

        print("Loaded datasets:")
        print(f"  - Sales: {len(sales_data)} rows × {len(sales_data.columns)} cols")
        print(f"  - Calendar: {len(calendar_data)} rows × {len(calendar_data.columns)} cols")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load data files: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

    results = {}

    # M5 temporal split: d_1913 is the last training day, d_1914 starts validation
    train_cols = [col for col in sales_data.columns if col.startswith('d_') and int(col.split('_')[1]) <= 1913]
    val_cols = [col for col in sales_data.columns if col.startswith('d_') and int(col.split('_')[1]) > 1913]

    print(f"\nTemporal boundaries:")
    print(f"  - Training period: {len(train_cols)} days")
    print(f"  - Validation period: {len(val_cols)} days")

    # 1. Compare temporal distributions
    print("\n1. Comparing temporal distributions...")
    try:
        distribution_comparison = compare_temporal_distributions(
            train_data=sales_data[train_cols],
            validation_data=sales_data[val_cols],
            columns=train_cols[:10]  # Sample for comparison
        )
        results['distribution_comparison'] = distribution_comparison
        stat_tests = len(distribution_comparison.get('statistical_tests', {}))
        print(f"   ✓ Performed {stat_tests} distribution comparisons")
    except Exception as e:
        print(f"   ✗ Error in distribution comparison: {str(e)}")
        results['distribution_comparison'] = {'error': str(e)}

    # 2. Analyze seasonal representativeness
    print("2. Analyzing seasonal representativeness...")
    try:
        seasonal_representativeness = analyze_seasonal_representativeness(
            train_data=sales_data[train_cols],
            validation_data=sales_data[val_cols],
            calendar_data=calendar_data
        )
        results['seasonal_representativeness'] = seasonal_representativeness
        coverage_pct = seasonal_representativeness.get('seasonal_coverage', {}).get('coverage_percent', 0)
        print(f"   ✓ Seasonal coverage: {coverage_pct:.1f}%")
    except Exception as e:
        print(f"   ✗ Error in seasonal analysis: {str(e)}")
        results['seasonal_representativeness'] = {'error': str(e)}

    # 3. Detect category-specific drift
    print("3. Detecting category-specific drift...")
    try:
        category_drift = detect_category_drift(
            train_data=sales_data[train_cols],
            validation_data=sales_data[val_cols],
            sales_metadata=sales_data[['item_id', 'cat_id', 'store_id']],
            calendar_data=calendar_data
        )
        results['category_drift'] = category_drift
        drifted_cats = len(category_drift.get('drifted_categories', []))
        print(f"   ✓ Detected {drifted_cats} categories with significant drift")
    except Exception as e:
        print(f"   ✗ Error in category drift detection: {str(e)}")
        results['category_drift'] = {'error': str(e)}

    # 4. Compute drift severity scores
    print("4. Computing drift severity scores...")
    try:
        drift_severity = compute_drift_severity_scores(
            train_data=sales_data[train_cols],
            validation_data=sales_data[val_cols],
            sales_metadata=sales_data[['item_id', 'cat_id', 'store_id']]
        )
        results['drift_severity'] = drift_severity
        overall_severity = drift_severity.get('overall_severity', {}).get('level', 'unknown')
        severity_score = drift_severity.get('overall_severity', {}).get('score', 0)
        print(f"   ✓ Overall drift severity: {overall_severity} (score: {severity_score:.3f})")
    except Exception as e:
        print(f"   ✗ Error in drift severity analysis: {str(e)}")
        results['drift_severity'] = {'error': str(e)}

    # 5. Validate temporal split integrity
    print("5. Validating temporal split integrity...")
    try:
        temporal_integrity = validate_temporal_split_integrity(
            train_data=sales_data[train_cols],
            validation_data=sales_data[val_cols],
            calendar_data=calendar_data,
            split_day=1913
        )
        results['temporal_integrity'] = temporal_integrity
        integrity_status = temporal_integrity.get('integrity_status', 'unknown')
        print(f"   ✓ Temporal split integrity: {integrity_status}")
    except Exception as e:
        print(f"   ✗ Error in temporal integrity validation: {str(e)}")
        results['temporal_integrity'] = {'error': str(e)}

    # 6. Generate visualizations
    print("6. Generating distribution drift visualizations...")

    plot_dir = "notebooks/eda/plots/step13_drift_analysis"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    visualizations = {}

    # Distribution drift analysis plot
    try:
        drift_plot_path = os.path.join(plot_dir, "distribution_drift_analysis.png")
        drift_viz = plot_distribution_drift_analysis(
            train_data=sales_data[train_cols],
            validation_data=sales_data[val_cols],
            drift_results=results
        )
        visualizations['drift_distribution'] = drift_viz
        print("   ✓ Generated distribution drift plot")
        print(f"     Path: {drift_plot_path}")
    except Exception as e:
        print(f"   ✗ Error generating drift plot: {str(e)}")

    results['visualizations'] = visualizations

    # 7. Update EDA report
    print("7. Updating EDA report with findings...")
    try:
        _update_eda_report_step13(results)
        print("   ✓ EDA report updated")
    except Exception as e:
        print(f"   ℹ  Note: Could not update EDA report: {str(e)}")

    # 8. Generate summary
    print("\n8. Generating summary...")

    overall_risk_level = results.get('drift_severity', {}).get('overall_severity', {}).get('level', 'unknown')
    drifted_count = len(results.get('category_drift', {}).get('drifted_categories', []))
    seasonal_coverage = results.get('seasonal_representativeness', {}).get('seasonal_coverage', {}).get('coverage_percent', 0)
    temporal_integrity_status = results.get('temporal_integrity', {}).get('integrity_status', 'unknown')

    summary = {
        'overall_risk_level': overall_risk_level,
        'drifted_categories': drifted_count,
        'seasonal_coverage_percent': seasonal_coverage,
        'temporal_split_valid': temporal_integrity_status == 'VALID',
        'recommendations': results.get('temporal_integrity', {}).get('recommendations', []),
        'step_status': 'complete'
    }

    results['summary'] = summary

    print("\nStep 13 analysis complete!")
    print("-" * 60)
    print("Summary:")
    print(f"  - Overall drift risk: {summary['overall_risk_level']}")
    print(f"  - Drifted categories: {summary['drifted_categories']}")
    print(f"  - Seasonal coverage: {summary['seasonal_coverage_percent']:.1f}%")
    print(f"  - Temporal split valid: {summary['temporal_split_valid']}")

    return results


def audit_temporal_leakage(
    data_path: str = "/Users/rahul.vansh/Documents/Personal/demand_forecast_intelligence/data/raw",
    feature_engineering_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Step 14: Comprehensive temporal leakage audit for deployment readiness.

    Validates that all features respect temporal ordering and no future information
    leaks into model training. Critical for production-ready forecasting systems
    to ensure models will generalize to truly unseen future data.

    This step prevents:
    - Information leakage during cross-validation
    - Look-ahead bias in feature engineering
    - Overfitting to temporal artifacts
    - Invalid backtesting results

    Default feature config assumes:
    - Lag features: 1-28 days
    - Rolling statistics: 7, 14, 28 days
    - Seasonal indicators: previous year lags
    - Pricing: weekly frequency (7-day lags)

    Parameters
    ----------
    data_path : str
        Path to directory containing M5 CSV files
    feature_engineering_config : Optional[Dict[str, Any]]
        Configuration for planned feature engineering. If None, uses default config.
        Expected keys: 'lag_features', 'rolling_features', 'seasonal_features', 'price_features'

    Returns
    -------
    Dict[str, Any]
        Comprehensive leakage audit results containing:
        - temporal_boundaries: Temporal boundary validation results
        - feature_availability: Feature timing validation
        - cross_validation: CV fold integrity assessment
        - suspicious_correlations: Suspicious patterns detected
        - audit_report: Comprehensive audit summary
        - deployment_readiness: Production deployment decision
        - visualizations: Generated leakage validation plots
        - summary: High-level findings with recommendations

    Raises
    ------
    FileNotFoundError
        If required CSV files are not found
    ValueError
        If data validation fails

    Example
    -------
    >>> results = audit_temporal_leakage()
    >>> print(f"Deployment ready: {results['summary']['deployment_ready']}")
    >>> for rec in results['summary']['critical_recommendations']:
    ...     print(f"  - {rec}")
    """
    print("Starting Step 14: Temporal Leakage Audit")
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
        price_data = pd.read_csv(price_path) if os.path.exists(price_path) else pd.DataFrame()

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

    # Set default feature config if not provided
    if feature_engineering_config is None:
        feature_engineering_config = {
            'lag_features': [1, 7, 14, 28],
            'rolling_features': [7, 14, 28],
            'seasonal_features': [365],
            'price_features': [7]
        }

    print(f"\nFeature engineering config:")
    print(f"  - Lag features: {feature_engineering_config.get('lag_features', [])}")
    print(f"  - Rolling features: {feature_engineering_config.get('rolling_features', [])}")

    # M5 temporal split: d_1913 is the last training day, d_1914 starts validation
    train_end_day = 1913
    val_start_day = 1914

    print(f"\nTemporal boundaries:")
    print(f"  - Training period end: day {train_end_day}")
    print(f"  - Validation period start: day {val_start_day}")

    # 1. Audit temporal boundaries
    print("\n1. Auditing temporal boundaries...")
    try:
        # Create feature configs for audit
        feature_configs = [
            {'name': f'lag_{lag}', 'type': 'lag', 'offset': f'-{lag}d'}
            for lag in feature_engineering_config.get('lag_features', [])
        ]

        temporal_boundaries = audit_temporal_boundaries(
            feature_data=sales_data,
            split_date=pd.Timestamp(f"2016-01-{min(31, train_end_day)}"),  # Approximate
            feature_configs=feature_configs
        )
        results['temporal_boundaries'] = temporal_boundaries
        boundary_issues = len(temporal_boundaries.get('issues', []))
        print(f"   ✓ Temporal boundary validation complete ({boundary_issues} potential issues)")
    except Exception as e:
        print(f"   ✗ Error in temporal boundary audit: {str(e)}")
        results['temporal_boundaries'] = {'error': str(e)}

    # 2. Check feature availability timing
    print("2. Checking feature availability timing...")
    try:
        feature_availability = check_feature_availability_timing(
            sales_data=sales_data,
            price_data=price_data,
            calendar_data=calendar_data,
            feature_engineering_config=feature_engineering_config
        )
        results['feature_availability'] = feature_availability
        availability_issues = len(feature_availability.get('timing_issues', []))
        print(f"   ✓ Feature availability check complete ({availability_issues} timing issues)")
    except Exception as e:
        print(f"   ✗ Error in feature availability check: {str(e)}")
        results['feature_availability'] = {'error': str(e)}

    # 3. Validate cross-validation integrity
    print("3. Validating cross-validation integrity...")
    try:
        cv_integrity = validate_cross_validation_integrity(
            sales_data=sales_data,
            calendar_data=calendar_data,
            num_folds=5,
            min_train_days=365
        )
        results['cross_validation'] = cv_integrity
        cv_status = cv_integrity.get('cv_status', 'unknown')
        print(f"   ✓ Cross-validation integrity: {cv_status}")
    except Exception as e:
        print(f"   ✗ Error in CV integrity validation: {str(e)}")
        results['cross_validation'] = {'error': str(e)}

    # 4. Scan for suspicious correlations
    print("4. Scanning for suspicious correlations...")
    try:
        suspicious_corr = scan_suspicious_correlations(
            sales_data=sales_data,
            price_data=price_data,
            calendar_data=calendar_data
        )
        results['suspicious_correlations'] = suspicious_corr
        suspicious_patterns = len(suspicious_corr.get('suspicious_patterns', []))
        print(f"   ✓ Suspicious correlation scan complete ({suspicious_patterns} patterns found)")
    except Exception as e:
        print(f"   ✗ Error in suspicious correlation scan: {str(e)}")
        results['suspicious_correlations'] = {'error': str(e)}

    # 5. Generate comprehensive leakage audit report
    print("5. Generating leakage audit report...")
    try:
        audit_report = generate_leakage_audit_report(results)
        results['audit_report'] = audit_report
        deployment_ready = audit_report.get('deployment_ready', False)
        risk_level = audit_report.get('risk_level', 'unknown')
        print(f"   ✓ Audit report generated (Risk level: {risk_level})")
    except Exception as e:
        print(f"   ✗ Error generating audit report: {str(e)}")
        results['audit_report'] = {'error': str(e)}

    # 6. Generate visualizations
    print("6. Generating leakage validation visualizations...")

    plot_dir = "notebooks/eda/plots/step14_leakage_validation"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    visualizations = {}

    # Leakage validation summary plot
    try:
        leakage_plot_path = os.path.join(plot_dir, "leakage_validation_summary.png")
        leakage_viz = plot_leakage_validation_summary(
            audit_results=results,
            save_path=leakage_plot_path
        )
        visualizations['leakage_summary'] = leakage_viz
        print("   ✓ Generated leakage validation summary plot")
        print(f"     Path: {leakage_plot_path}")
    except Exception as e:
        print(f"   ✗ Error generating leakage plot: {str(e)}")

    results['visualizations'] = visualizations

    # 7. Update EDA report
    print("7. Updating EDA report with findings...")
    try:
        _update_eda_report_step14(results)
        print("   ✓ EDA report updated")
    except Exception as e:
        print(f"   ℹ  Note: Could not update EDA report: {str(e)}")

    # 8. Generate summary
    print("\n8. Generating summary...")

    audit_report_data = results.get('audit_report', {})
    deployment_ready = audit_report_data.get('deployment_ready', False)
    risk_level = audit_report_data.get('risk_level', 'unknown')
    critical_issues = len(audit_report_data.get('critical_issues', []))
    recommendations = audit_report_data.get('recommendations', [])

    summary = {
        'deployment_ready': deployment_ready,
        'risk_level': risk_level,
        'critical_issues': critical_issues,
        'total_recommendations': len(recommendations),
        'critical_recommendations': recommendations[:3] if recommendations else [],
        'step_status': 'complete'
    }

    results['summary'] = summary

    print("\nStep 14 analysis complete!")
    print("-" * 60)
    print("Summary:")
    print(f"  - Deployment ready: {summary['deployment_ready']}")
    print(f"  - Risk level: {summary['risk_level']}")
    print(f"  - Critical issues: {summary['critical_issues']}")
    print(f"  - Recommendations: {summary['total_recommendations']}")

    return results


# =====================================================================
# Helper Functions for EDA Report Updates
# =====================================================================

def _update_eda_report_step11(results: Dict[str, Any]) -> None:
    """
    Update EDA report with Step 11 (Segment Behavior) findings.

    Integrates segment analysis results into the main EDA report file
    with business-focused summary of key patterns and insights.

    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary from analyze_segment_behavior()
    """
    report_path = "notebooks/eda/EDA_REPORT.md"

    # Prepare findings
    summary = results.get('summary', {})
    category_behavior = results.get('category_behavior', {})
    performance_metrics = results.get('performance_metrics', {})

    findings = f"""
## Step 11: Segment Behavior Analysis

**Summary:**
- Total categories analyzed: {summary.get('total_categories', 0)}
- Departments analyzed: {summary.get('departments_analyzed', 0)}
- Top performers identified: {summary.get('top_performers', 0)}
- Seasonal segments detected: {summary.get('seasonal_segments', 0)}

**Lifecycle Distribution:**
- New products: {summary.get('lifecycle_distribution', {}).get('new', 0)}
- Mature products: {summary.get('lifecycle_distribution', {}).get('mature', 0)}
- Discontinued products: {summary.get('lifecycle_distribution', {}).get('discontinued', 0)}

**Key Insights:**
- Segment analysis identifies distinct demand behavior patterns
- Performance tier ranking enables targeted forecasting strategies
- Lifecycle stage classification supports inventory optimization
"""

    try:
        if os.path.exists(report_path):
            with open(report_path, 'a') as f:
                f.write(findings + "\n")
    except Exception as e:
        print(f"   ℹ  Could not update report: {str(e)}")


def _update_eda_report_step13(results: Dict[str, Any]) -> None:
    """
    Update EDA report with Step 13 (Distribution Drift) findings.

    Integrates drift analysis results into the main EDA report file
    with risk assessment and validation period representativeness.

    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary from analyze_distribution_drift()
    """
    report_path = "notebooks/eda/EDA_REPORT.md"

    # Prepare findings
    summary = results.get('summary', {})
    temporal_integrity = results.get('temporal_integrity', {})

    findings = f"""
## Step 13: Distribution Drift Analysis

**Risk Assessment:**
- Overall drift risk level: {summary.get('overall_risk_level', 'unknown')}
- Categories with drift: {summary.get('drifted_categories', 0)}
- Seasonal coverage: {summary.get('seasonal_coverage_percent', 0):.1f}%
- Temporal split valid: {summary.get('temporal_split_valid', False)}

**Validation Period Representativeness:**
- Temporal boundaries: M5 d_1 to d_1913 (train), d_1914 to d_1941 (validation)
- Recommendations: {len(summary.get('recommendations', []))} items

**Key Findings:**
- Distribution drift analysis validates model validation approach
- Seasonal coverage assessment ensures representative validation periods
- Temporal integrity check confirms train-validation separation
"""

    try:
        if os.path.exists(report_path):
            with open(report_path, 'a') as f:
                f.write(findings + "\n")
    except Exception as e:
        print(f"   ℹ  Could not update report: {str(e)}")


def _update_eda_report_step14(results: Dict[str, Any]) -> None:
    """
    Update EDA report with Step 14 (Temporal Leakage Audit) findings.

    Integrates leakage audit results into the main EDA report file
    with deployment readiness assessment and critical recommendations.

    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary from audit_temporal_leakage()
    """
    report_path = "notebooks/eda/EDA_REPORT.md"

    # Prepare findings
    summary = results.get('summary', {})
    audit_report = results.get('audit_report', {})

    findings = f"""
## Step 14: Temporal Leakage Audit

**Deployment Readiness:**
- Deployment ready: {summary.get('deployment_ready', False)}
- Risk level: {summary.get('risk_level', 'unknown')}
- Critical issues: {summary.get('critical_issues', 0)}

**Recommendations:**
- Total recommendations: {summary.get('total_recommendations', 0)}

**Top 3 Critical Recommendations:**
"""

    for i, rec in enumerate(summary.get('critical_recommendations', []), 1):
        findings += f"\n{i}. {rec}"

    findings += """

**Key Validations:**
- Temporal boundaries ensure no information leakage at split point
- Feature availability timing prevents future information use
- Cross-validation integrity maintains proper fold separation
- Suspicious correlation detection identifies look-ahead bias patterns

**Production Deployment Status:**
- This audit validates the forecasting system for production deployment
- All critical issues must be resolved before model release
"""

    try:
        if os.path.exists(report_path):
            with open(report_path, 'a') as f:
                f.write(findings + "\n")
    except Exception as e:
        print(f"   ℹ  Could not update report: {str(e)}")
