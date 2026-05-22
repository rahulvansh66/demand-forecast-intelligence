"""
Segment analysis utilities for demand forecasting EDA (Steps 11, 13, 14).

Provides functions for analyzing behavioral patterns across product categories
and departments to support segmentation model development.

Functions:
- analyze_category_behavior_patterns: Category behavior across FOODS/HOUSEHOLD/HOBBIES
- analyze_department_segment_patterns: Department-level segment analysis
- compute_segment_performance_metrics: Segment ROI and performance ranking
- analyze_segment_seasonality_patterns: Seasonality detection by segment
- detect_segment_lifecycle_stages: Product lifecycle stage classification
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def analyze_category_behavior_patterns(
    data: pd.DataFrame,
    category_col: str = 'cat_id',
    sales_col: str = 'daily_sales'
) -> Dict[str, Any]:
    """
    Analyze behavioral patterns across product categories.

    Examines demand patterns (mean, variability, intermittency) for FOODS,
    HOUSEHOLD, and HOBBIES categories. Uses statistical hypothesis testing
    to determine significant differences in category behavior patterns.

    Business context: Different product categories exhibit different demand
    characteristics (e.g., FOODS more stable, HOBBIES more volatile). This
    analysis supports category-specific demand forecasting strategies.

    Parameters
    ----------
    data : pd.DataFrame
        Aggregated sales data with category column
    category_col : str
        Column name containing product categories
    sales_col : str
        Column name containing daily sales values

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - behavioral_metrics: Dict of each category with mean, std_dev, cv,
                             intermittency, and variability_class
        - statistical_tests: Kruskal-Wallis test results with interpretation
        - business_interpretation: Text summary of category differences
    """
    if data.empty or len(data) == 0:
        return {
            'behavioral_metrics': {},
            'statistical_tests': {
                'test_name': 'Kruskal-Wallis H-test',
                'p_value': np.nan,
                'interpretation': 'Insufficient data for statistical test'
            },
            'business_interpretation': 'No data available for analysis'
        }

    # Remove NaN values
    data_clean = data[[category_col, sales_col]].dropna()

    if len(data_clean) == 0:
        return {
            'behavioral_metrics': {},
            'statistical_tests': {
                'test_name': 'Kruskal-Wallis H-test',
                'p_value': np.nan,
                'interpretation': 'No valid data after cleaning'
            },
            'business_interpretation': 'No valid data for analysis'
        }

    behavioral_metrics = {}
    category_groups = []

    # Calculate metrics for each category
    for category in data_clean[category_col].unique():
        cat_data = data_clean[data_clean[category_col] == category][sales_col]

        if len(cat_data) == 0:
            continue

        mean = float(cat_data.mean())
        std_dev = float(cat_data.std())

        # Coefficient of variation
        if abs(mean) < 1e-10:
            cv = 0.0 if std_dev == 0 else np.inf
        else:
            cv = float(std_dev / abs(mean))

        # Intermittency score (proportion of near-zero values)
        threshold = mean * 0.01 if mean > 0 else 0
        intermittency = float((cat_data <= threshold).sum() / len(cat_data))

        # Variability classification
        if intermittency < 0.25:
            variability_class = "Smooth" if cv < 0.5 else "Erratic"
        else:
            variability_class = "Intermittent" if cv < 0.5 else "Lumpy"

        behavioral_metrics[category] = {
            'mean_sales': mean,
            'std_dev': std_dev,
            'coefficient_of_variation': cv,
            'intermittency_score': intermittency,
            'variability_class': variability_class,
            'sample_size': len(cat_data)
        }

        category_groups.append(cat_data.values)

    # Perform Kruskal-Wallis H-test for statistical significance
    statistical_tests = {
        'test_name': 'Kruskal-Wallis H-test',
        'p_value': np.nan,
        'interpretation': 'Test not performed'
    }

    if len(category_groups) >= 2:
        try:
            h_stat, p_value = stats.kruskal(*category_groups)
            statistical_tests['h_statistic'] = float(h_stat)
            statistical_tests['p_value'] = float(p_value)

            if p_value < 0.05:
                statistical_tests['interpretation'] = (
                    'Significant differences in category behavior patterns detected (p < 0.05). '
                    'Categories require different forecasting strategies.'
                )
            else:
                statistical_tests['interpretation'] = (
                    'No significant differences in category behavior patterns (p >= 0.05). '
                    'Categories can use unified forecasting approach.'
                )
        except Exception:
            statistical_tests['interpretation'] = 'Insufficient valid data for test'

    # Generate business interpretation
    if behavioral_metrics:
        most_stable = min(
            behavioral_metrics.items(),
            key=lambda x: x[1]['coefficient_of_variation'] if not np.isinf(x[1]['coefficient_of_variation']) else np.inf
        )
        most_volatile = max(
            behavioral_metrics.items(),
            key=lambda x: x[1]['coefficient_of_variation'] if not np.isinf(x[1]['coefficient_of_variation']) else -np.inf
        )

        business_interpretation = (
            f"Category '{most_stable[0]}' exhibits most stable demand ({most_stable[1]['variability_class']}), "
            f"while '{most_volatile[0]}' is most volatile ({most_volatile[1]['variability_class']}). "
            f"Recommend category-specific inventory policies and forecasting techniques."
        )
    else:
        business_interpretation = "Insufficient data for business interpretation"

    return {
        'behavioral_metrics': behavioral_metrics,
        'statistical_tests': statistical_tests,
        'business_interpretation': business_interpretation
    }


def analyze_department_segment_patterns(
    data: pd.DataFrame,
    department_col: str = 'dept_id',
    sales_col: str = 'daily_sales'
) -> Dict[str, Any]:
    """
    Analyze demand patterns across department segments.

    Compares department-level performance metrics, identifies high/low performers,
    and provides cross-department benchmarking for inventory strategy.

    Business context: Different departments (e.g., grocery, clothing, toys) serve
    different customer needs with varying demand characteristics. Department-level
    segmentation enables targeted supply chain optimization.

    Parameters
    ----------
    data : pd.DataFrame
        Sales data with department identifier
    department_col : str
        Column name containing department identifiers
    sales_col : str
        Column name containing daily sales values

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - department_metrics: Per-department mean, std_dev, cv, sample_size
        - cross_department_comparison: Ranking and performance gaps
        - segment_recommendations: Actionable insights for each department
    """
    if data.empty or len(data) == 0:
        return {
            'department_metrics': {},
            'cross_department_comparison': {},
            'segment_recommendations': []
        }

    data_clean = data[[department_col, sales_col]].dropna()

    if len(data_clean) == 0:
        return {
            'department_metrics': {},
            'cross_department_comparison': {},
            'segment_recommendations': []
        }

    department_metrics = {}

    # Calculate metrics for each department
    for dept in data_clean[department_col].unique():
        dept_data = data_clean[data_clean[department_col] == dept][sales_col]

        if len(dept_data) == 0:
            continue

        mean = float(dept_data.mean())
        std_dev = float(dept_data.std())

        # Coefficient of variation
        if abs(mean) < 1e-10:
            cv = 0.0 if std_dev == 0 else np.inf
        else:
            cv = float(std_dev / abs(mean))

        department_metrics[dept] = {
            'mean_sales': mean,
            'std_dev': std_dev,
            'coefficient_of_variation': cv,
            'min_sales': float(dept_data.min()),
            'max_sales': float(dept_data.max()),
            'median_sales': float(dept_data.median()),
            'sample_size': len(dept_data)
        }

    # Cross-department comparison
    cross_comparison = {}
    if department_metrics:
        avg_sales = [m['mean_sales'] for m in department_metrics.values()]
        overall_avg = np.mean(avg_sales)

        cross_comparison['overall_average'] = float(overall_avg)
        cross_comparison['high_performers'] = [
            dept for dept, metrics in department_metrics.items()
            if metrics['mean_sales'] > overall_avg * 1.2
        ]
        cross_comparison['low_performers'] = [
            dept for dept, metrics in department_metrics.items()
            if metrics['mean_sales'] < overall_avg * 0.8
        ]

    # Segment recommendations
    recommendations = []
    for dept, metrics in department_metrics.items():
        cv = metrics['coefficient_of_variation']
        if np.isinf(cv):
            cv_class = "Highly variable"
            recommendation = "Requires specialized forecasting; consider manual review"
        elif cv < 0.3:
            cv_class = "Stable"
            recommendation = "Suitable for automated forecasting; consider just-in-time inventory"
        elif cv < 0.7:
            cv_class = "Moderate variability"
            recommendation = "Standard forecasting with safety stock; monitor demand trends"
        else:
            cv_class = "High variability"
            recommendation = "Use robust forecasting; maintain elevated safety stock"

        recommendations.append({
            'department': dept,
            'variability_class': cv_class,
            'mean_daily_sales': metrics['mean_sales'],
            'recommendation': recommendation
        })

    return {
        'department_metrics': department_metrics,
        'cross_department_comparison': cross_comparison,
        'segment_recommendations': recommendations
    }


def compute_segment_performance_metrics(
    data: pd.DataFrame,
    segment_col: str = 'segment_id',
    sales_col: str = 'daily_sales'
) -> Dict[str, Any]:
    """
    Compute performance metrics and ranking for segments.

    Calculates revenue contribution, growth stability, and ROI-related metrics
    for each segment. Enables prioritization for inventory investment and
    marketing focus.

    Business context: Segments contribute differently to total revenue and exhibit
    different growth trajectories. Performance ranking guides inventory allocation
    and promotional budget distribution.

    Parameters
    ----------
    data : pd.DataFrame
        Sales data with segment identifiers
    segment_col : str
        Column name containing segment identifiers
    sales_col : str
        Column name containing daily sales values

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - performance_metrics: Total volume, avg daily, revenue %, growth metrics
        - segment_ranking: Ranked list of segments by performance
        - business_insights: ROI and allocation recommendations
    """
    if data.empty or len(data) == 0:
        return {
            'performance_metrics': {},
            'segment_ranking': [],
            'business_insights': 'No data available'
        }

    data_clean = data[[segment_col, sales_col]].dropna()

    if len(data_clean) == 0:
        return {
            'performance_metrics': {},
            'segment_ranking': [],
            'business_insights': 'No valid data after cleaning'
        }

    performance_metrics = {}
    total_sales = data_clean[sales_col].sum()

    # Calculate metrics for each segment
    for segment in data_clean[segment_col].unique():
        seg_data = data_clean[data_clean[segment_col] == segment][sales_col]

        if len(seg_data) == 0:
            continue

        total_volume = float(seg_data.sum())
        revenue_percentage = (total_volume / total_sales * 100) if total_sales > 0 else 0

        performance_metrics[segment] = {
            'total_sales_volume': total_volume,
            'average_daily_sales': float(seg_data.mean()),
            'std_dev_daily': float(seg_data.std()),
            'revenue_contribution_percent': revenue_percentage,
            'min_daily_sales': float(seg_data.min()),
            'max_daily_sales': float(seg_data.max()),
            'sample_size': len(seg_data)
        }

    # Create ranking by revenue contribution
    segment_ranking = []
    for segment, metrics in sorted(
        performance_metrics.items(),
        key=lambda x: x[1]['revenue_contribution_percent'],
        reverse=True
    ):
        segment_ranking.append({
            'segment': segment,
            'rank': len(segment_ranking) + 1,
            'revenue_contribution_percent': metrics['revenue_contribution_percent'],
            'total_sales_volume': metrics['total_sales_volume'],
            'average_daily_sales': metrics['average_daily_sales']
        })

    # Business insights
    insights = "Segments ranked by revenue contribution. Top performers warrant increased inventory "
    insights += "investment and promotional focus. Low performers may be candidates for SKU rationalization."

    return {
        'performance_metrics': performance_metrics,
        'segment_ranking': segment_ranking,
        'business_insights': insights
    }


def analyze_segment_seasonality_patterns(
    data: pd.DataFrame,
    segment_col: str = 'segment_id',
    date_col: str = 'date',
    sales_col: str = 'daily_sales'
) -> Dict[str, Any]:
    """
    Analyze seasonality patterns within segments.

    Detects periodic demand variations (weekly, monthly) and quantifies
    seasonal strength. Supports seasonal demand forecasting and inventory
    planning for each segment.

    Business context: Some segments exhibit strong seasonality (e.g., holidays,
    seasonal clothing) while others are relatively stable. Accurate seasonality
    detection improves forecast accuracy and inventory turnover.

    Parameters
    ----------
    data : pd.DataFrame
        Time-series sales data with date and segment columns
    segment_col : str
        Column name containing segment identifiers
    date_col : str
        Column name containing dates
    sales_col : str
        Column name containing daily sales values

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - seasonality_metrics: Seasonal strength, detected periods for each segment
        - detected_patterns: Summary of periodicity findings
        - business_implications: Recommendations for seasonal inventory planning
    """
    if data.empty or len(data) == 0:
        return {
            'seasonality_metrics': {},
            'detected_patterns': [],
            'business_implications': 'No data available'
        }

    # Ensure date column is datetime
    data_copy = data.copy()
    data_copy[date_col] = pd.to_datetime(data_copy[date_col])
    data_clean = data_copy[[segment_col, date_col, sales_col]].dropna()

    if len(data_clean) == 0:
        return {
            'seasonality_metrics': {},
            'detected_patterns': [],
            'business_implications': 'No valid data after cleaning'
        }

    seasonality_metrics = {}
    detected_patterns = []

    # Analyze each segment
    for segment in data_clean[segment_col].unique():
        seg_data = data_clean[data_clean[segment_col] == segment].sort_values(date_col)

        if len(seg_data) < 14:  # Need at least 2 weeks for pattern detection
            continue

        sales_values = seg_data[sales_col].values

        # Calculate autocorrelation at common seasonal lags
        weekly_lag = min(7, len(sales_values) - 1)
        monthly_lag = min(30, len(sales_values) - 1)

        # Autocorrelation calculation
        mean_sales = np.mean(sales_values)
        c0 = np.sum((sales_values - mean_sales) ** 2) / len(sales_values)

        # Weekly autocorrelation
        if weekly_lag < len(sales_values):
            weekly_acf = np.sum(
                (sales_values[:-weekly_lag] - mean_sales) *
                (sales_values[weekly_lag:] - mean_sales)
            ) / len(sales_values)
            weekly_acf = weekly_acf / c0 if c0 > 0 else 0
        else:
            weekly_acf = 0

        # Seasonal strength (simplified)
        seasonal_strength = abs(weekly_acf) if not np.isnan(weekly_acf) else 0

        # Determine if seasonality is detected
        seasonality_detected = seasonal_strength > 0.3

        seasonality_metrics[segment] = {
            'seasonal_strength': float(seasonal_strength),
            'periodicity_detected': seasonality_detected,
            'weekly_autocorrelation': float(weekly_acf),
            'observation_count': len(seg_data),
            'date_range': {
                'start': str(seg_data[date_col].min().date()),
                'end': str(seg_data[date_col].max().date())
            }
        }

        if seasonality_detected:
            detected_patterns.append({
                'segment': segment,
                'pattern_type': 'Weekly' if weekly_acf > 0.3 else 'Complex',
                'strength': 'Strong' if seasonal_strength > 0.5 else 'Moderate',
                'recommendation': 'Use seasonal decomposition in forecasting models'
            })

    # Business implications
    if detected_patterns:
        implications = (
            f"Seasonality detected in {len(detected_patterns)} segments. "
            f"Recommend using seasonal ARIMA or exponential smoothing models. "
            f"Adjust safety stock levels accordingly."
        )
    else:
        implications = "Limited seasonality detected. Standard forecasting methods appropriate."

    return {
        'seasonality_metrics': seasonality_metrics,
        'detected_patterns': detected_patterns,
        'business_implications': implications
    }


def detect_segment_lifecycle_stages(
    data: pd.DataFrame,
    segment_col: str = 'segment_id',
    date_col: str = 'date',
    sales_col: str = 'daily_sales'
) -> Dict[str, Any]:
    """
    Detect product lifecycle stages for segments.

    Classifies segments into introduction, growth, maturity, or decline phases
    based on sales trend analysis. Guides targeted strategies for inventory
    and marketing investment.

    Business context: Product segments evolve through predictable lifecycle stages
    requiring different strategies. Early detection enables proactive inventory
    and SKU management decisions.

    Parameters
    ----------
    data : pd.DataFrame
        Time-series sales data with date and segment columns
    segment_col : str
        Column name containing segment identifiers
    date_col : str
        Column name containing dates
    sales_col : str
        Column name containing daily sales values

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - lifecycle_stages: Detected stage for each segment with confidence
        - stage_characteristics: Growth rate, volatility metrics by stage
        - strategic_recommendations: Actions for each stage (inventory, marketing)
    """
    if data.empty or len(data) == 0:
        return {
            'lifecycle_stages': [],
            'stage_characteristics': [],
            'strategic_recommendations': []
        }

    # Ensure date column is datetime
    data_copy = data.copy()
    data_copy[date_col] = pd.to_datetime(data_copy[date_col])
    data_clean = data_copy[[segment_col, date_col, sales_col]].dropna()

    if len(data_clean) == 0:
        return {
            'lifecycle_stages': [],
            'stage_characteristics': [],
            'strategic_recommendations': []
        }

    lifecycle_stages = []
    all_characteristics = []

    # Analyze each segment
    for segment in data_clean[segment_col].unique():
        seg_data = data_clean[data_clean[segment_col] == segment].sort_values(date_col)

        if len(seg_data) < 10:  # Need at least 10 observations
            continue

        sales_values = seg_data[sales_col].values

        # Divide into early and late periods
        mid_point = len(sales_values) // 2
        early_period = sales_values[:mid_point]
        late_period = sales_values[mid_point:]

        early_mean = np.mean(early_period)
        late_mean = np.mean(late_period)

        # Calculate growth rate
        if early_mean > 0:
            growth_rate = (late_mean - early_mean) / early_mean
        else:
            growth_rate = 0 if late_mean == 0 else np.inf

        # Calculate volatility
        overall_cv = (np.std(sales_values) / np.mean(sales_values)) if np.mean(sales_values) > 0 else 0

        # Determine lifecycle stage
        if growth_rate > 0.2:
            stage_name = 'Growth'
            stage_description = 'Sales increasing, expanding market share'
        elif growth_rate > -0.1:
            stage_name = 'Maturity'
            stage_description = 'Stable sales, established market position'
        else:
            stage_name = 'Decline'
            stage_description = 'Sales decreasing, market saturation'

        if early_mean < late_mean * 0.5 and early_mean < np.mean(sales_values) * 0.3:
            stage_name = 'Introduction'
            stage_description = 'Early-stage product, ramping up sales'

        lifecycle_stages.append({
            'segment': segment,
            'stage_name': stage_name,
            'growth_rate': float(growth_rate),
            'early_period_avg': float(early_mean),
            'late_period_avg': float(late_mean),
            'stage_description': stage_description
        })

        # Stage characteristics
        all_characteristics.append({
            'characteristic_name': f"{segment}_{stage_name}",
            'stage': stage_name,
            'segment': segment,
            'volatility': 'High' if overall_cv > 0.5 else 'Moderate' if overall_cv > 0.2 else 'Low',
            'trend': 'Increasing' if growth_rate > 0 else 'Decreasing' if growth_rate < -0.05 else 'Stable'
        })

    # Strategic recommendations by stage
    strategic_recommendations = []
    stage_strategies = {
        'Introduction': {
            'inventory': 'Conservative, test demand levels, frequent replenishment',
            'marketing': 'Heavy promotion, customer acquisition focus',
            'forecasting': 'Use judgmental methods, monitor closely'
        },
        'Growth': {
            'inventory': 'Increase stock levels, plan for rapid scaling',
            'marketing': 'Maintain promotion, build brand loyalty',
            'forecasting': 'Upgrade forecasting models as patterns emerge'
        },
        'Maturity': {
            'inventory': 'Optimize stock levels, implement just-in-time',
            'marketing': 'Promotional support, maintain market share',
            'forecasting': 'Use standard forecasting, focus on efficiency'
        },
        'Decline': {
            'inventory': 'Reduce stock, manage aging inventory',
            'marketing': 'Selective promotion, consider discontinuation',
            'forecasting': 'Prepare for phase-out, reduce forecast accuracy requirements'
        }
    }

    for stage in ['Introduction', 'Growth', 'Maturity', 'Decline']:
        strategic_recommendations.append({
            'lifecycle_stage': stage,
            'inventory_strategy': stage_strategies[stage]['inventory'],
            'marketing_strategy': stage_strategies[stage]['marketing'],
            'forecasting_approach': stage_strategies[stage]['forecasting']
        })

    return {
        'lifecycle_stages': lifecycle_stages,
        'stage_characteristics': all_characteristics,
        'strategic_recommendations': strategic_recommendations
    }
