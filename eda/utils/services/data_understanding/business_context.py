"""
Business Context Service for EDA Step 1: Business Problem Understanding.

This service implements the first step of the EDA framework, focusing on defining
the business problem clearly and establishing critical constraints for data leakage prevention.

Framework Correlation:
- EDA Step 1: "Understand the problem first"
- Focus: Business objectives, available features, and leakage prevention
- Output: Clear problem definition with visualization and business impact analysis
"""

from typing import Any, Dict, List
import matplotlib.pyplot as plt
from datetime import datetime

from utils.core.base_service import BaseEDAService
from utils.core.context import EDAContext


class BusinessContextService(BaseEDAService):
    """
    Service for analyzing business context and problem definition.

    This service implements EDA Step 1 by:
    1. Defining clear forecasting and segmentation objectives
    2. Identifying available vs forbidden features at prediction time
    3. Establishing data leakage prevention rules
    4. Creating problem definition timeline visualization
    5. Generating comprehensive business impact analysis

    The service is critical for preventing data leakage by establishing temporal
    boundaries and feature availability constraints upfront.
    """

    def __init__(self, ctx: EDAContext) -> None:
        """Initialize with EDA context."""
        super().__init__(ctx)

    def analyze_problem_definition(self) -> Dict[str, Any]:
        """
        Analyze and define the core business problems.

        Returns:
            Dictionary containing forecasting and segmentation objectives
        """
        # Safe access to config with null check
        m5_config = {}
        if self.ctx.config is not None:
            m5_config = self.ctx.config.get("m5_specifics", {})

        forecasting_objective = {
            "target_variable": "unit_sales",
            "problem_type": "multivariate_time_series_forecasting",
            "forecast_horizon": m5_config.get("forecast_horizon", 28),
            "granularity": "item_store_daily",
            "prediction_frequency": "daily",
            "evaluation_metrics": ["WRMSSE", "RMSE", "MAE", "MAPE"],
            "business_goal": "Optimize inventory planning and reduce stockouts/overstock",
            "success_criteria": {
                "primary": "WRMSSE < 0.5 (competition baseline)",
                "operational": "Reduce inventory costs by 10-15%",
                "business": "Improve customer satisfaction via reduced stockouts"
            }
        }

        segmentation_objective = {
            "target_segments": [
                "smooth_regular", "intermittent_low", "intermittent_high",
                "lumpy_seasonal", "erratic_volatile"
            ],
            "segmentation_features": [
                "intermittency_rate", "coefficient_of_variation",
                "trend_strength", "seasonality_strength", "zero_ratio"
            ],
            "business_purpose": "Tailor forecasting models to demand patterns",
            "segment_strategies": {
                "smooth_regular": "Standard ARIMA/ETS models",
                "intermittent_low": "Croston's method variations",
                "intermittent_high": "Bootstrap intermittent demand",
                "lumpy_seasonal": "Seasonal intermittent models",
                "erratic_volatile": "Machine learning ensemble"
            }
        }

        return {
            "forecasting_objective": forecasting_objective,
            "segmentation_objective": segmentation_objective
        }

    def identify_available_features(self) -> Dict[str, Any]:
        """
        Identify features available vs forbidden at prediction time.

        Returns:
            Dictionary categorizing features by availability at prediction time
        """
        # Features available at prediction time (no leakage)
        available_features = [
            # Historical sales (lagged appropriately)
            "sales_lag_1_to_28", "sales_rolling_mean_7_14_28",
            "sales_seasonal_lag_365", "sales_trend_components",

            # Calendar features (known in advance)
            "date", "day_of_week", "day_of_month", "month", "quarter",
            "is_weekend", "is_holiday", "days_to_holiday",

            # Event features (planned events only)
            "planned_events", "snap_ca", "snap_tx", "snap_wi",
            "cultural_events", "national_holidays",

            # Item/store static attributes
            "item_id", "store_id", "category_id", "department_id",
            "state_id", "store_type", "item_category",

            # Historical pricing (lagged appropriately)
            "price_lag_1_to_7", "price_changes_historical",
            "promotional_periods_historical"
        ]

        # Features forbidden at prediction time (cause leakage)
        forbidden_features = [
            # Future sales (direct target leakage)
            "sales_future", "sales_same_period",

            # Future pricing (not known at prediction time)
            "sell_prices_future", "promotional_prices_future",

            # Future events (unknown events)
            "sporting_events_future", "weather_events_future",

            # Aggregated statistics including future data
            "sales_annual_total", "category_performance_full_year",

            # Competition data (typically not available)
            "competitor_sales", "market_share_data",

            # Inventory levels (usually not available for forecasting)
            "inventory_on_hand", "stock_levels"
        ]

        # Time-sensitive feature rules
        temporal_rules = {
            "cutoff_date": "2016-04-24",  # d_1914 - validation start
            "max_lag_days": 1,  # Minimum lag for sales features
            "price_lag_days": 1,  # Minimum lag for pricing features
            "event_lead_time": 0,  # Events can be same-day if planned
            "calendar_lead_time": -365  # Calendar features known 1 year ahead
        }

        return {
            "available_features": available_features,
            "forbidden_features": forbidden_features,
            "temporal_rules": temporal_rules,
            "feature_count": {
                "available": len(available_features),
                "forbidden": len(forbidden_features)
            }
        }

    def define_leakage_risks(self) -> Dict[str, Any]:
        """
        Define specific data leakage risks for the M5 forecasting problem.

        Returns:
            Dictionary categorizing different types of leakage risks
        """
        temporal_leakage = {
            "description": "Using future information to predict the past",
            "examples": [
                "Using sales from d_1942+ to predict d_1914-1941",
                "Including future promotional events in feature engineering",
                "Using full-period statistics that include validation/test periods"
            ],
            "prevention": [
                "Strict temporal cutoff at d_1913 for training features",
                "Time-aware cross-validation with forward chaining",
                "Rolling window feature engineering with proper lags"
            ],
            "risk_level": "CRITICAL"
        }

        pricing_leakage = {
            "description": "Using future pricing information not available at prediction time",
            "examples": [
                "Using promotional prices from future periods",
                "Including competitor pricing not known in advance",
                "Price optimization feedback loops"
            ],
            "prevention": [
                "Use only historical pricing with appropriate lags",
                "Separate price prediction from demand prediction",
                "Model price changes probabilistically for scenarios"
            ],
            "risk_level": "HIGH"
        }

        event_leakage = {
            "description": "Using event information not available at forecast time",
            "examples": [
                "Including unplanned sporting event outcomes",
                "Using weather events known only after occurrence",
                "Cultural events not scheduled in advance"
            ],
            "prevention": [
                "Use only pre-scheduled/planned events",
                "Model weather probabilistically using historical patterns",
                "Distinguish planned vs reactive events clearly"
            ],
            "risk_level": "MEDIUM"
        }

        aggregation_leakage = {
            "description": "Statistics calculated across time periods that include the prediction target",
            "examples": [
                "Annual sales totals including validation period",
                "Category performance metrics spanning train/validation",
                "Global scaling factors computed on full dataset"
            ],
            "prevention": [
                "Calculate all statistics on training period only",
                "Use expanding window for aggregations",
                "Time-aware normalization and scaling"
            ],
            "risk_level": "HIGH"
        }

        return {
            "temporal_leakage": temporal_leakage,
            "pricing_leakage": pricing_leakage,
            "event_leakage": event_leakage,
            "aggregation_leakage": aggregation_leakage
        }

    def validate_prerequisites(self) -> List[str]:
        """
        Validate prerequisites for business context analysis.

        Returns:
            List of missing prerequisites (empty for Step 1 as it has no dependencies)
        """
        # Business context analysis is Step 1 - no dependencies
        return []

    def execute(self) -> Dict[str, Any]:
        """
        Execute complete business context analysis.

        Returns:
            Dictionary containing complete Step 1 analysis results
        """
        # Run all analysis components
        problem_definition = self.analyze_problem_definition()
        available_features = self.identify_available_features()
        leakage_risks = self.define_leakage_risks()

        # Generate visualization
        self._create_problem_definition_plot()

        # Compile key findings
        key_findings = [
            "Dual-objective problem: 28-day forecasting + demand pattern segmentation",
            "Target evaluation metric: WRMSSE (Walmart custom accuracy measure)",
            f"Available features: {available_features['feature_count']['available']} vs forbidden: {available_features['feature_count']['forbidden']}",
            "Critical temporal cutoff: d_1913 (April 24, 2016) prevents leakage",
            "Five demand segments identified for specialized model strategies",
            "Pricing and event timing create major leakage risks requiring careful handling"
        ]

        # Create comprehensive result
        result = {
            "problem_definition": problem_definition,
            "available_features": available_features,
            "leakage_risks": leakage_risks,
            "key_findings": key_findings,
            "business_impact": {
                "cost_reduction_target": "10-15% inventory cost reduction",
                "service_level_improvement": "Reduced stockouts and overstock",
                "model_differentiation": "Segment-specific strategies vs one-size-fits-all",
                "competition_benchmark": "WRMSSE < 0.5 for competitive performance"
            }
        }

        # Cache result
        self._save_step_result("step_1", result)

        return {"step_1": result}

    def generate_summary(self) -> str:
        """
        Generate human-readable summary of business context analysis.

        Returns:
            Formatted summary string
        """
        result = self._get_step_result("step_1")
        if not result:
            return "No business context analysis results available."

        summary = """# Business Context Analysis (EDA Step 1)

## Problem Definition Summary
The M5 dataset presents a dual-objective retail forecasting challenge:

### Primary Objective: Demand Forecasting
- **Target**: 28-day ahead unit sales prediction per item-store combination
- **Evaluation**: WRMSSE metric (Walmart's custom hierarchical accuracy measure)
- **Business Goal**: Optimize inventory planning to reduce costs by 10-15%
- **Success Criteria**: Achieve WRMSSE < 0.5 for competitive performance

### Secondary Objective: Demand Pattern Segmentation
- **Purpose**: Tailor forecasting strategies to different demand behaviors
- **Segments**: 5 categories (smooth, intermittent low/high, lumpy, erratic)
- **Strategy**: Apply specialized models per segment vs one-size-fits-all approach

## Critical Data Leakage Prevention
### Temporal Boundary
- **Strict Cutoff**: d_1913 (April 24, 2016) for all training features
- **Validation Period**: d_1914 to d_1941 (28 days)
- **No Future Information**: Zero tolerance for post-cutoff data in training

### Feature Availability Assessment
"""

        if result.get("available_features"):
            available_count = result["available_features"]["feature_count"]["available"]
            forbidden_count = result["available_features"]["feature_count"]["forbidden"]

            summary += f"""- **Available Features**: {available_count} (historical sales, calendar, static attributes)
- **Forbidden Features**: {forbidden_count} (future sales, pricing, unplanned events)
- **Risk Level**: {len(result.get("leakage_risks", {}))} major leakage categories identified

"""

        summary += """## Business Impact Potential
- **Cost Reduction**: 10-15% inventory optimization through better demand prediction
- **Service Level**: Reduced stockouts and overstock situations
- **Competitive Advantage**: Segment-specific modeling vs generic approaches
- **Model Performance**: Target WRMSSE < 0.5 for industry-leading accuracy

## Next Steps
1. Proceed to Data Overview (Step 2) with established temporal boundaries
2. Feature engineering must respect leakage prevention rules
3. Model development should leverage segment-specific strategies
4. Validation framework must use time-aware cross-validation

---
*Analysis completed: EDA Step 1 - Business Context Understanding*
"""

        return summary

    def _create_problem_definition_plot(self) -> str:
        """
        Create visualization showing the problem definition timeline.

        Returns:
            String path to the created plot file
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Timeline data
        dates = [
            datetime(2011, 1, 29),  # d_1 start
            datetime(2016, 4, 24),  # d_1913 - training end
            datetime(2016, 5, 22),  # d_1941 - validation end
            datetime(2016, 6, 19)   # d_1969 - evaluation end
        ]

        periods = ['Training\n(d_1 to d_1913)', 'Validation\n(d_1914 to d_1941)',
                  'Evaluation\n(d_1942 to d_1969)']
        colors = ['#2E86AB', '#A23B72', '#F18F01']

        # Plot 1: Timeline
        ax1.set_title('M5 Forecasting Problem Timeline', fontsize=16, fontweight='bold', pad=20)

        # Draw timeline
        for i, (start, end, period, color) in enumerate(zip(dates[:-1], dates[1:], periods, colors)):
            duration = (end - start).days
            ax1.barh(0, duration, left=(start - dates[0]).days, height=0.3,
                    color=color, alpha=0.7, label=period)

            # Add period labels
            mid_point = (start - dates[0]).days + duration/2
            ax1.text(mid_point, 0, f'{period}\n{duration} days',
                    ha='center', va='center', fontweight='bold', fontsize=10)

        # Add critical boundary marker
        cutoff_days = (datetime(2016, 4, 24) - dates[0]).days
        ax1.axvline(x=cutoff_days, color='red', linestyle='--', linewidth=3, alpha=0.8)
        ax1.text(cutoff_days, 0.6, 'LEAKAGE PREVENTION\nBOUNDARY', ha='center',
                color='red', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red"))

        ax1.set_xlim(0, (dates[-1] - dates[0]).days)
        ax1.set_ylim(-0.5, 1)
        ax1.set_xlabel('Days since start (d_1 = Jan 29, 2011)', fontsize=12)
        ax1.set_yticks([])
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3)

        # Plot 2: Business Objectives
        ax2.set_title('Dual Business Objectives', fontsize=16, fontweight='bold', pad=20)

        objectives = ['28-Day\nForecasting', 'Demand Pattern\nSegmentation']
        metrics = ['WRMSSE < 0.5', '5 Behavioral\nSegments']
        impacts = ['10-15% Cost\nReduction', 'Specialized\nStrategies']

        # Create objective boxes
        colors_obj = ['#4CAF50', '#FF9800']
        y_positions = [1, 0]

        for i, (obj, metric, impact, color, y) in enumerate(zip(objectives, metrics, impacts, colors_obj, y_positions)):
            # Objective box
            ax2.text(0.1, y, obj, ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3, edgecolor=color))

            # Arrow
            ax2.annotate('', xy=(0.4, y), xytext=(0.25, y),
                        arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

            # Metric box
            ax2.text(0.5, y, metric, ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

            # Arrow
            ax2.annotate('', xy=(0.75, y), xytext=(0.65, y),
                        arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

            # Impact box
            ax2.text(0.85, y, impact, ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

        ax2.set_xlim(0, 1)
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.axis('off')

        # Add overall framework note
        plt.figtext(0.5, 0.02, 'EDA Step 1: Business Context Analysis - Foundation for Leakage-Free Modeling',
                   ha='center', fontsize=12, style='italic')

        plt.tight_layout()

        # Save plot
        plot_path = self._create_plot_path("step1_business_context", "problem_definition_timeline")
        full_plot_path = self.ctx.plots_dir / plot_path
        full_plot_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(full_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(full_plot_path)