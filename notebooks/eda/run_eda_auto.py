#!/usr/bin/env python3
"""
Run EDA analysis with complete output capture (non-interactive version).

This script runs your EDA analysis functions and saves all outputs
automatically without requiring user input.
"""

import os
import sys

# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from output_manager import EDAOutputManager


def main():
    """Run EDA analysis with output capture."""

    # Import your EDA functions
    try:
        # Import new EDA steps (1, 2, 3, 5)
        from eda_steps_1_3_5 import (
            analyze_m5_problem_context,
            inspect_m5_dataset_structure,
            check_m5_data_quality,
            analyze_m5_individual_features
        )

        # Import existing EDA steps (6-14)
        from eda_analysis import (
            study_feature_target_relationships,
            study_feature_feature_relationships,
            analyze_time_series_patterns,
            analyze_missing_values_deeply,
            identify_outliers_and_anomalies,
            analyze_segment_behavior,
            analyze_distribution_drift,
            audit_temporal_leakage
        )
    except ImportError as e:
        print(f"❌ Error importing EDA functions: {e}")
        print("Make sure eda_analysis.py and eda_steps_1_3_5.py are in the same directory")
        return

    # Initialize output manager
    output_manager = EDAOutputManager("notebooks/eda/outputs")

    print("🚀 EDA Analysis with Output Capture (Auto Mode)")
    print(f"📁 Output directory: notebooks/eda/outputs")
    print(f"🆔 Session ID: {output_manager.session_id}")
    print("=" * 80)

    # Define all available steps with logical ordering (1-5, then 6-14)
    steps_to_run = [
        # NEW EDA STEPS (1-5)
        ("1", "M5 Problem Context Analysis", analyze_m5_problem_context),
        ("2", "M5 Dataset Structure Inspection", inspect_m5_dataset_structure),
        ("3", "M5 Data Quality Assessment", check_m5_data_quality),
        ("5", "M5 Individual Features Analysis", analyze_m5_individual_features),

        # EXISTING EDA STEPS (6-14)
        ("6", "Feature-Target Relationships", study_feature_target_relationships),
        ("7", "Feature-Feature Relationships", study_feature_feature_relationships),
        ("8", "Time Series Patterns", analyze_time_series_patterns),
        ("9", "Missing Values Analysis", analyze_missing_values_deeply),
        ("10", "Outliers and Anomalies", identify_outliers_and_anomalies),
        ("11", "Segment Behavior", analyze_segment_behavior),
        ("13", "Distribution Drift", analyze_distribution_drift),
        ("14", "Temporal Leakage Audit", audit_temporal_leakage),
    ]

    print(f"🔄 Running {len(steps_to_run)} EDA steps automatically...")

    all_results = {}

    for step_num, step_name, step_function in steps_to_run:
        full_step_name = f"Step {step_num}: {step_name}"

        print(f"\n{'='*80}")
        print(f"🔄 Running {full_step_name}")
        print(f"{'='*80}")

        try:
            result, captured_output = output_manager.capture_function_output(step_function)
            all_results[step_function.__name__] = result

            if result and 'error' in result:
                print(f"⚠️  {full_step_name} completed with errors")
            else:
                print(f"✅ {full_step_name} completed successfully")

        except Exception as e:
            print(f"❌ Error in {full_step_name}: {str(e)}")
            all_results[step_function.__name__] = {'error': str(e)}

    # Create overall session summary
    print(f"\n{'='*80}")
    print("📋 Creating Session Summary...")
    output_manager.create_session_summary(all_results)

    print(f"\n🎉 EDA Analysis Complete!")
    print(f"📁 All outputs saved to: notebooks/eda/outputs")
    print(f"🆔 Session ID: {output_manager.session_id}")
    print("\nOutput Structure:")
    print(f"  📁 logs/        - Function execution logs")
    print(f"  📁 results/     - Structured analysis results (JSON, CSV)")
    print(f"  📁 summaries/   - Human-readable summaries")
    print(f"  📁 plots/       - Generated visualizations")
    print(f"  📄 EDA_SESSION_SUMMARY_{output_manager.session_id}.md - Complete session overview")

    # Show a sample of what was captured
    print(f"\n📊 Results Summary:")
    for func_name, result in all_results.items():
        if 'error' in result:
            print(f"  ❌ {func_name}: Error - {result['error']}")
        elif 'summary' in result:
            summary = result['summary']
            print(f"  ✅ {func_name}: {len(summary)} summary items")
        else:
            print(f"  ✅ {func_name}: Completed")

    return all_results


if __name__ == "__main__":
    main()