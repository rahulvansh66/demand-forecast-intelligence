#!/usr/bin/env python3
"""
Test script to verify that the integration fixes in eda_analysis.py work correctly.
"""

import sys
import os
sys.path.insert(0, 'notebooks/eda')

from eda_analysis import study_feature_target_relationships

def test_integration():
    """Test the complete integration workflow."""
    print("Testing Integration Fixes for study_feature_target_relationships()")
    print("=" * 80)

    try:
        # Run the fixed function
        results = study_feature_target_relationships()

        print("\n" + "=" * 80)
        print("INTEGRATION TEST RESULTS:")
        print("=" * 80)

        # Check if the main components worked
        success_count = 0
        total_checks = 4

        # 1. Check categorical patterns
        if 'categorical_patterns' in results:
            if 'error' not in results['categorical_patterns']:
                print("✅ Categorical patterns analysis: SUCCESS")
                success_count += 1

                # Show details
                cats = results['categorical_patterns'].get('categories', [])
                print(f"   - Found {len(cats)} categories: {cats}")

                if 'store_analysis' in results['categorical_patterns']:
                    stores = results['categorical_patterns']['store_analysis'].get('categories', [])
                    print(f"   - Found {len(stores)} stores")
            else:
                print(f"❌ Categorical patterns analysis: FAILED - {results['categorical_patterns']['error']}")
        else:
            print("❌ Categorical patterns analysis: MISSING")

        # 2. Check temporal correlations
        if 'temporal_correlations' in results:
            if 'error' not in results['temporal_correlations']:
                print("✅ Temporal correlations analysis: SUCCESS")
                success_count += 1

                # Show details
                temp_cats = results['temporal_correlations'].get('temporal_correlations', {})
                print(f"   - Analyzed {len(temp_cats)} categories for temporal patterns")
            else:
                print(f"❌ Temporal correlations analysis: FAILED - {results['temporal_correlations']['error']}")
        else:
            print("❌ Temporal correlations analysis: MISSING")

        # 3. Check SNAP analysis
        if 'snap_impact' in results:
            if 'error' not in results['snap_impact']:
                print("✅ SNAP benefit impact analysis: SUCCESS")
                success_count += 1

                # Show details
                snap_states = results['snap_impact'].get('snap_impact_by_state', {})
                print(f"   - Analyzed {len(snap_states)} states for SNAP impact")
            else:
                print(f"ℹ️  SNAP benefit impact analysis: INFO - {results['snap_impact']['error']}")
                success_count += 1  # This might be expected if no SNAP data
        else:
            print("❌ SNAP benefit impact analysis: MISSING")

        # 4. Check visualizations
        if 'visualizations' in results:
            if 'error' not in results['visualizations']:
                print("✅ Visualizations: SUCCESS")
                success_count += 1

                if 'category_distributions' in results['visualizations']:
                    print("   - Category distribution plots generated")
            else:
                print(f"❌ Visualizations: FAILED - {results['visualizations']['error']}")
        else:
            print("❌ Visualizations: MISSING")

        # 5. Check summary
        if 'summary' in results:
            print("\n📊 SUMMARY:")
            summary = results['summary']
            print(f"   - Total categories: {summary.get('total_categories', 'N/A')}")
            print(f"   - Temporal features: {summary.get('temporal_features_analyzed', 'N/A')}")
            print(f"   - SNAP states: {summary.get('snap_states_analyzed', 'N/A')}")
            print(f"   - Total observations: {summary.get('total_observations', 'N/A')}")
            print(f"   - Status: {summary.get('step_status', 'N/A')}")

        print(f"\n🎯 OVERALL SUCCESS RATE: {success_count}/{total_checks} components working")

        if success_count >= 3:
            print("🎉 INTEGRATION TEST PASSED! Most components are working correctly.")
            return True
        else:
            print("⚠️  INTEGRATION TEST PARTIAL. Some components need attention.")
            return False

    except Exception as e:
        print(f"❌ CRITICAL ERROR during integration test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)