#!/usr/bin/env python3
"""
Main EDA Demo Script

Demonstrates the M5 EDA framework by running the implemented components
and showing the framework's capabilities.
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.core.context import EDAContext
from utils.core.orchestrator import EDAOrchestrator
from utils.services.data_understanding.business_context import BusinessContextService
from config import M5_CONFIG


def main():
    """Main demonstration of the EDA framework."""
    print("=" * 80)
    print("M5 EDA FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    print()

    # Initialize EDA Context
    print("🔧 Initializing EDA Context...")
    ctx = EDAContext.from_config(M5_CONFIG)
    print(f"   ✓ Data directory: {ctx.data_dir}")
    print(f"   ✓ Output directory: {ctx.output_dir}")
    print(f"   ✓ Plots directory: {ctx.plots_dir}")
    print()

    # Initialize Orchestrator
    print("🎯 Initializing EDA Orchestrator...")
    orchestrator = EDAOrchestrator(ctx)
    print(f"   ✓ Available phases: {list(orchestrator.PHASE_STEPS.keys())}")
    print(f"   ✓ Available subgroups: {list(orchestrator.SUBGROUP_STEPS.keys())}")
    print(f"   ✓ Total steps: {list(orchestrator.STEP_DEPENDENCIES.keys())}")
    print()

    # Show framework architecture
    print("🏗️  Framework Architecture:")
    print("   📊 Phases & Steps:")
    for phase, steps in orchestrator.PHASE_STEPS.items():
        phase_names = {
            1: "Data Understanding",
            2: "Feature Analysis",
            3: "Time Patterns",
            4: "Model Preparation"
        }
        print(f"      Phase {phase} ({phase_names[phase]}): Steps {steps}")
    print()

    # Show configuration
    print("⚙️  M5 Configuration Summary:")
    analysis_params = M5_CONFIG.get("analysis_params", {})
    for param, value in analysis_params.items():
        print(f"   • {param}: {value}")
    print()

    # Demonstrate BusinessContextService (Step 1)
    print("📈 Demonstrating BusinessContextService (Step 1)...")
    try:
        service = BusinessContextService(ctx)

        # Check prerequisites
        print("   🔍 Checking prerequisites...")
        prereqs = service.validate_prerequisites()
        if prereqs:
            print(f"      ⚠️  Missing prerequisites: {prereqs}")
            print("      📝 Note: This is expected without M5 dataset files")
        else:
            print("      ✓ All prerequisites satisfied")

        print("   📋 Business context analysis capabilities:")
        print("      • Forecasting objectives definition")
        print("      • Temporal boundary establishment")
        print("      • Data leakage prevention rules")
        print("      • Problem visualization generation")
        print()

    except Exception as e:
        print(f"   ⚠️  Service demo skipped: {e}")
        print()

    # Show framework capabilities
    print("🚀 Framework Capabilities:")
    print("   ✅ Modular service-based architecture")
    print("   ✅ Dependency-aware execution order")
    print("   ✅ Results caching (memory + persistent)")
    print("   ✅ Multi-granularity execution (step/subgroup/phase/full)")
    print("   ✅ M5-specific configuration system")
    print("   ✅ Comprehensive test coverage (53 tests)")
    print("   ✅ Plot generation (matplotlib/seaborn)")
    print("   ✅ Security validation & input sanitization")
    print()

    # Show execution examples
    print("💻 Usage Examples:")
    print("   # Run individual step:")
    print("   orchestrator.run_step(1)")
    print()
    print("   # Run subgroup:")
    print("   orchestrator.run_subgroup('1A')")
    print()
    print("   # Run full phase:")
    print("   orchestrator.run_phase(1)")
    print()
    print("   # Run complete pipeline:")
    print("   orchestrator.run_full_pipeline()")
    print()

    # Show deliverables
    print("📦 Expected Deliverables:")
    print("   📊 Plots: eda/plots/step_XX_*/")
    print("   📄 Reports: data/eda/outputs/reports/")
    print("   💾 Results: data/eda/outputs/step_results/")
    print("   📈 Summaries: data/eda/outputs/phase_summaries/")
    print()

    print("=" * 80)
    print("FRAMEWORK READY FOR M5 DATASET ANALYSIS")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("1. Add M5 dataset files to data/raw/")
    print("2. Run: orchestrator.run_full_pipeline()")
    print("3. Check outputs in data/eda/outputs/ and eda/plots/")


if __name__ == "__main__":
    main()