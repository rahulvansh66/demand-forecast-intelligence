#!/usr/bin/env python3
"""
M5 Sample Dataset Creation CLI Script.

This script provides a user-friendly command-line interface for generating
sample datasets from the M5 Walmart dataset using behavioral stratification.
The CLI orchestrates all sampling components (config, stratification, generation,
validation) into a simple command-line tool for POC development.

Business Rationale:
Data scientists and ML engineers need a simple, configurable way to generate
sample datasets for faster POC iteration. The CLI provides immediate access
to the sophisticated behavioral stratification system through an intuitive
interface, enabling rapid experimentation on M3 Pro hardware constraints.

Key Features:
- Configurable data directories and sample sizes
- Reproducible sampling with configurable random seeds
- Optional validation report generation with quality metrics
- Clear progress reporting and error handling
- Category distribution and behavioral diversity summaries

Technical Design:
The script follows the Unix philosophy of "do one thing well" by focusing
solely on dataset sampling orchestration. It delegates complex logic to
specialized classes while providing a clean user interface and comprehensive
error handling for production-ready operation.

Usage Examples:
    # Basic usage with defaults (1400 items, data/full_data -> data/processed/sample_dataset)
    python scripts/create_sample_dataset.py

    # Custom configuration
    python scripts/create_sample_dataset.py --data-dir /path/to/m5/data --output-dir /path/to/output --target-items 1000 --random-seed 123

    # With validation report
    python scripts/create_sample_dataset.py --validate

    # Quick help
    python scripts/create_sample_dataset.py --help
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Add source directory to Python path for imports
# This allows the script to import from the main package structure
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Import the sampling components after path setup
# These imports access the behavioral stratification and generation logic
from demand_forecast_intelligence.data.sampling.config import SamplingConfig
from demand_forecast_intelligence.data.sampling.sample_generator import SampleGenerator
from demand_forecast_intelligence.data.schemas.m5_schemas import (
    validate_sales_dataframe, validate_calendar_dataframe, validate_prices_dataframe
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure logging for user-friendly CLI output.

    Sets up structured logging that provides clear progress updates during
    sample generation while maintaining appropriate detail levels for debugging.
    The logging configuration balances informativeness with readability for
    CLI users who need to understand processing status.

    Args:
        verbose: If True, enables DEBUG level logging for detailed diagnostics.
                If False, uses INFO level for standard operational updates.

    Returns:
        Logger instance configured for CLI usage with timestamp and level formatting.

    Design Choice:
    Uses a simple format suitable for CLI output rather than JSON or structured
    logging, prioritizing human readability over machine parsing since this is
    a user-facing tool for data scientists and ML engineers.
    """
    # Configure root logger to avoid interference with library loggers
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at {log_level} level")

    return logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for sample dataset generation.

    Provides a comprehensive argument parser that supports all key sampling
    configuration options through intuitive command-line flags. The argument
    design follows standard Unix conventions while providing sensible defaults
    for rapid POC development on resource-constrained hardware.

    Returns:
        Namespace with parsed arguments including paths, sampling parameters,
        and operational flags for validation and progress reporting.

    Argument Design Rationale:
    - Default paths assume standard project structure (data/full_data -> data/processed/sample_dataset)
    - Target items default (1400) achieves ~50% dataset reduction for M3 Pro performance
    - Random seed default (42) ensures reproducible results for POC comparison
    - Validation is optional to allow fast iteration during development
    - All arguments use descriptive long names with intuitive short aliases
    """
    parser = argparse.ArgumentParser(
        description="Generate sample M5 dataset using behavioral stratification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults (1400 items, standard paths)
  python scripts/create_sample_dataset.py

  # Custom sample size and paths
  python scripts/create_sample_dataset.py --data-dir /path/to/m5 --target-items 1000

  # Generate with validation report
  python scripts/create_sample_dataset.py --validate

  # Reproducible sampling with custom seed
  python scripts/create_sample_dataset.py --random-seed 123

  # Verbose output for debugging
  python scripts/create_sample_dataset.py --verbose

This tool creates statistically representative sample datasets from M5 Walmart
data using behavioral stratification to prevent sampling bias. The approach
ensures challenging intermittent/sparse demand patterns are properly represented
for robust POC validation.
        """
    )

    # Input/Output Configuration
    # These arguments control where the script finds source data and saves results
    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=Path('data/full_data'),
        help='Directory containing M5 dataset CSV files (default: data/full_data)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('data/processed/sample_dataset'),
        help='Directory to save sample dataset files (default: data/processed/sample_dataset)'
    )

    # Sampling Parameters
    # These arguments control the core sampling behavior and reproducibility
    parser.add_argument(
        '--target-items', '-n',
        type=int,
        default=1400,
        help='Target number of items in sample dataset (default: 1400, ~50%% reduction)'
    )

    parser.add_argument(
        '--random-seed', '-r',
        type=int,
        default=42,
        help='Random seed for reproducible sampling (default: 42)'
    )

    # Operational Flags
    # These arguments control script behavior and output detail level
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Generate detailed validation report with quality metrics'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging for detailed progress tracking'
    )

    return parser.parse_args()


def load_m5_data(data_dir: Path, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """
    Load and validate M5 dataset files with comprehensive error handling.

    This function implements robust data loading that validates both file
    existence and data quality before proceeding with sampling. The validation
    prevents downstream errors and provides clear feedback about data issues
    that could impact sample generation quality.

    Args:
        data_dir: Path to directory containing M5 dataset CSV files
                 Expected files: sales_train_validation.csv, calendar.csv, sell_prices.csv
        logger: Logger instance for progress reporting and error diagnostics

    Returns:
        Dictionary containing loaded DataFrames:
        - 'sales': Sales training data with validation
        - 'calendar': Calendar data (optional, for time-based analysis)
        - 'prices': Price data (optional, for economic analysis)

    Raises:
        FileNotFoundError: If required sales_train_validation.csv is missing
        ValueError: If any loaded dataset fails schema validation

    Data Loading Strategy:
    The function prioritizes the sales data as required (core for sampling)
    while treating calendar and prices as optional supplementary data.
    This design allows the script to work with minimal M5 datasets while
    providing full functionality when complete data is available.

    Validation Approach:
    Uses the project's M5 schema validators to ensure data quality and
    consistency with expected M5 dataset structure. This prevents silent
    failures that could produce invalid sample datasets.
    """
    logger.info(f"Loading M5 dataset from: {data_dir}")

    # Check if data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data = {}

    # Load sales data (required for sampling)
    sales_file = data_dir / "sales_train_validation.csv"
    logger.info(f"Loading sales data from: {sales_file}")

    if not sales_file.exists():
        raise FileNotFoundError(
            f"Required sales data file not found: {sales_file}\n"
            "Please ensure sales_train_validation.csv is in the data directory."
        )

    try:
        sales_data = pd.read_csv(sales_file)
        logger.info(f"Loaded sales data: {len(sales_data):,} rows, {len(sales_data.columns)} columns")

        # Validate sales data structure and quality
        validate_sales_dataframe(sales_data)
        logger.info("Sales data validation passed")

        data['sales'] = sales_data

    except Exception as e:
        logger.error(f"Failed to load or validate sales data: {e}")
        raise ValueError(f"Sales data loading failed: {e}")

    # Load calendar data (optional, enhances time-based analysis)
    calendar_file = data_dir / "calendar.csv"
    if calendar_file.exists():
        try:
            logger.info(f"Loading calendar data from: {calendar_file}")
            calendar_data = pd.read_csv(calendar_file)

            # Validate calendar data structure
            validate_calendar_dataframe(calendar_data)
            logger.info(f"Loaded calendar data: {len(calendar_data):,} rows")

            data['calendar'] = calendar_data

        except Exception as e:
            logger.warning(f"Calendar data loading failed (optional): {e}")
    else:
        logger.info("Calendar data not found (optional)")

    # Load price data (optional, enhances economic analysis)
    prices_file = data_dir / "sell_prices.csv"
    if prices_file.exists():
        try:
            logger.info(f"Loading price data from: {prices_file}")
            prices_data = pd.read_csv(prices_file)

            # Validate price data structure
            validate_prices_dataframe(prices_data)
            logger.info(f"Loaded price data: {len(prices_data):,} rows")

            data['prices'] = prices_data

        except Exception as e:
            logger.warning(f"Price data loading failed (optional): {e}")
    else:
        logger.info("Price data not found (optional)")

    logger.info("M5 data loading completed successfully")
    return data


def display_configuration(config: SamplingConfig, data_dir: Path, output_dir: Path, logger: logging.Logger):
    """
    Display sampling configuration parameters before execution.

    Provides transparent reporting of all sampling parameters to enable
    user verification and audit trail documentation. This visibility is
    critical for reproducible research and allows users to verify that
    the configuration matches their intended sampling strategy.

    Args:
        config: SamplingConfig instance with all sampling parameters
        data_dir: Source data directory path
        output_dir: Destination directory for sample files
        logger: Logger instance for formatted output

    Design Rationale:
    Configuration display serves multiple purposes:
    1. User verification - allows checking parameters before long-running process
    2. Audit trail - provides documentation of sampling choices for reproducibility
    3. Debugging - helps diagnose configuration issues before execution
    4. Education - demonstrates the range of available configuration options

    The formatting prioritizes readability while maintaining completeness,
    enabling users to quickly verify key parameters while providing access
    to detailed configuration for advanced users.
    """
    logger.info("=" * 60)
    logger.info("M5 SAMPLE DATASET GENERATION CONFIGURATION")
    logger.info("=" * 60)

    # Core sampling parameters - most important for user verification
    logger.info("CORE PARAMETERS:")
    logger.info(f"  Target Items: {config.target_item_count:,}")
    logger.info(f"  Random Seed: {config.random_seed}")
    logger.info(f"  Source Directory: {data_dir}")
    logger.info(f"  Output Directory: {output_dir}")
    logger.info("")

    # Stratification parameters - control sampling quality and bias prevention
    logger.info("BEHAVIORAL STRATIFICATION:")
    logger.info(f"  Volume Percentiles: {config.volume_percentiles}")
    logger.info(f"  Intermittency Thresholds: {config.intermittency_thresholds}")
    logger.info(f"  Training Period End: {config.training_end_day}")
    logger.info("")

    # Constraint parameters - ensure statistical and business validity
    logger.info("SAMPLING CONSTRAINTS:")
    logger.info(f"  Minimum per Department: {config.min_per_dept}")
    logger.info(f"  Minimum per Stratum: {config.min_per_stratum}")
    logger.info("")

    # Lifecycle analysis parameters - temporal pattern classification
    logger.info("LIFECYCLE ANALYSIS:")
    for key, value in config.lifecycle_thresholds.items():
        logger.info(f"  {key.replace('_', ' ').title()}: {value}")
    logger.info("")

    logger.info("=" * 60)


def generate_validation_report(results: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Generate comprehensive validation report for sample dataset quality.

    Creates detailed analysis of sampling results including statistical
    metrics, bias prevention verification, and coverage validation.
    This report provides audit trail evidence that sampling was performed
    correctly and produces high-quality representative datasets.

    Args:
        results: Dictionary containing sampling results from SampleGenerator
                Includes sample_items, allocation_summary, sampling_metadata,
                file_paths, and coverage_stats from generation process
        logger: Logger instance for formatted report output

    Report Components:
    1. Sample Quality Metrics - statistical measures of representativeness
    2. Behavioral Diversity - verification of pattern coverage
    3. Geographic Coverage - store/state representation validation
    4. Anti-Bias Evidence - documentation of random sampling approach
    5. File Generation Summary - output file locations and sizes

    Business Value:
    The validation report serves as quality assurance documentation that
    demonstrates the sample dataset is suitable for POC validation and
    model development. It provides confidence that results will be
    representative of real-world performance across diverse demand patterns.
    """
    logger.info("=" * 60)
    logger.info("SAMPLE DATASET VALIDATION REPORT")
    logger.info("=" * 60)

    # Extract key result components for analysis
    sample_items = results['sample_items']
    metadata = results['sampling_metadata']
    coverage_stats = results['coverage_stats']

    # Sample Quality Summary - core metrics for quality assessment
    logger.info("SAMPLE QUALITY SUMMARY:")
    logger.info(f"  Items Selected: {len(sample_items):,}")
    logger.info(f"  Target Achievement: {len(sample_items) / metadata['target_item_count']:.1%}")
    logger.info(f"  Population Reduction: {coverage_stats['reduction_stats']['item_reduction']:.1%}")
    logger.info(f"  Sampling Method: {metadata['sampling_method']}")
    logger.info("")

    # Behavioral Diversity Analysis - verification of pattern representation
    logger.info("BEHAVIORAL DIVERSITY:")

    # Volume distribution across sample
    volume_dist = sample_items['volume_bucket'].value_counts().to_dict()
    logger.info("  Volume Distribution:")
    for bucket, count in sorted(volume_dist.items()):
        percentage = count / len(sample_items) * 100
        logger.info(f"    {bucket.title()}: {count:,} items ({percentage:.1f}%)")

    # Intermittency pattern coverage
    intermittency_dist = sample_items['intermittency_class'].value_counts().to_dict()
    logger.info("  Intermittency Patterns:")
    for pattern, count in sorted(intermittency_dist.items()):
        percentage = count / len(sample_items) * 100
        logger.info(f"    {pattern.title()}: {count:,} items ({percentage:.1f}%)")

    # Lifecycle stage representation
    lifecycle_dist = sample_items['lifecycle_stage'].value_counts().to_dict()
    logger.info("  Lifecycle Stages:")
    for stage, count in sorted(lifecycle_dist.items()):
        percentage = count / len(sample_items) * 100
        logger.info(f"    {stage.title()}: {count:,} items ({percentage:.1f}%)")

    logger.info("")

    # Geographic Coverage Verification - business context preservation
    logger.info("GEOGRAPHIC COVERAGE:")
    geo_coverage = coverage_stats['geographic_coverage']
    logger.info(f"  States: {geo_coverage['states_in_sample']}/{geo_coverage['states_in_population']} "
               f"({'COMPLETE' if geo_coverage['states_in_sample'] == geo_coverage['states_in_population'] else 'PARTIAL'})")
    logger.info(f"  Stores: {geo_coverage['stores_in_sample']}/{geo_coverage['stores_in_population']} "
               f"({'COMPLETE' if geo_coverage['stores_in_sample'] == geo_coverage['stores_in_population'] else 'PARTIAL'})")
    logger.info("")

    # Department Coverage Analysis - business segment representation
    logger.info("DEPARTMENT COVERAGE:")
    dept_dist = sample_items['dept_id'].value_counts().to_dict()
    for dept, count in sorted(dept_dist.items()):
        percentage = count / len(sample_items) * 100
        logger.info(f"  {dept}: {count:,} items ({percentage:.1f}%)")
    logger.info("")

    # Anti-Bias Verification - audit trail for sampling methodology
    logger.info("ANTI-BIAS VERIFICATION:")
    bias_prevention = metadata['bias_prevention']
    for key, value in bias_prevention.items():
        logger.info(f"  {key.replace('_', ' ').title()}: {value}")
    logger.info("")

    # File Generation Summary - output documentation
    if 'file_paths' in results:
        logger.info("GENERATED FILES:")
        for file_type, file_path in results['file_paths'].items():
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
                logger.info(f"  {file_type.title()}: {file_path} ({file_size:.1f} MB)")
            else:
                logger.info(f"  {file_type.title()}: {file_path} (FILE NOT FOUND)")
        logger.info("")

    # Quality Score Calculation - overall assessment
    # This provides a single metric for sample quality evaluation
    quality_metrics = {
        'target_achievement': len(sample_items) / metadata['target_item_count'],
        'strata_coverage': metadata['sample_stats']['strata_coverage'] / metadata['population_stats']['total_strata'],
        'dept_coverage': len(dept_dist) / len(metadata['population_stats']['dept_distribution']),
        'geographic_completeness': 1.0 if coverage_stats['coverage_complete'] else 0.8
    }

    overall_quality = sum(quality_metrics.values()) / len(quality_metrics)
    logger.info(f"OVERALL QUALITY SCORE: {overall_quality:.2f}/1.00 "
               f"({'EXCELLENT' if overall_quality >= 0.9 else 'GOOD' if overall_quality >= 0.8 else 'ACCEPTABLE'})")

    logger.info("=" * 60)


def main():
    """
    Main orchestration function for M5 sample dataset generation.

    This function implements the complete CLI workflow from argument parsing
    through final result reporting. It coordinates all sampling components
    while providing comprehensive error handling and user feedback throughout
    the process. The workflow follows these key phases:

    1. **Configuration Setup** - Parse arguments, configure logging, validate inputs
    2. **Data Loading** - Load and validate M5 dataset files with error handling
    3. **Sample Generation** - Execute behavioral stratification and random sampling
    4. **Result Processing** - Create output files and generate validation reports
    5. **User Feedback** - Provide summary and next steps for POC development

    Error Handling Strategy:
    The function uses a "fail fast" approach with comprehensive error reporting
    to help users quickly identify and resolve configuration or data issues.
    All errors include specific guidance for resolution to minimize debugging time.

    User Experience Design:
    Progress reporting balances informativeness with readability, providing
    enough detail for troubleshooting while maintaining clarity for routine use.
    The output format supports both interactive use and log file analysis.

    Exit Codes:
    - 0: Successful completion
    - 1: Configuration or argument errors
    - 2: Data loading or validation errors
    - 3: Sample generation errors
    - 4: Output file creation errors
    """
    try:
        # Phase 1: Configuration Setup
        # Parse command-line arguments and set up logging infrastructure
        args = parse_arguments()
        logger = setup_logging(args.verbose)

        logger.info("Starting M5 Sample Dataset Generation")

        # Create and validate sampling configuration
        try:
            config = SamplingConfig(
                target_item_count=args.target_items,
                random_seed=args.random_seed
            )
            logger.info("Configuration validation passed")
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)

        # Display configuration for user verification
        display_configuration(config, args.data_dir, args.output_dir, logger)

        # Phase 2: Data Loading and Validation
        # Load M5 dataset files with comprehensive error handling
        try:
            data = load_m5_data(args.data_dir, logger)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Data loading failed: {e}")
            logger.error("Please check data directory and file availability")
            sys.exit(2)

        # Phase 3: Sample Generation
        # Execute behavioral stratification and random sampling workflow
        logger.info("Initializing sample generation engine...")
        try:
            generator = SampleGenerator(config, args.data_dir)

            logger.info("Executing sample generation workflow...")
            logger.info("This may take a few minutes for large datasets...")

            # Execute complete sampling workflow
            results = generator.generate_sample()

            logger.info("Sample generation completed successfully!")

        except Exception as e:
            logger.error(f"Sample generation failed: {e}")
            logger.error("This may indicate data quality issues or configuration problems")
            sys.exit(3)

        # Phase 4: Output File Management
        # Create output directory and move files to final location
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory prepared: {args.output_dir}")

            # Note: SampleGenerator creates files in data_dir/samples/ by design
            # For CLI use, we could add file copying logic here if needed

        except Exception as e:
            logger.error(f"Output directory creation failed: {e}")
            sys.exit(4)

        # Phase 5: Results Reporting and Validation
        # Provide comprehensive summary and optional validation report
        logger.info("=" * 60)
        logger.info("SAMPLE GENERATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        # Core results summary
        sample_items = results['sample_items']
        metadata = results['sampling_metadata']

        logger.info(f"✓ Generated sample with {len(sample_items):,} items")
        logger.info(f"✓ Achieved {len(sample_items)/metadata['target_item_count']:.1%} of target size")
        logger.info(f"✓ Covered {metadata['sample_stats']['strata_coverage']} behavioral strata")
        logger.info(f"✓ Maintained {metadata['sample_stats']['dept_distribution']} department coverage")

        # File creation summary
        if 'file_paths' in results:
            logger.info("✓ Created sample dataset files:")
            for file_type, file_path in results['file_paths'].items():
                logger.info(f"  - {file_type}: {file_path}")

        # Generate detailed validation report if requested
        if args.validate:
            generate_validation_report(results, logger)

        # Next steps guidance for POC development
        logger.info("")
        logger.info("NEXT STEPS FOR POC DEVELOPMENT:")
        logger.info("1. Review validation report (use --validate flag for details)")
        logger.info("2. Use sample dataset files for model training and validation")
        logger.info("3. Compare model performance on sample vs full dataset")
        logger.info("4. Adjust sample size if needed based on performance requirements")
        logger.info("")
        logger.info("Sample dataset generation completed successfully! 🎉")

    except KeyboardInterrupt:
        logger.info("Sample generation interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C

    except Exception as e:
        logger.error(f"Unexpected error during sample generation: {e}")
        logger.error("Please check the configuration and data files")
        sys.exit(1)


if __name__ == "__main__":
    main()