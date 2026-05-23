"""
Enhanced visualization utilities for EDA Steps 1, 2, 3, and 5.

Provides advanced static plotting functions for:
- Product hierarchy tree diagrams (Step 1)
- Timeline validation charts (Step 1)
- Data coverage heatmaps (Step 2)
- Table relationship diagrams (Step 2)
- Data quality dashboards (Step 3)
- Price anomaly detection plots (Step 3)
- Feature distribution comparisons (Step 5)
- Temporal correlation heatmaps (Step 5)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, Any, List, Optional
from pathlib import Path
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Set consistent plot style
plt.style.use('default')
sns.set_palette("husl")

# Constants for consistent styling
PLOT_DPI = 300
FIGURE_SIZE = (12, 8)
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10


def plot_hierarchy_tree(hierarchy_stats: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Create product hierarchy tree diagram for Step 1.

    Shows categories → departments → items structure with item counts at each level.

    Parameters
    ----------
    hierarchy_stats : Dict[str, Any]
        Statistics containing category, department, and item hierarchies
    output_dir : str
        Directory to save the plot

    Returns
    -------
    Dict[str, Any]
        Plot creation results and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not hierarchy_stats.get('categories'):
        raise ValueError("hierarchy_stats must contain 'categories' data")

    # Create figure with appropriate size for tree diagram
    fig, ax = plt.subplots(figsize=(14, 10))

    categories = hierarchy_stats['categories']
    total_items = hierarchy_stats.get('total_items', 0)

    # Set up tree structure
    y_positions = {}
    y_current = 0.9

    # Root node
    ax.text(0.5, 0.95, f'M5 Product Hierarchy\n({total_items:,} Total Items)',
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

    # Category level
    cat_count = len(categories)
    cat_spacing = 0.8 / cat_count if cat_count > 1 else 0.4
    cat_start = 0.1 + (0.8 - cat_spacing * (cat_count - 1)) / 2

    for i, (cat_name, cat_data) in enumerate(categories.items()):
        cat_x = cat_start + i * cat_spacing
        cat_y = 0.75
        y_positions[cat_name] = (cat_x, cat_y)

        # Draw category node
        ax.text(cat_x, cat_y, f'{cat_name}\n({cat_data["item_count"]:,} items)',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

        # Draw line from root to category
        ax.plot([0.5, cat_x], [0.9, cat_y], 'k-', alpha=0.6, linewidth=2)

        # Department level
        departments = cat_data.get('departments', [])
        if departments:
            dept_count = len(departments)
            dept_spacing = cat_spacing * 0.8 / dept_count if dept_count > 1 else cat_spacing * 0.4
            dept_start = cat_x - dept_spacing * (dept_count - 1) / 2

            for j, dept_name in enumerate(departments):
                dept_x = dept_start + j * dept_spacing
                dept_y = 0.5

                # Get department item count if available
                dept_item_count = hierarchy_stats.get('departments', {}).get(dept_name, {}).get('item_count', '?')

                # Draw department node
                ax.text(dept_x, dept_y, f'{dept_name}\n({dept_item_count} items)',
                        ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))

                # Draw line from category to department
                ax.plot([cat_x, dept_x], [cat_y, dept_y], 'k-', alpha=0.5, linewidth=1)

    # Add summary statistics
    summary_text = f"""
    Hierarchy Summary:
    • {len(categories)} Categories
    • {len(hierarchy_stats.get('departments', {}))} Departments
    • {total_items:,} Items
    """

    ax.text(0.02, 0.25, summary_text, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Configure plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('M5 Product Hierarchy Tree Structure', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=20)

    # Save plot
    plot_path = output_path / "hierarchy_tree.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    return {
        'plot_path': str(plot_path),
        'summary': f'Hierarchy tree with {len(categories)} categories, {total_items:,} total items',
        'categories_plotted': len(categories),
        'total_items': total_items
    }


def plot_timeline_validation(temporal_stats: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Create timeline validation chart for Step 1.

    Shows training/validation period boundaries with gaps and overlaps highlighted.

    Parameters
    ----------
    temporal_stats : Dict[str, Any]
        Statistics containing date ranges and period information
    output_dir : str
        Directory to save the plot

    Returns
    -------
    Dict[str, Any]
        Plot creation results and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not temporal_stats.get('date_range') or not temporal_stats.get('periods'):
        raise ValueError("temporal_stats must contain 'date_range' and 'periods' data")

    fig, ax = plt.subplots(figsize=(14, 6))

    date_range = temporal_stats['date_range']
    periods = temporal_stats['periods']

    # Convert dates to datetime
    overall_start = pd.to_datetime(date_range['start'])
    overall_end = pd.to_datetime(date_range['end'])

    # Plot overall timeline
    ax.barh(0, (overall_end - overall_start).days, left=0, height=0.3,
            color='lightgray', alpha=0.5, label='Overall Dataset')

    # Plot training period
    if 'training' in periods:
        train_start = pd.to_datetime(periods['training']['start'])
        train_end = pd.to_datetime(periods['training']['end'])
        train_offset = (train_start - overall_start).days
        train_duration = (train_end - train_start).days

        ax.barh(1, train_duration, left=train_offset, height=0.4,
                color='green', alpha=0.7, label=f'Training ({train_duration} days)')

        # Add text annotation
        ax.text(train_offset + train_duration/2, 1.25,
                f'Training\n{train_start.strftime("%Y-%m-%d")} to {train_end.strftime("%Y-%m-%d")}',
                ha='center', va='bottom', fontsize=10)

    # Plot validation period
    if 'validation' in periods:
        val_start = pd.to_datetime(periods['validation']['start'])
        val_end = pd.to_datetime(periods['validation']['end'])
        val_offset = (val_start - overall_start).days
        val_duration = (val_end - val_start).days

        ax.barh(2, val_duration, left=val_offset, height=0.4,
                color='blue', alpha=0.7, label=f'Validation ({val_duration} days)')

        # Add text annotation
        ax.text(val_offset + val_duration/2, 2.25,
                f'Validation\n{val_start.strftime("%Y-%m-%d")} to {val_end.strftime("%Y-%m-%d")}',
                ha='center', va='bottom', fontsize=10)

    # Highlight gaps and overlaps if they exist
    gaps = temporal_stats.get('gaps', [])
    overlaps = temporal_stats.get('overlaps', [])

    if gaps:
        for gap in gaps:
            gap_start = pd.to_datetime(gap['start'])
            gap_end = pd.to_datetime(gap['end'])
            gap_offset = (gap_start - overall_start).days
            gap_duration = (gap_end - gap_start).days
            ax.barh(3, gap_duration, left=gap_offset, height=0.2,
                    color='red', alpha=0.8, label='Data Gap')

    if overlaps:
        for overlap in overlaps:
            overlap_start = pd.to_datetime(overlap['start'])
            overlap_end = pd.to_datetime(overlap['end'])
            overlap_offset = (overlap_start - overall_start).days
            overlap_duration = (overlap_end - overlap_start).days
            ax.barh(3.5, overlap_duration, left=overlap_offset, height=0.2,
                    color='orange', alpha=0.8, label='Period Overlap')

    # Configure plot
    ax.set_ylim(-0.5, 4)
    ax.set_xlabel('Days from Dataset Start', fontsize=LABEL_FONTSIZE)
    ax.set_title('M5 Dataset Timeline Validation', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add date labels on x-axis
    total_days = (overall_end - overall_start).days
    tick_positions = np.linspace(0, total_days, 6)
    tick_labels = [(overall_start + pd.Timedelta(days=int(pos))).strftime('%Y-%m-%d')
                   for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)

    # Remove y-axis ticks
    ax.set_yticks([])

    # Save plot
    plot_path = output_path / "timeline_validation.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    return {
        'plot_path': str(plot_path),
        'total_days': (overall_end - overall_start).days,
        'periods_validated': len(periods),
        'gaps_found': len(gaps),
        'overlaps_found': len(overlaps)
    }


def plot_data_coverage_heatmap(coverage_stats: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Create multi-table data coverage heatmap for Step 2.

    Shows missing data patterns across tables and time periods with color coding.

    Parameters
    ----------
    coverage_stats : Dict[str, Any]
        Statistics containing table coverage and temporal coverage data
    output_dir : str
        Directory to save the plot

    Returns
    -------
    Dict[str, Any]
        Plot creation results and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not coverage_stats.get('tables'):
        raise ValueError("coverage_stats must contain 'tables' data")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Overall table coverage
    tables = coverage_stats['tables']
    table_names = list(tables.keys())
    coverage_values = [tables[table]['coverage'] for table in table_names]
    missing_percentages = [tables[table]['missing_percentage'] for table in table_names]

    # Create coverage bar chart
    bars = ax1.bar(table_names, coverage_values, color=['green' if c >= 0.9 else 'orange' if c >= 0.7 else 'red' for c in coverage_values])
    ax1.set_ylabel('Coverage Percentage', fontsize=LABEL_FONTSIZE)
    ax1.set_title('Data Coverage by Table', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax1.set_ylim(0, 1)

    # Add percentage labels on bars
    for bar, coverage, missing in zip(bars, coverage_values, missing_percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{coverage:.1%}\n({missing:.1f}% missing)',
                ha='center', va='bottom', fontsize=10)

    # Plot 2: Temporal coverage heatmap
    temporal_coverage = coverage_stats.get('temporal_coverage', {})
    if temporal_coverage:
        years = list(temporal_coverage.keys())
        tables_in_temporal = list(next(iter(temporal_coverage.values())).keys())

        # Create coverage matrix
        coverage_matrix = []
        for year in years:
            year_coverage = [temporal_coverage[year].get(table, 0) for table in tables_in_temporal]
            coverage_matrix.append(year_coverage)

        coverage_matrix = np.array(coverage_matrix)

        # Create heatmap
        im = ax2.imshow(coverage_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(range(len(tables_in_temporal)))
        ax2.set_xticklabels(tables_in_temporal)
        ax2.set_yticks(range(len(years)))
        ax2.set_yticklabels(years)
        ax2.set_title('Coverage by Year and Table', fontsize=TITLE_FONTSIZE, fontweight='bold')

        # Add text annotations
        for i in range(len(years)):
            for j in range(len(tables_in_temporal)):
                text = ax2.text(j, i, f'{coverage_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black" if coverage_matrix[i, j] > 0.5 else "white")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Coverage Percentage', rotation=270, labelpad=15)
    else:
        ax2.text(0.5, 0.5, 'No temporal coverage data available',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Temporal Coverage (No Data)', fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Save plot
    plot_path = output_path / "data_coverage_heatmap.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    avg_coverage = np.mean(coverage_values)

    return {
        'plot_path': str(plot_path),
        'tables_analyzed': len(table_names),
        'average_coverage': avg_coverage,
        'lowest_coverage_table': table_names[np.argmin(coverage_values)],
        'temporal_periods': len(temporal_coverage) if temporal_coverage else 0
    }


def plot_table_relationships(relationship_stats: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Create table relationship diagram for Step 2.

    Shows connections between tables with join success rates and key columns.

    Parameters
    ----------
    relationship_stats : Dict[str, Any]
        Statistics containing table relationships and primary keys
    output_dir : str
        Directory to save the plot

    Returns
    -------
    Dict[str, Any]
        Plot creation results and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not relationship_stats.get('tables') or not relationship_stats.get('relationships'):
        raise ValueError("relationship_stats must contain 'tables' and 'relationships' data")

    fig, ax = plt.subplots(figsize=(12, 8))

    tables = relationship_stats['tables']
    relationships = relationship_stats['relationships']
    primary_keys = relationship_stats.get('primary_keys', {})

    # Position tables in a circular layout
    n_tables = len(tables)
    angles = np.linspace(0, 2*np.pi, n_tables, endpoint=False)
    radius = 0.4
    center = (0.5, 0.5)

    table_positions = {}
    for i, table in enumerate(tables):
        x = center[0] + radius * np.cos(angles[i])
        y = center[1] + radius * np.sin(angles[i])
        table_positions[table] = (x, y)

        # Draw table box
        keys_text = ', '.join(primary_keys.get(table, ['N/A'])[:3])  # Show up to 3 keys
        if len(primary_keys.get(table, [])) > 3:
            keys_text += ', ...'

        box_text = f'{table.upper()}\nKeys: {keys_text}'
        ax.text(x, y, box_text, ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

    # Draw relationships
    for rel_name, rel_data in relationships.items():
        # Parse relationship name to get table names
        if '_' in rel_name:
            table1, table2 = rel_name.split('_', 1)
        else:
            continue  # Skip if can't parse

        if table1 in table_positions and table2 in table_positions:
            x1, y1 = table_positions[table1]
            x2, y2 = table_positions[table2]

            # Draw connection line
            success_rate = rel_data['join_success_rate']
            line_color = 'green' if success_rate >= 0.9 else 'orange' if success_rate >= 0.7 else 'red'
            line_width = 2 + success_rate * 3  # Thicker lines for better join rates

            ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=line_width, alpha=0.7)

            # Add relationship label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            key_columns = ', '.join(rel_data.get('key_columns', ['Unknown']))
            ax.text(mid_x, mid_y, f'{success_rate:.1%}\n{key_columns}',
                   ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Add legend
    legend_elements = [
        mpatches.Patch(color='green', label='Join Success ≥ 90%'),
        mpatches.Patch(color='orange', label='Join Success 70-89%'),
        mpatches.Patch(color='red', label='Join Success < 70%')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    # Configure plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('M5 Table Relationships and Join Analysis', fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Save plot
    plot_path = output_path / "table_relationships.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    avg_join_success = np.mean([rel['join_success_rate'] for rel in relationships.values()])

    return {
        'plot_path': str(plot_path),
        'tables_analyzed': len(tables),
        'relationships_mapped': len(relationships),
        'average_join_success': avg_join_success
    }


def plot_data_quality_dashboard(quality_stats: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Create comprehensive data quality overview for Step 3.

    Multi-panel plot showing different quality metrics with anomaly counts.

    Parameters
    ----------
    quality_stats : Dict[str, Any]
        Statistics containing quality metrics, anomaly counts, and missing data
    output_dir : str
        Directory to save the plot

    Returns
    -------
    Dict[str, Any]
        Plot creation results and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not quality_stats.get('overall_score'):
        raise ValueError("quality_stats must contain 'overall_score'")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Overall Quality Score
    overall_score = quality_stats['overall_score']
    colors = ['red' if overall_score < 0.6 else 'orange' if overall_score < 0.8 else 'green']

    wedges, texts, autotexts = ax1.pie([overall_score, 1-overall_score],
                                       labels=['Quality Score', 'Issues'],
                                       colors=[colors[0], 'lightgray'],
                                       autopct=lambda pct: f'{pct:.1f}%' if pct == overall_score*100 else '',
                                       startangle=90)
    ax1.set_title(f'Overall Data Quality Score: {overall_score:.1%}',
                 fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Panel 2: Anomaly Counts
    anomaly_counts = quality_stats.get('anomaly_counts', {})
    if anomaly_counts:
        anomaly_types = list(anomaly_counts.keys())
        anomaly_values = list(anomaly_counts.values())

        bars = ax2.bar(anomaly_types, anomaly_values,
                      color=['red', 'orange', 'darkred'][:len(anomaly_types)])
        ax2.set_ylabel('Count', fontsize=LABEL_FONTSIZE)
        ax2.set_title('Anomaly Detection Results', fontsize=TITLE_FONTSIZE, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')

    # Panel 3: Missing Data by Table
    missing_data = quality_stats.get('missing_data', {})
    if missing_data:
        tables = list(missing_data.keys())
        missing_percentages = [missing_data[table] * 100 for table in tables]

        colors_missing = ['green' if p < 5 else 'orange' if p < 15 else 'red' for p in missing_percentages]
        bars = ax3.bar(tables, missing_percentages, color=colors_missing)
        ax3.set_ylabel('Missing Data %', fontsize=LABEL_FONTSIZE)
        ax3.set_title('Missing Data by Table', fontsize=TITLE_FONTSIZE, fontweight='bold')
        ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
        ax3.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='15% threshold')
        ax3.legend()

        # Add value labels
        for bar, pct in zip(bars, missing_percentages):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%',
                    ha='center', va='bottom')

    # Panel 4: Quality Metrics Breakdown
    quality_metrics = quality_stats.get('quality_metrics', {})
    if quality_metrics:
        metrics = list(quality_metrics.keys())
        values = list(quality_metrics.values())

        y_pos = np.arange(len(metrics))
        bars = ax4.barh(y_pos, values, color=['green' if v >= 0.8 else 'orange' if v >= 0.6 else 'red' for v in values])
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(metrics)
        ax4.set_xlabel('Score', fontsize=LABEL_FONTSIZE)
        ax4.set_title('Quality Metrics Breakdown', fontsize=TITLE_FONTSIZE, fontweight='bold')
        ax4.set_xlim(0, 1)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{value:.2f}',
                    ha='left', va='center')

    # Save plot
    plot_path = output_path / "data_quality_dashboard.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    total_anomalies = sum(anomaly_counts.values()) if anomaly_counts else 0

    return {
        'plot_path': str(plot_path),
        'overall_score': overall_score,
        'total_anomalies': total_anomalies,
        'tables_with_missing_data': len(missing_data) if missing_data else 0
    }


def plot_price_anomaly_detection(price_stats: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Create price anomaly detection plots for Step 3.

    Shows price distributions and outlier identification with anomaly markers.

    Parameters
    ----------
    price_stats : Dict[str, Any]
        Statistics containing price distributions and anomaly information
    output_dir : str
        Directory to save the plot

    Returns
    -------
    Dict[str, Any]
        Plot creation results and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not price_stats.get('price_distribution') or not price_stats.get('anomalies'):
        raise ValueError("price_stats must contain 'price_distribution' and 'anomalies' data")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    price_dist = price_stats['price_distribution']
    anomalies = price_stats['anomalies']

    # Panel 1: Price Distribution Statistics
    stats_labels = ['Mean', 'Std', 'Min', 'Max']
    stats_values = [price_dist['mean'], price_dist['std'], price_dist['min'], price_dist['max']]

    bars = ax1.bar(stats_labels, stats_values, color=['blue', 'green', 'orange', 'red'])
    ax1.set_ylabel('Price ($)', fontsize=LABEL_FONTSIZE)
    ax1.set_title('Price Distribution Statistics', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax1.set_yscale('log')  # Log scale due to wide range

    # Add value labels
    for bar, value in zip(bars, stats_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${value:.2f}',
                ha='center', va='bottom', rotation=0 if value < 100 else 90)

    # Panel 2: Percentile Distribution
    percentiles = price_dist.get('percentiles', {})
    if percentiles:
        perc_labels = list(percentiles.keys())
        perc_values = list(percentiles.values())

        ax2.plot(perc_labels, perc_values, 'bo-', linewidth=2, markersize=8)
        ax2.set_ylabel('Price ($)', fontsize=LABEL_FONTSIZE)
        ax2.set_xlabel('Percentile', fontsize=LABEL_FONTSIZE)
        ax2.set_title('Price Percentile Distribution', fontsize=TITLE_FONTSIZE, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for label, value in zip(perc_labels, perc_values):
            ax2.annotate(f'${value:.2f}', (label, value), textcoords="offset points",
                        xytext=(0,10), ha='center')

    # Panel 3: Anomaly Counts
    anomaly_types = []
    anomaly_counts = []

    for anom_type, anom_data in anomalies.items():
        if isinstance(anom_data, dict) and 'count' in anom_data:
            anomaly_types.append(anom_type.replace('_', ' ').title())
            anomaly_counts.append(anom_data['count'])

    if anomaly_types:
        colors_anom = ['red', 'orange', 'darkred'][:len(anomaly_types)]
        bars = ax3.bar(anomaly_types, anomaly_counts, color=colors_anom)
        ax3.set_ylabel('Count', fontsize=LABEL_FONTSIZE)
        ax3.set_title('Price Anomalies Detected', fontsize=TITLE_FONTSIZE, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, count in zip(bars, anomaly_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom')

    # Panel 4: Anomaly Examples
    ax4.axis('off')

    # Create text summary of anomaly examples
    summary_text = "Anomaly Examples:\n\n"

    for anom_type, anom_data in anomalies.items():
        if isinstance(anom_data, dict) and 'examples' in anom_data:
            examples = anom_data['examples'][:3]  # Show up to 3 examples
            examples_str = ', '.join([f'${ex:.2f}' for ex in examples])
            summary_text += f"• {anom_type.replace('_', ' ').title()}: {examples_str}\n"

            if 'threshold' in anom_data:
                summary_text += f"  (Threshold: ${anom_data['threshold']:.2f})\n"
            elif 'threshold_ratio' in anom_data:
                summary_text += f"  (Ratio threshold: {anom_data['threshold_ratio']:.1f}x)\n"
            summary_text += "\n"

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    ax4.set_title('Detailed Anomaly Information', fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Save plot
    plot_path = output_path / "price_anomaly_detection.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    total_anomalies = sum([anom['count'] for anom in anomalies.values() if isinstance(anom, dict) and 'count' in anom])

    return {
        'plot_path': str(plot_path),
        'total_anomalies': total_anomalies,
        'anomaly_types': len([anom for anom in anomalies.values() if isinstance(anom, dict) and 'count' in anom]),
        'price_range': price_dist['max'] - price_dist['min']
    }


def plot_feature_distributions(feature_stats: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Create feature distribution comparison plots for Step 5.

    Shows categorical and geographic feature distributions with performance variations.

    Parameters
    ----------
    feature_stats : Dict[str, Any]
        Statistics containing categorical and geographic feature distributions
    output_dir : str
        Directory to save the plot

    Returns
    -------
    Dict[str, Any]
        Plot creation results and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    plot_count = 0

    # Plot categorical features
    categorical_features = feature_stats.get('categorical_features', {})
    for feature_name, feature_data in categorical_features.items():
        if plot_count >= 4:
            break

        ax = axes[plot_count]

        categories = list(feature_data.keys())
        counts = [feature_data[cat]['count'] for cat in categories]
        avg_sales = [feature_data[cat]['avg_sales'] for cat in categories]

        # Create dual-axis plot
        ax2 = ax.twinx()

        # Bar plot for counts
        bars = ax.bar(categories, counts, alpha=0.7, color='lightblue', label='Item Count')
        ax.set_ylabel('Item Count', fontsize=LABEL_FONTSIZE, color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        # Line plot for average sales
        line = ax2.plot(categories, avg_sales, 'ro-', linewidth=2, markersize=8, label='Avg Sales')
        ax2.set_ylabel('Average Sales', fontsize=LABEL_FONTSIZE, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_title(f'{feature_name.replace("_", " ").title()} Distribution',
                    fontsize=TITLE_FONTSIZE, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', color='blue')

        for i, (cat, sales) in enumerate(zip(categories, avg_sales)):
            ax2.annotate(f'{sales:.1f}', (i, sales), textcoords="offset points",
                        xytext=(0,10), ha='center', color='red')

        plot_count += 1

    # Plot geographic features
    geographic_features = feature_stats.get('geographic_features', {})
    if geographic_features and plot_count < 4:
        performance_by_state = geographic_features.get('performance_by_state', {})
        if performance_by_state:
            ax = axes[plot_count]

            states = list(performance_by_state.keys())
            sales_volumes = [performance_by_state[state]['sales_volume'] for state in states]
            store_counts = [performance_by_state[state]['store_count'] for state in states]

            # Create dual-axis plot
            ax2 = ax.twinx()

            # Bar plot for sales volume
            bars = ax.bar(states, sales_volumes, alpha=0.7, color='lightgreen', label='Sales Volume')
            ax.set_ylabel('Sales Volume', fontsize=LABEL_FONTSIZE, color='green')
            ax.tick_params(axis='y', labelcolor='green')

            # Line plot for store count
            line = ax2.plot(states, store_counts, 'mo-', linewidth=2, markersize=8, label='Store Count')
            ax2.set_ylabel('Store Count', fontsize=LABEL_FONTSIZE, color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')

            ax.set_title('Geographic Performance by State', fontsize=TITLE_FONTSIZE, fontweight='bold')

            # Add value labels
            for bar, volume in zip(bars, sales_volumes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(volume):,}', ha='center', va='bottom', color='green')

            for i, (state, count) in enumerate(zip(states, store_counts)):
                ax2.annotate(f'{count}', (i, count), textcoords="offset points",
                            xytext=(0,10), ha='center', color='purple')

            plot_count += 1

    # Hide unused subplots
    for i in range(plot_count, 4):
        axes[i].axis('off')

    # Save plot
    plot_path = output_path / "feature_distributions.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    total_categories = sum(len(features) for features in categorical_features.values())

    return {
        'plot_path': str(plot_path),
        'categorical_features_plotted': len(categorical_features),
        'total_categories': total_categories,
        'geographic_analysis': len(geographic_features) > 0
    }


def plot_temporal_correlations(correlation_stats: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Create temporal correlation heatmaps for Step 5.

    Shows correlations between temporal features and sales with significance levels.

    Parameters
    ----------
    correlation_stats : Dict[str, Any]
        Statistics containing correlation matrix and significance levels
    output_dir : str
        Directory to save the plot

    Returns
    -------
    Dict[str, Any]
        Plot creation results and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not correlation_stats.get('correlation_matrix'):
        raise ValueError("correlation_stats must contain 'correlation_matrix' data")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    correlation_matrix = correlation_stats['correlation_matrix']
    temporal_features = correlation_stats.get('temporal_features', list(correlation_matrix['sales'].keys()))

    # Create correlation matrix for heatmap
    corr_data = []
    feature_names = []

    for feature in temporal_features:
        if feature in correlation_matrix['sales']:
            corr_data.append(correlation_matrix['sales'][feature])
            feature_names.append(feature.replace('_', ' ').title())

    corr_matrix = np.array(corr_data).reshape(-1, 1)

    # Plot 1: Correlation heatmap
    im = ax1.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xticks([0])
    ax1.set_xticklabels(['Sales'])
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names)
    ax1.set_title('Temporal Feature Correlations with Sales', fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Add correlation values as text
    for i, corr_val in enumerate(corr_data):
        color = 'white' if abs(corr_val) > 0.3 else 'black'
        ax1.text(0, i, f'{corr_val:.3f}', ha='center', va='center', color=color, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)

    # Plot 2: Significance levels
    significance_levels = correlation_matrix.get('significance_levels', {})
    if significance_levels:
        sig_data = []
        sig_labels = []

        for feature in temporal_features:
            if feature in significance_levels:
                sig_level = significance_levels[feature]
                sig_data.append(sig_level)
                sig_labels.append(feature.replace('_', ' ').title())

        # Create significance plot
        colors = ['green' if p <= 0.01 else 'orange' if p <= 0.05 else 'red' for p in sig_data]
        bars = ax2.barh(range(len(sig_labels)), sig_data, color=colors)
        ax2.set_yticks(range(len(sig_labels)))
        ax2.set_yticklabels(sig_labels)
        ax2.set_xlabel('P-value', fontsize=LABEL_FONTSIZE)
        ax2.set_title('Statistical Significance of Correlations', fontsize=TITLE_FONTSIZE, fontweight='bold')
        ax2.axvline(x=0.05, color='orange', linestyle='--', alpha=0.7, label='p=0.05')
        ax2.axvline(x=0.01, color='green', linestyle='--', alpha=0.7, label='p=0.01')
        ax2.legend()
        ax2.set_xscale('log')

        # Add p-value labels
        for i, (bar, p_val) in enumerate(zip(bars, sig_data)):
            width = bar.get_width()
            ax2.text(width * 1.1, bar.get_y() + bar.get_height()/2.,
                    f'{p_val:.4f}',
                    ha='left', va='center', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No significance data available',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Statistical Significance (No Data)', fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Save plot
    plot_path = output_path / "temporal_correlations.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    # Calculate summary statistics
    significant_correlations = sum(1 for p in significance_levels.values() if p <= 0.05) if significance_levels else 0
    strong_correlations = sum(1 for corr in corr_data if abs(corr) >= 0.3)

    return {
        'plot_path': str(plot_path),
        'features_analyzed': len(temporal_features),
        'significant_correlations': significant_correlations,
        'strong_correlations': strong_correlations,
        'max_correlation': max(corr_data) if corr_data else 0
    }