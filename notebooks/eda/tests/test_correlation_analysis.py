"""
Tests for correlation analysis module.

This module contains tests for categorical sales pattern analysis
and correlation analysis functionality.
"""

import pandas as pd
import pytest
import sys
import os

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.correlation_analysis import analyze_categorical_sales_patterns


def test_analyze_categorical_sales_patterns():
    """
    Test basic categorical sales pattern analysis functionality.

    This test verifies that the analyze_categorical_sales_patterns function
    can correctly analyze sales patterns by category and return meaningful
    statistical summaries.
    """
    # Create sample data
    sample_data = pd.DataFrame({
        'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'sales': [100, 150, 120, 200, 250, 220, 50, 75, 60]
    })

    # Call the function
    result = analyze_categorical_sales_patterns(sample_data, 'category', 'sales')

    # Verify result is a dictionary
    assert isinstance(result, dict), "Result should be a dictionary"

    # Verify result contains expected keys
    assert 'categories' in result, "Result should contain 'categories' key"
    assert 'summary_stats' in result, "Result should contain 'summary_stats' key"

    # Verify categories are identified
    assert len(result['categories']) == 3, "Should identify 3 categories"

    # Verify summary stats have required fields
    assert 'mean' in result['summary_stats'], "Summary stats should contain 'mean'"
    assert 'std' in result['summary_stats'], "Summary stats should contain 'std'"
    assert 'count' in result['summary_stats'], "Summary stats should contain 'count'"
