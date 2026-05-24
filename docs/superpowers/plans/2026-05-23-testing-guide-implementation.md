# Testing Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a comprehensive testing guide that teaches testing concepts through actual EDA test examples using a tutorial-style approach.

**Architecture:** Tutorial-based documentation that starts with concrete test code examples and generalizes concepts naturally. Uses real project test files as teaching material with annotated code examples and cross-references.

**Tech Stack:** Markdown documentation, Python code examples extracted from existing test files, pytest framework examples

---

### Task 1: Setup Documentation Structure

**Files:**
- Create: `docs/testing/testing-guide-comprehensive.md`
- Create: `docs/testing/examples/unit-test-examples.py`
- Create: `docs/testing/examples/integration-test-examples.py`  
- Create: `docs/testing/examples/edge-case-examples.py`
- Create: `docs/testing/quick-reference.md`

- [ ] **Step 1: Create testing documentation directory structure**

```bash
mkdir -p docs/testing/examples
```

- [ ] **Step 2: Create main tutorial document with header**

File: `docs/testing/testing-guide-comprehensive.md`

```markdown
# Testing Guide: Learn by Example

A beginner-friendly guide to testing concepts using real examples from the Demand Forecast Intelligence project's EDA testing suite.

## Table of Contents

1. [Foundation with Real Code](#part-1-foundation-with-real-code)
2. [Testing Types in Action](#part-2-testing-types-in-action)  
3. [Testing Patterns Across Your Project](#part-3-testing-patterns-across-your-project)

---

*This guide uses actual test files from our project. All examples are real, working code that you can find and run in the codebase.*
```

- [ ] **Step 3: Create placeholder example files**

File: `docs/testing/examples/unit-test-examples.py`

```python
"""
Unit test examples extracted from the EDA testing suite.
These are real, working examples from notebooks/eda/tests/
"""

# Examples will be populated in subsequent tasks
```

- [ ] **Step 4: Commit initial structure**

```bash
git add docs/testing/
git commit -m "docs: create testing guide directory structure

Initialize tutorial-style testing documentation framework"
```

### Task 2: Part 1 - Foundation with Real Code (Section 1.1)

**Files:**
- Modify: `docs/testing/testing-guide-comprehensive.md`
- Modify: `docs/testing/examples/unit-test-examples.py`

- [ ] **Step 1: Extract and annotate the main teaching example**

File: `docs/testing/examples/unit-test-examples.py`

```python
"""
Unit test examples extracted from the EDA testing suite.
"""

# Example 1: Basic unit test structure from test_temporal_analysis.py
def test_analyze_time_structure_basic():
    """
    TEACHING EXAMPLE: This is a unit test that tests ONE function in isolation.
    
    Structure:
    1. Setup: Create test data (sales_data, calendar_data)
    2. Execute: Call the function we want to test  
    3. Assert: Check that the result matches our expectations
    """
    # SETUP: Create sample sales data (this is test data, not real M5 data)
    sales_data = pd.DataFrame({
        'id': ['FOODS_1_001_CA_1_validation'],
        'd_1': [5],    # Day 1 sales: 5 units
        'd_2': [3],    # Day 2 sales: 3 units  
        'd_3': [7]     # Day 3 sales: 7 units
    })

    # SETUP: Create sample calendar data to match
    calendar_data = pd.DataFrame({
        'd': ['d_1', 'd_2', 'd_3'],
        'date': pd.to_datetime(['2011-01-29', '2011-01-30', '2011-01-31'])
    })

    # EXECUTE: Call the function we're testing
    result = analyze_time_structure(sales_data, calendar_data)

    # ASSERT: Check that the function returns what we expect
    assert isinstance(result, dict)           # Should return a dictionary
    assert 'time_range' in result            # Should have time_range key
    assert 'frequency_validation' in result  # Should have frequency info
    assert result['total_days'] == 3         # Should count 3 days correctly
    assert result['frequency'] == 'Daily'    # Should detect daily frequency
```

- [ ] **Step 2: Write Section 1.1 - Let's Look at a Real Test**

File: `docs/testing/testing-guide-comprehensive.md` (append after table of contents)

```markdown
## Part 1: Foundation with Real Code

### Section 1.1: "Let's Look at a Real Test"

Let's start by examining an actual test from our project. Here's a real test from [`notebooks/eda/tests/test_temporal_analysis.py`](../../notebooks/eda/tests/test_temporal_analysis.py):

```python
def test_analyze_time_structure_basic():
    # Create sample sales data
    sales_data = pd.DataFrame({
        'id': ['FOODS_1_001_CA_1_validation'],
        'd_1': [5],
        'd_2': [3], 
        'd_3': [7]
    })

    # Create sample calendar data
    calendar_data = pd.DataFrame({
        'd': ['d_1', 'd_2', 'd_3'],
        'date': pd.to_datetime(['2011-01-29', '2011-01-30', '2011-01-31'])
    })

    result = analyze_time_structure(sales_data, calendar_data)

    assert isinstance(result, dict)
    assert 'time_range' in result
    assert result['total_days'] == 3
    assert result['frequency'] == 'Daily'
```

**What's happening here?** This test is checking that our `analyze_time_structure()` function correctly processes M5 dataset format and returns the right information about time periods.

**The anatomy of this test:**

1. **Setup** (lines 2-12): Create fake but realistic data that looks like the M5 dataset
2. **Execute** (line 14): Call the function we want to test with our test data  
3. **Assert** (lines 16-19): Check that the function behaves correctly

**Why use fake data?** We create small, controlled test data instead of using the real 30,490-row M5 dataset because:
- Tests run faster (3 rows vs 30,490 rows)
- We control exactly what inputs the function receives
- We can predict exactly what outputs it should produce
- If the test fails, we know it's a code problem, not a data problem

Try running this test yourself:
```bash
cd /Users/rahul.vansh/Documents/Personal/demand_forecast_intelligence
pytest notebooks/eda/tests/test_temporal_analysis.py::test_analyze_time_structure_basic -v
```
```

- [ ] **Step 3: Commit Section 1.1**

```bash
git add docs/testing/
git commit -m "docs: add section 1.1 - walkthrough of real test example

Introduce test anatomy using actual temporal analysis test"
```

### Task 3: Part 1 - Foundation with Real Code (Section 1.2)

**Files:**
- Modify: `docs/testing/testing-guide-comprehensive.md`

- [ ] **Step 1: Write Section 1.2 - What Type of Test Was That?**

File: `docs/testing/testing-guide-comprehensive.md` (append)

```markdown
### Section 1.2: "What Type of Test Was That?"

The test we just looked at is called a **unit test**. Here's why:

**Unit Test Definition**: A test that examines one function (or "unit" of code) in isolation, without depending on other parts of the system.

**How our example fits this definition:**
- ✅ **Tests one function**: Only `analyze_time_structure()` is being tested
- ✅ **Uses controlled inputs**: We provide exact test data, not real files
- ✅ **Independent**: This test doesn't call other functions in our EDA pipeline
- ✅ **Predictable**: Same input always produces same output

**The Testing Pyramid**

Unit tests form the foundation of the "testing pyramid":

```
        /\
       /  \    E2E Tests (Few)
      /    \   Integration Tests (Some)  
     /      \  Unit Tests (Many)
    /________\
```

**Why many unit tests?**
- **Fast**: Testing 3 rows takes milliseconds vs seconds for full datasets
- **Focused**: When a unit test fails, you know exactly which function broke
- **Reliable**: No external dependencies means tests don't randomly fail
- **Documentation**: Tests show exactly how functions should be used

**In our EDA context**, unit tests catch problems like:
- Time structure validation returning wrong date ranges
- Statistical calculations producing NaN values  
- Data type mismatches between M5 format expectations

Each unit test is like having a specialist inspector who only checks one specific thing, but checks it very thoroughly.
```

- [ ] **Step 2: Commit Section 1.2**

```bash
git add docs/testing/testing-guide-comprehensive.md
git commit -m "docs: add section 1.2 - unit testing concepts

Define unit testing using the temporal analysis example"
```

### Task 4: Part 1 - Foundation with Real Code (Section 1.3)

**Files:**
- Modify: `docs/testing/testing-guide-comprehensive.md`

- [ ] **Step 1: Write Section 1.3 - Why This Matters for EDA**

File: `docs/testing/testing-guide-comprehensive.md` (append)

```markdown
### Section 1.3: "Why This Matters for EDA"

**Business Impact**: In demand forecasting, incorrect time structure analysis leads to wrong predictions, which means:
- Overstocking costs money (excess inventory)
- Understocking loses sales (empty shelves)  
- At Walmart scale, even 1% accuracy improvement = millions of dollars

**Real Consequences of Untested EDA Code**:

**Scenario 1**: `analyze_time_structure()` has a bug that miscounts days
- Bug: Function returns `total_days: 1912` instead of correct `1913` 
- Impact: Forecasting model trains on wrong time window
- Result: All predictions shifted by 1 day, inventory decisions based on wrong timeline

**Scenario 2**: Time frequency validation fails silently  
- Bug: Function doesn't detect missing weekends in data
- Impact: Model assumes daily frequency when it's actually weekdays-only
- Result: Weekend demand predictions are completely wrong

**How Our Test Prevents This**:

```python
assert result['total_days'] == 3         # Catches day counting bugs
assert result['frequency'] == 'Daily'    # Catches frequency detection bugs
```

If someone changes the `analyze_time_structure()` function and introduces a bug, this test will fail immediately. No broken code reaches production.

**Project-Specific Benefits**:

1. **M5 Dataset Confidence**: M5 has 1,913 days of data across 3,049 items. Manual verification is impossible—tests ensure our processing is correct.

2. **Refactoring Safety**: When we optimize EDA functions for performance, tests prove we didn't break functionality.

3. **Team Collaboration**: New developers can modify EDA code confidently, knowing tests will catch mistakes.

4. **Debugging Speed**: When something breaks, tests help isolate whether the problem is in data loading, time analysis, statistical calculations, or visualization.

**The EDA Pipeline Trust Chain**: Each tested function becomes a reliable building block for the next function. Time structure → seasonal patterns → trend analysis → forecasting. If the foundation is solid (tested), the whole pipeline is solid.
```

- [ ] **Step 2: Commit Section 1.3**

```bash
git add docs/testing/testing-guide-comprehensive.md  
git commit -m "docs: add section 1.3 - EDA testing business justification

Connect testing concepts to demand forecasting business impact"
```

### Task 5: Part 2 - Testing Types in Action (Section 2.1)

**Files:**
- Modify: `docs/testing/testing-guide-comprehensive.md`
- Modify: `docs/testing/examples/unit-test-examples.py`

- [ ] **Step 1: Extract more unit test examples**

File: `docs/testing/examples/unit-test-examples.py` (append)

```python
# Example 2: Multiple test methods for one function (from TestTimeStructureAnalysis class)

def test_time_structure_with_actual_m5_format():
    """
    TEACHING EXAMPLE: Testing the same function with MORE REALISTIC data
    
    This shows how we test edge cases and different scenarios for the same function.
    Notice: Still a unit test because we only test analyze_time_structure().
    """
    # More realistic M5-style data with all the columns
    sales_data = pd.DataFrame({
        'id': ['FOODS_1_001_CA_1_validation', 'HOUSEHOLD_1_001_TX_1_validation'],
        'item_id': ['FOODS_1_001', 'HOUSEHOLD_1_001'],
        'cat_id': ['FOODS', 'HOUSEHOLD'],
        'dept_id': ['FOODS_1', 'HOUSEHOLD_1'], 
        'store_id': ['CA_1', 'TX_1'],
        'state_id': ['CA', 'TX'],
        'd_1': [5, 2],
        'd_2': [3, 1],
        'd_3': [7, 0]
    })

    calendar_data = pd.DataFrame({
        'd': ['d_1', 'd_2', 'd_3'],
        'date': pd.to_datetime(['2011-01-29', '2011-01-30', '2011-01-31'])
    })

    result = analyze_time_structure(sales_data, calendar_data)

    # Different assertions for this scenario
    assert result['total_days'] == 3
    assert result['total_series'] == 2  # Now we have 2 product-store combinations
    assert result['frequency_validation']['structure_consistent'] is True
    assert result['panel_structure']['entities'] == 2
    assert result['panel_structure']['time_periods'] == 3

def test_time_structure_with_missing_dates():
    """
    TEACHING EXAMPLE: Testing EDGE CASES - what happens with problematic data?
    
    This is still a unit test, but it's testing how our function handles
    bad or unusual inputs.
    """
    sales_data = pd.DataFrame({
        'd_1': [10],
        'd_2': [12], 
        'd_3': [15],
        'd_4': [18]
    })

    # Notice: Missing Feb 1st (gap in dates)
    calendar_data = pd.DataFrame({
        'd': ['d_1', 'd_2', 'd_3', 'd_4'],
        'date': pd.to_datetime(['2011-01-29', '2011-01-31', '2011-02-02', '2011-02-04'])
    })

    result = analyze_time_structure(sales_data, calendar_data)

    # Should still work, but detect the missing dates
    assert 'time_range' in result
    assert result['missing_dates'] >= 0  # Should count missing dates
```

- [ ] **Step 2: Write Section 2.1 - Unit Testing Deep Dive**

File: `docs/testing/testing-guide-comprehensive.md` (append)

```markdown
## Part 2: Testing Types in Action

### Section 2.1: Unit Testing Deep Dive

Let's look at how our project uses **test classes** to organize multiple unit tests for the same function. In [`test_temporal_analysis.py`](../../notebooks/eda/tests/test_temporal_analysis.py), you'll find:

```python
class TestTimeStructureAnalysis:
    """Tests for time structure validation and analysis."""

    def test_analyze_time_structure_basic(self):
        # Basic scenario (we saw this already)

    def test_time_structure_with_actual_m5_format(self):  
        # More realistic M5 data format

    def test_time_structure_with_missing_dates(self):
        # Edge case: what if dates are missing?
```

**Why multiple tests for one function?** Each test covers a different **scenario**:

1. **Happy path**: Normal, expected inputs (`test_analyze_time_structure_basic`)
2. **Realistic data**: Full M5 format with all columns (`test_time_structure_with_actual_m5_format`)  
3. **Edge cases**: Problematic inputs (`test_time_structure_with_missing_dates`)

**Test Independence**: Each test can run alone. If `test_time_structure_with_missing_dates` fails, the other tests still pass. This helps you pinpoint exactly which scenario is broken.

**Pattern Recognition**: Look for this pattern across our EDA tests:

```
TestTimeStructureAnalysis     → Tests analyze_time_structure()
TestSeasonalPatternDetection  → Tests detect_seasonal_patterns()  
TestTrendAnalysis            → Tests analyze_trend_components()
TestAutocorrelationAnalysis  → Tests compute_autocorrelation_analysis()
```

Each class focuses on **one function** but tests it **thoroughly** with multiple scenarios.

**Running specific test classes**:
```bash
# Run all time structure tests
pytest notebooks/eda/tests/test_temporal_analysis.py::TestTimeStructureAnalysis -v

# Run just the edge case test
pytest notebooks/eda/tests/test_temporal_analysis.py::TestTimeStructureAnalysis::test_time_structure_with_missing_dates -v
```

**What makes a good unit test?**
- ✅ **Fast**: Runs in milliseconds
- ✅ **Isolated**: Doesn't depend on files, databases, or other functions
- ✅ **Repeatable**: Same result every time  
- ✅ **Self-checking**: Pass/fail is automatic (assert statements)
- ✅ **Focused**: Tests one specific behavior
```

- [ ] **Step 3: Commit Section 2.1**

```bash
git add docs/testing/
git commit -m "docs: add section 2.1 - unit testing deep dive

Show test class organization and multiple test scenarios"
```

### Task 6: Part 2 - Testing Types in Action (Section 2.2)

**Files:**
- Modify: `docs/testing/testing-guide-comprehensive.md`
- Modify: `docs/testing/examples/integration-test-examples.py`

- [ ] **Step 1: Extract integration test example**

File: `docs/testing/examples/integration-test-examples.py`

```python
"""
Integration test examples extracted from the EDA testing suite.
These test multiple functions working together.
"""

def test_full_temporal_analysis_pipeline():
    """
    TEACHING EXAMPLE: Integration test - multiple functions working together
    
    Unlike unit tests that test ONE function, this integration test checks that
    multiple functions in our EDA pipeline can work together correctly.
    
    Functions tested together:
    1. analyze_time_structure() 
    2. detect_seasonal_patterns()
    3. analyze_trend_components()
    4. compute_autocorrelation_analysis()
    """
    # Setup: Create realistic M5-style data (larger dataset for integration test)
    np.random.seed(42)  # Reproducible random data
    sales_data = pd.DataFrame({
        'id': ['FOODS_1_001_CA_1_validation', 'HOUSEHOLD_1_001_TX_1_validation'],
        'item_id': ['FOODS_1_001', 'HOUSEHOLD_1_001'],
        'cat_id': ['FOODS', 'HOUSEHOLD'],
        'dept_id': ['FOODS_1', 'HOUSEHOLD_1'],
        'store_id': ['CA_1', 'TX_1'],
        'state_id': ['CA', 'TX'],
        # 100 days of data (vs 3 days in unit tests)
        **{f'd_{i}': [np.random.randint(5, 50), np.random.randint(1, 20)]
           for i in range(1, 101)}
    })

    calendar_data = pd.DataFrame({
        'd': [f'd_{i}' for i in range(1, 101)],
        'date': pd.date_range('2011-01-29', periods=100)
    })

    # Execute: Run the complete temporal analysis workflow
    time_structure = analyze_time_structure(sales_data, calendar_data)
    seasonal_patterns = detect_seasonal_patterns(sales_data, calendar_data) 
    trend_analysis = analyze_trend_components(sales_data, calendar_data)
    autocorr_analysis = compute_autocorrelation_analysis(sales_data, max_lags=50)

    # Assert: Check that all functions produced expected results
    # AND that they work together (no data type mismatches, etc.)
    assert time_structure['total_days'] == 100
    assert 'FOODS' in seasonal_patterns['seasonal_patterns']
    assert 'linear_trend' in trend_analysis  
    assert 'autocorrelations' in autocorr_analysis
    
    # Integration-specific check: Results should be consistent
    # Time structure says 100 days, trend analysis should use 100 days too
    assert time_structure['total_days'] == len(trend_analysis['time_series_data'])
```

- [ ] **Step 2: Write Section 2.2 - Integration Testing Walkthrough**

File: `docs/testing/testing-guide-comprehensive.md` (append)

```markdown
### Section 2.2: Integration Testing Walkthrough

Now let's look at a different type of test: an **integration test**. Here's a real example from our project:

```python
def test_full_temporal_analysis_pipeline():
    """Test complete temporal analysis workflow."""
    # ... create test data ...
    
    # Run ALL temporal analysis functions together
    time_structure = analyze_time_structure(sales_data, calendar_data)
    seasonal_patterns = detect_seasonal_patterns(sales_data, calendar_data)
    trend_analysis = analyze_trend_components(sales_data, calendar_data) 
    autocorr_analysis = compute_autocorrelation_analysis(sales_data, max_lags=50)
    
    # Check that everything worked
    assert time_structure['total_days'] == 100
    assert 'FOODS' in seasonal_patterns['seasonal_patterns'] 
    assert 'linear_trend' in trend_analysis
    assert 'autocorrelations' in autocorr_analysis
```

**Integration Test Definition**: A test that checks multiple functions or components working together as a system.

**Key Differences from Unit Tests**:

| Unit Test | Integration Test |
|-----------|------------------|
| Tests **one function** | Tests **multiple functions together** |
| Small, controlled data (3 rows) | Larger, realistic data (100 rows) |
| Runs in milliseconds | Runs in seconds |
| Catches function-level bugs | Catches interaction bugs |

**What Integration Tests Catch**:

1. **Data Flow Problems**: Function A returns a DataFrame, but Function B expects a Series
2. **Type Mismatches**: Time structure returns dates as strings, but seasonal analysis expects datetime objects  
3. **Performance Issues**: Individual functions are fast, but together they're too slow
4. **Missing Dependencies**: Function B needs results from Function A to work correctly

**Real Example from Our EDA Pipeline**:
- `analyze_time_structure()` identifies the date range
- `detect_seasonal_patterns()` uses that date range to compute seasonality
- If time structure returns wrong dates, seasonal analysis fails
- Unit tests won't catch this—they test functions separately
- Integration test catches this because it runs them together

**When to Write Integration Tests**:
- ✅ After you have working unit tests
- ✅ When functions need to work together in sequence  
- ✅ When testing end-to-end workflows (like "complete EDA analysis")
- ✅ When data flows between multiple functions

**Running Integration Tests**:
```bash
# Run integration tests (slower, but more comprehensive)
pytest notebooks/eda/tests/test_temporal_analysis.py::TestTemporalAnalysisIntegration -v
```
```

- [ ] **Step 3: Commit Section 2.2**

```bash
git add docs/testing/
git commit -m "docs: add section 2.2 - integration testing walkthrough  

Explain integration testing using temporal analysis pipeline example"
```

### Task 7: Part 2 - Testing Types in Action (Section 2.3)

**Files:**
- Modify: `docs/testing/testing-guide-comprehensive.md`
- Modify: `docs/testing/examples/edge-case-examples.py`

- [ ] **Step 1: Extract functional and non-functional test examples**

File: `docs/testing/examples/edge-case-examples.py`

```python
"""
Edge case and non-functional test examples from the EDA testing suite.
"""

# FUNCTIONAL TESTING EXAMPLE: Testing business logic correctness
def test_seasonal_patterns_multiple_categories():
    """
    FUNCTIONAL TEST: Does seasonal detection work correctly for business categories?
    
    This tests the BUSINESS LOGIC: "Different product categories should have 
    different seasonal patterns, and our function should detect them correctly."
    """
    # Create data with intentional seasonal patterns
    np.random.seed(42)
    foods_data = [np.random.randint(8, 15) for _ in range(5)]      # Higher baseline sales
    household_data = [np.random.randint(2, 8) for _ in range(5)]  # Lower baseline sales

    sales_data = pd.DataFrame({
        'cat_id': ['FOODS'] * 5 + ['HOUSEHOLD'] * 5,
        **{f'd_{i}': foods_data + household_data for i in range(1, 31)}
    })

    calendar_data = pd.DataFrame({
        'd': [f'd_{i}' for i in range(1, 31)]
    })

    result = detect_seasonal_patterns(sales_data, calendar_data, hierarchy_level='category')

    # FUNCTIONAL ASSERTION: Business logic should work correctly
    assert 'FOODS' in result['seasonal_patterns']      # Should detect FOODS category
    assert 'HOUSEHOLD' in result['seasonal_patterns']  # Should detect HOUSEHOLD category  
    assert 'weekly_seasonality' in result['seasonal_patterns']['FOODS']  # Should find patterns


# NON-FUNCTIONAL TESTING EXAMPLES: Testing system qualities, not business logic

def test_large_dataset():
    """
    NON-FUNCTIONAL TEST: Performance - can our function handle large datasets?
    
    This isn't testing business logic (what the function does), but rather
    system quality (how well it performs with lots of data).
    """
    # Large dataset: 200 days, 100 product-store combinations  
    sales_data = pd.DataFrame({
        f'd_{i}': [np.random.randint(0, 100) for _ in range(100)]
        for i in range(1, 201)
    })

    calendar_data = pd.DataFrame({
        'd': [f'd_{i}' for i in range(1, 201)],
        'date': pd.date_range('2011-01-29', periods=200)
    })

    # Performance test: Should complete without timeout or memory errors
    result = analyze_time_structure(sales_data, calendar_data)
    assert result['total_days'] == 200  # Should handle large dataset correctly

def test_minimal_data():
    """
    NON-FUNCTIONAL TEST: Robustness - edge case with minimal valid input
    
    Tests system quality: "Does our function gracefully handle the smallest 
    possible valid input?"
    """
    # Minimal valid data: just 2 days
    sales_data = pd.DataFrame({
        'd_1': [10],
        'd_2': [12]
    })

    calendar_data = pd.DataFrame({
        'd': ['d_1', 'd_2'], 
        'date': pd.to_datetime(['2011-01-29', '2011-01-30'])
    })

    # Robustness test: Should work even with minimal data
    result = analyze_time_structure(sales_data, calendar_data)
    assert result['total_days'] == 2  # Should handle minimal case

def test_constant_series():
    """
    NON-FUNCTIONAL TEST: Edge case handling - what if data never changes?
    
    Tests robustness: "What happens with zero-variance time series?"
    """
    # Constant data: same value every day
    sales_data = pd.DataFrame({
        'd_' + str(i): [50, 50, 50]  # Never changes
        for i in range(1, 31)
    })

    calendar_data = pd.DataFrame({
        'd': [f'd_{i}' for i in range(1, 31)]
    })

    result = analyze_trend_components(sales_data, calendar_data)

    # Should handle constant series gracefully (slope ≈ 0)
    assert abs(result['linear_trend']['slope']) < 0.1
```

- [ ] **Step 2: Write Section 2.3 - Functional vs Non-Functional Testing**

File: `docs/testing/testing-guide-comprehensive.md` (append)

```markdown
### Section 2.3: Functional vs Non-Functional Testing

Our EDA tests cover two different types of concerns:

## Functional Testing: "Does it do the right thing?"

**Functional tests** check that business logic works correctly. They answer: "Given this input, does the function produce the correct business result?"

**Example from our seasonal pattern detection**:

```python
def test_seasonal_patterns_multiple_categories():
    # Test data with FOODS and HOUSEHOLD categories
    result = detect_seasonal_patterns(sales_data, calendar_data, hierarchy_level='category')
    
    # FUNCTIONAL ASSERTIONS: Business logic correctness
    assert 'FOODS' in result['seasonal_patterns']      # Should find FOODS
    assert 'HOUSEHOLD' in result['seasonal_patterns']  # Should find HOUSEHOLD  
```

**What this tests**: The business requirement that "our system should correctly identify seasonal patterns for different product categories in retail data."

## Non-Functional Testing: "Does it do it well?"

**Non-functional tests** check system qualities like performance, robustness, and scalability. They answer: "Does the function behave well under different conditions?"

**Examples from our edge case testing**:

### Performance Testing
```python  
def test_large_dataset():
    # 200 days × 100 product combinations = 20,000 data points
    sales_data = pd.DataFrame({...})  # Large dataset
    
    result = analyze_time_structure(sales_data, calendar_data)
    # Should complete without timing out or running out of memory
```

### Robustness Testing  
```python
def test_minimal_data():
    sales_data = pd.DataFrame({'d_1': [10], 'd_2': [12]})  # Only 2 days
    
    result = analyze_time_structure(sales_data, calendar_data) 
    # Should work even with minimal valid input
```

### Edge Case Handling
```python
def test_constant_series():
    # All sales values are identical (no variation)
    sales_data = pd.DataFrame({f'd_{i}': [50, 50, 50] for i in range(1, 31)})
    
    result = analyze_trend_components(sales_data, calendar_data)
    # Should handle zero-variance data gracefully
```

## Why Both Types Matter for EDA

**Functional Testing Prevents**: Wrong business insights
- Seasonal patterns detected for wrong categories
- Trend directions calculated incorrectly  
- Missing value analysis missing actual patterns

**Non-Functional Testing Prevents**: System failures in production  
- Memory crashes on large M5 dataset (30,490 rows)
- Infinite loops on edge case data
- Performance degradation over time

**In Our Project Context**:
- **Functional**: "Does seasonal detection correctly identify weekly patterns in FOODS category?"
- **Non-Functional**: "Can seasonal detection handle 3,049 M5 items without crashing?"

Both are essential for reliable demand forecasting at Walmart scale.

## Finding These Tests in Our Codebase

**Functional tests** are usually in the main test classes:
- `TestSeasonalPatternDetection` 
- `TestTrendAnalysis`
- `TestAutocorrelationAnalysis`

**Non-functional tests** are often in edge case classes:
- `TestTemporalAnalysisEdgeCases`
- Tests with names like `test_large_dataset`, `test_minimal_data`
```

- [ ] **Step 3: Commit Section 2.3**

```bash
git add docs/testing/
git commit -m "docs: add section 2.3 - functional vs non-functional testing

Distinguish business logic tests from system quality tests"
```

### Task 8: Part 3 - Testing Patterns Across Your Project

**Files:**
- Modify: `docs/testing/testing-guide-comprehensive.md`

- [ ] **Step 1: Write Section 3.1 - Pattern Recognition**

File: `docs/testing/testing-guide-comprehensive.md` (append)

```markdown
## Part 3: Testing Patterns Across Your Project

### Section 3.1: Pattern Recognition

Now that you understand the concepts, let's see how the same testing patterns appear consistently across our EDA test suite:

## Pattern 1: Class-Based Organization

**Every test file follows this structure**:

```python
# notebooks/eda/tests/test_temporal_analysis.py
class TestTimeStructureAnalysis:      # One class per function
class TestSeasonalPatternDetection:   # One class per function  
class TestTrendAnalysis:              # One class per function
class TestAutocorrelationAnalysis:    # One class per function
class TestTemporalAnalysisIntegration:  # Integration tests
class TestTemporalAnalysisEdgeCases:    # Edge cases

# notebooks/eda/tests/test_data_quality.py  
class TestMissingPatternAnalysis:     # One class per function
class TestMissingMechanismCharacterization:  # One class per function
class TestSalesOutlierDetection:      # One class per function
class TestDataQualityIntegration:     # Integration tests

# notebooks/eda/tests/test_statistical_analysis.py
class TestDistributionStats:          # One class per function
class TestVariationMetrics:           # One class per function  
class TestOutlierAnalysis:            # One class per function
```

**Why this pattern works**: Each class focuses on thoroughly testing one function with multiple scenarios.

## Pattern 2: Test Data Creation

**Every test creates small, controlled data**:

```python
# From test_temporal_analysis.py
sales_data = pd.DataFrame({
    'id': ['FOODS_1_001_CA_1_validation'],
    'd_1': [5], 'd_2': [3], 'd_3': [7]    # 3 days of data
})

# From test_data_quality.py  
sales_data = pd.DataFrame({
    'd_1': [10, np.nan, 15],              # Mix of valid and missing
    'd_2': [12, 14, np.nan], 
    'd_3': [np.nan, 16, 18]
})

# From test_statistical_analysis.py
test_data = np.array([1, 2, 3, 4, 5, 100])  # Normal values + outlier
```

**Pattern**: Small datasets (3-10 rows) that contain exactly the conditions we want to test.

## Pattern 3: Assert Patterns by Domain

**Data Quality Tests**: Check data structure and completeness
```python
# From test_data_quality.py
assert result['missing_percentage'] == 33.33
assert result['missing_mechanism'] == 'MCAR'
assert 'outlier_indices' in result
```

**Statistical Analysis Tests**: Check mathematical correctness
```python  
# From test_statistical_analysis.py
assert abs(result['mean'] - 21.0) < 0.01
assert result['distribution_type'] == 'right_skewed'
assert result['outlier_count'] == 1
```

**Temporal Analysis Tests**: Check time-related business logic
```python
# From test_temporal_analysis.py
assert result['frequency'] == 'Daily'
assert result['total_days'] == 3
assert 'seasonal_patterns' in result
```

## Pattern 4: Integration Test Structure

**Every module has one integration test class** that combines all its functions:

- `TestTemporalAnalysisIntegration` → Complete time series workflow
- `TestDataQualityIntegration` → Complete data validation workflow  
- `TestStatisticalAnalysisIntegration` → Complete statistical workflow

**Pattern**: Integration tests use larger datasets (50-100 rows) and run multiple functions in sequence.

## Pattern 5: Edge Case Coverage

**Every module tests the same edge cases**:

- **Minimal data**: Smallest valid input
- **Large data**: Performance stress test
- **Missing data**: How does it handle NaN values?
- **Constant data**: Zero variance scenarios
- **Invalid data**: What happens with bad inputs?

**Find these patterns yourself**:
```bash
# Search for edge case test names across all files
grep -r "test_.*minimal\|test_.*large\|test_.*constant" notebooks/eda/tests/
```

This consistency makes our test suite predictable and maintainable.
```

- [ ] **Step 2: Write Section 3.2 - Test Organization Strategies**

File: `docs/testing/testing-guide-comprehensive.md` (append)

```markdown
### Section 3.2: Test Organization Strategies

## Naming Conventions for Discoverability

Our project uses consistent naming that makes tests easy to find and understand:

**Test File Names**:
```
test_temporal_analysis.py      → Tests utils/temporal_analysis.py  
test_data_quality.py          → Tests utils/data_quality.py
test_statistical_analysis.py  → Tests utils/statistical_analysis.py
```

**Test Class Names**:
```
TestTimeStructureAnalysis     → Tests analyze_time_structure() function
TestSeasonalPatternDetection  → Tests detect_seasonal_patterns() function
TestTrendAnalysis            → Tests analyze_trend_components() function
```

**Test Method Names**:
```
test_analyze_time_structure_basic()           → Happy path scenario
test_time_structure_with_actual_m5_format()   → Realistic data scenario  
test_time_structure_with_missing_dates()      → Edge case scenario
```

**Pattern**: `test_[function_name]_[scenario_description]()`

## Setup and Teardown with pytest

**Our tests use pytest fixtures for common setup**:

```python
# From notebooks/eda/tests/conftest.py (if it exists)
@pytest.fixture
def sample_m5_sales_data():
    """Reusable M5-format test data."""
    return pd.DataFrame({
        'id': ['FOODS_1_001_CA_1_validation'],
        'item_id': ['FOODS_1_001'], 
        'cat_id': ['FOODS'],
        'd_1': [5], 'd_2': [3], 'd_3': [7]
    })

@pytest.fixture  
def sample_calendar_data():
    """Reusable calendar test data."""
    return pd.DataFrame({
        'd': ['d_1', 'd_2', 'd_3'],
        'date': pd.to_datetime(['2011-01-29', '2011-01-30', '2011-01-31'])
    })

# Usage in tests
def test_with_fixtures(sample_m5_sales_data, sample_calendar_data):
    result = analyze_time_structure(sample_m5_sales_data, sample_calendar_data)
    assert result['total_days'] == 3
```

**Benefits**:
- ✅ **DRY**: Don't repeat test data setup in every test
- ✅ **Consistency**: All tests use the same baseline data format  
- ✅ **Maintainability**: Change data format once, all tests update

## Directory Structure Strategy

**Our EDA testing follows a clear structure**:

```
notebooks/eda/tests/
├── __init__.py                    # Makes it a Python package
├── test_temporal_analysis.py      # Time series analysis tests
├── test_data_quality.py          # Missing values, outliers  
├── test_statistical_analysis.py  # Distribution, variation metrics
├── test_correlation_analysis.py  # Category patterns, correlations
├── test_leakage_validation.py    # Data leakage prevention
├── test_drift_analysis.py        # Distribution drift detection
├── test_segment_analysis.py      # Product segmentation
├── test_visualization.py         # Plot generation tests
├── test_integration.py           # End-to-end workflow tests
└── test_eda_pipeline_integration.py  # Full pipeline tests
```

**Organization principle**: One test file per utility module, plus integration test files.

**Why this works**:
- **Parallel development**: Different developers can work on different modules
- **Targeted testing**: Run tests for only the module you're changing
- **Clear ownership**: Each test file has one clear responsibility
```

- [ ] **Step 3: Write Section 3.3 - Running and Debugging Tests**

File: `docs/testing/testing-guide-comprehensive.md` (append)

```markdown
### Section 3.3: Running and Debugging Tests

## Command Line Execution with pytest

**Run all EDA tests**:
```bash
cd /Users/rahul.vansh/Documents/Personal/demand_forecast_intelligence
pytest notebooks/eda/tests/ -v
```

**Run tests for one module**:
```bash
# Only temporal analysis tests
pytest notebooks/eda/tests/test_temporal_analysis.py -v

# Only data quality tests  
pytest notebooks/eda/tests/test_data_quality.py -v
```

**Run one specific test class**:
```bash
# Only time structure analysis tests
pytest notebooks/eda/tests/test_temporal_analysis.py::TestTimeStructureAnalysis -v
```

**Run one specific test method**:
```bash
# Only the basic test
pytest notebooks/eda/tests/test_temporal_analysis.py::TestTimeStructureAnalysis::test_analyze_time_structure_basic -v
```

**Useful pytest flags**:
```bash
-v, --verbose     # Show individual test names and results
-s               # Show print statements from tests  
-x               # Stop on first failure
--tb=short       # Shorter error tracebacks
--tb=long        # Detailed error tracebacks
-k "pattern"     # Run tests matching name pattern
```

## Reading Test Output and Failure Messages

**Successful test run**:
```
notebooks/eda/tests/test_temporal_analysis.py::TestTimeStructureAnalysis::test_analyze_time_structure_basic PASSED [100%]
```

**Failed test with helpful error**:
```
FAILED notebooks/eda/tests/test_temporal_analysis.py::TestTimeStructureAnalysis::test_analyze_time_structure_basic

>       assert result['total_days'] == 3
E       AssertionError: assert 4 == 3
E        +  where 4 = {'total_days': 4, 'frequency': 'Daily', ...}['total_days']

notebooks/eda/tests/test_temporal_analysis.py:49: AssertionError
```

**How to read this**:
- **Location**: `test_temporal_analysis.py:49` → Check line 49
- **Problem**: Expected 3 days, got 4 days  
- **Root cause**: Either test data is wrong, or function logic is wrong

## Using Project's pyproject.toml Configuration

Our project configures pytest in [`pyproject.toml`](../../pyproject.toml):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]           # Also run main project tests
python_files = ["test_*.py"]    # Test files must start with "test_"
python_classes = ["Test*"]      # Test classes must start with "Test"  
python_functions = ["test_*"]   # Test functions must start with "test_"
addopts = "--strict-markers --disable-warnings"
```

**What this means**:
- ✅ **Auto-discovery**: pytest automatically finds our test files
- ✅ **Naming enforcement**: Tests must follow naming conventions
- ✅ **Clean output**: Warnings are disabled for cleaner test results

## Debugging Test Failures

**Step 1: Read the error message carefully**
- Which assertion failed?
- What was the expected vs actual value?
- Which line of code caused the failure?

**Step 2: Run just that one test**
```bash
pytest notebooks/eda/tests/test_temporal_analysis.py::TestTimeStructureAnalysis::test_analyze_time_structure_basic -v -s
```

**Step 3: Add debug print statements**
```python
def test_analyze_time_structure_basic():
    # ... setup code ...
    
    result = analyze_time_structure(sales_data, calendar_data)
    print(f"DEBUG: result = {result}")  # Add this line
    
    assert result['total_days'] == 3
```

**Step 4: Check if the function changed**  
Maybe someone modified `analyze_time_structure()` and the test needs updating.

**Step 5: Check test data**
Maybe the test data doesn't match what we think it contains.

**Common debugging workflow**:
1. **Isolate**: Run only the failing test
2. **Inspect**: Add print statements to see actual vs expected values
3. **Compare**: Check if function behavior or test expectations changed
4. **Fix**: Update either the function or the test (whichever is wrong)
```

- [ ] **Step 4: Commit Section 3**

```bash
git add docs/testing/testing-guide-comprehensive.md
git commit -m "docs: add part 3 - testing patterns across project

Show consistent patterns and practical usage across EDA test suite"
```

### Task 9: Create Quick Reference Guide

**Files:**
- Modify: `docs/testing/quick-reference.md`

- [ ] **Step 1: Create quick reference guide**

File: `docs/testing/quick-reference.md`

```markdown
# Testing Quick Reference

Quick lookup guide for testing concepts and commands in the Demand Forecast Intelligence project.

## Test Types Summary

| Type | Purpose | Example | File Location |
|------|---------|---------|---------------|
| **Unit** | Test one function in isolation | `test_analyze_time_structure_basic()` | `test_temporal_analysis.py` |
| **Integration** | Test multiple functions together | `test_full_temporal_analysis_pipeline()` | `test_temporal_analysis.py` |  
| **Functional** | Test business logic correctness | `test_seasonal_patterns_multiple_categories()` | `test_temporal_analysis.py` |
| **Non-Functional** | Test system qualities (performance, edge cases) | `test_large_dataset()` | `test_temporal_analysis.py` |

## Common Test Commands

```bash
# Run all EDA tests
pytest notebooks/eda/tests/ -v

# Run tests for one module  
pytest notebooks/eda/tests/test_temporal_analysis.py -v

# Run one test class
pytest notebooks/eda/tests/test_temporal_analysis.py::TestTimeStructureAnalysis -v

# Run one specific test
pytest notebooks/eda/tests/test_temporal_analysis.py::TestTimeStructureAnalysis::test_analyze_time_structure_basic -v

# Run with debug output
pytest notebooks/eda/tests/test_temporal_analysis.py -v -s

# Stop on first failure  
pytest notebooks/eda/tests/ -x

# Run tests matching pattern
pytest notebooks/eda/tests/ -k "time_structure" -v
```

## Test File Organization

```
notebooks/eda/tests/
├── test_temporal_analysis.py      # Time series: structure, seasonality, trends
├── test_data_quality.py          # Missing values, outliers, data validation
├── test_statistical_analysis.py  # Distributions, variations, mathematical functions  
├── test_correlation_analysis.py  # Category patterns, correlations
├── test_leakage_validation.py    # Prevent data leakage in ML pipeline
├── test_drift_analysis.py        # Distribution drift detection
├── test_segment_analysis.py      # Product behavioral segmentation
├── test_visualization.py         # Plot generation and validation
└── test_integration.py           # End-to-end EDA workflow tests
```

## Test Class Naming Pattern

```python
TestTimeStructureAnalysis     # Tests analyze_time_structure() function
TestSeasonalPatternDetection  # Tests detect_seasonal_patterns() function  
TestTrendAnalysis            # Tests analyze_trend_components() function
TestTemporalAnalysisIntegration  # Integration tests for temporal module
TestTemporalAnalysisEdgeCases    # Edge cases for temporal module
```

## Common Assertion Patterns

**Data Structure Assertions**:
```python
assert isinstance(result, dict)
assert 'expected_key' in result
assert len(result['data']) == expected_count
```

**Business Logic Assertions**:
```python
assert result['frequency'] == 'Daily'
assert result['total_days'] == 3  
assert 'FOODS' in result['seasonal_patterns']
```

**Mathematical Assertions (with tolerance)**:
```python
assert abs(result['mean'] - 21.0) < 0.01
assert result['correlation'] > 0.5
```

**Edge Case Assertions**:
```python
assert result is not None
assert not np.isnan(result['value'])
assert result['status'] == 'success'
```

## Debugging Failed Tests

1. **Read error message**: Which assertion failed? Expected vs actual?
2. **Run isolated**: `pytest path/to/test.py::test_name -v -s`  
3. **Add debug prints**: `print(f"DEBUG: result = {result}")`
4. **Check function**: Did the implementation change?
5. **Check test data**: Does test data match expectations?

## Test Data Patterns

**Minimal Valid Data** (unit tests):
```python
sales_data = pd.DataFrame({
    'd_1': [5], 'd_2': [3], 'd_3': [7]  # 3 days, 1 product
})
```

**Realistic M5 Format** (integration tests):
```python
sales_data = pd.DataFrame({
    'id': ['FOODS_1_001_CA_1_validation'],
    'item_id': ['FOODS_1_001'],
    'cat_id': ['FOODS'],
    'd_1': [5], 'd_2': [3], 'd_3': [7]
})
```

**Edge Case Data**:
```python
# Missing values
sales_data = pd.DataFrame({'d_1': [10, np.nan, 15]})

# Constant values  
sales_data = pd.DataFrame({'d_1': [50, 50, 50]})

# Large dataset
sales_data = pd.DataFrame({f'd_{i}': [random_value] for i in range(1, 201)})
```

## When to Write Each Test Type

- **Unit Tests**: Always write first, test each function individually
- **Integration Tests**: After unit tests pass, test functions working together  
- **Functional Tests**: When testing business requirements (seasonality, trends)
- **Non-Functional Tests**: For performance, edge cases, error handling

## Project-Specific Context

**M5 Dataset**: 1,913 days × 30,490 products from Walmart
**EDA Goal**: Analyze temporal patterns for demand forecasting
**Business Impact**: Test failures = wrong predictions = inventory problems
**Scale**: Tests must work on both sample data (fast) and full M5 (realistic)
```

- [ ] **Step 2: Commit quick reference**

```bash
git add docs/testing/quick-reference.md
git commit -m "docs: add testing quick reference guide

Condensed lookup guide for test commands and patterns"
```

### Task 10: Final Integration and Cross-References

**Files:**
- Modify: `docs/testing/testing-guide-comprehensive.md`

- [ ] **Step 1: Add conclusion section with cross-references**

File: `docs/testing/testing-guide-comprehensive.md` (append)

```markdown
---

## Conclusion: Next Steps

Congratulations! You now understand the four main types of testing in our EDA project and can read, run, and debug our test suite.

## What You've Learned

✅ **Unit Testing**: Test individual functions like `analyze_time_structure()` in isolation  
✅ **Integration Testing**: Test multiple functions working together in workflows  
✅ **Functional Testing**: Test business logic correctness (seasonal patterns, trends)  
✅ **Non-Functional Testing**: Test system qualities (performance, edge cases)

## Practical Next Steps

**1. Explore the Test Files**
Start reading these files to see the patterns in action:
- [`notebooks/eda/tests/test_temporal_analysis.py`](../../notebooks/eda/tests/test_temporal_analysis.py) - Main teaching example
- [`notebooks/eda/tests/test_data_quality.py`](../../notebooks/eda/tests/test_data_quality.py) - Data validation patterns  
- [`notebooks/eda/tests/test_statistical_analysis.py`](../../notebooks/eda/tests/test_statistical_analysis.py) - Mathematical testing

**2. Run Tests Yourself**
```bash
cd /Users/rahul.vansh/Documents/Personal/demand_forecast_intelligence

# Start with one test to see it work
pytest notebooks/eda/tests/test_temporal_analysis.py::TestTimeStructureAnalysis::test_analyze_time_structure_basic -v

# Then run a whole module
pytest notebooks/eda/tests/test_temporal_analysis.py -v

# Finally run all EDA tests  
pytest notebooks/eda/tests/ -v
```

**3. Write Your First Test**
When you add a new EDA function, follow this pattern:
```python
class TestYourNewFunction:
    def test_your_function_basic(self):
        # Setup: Create small test data
        # Execute: Call your function  
        # Assert: Check the results
```

**4. Use the Quick Reference**
Keep [`docs/testing/quick-reference.md`](quick-reference.md) handy for commands and patterns.

## Additional Resources

**Project Documentation**:
- [Main Project README](../../README.md) - Project overview and setup
- [CLAUDE.md](../../CLAUDE.md) - Development guidelines and commands

**EDA Implementation**:
- [`notebooks/eda/eda_analysis.py`](../../notebooks/eda/eda_analysis.py) - Main EDA functions being tested
- [`notebooks/eda/utils/`](../../notebooks/eda/utils/) - Utility modules with test coverage

**pytest Documentation**:
- [pytest official docs](https://docs.pytest.org/) - Complete testing framework guide
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html) - Advanced test setup patterns

## Remember: Tests Are Documentation

Good tests serve as **executable documentation** that shows:
- How functions should be called
- What inputs they expect  
- What outputs they produce
- What edge cases they handle

When you read our test files, you're reading the most up-to-date documentation of how our EDA system actually works.

Happy testing! 🧪
```

- [ ] **Step 2: Add table of contents links**

File: `docs/testing/testing-guide-comprehensive.md` (modify table of contents section)

```markdown
## Table of Contents

1. [Foundation with Real Code](#part-1-foundation-with-real-code)
   - 1.1 [Let's Look at a Real Test](#section-11-lets-look-at-a-real-test)
   - 1.2 [What Type of Test Was That?](#section-12-what-type-of-test-was-that)
   - 1.3 [Why This Matters for EDA](#section-13-why-this-matters-for-eda)

2. [Testing Types in Action](#part-2-testing-types-in-action)  
   - 2.1 [Unit Testing Deep Dive](#section-21-unit-testing-deep-dive)
   - 2.2 [Integration Testing Walkthrough](#section-22-integration-testing-walkthrough)
   - 2.3 [Functional vs Non-Functional Testing](#section-23-functional-vs-non-functional-testing)

3. [Testing Patterns Across Your Project](#part-3-testing-patterns-across-your-project)
   - 3.1 [Pattern Recognition](#section-31-pattern-recognition)  
   - 3.2 [Test Organization Strategies](#section-32-test-organization-strategies)
   - 3.3 [Running and Debugging Tests](#section-33-running-and-debugging-tests)

4. [Quick Reference](quick-reference.md) - Commands and patterns lookup
5. [Code Examples](examples/) - Extracted examples from real test files

---
```

- [ ] **Step 3: Final commit of complete guide**

```bash
git add docs/testing/testing-guide-comprehensive.md
git commit -m "docs: complete testing guide with cross-references and next steps

Finalize comprehensive tutorial-style testing documentation"
```

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-23-testing-guide-implementation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Explore project context - check EDA test files and structure", "status": "completed", "activeForm": "Exploring project context and test file structure"}, {"content": "Ask clarifying questions about documentation scope and audience", "status": "completed", "activeForm": "Asking clarifying questions about documentation requirements"}, {"content": "Propose 2-3 approaches for documentation structure", "status": "completed", "activeForm": "Proposing documentation approaches"}, {"content": "Present design for testing guide", "status": "completed", "activeForm": "Presenting testing documentation design"}, {"content": "Write design doc", "status": "completed", "activeForm": "Writing design specification"}, {"content": "Spec self-review", "status": "completed", "activeForm": "Reviewing spec for consistency"}, {"content": "User reviews written spec", "status": "completed", "activeForm": "User reviewing written specification"}, {"content": "Transition to implementation", "status": "completed", "activeForm": "Transitioning to implementation"}]