# Testing Quick Reference Guide

A concise overview of testing types for the Demand Forecast Intelligence EDA module.

## 1. Unit Tests

**Purpose**: Test individual functions or components in isolation  
**Scope**: Single function, method, or class  
**Speed**: Very fast (milliseconds)  
**Dependencies**: None (mocked if needed)

**Example**: Testing a data validation function with different input types
```python
def test_validate_sales_data():
    assert validate_sales_data([1, 2, 3]) == True
    assert validate_sales_data([-1, 2, 3]) == False
```

---

## 2. Integration Tests

**Purpose**: Test how multiple components work together  
**Scope**: Multiple modules, classes, or external systems  
**Speed**: Moderate (seconds)  
**Dependencies**: Real components, may include databases or APIs

**Example**: Testing data loading and preprocessing pipeline
```python
def test_data_pipeline_integration():
    raw_data = load_sales_data()
    processed_data = preprocess_sales_data(raw_data)
    assert processed_data.shape[0] > 0
    assert processed_data['sales'].dtype == 'float64'
```

---

## 3. Edge Case Tests

**Purpose**: Test boundary conditions and unusual inputs  
**Scope**: Any level (unit, integration, system)  
**Speed**: Varies  
**Dependencies**: Depends on test type

**Example**: Testing with extreme or unusual data conditions
```python
def test_edge_cases():
    # Empty dataset
    assert handle_empty_data([]) == "No data available"
    
    # Single data point
    assert calculate_trend([100]) == "Insufficient data"
    
    # All zeros
    assert calculate_variance([0, 0, 0]) == 0
```

---

## 4. Property-Based Tests

**Purpose**: Test properties that should hold for all valid inputs  
**Scope**: Function behavior across input space  
**Speed**: Slower (generates many test cases)  
**Dependencies**: Property testing library (like Hypothesis)

**Example**: Testing mathematical properties of forecasting functions
```python
@given(sales_data=lists(floats(min_value=0, max_value=1000)))
def test_forecast_properties(sales_data):
    if len(sales_data) >= 7:  # Minimum data requirement
        forecast = generate_forecast(sales_data)
        # Property: forecast should be non-negative
        assert all(f >= 0 for f in forecast)
        # Property: forecast length should match horizon
        assert len(forecast) == FORECAST_HORIZON
```

---

## When to Use Each Type

| Test Type | Use When | EDA Context Example |
|-----------|----------|-------------------|
| **Unit** | Testing pure functions, calculations | Data validation, statistical functions |
| **Integration** | Testing component interactions | Data loading + preprocessing pipelines |
| **Edge Case** | Testing boundary conditions | Empty datasets, extreme values, missing data |
| **Property-Based** | Testing mathematical properties | Forecast constraints, data transformation invariants |

## Test Organization

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component interaction tests  
├── edge_cases/     # Boundary condition tests
└── properties/     # Property-based tests
```