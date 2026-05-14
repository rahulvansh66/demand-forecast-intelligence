# Walmart M5 Dataset Schema Documentation Design

**Date:** 2026-05-14  
**Purpose:** Design specification for creating comprehensive schema documentation of the Walmart M5 dataset  
**Output:** `my-docs/project-info/schema-info.md`

## Overview

This design specifies how to create structured schema documentation for the Walmart M5 dataset used in the retail demand forecasting copilot. The documentation will serve both ML engineers implementing forecasting models and business stakeholders understanding the data pipeline.

## Design Principles

1. **Business Process Flow Organization** - Structure tables by their role in the demand forecasting pipeline
2. **Balanced Technical/Business Context** - Include both data types and business interpretation
3. **Inline Relationship References** - Show table connections within field descriptions
4. **Point/Subpoint Format** - Use bullet-style formatting for enhanced readability

## Document Structure

### 1. Introduction Section
- Brief overview of Walmart M5 dataset source (Kaggle competition)
- Purpose in demand forecasting context
- Dataset scope and time periods covered
- How to interpret the "database table" concept for CSV files

### 2. Core Sales Data Tables
Tables containing the primary transactional sales information:

**2.1 sales_train_validation.csv**
- Training data with daily sales by item-store combination
- Covers d_1 to d_1913 (Jan 2011 - May 2016)
- 30,490 rows representing unique item-store pairs

**2.2 sales_train_evaluation.csv** 
- Extended dataset including evaluation period
- Covers d_1 to d_1941 (extends 28 days beyond validation)
- Same structure as validation file but longer time series

### 3. Supporting Dimension Tables
Tables providing contextual data to enrich sales analysis:

**3.1 calendar.csv**
- Date dimension with events, holidays, SNAP benefits
- Maps day identifiers (d_1, d_2, etc.) to actual calendar dates
- 1,969 rows covering full dataset timespan

**3.2 sell_prices.csv**
- Weekly pricing data by store-item combination
- Enables price-demand relationship analysis
- 6.8M+ rows with weekly granularity

### 4. Model Output Tables
Competition submission format for forecast outputs:

**4.1 sample_submission.csv**
- Template for 28-day forecast submissions
- Shows expected output format for ML models
- Covers both validation and evaluation forecasts

### 5. Database Relationships Summary
High-level overview of how tables connect and data flows through the forecasting pipeline.

## Field Documentation Format

Each table will use this standardized format:

```markdown
### [filename.csv] - [Business Purpose]
[1-2 sentence description of table's role in forecasting]

**Technical Details:** [Row count, time period, update frequency]

**Fields:**

• **field_name** (DataType)
  - [Primary description of what this field represents]
  - Links to: [Related tables and how they connect]
  - [Business context or constraints]
  - [Special handling notes if applicable]
```

## Content Guidelines

### Technical Specifications
- Include accurate data types (String, Integer, Float, Date)
- Provide row counts and dataset dimensions
- Note primary/foreign key relationships
- Mention data ranges and constraints

### Business Context
- Explain field meaning in retail/forecasting terms
- Connect to demand forecasting use cases
- Note relevant business rules or logic
- Highlight fields critical for ML pipeline

### Relationship Documentation
- Use "Links to:" format for table connections
- Explain join conditions and cardinality
- Reference specific fields in related tables
- Show data flow direction where relevant

## Implementation Approach

1. **Analyze Dataset Structure** - Examine CSV headers, sample data, and relationships
2. **Document Core Tables First** - Start with sales data as primary entities
3. **Add Supporting Tables** - Layer in calendar and pricing dimensions
4. **Cross-Reference Validation** - Ensure all relationships are documented consistently
5. **Business Context Review** - Verify descriptions serve both technical and business audiences

## Success Criteria

- Complete field documentation for all 5 CSV files
- Clear explanation of table relationships and data flow
- Balanced technical accuracy with business interpretation
- Consistent formatting using point/subpoint structure
- Actionable information for ML pipeline development

## Deliverable

Single markdown file: `my-docs/project-info/schema-info.md` containing comprehensive schema documentation following this design specification.