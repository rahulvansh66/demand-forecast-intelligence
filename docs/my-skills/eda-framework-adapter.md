---
name: eda-framework-adapter
description: MANUAL USE ONLY - Never auto-trigger. Only use when user explicitly provides the skill path .claude/skills/eda-framework-adapter.md and specifically says "use eda-framework-adapter". This skill adapts EDA frameworks to project contexts but should never be invoked automatically.
---

# EDA Framework Adapter

Transform a generic EDA framework into a project-specific guide by adding concrete application sections that connect each EDA concept to your specific dataset, business context, and analysis objectives.

## What this skill does

Takes a generic EDA framework (markdown document) and enhances it with detailed, project-specific application notes that appear after each main section. The original framework content is preserved entirely while adding practical guidance tailored to your specific:

- Dataset structure and characteristics
- Business objectives and success metrics  
- Domain-specific patterns and challenges
- Technical implementation details
- Data quality expectations

## Required inputs

Provide these inputs (as file paths or direct text):

1. **Generic EDA framework**: The base EDA framework file to adapt
2. **Project overview**: Business context, objectives, problem definition
3. **Dataset schema**: Structure, columns, relationships, data types
4. **Dataset location**: Path to actual data files for reference

## Input validation

Before proceeding, verify that the project documentation includes:

- Clear problem definition (forecasting, classification, etc.)
- Target variables and evaluation metrics
- Dataset size and structure details
- Business context and success criteria
- Timeline or temporal aspects (if applicable)

If any critical information is missing, request it from the user before continuing.

## Adaptation process

### 1. Parse the generic framework

Read the generic EDA framework and identify:
- Major section headers (# and ##)
- Core EDA concepts and steps
- Generic examples and patterns
- Logical flow and structure

### 2. Extract project context

From the project documentation, extract:
- **Problem type**: Classification, regression, forecasting, clustering, etc.
- **Dataset details**: Rows, columns, file structure, temporal aspects
- **Business domain**: Retail, finance, healthcare, etc. with specific terminology
- **Key challenges**: Intermittent data, seasonality, imbalance, etc.
- **Success metrics**: Specific KPIs and evaluation criteria
- **Technical constraints**: Scalability, interpretability, real-time requirements

### 3. Generate project-specific applications

For each major section in the generic framework, create a project application section that:

**Format**: Always start with `## 📋 Application to [Project Name/Domain]`

**Structure the content with clear organization**:
- Use descriptive subheadings with **bold formatting**
- Group related concepts together logically
- Include specific examples with concrete values
- Reference actual dataset files, columns, and business metrics
- Explain the "why" behind project-specific considerations

**Content guidelines**:
- **Be concrete**: Replace generic examples with project-specific ones
- **Use actual values**: Mention real column names, data ranges, business KPIs
- **Connect to business**: Explain how each EDA step relates to business decisions
- **Address domain challenges**: Highlight specific issues relevant to the domain
- **Include implementation details**: Specific techniques, thresholds, validation approaches

### 4. Integration pattern

Insert project applications immediately after each main framework section:
```markdown
# [Original Framework Section]
[Original generic content preserved exactly]

## 📋 Application to [Project Context]
[Project-specific guidance here]

---

# [Next Original Framework Section]
```

## Example section structure

```markdown
## 📋 Application to Retail Demand Forecasting Project

**Dataset-Specific Considerations:**
- **Sales data**: 30,490 item-store combinations × 1,969 daily observations
- **Zero-inflation**: Expect high proportion of zero sales days (intermittent demand)
- **Hierarchical structure**: Categories → departments → items → stores

**Business Context Integration:**
- **Forecasting horizon**: 28-day inventory planning cycles
- **Evaluation metrics**: WRMSSE (Walmart's weighted metric), RMSE, MAE
- **Seasonality critical**: Weekly patterns, holiday effects, promotional cycles

**Domain-Specific Quality Checks:**
- **Sales range validation**: Daily units should be 0-10,000 (flag outliers)
- **Price consistency**: Week-over-week changes >200% indicate data errors
- **Calendar alignment**: Ensure pricing weeks align with sales days

**Key Implementation Priorities:**
- Time-based validation (never random splits for time series)
- Lag feature engineering (1, 7, 14, 28-day lags)
- Price elasticity analysis integration
- Intermittent demand handling (zero-inflated models)
```

## Output requirements

Create a modified markdown file that:

1. **Preserves all original content** exactly as written
2. **Adds project sections** after each main EDA topic
3. **Maintains proper markdown formatting** and structure
4. **Uses consistent section markers** (📋 emoji + "Application to [Project]")
5. **Includes concrete examples** with real project values
6. **References specific files/paths** from the project context

## Validation checklist

Before completing, verify the adapted framework:

- [ ] All original framework content is preserved
- [ ] Project applications are added to every major section
- [ ] Examples use concrete project details (not generic placeholders)
- [ ] Business context is clearly connected to technical concepts  
- [ ] Domain-specific challenges are addressed
- [ ] Formatting is consistent and professional
- [ ] File references and paths are accurate

## Common adaptation patterns

**For time-series projects**: Emphasize temporal validation, seasonality detection, lag features, and leakage prevention

**For classification projects**: Focus on class imbalance, feature importance, cross-validation strategies

**For high-cardinality data**: Address overfitting risks, embedding strategies, hierarchical modeling

**For business analytics**: Connect each EDA step to specific business decisions and KPIs

**For regulated domains**: Highlight compliance requirements, interpretability needs, bias detection

## Success criteria

The adapted framework should enable a data scientist to:

1. Understand exactly how each EDA concept applies to this specific project
2. Know what to look for in this particular dataset
3. Recognize domain-specific patterns and issues
4. Make informed modeling decisions based on project constraints
5. Connect technical findings to business value

The result transforms a generic guide into a practical, actionable roadmap tailored to the specific project context.