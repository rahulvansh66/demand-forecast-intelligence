Create a brand new skill 

What should this skill enable Claude to do?
- eda on original data, more about it mentioned here @docs/my-docs/project-info/detailed_eda_checklist.md . to konw about schema, Refer dataset schema `docs/project-info/schema-info.md` to understand dataset.
When should it trigger? (what kinds of user requests or contexts) 
- manually when i tag in prompt, not auto trigger or auto lookup for this skill
Do you have a specific workflow or set of steps in mind?
- i mentioned here @docs/my-docs/project-info/detailed_eda_checklist.md

----Futher improvement
use skill @docs/outcomes-info/EDA/demand-forecast-eda-skill.md

task1:
in analysis, @notebooks/EDA apply concept like following in time series analysis, 
- Dickey-Fuller Test
- ACF & PACF plots and scores

task2:
create detailed report of all analysis performed, you cant descript analysis where plots used but you can utilise either data used for plot or equivalnet statistic test score 

after performing both task update skill @docs/outcomes-info/EDA/demand-forecast-eda-skill.md

<!-- Create a jupyter notebook in @notebooks/EDA to perform eda on original dataset, create required python files in @notebooks/EDA/utils and use it in notebook to avoid long codes and able to give more focus on showing charts and results in the jupyter notebook.
 


EDA should include following (at least but not limited to this):
 
Basics data scanity 
- load data
- Initial data inspection
- Basic Data quality checks: Missing values, Invalid values, Unique values, Cardinality, Range issues, data types checking, Duplicate IDs, Target distribution
- Light cleaning : Rename messy columns, Strip spaces, Fix data types, Convert dates. Remove exact duplicates, Handle obvious invalid records

Perform EDA

- Descriptive statistics
- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- EDA related to time series 

Note: 
- For analysis feel free to use both plots and statistics score/test wherever it make sense. 
- EDA should include Detect outliers and anomalies detection,  -->

Report generation 

Store report in md at @docs/outcomes-info/EDA and @notebooks/EDA

Generate report to add all Findings from Basics data scanity and eda


