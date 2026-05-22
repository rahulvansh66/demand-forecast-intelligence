Use brainstroming skill and plan

Read and understand eda framework @docs/project-info/claude/eda_framework.md which is guide of how to perform eda and then plan the eda workflow for from step 1(Understand the problem first) to 5(Analyze individual features). 

Add Python code to eda_analysis.py in @notebooks/EDA to generate the required plots for analysis and perform the necessary mathematical scoring, statistical calculations, and statistical tests.

The goal is to help me understand the data in both ways: through visualizations and numeric/statistical results

Modularize the supporting code into separate utility files under @notebooks/eda/utils.

Log all findings, including calculations, statistics, scores, and statistical test results, in eda_report.md in a clear and readable format. The goal is for eda_report.md to provide all mathematical facts about the data so I can review them later before planning preprocessing or model building.

Notes:

- Store all plots in @notebooks/eda/plots
- Store the report at @notebooks/eda/eda_report.md
- Schema information available @docs/project-info/schema-info.md
- Perform analysis dataset stored at @data/raw, and 
- Add comments understand code and correlate with given eda framework


====

What's your primary goal for this EDA implementation is, as data scientist, I want to understand data, generate plots and well docuemented eda_report.md to Log all stats findings, including calculations, statistics, scores, and statistical test results which further helpful for data pre processing and model development