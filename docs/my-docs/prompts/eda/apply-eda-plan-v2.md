Use brainstroming skill

Read and understand eda framework @docs/project-info/claude/eda_framework_v2.md which is guide of how to perform eda.

Add Python code in @notebooks/EDA to generate the required plots for analysis and perform and print the necessary mathematical scoring, statistical calculations, and statistical tests. 


C. Hybrid approach - Core utilities + both notebook interfaces for exploration AND script pipeline for automated execution. Modularize the code, into separate utility files under @notebooks/eda/utils.

The goal is to help me understand the data in both ways: through visualizations and numeric/statistical results. 

Keep test cases for eda very minimal.

After you are done with implementation, document implementation @docs/implementation-info

Notes:

- Store all plots in @notebooks/eda/plots
- Dataset schema information available @docs/project-info/schema-info.md
- Perform analysis dataset stored at @data/raw, and 
- Add comments understand code and correlate with given eda framework