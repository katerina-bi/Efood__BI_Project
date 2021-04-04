## efoodBiProject
This repository includes all the necessary files and data required in order to replicate the efood assignment.
## Folders						
- Input
Input is in the path Input_Files/PY_analysis (.zip format which needs to be unzipped using the appropriate program). There are two kinds of inputs:
1. It includes the data I received from the BigQuery. These data were extracted in json format and are used as an input for the analysis-generated code. 
2. Data used as an input for the business analytics service by Microsoft, PowerBI. They are the output of the afformentioned analysis (input 1).
- Code 
The code resigns in the folder named *Code*. In the process of generating the appropriate code, two kinds of methodologies were applied. For both methodlogies RMF KPIs were created.
1. The first methodology is based on segment-creating rules.
2. The second methodology is a K-means statistical analysis in order to properly identify the number of segments. A scoring was applied to the segmented customer groups based on the frequency, recency, and average basket amount.
- Dashboard and Presentation
The presentation and the dashboard are located in the *Visualization* folder (in .pptx and .pbix formats respectively. The dashboard file includes three tabs.
1. The first tab presents the first methodology that was used in the analysis.
2. The second tab presents the K-means analysis that was used in the second methodology. 
3. The third tab presents the filtered insights per cuisine in order to answer marketing questions involving certain cuisines (i.e. breakfast.\
- Updated Version
The folder *Part I* contains the updated version of the code from the first assignment.
## Instructions
In order to replicate this process an analyst must do the following: 
1. Download and unzip *bq_results3* from Efood__BI_Project/Input_Files/Py_analysis/. 
2. Download the Python code from Efood__BI_Project/Code/ and change the path that reads the input at code line 23 to your respective path. Make sure that all the libraries mentioned from line 3 to line 20 are downloaded and installed using the import command. Change the output that the output is saved in line 74 and then run the program. 
## Tools
- Stack Overflow for help in code implementation.
- Microsoft's PowerBI.
- Microsoft's Excel, Powerpoint.
- Microsoft's Visual Studio Code as an editor.
- Code was written in Python language (version 3.9).
- The Python packages used were: json, pandas, numpy, matplotlib.pyplot, seaborn, datetime, scipy, sklearn.
