import pandas as pd
import numpy as np
import os

script_dir=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.dirname(script_dir)
data_path=os.path.join(project_root,"Adult Dataset","output.csv")
output=os.path.join(project_root,"output")

df=pd.read_csv(data_path)
outlier_file=os.path.join(output, "outliers_report.csv")

numerical_columns=df.select_dtypes(include=['int64','float64']).columns
outliers_results=[]

for col in numerical_columns:
      q1=df[col].quantile(0.25)
      q2=df[col].quantile(0.75)
      iqr=q2-q1
      lower_bound=q1-1.5*iqr
      upper_bound=q2+1.5*iqr
      outliers=df[(df[col]<lower_bound) | (df[col]>upper_bound)]

      outliers_results.append({
            'Column': col,
            'Q1:':q1,
            'Q2':q2,
            'IQR':iqr,
            'Lower_Bound':lower_bound,
            'Upper_Bound':upper_bound,
            'Outlier_count':len(outliers),
            'Outlier_percentage':(len(outliers)/len(df)*100)
      })
outlier_df=pd.DataFrame(outliers_results)
outlier_df.to_csv(outlier_file,index=False)

total_outliers=outlier_df['Outlier_count'].sum()
print(f"      ✓ Found {total_outliers:,} total outliers across {len(numerical_columns)} numerical features")
print(f"      ✓ Report saved to: {os.path.basename(outlier_file)}\n")