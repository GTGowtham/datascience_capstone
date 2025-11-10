import pandas as pd
import numpy as np
import os 

script_dir=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.dirname(script_dir)
data_path=os.path.join(project_root,"Adult Dataset","output.csv")
output=os.path.join(project_root,"output")

df=pd.read_csv(data_path)
missing_file = os.path.join(output, "missing_values_report.csv")

missing_counts={}

for col in df.columns:
    if df[col].dtype=="object":
        missing_count=(df[col]=='?').sum()
    else:
        missing_count=df[col].isnull().sum()
    missing_counts[col]=missing_count
print(missing_counts)

missing_df = pd.DataFrame({
'Column': missing_counts.keys(),
'Missing_Count': missing_counts.values(),
'Percentage': [(v/len(df)*100) for v in missing_counts.values()]
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
missing_df.to_csv(missing_file, index=False)

total_missing = sum(missing_counts.values())
rows_with_missing = df[df.isin(['?']).any(axis=1)].shape[0]

print(f"      ✓ Found {total_missing:,} missing values in {len(missing_df)} columns")
print(f"      ✓ {rows_with_missing:,} rows contain at least one missing value")
print(f"      ✓ Report saved to: {os.path.basename(missing_file)}\n")


