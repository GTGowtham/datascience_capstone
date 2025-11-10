import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

script_dir=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.dirname(script_dir)
data_path=os.path.join(project_root,"Adult Dataset","output.csv")
output=os.path.join(project_root,"output")

df=pd.read_csv(data_path)


summary_file = os.path.join(output, "01_exploration_summary.txt")

with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("ADULT DATASET - EXPLORATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("DATASET OVERVIEW\n")
    f.write("-"*80 + "\n")
    f.write(f"Total Records: {len(df):,}\n")
    f.write(f"Total Features: {len(df.columns)}\n")
    f.write(f"Dataset Shape: {df.shape}\n\n")
    
    f.write("COLUMN NAMES\n")
    f.write("-"*80 + "\n")
    for i, col in enumerate(df.columns, 1):
        f.write(f"{i:2d}. {col}\n")
    
    f.write("\n\nFIRST 10 ROWS\n")
    f.write("-"*80 + "\n")
    f.write(df.head(10).to_string())
    
    f.write("\n\n\nDATA TYPES\n")
    f.write("-"*80 + "\n")
    f.write(df.dtypes.to_string())

print(f"Summary saved to: {os.path.basename(summary_file)}\n")

# ==================== STATISTICAL SUMMARY ====================
print("[3/6] Generating statistical summary...")
stats_file = os.path.join(output, "statistical_summary.csv")

# Numerical statistics
numerical_stats = df.describe().T
numerical_stats.to_csv(stats_file)

print(f"Statistics saved to: {os.path.basename(stats_file)}\n")