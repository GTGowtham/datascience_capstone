import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt

script_dir=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.dirname(script_dir)
data_path=os.path.join(project_root,"Adult Dataset","output.csv")
output=os.path.join(project_root,"output")

df=pd.read_csv(data_path)
income_distribution_file=os.path.join(output, "income_distribution.png")

plt.figure(figsize=(10,8))
income_counts=df['income'].value_counts()
plt.bar(income_counts.index, income_counts.values, color=['coral', 'lightgreen'])
plt.xlabel("income category", fontsize=12)
plt.ylabel("count", fontsize=12)
plt.title("income distribution", fontsize=14, fontweight='bold')
for i,v in enumerate(income_counts.values):
      plt.text(i,v+1000,f'{v:,}', ha='center',fontsize=12)
plt.grid(True,alpha=0.3,axis='y')
plt.tight_layout()
income_plot=os.path.join(output,"income_distribution.png")
plt.savefig(income_plot, dpi=300, bbox_inches='tight')
plt.close()