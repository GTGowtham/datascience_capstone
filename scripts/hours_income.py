import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

script_dir=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.dirname(script_dir)
data_path=os.path.join(project_root,"Adult Dataset","output.csv")
output=os.path.join(project_root,"output")

df=pd.read_csv(data_path)
income_hours_file=os.path.join(output,"hours_income.png")

plt.figure(figsize=(10,8))
sns.boxplot(x='income',y='hours-per-week',hue='income',data=df,palette='Set2',legend=False)
plt.xlabel("income category",fontsize=12)
plt.ylabel("hours per week",fontsize=12)
plt.title("working hours by income level",fontsize=14,fontweight='bold')
plt.grid(True,alpha=0.3,axis='y')
plt.tight_layout()
plt.savefig(income_hours_file,dpi=300,bbox_inches='tight')
plt.close()


