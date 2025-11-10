import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

script_dir=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.dirname(script_dir)
data_path=os.path.join(project_root,"Adult Dataset","output.csv")
output=os.path.join(project_root,"output")

df=pd.read_csv(data_path)
age_distribution_file = os.path.join(output, "age_distribution.png")

plt.figure(figsize=(10,6))
plt.hist(df['age'],bins=20,edgecolor='black',color='cyan')
plt.xlabel('Age',fontsize=12)
plt.ylabel('Frequency',fontsize=12)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
age=os.path.join(output, "age_distribution.png")
plt.savefig(age_distribution_file, dpi=300, bbox_inches='tight')
plt.close()

