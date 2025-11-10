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
education_distribution_file=os.path.join(output,"education_distribution.png")

plt.figure(figsize=(10,8))
education_counts=df['education'].value_counts()
plt.bar(range(len(education_counts)),education_counts.values,color='steelblue')
plt.xlabel('Education level',fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.title('Education Distribution',fontsize=14,fontweight='bold')
plt.xticks(range(len(education_counts)),education_counts.index,rotation=45,ha='right')
plt.grid(True,alpha=0.3,axis='y')
plt.tight_layout()
plt.savefig(education_distribution_file,dpi=300,bbox_inches='tight')
plt.close()


