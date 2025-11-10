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
corelation_file=os.path.join(output,"corelation_heatmap.png")
numerical_features=['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']

correlation_matrix=df[numerical_features].corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,fmt='.2f',cmap='coolwarm',center=0,square=True
      ,linewidths=1,cbar_kws={"shrink":0.8})
plt.title('Feature_corelation_map',fontsize=14,fontweight='bold')
plt.savefig(corelation_file,dpi=300,bbox_inches='tight')
plt.close()
