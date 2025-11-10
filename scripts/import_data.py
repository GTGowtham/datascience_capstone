import pandas as pd
import numpy as np
import os

script_dir=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.dirname(script_dir)
data_path=os.path.join(project_root,"Adult Dataset","adult.data")
output=os.path.join(project_root,"output")

columns=[ 'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

df = pd.read_csv(data_path, names=columns, skipinitialspace=True)
print(f"✓ Loaded {len(df):,} records with {len(df.columns)} features\n")


data=os.path.join(project_root,"Adult Dataset","output.csv")

df.to_csv(data, index=False)
