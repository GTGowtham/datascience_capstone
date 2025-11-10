# Data Exploration Project - Skills and Process Guide

## 📚 Table of Contents
1. [Required Skills Overview](#required-skills-overview)
2. [Technical Skills Breakdown](#technical-skills-breakdown)
3. [Step-by-Step Process Guide](#step-by-step-process-guide)
4. [Understanding Output Files](#understanding-output-files)
5. [Practical Examples](#practical-examples)
6. [Learning Path](#learning-path)

---

## 🎯 Required Skills Overview

To successfully complete this data exploration and analysis project, you need a combination of technical and analytical skills:

### Essential Skills (Must Have)
- ✅ **Python Programming** - Basic to Intermediate
- ✅ **Data Manipulation** - Using pandas library
- ✅ **Statistical Concepts** - Mean, median, outliers, correlation
- ✅ **File Operations** - Reading CSV files, working with directories
- ✅ **Command Line** - Basic terminal/PowerShell commands

### Recommended Skills (Good to Have)
- 📊 **Data Visualization** - Creating meaningful charts
- 📈 **Statistical Analysis** - Hypothesis testing, correlation analysis
- 🧹 **Data Cleaning** - Handling missing values and duplicates
- 💡 **Problem Solving** - Breaking down complex problems
- 📝 **Documentation** - Writing clear code comments

### Advanced Skills (For Next Steps)
- 🤖 **Machine Learning** - Building predictive models
- 🔍 **Feature Engineering** - Creating new features
- 📊 **Advanced Visualization** - Interactive dashboards
- ⚙️ **Model Optimization** - Hyperparameter tuning

---

## 🔧 Technical Skills Breakdown

### 1. Python Programming Fundamentals

#### What You Need to Know:

**Variables and Data Types**
```python
# Integers and Floats
age = 25
salary = 50000.50

# Strings
name = "John Doe"
education = "Bachelors"

# Lists
ages = [25, 30, 35, 40]
countries = ["USA", "UK", "India"]

# Dictionaries
person = {
    'name': 'John',
    'age': 25,
    'income': '>50K'
}
```

**Control Flow**
```python
# If-Else Statements
if income > 50000:
    category = "High earner"
else:
    category = "Low earner"

# For Loops
for column in df.columns:
    print(f"Column: {column}")

# While Loops
count = 0
while count < 10:
    print(count)
    count += 1
```

**Functions**
```python
# Creating a function to calculate outliers
def detect_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers

# Using the function
outliers = detect_outliers(df, 'age')
print(f"Found {len(outliers)} outliers")
```

---

### 2. Pandas Library - Data Manipulation

#### Reading Data
```python
import pandas as pd

# Read CSV file
df = pd.read_csv('data.csv')

# Read with custom column names
column_names = ['age', 'workclass', 'education']
df = pd.read_csv('data.csv', names=column_names)

# Read with options
df = pd.read_csv('data.csv', 
                 skipinitialspace=True,  # Remove leading spaces
                 na_values=['?', 'NA'])   # Treat '?' as missing
```

#### Exploring Data
```python
# View first rows
df.head()          # First 5 rows
df.head(10)        # First 10 rows

# View last rows
df.tail()

# Dataset information
df.info()          # Column types and counts
df.shape           # (rows, columns)
df.columns         # Column names
df.dtypes          # Data types

# Statistical summary
df.describe()      # For numerical columns
df.describe(include='object')  # For categorical columns
```

#### Selecting Data
```python
# Select single column
ages = df['age']

# Select multiple columns
subset = df[['age', 'education', 'income']]

# Select rows by condition
high_earners = df[df['income'] == '>50K']
young_people = df[df['age'] < 30]

# Multiple conditions (AND)
young_high_earners = df[(df['age'] < 30) & (df['income'] == '>50K')]

# Multiple conditions (OR)
extreme_ages = df[(df['age'] < 20) | (df['age'] > 65)]

# Using .loc (by label)
row_10_to_20 = df.loc[10:20]

# Using .iloc (by position)
first_10_rows = df.iloc[:10]
```

#### Data Cleaning
```python
# Check for missing values
df.isnull().sum()
df.isna().sum()

# Check for specific value (like '?')
(df == '?').sum()

# Remove rows with missing values
df_clean = df.dropna()

# Remove specific rows
df_clean = df[df['workclass'] != '?']

# Fill missing values
df['age'].fillna(df['age'].mean(), inplace=True)

# Remove duplicates
df_clean = df.drop_duplicates()
```

#### Data Transformation
```python
# Create new column
df['age_group'] = df['age'] // 10 * 10

# Apply function to column
df['income_binary'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Replace values
df['workclass'].replace('?', 'Unknown', inplace=True)

# Convert data types
df['age'] = df['age'].astype(int)
df['income'] = df['income'].astype('category')
```

#### Aggregation and Grouping
```python
# Value counts
df['education'].value_counts()

# Group by and aggregate
df.groupby('education')['income'].value_counts()
df.groupby('workclass')['age'].mean()

# Multiple aggregations
df.groupby('education').agg({
    'age': ['mean', 'median', 'std'],
    'hours-per-week': ['mean', 'max']
})
```

---

### 3. Statistical Concepts

#### Measures of Central Tendency
```python
# Mean (average)
mean_age = df['age'].mean()
# Example: 38.5 years

# Median (middle value)
median_age = df['age'].median()
# Example: 37.0 years

# Mode (most frequent)
mode_education = df['education'].mode()[0]
# Example: "HS-grad"
```

#### Measures of Spread
```python
# Standard Deviation (how spread out the data is)
std_age = df['age'].std()
# Example: 13.6 years

# Variance
var_age = df['age'].var()

# Range
age_range = df['age'].max() - df['age'].min()
# Example: 73 years (90 - 17)

# Interquartile Range (IQR)
Q1 = df['age'].quantile(0.25)  # 25th percentile
Q3 = df['age'].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1
# Example: Q1=28, Q3=48, IQR=20
```

#### Outlier Detection (IQR Method)
```python
# Calculate bounds
Q1 = df['hours-per-week'].quantile(0.25)
Q3 = df['hours-per-week'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['hours-per-week'] < lower_bound) | 
              (df['hours-per-week'] > upper_bound)]

print(f"Lower bound: {lower_bound}")  # Example: 20 hours
print(f"Upper bound: {upper_bound}")  # Example: 60 hours
print(f"Number of outliers: {len(outliers)}")  # Example: 3,245
```

#### Correlation Analysis
```python
# Correlation between two variables
correlation = df['age'].corr(df['hours-per-week'])
# Example: 0.068 (weak positive correlation)

# Correlation matrix
correlation_matrix = df[['age', 'education-num', 'hours-per-week', 'capital-gain']].corr()

# Interpretation:
# -1.0 to -0.7: Strong negative correlation
# -0.7 to -0.3: Moderate negative correlation
# -0.3 to 0.3: Weak or no correlation
#  0.3 to 0.7: Moderate positive correlation
#  0.7 to 1.0: Strong positive correlation
```

---

### 4. Data Visualization

#### Basic Plots with Matplotlib
```python
import matplotlib.pyplot as plt

# Histogram - Distribution of a variable
plt.figure(figsize=(10, 6))
plt.hist(df['age'], bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

# Bar plot - Categorical data
income_counts = df['income'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(income_counts.index, income_counts.values)
plt.xlabel('Income Category')
plt.ylabel('Count')
plt.title('Income Distribution')
plt.show()

# Scatter plot - Relationship between variables
plt.figure(figsize=(10, 6))
plt.scatter(df['age'], df['hours-per-week'], alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Hours per Week')
plt.title('Age vs Hours Worked')
plt.show()
```

#### Advanced Plots with Seaborn
```python
import seaborn as sns

# Box plot - Show outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x='income', y='age', data=df)
plt.title('Age Distribution by Income Level')
plt.show()

# Heatmap - Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Count plot - Categorical distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='education', data=df, order=df['education'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Education Level Distribution')
plt.show()
```

---

## 🔄 Step-by-Step Process Guide

### Phase 1: Project Setup

#### Step 1.1: Environment Setup
```powershell
# Create project directory
mkdir data_exploration
cd data_exploration

# Create folder structure
mkdir scripts
mkdir "Adult Dataset"
mkdir knowledge

# Install required libraries
pip install pandas numpy matplotlib seaborn scipy
```

#### Step 1.2: Verify Installation
```python
# test_installation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("✓ All libraries installed successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
```

---

### Phase 2: Initial Data Exploration

#### Step 2.1: Load and Inspect Data
```python
import pandas as pd
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "Adult Dataset", "adult.data")

# Define column names
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Load data
df = pd.read_csv(data_path, names=column_names, skipinitialspace=True)

# Initial inspection
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nColumn types:")
print(df.dtypes)
print(f"\nBasic statistics:")
print(df.describe())
```

#### Step 2.2: Identify Data Quality Issues
```python
# Missing values
print("Missing Values Analysis:")
for col in df.columns:
    if df[col].dtype == 'object':
        missing = (df[col] == '?').sum()
    else:
        missing = df[col].isnull().sum()
    
    if missing > 0:
        percentage = (missing / len(df)) * 100
        print(f"{col}: {missing} ({percentage:.2f}%)")

# Duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Data distribution
print("\nTarget Variable Distribution:")
print(df['income'].value_counts())
print(df['income'].value_counts(normalize=True) * 100)
```

---

### Phase 3: Data Cleaning

#### Step 3.1: Handle Missing Values
```python
# Method 1: Remove rows with missing values
print(f"Original shape: {df.shape}")
df_clean = df[df != '?'].dropna()
print(f"After removing missing: {df_clean.shape}")

# Method 2: Replace missing with mode (for categorical)
for col in df.select_dtypes(include=['object']).columns:
    mode_value = df[col].mode()[0]
    df[col].replace('?', mode_value, inplace=True)

# Method 3: Replace missing with mean (for numerical)
df['age'].fillna(df['age'].mean(), inplace=True)
```

#### Step 3.2: Remove Outliers
```python
def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers before removal
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"{column}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
    
    # Remove outliers
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Apply to numerical columns
numerical_cols = ['age', 'hours-per-week', 'capital-gain', 'capital-loss']
df_no_outliers = df.copy()

for col in numerical_cols:
    df_no_outliers = remove_outliers_iqr(df_no_outliers, col)

print(f"\nOriginal size: {len(df)}")
print(f"After removing outliers: {len(df_no_outliers)}")
print(f"Rows removed: {len(df) - len(df_no_outliers)}")
```

---

### Phase 4: Analysis and Insights

#### Step 4.1: Weekly Working Hours Impact on Earning Potential
```python
# Group by income and calculate statistics
hours_analysis = df.groupby('income')['hours-per-week'].agg([
    'mean', 'median', 'std', 'min', 'max'
]).round(2)

print("Hours per Week by Income Level:")
print(hours_analysis)

# Statistical test - Are the differences significant?
from scipy import stats

low_income_hours = df[df['income'] == '<=50K']['hours-per-week']
high_income_hours = df[df['income'] == '>50K']['hours-per-week']

t_stat, p_value = stats.ttest_ind(low_income_hours, high_income_hours)
print(f"\nT-test results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ Significant difference in working hours between income groups")
else:
    print("✗ No significant difference")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='income', y='hours-per-week', data=df)
plt.title('Working Hours by Income Level')
plt.ylabel('Hours per Week')
plt.xlabel('Income Category')
plt.show()
```

#### Step 4.2: Features Highly Correlated with Earning Potential
```python
# Encode income as binary
df['income_binary'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Calculate correlation with income
numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                     'capital-loss', 'hours-per-week']

correlations = {}
for feature in numerical_features:
    corr = df[feature].corr(df['income_binary'])
    correlations[feature] = corr

# Sort by absolute correlation
corr_sorted = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("Features Correlated with Income (sorted by strength):")
print("-" * 60)
for feature, corr in corr_sorted:
    strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.1 else "Weak"
    direction = "Positive" if corr > 0 else "Negative"
    print(f"{feature:20s}: {corr:6.3f} ({strength} {direction})")

# Visualization
plt.figure(figsize=(10, 6))
features = [item[0] for item in corr_sorted]
values = [item[1] for item in corr_sorted]

plt.barh(features, values, color=['green' if v > 0 else 'red' for v in values])
plt.xlabel('Correlation with Income')
plt.title('Feature Correlation with Earning Potential')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.show()
```

#### Step 4.3: Education Years vs Earning Potential
```python
# Analysis
education_income = df.groupby('education-num')['income'].apply(
    lambda x: (x == '>50K').sum() / len(x) * 100
).round(2)

print("Percentage of High Earners by Years of Education:")
print(education_income)

# Correlation
corr_education = df['education-num'].corr(df['income_binary'])
print(f"\nCorrelation between education years and income: {corr_education:.3f}")

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='education-num', y='income_binary', data=df)
plt.title('Education Years vs Income')
plt.xlabel('Years of Education')
plt.ylabel('Income (0=Low, 1=High)')

plt.subplot(1, 2, 2)
plt.plot(education_income.index, education_income.values, marker='o')
plt.xlabel('Years of Education')
plt.ylabel('Percentage Earning >50K')
plt.title('High Earner Percentage by Education')
plt.grid(True)
plt.tight_layout()
plt.show()
```

#### Step 4.4: Age vs Earning Potential
```python
# Create age groups
df['age_group'] = pd.cut(df['age'], 
                         bins=[0, 25, 35, 45, 55, 65, 100],
                         labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])

# Analysis by age group
age_income = df.groupby('age_group')['income'].apply(
    lambda x: (x == '>50K').sum() / len(x) * 100
).round(2)

print("Percentage of High Earners by Age Group:")
print(age_income)

# Correlation
corr_age = df['age'].corr(df['income_binary'])
print(f"\nCorrelation between age and income: {corr_age:.3f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Distribution by age group
df.groupby(['age_group', 'income']).size().unstack().plot(kind='bar', ax=axes[0])
axes[0].set_title('Income Distribution by Age Group')
axes[0].set_xlabel('Age Group')
axes[0].set_ylabel('Count')
axes[0].legend(title='Income')

# Plot 2: Percentage of high earners
axes[1].bar(age_income.index, age_income.values, color='steelblue')
axes[1].set_title('High Earner Percentage by Age')
axes[1].set_xlabel('Age Group')
axes[1].set_ylabel('Percentage Earning >50K')
axes[1].tick_params(axis='x', rotation=45)

# Plot 3: Scatter plot
axes[2].scatter(df['age'], df['income_binary'], alpha=0.1)
axes[2].set_title('Age vs Income (Binary)')
axes[2].set_xlabel('Age')
axes[2].set_ylabel('Income (0=Low, 1=High)')

plt.tight_layout()
plt.show()
```

---

## � Understanding Output Files

When you run the `02_clean_exploration.py` script, it generates 9 files in the `output` folder. Here's what each file contains and how to use it:

### Text/CSV Reports

#### 1. **01_exploration_summary.txt**
**What it contains:**
- Dataset shape (rows and columns)
- List of all column names
- First 10 rows of data
- Data types for each column

**How to use it:**
- Open in any text editor (Notepad, VS Code, etc.)
- Quick reference for dataset structure
- Share with team members for overview

**Example content:**
```
================================================================================
ADULT DATASET - EXPLORATION SUMMARY
================================================================================

DATASET OVERVIEW
--------------------------------------------------------------------------------
Total Records: 32,561
Total Features: 15
Dataset Shape: (32561, 15)

COLUMN NAMES
--------------------------------------------------------------------------------
 1. age
 2. workclass
 3. education
...
```

#### 2. **02_statistical_summary.csv**
**What it contains:**
- Descriptive statistics for all numerical columns
- Mean, standard deviation, min, max
- 25th, 50th (median), 75th percentiles

**How to use it:**
- Open in Excel, Google Sheets, or pandas
- Compare statistics across features
- Identify ranges and distributions

**Example:**
| Column | count | mean | std | min | 25% | 50% | 75% | max |
|--------|-------|------|-----|-----|-----|-----|-----|-----|
| age | 32561 | 38.58 | 13.64 | 17 | 28 | 37 | 48 | 90 |
| hours-per-week | 32561 | 40.44 | 12.35 | 1 | 40 | 40 | 45 | 99 |

#### 3. **03_missing_values_report.csv**
**What it contains:**
- Columns that have missing values
- Count of missing values per column
- Percentage of missing data

**How to use it:**
- Identify which columns need cleaning
- Decide on imputation strategies
- Calculate data quality score

**Example:**
| Column | Missing_Count | Percentage |
|--------|---------------|------------|
| workclass | 1,836 | 5.64% |
| occupation | 1,843 | 5.66% |
| native-country | 583 | 1.79% |

#### 4. **04_outliers_report.csv**
**What it contains:**
- Outlier analysis for each numerical column
- Q1, Q3, and IQR values
- Upper and lower bounds
- Count and percentage of outliers

**How to use it:**
- Identify features with many outliers
- Decide whether to remove or keep outliers
- Understand data quality issues

**Example:**
| Column | Q1 | Q3 | IQR | Lower_Bound | Upper_Bound | Outlier_Count | Outlier_Percentage |
|--------|----|----|-----|-------------|-------------|---------------|-------------------|
| age | 28 | 48 | 20 | -2 | 78 | 1,234 | 3.79% |

### Visualization Files (PNG Images)

#### 5. **05_age_distribution.png**
**What it shows:**
- Histogram of age distribution
- Frequency of each age group

**How to interpret:**
- Most common age ranges
- Whether data is normally distributed
- Age demographics of the dataset

**What to look for:**
- Peak age groups (modal age)
- Skewness (left or right)
- Gaps or unusual patterns

#### 6. **06_income_distribution.png**
**What it shows:**
- Bar chart of income categories
- Count of people in each category (<=50K vs >50K)

**How to interpret:**
- Class balance (is it imbalanced?)
- Majority class vs minority class
- Percentage breakdown

**What to look for:**
- Is one class much larger than the other?
- This affects prediction model performance

#### 7. **07_hours_by_income.png**
**What it shows:**
- Box plot comparing working hours between income groups
- Median, quartiles, and outliers for each group

**How to interpret:**
- Do high earners work more hours?
- What's the typical range for each group?
- Are there outliers (people working unusual hours)?

**What to look for:**
- Median line position
- Box size (variance)
- Outlier points (dots beyond whiskers)

#### 8. **08_education_distribution.png**
**What it shows:**
- Bar chart of education level frequencies
- Count of people at each education level

**How to interpret:**
- Most common education levels
- Education diversity in the dataset
- Potential class imbalances

**What to look for:**
- Which education levels are most represented?
- Are there rare categories?

#### 9. **09_correlation_heatmap.png**
**What it shows:**
- Correlation matrix of numerical features
- Color-coded relationships (red = positive, blue = negative)

**How to interpret:**
- Which features are related to each other?
- Correlation values range from -1 to +1
- Values close to 0 = no correlation

**What to look for:**
- Strong correlations (> 0.7 or < -0.7)
- Features correlated with target variable
- Multicollinearity (features correlated with each other)

**Color guide:**
- 🔴 Dark Red: Strong positive correlation (~1.0)
- 🟠 Light Red/Pink: Moderate positive correlation (0.3-0.7)
- ⚪ White: No correlation (~0.0)
- 🔵 Light Blue: Moderate negative correlation (-0.3 to -0.7)
- 🟦 Dark Blue: Strong negative correlation (~-1.0)

---

### How to Open and Use Output Files

#### For CSV Files:
```powershell
# Open in Excel
start 02_statistical_summary.csv

# Open in pandas (Python)
import pandas as pd
df = pd.read_csv('output/02_statistical_summary.csv')
print(df)
```

#### For PNG Images:
```powershell
# Open in default image viewer
start 05_age_distribution.png

# View in VS Code
code 05_age_distribution.png
```

#### For Text Files:
```powershell
# Open in Notepad
notepad 01_exploration_summary.txt

# View in PowerShell
Get-Content 01_exploration_summary.txt
```

---

### Saving Output Files for Presentations

All images are saved at **300 DPI** (publication quality) and can be directly used in:
- PowerPoint presentations
- Research papers
- Reports and documentation
- Jupyter notebooks
- Websites and blogs

Simply copy the PNG files from the `output` folder and insert them into your documents.

---

## 📊 Practical Examples

Here's a complete example combining all steps:

```python
"""
Complete Data Analysis Script
Author: Data Science Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# ========== SETUP ==========
print("="*80)
print("ADULT DATASET ANALYSIS")
print("="*80)

# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "Adult Dataset", "adult.data")

# Column names
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Load data
df = pd.read_csv(data_path, names=column_names, skipinitialspace=True)
print(f"\n✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ========== DATA CLEANING ==========
print("\n" + "="*80)
print("DATA CLEANING")
print("="*80)

# Remove missing values
original_size = len(df)
df = df[(df != '?').all(axis=1)]
print(f"✓ Removed {original_size - len(df)} rows with missing values")

# Remove outliers
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    return data[(data[column] >= Q1 - 1.5*IQR) & (data[column] <= Q3 + 1.5*IQR)]

before_outlier_removal = len(df)
for col in ['age', 'hours-per-week', 'capital-gain', 'capital-loss']:
    df = remove_outliers(df, col)
print(f"✓ Removed {before_outlier_removal - len(df)} outlier rows")
print(f"✓ Final dataset size: {len(df)} rows")

# ========== ANALYSIS ==========
print("\n" + "="*80)
print("ANALYSIS RESULTS")
print("="*80)

# Encode income
df['income_binary'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# 1. Working hours impact
print("\n1. IMPACT OF WORKING HOURS ON INCOME:")
print("-" * 60)
hours_by_income = df.groupby('income')['hours-per-week'].mean()
for income_level, hours in hours_by_income.items():
    print(f"   {income_level}: {hours:.2f} hours/week average")

# 2. Feature correlations
print("\n2. FEATURES CORRELATED WITH INCOME:")
print("-" * 60)
features = ['age', 'education-num', 'hours-per-week', 'capital-gain']
for feature in features:
    corr = df[feature].corr(df['income_binary'])
    print(f"   {feature:20s}: {corr:.3f}")

# 3. Education impact
print("\n3. EDUCATION IMPACT:")
print("-" * 60)
edu_impact = df.groupby('education-num')['income_binary'].mean() * 100
print(f"   Correlation: {df['education-num'].corr(df['income_binary']):.3f}")
print(f"   10 years education: {edu_impact[10]:.1f}% earn >50K")
print(f"   16 years education: {edu_impact[16]:.1f}% earn >50K")

# 4. Age impact
print("\n4. AGE IMPACT:")
print("-" * 60)
age_corr = df['age'].corr(df['income_binary'])
print(f"   Correlation: {age_corr:.3f}")
df['age_group'] = pd.cut(df['age'], bins=[0,30,40,50,100], labels=['<30','30-40','40-50','50+'])
for age_group, data in df.groupby('age_group'):
    pct = (data['income'] == '>50K').sum() / len(data) * 100
    print(f"   {age_group}: {pct:.1f}% earn >50K")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
```

---

## 🎓 Learning Path

### Week 1: Python Fundamentals
- **Day 1-2**: Variables, data types, operators
- **Day 3-4**: Control flow (if/else, loops)
- **Day 5-6**: Functions and modules
- **Day 7**: Practice exercises

### Week 2: Pandas Basics
- **Day 1-2**: Reading and writing data
- **Day 3-4**: Data selection and filtering
- **Day 5-6**: Data cleaning techniques
- **Day 7**: Mini-project

### Week 3: Statistics and Analysis
- **Day 1-2**: Descriptive statistics
- **Day 3-4**: Correlation and relationships
- **Day 5-6**: Outlier detection methods
- **Day 7**: Statistical tests

### Week 4: Visualization
- **Day 1-3**: Matplotlib basics
- **Day 4-5**: Seaborn for statistical plots
- **Day 6-7**: Final project

---

## 📚 Recommended Resources

### Online Courses
- **Python for Data Analysis** - Coursera
- **Pandas Tutorial** - DataCamp
- **Statistics for Data Science** - Udemy

### Books
- "Python for Data Analysis" by Wes McKinney
- "Hands-On Machine Learning" by Aurélien Géron
- "Data Science from Scratch" by Joel Grus

### Practice Platforms
- **Kaggle** - Real datasets and competitions
- **HackerRank** - Python challenges
- **LeetCode** - Problem-solving practice

### Documentation
- [Pandas Official Docs](https://pandas.pydata.org/docs/)
- [NumPy User Guide](https://numpy.org/doc/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/)

---

## ✅ Skills Checklist

Use this checklist to track your progress:

### Python Programming
- [ ] Can write variables and basic data types
- [ ] Understand if/else statements
- [ ] Can write for loops
- [ ] Can create and call functions
- [ ] Understand file paths and os module

### Pandas
- [ ] Can read CSV files
- [ ] Can select rows and columns
- [ ] Can filter data with conditions
- [ ] Can handle missing values
- [ ] Can group and aggregate data

### Statistics
- [ ] Understand mean, median, mode
- [ ] Know standard deviation and variance
- [ ] Can detect outliers using IQR
- [ ] Understand correlation
- [ ] Can interpret statistical results

### Visualization
- [ ] Can create basic plots
- [ ] Can create histograms
- [ ] Can create scatter plots
- [ ] Can customize plot appearance
- [ ] Can interpret visualizations

### Analysis
- [ ] Can formulate research questions
- [ ] Can clean messy data
- [ ] Can find patterns in data
- [ ] Can draw insights from statistics
- [ ] Can present findings clearly

---

**Document Version**: 1.0  
**Last Updated**: November 7, 2025  
**Author**: Data Science Team
