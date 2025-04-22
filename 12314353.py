import pandas as pd

# Load dataset
df = pd.read_excel(r"C:\Users\sai30\OneDrive\Desktop\NewPython.py\Employee_Details_0.xlsx")
df.columns = df.columns.str.strip()

# Correct column name
df.rename(columns={'MartitalStatus': 'MaritalStatus'}, inplace=True)

# Convert date columns
df['BirthDate'] = pd.to_datetime(df['BirthDate'], origin='1899-12-30', errors='coerce')
df['Schedule RetirementDate'] = pd.to_datetime(df['Schedule RetirementDate'], origin='1899-12-30', errors='coerce')

# Calculate age
df['Age'] = pd.Timestamp.now().year - df['BirthDate'].dt.year

# Clean categorical columns
string_cols = ['Gender', 'IsTeaching', 'Community Name', 'Religion Name', 'Caste Name', 'MaritalStatus']
for col in string_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# Replace NaN values in SikkimSubjectNo
df['SikkimSubjectNo'] = df['SikkimSubjectNo'].fillna(0)

# Remove rows with missing values and duplicates
df_cleaned = df.dropna()
df_cleaned = df_cleaned.drop_duplicates()

# Check dataset info
print(" DataFrame Info (After Cleaning):")
df_cleaned.info()

# Statistical summary
print("\n Statistical Summary (Numerical Columns):")
print(df_cleaned.describe())



import matplotlib.pyplot as plt
import seaborn as sns

# Reload data 
df = pd.read_excel(r"C:\Users\sai30\OneDrive\Desktop\NewPython.py\Employee_Details_0.xlsx")
df.columns = df.columns.str.strip()
df.rename(columns={'MartitalStatus': 'MaritalStatus'}, inplace=True)
df['BirthDate'] = pd.to_datetime(df['BirthDate'], origin='1899-12-30', errors='coerce')
df['Schedule RetirementDate'] = pd.to_datetime(df['Schedule RetirementDate'], origin='1899-12-30', errors='coerce')
df['Age'] = pd.Timestamp.now().year - df['BirthDate'].dt.year

string_cols = ['Gender', 'IsTeaching', 'Community Name', 'Religion Name', 'Caste Name', 'MaritalStatus']
for col in string_cols:
    df[col] = df[col].astype(str).str.strip().str.title()



# Check dataset shape
print(" Dataset Shape:", df.shape)
print(" Column Types & Nulls:")
print(df.info())
print("\n Description:")
print(df.describe(include='all'))



# Pie chart - Gender distribution
df = pd.read_excel(r"C:\Users\sai30\OneDrive\Desktop\NewPython.py\Employee_Details_0.xlsx")
if 'Gender' in df.columns:
    gender_counts = df['Gender'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title("Gender Distribution", fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
else:
    print("❌ 'Gender' column not found in the dataset.")



# Histogram - Age distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'].dropna(), bins=25, kde=True, color='skyblue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()



# Countplot - Teaching vs Non-Teaching
plt.figure(figsize=(6,4))
sns.countplot(x='IsTeaching', data=df)
plt.title("Teaching vs Non-Teaching")
plt.show()



# Top 10 Caste / Community / Religion
for col in ['Caste Name', 'Community Name', 'Religion Name']:
    plt.figure(figsize=(10,4))
    top_vals = df[col].value_counts().head(10)
    sns.barplot(x=top_vals.values, y=top_vals.index, palette='pastel')
    plt.title(f"Top 10 {col}")
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.show()



# Marital status
plt.figure(figsize=(6,4))
sns.countplot(x='MaritalStatus', data=df)
plt.title("Marital Status")
plt.show()



# Designation
plt.figure(figsize=(10,5))
top_designations = df['Designation'].value_counts().head(10)
sns.barplot(x=top_designations.values, y=top_designations.index, palette='muted')
plt.title("Top 10 Designations")
plt.xlabel("Count")
plt.ylabel("Designation")
plt.show()



# Heatmap - Correlation
df_numeric = df.copy()
for col in df_numeric.columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
df_numeric = df_numeric.dropna(axis=1, how='all')
df_numeric = df_numeric.loc[:, df_numeric.nunique() > 1]
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()



# Donut chart - Appointment Status
status_counts = df['Appointment Status'].value_counts()
fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(
    status_counts,
    labels=None,
    autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '',
    startangle=140,
    pctdistance=0.85,
    wedgeprops={'width': 0.4}
)
ax.legend(
    wedges, status_counts.index,
    title='Appointment Status',
    loc='center left',
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=10
)
plt.title('Appointment Status Distribution (Donut Chart with Legend)', fontsize=14)
plt.tight_layout()
plt.show()



# Boxplot - Hierarchy vs Teaching
plt.figure(figsize=(10, 6))
sns.boxplot(x='IsTeaching', y='Hierarchy', data=df, palette='Set2')
plt.title("Distribution of Hierarchy by Teaching Status")
plt.xlabel("Is Teaching")
plt.ylabel("Hierarchy Level")
plt.tight_layout()
plt.show()



# Skewness & Kurtosis
import numpy as np
from scipy.stats import skew, kurtosis

df = pd.read_excel(r"C:\Users\sai30\OneDrive\Desktop\NewPython.py\Employee_Details_0.xlsx")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print(" Dataset Shape:", df.shape)
print(" Column Names:\n", df.columns.tolist())
print("\n Data Types:\n", df.dtypes)

numerical = df.select_dtypes(include=[np.number])
print("\n Numerical Summary (Full):")
print(numerical.describe().T)

print("\n Skewness & Kurtosis:")
for col in numerical.columns:
    print(f"\n🔸 {col}")
    print(f"Skewness : {skew(df[col].dropna()):.2f}")
    print(f"Kurtosis : {kurtosis(df[col].dropna()):.2f}")



# Central tendency & spread
df = pd.read_excel(r"C:\Users\sai30\OneDrive\Desktop\NewPython.py\Employee_Details_0.xlsx")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

numerical = df.select_dtypes(include=[np.number])
print("\n📏 Central Tendency + Spread:")
for col in numerical.columns:
    col_data = df[col].dropna()
    print(f"\n🔹 {col}")
    print(f"Mean      : {col_data.mean():.2f}")
    print(f"Median    : {col_data.median():.2f}")
    print(f"Mode      : {col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A'}")
    print(f"Min       : {col_data.min()}")
    print(f"Max       : {col_data.max()}")
    print(f"Range     : {col_data.max() - col_data.min()}")
    print(f"Std Dev   : {col_data.std():.2f}")
    print(f"IQR       : {col_data.quantile(0.75) - col_data.quantile(0.25):.2f}")



# Outlier detection using IQR
from statsmodels.stats.weightstats import ztest
from scipy.stats import ttest_1samp, ttest_ind, chi2_contingency

df = pd.read_excel(r"C:\Users\sai30\OneDrive\Desktop\NewPython.py\Employee_Details_0.xlsx")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
numerical = df.select_dtypes(include=[np.number])

print("\n Outlier Detection using IQR Method:")
for col in numerical.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\n🔹 {col}")
    print(f"IQR Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Outliers Detected: {outliers.shape[0]}")



# One-sample T-test
if 'hierarchy' in df.columns:
    t_stat, p_value = ttest_1samp(df['hierarchy'].dropna(), 100)
    print("\n One-sample T-Test: Is Hierarchy ≠ 100?")
    print(f"T-statistic: {t_stat:.2f}")
    print(f"P-value: {p_value:.4f}")
else:
    print("\n 'hierarchy' column not found for t-test.")



# One-sample Z-test
if 'hierarchy' in df.columns:
    hierarchy_data = df['hierarchy'].dropna()
    z_stat, p_value = ztest(hierarchy_data, value=100)
    print("\n One-sample Z-Test: Is Hierarchy ≠ 100?")
    print(f"Z-statistic: {z_stat:.2f}")
    print(f"P-value: {p_value:.4f}")
else:
    print("\n 'hierarchy' column not found for Z-test.")
