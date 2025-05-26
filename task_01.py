import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


df = pd.read_csv("C:\\Users\\Arpit\\Desktop\\Arpit Python\\Elevate labs AIML\\Task 1\\Titanic-Dataset (1).csv")


df['Age'].fillna(df['Age'].mean(), inplace=True)

#
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cols_to_analyze = [col for col in numerical_cols if col not in ['PassengerId', 'Survived', 'Pclass']]


plt.figure(figsize=(15, 10))
for i, column in enumerate(cols_to_analyze, 1):
    plt.subplot(len(cols_to_analyze), 2, i)
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()


def remove_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
       
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        print(f"Removed {len(df) - len(df_clean)} outliers from {col}")
    
    return df_clean


df_no_outliers = remove_outliers(df, cols_to_analyze)

plt.figure(figsize=(15, 10))
for i, column in enumerate(cols_to_analyze, 1):
    plt.subplot(len(cols_to_analyze), 2, i)
    sns.boxplot(x=df_no_outliers[column])
    plt.title(f'Boxplot of {column} (After removing outliers)')
plt.tight_layout()
plt.show()


def normalize_min_max(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def standardize(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def robust_scale(df, columns):
    scaler = RobustScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

df_no_outliers = standardize(df_no_outliers, cols_to_analyze)

print("Data after removing outliers and standardization:")
print(df_no_outliers.head()) 
