
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

df = pd.read_csv("C:\\Users\\Arpit\\Desktop\\Internships Arpit\\Elevate labs AIML internship\\Titanic-Dataset (2).csv")  # Make sure 'train.csv' is in your working directory

print("First 5 rows of the dataset:")
display(df.head())

print("\nBasic Info:")
df.info()

print("\nSummary Statistics:")
display(df.describe())

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['Age'].dropna(), kde=True, bins=30, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['Fare'], kde=True, bins=30, color='orange')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Passenger Class vs Age')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']])
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()
