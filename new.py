#Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

# Loading the dataset
df = pd.read_csv("//workspaces//codespaces-jupyter//healthcare_dataset (13).csv")

# Displaying the first 10 records
print(df.head(10))

# Missing values per variable
print(df.isnull().sum())

# Number of male and female patients
print(df['Gender'].value_counts())

# Average BMI of the patients
print("Average BMI:", df['BMI'].mean())

# Descriptive statistics
# Summary statistics
print(df[['Age', 'BMI', 'Hospital_Visits_Per_Year', 'Exercise_Frequency']].describe())

# Patient with highest hospital visits
print(df.loc[df['Hospital_Visits_Per_Year'].idxmax()])

# Most common cholesterol level category
print(df['Cholesterol_Level'].mode()[0])

#Data visualization
# Histogram of patient ages
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Histogram of Patient Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
# Saving the plot as an image file
plt.savefig('patient_ages_histogram.png')

# Boxplot of BMI by Gender
plt.figure(figsize=(8,5))
sns.boxplot(x='Gender', y='BMI', data=df)
plt.title('BMI Distribution by Gender')
# Saving the plot as an image file
plt.savefig('bmi_by_gender_boxplot.png')

# Countplot of Smoking Status
plt.figure(figsize=(8,5))
sns.countplot(x='Smoking_Status', data=df)
plt.title('Smoking Status Distribution')
# Saving the plot as an image file
plt.savefig('smoking_status_countplot.png')

# Scatter plot: BMI vs Exercise Frequency
plt.figure(figsize=(8,5))
sns.scatterplot(x='Exercise_Frequency', y='BMI', data=df)
plt.title('BMI vs Exercise Frequency')
# Saving the plot as an image file
plt.savefig('bmi_vs_exercise_frequency_scatterplot.png')

# Pie chart: Medication Adherence
plt.figure(figsize=(6,6))
df['Medication_Adherence'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Medication Adherence Distribution')
plt.ylabel('')
# Saving the plot as an image file
plt.savefig('medication_adherence_pie_chart.png')

# Heatmap of correlations
plt.figure(figsize=(10,6))
corr = df[['Age', 'BMI', 'Hospital_Visits_Per_Year', 'Exercise_Frequency']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
# Saving the plot as an image file
plt.savefig('correlation_heatmap.png')

# Chi square tests for independence
# Gender vs Smoking_Status
contingency1 = pd.crosstab(df['Gender'], df['Smoking_Status'])
chi2_1, p_1, _, _ = chi2_contingency(contingency1)
print(f"Chi-Square Test (Gender vs Smoking): p-value = {p_1}")

#Diabetes vs Blood_Pressure
contingency2 = pd.crosstab(df['Diabetes'], df['Blood_Pressure'])
chi2_2, p_2, _, _ = chi2_contingency(contingency2)
print(f"Chi-Square Test (Diabetes vs BP): p-value = {p_2}")

# T-tests
# BMI by Gender
male_bmi = df[df['Gender'] == 'Male']['BMI']
female_bmi = df[df['Gender'] == 'Female']['BMI']
t_stat, p_val = ttest_ind(male_bmi, female_bmi, nan_policy='omit')
print(f"T-Test (BMI Male vs Female): p-value = {p_val}")

# One way ANOVA 
groups = [group['Exercise_Frequency'].dropna() for name, group in df.groupby('Medication_Adherence')]
anova_stat, anova_p = f_oneway(*groups)
print(f"ANOVA (Exercise Frequency by Medication Adherence): p-value = {anova_p}")

# Correlation analysis
# Correlation matrix
corr_matrix = df[['Age', 'BMI', 'Hospital_Visits_Per_Year', 'Exercise_Frequency']].corr()
print(corr_matrix)
# Visualizing the correlation matrix using a heat map
plt.figure(figsize = (8,6))
sns.heatmap(corr_matrix, annot=True, cmap='viridis')
plt.title('Correlation Matrix Heatmap')
# Saving the plot as an image file
plt.savefig('correlation_matrix_heatmap.png')


# Strongest correlations
corr_pairs = corr_matrix.unstack().sort_values()
strongest_neg = corr_pairs[corr_pairs < 0].idxmin()
strongest_pos = corr_pairs[corr_pairs > 0].idxmax()
print(f"Strongest Positive Correlation: {strongest_pos} = {corr_pairs[strongest_pos]:.2f}")
print(f"Strongest Negative Correlation: {strongest_neg} = {corr_pairs[strongest_neg]:.2f}")
























