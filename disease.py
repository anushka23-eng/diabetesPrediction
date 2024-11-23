# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

# Step 1: Load the dataset
# Replace 'diabetes.csv' with your actual file path
data = pd.read_csv("diabetes.csv")  # Ensure correct path to the dataset

# Step 2: Overview of the dataset
print("Dataset Overview:")
print(data.head())  # Displays first 5 rows
print("\nDataset Info:")
print(data.info())  # Summary of the dataset's columns and missing values
print("\nSummary Statistics:")
print(data.describe())  # Descriptive statistics for numeric columns

# Step 3: Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())  # Check for null/missing values

# Step 4: Visualize Target Distribution
sns.countplot(x='Outcome', data=data, palette='viridis')
plt.title("Target Distribution")
plt.show()

# Step 5: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Step 6: Preprocessing
# Handle missing values if any (not needed if no missing values)
# data.fillna(data.mean(), inplace=True)  # Example, if there are missing values

# Split dataset into features (X) and target (y)
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Target variable

# Normalize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check the class distribution after SMOTE
print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_smote.value_counts())

# Step 9: Modeling - Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_smote, y_train_smote)

# Predictions and evaluation for Logistic Regression
y_pred = log_reg.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, y_pred))

# Confusion matrix for Logistic Regression
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Logistic Regression)")
plt.show()

# Step 10: Modeling - LightGBM
lgbm = lgb.LGBMClassifier()
lgbm.fit(X_train_smote, y_train_smote)

# Predictions and evaluation for LightGBM
y_pred_lgbm = lgbm.predict(X_test)
print("\nLightGBM Accuracy:", accuracy_score(y_test, y_pred_lgbm))
print("LightGBM ROC-AUC:", roc_auc_score(y_test, y_pred_lgbm))

# Confusion matrix for LightGBM
conf_matrix_lgbm = confusion_matrix(y_test, y_pred_lgbm)
sns.heatmap(conf_matrix_lgbm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (LightGBM)")
plt.show()

# Step 11: Model Comparison
print("\nModel Comparison:")
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgbm))
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, y_pred))
print("LightGBM ROC-AUC:", roc_auc_score(y_test, y_pred_lgbm))

# Final conclusion based on ROC-AUC
if roc_auc_score(y_test, y_pred_lgbm) > roc_auc_score(y_test, y_pred):
    print("\nLightGBM model performs better.")
else:
    print("\nLogistic Regression model performs better.")
