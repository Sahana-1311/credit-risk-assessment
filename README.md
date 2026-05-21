# Credit Risk Assessment & Decision Support System

A binary classification system to predict loan default risk using machine learning.

## Overview
Built on a 10,000-record loan dataset to help risk analysts make data-driven lending decisions. Achieves 91% accuracy and ROC-AUC of 0.94.

## Key Features
- Random Forest and XGBoost classifiers with GridSearchCV hyperparameter tuning
- SMOTE oversampling to handle severe class imbalance (8% default rate) — boosted minority-class recall from 61% to 79%
- Top 5 risk drivers identified via permutation importance: debt-to-income ratio, credit utilisation, missed payments, loan tenure, employment type
- Threshold-tuning module for analysts to shift the decision boundary and visualise precision-recall trade-offs

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib

## Results
- Accuracy: 91%
- ROC-AUC: 0.94
- Minority-class recall improved by 18 percentage points after SMOTE
