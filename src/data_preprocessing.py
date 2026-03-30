# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
def load_data(path):
    return pd.read_csv(path)

# Clean + encode
def preprocess_data(df):
    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Drop ID
    df.drop('customerID', axis=1, inplace=True)

    # Target encoding
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Split
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    return X, y

# Scaling
def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler