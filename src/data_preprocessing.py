import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, training=True):

    # Fix TotalCharges
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df.dropna(inplace=True)

    # Drop ID safely
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    if training:
        # Target encoding
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

        # Split
        X = df.drop('Churn', axis=1)
        y = df['Churn']

        # One-hot encoding
        X = pd.get_dummies(X, drop_first=True)

        return X, y

    else:
        # Only features during prediction
        df = pd.get_dummies(df, drop_first=True)
        return df, None


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler