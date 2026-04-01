import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn

from data_preprocessing import preprocess_data, scale_data

# =========================
# ✅ FIX: Set tracking location
# =========================
mlflow.set_tracking_uri("file:./mlruns")

# =========================
# Start MLflow Experiment
# =========================
mlflow.set_experiment("Telco Churn Prediction")

with mlflow.start_run():

    # =========================
    # 1. Load Data
    # =========================
    df = pd.read_csv("data/Telco-Customer-Churn.csv")

    # =========================
    # 2. Preprocess
    # =========================
    X, y = preprocess_data(df)

    # =========================
    # 3. Split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # 4. Scale
    # =========================
    X_train, X_test, scaler = scale_data(X_train, X_test)

    # =========================
    # 5. Train model
    # =========================
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    # =========================
    # 6. Threshold tuning
    # =========================
    threshold = 0.4
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    # =========================
    # 7. Evaluate
    # =========================
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # =========================
    # 🔥 MLflow Logging
    # =========================
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("threshold", threshold)

    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # =========================
    # Save model locally
    # =========================
    model_data = {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "columns": X.columns.tolist()
    }

    os.makedirs("models", exist_ok=True)

    with open("models/model_bundle.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print("✅ Model saved successfully")
