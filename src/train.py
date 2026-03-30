import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
from  data_preprocessing import preprocess_data,scale_data

import pandas as pd

# 1. Load
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# 2. Preprocess
X, y = preprocess_data(df)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Scale
X_train, X_test, scaler = scale_data(X_train, X_test)

# 5. Train model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# 6. Threshold tuning
threshold = 0.4
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > threshold).astype(int)

# 7. Evaluate
print(classification_report(y_test, y_pred))

# 8. Save model
model_data = {
    "model": model,
    "scaler": scaler,
    "threshold": threshold,
    "columns": X.columns.tolist()
}

# Ensure folder exists
os.makedirs("models", exist_ok=True)

# Save
with open("models/model_bundle.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("✅ Model saved successfully")

