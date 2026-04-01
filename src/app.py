import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# =========================
# Load paths properly
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
threshold_path = os.path.join(BASE_DIR, "models", "threshold.pkl")

# =========================
# Load objects
# =========================
model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
threshold = pickle.load(open(threshold_path, "rb"))

# =========================
# Home route
# =========================
@app.route("/")
def home():
    return "🚀 Churn Prediction API is running!"

# =========================
# Prediction route
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # ⚠️ IMPORTANT:
        # You must apply SAME preprocessing as training
        # For now assuming input is already numeric & aligned

        # Scale input
        X = scaler.transform(df)

        # Predict
        prob = model.predict_proba(X)[0][1]
        pred = "Churn" if prob > threshold else "No Churn"

        return jsonify({
            "churn_probability": round(float(prob), 4),
            "prediction": pred
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

# =========================
# Run app
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)