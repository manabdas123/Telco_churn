import os
import pickle
import pandas as pd
from flask import Flask, request, render_template

from data_preprocessing import preprocess_data, scale_data  # reuse logic


app = Flask(__name__)

# =========================
# Load Model + Columns
# =========================
model_path = os.path.join("models", "model_bundle.pkl")
columns_path = os.path.join("models", "columns.pkl")

with open(model_path, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data["scaler"]
threshold = model_data["threshold"]

with open(columns_path, "rb") as f:
    columns = pickle.load(f)

# =========================
# Routes
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # =========================
        # 1. Get input from form
        # =========================
        input_data = request.form.to_dict()

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # =========================
        # 2. Preprocess
        # =========================
        input_df, _ = preprocess_data(input_df, training=False)

        # =========================
        # 3. Align columns
        # =========================
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # =========================
        # 4. Scale
        # =========================
        input_scaled = scaler.transform(input_df)

        # =========================
        # 5. Predict
        # =========================
        prob = model.predict_proba(input_scaled)[0][1]
        prediction = int(prob > threshold)

        result = "Churn" if prediction == 1 else "No Churn"

        return render_template("index.html", prediction_text=f"Result: {result} (Prob: {prob:.2f})")

    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)