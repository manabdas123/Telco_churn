import pickle
import pandas as pd

# Load model bundle
def load_model():
    return pickle.load(open("models/model_bundle.pkl", "rb"))

# Load once (IMPORTANT)
data = load_model()

model = data['model']
scaler = data['scaler']
threshold = data['threshold']
columns = data['columns']


def predict(input_dict):
    # Convert input to DataFrame
    df = pd.DataFrame([input_dict])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align columns (VERY IMPORTANT)
    df = df.reindex(columns=columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict probability
    prob = model.predict_proba(df_scaled)[:, 1]

    # Apply threshold
    pred = int(prob > threshold)

    return prob[0], pred


# 🔥 TEST BLOCK (Run this file directly)
if __name__ == "__main__":
    sample_input = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.5,
        "TotalCharges": 850.5
    }

    prob, pred = predict(sample_input)

    print("🔹 Churn Probability:", round(prob, 4))
    print("🔹 Prediction:", "Churn" if pred == 1 else "No Churn")