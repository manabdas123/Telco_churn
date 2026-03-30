# src/predict.py

import pickle
import pandas as pd

def load_model():
    return pickle.load(open("models/model_bundle.pkl", "rb"))

def predict(input_dict):
    data = load_model()

    model = data['model']
    scaler = data['scaler']
    threshold = data['threshold']
    columns = data['columns']

    # Convert input to dataframe
    df = pd.DataFrame([input_dict])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align columns (VERY IMPORTANT 🔥)
    df = df.reindex(columns=columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    prob = model.predict_proba(df_scaled)[:,1]
    pred = (prob > threshold).astype(int)

    return pred[0]