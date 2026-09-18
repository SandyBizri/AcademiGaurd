import joblib
import pandas as pd


def predict_student(student_data, threshold=0.35):
    # Load trained model
    model = joblib.load("models/risk_model.pkl")

    # Convert student data to DataFrame
    input_df = pd.DataFrame([student_data])

    # Keep only features the model was trained on
    model_features = model.feature_names_in_
    input_df = input_df[[f for f in model_features if f in input_df.columns]]

    # Predict probability for class 1 (At Risk)
    prob = model.predict_proba(input_df)[:, 1][0]

    # Determine label based on threshold
    label = "At Risk" if prob > threshold else "Low Risk"

    return label, prob
