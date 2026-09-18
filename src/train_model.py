import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train():
    # Load processed dataset
    df = pd.read_csv("data/processed/students_with_risk.csv")

    # Drop G3 if exists (we only want G1 and G2)
    if "G3" in df.columns:
        df = df.drop(columns=["G3"])

    # Create binary target (0 = Low Risk, 1 = Medium/High Risk)
    df["target"] = df["risk_level"].apply(
        lambda x: 1 if x in ["Medium", "High"] else 0
    )

    # Define features (X) and label (y) - only use G1, G2 and other predictors
    X = df.drop(["risk_level", "risk_score", "target"], axis=1)
    y = df["target"]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create Random Forest model with class balancing
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict probabilities on test set
    y_probs = model.predict_proba(X_test)[:, 1]

    # Lower threshold to increase recall
    threshold = 0.35
    y_pred = (y_probs > threshold).astype(int)

    # Evaluate model
    print("\nModel Evaluation:\n")
    print(classification_report(y_test, y_pred))

    # Save trained model
    joblib.dump(model, "models/risk_model.pkl")
    print("\nModel saved to models/risk_model.pkl")

    return model
