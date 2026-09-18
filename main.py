from src.train_model import train
from src.predict import predict_student


def main():
    # Train the model (Week 2 ML)
    print("Training model...")
    train()

    # Test prediction with a sample student (no G3)
    print("\nTesting prediction with sample student...\n")
    sample_student = {
        "G1": 10,
        "G2": 9,
        "studytime": 2,
        "failures": 1,
        "absences": 15,
        "internet": 1,
        "higher": 1
    }

    label, probability = predict_student(sample_student)
    print(f"Prediction:{label} (Probability At Risk:{probability:.2f})")


if __name__ == "__main__":
    main()
