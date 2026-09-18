from ucimlrepo import fetch_ucirepo
import pandas as pd
import os


def load_student_data():
    dataset = fetch_ucirepo(id=320)
    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/student_performance.csv", index=False)
    return df
