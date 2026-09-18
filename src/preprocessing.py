def preprocess_data(df):

    df = df.copy()

    binary_columns = ["schoolsup", "famsup", "paid", "activities",
                      "nursery", "higher", "internet", "romantic"]

    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})

    selected_features = [
        "G1", "G2",
        "studytime",
        "failures",
        "absences",
        "internet",
        "higher"
    ]

    df = df[selected_features]

    return df
