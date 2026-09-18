def calculate_risk(row):
    score = 0
    if row["G1"] < 10:
        score += 2
    if row["G2"] < 10:
        score += 2
    if row["failures"] >= 1:
        score += 2
    if row["studytime"] <= 1:
        score += 1
    if row["absences"] > 10:
        score += 1
    if row["internet"] == 0:
        score += 1
    if row["higher"] == 0:
        score += 1
    return score


def classify_risk(score):
    if score >= 6:
        return "High"
    elif score >= 3:
        return "Medium"
    else:
        return "Low"
