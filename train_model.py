import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

RANDOM_SEED = 42
N_SAMPLES = 500
DATASET_PATH = "dataset.csv"
MODEL_PATH = "model.pkl"

INTERNAL_MAX = 40
INTERNAL_WEIGHT = 70
ATTENDANCE_WEIGHT = 30
MIN_PASSING_INTERNAL_MARKS = 14
MIN_PASSING_ATTENDANCE = 50
GRADE_THRESHOLDS = [
    (85, "A+ Grade"),
    (75, "A Grade"),
    (65, "B Grade"),
    (55, "C Grade"),
    (45, "D Grade"),
]


def assign_grade(internal_marks, attendance, score):
    if internal_marks < MIN_PASSING_INTERNAL_MARKS or attendance < MIN_PASSING_ATTENDANCE:
        return "Fail"
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "Fail"


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a synthetic student grade dataset and return it as a DataFrame."""
    rng = np.random.default_rng(seed)

    internal_marks = rng.integers(0, INTERNAL_MAX + 1, n_samples)
    attendance = rng.integers(0, 101, n_samples)

    base_score = (internal_marks / INTERNAL_MAX) * INTERNAL_WEIGHT + (attendance / 100) * ATTENDANCE_WEIGHT
    noise = rng.normal(0, 3, n_samples)  
    score = np.clip(base_score + noise, 0, 100)

    grades = [
        assign_grade(im, att, sc)
        for im, att, sc in zip(internal_marks, attendance, score)
    ]

    return pd.DataFrame({
        "Internal Marks": internal_marks,
        "Attendance": attendance,
        "Grade": grades,
    })


def train_and_save_model(df: pd.DataFrame):
    """Train the RandomForest classifier on the dataset and persist it to disk."""
    feature_names = ["Internal Marks", "Attendance"]
    X = df[feature_names]
    y = df["Grade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {test_accuracy:.4f}")

    joblib.dump(
        {"model": model, "feature_names": feature_names, "test_accuracy": test_accuracy},
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv(DATASET_PATH, index=False)
    print(f"Dataset saved to {DATASET_PATH} ({len(dataset)} rows)")

    train_and_save_model(dataset)
