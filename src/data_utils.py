from pathlib import Path
import pandas as pd

EXPECTED_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"

def load_dataset(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download creditcard.csv from Kaggle and place it in data/raw/."
        )

    df = pd.read_csv(path)
    required = EXPECTED_FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[required].copy()
    df = df.dropna(subset=required)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    invalid_labels = set(df[TARGET].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"Class must contain only 0 and 1. Found: {invalid_labels}")

    for column in EXPECTED_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=EXPECTED_FEATURES)
    return df
