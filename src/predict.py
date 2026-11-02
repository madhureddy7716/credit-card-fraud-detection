from pathlib import Path
import argparse
import joblib
import pandas as pd
from src.data_utils import EXPECTED_FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud_detection_model.joblib"

def main():
    parser = argparse.ArgumentParser(description="Score transactions using the trained fraud model.")
    parser.add_argument("--input", required=True, help="CSV with Time, V1-V28, Amount columns")
    parser.add_argument("--output", default="reports/predictions.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.50, help="Fraud probability threshold")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run: python -m src.train")
    df = pd.read_csv(args.input)
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    model = joblib.load(MODEL_PATH)
    X = df[EXPECTED_FEATURES].copy()
    probabilities = model.predict_proba(X)[:, 1]
    results = df.copy()
    results["fraud_probability"] = probabilities
    results["prediction"] = (probabilities >= args.threshold).astype(int)
    results["prediction_label"] = results["prediction"].map({0: "Legitimate", 1: "Potential Fraud"})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Scored {len(results):,} transactions.")
    print(f"Potential fraud flagged: {(results['prediction'] == 1).sum():,}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
