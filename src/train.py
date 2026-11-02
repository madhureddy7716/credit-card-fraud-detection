
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_utils import (
    load_dataset,
    EXPECTED_FEATURES,
    TARGET,
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "fraud_detection_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Default probability threshold for classifying a transaction.
THRESHOLD = 0.50


# ============================================================
# 2. FEATURE PREPROCESSING
# ============================================================

def make_preprocessor():
    """
    Scale Time and Amount.

    V1-V28 are already anonymized PCA-transformed features.
    """

    return ColumnTransformer(
        transformers=[
            (
                "scale_time_amount",
                StandardScaler(),
                ["Time", "Amount"],
            ),
            (
                "pca_features",
                "passthrough",
                [f"V{i}" for i in range(1, 29)],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ============================================================
# 3. DEFINE MACHINE LEARNING MODELS
# ============================================================

def make_models():
    """
    Create two classification pipelines.

    Model 1: Logistic Regression
    Model 2: Random Forest
    """

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=16,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


# ============================================================
# 4. SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(y_true, y_pred, model_name):
    """
    Create a confusion matrix using Matplotlib only.
    Seaborn is not required.
    """

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    figure, axis = plt.subplots(figsize=(7, 5))

    image = axis.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"Confusion Matrix - {model_name}",
    )

    # Display the values inside the matrix.
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            axis.text(
                column,
                row,
                f"{cm[row, column]:,}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )

    figure.tight_layout()

    output_path = REPORT_DIR / "confusion_matrix.png"

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Confusion matrix saved: {output_path}")


# ============================================================
# 5. SAVE PRECISION-RECALL CURVE
# ============================================================

def save_precision_recall_curve(
    y_true,
    probabilities,
    model_name,
):
    """
    Plot precision against recall.

    This is useful when fraud cases are rare.
    """

    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        recall,
        precision,
        linewidth=2,
    )

    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")

    axis.set_title(
        f"Precision-Recall Curve - {model_name}"
    )

    axis.grid(True, alpha=0.3)

    figure.tight_layout()

    output_path = REPORT_DIR / "precision_recall_curve.png"

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Precision-recall curve saved: {output_path}")


# ============================================================
# 6. MAIN TRAINING PIPELINE
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )

    print("=" * 70)
    print("CREDIT CARD FRAUD DETECTION")
    print("MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: CHECK DATASET
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "\nDataset not found!\n"
            f"Expected location: {DATA_PATH}\n\n"
            "Download creditcard.csv from Kaggle and place it inside:\n"
            "data/raw/creditcard.csv\n"
        )

    print("\n[1/8] Loading dataset...")

    df = load_dataset(DATA_PATH)

    if df.empty:
        raise ValueError(
            "The dataset contains no usable rows."
        )

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "Training requires both classes: 0 and 1."
        )

    print(f"Total transactions: {len(df):,}")

    fraud_count = int((df[TARGET] == 1).sum())
    legitimate_count = int((df[TARGET] == 0).sum())

    print(f"Fraud transactions: {fraud_count:,}")
    print(f"Legitimate transactions: {legitimate_count:,}")

    print(
        "Fraud percentage: "
        f"{df[TARGET].mean() * 100:.4f}%"
    )

    # --------------------------------------------------------
    # STEP 2: PREPARE FEATURES AND TARGET
    # --------------------------------------------------------

    print("\n[2/8] Preparing features...")

    X = df[EXPECTED_FEATURES].copy()
    y = df[TARGET].copy()

    if y.value_counts().min() < 3:
        raise ValueError(
            "At least three examples of each class are required "
            "for the stratified training and evaluation splits."
        )

    # --------------------------------------------------------
    # STEP 3: SPLIT DATA
    # --------------------------------------------------------

    print("\n[3/8] Splitting dataset...")

    # 15% test data.
    X_trainval, X_test, y_trainval, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.15,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    # 15% of the complete dataset for validation.
    # The remaining 70% is used for training.
    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_trainval,
            y_trainval,
            test_size=0.15 / 0.85,
            stratify=y_trainval,
            random_state=RANDOM_STATE,
        )
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Testing rows: {len(X_test):,}")

    # --------------------------------------------------------
    # STEP 4: TRAIN AND COMPARE MODELS
    # --------------------------------------------------------

    print("\n[4/8] Training machine learning models...")

    validation_results = []
    fitted_models = {}

    for model_name, model in make_models().items():

        print("\n" + "-" * 60)
        print(f"Training: {model_name}")
        print("-" * 60)

        model.fit(
            X_train,
            y_train,
        )

        validation_probabilities = model.predict_proba(
            X_val
        )[:, 1]

        validation_pr_auc = average_precision_score(
            y_val,
            validation_probabilities,
        )

        validation_roc_auc = roc_auc_score(
            y_val,
            validation_probabilities,
        )

        validation_results.append(
            {
                "model": model_name,
                "validation_pr_auc": float(
                    validation_pr_auc
                ),
                "validation_roc_auc": float(
                    validation_roc_auc
                ),
            }
        )

        fitted_models[model_name] = model

        print(
            f"Validation PR-AUC: "
            f"{validation_pr_auc:.4f}"
        )

        print(
            f"Validation ROC-AUC: "
            f"{validation_roc_auc:.4f}"
        )

    # --------------------------------------------------------
    # STEP 5: SELECT THE BEST MODEL
    # --------------------------------------------------------

    print("\n[5/8] Comparing models...")

    results_df = pd.DataFrame(
        validation_results
    ).sort_values(
        by="validation_pr_auc",
        ascending=False,
    )

    comparison_path = (
        REPORT_DIR / "model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False,
    )

    best_name = str(
        results_df.iloc[0]["model"]
    )

    best_model = fitted_models[best_name]

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    print(
        f"\nSelected model: {best_name}"
    )

    print(
        "Selection criterion: validation Average Precision (PR-AUC)"
    )

    # --------------------------------------------------------
    # STEP 6: EVALUATE ON THE HELD-OUT TEST SET
    # --------------------------------------------------------

    print("\n[6/8] Evaluating selected model...")

    # The test set is not used for model selection.
    test_probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= THRESHOLD
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )

    metrics = {
        "selected_model": best_name,
        "threshold": THRESHOLD,
        "test_rows": int(len(X_test)),
        "test_fraud_cases": int(y_test.sum()),

        "precision": float(
            precision_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),

        "f1_score": float(
            f1_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),

        "pr_auc_average_precision": float(
            average_precision_score(
                y_test,
                test_probabilities,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                test_probabilities,
            )
        ),

        "confusion_matrix": cm.tolist(),

        "classification_report": classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=[
                "Legitimate",
                "Fraud",
            ],
            output_dict=True,
            zero_division=0,
        ),
    }

    # --------------------------------------------------------
    # STEP 7: SAVE MODEL AND REPORTS
    # --------------------------------------------------------

    print("\n[7/8] Saving model and reports...")

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    metrics_path = (
        REPORT_DIR / "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    # Save confusion matrix.
    save_confusion_matrix(
        y_test,
        test_predictions,
        best_name,
    )

    # Save precision-recall curve.
    save_precision_recall_curve(
        y_test,
        test_probabilities,
        best_name,
    )

    # --------------------------------------------------------
    # STEP 8: DISPLAY FINAL RESULTS
    # --------------------------------------------------------

    print("\n[8/8] Training completed successfully!")

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Selected model : {best_name}")

    print(
        f"Test rows      : {len(X_test):,}"
    )

    print(
        f"Test frauds    : {int(y_test.sum()):,}"
    )

    print(
        f"Threshold      : {THRESHOLD:.2f}"
    )

    print(
        f"Precision      : {metrics['precision']:.4f}"
    )

    print(
        f"Recall         : {metrics['recall']:.4f}"
    )

    print(
        f"F1-score       : {metrics['f1_score']:.4f}"
    )

    print(
        f"PR-AUC         : "
        f"{metrics['pr_auc_average_precision']:.4f}"
    )

    print(
        f"ROC-AUC        : {metrics['roc_auc']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        "                 Predicted Legitimate | Predicted Fraud"
    )

    print(
        f"Actual Legitimate       {cm[0, 0]:>8,}       "
        f"{cm[0, 1]:>8,}"
    )

    print(
        f"Actual Fraud            {cm[1, 0]:>8,}       "
        f"{cm[1, 1]:>8,}"
    )

    print("\nSaved files:")
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {metrics_path}")
    print(f"Comparison: {comparison_path}")

    print(
        f"Confusion matrix: "
        f"{REPORT_DIR / 'confusion_matrix.png'}"
    )

    print(
        f"Precision-recall curve: "
        f"{REPORT_DIR / 'precision_recall_curve.png'}"
    )

    print("\nNote:")
    print(
        "This project is an educational ML prototype, "
        "not a production banking fraud detection system."
    )

    print("=" * 70)


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except FileNotFoundError as error:
        print(f"\nFILE ERROR:\n{error}")

    except ValueError as error:
        print(f"\nDATA ERROR:\n{error}")

    except KeyboardInterrupt:
        print(
            "\nTraining interrupted by the user."
        )

