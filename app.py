
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CREDIT CARD FRAUD DETECTION DASHBOARD
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.joblib"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(120deg, #14213d, #17365d);
        border: 1px solid #315477;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }

    .hero p {
        color: #d5e4f5;
        margin-bottom: 0;
    }

    div[data-testid="stMetric"] {
        background-color: #171c26;
        border: 1px solid #303746;
        padding: 18px;
        border-radius: 12px;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        padding-top: 25px;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL AND METRICS
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run the training script first."
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the trained model.")
    st.code(str(error))
    st.info(
        "Check that models/fraud_detection_model.joblib exists "
        "and that your Python environment has the required packages."
    )
    st.stop()

metrics = load_metrics()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ Credit Card Fraud Detection</h1>
        <p>
            Machine-learning dashboard for identifying potentially
            fraudulent transactions using an anonymized transaction dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🛡️ Fraud Detection")
st.sidebar.caption("Machine Learning Project")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Transaction Prediction",
        "About the Project",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "This application is a machine-learning demonstration. "
    "Predictions should not be treated as confirmed fraud."
)


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    st.header("📊 Model Performance")

    selected_model = metrics.get(
        "selected_model",
        "Random Forest",
    )

    if isinstance(selected_model, dict):
        selected_model = selected_model.get(
            "name", "Random Forest"
        )

    precision = metrics.get("precision", None)
    recall = metrics.get("recall", None)
    f1_score = metrics.get("f1_score", metrics.get("f1", None))
    pr_auc = metrics.get(
        "average_precision",
        metrics.get("pr_auc", None),
    )
    roc_auc = metrics.get("roc_auc", None)

    def format_metric(value):
        if value is None:
            return "N/A"

        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "N/A"

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Selected Model", str(selected_model))
    col2.metric("Precision", format_metric(precision))
    col3.metric("Recall", format_metric(recall))
    col4.metric("PR-AUC", format_metric(pr_auc))

    col5, col6 = st.columns(2)
    col5.metric("F1 Score", format_metric(f1_score))
    col6.metric("ROC-AUC", format_metric(roc_auc))

    st.caption(
        "Metrics are taken from your saved evaluation report. "
        "If available, they represent performance on the held-out test set."
    )

    st.divider()

    st.subheader("📁 Saved Evaluation Reports")

    report_files = [
        (
            "Confusion Matrix",
            BASE_DIR / "reports" / "confusion_matrix.png",
        ),
        (
            "Precision-Recall Curve",
            BASE_DIR / "reports" / "precision_recall_curve.png",
        ),
    ]

    available_reports = [
        (name, path)
        for name, path in report_files
        if path.exists()
    ]

    if available_reports:
        columns = st.columns(len(available_reports))

        for column, (name, path) in zip(
            columns, available_reports
        ):
            with column:
                st.markdown(f"**{name}**")
                st.image(str(path), use_container_width=True)
    else:
        st.info(
            "Evaluation plots were not found. "
            "Run the training script to generate them."
        )

    comparison_path = (
        BASE_DIR / "reports" / "model_comparison.csv"
    )

    if comparison_path.exists():
        st.subheader("🤖 Model Comparison")

        try:
            comparison = pd.read_csv(comparison_path)
            st.dataframe(
                comparison,
                width="stretch",
                hide_index=True,
            )
        except Exception as error:
            st.warning(
                f"Could not load the model comparison: {error}"
            )


# ============================================================
# TRANSACTION PREDICTION PAGE
# ============================================================

elif page == "Transaction Prediction":

    st.header("🔍 Transaction Prediction")

    st.write(
        "Upload a CSV file containing the transaction features "
        "required by your trained model."
    )

    st.info(
        "Required columns: Time, V1 through V28, and Amount. "
        "The Class column is optional and is not used to generate predictions."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        help="Upload a CSV containing the required transaction features.",
    )

    threshold = st.slider(
        "Fraud probability threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        help=(
            "Transactions with predicted fraud probabilities "
            "at or above this threshold are flagged."
        ),
    )

    if uploaded_file is not None:

        try:
            input_df = pd.read_csv(uploaded_file)

            if input_df.empty:
                st.error("The uploaded CSV contains no transactions.")
                st.stop()

            missing_columns = [
                column
                for column in FEATURES
                if column not in input_df.columns
            ]

            if missing_columns:
                st.error(
                    "Your CSV is missing required columns:"
                )
                st.code(", ".join(missing_columns))
                st.stop()

            # Keep only model features, in the correct order.
            model_input = input_df[FEATURES].copy()

            # Convert feature values to numbers.
            for column in FEATURES:
                model_input[column] = pd.to_numeric(
                    model_input[column],
                    errors="coerce",
                )

            if model_input.isna().any().any():
                bad_columns = model_input.columns[
                    model_input.isna().any()
                ].tolist()

                st.error(
                    "Missing or invalid numeric values were found "
                    "in these columns:"
                )
                st.code(", ".join(bad_columns))
                st.stop()

            if not np.isfinite(
                model_input.to_numpy(dtype=float)
            ).all():
                st.error(
                    "The uploaded CSV contains infinite or invalid numeric values."
                )
                st.stop()

            if (model_input["Amount"] < 0).any():
                st.error(
                    "Transaction amounts must be zero or greater."
                )
                st.stop()

            if (model_input["Time"] < 0).any():
                st.error(
                    "Time values must be zero or greater."
                )
                st.stop()

            # Generate fraud probabilities.
            probabilities = model.predict_proba(
                model_input
            )[:, 1]

            predictions = (
                probabilities >= threshold
            ).astype(int)

            # Build results without modifying the uploaded input.
            results = input_df.copy()

            results["fraud_probability"] = probabilities
            results["predicted_class"] = predictions

            results["risk_level"] = np.select(
                [
                    probabilities >= 0.80,
                    probabilities >= 0.50,
                    probabilities >= 0.20,
                ],
                [
                    "High",
                    "Medium",
                    "Low",
                ],
                default="Very Low",
            )

            results["prediction_label"] = np.where(
                predictions == 1,
                "Potential Fraud",
                "Likely Legitimate",
            )

            total_transactions = len(results)
            fraud_count = int(predictions.sum())
            legitimate_count = (
                total_transactions - fraud_count
            )
            fraud_rate = (
                fraud_count / total_transactions * 100
            )

            st.success(
                f"Successfully scored {total_transactions:,} transactions."
            )

            # ------------------------------------------------
            # KEY PERFORMANCE INDICATORS
            # ------------------------------------------------

            st.subheader("📌 Prediction Overview")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Transactions Scored",
                f"{total_transactions:,}",
            )

            c2.metric(
                "Potential Fraud Flagged",
                f"{fraud_count:,}",
            )

            c3.metric(
                "Likely Legitimate",
                f"{legitimate_count:,}",
            )

            c4.metric(
                "Flag Rate",
                f"{fraud_rate:.2f}%",
            )

            st.caption(
                f"Current classification threshold: {threshold:.2f}"
            )

            st.divider()

            # ------------------------------------------------
            # CHARTS
            # ------------------------------------------------

            st.subheader("📈 Prediction Analytics")

            chart1, chart2 = st.columns(2)

            with chart1:
                st.markdown("**Transaction Classification**")

                classification_data = pd.DataFrame(
                    {
                        "Classification": [
                            "Potential Fraud",
                            "Likely Legitimate",
                        ],
                        "Transactions": [
                            fraud_count,
                            legitimate_count,
                        ],
                    }
                ).set_index("Classification")

                st.bar_chart(classification_data)

            with chart2:
                st.markdown("**Predicted Risk Levels**")

                risk_order = [
                    "High",
                    "Medium",
                    "Low",
                    "Very Low",
                ]

                risk_counts = (
                    results["risk_level"]
                    .value_counts()
                    .reindex(risk_order, fill_value=0)
                )

                st.bar_chart(
                    risk_counts.rename("Transactions")
                )

            st.divider()

            # ------------------------------------------------
            # FILTERS
            # ------------------------------------------------

            st.subheader("🧾 Transaction Results")

            filter_col1, filter_col2 = st.columns(2)

            with filter_col1:
                classification_filter = st.selectbox(
                    "Filter by classification",
                    [
                        "All Transactions",
                        "Potential Fraud",
                        "Likely Legitimate",
                    ],
                )

            with filter_col2:
                risk_filter = st.selectbox(
                    "Filter by risk level",
                    [
                        "All Risk Levels",
                        "High",
                        "Medium",
                        "Low",
                        "Very Low",
                    ],
                )

            filtered_results = results.copy()

            if classification_filter == "Potential Fraud":
                filtered_results = filtered_results[
                    filtered_results["predicted_class"] == 1
                ]

            elif classification_filter == "Likely Legitimate":
                filtered_results = filtered_results[
                    filtered_results["predicted_class"] == 0
                ]

            if risk_filter != "All Risk Levels":
                filtered_results = filtered_results[
                    filtered_results["risk_level"] == risk_filter
                ]

            st.write(
                f"Showing {len(filtered_results):,} "
                f"of {total_transactions:,} transactions."
            )

            st.dataframe(
                filtered_results,
                width="stretch",
                hide_index=True,
            )

            # ------------------------------------------------
            # DOWNLOAD RESULTS
            # ------------------------------------------------

            st.subheader("📥 Download Reports")

            download_col1, download_col2 = st.columns(2)

            with download_col1:
                all_csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download All Predictions",
                    data=all_csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with download_col2:
                fraud_results = results[
                    results["predicted_class"] == 1
                ]

                fraud_csv = fraud_results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Flagged Transactions",
                    data=fraud_csv,
                    file_name="flagged_fraud_transactions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.warning(
                "A flagged transaction is a model prediction, not proof "
                "of fraud. Review suspicious transactions before taking action."
            )

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")

        except pd.errors.ParserError:
            st.error(
                "Unable to read the CSV. Check its format and try again."
            )

        except Exception as error:
            st.error("Prediction failed.")
            st.exception(error)

    else:
        st.info(
            "Upload a CSV file to begin transaction prediction."
        )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About the Project":

    st.header("ℹ️ About This Project")

    st.markdown(
        """
        ### Project Overview

        Credit card fraud detection is a binary classification problem.
        The model estimates whether a transaction resembles a fraudulent
        transaction based on its input features.

        ### Technologies

        - Python
        - Pandas and NumPy
        - Scikit-learn
        - Joblib
        - Streamlit
        - Matplotlib-generated evaluation plots

        ### Machine Learning Workflow

        1. Load and validate the dataset.
        2. Prepare the features and target.
        3. Split the data into training, validation, and test sets.
        4. Train and compare machine-learning models.
        5. Select a model using validation Average Precision.
        6. Evaluate the selected model on held-out test data.
        7. Save the trained model and evaluation reports.
        8. Use the dashboard to score new CSV transactions.

        ### Important Limitations

        - Predictions depend on the quality and distribution of the training data.
        - A probability score is not proof that a transaction is fraudulent.
        - The default threshold may not be suitable for every application.
        - This project is a demonstration, not a production banking security system.
        - Real deployments require additional validation, security, monitoring,
          and appropriate handling of sensitive data.
        """
    )

    st.subheader("📂 Project Files")

    files = [
        "app.py",
        "src/train.py",
        "src/predict.py",
        "src/data_utils.py",
        "models/fraud_detection_model.joblib",
        "reports/metrics.json",
        "reports/model_comparison.csv",
        "reports/confusion_matrix.png",
        "reports/precision_recall_curve.png",
    ]

    for file_name in files:
        path = BASE_DIR / file_name

        if path.exists():
            st.success(f"✓ {file_name}")
        else:
            st.warning(f"Not found: {file_name}")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Credit Card Fraud Detection | Machine Learning Project<br>
        Predictions are estimates and must not be treated as confirmed fraud.
    </div>
    """,
    unsafe_allow_html=True,
)

