# Credit Card Fraud Detection (Machine Learning)

An educational end-to-end ML project using the Kaggle ULB credit-card fraud dataset. It includes data validation, class-imbalance-aware models, model comparison, test metrics, plots, CSV scoring, and a Streamlit dashboard.

## Dataset

Download `creditcard.csv` from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

Place the CSV at `data/raw/creditcard.csv`. Do not commit the dataset to GitHub. It is large and has its own usage terms. The dataset normally contains 284,807 transactions and the target column `Class` (`0` = legitimate, `1` = fraud). `V1`–`V28` are anonymized PCA-transformed features.

## Windows CMD setup

```bat
cd %USERPROFILE%\Desktop
mkdir credit_card_fraud_detection
cd credit_card_fraud_detection
```

If you downloaded the project ZIP, extract it first and open CMD in the extracted project folder.

Create and activate a virtual environment:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the dataset from Kaggle, extract `creditcard.csv`, and put it here:

```text
data\raw\creditcard.csv
```

Train models and create reports:

```bat
python -m src.train
```

Start the dashboard:

```bat
streamlit run app.py
```

Score another CSV (must contain Time, V1–V28, Amount):

```bat
python -m src.predict --input data\raw\new_transactions.csv --output reports\predictions.csv --threshold 0.50
```

## Project structure

```text
credit_card_fraud_detection/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/
│       └── creditcard.csv          # Download manually; not included
├── models/
│   └── fraud_detection_model.joblib # Created after training
├── reports/
│   ├── metrics.json                # Created after training
│   ├── model_comparison.csv         # Created after training
│   ├── confusion_matrix.png         # Created after training
│   └── precision_recall_curve.png   # Created after training
└── src/
    ├── __init__.py
    ├── data_utils.py
    ├── train.py
    └── predict.py
```

## What the pipeline does

1. Checks required columns, numeric values, and target labels.
2. Creates stratified train, validation, and test splits.
3. Trains Logistic Regression and Random Forest using class weights.
4. Selects a model using validation Average Precision / PR-AUC.
5. Evaluates the selected model once on the held-out test set.
6. Saves the trained model, metrics, model comparison, confusion matrix, and precision-recall curve.
7. Uses the trained model in a Streamlit dashboard or to score a CSV.

## Why not accuracy alone?

Fraud cases are extremely rare in this dataset. A model can predict nearly every transaction as legitimate and still show very high accuracy. Review precision, recall, F1-score, Average Precision (PR-AUC), ROC-AUC, and the confusion matrix. The probability threshold is a business decision and should be tuned on validation data based on the cost of false alarms versus missed fraud.

## Important limitations

- The features are anonymized and the data is historical; it is not a live banking feed.
- This is a learning project, not a production fraud-prevention service.
- Predictions are risk indicators and require appropriate review.
- Do not upload real customer card data or sensitive financial information into this demo.
