# Predicting Loan Payback

MSBA 265 Final Project — Ziyan Chen
Kaggle Playground Series S5E11: https://www.kaggle.com/competitions/playground-series-s5e11

## Goal
Identify which borrower and loan characteristics are related to loan repayment, and build a classification model to predict the probability that a borrower pays back the loan. Submissions are scored with ROC AUC.

## Setup
1. Clone this repository and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # Mac
   pip install -r requirements.txt
   ```
2. The raw data (`train.csv`, `test.csv`, `sample_submission.csv`) from the
   [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s5e11/data)
   is included in `data/`. Running `python src/eda_audit.py` regenerates
   `train_clean.csv` and `test_clean.csv`.

## How to run
- **Full audit and cleaning pipeline:** `python src/eda_audit.py`
  - Writes audit tables and figures to `reports/`
  - Writes `train_clean.csv` and `test_clean.csv` to `data/`
- **Interactive EDA:** open `01_eda.ipynb` and run all cells

## Project structure
```
data/              raw and cleaned data
reports/           audit tables and figures
src/eda_audit.py   reproducible audit and cleaning pipeline
01_eda.ipynb       exploratory data analysis with interpretations
requirements.txt   Python dependencies
```

## Key EDA findings
- 79.9% of loans were repaid, so the classes are imbalanced; ROC AUC is used instead of accuracy.
- Employment status is the strongest signal: only 7.8% of unemployed borrowers repaid, versus 89.4% of employed.
- Payback drops sharply in the top two DTI deciles (from about 78% to 49%), so DTI risk is nonlinear.
- Credit score and loan grade are highly correlated (r = −0.91), so logistic regression will keep only one of them.
- Outliers were kept: high-DTI outliers are real high-risk borrowers (45% payback vs. 81% for others).

## Next steps
Baseline → logistic regression → LightGBM / XGBoost, evaluated with stratified 5-fold cross-validation.
