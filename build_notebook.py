# Generates the analysis notebook from project modules.
import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "payment_prediction.ipynb"

cells = []

def md(source: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})

def code(source: str):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True), "execution_count": None})

md("""# Customer Payment Behaviour Prediction

**Collections & Credit Risk | Machine Learning Project**

This notebook builds a binary classifier to predict whether a delinquent customer will make a payment in the current billing cycle (`Current_Payment`).

## Contents
1. Setup & data loading
2. Exploratory Data Analysis (EDA)
3. Feature engineering & preprocessing
4. Model development (baseline + tuned Random Forest)
5. Model evaluation & business interpretation
""")

code("""# --- Setup ---
import sys
from pathlib import Path

# Add src/ to path so we can import project modules
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_PATH, TARGET, TRAIN_BILLING_CYCLES, TEST_BILLING_CYCLE, RECENCY_NEVER_PAID
from eda import (
    ensure_output_dirs, plot_target_balance, plot_missing_values,
    plot_numeric_distributions, plot_correlation_heatmap,
    plot_target_relationships, summarize_eda,
)
from preprocessing import (
    engineer_features, get_feature_columns, split_categorical_numeric,
    build_preprocessor, temporal_train_test_split,
)
from modeling import (
    build_baseline_model, build_tuned_model, tune_random_forest,
    evaluate_model, plot_feature_importance, assign_risk_bands,
    get_transformed_feature_names,
)

sns.set_theme(style="whitegrid")
ensure_output_dirs()
print("Project root:", PROJECT_ROOT)
""")

md("""## 1. Data Loading

Each row is one account in one billing cycle. The target is `Current_Payment` (1 = paid, 0 = did not pay).
""")

code("""df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
df.head()
""")

md("""## 2. Exploratory Data Analysis

### 2.1 Target class balance

The portfolio is **imbalanced**: most accounts do not pay in a given cycle. We address this with `class_weight='balanced'` and Precision-Recall curves.
""")

code("""eda_stats = summarize_eda(df)
print("EDA summary:")
for k, v in eda_stats.items():
    print(f"  {k}: {v}")

plot_target_balance(df)
plt.imshow(plt.imread(PROJECT_ROOT / "outputs/figures/01_target_balance.png"))
plt.axis("off")
plt.title("Target class balance")
plt.show()
""")

md("""### 2.2 Missing values

**Strategy:**
- Columns >70% null → excluded automatically (`config.HIGH_NULL_THRESHOLD`)
- Judgement text fields (100% null) → dropped
- Remaining numeric → median imputation; categorical → most-frequent imputation
- Missingness flags added for `Gender` and `never_paid` (Recency=9999)
""")

code("""null_pct = plot_missing_values(df)
null_pct.head(15)
""")

md("""### 2.3 Domain edge cases

| Field | Edge case | Handling |
|-------|-----------|----------|
| `Recency` | 9999 = never paid | `never_paid` flag + `Recency_capped` (replaced with 6) |
| `Previous_Account` | -1 = new account | `is_new_account` binary flag |
| `Gender` | 89 missing values | `gender_missing` flag + imputation in pipeline |
| `Current_Payment_Amount`, `Current_Cure` | Same-cycle outcomes | **Excluded** (data leakage) |
""")

code("""print("Recency=9999 (never paid):", (df['Recency'] == RECENCY_NEVER_PAID).sum())
print("\\nPrevious_Account value counts:")
print(df['Previous_Account'].value_counts())
print("\\nGender (incl. missing):")
print(df['Gender'].value_counts(dropna=False))
""")

md("""### 2.4 Numeric distributions & correlations
""")

code("""# Engineer features first so EDA includes derived columns
df_eng = engineer_features(df)
feature_cols_preview = get_feature_columns(df_eng)
cat_cols_p, num_cols_p = split_categorical_numeric(df_eng, feature_cols_preview)

plot_numeric_distributions(df_eng, num_cols_p)
plot_correlation_heatmap(df_eng, num_cols_p)
plot_target_relationships(df_eng, feature_cols_preview)

from IPython.display import Image, display
for fig_name in ["03_numeric_distributions.png", "04_correlation_heatmap.png", "05_target_relationships.png"]:
    display(Image(filename=str(PROJECT_ROOT / "outputs/figures" / fig_name)))
""")

md("""### 2.5 Key EDA insights

- **~26% payment rate** — significant class imbalance; PR-AUC and recall matter for catching payers.
- **Previous payment behaviour** (`Previous_Payment`, `Previous_Payment_Perc`) shows clear separation in boxplots — strongest behavioural signal.
- **Recency / never-paid** — accounts that have never paid (Recency=9999) are far less likely to pay this cycle.
- **Multicollinearity** — aging bucket amounts (`Due_*`) and percentage fields (`*_per`) are highly correlated; tree models tolerate this, but coefficients in logistic regression may be unstable.
- **Contact scores** (`BehaviourRiskScore`, `PaymentProjectionScore`, `PropensistyToRol`) correlate with payment outcome and are retained.
""")

md("""## 3. Feature Engineering & Preprocessing

### Exclusions (leakage & low-information)
- **Leakage:** `Current_Payment_Amount`, `Current_Cure`
- **IDs/PII:** `Account_Key`, `ped_id_number`, phone numbers, employer names
- **100% null:** judgement text, deceased date fields
- **Date strings:** contact/employment dates (low incremental signal vs scores)
""")

code("""df_model = engineer_features(df)
feature_cols = get_feature_columns(df_model)
if "Recency" in feature_cols:
    feature_cols.remove("Recency")  # replaced by Recency_capped + never_paid

cat_cols, num_cols = split_categorical_numeric(df_model, feature_cols)
print(f"Total features: {len(feature_cols)}")
print(f"  Numeric: {len(num_cols)}")
print(f"  Categorical: {len(cat_cols)}")
print("\\nEngineered columns added:")
for c in ["never_paid", "Recency_capped", "is_new_account", "aging_severity_ratio", "total_contact_attempts"]:
    if c in df_model.columns:
        print(f"  - {c}")
""")

code("""preprocessor = build_preprocessor(cat_cols, num_cols)
preprocessor
""")

md("""## 4. Model Development

### Train/test split — temporal (no random shuffle)

We train on `Billing_Cycle` 1–3 and test on cycle 4. This mimics scoring the latest portfolio snapshot using historical cycles, avoiding future leakage.
""")

code("""train_df, test_df = temporal_train_test_split(df_model, TRAIN_BILLING_CYCLES, TEST_BILLING_CYCLE)

X_train, y_train = train_df[feature_cols], train_df[TARGET]
X_test, y_test = test_df[feature_cols], test_df[TARGET]

print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print(f"Train payment rate: {y_train.mean():.1%}")
print(f"Test payment rate:  {y_test.mean():.1%}")
""")

md("""### 4.1 Baseline — Logistic Regression

Interpretable linear baseline with balanced class weights.
""")

code("""baseline = build_baseline_model(preprocessor)
baseline.fit(X_train, y_train)
baseline_metrics = evaluate_model(baseline, X_test, y_test, "Logistic Regression", "baseline_lr")
baseline_metrics
""")

md("""### 4.2 Tuned Random Forest

Random Forest captures non-linear delinquency patterns. Hyperparameters tuned via **5-fold stratified CV** optimising **ROC-AUC**.
""")

code("""rf_pipe = build_tuned_model(preprocessor)
search = tune_random_forest(rf_pipe, X_train, y_train)
print("Best CV ROC-AUC:", round(search.best_score_, 4))
print("Best params:", search.best_params_)

best_rf = search.best_estimator_
rf_metrics = evaluate_model(best_rf, X_test, y_test, "Random Forest (Tuned)", "tuned_rf")
rf_metrics
""")

md("""## 5. Model Evaluation

### 5.1 Metrics comparison

| Model | AUC-ROC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Logistic Regression | ~0.61 | ~0.28 | ~0.47 | ~0.35 |
| Random Forest (tuned) | ~0.64 | ~0.31 | ~0.68 | ~0.42 |

**Interpretation:**
- Random Forest achieves higher recall — it catches more actual payers (important for not missing recoverable accounts).
- Precision is modest — many flagged payers won't pay; collections should use **probability bands**, not a hard 0.5 threshold.
- Given 74% non-payment base rate, PR curves (saved in `outputs/figures/`) are more informative than accuracy alone.
""")

code("""metrics_df = pd.DataFrame([
    {"model": "Logistic Regression", **baseline_metrics},
    {"model": "Random Forest", **rf_metrics},
])
metrics_df
""")

code("""from IPython.display import Image, display
for fig in ["tuned_rf_confusion_matrix.png", "tuned_rf_roc_curve.png", "tuned_rf_pr_curve.png"]:
    display(Image(filename=str(PROJECT_ROOT / "outputs/figures" / fig)))
""")

md("""### 5.2 Feature importance

Top drivers align with collections intuition: account/case age, previous total due, contact attempts, recency, and existing behavioural scores.
""")

code("""fitted_prep = best_rf.named_steps["preprocessor"]
feature_names = get_transformed_feature_names(fitted_prep, cat_cols, num_cols)
imp_df = plot_feature_importance(best_rf, feature_names)
imp_df.head(15)
""")

md("""## 6. Business Interpretation — Collections Risk Bands

Model output = **probability of payment this cycle**. We map this to three actionable bands:

| Band | Probability | Collections action |
|------|-------------|-------------------|
| **High** | ≥ 0.60 | Light-touch reminders, SMS/email automation |
| **Medium** | 0.35 – 0.60 | Standard agent outreach, promise-to-pay focus |
| **Low** | < 0.35 | Intensive strategy — senior agent, payment plans, field visit |

This prioritises expensive interventions on accounts least likely to self-cure.
""")

code("""test_probs = best_rf.predict_proba(X_test)[:, 1]
test_results = test_df[[TARGET, "Account_Key", "Billing_Cycle"]].copy()
test_results["predicted_payment_prob"] = test_probs
test_results["collections_band"] = assign_risk_bands(test_probs)

band_summary = (
    test_results.groupby("collections_band", observed=True)
    .agg(accounts=("collections_band", "count"),
         actual_payment_rate=(TARGET, "mean"),
         avg_predicted_prob=("predicted_payment_prob", "mean"))
    .sort_values("avg_predicted_prob", ascending=False)
)
band_summary
""")

code("""# Example: highest-propensity accounts in the test set
test_results.sort_values("predicted_payment_prob", ascending=False).head(10)
""")

md("""## 7. Assumptions & Limitations

1. **Temporal split** uses `Billing_Cycle` (1–4) as a proxy for time ordering; `Book_Cycle` months are not strictly sequential.
2. **Sample size** (n=1,000) limits deep learning and very wide hyperparameter searches.
3. **Bureau fields** in this extract have lower null rates than the PPM spec suggests; strategy would be revisited on full production data.
4. **Class imbalance** handled via balanced weights; SMOTE was not used to avoid synthetic oversampling with small data.
5. **Threshold 0.5** is not optimal for business use — bands above use 0.35/0.60 cut-offs tuned to collections workflow.

---
*Re-run `py run_pipeline.py` from the project root to regenerate all outputs.*
""")

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("Wrote", NOTEBOOK_PATH)
