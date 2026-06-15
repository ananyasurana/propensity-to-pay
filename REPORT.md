# Customer Payment Behaviour Prediction — Project Summary

**Author:** ML Intern Project  
**Domain:** Collections & Credit Risk  
**Target:** `Current_Payment` — binary flag for payment in the current billing cycle

---

## 1. Objective & Approach

We built a binary classification model to predict whether delinquent customers will make a payment in the current billing cycle. The model uses account balances, delinquency aging, prior-cycle payment behaviour, contact history, behavioural scores, demographics, and credit bureau attributes — while strictly excluding same-cycle outcome fields (`Current_Payment_Amount`, `Current_Cure`) to prevent data leakage.

The workflow comprised exploratory analysis, domain-aware feature engineering, temporal train/test splitting, a logistic regression baseline, and a tuned Random Forest classifier with stratified cross-validation.

---

## 2. Data Overview

| Metric | Value |
|--------|-------|
| Records | 1,000 (one row per account per billing cycle) |
| Features (raw) | 146 |
| Payment rate | 25.9% (imbalanced — 74.1% non-payment) |
| Never-paid accounts (`Recency=9999`) | 247 (24.7%) |
| New accounts (`Previous_Account=-1`) | 255 (25.5%) |
| Missing gender | 89 (8.9%) |

**Billing cycles:** Four working periods (`Billing_Cycle` 1–4) across multiple book portfolios (`Book_key`).

---

## 3. Key EDA Findings

1. **Class imbalance** — Only ~26% of accounts pay in a given cycle. Accuracy is misleading; AUC-ROC, Precision-Recall, and recall for the positive class are primary metrics.

2. **Previous payment behaviour is the strongest signal** — `Previous_Payment` and `Previous_Payment_Perc` show the clearest separation between payers and non-payers.

3. **Recency edge case** — 9999 indicates the account has never paid. We created a `never_paid` binary flag and capped numeric recency at 6 rather than treating 9999 as a continuous value.

4. **Missing values** — Judgement text fields are 100% null and were dropped. Contact phone numbers and employer names (high-cardinality PII) were excluded. Remaining bureau fields (credit score, EagleEye flags, income) were retained with median/mode imputation. Columns exceeding 70% null are auto-excluded via configuration.

5. **Multicollinearity** — Aging bucket amounts and their percentage counterparts are highly correlated. Tree-based models handle this; it may inflate coefficient variance in logistic regression.

---

## 4. Feature Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| Exclude leakage columns | `Current_Payment_Amount` and `Current_Cure` are contemporaneous with the target |
| `never_paid` flag | Captures never-paid accounts without distorting recency scale |
| `is_new_account` flag | Separates -1 (no prior cycle) from 0 (existed, no payment) |
| `aging_severity_ratio` | Summarises delinquency depth as max bucket / total due |
| `total_contact_attempts` | Aggregates SMS, email, letter, and attempt counts |
| Drop date/PII columns | Contact dates and phone numbers add noise and privacy risk |
| One-hot encode categoricals | Book type, marital status, credit score category |
| Standardise numerics | Required for logistic regression; harmless for Random Forest |

Final feature set: **115 columns** (66 numeric, 49 categorical) after exclusions and engineering.

---

## 5. Modelling Strategy

**Train/test split:** Temporal — train on `Billing_Cycle` ∈ {1, 2, 3} (743 rows), test on cycle 4 (257 rows). No random shuffling, to simulate scoring the latest portfolio.

**Class imbalance:** `class_weight='balanced'` in both models. SMOTE was not applied given the small sample size.

**Baseline:** Logistic Regression — interpretable, fast, provides coefficient directionality.

**Production candidate:** Random Forest with GridSearchCV (5-fold stratified, optimising ROC-AUC).

Best hyperparameters: `n_estimators=200`, `max_depth=None`, `min_samples_leaf=5`.

---

## 6. Results

| Model | AUC-ROC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Logistic Regression | 0.612 | 0.280 | 0.467 | 0.350 |
| Random Forest (tuned) | 0.639 | 0.306 | 0.683 | 0.423 |

**Top feature importances (Random Forest):** Case age, previous total due, account age, contact attempts (`PAttempts`), existing account status, never-paid flag, recency, propensity-to-roll score, and contact score.

The tuned Random Forest improves recall substantially (68% of actual payers identified vs 47% for logistic regression), at the cost of moderate precision. For collections, **missing a likely payer is more costly than over-contacting a non-payer**, making recall-oriented tuning appropriate.

---

## 7. Business Recommendations

Model output should be deployed as a **payment probability score**, segmented into three collections bands:

| Band | Probability threshold | Recommended action |
|------|----------------------|-------------------|
| High propensity | ≥ 0.60 | Automated reminders, minimal agent cost |
| Medium | 0.35 – 0.60 | Standard outreach, promise-to-pay negotiation |
| Low propensity | < 0.35 | Intensive collections — senior agents, payment plans |

On the held-out test set, the **low band (65 accounts) had 0% actual payment rate**, while the medium band achieved ~33% — validating discriminative power at the extremes.

**Next steps for production:**
1. Retrain on the full portfolio with strictly chronological validation (by `Book_Cycle` month).
2. Calibrate probabilities (Platt scaling or isotonic regression) for reliable band cut-offs.
3. A/B test band-based routing against the current manual prioritisation.
4. Monitor drift in payment rate and feature distributions monthly.

---

## 8. Assumptions

- `Billing_Cycle` order (1→4) is a valid temporal proxy.
- Behavioural scores (`PropensistyToRol`, `PaymentProjectionScore`) are computed from prior-cycle data only (if not, they would leak — this should be verified with the PPM data definition).
- EagleEye and credit bureau attributes reflect the state at cycle close, not after payment.
- The 1,000-row sample is representative of the broader portfolio.

---

*Code: `notebooks/payment_prediction.ipynb` and `run_pipeline.py` | Outputs: `outputs/figures/`*
