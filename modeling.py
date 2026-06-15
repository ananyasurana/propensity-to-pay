"""Model training, hyperparameter tuning, and evaluation."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from config import FIGURES_DIR, RANDOM_STATE, TARGET


def build_baseline_model(preprocessor) -> Pipeline:
    """
    Logistic Regression baseline with balanced class weights.

    Chosen as baseline because it is interpretable and fast; coefficients
    give directional insight for collections stakeholders.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_tuned_model(preprocessor) -> Pipeline:
    """
    Random Forest — handles non-linearities and mixed feature types well
    after one-hot encoding. Class weights address the 74/26 imbalance.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def tune_random_forest(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """
    Hyperparameter search with stratified 5-fold CV on the training set.

    We keep the search space modest given dataset size (n=1000).
    """
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [6, 10, None],
        "classifier__min_samples_leaf": [1, 5, 10],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
    save_prefix: str = "model",
) -> dict[str, float]:
    """Compute classification metrics and save diagnostic plots."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "auc_roc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"{save_prefix}_confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"{save_prefix}_roc_curve.png", dpi=120)
    plt.close(fig)

    # Precision-Recall curve (important for imbalanced data)
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax, name=model_name)
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"{save_prefix}_pr_curve.png", dpi=120)
    plt.close(fig)

    # Classification report as text artifact
    report = classification_report(y_test, y_pred, target_names=["No Payment", "Payment"])
    report_path = FIGURES_DIR.parent / f"{save_prefix}_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    return metrics


def plot_feature_importance(
    model: Pipeline,
    feature_names: list[str],
    top_n: int = 20,
    save_prefix: str = "rf",
) -> pd.DataFrame:
    """Extract and plot Random Forest feature importances."""
    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=imp_df, y="feature", x="importance", ax=ax, palette="mako")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"{save_prefix}_feature_importance.png", dpi=120)
    plt.close(fig)
    return imp_df


def assign_risk_bands(probabilities: np.ndarray) -> np.ndarray:
    """
    Map predicted payment probabilities to actionable collections bands.

    Business interpretation:
    - High propensity (>=0.60): low-touch / automated reminders — likely to self-cure
    - Medium (0.35–0.60): standard collections workflow
    - Low (<0.35): intensive outreach, senior agent, payment-plan offers
    """
    bands = np.empty(len(probabilities), dtype=object)
    bands[probabilities >= 0.60] = "High - Prioritise light touch"
    bands[(probabilities >= 0.35) & (probabilities < 0.60)] = "Medium - Standard outreach"
    bands[probabilities < 0.35] = "Low - Intensive collections"
    return bands


def get_transformed_feature_names(preprocessor, cat_cols: list[str], num_cols: list[str]) -> list[str]:
    """Recover feature names after ColumnTransformer + OneHotEncoder."""
    feature_names = list(num_cols)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_features = cat_encoder.get_feature_names_out(cat_cols)
    feature_names.extend(cat_features.tolist())
    return feature_names
