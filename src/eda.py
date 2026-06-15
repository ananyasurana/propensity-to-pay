"""Exploratory data analysis helpers and plot generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR, RECENCY_NEVER_PAID, TARGET


def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_target_balance(df: pd.DataFrame, save: bool = True) -> None:
    """Class balance bar chart for Current_Payment."""
    counts = df[TARGET].value_counts().sort_index()
    pct = df[TARGET].value_counts(normalize=True).sort_index() * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(counts.index.astype(str), counts.values, color="steelblue")
    axes[0].set_title("Target Class Counts")
    axes[0].set_xlabel("Current_Payment (0=No, 1=Yes)")
    axes[0].set_ylabel("Count")

    axes[1].bar(pct.index.astype(str), pct.values, color="coral")
    axes[1].set_title("Target Class Proportions (%)")
    axes[1].set_xlabel("Current_Payment")
    axes[1].set_ylabel("Percentage")

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "01_target_balance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_missing_values(df: pd.DataFrame, top_n: int = 25, save: bool = True) -> pd.Series:
    """Horizontal bar chart of columns with highest missing-value rates."""
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    plot_data = null_pct.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_data.plot(kind="barh", ax=ax, color="steelblue", legend=False)
    ax.set_xlabel("Missing (%)")
    ax.set_title(f"Top {top_n} Features by Missing-Value Rate")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "02_missing_values.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return null_pct


def plot_numeric_distributions(
    df: pd.DataFrame, numeric_cols: list[str], save: bool = True
) -> None:
    """Histogram grid for all numeric features (sampled if very large)."""
    cols = numeric_cols[:30]  # cap for readability in a single figure
    n = len(cols)
    if n == 0:
        return

    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.5))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numeric Feature Distributions", y=1.01, fontsize=12)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "03_numeric_distributions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(
    df: pd.DataFrame, numeric_cols: list[str], save: bool = True
) -> pd.DataFrame:
    """Correlation matrix for numeric features (subset for readability)."""
    # Select a focused set of business-relevant numerics to avoid an unreadable 100+ matrix
    focus = [
        c for c in numeric_cols
        if any(
            kw in c
            for kw in [
                "Balance", "Total_Due", "BAR", "Instalment", "Due_", "Recency",
                "Previous_", "PAttempts", "Propensisty", "Behaviour", "PaymentProjection",
                "Contact_Score", "Credit_Risk", "Age", "Deliquency", "DC",
                "Current_Payment", "never_paid", "aging_severity",
            ]
        )
    ]
    focus = list(dict.fromkeys(focus))  # dedupe preserving order
    if TARGET not in focus and TARGET in df.columns:
        focus.append(TARGET)

    corr = df[focus].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=False, cmap="RdBu_r", center=0,
        square=True, linewidths=0.3, ax=ax,
    )
    ax.set_title("Correlation Matrix — Key Numeric Features")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "04_correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return corr


def plot_target_relationships(
    df: pd.DataFrame, feature_cols: list[str], save: bool = True
) -> None:
    """Box/violin plots for top numeric features vs target."""
    key_features = [
        c for c in [
            "Recency_capped", "Previous_Payment_Perc", "Total_Due", "Deliquency",
            "BehaviourRiskScore", "PaymentProjectionScore", "PropensistyToRol",
            "PAttempts", "Previous_Payment", "never_paid",
        ]
        if c in df.columns
    ]

    n = len(key_features)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(key_features):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, ax=axes[i], palette="Set2", legend=False)
        axes[i].set_title(f"{col} by Payment Outcome")
        axes[i].set_xlabel("Current_Payment")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Target Class", y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "05_target_relationships.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def summarize_eda(df: pd.DataFrame) -> dict:
    """Return a dict of key EDA statistics for the report."""
    target_rate = df[TARGET].mean()
    never_paid = (df["Recency"] == RECENCY_NEVER_PAID).sum() if "Recency" in df.columns else 0
    new_accounts = (df["Previous_Account"] == -1).sum() if "Previous_Account" in df.columns else 0

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "payment_rate": round(target_rate, 4),
        "non_payment_rate": round(1 - target_rate, 4),
        "never_paid_accounts": int(never_paid),
        "new_accounts_prior_cycle": int(new_accounts),
        "gender_missing": int(df["Gender"].isna().sum()) if "Gender" in df.columns else 0,
    }
