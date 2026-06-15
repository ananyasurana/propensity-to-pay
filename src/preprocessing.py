"""Feature engineering and preprocessing pipeline."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    DROP_COLUMNS,
    HIGH_NULL_THRESHOLD,
    ID_AND_PII_COLUMNS,
    LEAKAGE_COLUMNS,
    RECENCY_NEVER_PAID,
    TARGET,
)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model-ready feature column names after exclusions."""
    exclude = set(
        [TARGET]
        + LEAKAGE_COLUMNS
        + ID_AND_PII_COLUMNS
        + DROP_COLUMNS
    )
    # Drop columns exceeding the high-null threshold (except those already listed)
    null_frac = df.isnull().mean()
    high_null = null_frac[null_frac > HIGH_NULL_THRESHOLD].index.tolist()
    exclude.update(high_null)

    return [c for c in df.columns if c not in exclude]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features and handle domain-specific edge cases.

    Decisions documented:
    - Recency 9999 → binary 'never_paid' flag; numeric recency capped at 6 for modelling
    - Previous_Account -1 → 'is_new_account' flag (account did not exist prior cycle)
    - Existing_Account -1 → same treatment for opening-cycle existence
    - Gender missing → separate 'gender_missing' flag; imputed as mode later
    """
    out = df.copy()

    # --- Recency: never-paid sentinel (9999) ---
    out["never_paid"] = (out["Recency"] == RECENCY_NEVER_PAID).astype(int)
    out["Recency_capped"] = out["Recency"].replace(RECENCY_NEVER_PAID, 6)

    # --- New account flags (value -1 means no prior-cycle record) ---
    if "Previous_Account" in out.columns:
        out["is_new_account"] = (out["Previous_Account"] == -1).astype(int)
    if "Existing_Account" in out.columns:
        out["was_existing_account"] = (out["Existing_Account"] == 1).astype(int)

    # --- Delinquency concentration: share of total due in oldest bucket ---
    aging_cols = [
        "Due_30days", "Due_60days", "Due_90days", "Due_120days",
        "Due_150days", "Due_180days", "Due_210days",
    ]
    present = [c for c in aging_cols if c in out.columns]
    if present and "Total_Due" in out.columns:
        out["max_aging_bucket"] = out[present].max(axis=1)
        out["aging_severity_ratio"] = np.where(
            out["Total_Due"] > 0,
            out["max_aging_bucket"] / out["Total_Due"],
            0,
        )

    # --- Contact intensity: total outreach attempts across channels ---
    contact_cols = ["PSMSs", "PEmails", "PLetters", "PAttempts"]
    present_contact = [c for c in contact_cols if c in out.columns]
    if present_contact:
        out["total_contact_attempts"] = out[present_contact].sum(axis=1)

    # --- Payment momentum: did they pay last cycle AND what fraction ---
    if "Previous_Payment" in out.columns and "Previous_Payment_Perc" in out.columns:
        out["prev_payment_signal"] = out["Previous_Payment"] * out["Previous_Payment_Perc"].fillna(0)

    # --- Gender missing flag ---
    if "Gender" in out.columns:
        out["gender_missing"] = out["Gender"].isna().astype(int)

    return out


def split_categorical_numeric(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], list[str]]:
    """Partition features into categorical vs numeric based on dtype and cardinality."""
    cat_cols = []
    num_cols = []
    force_numeric = {
        "Balance", "Total_Due", "BAR", "Instalment",
        "Previous_Total_Due", "Previous_Payment_Amount",
        "CreditScore1_CreditScore", "Income1_Income",
    }

    for col in feature_cols:
        series = df[col]
        # Object / string dtypes are always categorical
        if series.dtype == "object" or series.dtype.name in ("category", "string"):
            cat_cols.append(col)
            continue

        # Confirm values are actually numeric (guards against 'Unknown' in float columns)
        coerced = pd.to_numeric(series, errors="coerce")
        non_null = series.notna().sum()
        if non_null > 0 and coerced.notna().sum() / non_null < 0.95:
            cat_cols.append(col)
            continue

        if (
            col not in force_numeric
            and pd.api.types.is_integer_dtype(series)
            and series.nunique(dropna=False) <= 10
        ):
            # Low-cardinality integers treated as categorical (e.g. Book_key, bands)
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return cat_cols, num_cols


def build_preprocessor(
    cat_cols: list[str], num_cols: list[str]
) -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline.

    - Numeric: median imputation + standard scaling (helps logistic regression)
    - Categorical: most-frequent imputation + one-hot encoding
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )


def temporal_train_test_split(
    df: pd.DataFrame,
    train_cycles: list[int],
    test_cycle: int,
    cycle_col: str = "Billing_Cycle",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by billing cycle to respect temporal ordering.

    We train on cycles 1–3 and hold out cycle 4 as the most recent period,
    simulating production scoring on the latest portfolio snapshot.
    """
    train = df[df[cycle_col].isin(train_cycles)].copy()
    test = df[df[cycle_col] == test_cycle].copy()
    return train, test
