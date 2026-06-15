"""Model persistence and scoring helpers for the viewer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import (
    DATA_PATH,
    OUTPUT_DIR,
    TARGET,
    TEST_BILLING_CYCLE,
    TRAIN_BILLING_CYCLES,
)
from modeling import assign_risk_bands, build_tuned_model, tune_random_forest
from preprocessing import (
    build_preprocessor,
    engineer_features,
    get_feature_columns,
    split_categorical_numeric,
    temporal_train_test_split,
)

MODEL_PATH = OUTPUT_DIR / "payment_model.joblib"
METADATA_PATH = OUTPUT_DIR / "model_metadata.json"


@dataclass
class ModelBundle:
    """Trained model plus the feature schema needed for scoring."""

    model: Any
    feature_cols: list[str]
    cat_cols: list[str]
    num_cols: list[str]
    train_cycles: list[int]
    test_cycle: int

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        METADATA_PATH.write_text(
            json.dumps(
                {
                    "feature_cols": self.feature_cols,
                    "cat_cols": self.cat_cols,
                    "num_cols": self.num_cols,
                    "train_cycles": self.train_cycles,
                    "test_cycle": self.test_cycle,
                    "n_features": len(self.feature_cols),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ModelBundle":
        if not path.exists():
            raise FileNotFoundError(
                f"No saved model at {path}. Run `py run_pipeline.py` first."
            )
        return joblib.load(path)


def prepare_scoring_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply feature engineering and return model-ready feature matrix."""
    engineered = engineer_features(df)
    feature_cols = get_feature_columns(engineered)
    if "Recency" in feature_cols:
        feature_cols.remove("Recency")
    return engineered, feature_cols


def train_and_save_model(
    data_path: Path = DATA_PATH,
    *,
    quick: bool = False,
) -> ModelBundle:
    """
    Train the Random Forest model and persist it for the viewer.

    quick=True skips grid search (faster startup for demos).
    """
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)
    if "Recency" in feature_cols:
        feature_cols.remove("Recency")

    cat_cols, num_cols = split_categorical_numeric(df, feature_cols)
    train_df, _ = temporal_train_test_split(
        df, TRAIN_BILLING_CYCLES, TEST_BILLING_CYCLE
    )

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    preprocessor = build_preprocessor(cat_cols, num_cols)
    pipeline = build_tuned_model(preprocessor)

    if quick:
        pipeline.fit(X_train, y_train)
        model = pipeline
    else:
        search = tune_random_forest(pipeline, X_train, y_train)
        model = search.best_estimator_

    bundle = ModelBundle(
        model=model,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
        train_cycles=TRAIN_BILLING_CYCLES,
        test_cycle=TEST_BILLING_CYCLE,
    )
    bundle.save()
    return bundle


def score_dataframe(
    df: pd.DataFrame,
    bundle: ModelBundle,
    *,
    include_actual: bool = True,
) -> pd.DataFrame:
    """Score rows and attach probability, predicted class, and collections band."""
    engineered, feature_cols = prepare_scoring_frame(df)

    # Align to training schema — missing columns filled with NaN
    for col in bundle.feature_cols:
        if col not in engineered.columns:
            engineered[col] = pd.NA
    X = engineered[bundle.feature_cols]

    probs = bundle.model.predict_proba(X)[:, 1]
    preds = bundle.model.predict(X)

    result = df.copy()
    result["predicted_payment_prob"] = probs
    result["predicted_payment"] = preds.astype(int)
    result["collections_band"] = assign_risk_bands(probs)

    if include_actual and TARGET in result.columns:
        result["prediction_correct"] = result[TARGET] == result["predicted_payment"]

    return result


def load_or_train_model(*, quick: bool = False) -> ModelBundle:
    """Load saved model, or train and save if missing."""
    try:
        return ModelBundle.load()
    except FileNotFoundError:
        return train_and_save_model(quick=quick)
