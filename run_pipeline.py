<<<<<<< HEAD
"""
Customer Payment Behaviour Prediction — end-to-end pipeline.

Run from project root:
    py run_pipeline.py

Outputs saved to outputs/figures/ and outputs/*.txt
"""

import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from config import (
    DATA_PATH,
    OUTPUT_DIR,
    TARGET,
    TEST_BILLING_CYCLE,
    TRAIN_BILLING_CYCLES,
)
from eda import (
    ensure_output_dirs,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_numeric_distributions,
    plot_target_balance,
    plot_target_relationships,
    summarize_eda,
)
from modeling import (
    assign_risk_bands,
    build_baseline_model,
    build_tuned_model,
    evaluate_model,
    get_transformed_feature_names,
    plot_feature_importance,
    tune_random_forest,
)
from predict import ModelBundle
from preprocessing import (
    build_preprocessor,
    engineer_features,
    get_feature_columns,
    split_categorical_numeric,
    temporal_train_test_split,
)


def main() -> None:
    print("=" * 60)
    print("Customer Payment Behaviour Prediction Pipeline")
    print("=" * 60)

    ensure_output_dirs()

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # -------------------------------------------------------------------------
    # 2. Exploratory Data Analysis
    # -------------------------------------------------------------------------
    print("\n[2/6] Running EDA and saving plots...")
    eda_stats = summarize_eda(df)
    print(f"  Payment rate: {eda_stats['payment_rate']:.1%}")
    print(f"  Never-paid accounts (Recency=9999): {eda_stats['never_paid_accounts']}")

    plot_target_balance(df)
    null_pct = plot_missing_values(df)
    plot_target_relationships(df, df.columns.tolist())

    # -------------------------------------------------------------------------
    # 3. Feature engineering & preprocessing setup
    # -------------------------------------------------------------------------
    print("\n[3/6] Feature engineering...")
    df = engineer_features(df)

    feature_cols = get_feature_columns(df)
    # Drop original Recency in favour of engineered columns
    if "Recency" in feature_cols:
        feature_cols.remove("Recency")

    cat_cols, num_cols = split_categorical_numeric(df, feature_cols)
    print(f"  Features: {len(feature_cols)} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")

    # EDA on engineered numerics
    plot_numeric_distributions(df, num_cols)
    plot_correlation_heatmap(df, num_cols + [TARGET])

    # -------------------------------------------------------------------------
    # 4. Temporal train/test split
    # -------------------------------------------------------------------------
    print("\n[4/6] Temporal train/test split...")
    train_df, test_df = temporal_train_test_split(
        df, TRAIN_BILLING_CYCLES, TEST_BILLING_CYCLE
    )
    print(f"  Train: {len(train_df)} rows (Billing_Cycle {TRAIN_BILLING_CYCLES})")
    print(f"  Test:  {len(test_df)} rows (Billing_Cycle {TEST_BILLING_CYCLE})")
    print(f"  Train payment rate: {train_df[TARGET].mean():.1%}")
    print(f"  Test payment rate:  {test_df[TARGET].mean():.1%}")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]

    preprocessor = build_preprocessor(cat_cols, num_cols)

    # -------------------------------------------------------------------------
    # 5. Model training
    # -------------------------------------------------------------------------
    print("\n[5/6] Training models...")

    # Baseline: Logistic Regression
    print("  Fitting baseline Logistic Regression...")
    baseline = build_baseline_model(preprocessor)
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate_model(
        baseline, X_test, y_test,
        model_name="Logistic Regression (Baseline)",
        save_prefix="baseline_lr",
    )
    print(f"  Baseline AUC-ROC: {baseline_metrics['auc_roc']:.4f}")

    # Tuned Random Forest
    print("  Running Random Forest hyperparameter search (5-fold CV)...")
    rf_pipeline = build_tuned_model(preprocessor)
    search = tune_random_forest(rf_pipeline, X_train, y_train)
    best_rf = search.best_estimator_
    print(f"  Best CV AUC: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")

    rf_metrics = evaluate_model(
        best_rf, X_test, y_test,
        model_name="Random Forest (Tuned)",
        save_prefix="tuned_rf",
    )
    print(f"  Random Forest AUC-ROC: {rf_metrics['auc_roc']:.4f}")

    # Feature importance
    fitted_preprocessor = best_rf.named_steps["preprocessor"]
    fitted_preprocessor.fit(X_train, y_train)  # ensure fitted for name extraction
    feature_names = get_transformed_feature_names(fitted_preprocessor, cat_cols, num_cols)
    imp_df = plot_feature_importance(best_rf, feature_names, save_prefix="tuned_rf")
    imp_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # Persist model for the interactive viewer
    ModelBundle(
        model=best_rf,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
        train_cycles=TRAIN_BILLING_CYCLES,
        test_cycle=TEST_BILLING_CYCLE,
    ).save()
    print("  Saved model artifact for viewer:", OUTPUT_DIR / "payment_model.joblib")

    # -------------------------------------------------------------------------
    # 6. Business segmentation on test set
    # -------------------------------------------------------------------------
    print("\n[6/6] Assigning risk bands for collections prioritisation...")
    test_probs = best_rf.predict_proba(X_test)[:, 1]
    test_df = test_df.copy()
    test_df["predicted_payment_prob"] = test_probs
    test_df["collections_band"] = assign_risk_bands(test_probs)

    band_summary = (
        test_df.groupby("collections_band", observed=True)
        .agg(
            accounts=("collections_band", "count"),
            actual_payment_rate=(TARGET, "mean"),
            avg_predicted_prob=("predicted_payment_prob", "mean"),
        )
        .sort_values("avg_predicted_prob", ascending=False)
    )
    print("\nCollections band summary (test set):")
    print(band_summary.to_string())
    band_summary.to_csv(OUTPUT_DIR / "collections_bands_summary.csv")

    # Save metrics summary
    metrics_df = pd.DataFrame([
        {"model": "Logistic Regression (Baseline)", **baseline_metrics},
        {"model": "Random Forest (Tuned)", **rf_metrics},
    ])
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    # Save EDA summary
    pd.Series(eda_stats).to_csv(OUTPUT_DIR / "eda_summary.csv")
    null_pct.to_csv(OUTPUT_DIR / "missing_value_rates.csv")

    print("\n" + "=" * 60)
    print("Pipeline complete. Outputs in:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
=======
"""
Customer Payment Behaviour Prediction — end-to-end pipeline.

Run from project root:
    py run_pipeline.py

Outputs saved to outputs/figures/ and outputs/*.txt
"""

import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from config import (
    DATA_PATH,
    OUTPUT_DIR,
    TARGET,
    TEST_BILLING_CYCLE,
    TRAIN_BILLING_CYCLES,
)
from eda import (
    ensure_output_dirs,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_numeric_distributions,
    plot_target_balance,
    plot_target_relationships,
    summarize_eda,
)
from modeling import (
    assign_risk_bands,
    build_baseline_model,
    build_tuned_model,
    evaluate_model,
    get_transformed_feature_names,
    plot_feature_importance,
    tune_random_forest,
)
from predict import ModelBundle
from preprocessing import (
    build_preprocessor,
    engineer_features,
    get_feature_columns,
    split_categorical_numeric,
    temporal_train_test_split,
)


def main() -> None:
    print("=" * 60)
    print("Customer Payment Behaviour Prediction Pipeline")
    print("=" * 60)

    ensure_output_dirs()

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # -------------------------------------------------------------------------
    # 2. Exploratory Data Analysis
    # -------------------------------------------------------------------------
    print("\n[2/6] Running EDA and saving plots...")
    eda_stats = summarize_eda(df)
    print(f"  Payment rate: {eda_stats['payment_rate']:.1%}")
    print(f"  Never-paid accounts (Recency=9999): {eda_stats['never_paid_accounts']}")

    plot_target_balance(df)
    null_pct = plot_missing_values(df)
    plot_target_relationships(df, df.columns.tolist())

    # -------------------------------------------------------------------------
    # 3. Feature engineering & preprocessing setup
    # -------------------------------------------------------------------------
    print("\n[3/6] Feature engineering...")
    df = engineer_features(df)

    feature_cols = get_feature_columns(df)
    # Drop original Recency in favour of engineered columns
    if "Recency" in feature_cols:
        feature_cols.remove("Recency")

    cat_cols, num_cols = split_categorical_numeric(df, feature_cols)
    print(f"  Features: {len(feature_cols)} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")

    # EDA on engineered numerics
    plot_numeric_distributions(df, num_cols)
    plot_correlation_heatmap(df, num_cols + [TARGET])

    # -------------------------------------------------------------------------
    # 4. Temporal train/test split
    # -------------------------------------------------------------------------
    print("\n[4/6] Temporal train/test split...")
    train_df, test_df = temporal_train_test_split(
        df, TRAIN_BILLING_CYCLES, TEST_BILLING_CYCLE
    )
    print(f"  Train: {len(train_df)} rows (Billing_Cycle {TRAIN_BILLING_CYCLES})")
    print(f"  Test:  {len(test_df)} rows (Billing_Cycle {TEST_BILLING_CYCLE})")
    print(f"  Train payment rate: {train_df[TARGET].mean():.1%}")
    print(f"  Test payment rate:  {test_df[TARGET].mean():.1%}")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]

    preprocessor = build_preprocessor(cat_cols, num_cols)

    # -------------------------------------------------------------------------
    # 5. Model training
    # -------------------------------------------------------------------------
    print("\n[5/6] Training models...")

    # Baseline: Logistic Regression
    print("  Fitting baseline Logistic Regression...")
    baseline = build_baseline_model(preprocessor)
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate_model(
        baseline, X_test, y_test,
        model_name="Logistic Regression (Baseline)",
        save_prefix="baseline_lr",
    )
    print(f"  Baseline AUC-ROC: {baseline_metrics['auc_roc']:.4f}")

    # Tuned Random Forest
    print("  Running Random Forest hyperparameter search (5-fold CV)...")
    rf_pipeline = build_tuned_model(preprocessor)
    search = tune_random_forest(rf_pipeline, X_train, y_train)
    best_rf = search.best_estimator_
    print(f"  Best CV AUC: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")

    rf_metrics = evaluate_model(
        best_rf, X_test, y_test,
        model_name="Random Forest (Tuned)",
        save_prefix="tuned_rf",
    )
    print(f"  Random Forest AUC-ROC: {rf_metrics['auc_roc']:.4f}")

    # Feature importance
    fitted_preprocessor = best_rf.named_steps["preprocessor"]
    fitted_preprocessor.fit(X_train, y_train)  # ensure fitted for name extraction
    feature_names = get_transformed_feature_names(fitted_preprocessor, cat_cols, num_cols)
    imp_df = plot_feature_importance(best_rf, feature_names, save_prefix="tuned_rf")
    imp_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # Persist model for the interactive viewer
    ModelBundle(
        model=best_rf,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
        train_cycles=TRAIN_BILLING_CYCLES,
        test_cycle=TEST_BILLING_CYCLE,
    ).save()
    print("  Saved model artifact for viewer:", OUTPUT_DIR / "payment_model.joblib")

    # -------------------------------------------------------------------------
    # 6. Business segmentation on test set
    # -------------------------------------------------------------------------
    print("\n[6/6] Assigning risk bands for collections prioritisation...")
    test_probs = best_rf.predict_proba(X_test)[:, 1]
    test_df = test_df.copy()
    test_df["predicted_payment_prob"] = test_probs
    test_df["collections_band"] = assign_risk_bands(test_probs)

    band_summary = (
        test_df.groupby("collections_band", observed=True)
        .agg(
            accounts=("collections_band", "count"),
            actual_payment_rate=(TARGET, "mean"),
            avg_predicted_prob=("predicted_payment_prob", "mean"),
        )
        .sort_values("avg_predicted_prob", ascending=False)
    )
    print("\nCollections band summary (test set):")
    print(band_summary.to_string())
    band_summary.to_csv(OUTPUT_DIR / "collections_bands_summary.csv")

    # Save metrics summary
    metrics_df = pd.DataFrame([
        {"model": "Logistic Regression (Baseline)", **baseline_metrics},
        {"model": "Random Forest (Tuned)", **rf_metrics},
    ])
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    # Save EDA summary
    pd.Series(eda_stats).to_csv(OUTPUT_DIR / "eda_summary.csv")
    null_pct.to_csv(OUTPUT_DIR / "missing_value_rates.csv")

    print("\n" + "=" * 60)
    print("Pipeline complete. Outputs in:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
>>>>>>> c27659818b2768601dee142b1f82c857306b7e61
