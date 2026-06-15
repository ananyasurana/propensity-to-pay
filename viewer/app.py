"""
Interactive viewer for the Payment Behaviour Prediction project.

Run from project root:
    py -m streamlit run viewer/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DATA_PATH, FIGURES_DIR, OUTPUT_DIR, TARGET
from eda import summarize_eda
from predict import ModelBundle, load_or_train_model, score_dataframe, train_and_save_model

st.set_page_config(
    page_title="Payment Propensity Viewer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def get_model(quick: bool = False) -> ModelBundle:
    return load_or_train_model(quick=quick)


def account_label(row: pd.Series) -> str:
    return (
        f"Account {row.get('Account_Key', '?')} | "
        f"Cycle {row.get('Billing_Cycle', '?')} | "
        f"{row.get('Book_Cycle', '?')} | Book {row.get('Book_key', '?')}"
    )


def prob_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": "Payment Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc71" if probability >= 0.6 else "#f39c12" if probability >= 0.35 else "#e74c3c"},
                "steps": [
                    {"range": [0, 35], "color": "#fadbd8"},
                    {"range": [35, 60], "color": "#fdebd0"},
                    {"range": [60, 100], "color": "#d5f5e3"},
                ],
                "threshold": {
                    "line": {"color": "#2c3e50", "width": 2},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def band_badge(band: str) -> str:
    if "High" in band:
        return "🟢 " + band
    if "Medium" in band:
        return "🟡 " + band
    return "🔴 " + band


def render_sidebar(df: pd.DataFrame) -> tuple[pd.DataFrame, ModelBundle]:
    st.sidebar.title("Payment Propensity")
    st.sidebar.caption("Collections & Credit Risk — test viewer")

    model_exists = (OUTPUT_DIR / "payment_model.joblib").exists()
    if model_exists:
        st.sidebar.success("Trained model found")
    else:
        st.sidebar.warning("No saved model — will train on first load")

    if st.sidebar.button("Retrain model (full CV)", help="Runs grid search; may take ~1 min"):
        with st.spinner("Training model..."):
            bundle = train_and_save_model(quick=False)
            st.cache_resource.clear()
            st.sidebar.success("Model retrained and saved")
        return df, bundle

    if st.sidebar.button("Quick retrain (no CV)", help="Faster demo training"):
        with st.spinner("Quick training..."):
            bundle = train_and_save_model(quick=True)
            st.cache_resource.clear()
            st.sidebar.success("Quick model saved")
        return df, bundle

    bundle = get_model(quick=not model_exists)

    st.sidebar.divider()
    st.sidebar.subheader("Filter portfolio")
    cycles = sorted(df["Billing_Cycle"].dropna().unique().tolist())
    books = sorted(df["Book_key"].dropna().unique().tolist())

    selected_cycles = st.sidebar.multiselect(
        "Billing cycle", cycles, default=cycles, key="cycle_filter"
    )
    selected_books = st.sidebar.multiselect(
        "Book / portfolio", books, default=books, key="book_filter"
    )

    filtered = df[
        df["Billing_Cycle"].isin(selected_cycles) & df["Book_key"].isin(selected_books)
    ].copy()

    st.sidebar.metric("Accounts in view", len(filtered))
    if TARGET in filtered.columns:
        st.sidebar.metric("Actual payment rate", f"{filtered[TARGET].mean():.1%}")

    return filtered, bundle


def render_dashboard(scored: pd.DataFrame) -> None:
    st.header("Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts scored", len(scored))
    c2.metric("Avg payment probability", f"{scored['predicted_payment_prob'].mean():.1%}")
    c3.metric("Predicted payers", int(scored["predicted_payment"].sum()))
    if TARGET in scored.columns:
        c4.metric("Actual payers", int(scored[TARGET].sum()))

    col_l, col_r = st.columns(2)

    with col_l:
        band_counts = scored["collections_band"].value_counts().reset_index()
        band_counts.columns = ["band", "count"]
        fig = px.bar(
            band_counts,
            x="band",
            y="count",
            color="band",
            title="Collections band distribution",
            labels={"band": "Band", "count": "Accounts"},
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if TARGET in scored.columns:
            fig = px.histogram(
                scored,
                x="predicted_payment_prob",
                color=TARGET.astype(str),
                nbins=30,
                barmode="overlay",
                opacity=0.65,
                title="Predicted probability by actual outcome",
                labels={"predicted_payment_prob": "Payment probability", "color": "Paid?"},
                color_discrete_map={"0": "#e74c3c", "1": "#2ecc71"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(
                scored,
                x="predicted_payment_prob",
                nbins=30,
                title="Predicted probability distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

    if (OUTPUT_DIR / "model_metrics.csv").exists():
        st.subheader("Model metrics (held-out test set)")
        st.dataframe(pd.read_csv(OUTPUT_DIR / "model_metrics.csv"), hide_index=True, use_container_width=True)


def render_single_account(scored: pd.DataFrame, raw: pd.DataFrame) -> None:
    st.header("Score single account")
    st.caption("Pick an account from the filtered portfolio to inspect the model prediction.")

    labels = [account_label(row) for _, row in raw.iterrows()]
    label_to_idx = {label: i for i, label in enumerate(labels)}

    selected_label = st.selectbox("Select account", labels, index=0)
    idx = label_to_idx[selected_label]
    account_row = raw.iloc[idx]
    score_row = scored.iloc[idx]

    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
        prob = float(score_row["predicted_payment_prob"])
        st.plotly_chart(prob_gauge(prob), use_container_width=True)
        st.markdown(f"### {band_badge(score_row['collections_band'])}")
        st.markdown(
            f"**Predicted class:** {'Will pay' if score_row['predicted_payment'] == 1 else 'Will not pay'}"
        )
        if TARGET in score_row.index and pd.notna(score_row[TARGET]):
            actual = int(score_row[TARGET])
            correct = score_row.get("prediction_correct", actual == score_row["predicted_payment"])
            st.markdown(f"**Actual outcome:** {'Paid' if actual == 1 else 'Did not pay'}")
            st.markdown(f"**Prediction:** {'Correct' if correct else 'Incorrect'}")

    with col_info:
        st.subheader("Key account signals")
        highlight = {
            "Balance": account_row.get("Balance"),
            "Total_Due": account_row.get("Total_Due"),
            "Deliquency": account_row.get("Deliquency"),
            "Recency": account_row.get("Recency"),
            "Previous_Payment": account_row.get("Previous_Payment"),
            "Previous_Payment_Perc": account_row.get("Previous_Payment_Perc"),
            "PAttempts": account_row.get("PAttempts"),
            "PropensistyToRol": account_row.get("PropensistyToRol"),
            "BehaviourRiskScore": account_row.get("BehaviourRiskScore"),
            "PaymentProjectionScore": account_row.get("PaymentProjectionScore"),
            "Age": account_row.get("Age"),
        }
        for k, v in highlight.items():
            if pd.notna(v):
                if isinstance(v, float):
                    st.write(f"**{k}:** {v:,.2f}")
                else:
                    st.write(f"**{k}:** {v}")

        if account_row.get("Recency") == 9999:
            st.info("Recency = 9999 → account has **never made a payment**.")

    with st.expander("Full account record"):
        st.dataframe(
            pd.DataFrame(account_row).T.rename(columns={account_row.name: "value"}),
            use_container_width=True,
        )


def render_batch_scoring(bundle: ModelBundle) -> None:
    st.header("Batch scoring")
    st.caption("Upload a CSV with the same columns as `raw-data.csv` to score multiple accounts.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.info("No file uploaded. Use `raw-data.csv` or export a subset to test batch scoring.")
        if st.button("Score full raw-data.csv instead"):
            df = load_raw_data(str(DATA_PATH))
            with st.spinner("Scoring..."):
                results = score_dataframe(df, bundle)
            st.session_state["batch_results"] = results
            st.success(f"Scored {len(results)} accounts from raw-data.csv")
        return

    try:
        upload_df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(upload_df)}** rows, **{len(upload_df.columns)}** columns")

        if st.button("Run predictions"):
            with st.spinner("Scoring uploaded file..."):
                results = score_dataframe(upload_df, bundle, include_actual=TARGET in upload_df.columns)
            st.session_state["batch_results"] = results
            st.success("Scoring complete")

    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
        return

    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        display_cols = [
            c for c in [
                "Account_Key", "Billing_Cycle", "Book_Cycle", "Book_key",
                "predicted_payment_prob", "predicted_payment", "collections_band",
                TARGET, "prediction_correct",
            ]
            if c in results.columns
        ]
        st.dataframe(
            results[display_cols].sort_values("predicted_payment_prob", ascending=False),
            hide_index=True,
            use_container_width=True,
        )
        csv_bytes = results[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scored results",
            data=csv_bytes,
            file_name="scored_accounts.csv",
            mime="text/csv",
        )


def render_explorer(scored: pd.DataFrame) -> None:
    st.header("Data explorer")
    st.caption("Sort and filter scored accounts.")

    min_prob, max_prob = st.slider(
        "Payment probability range",
        0.0, 1.0, (0.0, 1.0),
        step=0.05,
    )
    band_filter = st.multiselect(
        "Collections band",
        sorted(scored["collections_band"].unique()),
        default=sorted(scored["collections_band"].unique()),
    )

    view = scored[
        (scored["predicted_payment_prob"] >= min_prob)
        & (scored["predicted_payment_prob"] <= max_prob)
        & (scored["collections_band"].isin(band_filter))
    ]

    display_cols = [
        c for c in [
            "Account_Key", "Billing_Num", "Billing_Cycle", "Book_Cycle", "Book_key",
            "Balance", "Total_Due", "Recency", "Previous_Payment",
            "predicted_payment_prob", "predicted_payment", "collections_band",
            TARGET, "prediction_correct",
        ]
        if c in view.columns
    ]

    st.dataframe(
        view[display_cols].sort_values("predicted_payment_prob", ascending=False),
        hide_index=True,
        use_container_width=True,
        height=450,
    )
    st.caption(f"Showing {len(view)} of {len(scored)} accounts")


def render_insights() -> None:
    st.header("Model & EDA insights")

    col1, col2 = st.columns(2)

    with col1:
        if (OUTPUT_DIR / "feature_importance.csv").exists():
            st.subheader("Top feature importances")
            imp = pd.read_csv(OUTPUT_DIR / "feature_importance.csv").head(15)
            fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Random Forest")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if (OUTPUT_DIR / "collections_bands_summary.csv").exists():
            st.subheader("Collections bands (test set)")
            st.dataframe(pd.read_csv(OUTPUT_DIR / "collections_bands_summary.csv"), hide_index=True)

    st.subheader("Saved EDA plots")
    if FIGURES_DIR.exists():
        figures = sorted(FIGURES_DIR.glob("*.png"))
        if figures:
            cols = st.columns(2)
            for i, fig_path in enumerate(figures):
                cols[i % 2].image(str(fig_path), caption=fig_path.stem.replace("_", " "), use_container_width=True)
        else:
            st.info("Run `py run_pipeline.py` to generate EDA figures.")
    else:
        st.info("Run `py run_pipeline.py` to generate outputs.")


def main() -> None:
    st.title("Customer Payment Behaviour — Test Viewer")
    st.markdown(
        "Explore predictions, score individual accounts, and run batch scoring "
        "against the trained **Random Forest** model."
    )

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at `{DATA_PATH}`. Place `raw-data.csv` in the project root.")
        return

    raw_df = load_raw_data(str(DATA_PATH))
    filtered_df, bundle = render_sidebar(raw_df)

    if filtered_df.empty:
        st.warning("No accounts match the sidebar filters.")
        return

    with st.spinner("Scoring filtered accounts..."):
        scored_df = score_dataframe(filtered_df, bundle)

    tab_dash, tab_single, tab_batch, tab_explore, tab_insights = st.tabs([
        "Dashboard",
        "Score account",
        "Batch scoring",
        "Explorer",
        "Insights",
    ])

    with tab_dash:
        render_dashboard(scored_df)
    with tab_single:
        render_single_account(scored_df, filtered_df)
    with tab_batch:
        render_batch_scoring(bundle)
    with tab_explore:
        render_explorer(scored_df)
    with tab_insights:
        render_insights()


if __name__ == "__main__":
    main()
