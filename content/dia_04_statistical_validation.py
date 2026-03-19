from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

from src.common.common import page_setup

page_setup()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PARQUET_PATH = PARQUET_PATH = (
    "example-data/20200505_Evosep_200SPD_SG06-16_MLHeLa_200ng_py8_S3-A1_1_2737_osw_features_f32_brotli_select_50000_rand_prec_id.parquet"
)

SCORE_COLS = [
    "var_bseries_score",
    "var_dotprod_score",
    "var_intensity_score",
    "var_isotope_correlation_score",
    "var_isotope_overlap_score",
    "var_library_corr",
    "var_library_dotprod",
    "var_library_manhattan",
    "var_library_rmsd",
    "var_library_rootmeansquare",
    "var_library_sangle",
    "var_log_sn_score",
    "var_manhattan_score",
    "var_massdev_score",
    "var_massdev_score_weighted",
    "var_mi_score",
    "var_mi_weighted_score",
    "var_norm_rt_score",
    "var_xcorr_coelution",
    "var_xcorr_coelution_weighted",
    "main_var_xcorr_shape",
    "var_xcorr_shape_weighted",
    "var_yseries_score",
    "var_im_xcorr_shape",
    "var_im_xcorr_coelution",
    "var_im_delta_score",
    "var_im_log_intensity",
]

MAIN_SCORE = "main_var_xcorr_shape"
SS_INITIAL_FDR = 0.15
SS_ITERATION_FDR = 0.05
SS_NUM_ITER = 3
TARGET_COLOR = "#F5793A"
DECOY_COLOR = "#0F2080"
MODEL_COLORS = {
    "LDA": "#2266CC",
    "SVM": "#CC2222",
    "XGBoost": "#22AA33",
    "MLP": "#11AACC",
}

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "s1_done": False,
    "feat_df": None,
    # scaling done invisibly after s1
    "X_scaled": None,
    "use_cols": None,
    "s2_done": False,
    "all_scores": {},
    "s3_done": False,
    "stats_results": {},
    "s4_done": False,
    "importance_cache": {},
    "s5_done": False,
    "scatter_cache": {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def load(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        st.error(f"Test Parquet file not found at {path}. ")


def scale_features(feat_df: pd.DataFrame):
    use_cols = [c for c in SCORE_COLS if feat_df[c].notna().mean() > 0.5]
    X_raw = (
        feat_df[use_cols].fillna(feat_df[use_cols].median()).to_numpy(dtype=np.float32)
    )
    return StandardScaler().fit_transform(X_raw), use_cols


# ----------------------------------------------
# Stages


@st.fragment
def render_stage_1() -> None:
    """Render Stage 1 inside a fragment so its widgets rerun independently."""
    st.markdown("---")
    st.subheader("Stage 1: Load Feature Data & Exploratory Analysis")

    s1_btn = st.button(
        "▶ Load Feature Data",
        type="primary",
        disabled=st.session_state.s1_done,
        key="s1_load_btn",
    )

    if s1_btn and not st.session_state.s1_done:
        with st.spinner("Loading / generating feature data and scaling..."):
            feat_df = load(PARQUET_PATH)
            X_scaled, use_cols = scale_features(feat_df)
            st.session_state.feat_df = feat_df
            st.session_state.X_scaled = X_scaled
            st.session_state.use_cols = use_cols
            st.session_state.importance_cache = {}
            st.session_state.s1_done = True
        st.rerun()

    if not st.session_state.s1_done:
        return

    feat_df = st.session_state.feat_df
    src = "from parquet" if os.path.exists(PARQUET_PATH) else "synthetically generated"
    st.success(f"Feature data loaded ({src}). Features scaled in background.")

    n_t = (feat_df["decoy"] == 0).sum()
    n_d = (feat_df["decoy"] == 1).sum()
    n_p = feat_df["group_id"].nunique()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total features", f"{len(feat_df):,}")
    c2.metric("Target features", f"{n_t:,}")
    c3.metric("Decoy features", f"{n_d:,}")
    c4.metric("Unique precursors", f"{n_p:,}")

    st.markdown(
        f"Each precursor has on average **{len(feat_df) / n_p:.1f}** candidate peak groups."
    )
    st.markdown(
        "Feature StandardScaling (zero mean, unit variance) is applied in the "
        "background — no feature selection is performed. The semi-supervised "
        "learning process implicitly weights features through the model."
    )

    st.markdown("#### Score distribution explorer")
    st.markdown(
        "Use the dropdown to inspect the separation between target and decoy "
        "distributions for any individual score. Well-separating scores (like "
        "`main_var_xcorr_shape`) will be the most informative for the model."
    )

    eda_score = st.selectbox(
        "Select a score to visualise:",
        options=SCORE_COLS,
        index=SCORE_COLS.index(MAIN_SCORE),
        key="eda_score_sel",
    )

    t_vals = (
        feat_df.loc[feat_df["decoy"] == 0, eda_score].dropna().astype(float).to_numpy()
    )
    d_vals = (
        feat_df.loc[feat_df["decoy"] == 1, eda_score].dropna().astype(float).to_numpy()
    )

    fig_eda = make_subplots(rows=1, cols=2, subplot_titles=["Histogram", "KDE Density"])
    for vals, label, col in [
        (t_vals, "Target", TARGET_COLOR),
        (d_vals, "Decoy", DECOY_COLOR),
    ]:
        h, e = np.histogram(vals, bins=60, density=True)
        fig_eda.add_trace(
            go.Bar(
                x=(e[:-1] + e[1:]) / 2,
                y=h,
                name=label,
                marker_color=col,
                opacity=0.70,
                legendgroup=label,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        x_g = np.linspace(vals.min() - 0.2, vals.max() + 0.2, 300)
        try:
            y_k = gaussian_kde(vals)(x_g)
        except Exception:
            y_k = np.zeros_like(x_g)
        fig_eda.add_trace(
            go.Scatter(
                x=x_g,
                y=y_k,
                mode="lines",
                name=label,
                line=dict(color=col, width=2),
                legendgroup=label,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig_eda.update_xaxes(title_text=eda_score)
    fig_eda.update_yaxes(title_text="Density", col=1)
    fig_eda.update_yaxes(title_text="Density", col=2)
    fig_eda.update_layout(
        height=330,
        barmode="overlay",
        legend=dict(title="Label"),
        title_text=f"Score distribution: {eda_score}",
    )
    st.plotly_chart(fig_eda, use_container_width=True)


# -----------------------------------------------------------------------------
# Page content

st.title("Statistical Validation of DIA Features")
st.markdown(
    """
This page demonstrates **semi-supervised statistical validation** of DIA feature scores.

The workflow mirrors a typical workflow in PyProphet:
1. Rank all candidate peak groups by **main score** (`main_var_xcorr_shape`)
2. Select **confident target training examples** (top-1 per group passing an FDR threshold) + all decoys
3. Train a discriminant model on this enriched training set
4. Re-score all features, tighten the FDR, repeat for several iterations
5. Normalise the final score by the decoy distribution → **d-score**
6. Compute **empirical p-values**, **π₀** (bootstrap), **q-values** and **PEP**
"""
)


render_stage_1()
