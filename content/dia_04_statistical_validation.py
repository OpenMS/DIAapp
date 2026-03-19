from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import scipy.stats
from scipy.stats import gaussian_kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.common.common import page_setup

page_setup()

# -----------------------------------------------------------------------------
# Constants

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

# -----------------------------------------------------------------------------
# Session state

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


# -----------------------------------------------------------------------------
# Statistical functions (matching PyProphet / bioconductor qvalue)


def pemp(stat: np.ndarray, stat0: np.ndarray) -> np.ndarray:
    """Empirical p-values (identical to bioconductor/qvalue empPvals)."""
    stat = np.asarray(stat, dtype=np.float64)
    stat0 = np.asarray(stat0, dtype=np.float64)
    m, m0 = len(stat), len(stat0)
    statc = np.concatenate([stat, stat0])
    v = np.array([True] * m + [False] * m0)
    perm = np.argsort(-statc, kind="mergesort")
    v = v[perm]
    u = np.where(v)[0]
    p = (u - np.arange(m)) / float(m0)
    ranks = np.floor(scipy.stats.rankdata(-stat)).astype(int) - 1
    p = p[ranks]
    p[p <= 1.0 / m0] = 1.0 / m0
    return np.clip(p, 0.0, 1.0)


def pi0est_bootstrap(p_values: np.ndarray) -> dict:
    """
    Estimate pi0 using the bootstrap method (bioconductor/qvalue bootstrap option).
    More robust than the smoother when there are few p-values near 1.
    """
    lambda_seq = np.arange(0.05, 1.0, 0.05)
    p = np.asarray(p_values, dtype=np.float64)
    p = p[np.isfinite(p)]
    m = len(p)
    pi0_lambda = np.array([np.mean(p >= l) / (1.0 - l) for l in lambda_seq])
    min_pi0 = np.percentile(pi0_lambda, 10)
    W = np.array([np.sum(p >= l) for l in lambda_seq], dtype=float)
    mse = (W / (m**2 * (1.0 - lambda_seq) ** 2)) * (1.0 - W / m) + (
        pi0_lambda - min_pi0
    ) ** 2
    pi0 = float(np.minimum(pi0_lambda[np.argmin(mse)], 1.0))
    pi0 = max(pi0, 1e-6)
    return {
        "pi0": pi0,
        "pi0_lambda": pi0_lambda,
        "lambda_": lambda_seq,
        "pi0_smooth": False,  # bootstrap → show P-P plot, not smoother plot
    }


def qvalue_from_pvalues(p_values: np.ndarray, pi0: float) -> np.ndarray:
    """q-values with pi0 correction (Storey & Tibshirani 2003)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    u = np.argsort(p)
    v = scipy.stats.rankdata(p, "max")
    q = np.minimum((pi0 * m * p) / v, 1.0)
    q[u[-1]] = min(q[u[-1]], 1.0)
    for i in range(m - 3, -1, -1):
        q[u[i]] = min(q[u[i]], q[u[i + 1]])
    return q


def lfdr_from_pvalues(p_values: np.ndarray, pi0: float) -> np.ndarray:
    """Local FDR (PEP) via KDE on probit-transformed p-values."""
    p = np.clip(np.asarray(p_values, dtype=np.float64), 1e-10, 1 - 1e-10)
    z = scipy.stats.norm.ppf(p)
    bw = 1.06 * np.std(z) * len(z) ** (-0.2)
    bw = max(bw, 0.01)
    try:
        kde = gaussian_kde(z, bw_method=bw)
        f_z = kde(z)
        phi_z = scipy.stats.norm.pdf(z)
        f_p = np.maximum(f_z / (phi_z + 1e-20), 1e-20)
        pep = np.clip(pi0 / f_p, 0.0, 1.0)
    except Exception:
        pep = np.full_like(p, pi0)
    return pep


def normalize_score_by_decoys(scores: np.ndarray, is_decoy: np.ndarray) -> np.ndarray:
    """Normalise so decoy distribution has mean=0, std=1."""
    d = scores[is_decoy == 1]
    mu = d.mean()
    sd = d.std(ddof=1) or 1.0
    return (scores - mu) / sd


def select_top_targets_by_fdr(
    t_sc: np.ndarray, d_sc: np.ndarray, fdr_cutoff: float
) -> float:
    """Score threshold at given empirical FDR."""
    sorted_t = np.sort(t_sc)
    sorted_d = np.sort(d_sc)
    try:
        p = pemp(sorted_t, sorted_d)
        pi0 = pi0est_bootstrap(p)["pi0"]
        q = qvalue_from_pvalues(p, pi0)
        passing = sorted_t[q <= fdr_cutoff]
        return float(passing.min()) if len(passing) else float(np.percentile(t_sc, 85))
    except Exception:
        return float(np.percentile(t_sc, 85))


# -----------------------------------------------------------------------------
# Semi-supervised learning loop


def run_semi_supervised(
    feat_df: pd.DataFrame,
    X_scaled: np.ndarray,
    model_name: str,
    n_iter: int = SS_NUM_ITER,
) -> np.ndarray:
    """
    PyProphet-style semi-supervised loop:
      1. Start from main_score rankings
      2. Select confident targets (top-1 per group passing FDR) + all decoys
      3. Train model → re-score → repeat
      4. Normalise final scores by decoy distribution
    """
    is_decoy = feat_df["decoy"].to_numpy(dtype=np.int32)
    group_ids = feat_df["group_id"].to_numpy()
    n = len(feat_df)
    clf_scores = feat_df[MAIN_SCORE].fillna(0).to_numpy(dtype=np.float64)

    def _top1_idx(sc):
        df_t = pd.DataFrame({"g": group_ids, "s": sc, "i": np.arange(n)})
        return (
            df_t.sort_values("s", ascending=False)
            .groupby("g", sort=False)["i"]
            .first()
            .to_numpy()
        )

    def _train_score(X_tr, y_tr):
        """Train model, score all features; ensure targets > decoys."""
        if model_name == "LDA":
            m = LinearDiscriminantAnalysis()
            m.fit(X_tr, y_tr)
            raw = m.decision_function(X_scaled)
        elif model_name == "SVM":
            m = Pipeline(
                [
                    ("sc", StandardScaler(with_std=False)),
                    ("clf", LinearSVC(max_iter=3000, C=0.05)),
                ]
            )
            m.fit(X_tr, y_tr)
            raw = m.decision_function(X_scaled)
        elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
            n_neg = int((y_tr == 1).sum())
            n_pos = int((y_tr == 0).sum())
            m = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
                n_jobs=1,
                scale_pos_weight=n_neg / max(n_pos, 1),
            )
            m.fit(X_tr, y_tr)
            raw = m.predict_proba(X_scaled)[:, 1]  # P(decoy)
        elif model_name == "MLP":
            m = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation="relu",
                solver="adam",
                max_iter=100,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
            )
            m.fit(X_tr, y_tr)
            raw = m.predict_proba(X_scaled)[:, 1]  # P(decoy)
        else:
            m = LinearDiscriminantAnalysis()
            m.fit(X_tr, y_tr)
            raw = m.decision_function(X_scaled)

        # Ensure targets score HIGHER than decoys
        if np.mean(raw[is_decoy == 0]) < np.mean(raw[is_decoy == 1]):
            raw = -raw
        return raw

    for iteration in range(n_iter + 1):
        fdr_thr = SS_INITIAL_FDR if iteration == 0 else SS_ITERATION_FDR
        top1_idx = _top1_idx(clf_scores)
        top1_mask = np.zeros(n, dtype=bool)
        top1_mask[top1_idx] = True

        t1_t = clf_scores[top1_mask & (is_decoy == 0)]
        t1_d = clf_scores[is_decoy == 1]

        if len(t1_t) < 20 or len(t1_d) < 20:
            break

        cutoff = select_top_targets_by_fdr(t1_t, t1_d, fdr_thr)
        conf_mask = top1_mask & (is_decoy == 0) & (clf_scores >= cutoff)
        train_mask = conf_mask | (is_decoy == 1)
        X_tr = X_scaled[train_mask]
        y_tr = is_decoy[train_mask].copy()

        if y_tr.sum() == 0 or (y_tr == 0).sum() == 0:
            break

        try:
            clf_scores = _train_score(X_tr, y_tr).astype(np.float64)
        except Exception:
            break
        clf_scores -= np.mean(clf_scores)

    return normalize_score_by_decoys(clf_scores, is_decoy)


def compute_full_stats(d_scores: np.ndarray, feat_df: pd.DataFrame) -> dict:
    """Top-1 per group → pemp p-values → pi0 bootstrap → q-values → PEP."""
    group_ids = feat_df["group_id"].to_numpy()
    is_decoy = feat_df["decoy"].to_numpy()
    n = len(feat_df)

    df_tmp = pd.DataFrame({"g": group_ids, "s": d_scores, "i": np.arange(n)})
    top1_idx = (
        df_tmp.sort_values("s", ascending=False)
        .groupby("g", sort=False)["i"]
        .first()
        .to_numpy()
    )
    top1_dec = is_decoy[top1_idx]
    top1_sc = d_scores[top1_idx]

    t_sc = np.sort(top1_sc[top1_dec == 0])
    d_sc = np.sort(top1_sc[top1_dec == 1])

    p_vals = pemp(t_sc, d_sc)
    pi0_res = pi0est_bootstrap(p_vals)
    q_vals = qvalue_from_pvalues(p_vals, pi0_res["pi0"])
    pep_vals = lfdr_from_pvalues(p_vals, pi0_res["pi0"])

    n_ids = int((q_vals <= 0.01).sum())
    # threshold in d-score space: minimum passing target score
    passing = t_sc[q_vals <= 0.01]
    thr = float(passing.min()) if len(passing) else float(t_sc.max())

    return {
        "t_scores": t_sc,
        "d_scores_dec": d_sc,
        "p_vals": p_vals,
        "q_vals": q_vals,
        "pep_vals": pep_vals,
        "pi0": pi0_res,
        "n_ids_1pct": n_ids,
        "threshold": thr,
    }


def compute_feature_importance_cache(
    feat_df: pd.DataFrame,
    X_scaled: np.ndarray,
    use_cols: list[str],
    all_scores: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Compute and cache feature importance data for each trained model."""
    is_decoy = feat_df["decoy"].to_numpy()
    group_ids = feat_df["group_id"].to_numpy()
    cache: dict[str, dict] = {}

    for nm, sc in all_scores.items():
        try:
            res = compute_full_stats(sc, feat_df)
            thr = res["threshold"]
            df_tmp = pd.DataFrame(
                {"g": group_ids, "s": sc, "i": np.arange(len(feat_df))}
            )
            top1_idx = (
                df_tmp.sort_values("s", ascending=False)
                .groupby("g", sort=False)["i"]
                .first()
                .to_numpy()
            )
            conf_mask = np.zeros(len(feat_df), dtype=bool)
            conf_mask[top1_idx] = True
            conf_mask &= (is_decoy == 0) & (sc >= thr)
            train_mask = conf_mask | (is_decoy == 1)
            X_tr = X_scaled[train_mask]
            y_tr = is_decoy[train_mask]

            imp_vals = None
            imp_std = None
            imp_label = ""

            if nm == "LDA":
                m = LinearDiscriminantAnalysis().fit(X_tr, y_tr)
                coef = m.coef_[0]
                if np.mean(m.decision_function(X_scaled)[is_decoy == 0]) < np.mean(
                    m.decision_function(X_scaled)[is_decoy == 1]
                ):
                    coef = -coef
                imp_vals = coef
                imp_label = "LDA coefficient"
            elif nm == "SVM":
                m = Pipeline(
                    [
                        ("sc", StandardScaler(with_std=False)),
                        ("clf", LinearSVC(max_iter=3000, C=0.05)),
                    ]
                ).fit(X_tr, y_tr)
                coef = m.named_steps["clf"].coef_[0]
                if np.mean(m.decision_function(X_scaled)[is_decoy == 0]) < np.mean(
                    m.decision_function(X_scaled)[is_decoy == 1]
                ):
                    coef = -coef
                imp_vals = coef
                imp_label = "SVM coefficient"
            elif nm == "XGBoost" and XGBOOST_AVAILABLE:
                m = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    verbosity=0,
                    random_state=42,
                    n_jobs=1,
                ).fit(X_tr, y_tr)
                imp_vals = m.feature_importances_
                imp_label = "XGBoost gain"
            elif nm == "MLP":
                m = MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    max_iter=200,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                ).fit(X_tr, y_tr)
                perm = permutation_importance(
                    m,
                    X_tr,
                    y_tr,
                    scoring="roc_auc",
                    n_repeats=10,
                    random_state=42,
                    n_jobs=1,
                    max_samples=min(len(X_tr), 2000),
                )
                imp_vals = perm.importances_mean
                imp_std = perm.importances_std
                imp_label = "Permutation importance (ROC-AUC)"

            if imp_vals is not None:
                cache[nm] = {
                    "feature": list(use_cols),
                    "importance": np.asarray(imp_vals, dtype=float).tolist(),
                    "importance_std": None
                    if imp_std is None
                    else np.asarray(imp_std, dtype=float).tolist(),
                    "label": imp_label,
                }
        except Exception as exc:
            cache[nm] = {
                "error": str(exc),
            }

    return cache


# ----------------------------------------------
# Stages


@st.fragment
def render_stage_1() -> None:
    """Render Stage 1 inside a fragment so its widgets rerun independently."""
    st.markdown("---")
    st.subheader("Load Feature Data & Exploratory Analysis")

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


@st.fragment
def render_stage_2() -> None:
    """Render Stage 2 inside a fragment so training does not rerun other stages."""
    if not st.session_state.s1_done:
        return

    st.markdown("---")
    st.subheader("Semi-Supervised Discriminant Learning")

    available_models = ["LDA", "SVM"]
    if XGBOOST_AVAILABLE:
        available_models.append("XGBoost")
    available_models += ["MLP"]

    st.markdown(
        f"""
Each model is trained using the **iterative semi-supervised** approach
(matching PyProphet's `StandardSemiSupervisedLearner`):

| Parameter | Value |
|---|---|
| Iterations | {SS_NUM_ITER} |
| Initial FDR (round 0) | {SS_INITIAL_FDR * 100:.0f}% |
| Iteration FDR (rounds 1–{SS_NUM_ITER}) | {SS_ITERATION_FDR * 100:.0f}% |
| Models | {", ".join(available_models)} |
"""
    )

    s2_btn = st.button(
        "▶ Run Semi-Supervised Learning",
        type="primary",
        disabled=st.session_state.s2_done,
        key="s2_run_btn",
    )

    if s2_btn and not st.session_state.s2_done:
        feat_df = st.session_state.feat_df
        X_scaled = st.session_state.X_scaled
        all_scores = {}
        progress = st.progress(0)
        for k, nm in enumerate(available_models):
            with st.spinner(
                f"Training {nm} ({SS_NUM_ITER} semi-supervised iterations)..."
            ):
                try:
                    all_scores[nm] = run_semi_supervised(feat_df, X_scaled, nm)
                except Exception as e:
                    st.warning(f"{nm} failed: {e}")
            progress.progress(int(100 * (k + 1) / len(available_models)))
        st.session_state.all_scores = all_scores
        st.session_state.importance_cache = {}
        st.session_state.s2_done = True
        st.rerun()

    if not st.session_state.s2_done:
        return

    feat_df = st.session_state.feat_df
    all_scores = st.session_state.all_scores

    rows = []
    for nm, sc in all_scores.items():
        try:
            res = compute_full_stats(sc, feat_df)
            try:
                top1 = res["t_scores"]
                top1d = res["d_scores_dec"]
                auc = roc_auc_score(
                    np.concatenate([np.ones(len(top1)), np.zeros(len(top1d))]),
                    np.concatenate([top1, top1d]),
                )
            except Exception:
                auc = float("nan")
            rows.append(
                {
                    "Model": nm,
                    "ROC-AUC (top-1)": round(auc, 4),
                    "IDs @ 1% FDR (top-1)": res["n_ids_1pct"],
                }
            )
        except Exception:
            rows.append(
                {"Model": nm, "ROC-AUC (top-1)": "error", "IDs @ 1% FDR (top-1)": 0}
            )

    st.markdown("#### Summary — semi-supervised model performance")
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    with st.expander("Semi-supervised loop pseudocode"):
        st.code(
            """# PyProphet-style semi-supervised learning (simplified)

clf_scores = main_score   # initialise from main score

for iteration in range(n_iter):
    fdr = ss_initial_fdr if iteration == 0 else ss_iteration_fdr

    # Select top-1 target per precursor group
    top1_targets = [max_score_feature for each precursor group (targets)]

    # FDR cutoff: empirical p-values → pi0 bootstrap → q-values
    p = pemp(sorted(top1_targets), sorted(all_decoy_scores))
    q = qvalue(p, pi0_bootstrap(p))
    cutoff = min_score_where(q <= fdr)

    # Confident training set: targets passing cutoff + all decoys
    X_train = concat(top1_targets[score >= cutoff], all_decoys)
    y_train = [0]*n_confident_targets + [1]*n_decoys

    # Retrain and rescore everything
    model.fit(X_train, y_train)
    clf_scores = model.score(X_all) - mean(model.score(X_all))

# Normalise by decoy distribution → d-score
d_scores = (clf_scores - mean(decoy_clf)) / std(decoy_clf)
""",
            language="python",
        )


@st.fragment
def render_stage_3() -> None:
    """Render Stage 3 inside a fragment so plotting/statistics rerun independently."""
    if not st.session_state.s2_done:
        return

    st.markdown("---")
    st.subheader("Score Distributions & Statistical Validation")

    s3_btn = st.button(
        "▶ Compute Statistics & Show Plots",
        type="primary",
        disabled=st.session_state.s3_done,
        key="s3_stats_btn",
    )

    if s3_btn and not st.session_state.s3_done:
        feat_df = st.session_state.feat_df
        all_scores = st.session_state.all_scores
        stats_results = {}
        for nm, sc in all_scores.items():
            try:
                stats_results[nm] = compute_full_stats(sc, feat_df)
            except Exception as e:
                st.warning(f"Stats failed for {nm}: {e}")
        st.session_state.stats_results = stats_results
        st.session_state.s3_done = True
        st.rerun()

    if not st.session_state.s3_done:
        return

    stats_results = st.session_state.stats_results

    st.markdown(
        """
### How the statistics are computed

**Empirical p-values** (`pemp`, matching bioconductor/qvalue `empPvals`):
For each top-1 target d-score, count how many decoy d-scores are ≥ it,
normalised by the total number of decoys — fully non-parametric.

**π₀ estimation** (bootstrap method):
The proportion of null features is estimated as:
the minimum MSE bootstrap estimate across a grid of lambda values,
where `pi0(lambda) = #{p >= lambda} / (m * (1-lambda))`.

**q-values** (Storey & Tibshirani 2003):
`q(p) = pi0 * m * p / rank(p)`, monotonised from right to left.

**PEP** (local FDR via probit-KDE):
`PEP(p) = pi0 / f(p)`, where f(p) is a KDE density estimate on
probit-transformed p-values.

**P-P plot** (target-decoy assumption check — Levitsky et al. 2017):
ECDF of target d-scores vs ECDF of decoy d-scores, interpolated onto a
common score grid. If the target-decoy model is valid, the P-P curve
should follow `y = pi0 * x` in the low-score (null) region and curve
upward toward (1,1) where true positives dominate.
"""
    )

    for nm, res in stats_results.items():
        t_sc = res["t_scores"]
        d_sc = res["d_scores_dec"]
        pvals = res["p_vals"]
        pi0_r = res["pi0"]
        pi0_v = pi0_r["pi0"]
        thr = res["threshold"]
        n_ids = res["n_ids_1pct"]

        fig3 = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                f"group d-score distributions — {nm}",
                f"group d-score densities — {nm}",
                f"p-value density histogram  (π₀ = {pi0_v:.3f})",
                "P-P Plot",
            ],
            vertical_spacing=0.20,
            horizontal_spacing=0.12,
        )

        for vals, label, col in [
            (t_sc, "target", TARGET_COLOR),
            (d_sc, "decoy", DECOY_COLOR),
        ]:
            h, e = np.histogram(vals, bins=40)
            fig3.add_trace(
                go.Bar(
                    x=(e[:-1] + e[1:]) / 2,
                    y=h,
                    name=label,
                    marker_color=col,
                    opacity=0.75,
                    legendgroup=label,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            x_g = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 300)
            try:
                y_k = gaussian_kde(vals)(x_g)
            except Exception:
                y_k = np.zeros_like(x_g)
            fig3.add_trace(
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
        for c in [1, 2]:
            fig3.add_vline(
                x=thr,
                line_dash="dash",
                line_color="grey",
                line_width=2,
                annotation_text=f"Cutoff @ 1%: {thr:.2f}" if c == 1 else "",
                row=1,
                col=c,
            )

        h_pv, e_pv = np.histogram(pvals, bins=20, density=True)
        fig3.add_trace(
            go.Bar(
                x=(e_pv[:-1] + e_pv[1:]) / 2,
                y=h_pv,
                marker_color="#5B9BD5",
                opacity=0.8,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig3.add_hline(y=pi0_v, line_color="red", line_width=2, row=2, col=1)

        x_t = np.sort(t_sc)
        y_t = np.arange(1, len(x_t) + 1) / len(x_t)
        x_d = np.sort(d_sc)
        y_d = np.arange(1, len(x_d) + 1) / len(x_d)
        x_seq = np.linspace(min(x_t.min(), x_d.min()), max(x_t.max(), x_d.max()), 1000)
        y_t_i = np.interp(x_seq, x_t, y_t)
        y_d_i = np.interp(x_seq, x_d, y_d)
        fig3.add_trace(
            go.Scatter(
                x=y_d_i,
                y=y_t_i,
                mode="markers",
                marker=dict(size=3, opacity=0.5, color="#5B9BD5"),
                name="Target vs Decoy ECDF",
                showlegend=True,
            ),
            row=2,
            col=2,
        )
        fig3.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="y = x (Perfect match)",
                showlegend=True,
            ),
            row=2,
            col=2,
        )
        fig3.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, pi0_v],
                mode="lines",
                line=dict(color="blue", dash="dot"),
                name=f"y = {pi0_v:.2f} × x",
                showlegend=True,
            ),
            row=2,
            col=2,
        )

        fig3.update_xaxes(title_text="d-score", row=1)
        fig3.update_xaxes(title_text="p-value", row=2, col=1)
        fig3.update_xaxes(title_text="Decoy ECDF", row=2, col=2)
        fig3.update_yaxes(title_text="# of groups", row=1, col=1)
        fig3.update_yaxes(title_text="density", row=1, col=2)
        fig3.update_yaxes(title_text="density", row=2, col=1)
        fig3.update_yaxes(title_text="Target ECDF", row=2, col=2)
        fig3.update_layout(
            height=720,
            barmode="overlay",
            margin=dict(t=80, b=50, l=60, r=40),
            title_text=f"{nm}  |  π₀ = {pi0_v:.3f}  |  IDs @ 1% FDR: {n_ids:,}",
            legend=dict(title="Label"),
        )
        st.plotly_chart(fig3, use_container_width=True)


@st.fragment
def render_stage_4() -> None:
    """Render Stage 4 inside a fragment so importance plots rerun independently."""
    if not st.session_state.s3_done:
        return

    st.markdown("---")
    st.subheader("Feature Importance")

    s4_btn = st.button(
        "▶ Show Feature Importance",
        type="primary",
        disabled=st.session_state.s4_done,
        key="s4_importance_btn",
    )

    if s4_btn and not st.session_state.s4_done:
        st.session_state.s4_done = True
        st.rerun()

    if not st.session_state.s4_done:
        return

    feat_df = st.session_state.feat_df
    X_scaled = st.session_state.X_scaled
    use_cols = st.session_state.use_cols
    all_scores = st.session_state.all_scores
    is_decoy = feat_df["decoy"].to_numpy()
    group_ids = feat_df["group_id"].to_numpy()

    st.markdown(
        """
Feature importance is derived from the **final-iteration** model fitted on the
semi-supervised training set for each scorer. Linear models show **signed** coefficients
(positive pushes toward targets, negative toward decoys). XGBoost gain and MLP
permutation importance are **magnitude-only** measures.
"""
    )

    if not st.session_state.importance_cache:
        with st.spinner(
            "Fitting feature-importance models (including permutation importance for MLP)..."
        ):
            st.session_state.importance_cache = compute_feature_importance_cache(
                feat_df=feat_df,
                X_scaled=X_scaled,
                use_cols=use_cols,
                all_scores=all_scores,
            )

    for nm, imp_res in st.session_state.importance_cache.items():
        if "error" in imp_res:
            st.warning(f"Importance unavailable for {nm}: {imp_res['error']}")
            continue

        imp_df = pd.DataFrame(
            {
                "feature": imp_res["feature"],
                "importance": imp_res["importance"],
                "importance_std": imp_res["importance_std"]
                if imp_res["importance_std"] is not None
                else [None] * len(imp_res["feature"]),
            }
        )
        imp_df = imp_df.sort_values("importance", ascending=True).tail(15)
        fig_imp = go.Figure(
            go.Bar(
                x=imp_df["importance"],
                y=imp_df["feature"],
                orientation="h",
                marker_color=MODEL_COLORS.get(nm, "#888888"),
                opacity=0.85,
                error_x=(
                    dict(type="data", array=imp_df["importance_std"], visible=True)
                    if imp_res["importance_std"] is not None
                    else None
                ),
            )
        )
        fig_imp.update_layout(
            title=f"{nm} — {imp_res['label']} (top 15)",
            xaxis_title="Importance",
            yaxis_title="",
            height=400,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown(
        """
        Across models, a small set of chromatography and library-matching features consistently drive discrimination, especially **var_xcorr_shape_weighted**, **var_dotprod_score**, and **var_intensity_score**, **var_mi_score**, suggesting these contain most of the target-versus-decoy signal. The exact rankings differ by model because each learner captures the feature relationships differently, but the overall pattern is consistent: weighted shape/coelution, spectral similarity, intensity, and RT-related features are the most informative, while a few negatively weighted features in LDA/SVM likely act as evidence against true targets rather than in favor of them.
        """
    )


@st.fragment
def _render_comparison(scatter_cache: dict, all_scores: dict) -> None:
    """Fragment: only this section rerenders when the dropdown changes."""
    if not scatter_cache:
        st.info("No models to compare (LDA may be the only model trained).")
        return

    others = list(scatter_cache.keys())
    sel = st.selectbox(
        "Compare LDA against:",
        options=others,
        key="model_cmp_sel",
    )

    cache = scatter_cache[sel]
    md = cache["marg_data"]
    thr_lda = cache["thr_lda"]
    thr_sel = cache["thr_other"]
    n_lda = cache["n_lda"]
    n_sel = cache["n_other"]

    st.markdown(
        f"Scatter: **LDA d-score** (x) vs **{sel} d-score** (y). "
        "Dashed lines = 1% FDR thresholds. Marginal histograms show score distributions for targets and decoys."
    )

    fig5 = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.78, 0.22],
        row_heights=[0.22, 0.78],
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for label, col in [("Target", TARGET_COLOR), ("Decoy", DECOY_COLOR)]:
        d = md[label]
        fig5.add_trace(
            go.Scattergl(
                x=d["x"],
                y=d["y"],
                mode="markers",
                marker=dict(color=col, size=2, opacity=0.20),
                name=label,
                legendgroup=label,
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        fig5.add_trace(
            go.Bar(
                x=d["ex"],
                y=d["hx"],
                name=label,
                marker_color=col,
                opacity=0.65,
                legendgroup=label,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig5.add_trace(
            go.Bar(
                x=d["hy"],
                y=d["ey"],
                orientation="h",
                name=label,
                marker_color=col,
                opacity=0.65,
                legendgroup=label,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig5.add_vline(
        x=thr_lda,
        line_dash="dash",
        line_color=MODEL_COLORS["LDA"],
        line_width=2,
        row=2,
        col=1,
    )
    fig5.add_vline(
        x=thr_lda,
        line_dash="dash",
        line_color=MODEL_COLORS["LDA"],
        line_width=2,
        row=1,
        col=1,
    )
    fig5.add_hline(
        y=thr_sel,
        line_dash="dash",
        line_color=MODEL_COLORS.get(sel, "#888888"),
        line_width=2,
        row=2,
        col=1,
    )
    fig5.add_hline(
        y=thr_sel,
        line_dash="dash",
        line_color=MODEL_COLORS.get(sel, "#888888"),
        line_width=2,
        row=2,
        col=2,
    )

    fig5.update_xaxes(title_text="LDA d-score", row=2, col=1)
    fig5.update_yaxes(title_text=f"{sel} d-score", row=2, col=1)
    fig5.update_layout(
        height=640,
        barmode="overlay",
        title_text=f"LDA ({n_lda:,} IDs) vs {sel} ({n_sel:,} IDs) @ 1% FDR",
        legend=dict(title="Label", orientation="h", y=1.03),
    )
    st.plotly_chart(fig5, use_container_width=True)

    r, p_corr = scipy.stats.pearsonr(all_scores["LDA"], all_scores[sel])
    st.metric(f"Pearson r (LDA vs {sel})", f"{r:.4f}", help=f"p = {p_corr:.2e}")


@st.fragment
def render_stage_5() -> None:
    """Render Stage 5 inside a fragment; the nested comparison fragment handles the dropdown."""
    if not st.session_state.s4_done:
        return

    st.markdown("---")
    st.subheader("Model Comparison — LDA Baseline vs Others")

    s5_btn = st.button(
        "▶ Pre-compute Comparison Data",
        type="primary",
        disabled=st.session_state.s5_done,
        key="s5_precompute_btn",
    )

    if s5_btn and not st.session_state.s5_done:
        feat_df = st.session_state.feat_df
        all_scores = st.session_state.all_scores

        if "LDA" not in all_scores:
            st.warning("LDA scores required for comparison — run Stage 2 first.")
            return

        is_decoy = feat_df["decoy"].to_numpy()
        lda_sc = all_scores["LDA"]
        scatter_cache = {}
        progress = st.progress(0)
        others = [nm for nm in all_scores if nm != "LDA"]
        try:
            res_lda = compute_full_stats(lda_sc, feat_df)
            thr_lda = res_lda["threshold"]
            n_lda = res_lda["n_ids_1pct"]
        except Exception:
            thr_lda, n_lda = float(lda_sc.max()), 0

        for k, nm in enumerate(others):
            sc2 = all_scores[nm]
            try:
                res2 = compute_full_stats(sc2, feat_df)
                thr2 = res2["threshold"]
                n2 = res2["n_ids_1pct"]
            except Exception:
                thr2, n2 = float(sc2.max()), 0

            t_mask = is_decoy == 0
            d_mask = is_decoy == 1
            marg_data = {}
            for mask, label in [(t_mask, "Target"), (d_mask, "Decoy")]:
                x_vals = lda_sc[mask]
                y_vals = sc2[mask]
                h_x, e_x = np.histogram(x_vals, bins=60, density=True)
                h_y, e_y = np.histogram(y_vals, bins=60, density=True)
                marg_data[label] = {
                    "x": x_vals.tolist(),
                    "y": y_vals.tolist(),
                    "hx": h_x.tolist(),
                    "ex": ((e_x[:-1] + e_x[1:]) / 2).tolist(),
                    "hy": h_y.tolist(),
                    "ey": ((e_y[:-1] + e_y[1:]) / 2).tolist(),
                }
            scatter_cache[nm] = {
                "marg_data": marg_data,
                "thr_lda": thr_lda,
                "thr_other": thr2,
                "n_lda": n_lda,
                "n_other": n2,
            }
            progress.progress(int(100 * (k + 1) / len(others)))

        st.session_state.scatter_cache = scatter_cache
        st.session_state.s5_done = True
        st.rerun()

    if not st.session_state.s5_done:
        return

    _render_comparison(st.session_state.scatter_cache, st.session_state.all_scores)


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

render_stage_2()

render_stage_3()

render_stage_4()

render_stage_5()

# -----------------------------------------------------------------------------
# References

st.markdown("---")
st.subheader("References")
st.markdown(
    """
1. **Käll L, Canterbury JD, Weston J, Noble WS, MacCoss MJ.**
   Semi-supervised learning for peptide identification from shotgun proteomics datasets.
   *Nat Methods.* 2007;4(11):923–925. https://doi.org/10.1038/nmeth1113

2. **Storey JD, Tibshirani R.**
   Statistical significance for genome-wide studies.
   *Proc Natl Acad Sci USA.* 2003;100(16):9440–9445. https://doi.org/10.1073/pnas.1530509100

3. **Elias JE, Gygi SP.**
   Target-decoy search strategy for increased confidence in large-scale protein
   identifications by mass spectrometry. *Nat Methods.* 2007;4(3):207–214.
   https://doi.org/10.1038/nmeth1019

4. **Levitsky LI, Ivanov MV, Lobas AA, Gorshkov MV.**
   Unbiased false discovery rate estimation for shotgun proteomics.
   *J Proteome Res.* 2017;16(2):393–397. https://doi.org/10.1021/acs.jproteome.6b00144

5. **Röst HL et al.**
   OpenSWATH enables automated, targeted analysis of data-independent acquisition MS data.
   *Nat Biotechnol.* 2014;32(3):219–223. https://doi.org/10.1038/nbt.2841
"""
)
