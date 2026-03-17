from __future__ import annotations

import io

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import streamlit as st

from src.common.common import page_setup

matplotlib.rcParams["font.family"] = "DejaVu Sans"

page_setup()

# -----------------------------------
#   Helpers


def fig_to_st(fig, caption: str | None = None, dpi: int = 130) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    st.image(buf, caption=caption, use_container_width=True)
    plt.close(fig)


def gauss(t, apex, sigma, amp=1.0):
    """Pure Gaussian chromatographic peak."""
    return amp * np.exp(-((t - apex) ** 2) / (2 * sigma**2))


def cross_correlation(x, y):
    """
    Normalised cross-correlation (delay=0 is central index).
    Returns (max_corr, optimal_delay_index).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sx = x.std()
    sy = y.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0, 0
    xn = (x - x.mean()) / sx
    yn = (y - y.mean()) / sy
    cc = np.correlate(xn, yn, mode="full")
    cc /= len(x)
    best_idx = int(np.argmax(cc))
    delay = best_idx - (len(x) - 1)
    return float(cc[best_idx]), delay


def xcorr_shape_score(xics: list[np.ndarray]) -> float:
    """VAR_XCORR_SHAPE — mean of all pairwise max-cross-correlations."""
    n = len(xics)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            v, _ = cross_correlation(xics[i], xics[j])
            vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


def xcorr_coelution_score(xics: list[np.ndarray]) -> float:
    """VAR_XCORR_COELUTION — mean(delays) + std(delays)."""
    n = len(xics)
    delays = []
    for i in range(n):
        for j in range(i + 1, n):
            _, d = cross_correlation(xics[i], xics[j])
            delays.append(abs(d))
    if not delays:
        return 0.0
    return float(np.mean(delays) + np.std(delays))


def spectral_angle(lib: np.ndarray, obs: np.ndarray) -> float:
    """Normalised spectral angle ∈ [0,1].  1 = perfect match."""
    lib = np.asarray(lib, dtype=float)
    obs = np.asarray(obs, dtype=float)
    nl = np.linalg.norm(lib)
    no = np.linalg.norm(obs)
    if nl < 1e-12 or no < 1e-12:
        return 0.0
    cos_theta = np.dot(lib / nl, obs / no)
    cos_theta = float(np.clip(cos_theta, -1, 1))
    return 1.0 - (2.0 * np.arccos(cos_theta) / np.pi)


def dot_product(lib: np.ndarray, obs: np.ndarray) -> float:
    """Square-root + L2 normalised dot product."""
    lib_s = np.sqrt(np.maximum(lib, 0))
    obs_s = np.sqrt(np.maximum(obs, 0))
    nl = np.linalg.norm(lib_s)
    no = np.linalg.norm(obs_s)
    if nl < 1e-12 or no < 1e-12:
        return 0.0
    return float(np.dot(lib_s / nl, obs_s / no))


def manhattan_score(lib: np.ndarray, obs: np.ndarray) -> float:
    """Sqrt-normalised Manhattan distance (lower = better match)."""
    lib_s = np.sqrt(np.maximum(lib, 0))
    obs_s = np.sqrt(np.maximum(obs, 0))
    sl = lib_s.sum()
    so = obs_s.sum()
    if sl < 1e-12 or so < 1e-12:
        return 1.0
    return float(np.sum(np.abs(lib_s / sl - obs_s / so)))


def pearson_corr(lib: np.ndarray, obs: np.ndarray) -> float:
    """Pearson correlation between library and observed intensity vectors."""
    if np.std(lib) < 1e-12 or np.std(obs) < 1e-12:
        return 0.0
    return float(np.corrcoef(lib, obs)[0, 1])


def rmsd_score(lib: np.ndarray, obs: np.ndarray) -> float:
    """Normalised RMSD (lower = better)."""
    mu_l = lib.mean()
    mu_o = obs.mean()
    if mu_l < 1e-12 or mu_o < 1e-12:
        return 1.0
    return float(np.sqrt(np.mean(np.abs(obs / mu_o - lib / mu_l))))


def score_badge(value: float, good_high: bool = True) -> str:
    """Return a coloured Streamlit metric delta string."""
    return "↑" if good_high else "↓"


# -----------------------------------
#   Page content

st.title("Feature Scoring in DIA")
st.markdown(
    """
During targeted data extraction and peak picking, we identify candidate peak groups that may correspond to the peptides in our spectral library. During this process, we compute a variety of scores that quantify how well the observed data matches our expectations based on the library. These scores are crucial for distinguishing true peptide detections from false positives during downstream statistical validation.

This page walks through a variety of computed scores used by
[OpenSWATH](https://openswath.org) with interactive controls so you can
**see how each score responds** when chromatographic quality, library match, or
mass accuracy changes.
"""
)

st.info(
    "**How to use this page:** Each section has sliders that simulate different "
    "data quality scenarios. Adjust them and watch the scores update in real time."
)

# --------------------------------
# Scoring pipeline overview

st.markdown("---")
st.subheader("The OpenSWATH Scoring Pipeline")

st.markdown(
    """
`scorePeakgroups()` is the central driver that computes a few main categories of scores for each candidate peak group detected in the XICs:
"""
)

SCORE_CATS = [
    ("Chromatographic", "#4A90D9", "XIC shape & coelution\ncross-correlation"),
    ("Library", "#5BAD72", "Spectral angle, dot-product\nManhattan, RMSD, Pearson RT"),
    (
        "Single Spectrum\n(DIA)",
        "#E8A838",
        "Isotope correlation\nmass deviation, b/y ions",
    ),
    ("Ion Mobility", "#9B59B6", "Ion mobility drift time\ncorrelation and deviation"),
    ("UIS", "#D95A4A", "Unique ion signature\n(PTM identification)"),
    # Not really used as much currently
    # ("Intensity", "#16A085", "Peak group intensity\nvs total XIC"),
    # ("Elution\nModel", "#7F8C8D", "EMG peak shape\nfit quality"),
]

fig_ov, ax_ov = plt.subplots(figsize=(14, 2.8))
ax_ov.set_xlim(0, 14)
ax_ov.set_ylim(0, 2.8)
ax_ov.axis("off")
box_w = 1.65
box_h = 1.35
xs_ov = np.linspace(0.9, 13.1, len(SCORE_CATS))

for i, ((label, color, sub), x) in enumerate(zip(SCORE_CATS, xs_ov)):
    ax_ov.add_patch(
        mpatches.FancyBboxPatch(
            (x - box_w / 2, 0.75),
            box_w,
            box_h,
            boxstyle="round,pad=0.10",
            facecolor=color,
            edgecolor="white",
            lw=1.5,
            alpha=0.88,
            zorder=3,
        )
    )
    ax_ov.text(
        x,
        1.44,
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="white",
        zorder=4,
        linespacing=1.25,
    )
    ax_ov.text(
        x,
        0.50,
        sub,
        ha="center",
        va="top",
        fontsize=7,
        color="#444444",
        linespacing=1.25,
        zorder=4,
    )

ax_ov.text(
    7.0,
    2.65,
    "OpenSWATH scorePeakgroups()- scoring categories",
    ha="center",
    va="top",
    fontsize=10.5,
    fontweight="bold",
)
fig_ov.tight_layout()
fig_to_st(fig_ov)

st.markdown(
    """
All scores are fed into **PyProphet** (a semi-supervised machine learning
classifier) which learns a linear discriminant that maximally separates
target from decoy peak groups.  The resulting discriminant score is used for
**FDR control** via the target-decoy method.
"""
)

# --------------------------------
# 2. Chromatographic Scores

st.markdown("---")
st.subheader("Chromatographic Scores")

st.markdown(
    r"""
Chromatographic scores measure how well the transition XICs **co-elute** and
share the same **peak shape**.  They are based on pairwise normalised
cross-correlations between every pair of transitions.

For each pair $(x, y)$ the cross-correlation at delay $d$ is:

$$Z[d] = \sum_{i=0}^{N-1} x_i\, y_{i+d}$$

where $x$ and $y$ are **standardised** (zero-mean, unit-variance) intensity
vectors.  The **maximum** $Z[d]$ gives the shape score, and the **delay $d$**
at that maximum gives the coelution score.

| Score | Definition | Good detection |
|---|---|---|
| `VAR_XCORR_SHAPE` | Mean of all pairwise max cross-correlations | → 1 |
| `VAR_XCORR_SHAPE_WEIGHTED` | Library-intensity weighted mean | → 1 |
| `VAR_XCORR_COELUTION` | Mean(delays) + SD(delays) | → 0 |
| `VAR_XCORR_COELUTION_WEIGHTED` | Library-intensity weighted | → 0 |
"""
)

st.markdown("#### Chromatographic Quality Demo")

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    rt_offset_max = st.slider(
        "Max RT offset between transitions (scans)",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Simulates how much transitions are staggered in RT. "
        "0 = perfect co-elution; higher = worse coelution score.",
    )
    noise_level = st.slider(
        "Noise level (% of peak height)",
        min_value=0,
        max_value=80,
        value=10,
        step=5,
        help="Gaussian noise added to each transition. "
        "High noise degrades both shape and coelution scores.",
    )
with col_ctrl2:
    n_transitions = st.slider(
        "Number of transitions",
        min_value=3,
        max_value=8,
        value=5,
        step=1,
    )
    apex_sigma = st.slider(
        "Peak width (sigma, scans)",
        min_value=2,
        max_value=15,
        value=5,
        step=1,
        help="Narrower peaks are harder to align → worse coelution.",
    )

# Simulate XICs
rng_xic = np.random.default_rng(42)
t = np.arange(0, 100)
APEX = 50.0
FRAG_COLORS = [
    "#CC1111",
    "#1199CC",
    "#22AA33",
    "#DD8800",
    "#8833CC",
    "#AA1177",
    "#117733",
    "#443399",
]
LIB_AMPS = np.array([1.00, 0.72, 0.55, 0.40, 0.28, 0.22, 0.18, 0.14])

xics_demo = []
offsets_used = []
for k in range(n_transitions):
    offset = (
        rng_xic.integers(-rt_offset_max, rt_offset_max + 1) if rt_offset_max > 0 else 0
    )
    offsets_used.append(offset)
    peak = gauss(t, APEX + offset, apex_sigma, amp=LIB_AMPS[k])
    noise = rng_xic.normal(0, noise_level / 100.0 * LIB_AMPS[k], len(t))
    xics_demo.append(np.maximum(peak + noise, 0))

# Compute scores
shape_score = xcorr_shape_score(xics_demo)
coelution_score = xcorr_coelution_score(xics_demo)

# Plot
fig_xic, axes_x = plt.subplots(
    2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
)

for k, (xic, col, off) in enumerate(
    zip(xics_demo, FRAG_COLORS[:n_transitions], offsets_used)
):
    axes_x[0].plot(
        t, xic, color=col, lw=1.8, alpha=0.85, label=f"y{k + 2} (offset={off:+d})"
    )
    axes_x[0].axvline(APEX + off, color=col, lw=0.9, linestyle=":", alpha=0.55)

axes_x[0].axvline(
    APEX, color="black", lw=2, linestyle="--", alpha=0.7, label="Expected apex"
)
axes_x[0].set_ylabel("Intensity", fontsize=10)
axes_x[0].set_title("Extracted Ion Chromatograms (XICs)", fontsize=11)
axes_x[0].legend(fontsize=8, ncol=4, loc="upper right")
axes_x[0].spines["top"].set_visible(False)
axes_x[0].spines["right"].set_visible(False)

# Summed XIC
summed = np.sum(xics_demo, axis=0)
axes_x[1].fill_between(t, summed, alpha=0.3, color="steelblue")
axes_x[1].plot(t, summed, color="navy", lw=1.8)
axes_x[1].axvline(APEX, color="black", lw=2, linestyle="--", alpha=0.7)
axes_x[1].set_ylabel("Summed", fontsize=10)
axes_x[1].set_xlabel("Retention Time (scans)", fontsize=10)
axes_x[1].spines["top"].set_visible(False)
axes_x[1].spines["right"].set_visible(False)

fig_xic.tight_layout()
fig_to_st(fig_xic)

# Score display
m1, m2 = st.columns(2)
m1.metric(
    "VAR_XCORR_SHAPE",
    f"{shape_score:.3f}",
    delta="↑ good (→ 1.0)" if shape_score > 0.7 else "↓ poor",
    delta_color="normal" if shape_score > 0.7 else "inverse",
    help="Mean of all pairwise max cross-correlations. Perfect co-elution = 1.",
)
m2.metric(
    "VAR_XCORR_COELUTION",
    f"{coelution_score:.2f}",
    delta="↑ poor (→ 0 is best)" if coelution_score > 1 else "↓ good",
    delta_color="inverse" if coelution_score > 1 else "normal",
    help="Mean(delays) + SD(delays). Perfect alignment = 0.",
)

with st.expander("How cross-correlation works — code"):
    st.code(
        '''def cross_correlation(x, y):
    """Normalised cross-correlation between two XIC arrays."""
    xn = (x - x.mean()) / x.std()   # standardise
    yn = (y - y.mean()) / y.std()
    cc = np.correlate(xn, yn, mode="full") / len(x)
    best_idx = np.argmax(cc)
    delay    = best_idx - (len(x) - 1)   # 0 = perfectly aligned
    return cc[best_idx], delay            # (shape_val, coelution_delay)

# VAR_XCORR_SHAPE    = mean of cc[best_idx] across all pairs
# VAR_XCORR_COELUTION = mean(|delays|) + std(|delays|) across all pairs''',
        language="python",
    )

# --------------------------------
#   Library Scores

st.markdown("---")
st.subheader("Library Scores")

st.markdown(
    r"""
Library scores compare the **observed peak-apex fragment intensities** against
the reference intensities stored in the spectral library.  A true detection
should produce intensities that closely mirror the library spectrum.

| Score | Formula (simplified) | Good detection |
|---|---|---|
| `VAR_LIBRARY_SANGLE` | $1 - \frac{2}{\pi}\arccos\!\left(\frac{b \cdot x}{\|b\|\|x\|}\right)$ | → 1 |
| `VAR_LIBRARY_DOTPROD` | $\frac{\sqrt{b}}{\|\sqrt{b}\|} \cdot \frac{\sqrt{x}}{\|\sqrt{x}\|}$ | → 1 |
| `VAR_LIBRARY_CORR` | Pearson $r(b, x)$ | → 1 |
| `VAR_LIBRARY_MANHATTAN` | $\sum_i \|\frac{\sqrt{b_i}}{\sum\sqrt{b}} - \frac{\sqrt{x_i}}{\sum\sqrt{x}}\|$ | → 0 |
| `VAR_LIBRARY_RMSD` | $\sqrt{\frac{1}{N}\sum_i\left(\frac{x_i}{\mu_x} - \frac{b_i}{\mu_b}\right)^2}$ | → 0 |

$b$ = library intensities, $x$ = observed intensities.
"""
)

st.markdown("#### Library Match Quality")

col_lib1, col_lib2 = st.columns(2)
with col_lib1:
    intensity_noise_pct = st.slider(
        "Intensity perturbation (% random deviation)",
        min_value=0,
        max_value=100,
        value=15,
        step=5,
        help="How much each observed fragment intensity deviates from the library. "
        "0% = perfect match; 100% = fully random.",
    )
with col_lib2:
    n_missing_frags = st.slider(
        "Missing/interfered transitions",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        help="Transitions with no signal (e.g. from co-eluting interference).",
    )

# Library intensities
N_FRAGS = 7
frag_labels = [f"y{i + 2}" for i in range(N_FRAGS)]
frag_mzs = [300 + i * 120 for i in range(N_FRAGS)]
lib_ints = np.array([1.00, 0.75, 0.88, 0.55, 0.42, 0.30, 0.20])

# Observed: perturb library + zero out missing
rng_lib = np.random.default_rng(17)
obs_ints = lib_ints.copy()
obs_ints += rng_lib.normal(0, intensity_noise_pct / 100.0, N_FRAGS)
obs_ints = np.maximum(obs_ints, 0.01)
for i in range(min(n_missing_frags, N_FRAGS)):
    obs_ints[-(i + 1)] = rng_lib.exponential(0.04)

obs_ints_norm = obs_ints / obs_ints.max()

# Scores
sa = spectral_angle(lib_ints, obs_ints_norm)
dp = dot_product(lib_ints, obs_ints_norm)
pr = pearson_corr(lib_ints, obs_ints_norm)
mh = manhattan_score(lib_ints, obs_ints_norm)
rmsd = rmsd_score(lib_ints, obs_ints_norm)

# Mirror plot
B_COLOR = "#2266CC"
Y_COLOR = "#CC2222"
fig_lib, ax_lib = plt.subplots(figsize=(11, 4.5))
x_pos = np.arange(N_FRAGS)
bar_w = 0.32

bars_lib = ax_lib.bar(
    x_pos - bar_w / 2,
    lib_ints,
    bar_w,
    color=B_COLOR,
    alpha=0.82,
    label="Library",
    edgecolor="white",
)
bars_obs = ax_lib.bar(
    x_pos + bar_w / 2,
    obs_ints_norm,
    bar_w,
    color=Y_COLOR,
    alpha=0.82,
    label="Observed (apex)",
    edgecolor="white",
)

for xi, (l, o) in zip(x_pos, zip(lib_ints, obs_ints_norm)):
    ax_lib.plot(
        [xi - bar_w / 2 + bar_w / 4, xi + bar_w / 2 + bar_w / 4],
        [l, o],
        color="#888888",
        lw=0.9,
        linestyle="--",
        alpha=0.6,
    )

ax_lib.set_xticks(x_pos)
ax_lib.set_xticklabels(frag_labels, fontsize=10)
ax_lib.set_ylabel("Relative Intensity", fontsize=10)
ax_lib.set_title("Library vs Observed Fragment Intensities", fontsize=11)
ax_lib.set_ylim(0, 1.25)
ax_lib.legend(fontsize=10)
ax_lib.spines["top"].set_visible(False)
ax_lib.spines["right"].set_visible(False)

# Spectral angle arc annotation
angle_deg = (1 - sa) * 90
ax_lib.text(
    N_FRAGS - 1 + 0.6,
    1.15,
    f"Spectral angle: {sa:.3f}\n(θ = {angle_deg:.1f}°)",
    ha="right",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEF3FF", edgecolor=B_COLOR, lw=0.8),
)

fig_lib.tight_layout()
fig_to_st(
    fig_lib,
    caption=(
        "Blue = library reference intensities. Red = observed peak-apex intensities. "
        "Dashed grey lines connect matching transitions."
    ),
)

# Score metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "VAR_LIBRARY_SANGLE", f"{sa:.3f}", help="Spectral angle ∈ [0,1]. 1 = perfect."
)
c2.metric(
    "VAR_LIBRARY_DOTPROD", f"{dp:.3f}", help="Sqrt-normalised dot product. 1 = perfect."
)
c3.metric("VAR_LIBRARY_CORR", f"{pr:.3f}", help="Pearson correlation. 1 = perfect.")
c4.metric("VAR_LIBRARY_MANHATTAN", f"{mh:.3f}", help="Manhattan distance. 0 = perfect.")
c5.metric("VAR_LIBRARY_RMSD", f"{rmsd:.3f}", help="Normalised RMSD. 0 = perfect.")

with st.expander("Score equations & code"):
    st.code(
        """def spectral_angle(lib, obs):
    cos_theta = np.dot(lib/||lib||, obs/||obs||)
    return 1 - (2/π) * arccos(cos_theta)   # 1 = perfect

def dot_product(lib, obs):
    lib_s = sqrt(lib);  obs_s = sqrt(obs)
    return dot(lib_s/||lib_s||, obs_s/||obs_s||)

def pearson_corr(lib, obs):
    return np.corrcoef(lib, obs)[0, 1]

def manhattan_score(lib, obs):
    lib_s = sqrt(lib)/sum(sqrt(lib))
    obs_s = sqrt(obs)/sum(sqrt(obs))
    return sum(|lib_s - obs_s|)           # 0 = perfect

def rmsd_score(lib, obs):
    return sqrt(mean((obs/mean(obs) - lib/mean(lib))^2))  # 0 = perfect""",
        language="python",
    )

# ---------------------------------
# Retention Time Scores

st.markdown("---")
st.subheader("Retention Time Score")

st.markdown(
    r"""
The RT score measures how much the **detected peak apex** deviates from the
**library-predicted retention time** (after iRT → experiment RT conversion).

$$\text{RT\_SCORE} = |\text{RT}_\text{exp} - \text{RT}_\text{library}|$$

$$\text{VAR\_NORM\_RT\_SCORE} = \frac{\text{RT\_SCORE}}{\text{rt\_normalization\_factor}}$$

The default normalization factor is **100 s**. A well-calibrated detection
should have `VAR_NORM_RT_SCORE` close to 0.
"""
)

col_rt1, col_rt2 = st.columns(2)
with col_rt1:
    rt_deviation = st.slider(
        "RT deviation from library prediction (s)",
        min_value=0,
        max_value=300,
        value=20,
        step=5,
        help="How far the detected apex is from the iRT-predicted RT.",
    )
    rt_norm_factor = st.slider(
        "RT normalization factor",
        min_value=50,
        max_value=300,
        value=100,
        step=10,
        help="Default = 100 s. Larger window = more permissive.",
    )

rt_score = float(rt_deviation)
var_norm_rt = rt_score / rt_norm_factor

t_rt = np.linspace(0, 400, 500)
lib_rt = 200.0
exp_rt = lib_rt + rt_deviation
peak_lib = gauss(t_rt, lib_rt, 8.0, 1.0)
peak_exp = gauss(t_rt, exp_rt, 8.0, 0.85)
win_lo = lib_rt - rt_norm_factor / 2
win_hi = lib_rt + rt_norm_factor / 2

fig_rt, ax_rt = plt.subplots(figsize=(11, 3.5))
ax_rt.fill_between(t_rt, peak_lib, alpha=0.2, color="#2266CC")
ax_rt.plot(t_rt, peak_lib, color="#2266CC", lw=2, label="Library-predicted apex")
ax_rt.fill_between(t_rt, peak_exp, alpha=0.2, color="#CC2222")
ax_rt.plot(t_rt, peak_exp, color="#CC2222", lw=2, label="Detected apex")

ax_rt.axvspan(win_lo, win_hi, alpha=0.06, color="green", label="RT extraction window")
ax_rt.axvline(lib_rt, color="#2266CC", lw=1.5, linestyle="--", alpha=0.7)
ax_rt.axvline(exp_rt, color="#CC2222", lw=1.5, linestyle="--", alpha=0.7)

if rt_deviation > 0:
    mid = (lib_rt + exp_rt) / 2
    y_ann = 0.55
    ax_rt.annotate(
        "",
        xy=(exp_rt, y_ann),
        xytext=(lib_rt, y_ann),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.4),
    )
    ax_rt.text(
        mid,
        y_ann + 0.04,
        f"Δ RT = {rt_deviation} s",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )

ax_rt.set_xlabel("Retention Time (s)", fontsize=10)
ax_rt.set_ylabel("Intensity", fontsize=10)
ax_rt.set_title("Retention Time Deviation", fontsize=11)
ax_rt.legend(fontsize=9, loc="upper right")
ax_rt.spines["top"].set_visible(False)
ax_rt.spines["right"].set_visible(False)
fig_rt.tight_layout()
fig_to_st(fig_rt)

col_rtm1, col_rtm2 = st.columns(2)
col_rtm1.metric("RT_SCORE (s)", f"{rt_score:.1f}")
col_rtm2.metric(
    "VAR_NORM_RT_SCORE",
    f"{var_norm_rt:.3f}",
    help="Smaller = closer to predicted RT. 0 = exact match.",
)

# ----------------------------------
# Spectrum Scores

st.markdown("---")
st.subheader("Spectrum Scores")

st.markdown(
    """
Single-spectrum scores use the **MS2 spectrum at the peak apex** rather than
the full XIC.  They assess the spectrum against theoretical.

**Key scores:**

| Score | What it measures |
|---|---|
| `VAR_ISOTOPE_CORRELATION_SCORE` | Pearson corr. of each fragment's isotope envelope vs theoretical averagine distribution |
| `VAR_ISOTOPE_OVERLAP_SCORE` | Evidence that the monoisotopic peak is shifted (interference from heavier species) |
| `VAR_MASSDEV_SCORE` | Mean ppm deviation of observed fragment m/z from theoretical |
| `VAR_BSERIES_SCORE` | Count of b-ions detected in the spectrum above intensity threshold |
| `VAR_YSERIES_SCORE` | Count of y-ions detected in the spectrum above intensity threshold |
| `VAR_MANHATTAN_SCORE` | Manhattan distance between observed and theoretical isotope distributions |
| `VAR_DOTPROD_SCORE` | Dot product between observed and theoretical isotope distributions |
"""
)

st.markdown("#### Spectral Quality at Peak Apex")

col_spec1, col_spec2 = st.columns(2)
with col_spec1:
    mass_error_ppm = st.slider(
        "Mass accuracy (ppm error)",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        help="Higher = worse mass accuracy → higher VAR_MASSDEV_SCORE.",
    )
    isotope_quality = st.slider(
        "Isotope pattern match (%)",
        min_value=20,
        max_value=100,
        value=85,
        step=5,
        help="How closely the observed isotope envelope matches averagine. "
        "100% = perfect theoretical match.",
    )
with col_spec2:
    n_b_ions = st.slider(
        "b-ions detected in apex spectrum",
        min_value=0,
        max_value=8,
        value=4,
        step=1,
    )
    n_y_ions = st.slider(
        "y-ions detected in apex spectrum",
        min_value=0,
        max_value=10,
        value=6,
        step=1,
    )

# Simulate isotope envelopes
rng_iso = np.random.default_rng(55)
n_isotopes = 5
theoretical_iso = np.array([0.55, 0.30, 0.11, 0.03, 0.01])  # averagine
obs_iso = theoretical_iso.copy()
noise_iso = rng_iso.normal(0, (100 - isotope_quality) / 100.0 * 0.15, n_isotopes)
obs_iso = np.maximum(obs_iso + noise_iso, 0)
obs_iso /= obs_iso.sum()

# Isotope correlation score (Pearson)
iso_corr = pearson_corr(theoretical_iso, obs_iso)

# Mass deviation score (simulated)
mass_dev_score = mass_error_ppm / 10.0  # scaled

fig_iso, axes_iso = plt.subplots(1, 2, figsize=(12, 4.0))

#  Left: isotope pattern
iso_labels = ["M", "M+1", "M+2", "M+3", "M+4"]
x_iso = np.arange(n_isotopes)
axes_iso[0].bar(
    x_iso - 0.2,
    theoretical_iso,
    0.38,
    color="#2266CC",
    alpha=0.80,
    label="Theoretical (averagine)",
    edgecolor="white",
)
axes_iso[0].bar(
    x_iso + 0.2,
    obs_iso,
    0.38,
    color="#CC2222",
    alpha=0.80,
    label="Observed",
    edgecolor="white",
)
axes_iso[0].set_xticks(x_iso)
axes_iso[0].set_xticklabels(iso_labels, fontsize=10)
axes_iso[0].set_ylabel("Relative Intensity", fontsize=10)
axes_iso[0].set_title("Isotope Pattern: Theoretical vs Observed", fontsize=10.5)
axes_iso[0].legend(fontsize=9)
axes_iso[0].spines["top"].set_visible(False)
axes_iso[0].spines["right"].set_visible(False)

#  Right: apex spectrum with b/y annotations
mz_axis = np.linspace(100, 1300, 800)
spectrum = np.zeros_like(mz_axis)

# Add simulated fragment peaks
AA_MASSES = [
    57.0,
    71.0,
    99.1,
    113.1,
    128.1,
    147.1,
    156.1,
    163.1,
    128.1,
    114.0,
    101.0,
    131.0,
]
b_mzs = [
    200
    + i * 110
    + rng_iso.normal(0, mass_error_ppm * 0.01 * (200 + i * 110) / 1e6 * 1e6)
    for i in range(min(n_b_ions, 8))
]
y_mzs = [
    250
    + i * 115
    + rng_iso.normal(0, mass_error_ppm * 0.01 * (250 + i * 115) / 1e6 * 1e6)
    for i in range(min(n_y_ions, 10))
]

for mz in b_mzs:
    idx = np.argmin(np.abs(mz_axis - mz))
    h = rng_iso.uniform(0.3, 1.0)
    axes_iso[1].plot([mz, mz], [0, h], color="#2266CC", lw=2.0, zorder=3)
    if h > 0.55:
        axes_iso[1].text(
            mz,
            h + 0.02,
            f"b{b_mzs.index(mz) + 1}",
            ha="center",
            fontsize=7.5,
            color="#2266CC",
            fontweight="bold",
        )

for mz in y_mzs:
    h = rng_iso.uniform(0.25, 0.95)
    axes_iso[1].plot([mz, mz], [0, h], color="#CC2222", lw=2.0, zorder=3)
    if h > 0.50:
        axes_iso[1].text(
            mz,
            h + 0.02,
            f"y{y_mzs.index(mz) + 1}",
            ha="center",
            fontsize=7.5,
            color="#CC2222",
            fontweight="bold",
        )

# Noise
for _ in range(20):
    nmz = rng_iso.uniform(120, 1250)
    nh = rng_iso.exponential(0.08)
    axes_iso[1].plot([nmz, nmz], [0, nh], color="#AAAAAA", lw=1.0, zorder=1)

axes_iso[1].set_xlabel("m/z (Da)", fontsize=10)
axes_iso[1].set_ylabel("Relative Intensity", fontsize=10)
axes_iso[1].set_title(
    f"Apex MS2 Spectrum (mass error ±{mass_error_ppm} ppm)", fontsize=10.5
)
axes_iso[1].set_ylim(-0.03, 1.20)
axes_iso[1].axhline(0, color="black", lw=0.7)

b_patch = mpatches.Patch(color="#2266CC", label=f"b-ions ({n_b_ions} detected)")
y_patch = mpatches.Patch(color="#CC2222", label=f"y-ions ({n_y_ions} detected)")
n_patch = mpatches.Patch(color="#AAAAAA", label="Noise / unmatched")
axes_iso[1].legend(handles=[b_patch, y_patch, n_patch], fontsize=8)
axes_iso[1].spines["top"].set_visible(False)
axes_iso[1].spines["right"].set_visible(False)

fig_iso.tight_layout()
fig_to_st(fig_iso)

c1s, c2s, c3s, c4s, c5s = st.columns(5)
c1s.metric(
    "VAR_ISOTOPE_CORRELATION",
    f"{iso_corr:.3f}",
    help="Pearson corr. between observed and theoretical isotope envelope. 1 = perfect.",
)
c2s.metric(
    "VAR_MASSDEV_SCORE (ppm)",
    f"{mass_error_ppm}",
    help="Mean ppm deviation of fragment ion m/z. 0 = perfect.",
)
c3s.metric(
    "VAR_BSERIES_SCORE",
    f"{n_b_ions}",
    help="Number of b-ions detected above intensity threshold.",
)
c4s.metric(
    "VAR_YSERIES_SCORE",
    f"{n_y_ions}",
    help="Number of y-ions detected above intensity threshold.",
)
iso_manhattan = float(np.sum(np.abs(obs_iso - theoretical_iso)))
c5s.metric(
    "VAR_MANHATTAN_SCORE",
    f"{iso_manhattan:.3f}",
    help="Manhattan distance between observed and theoretical isotope distributions.",
)
