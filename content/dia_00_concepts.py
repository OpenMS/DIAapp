from __future__ import annotations

import base64
import io

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import streamlit as st
from src.common.common import page_setup

page_setup()

# -----------------------------------
#   Helpers

def fig_to_st(fig, caption: str | None = None, dpi: int = 130) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    st.image(buf, caption=caption, use_container_width=True)
    plt.close(fig)


def load_gif_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
        
# -----------------------------------
#   Page content

st.title("Data-Independent Acquisition (DIA) — Core Concepts")
st.markdown(
    """
This page introduces the key concepts, terminology, and data structures behind
**SWATH-MS / Data-Independent Acquisition (DIA)** proteomics. Working through
these concepts will prepare you for the hands-on targeted data-extraction
section on the next page.
"""
)

# ------------------------------------
#   Brief MS Concepts

st.markdown("---")
st.subheader("Mass Spectrometry Based Proteomics: A Brief Primer")

st.markdown(
    """
In a typical **bottom-up proteomics** experiment, proteins extracted from a
biological sample are digested with a protease (most commonly *trypsin*) into
**peptides**. These peptides are separated by **liquid chromatography (LC)**
and then ionised and detected by a **mass spectrometer (MS)**.

A mass spectrometer measures the **mass-to-charge ratio (m/z)** of ions. For
peptides, two scan types are particularly informative:

| Scan type | What it measures |
|---|---|
| **MS1** (precursor scan) | Intact peptide ions eluting from the LC column |
| **MS2** (fragment scan) | Ions produced by fragmenting a selected precursor |

When a peptide is fragmented in the MS2, it breaks preferentially along the
**peptide backbone**. The dominant ion series are **b-ions** (N-terminal
fragments, retaining the N-terminus) and **y-ions** (C-terminal fragments,
retaining the C-terminus). Other ion types also exist — including **a-ions**
(b-ions minus CO), **x-ions** (y-ions plus CO₂), and **z-ions** (y-ions minus
NH₃) — but b- and y-ions are usually the most common and are the standard choice
for spectral libraries and DIA targeted extraction. Together, the pattern of b/y
ions forms a characteristic **fragment ion spectrum** that can be used to
identify the peptide sequence.
"""
)

# ===================================
#   Peptide Fragmentation Example

SEQUENCE = list("WNQLQAFWGTGK")
N = len(SEQUENCE)

B_COLOR = "#2266CC"
Y_COLOR = "#CC2222"

FIG_W   = 14.5
SEQ_Y   = 2.5      # y-level of amino-acid letters (backbone centre)
TICK_UP = 0.55     # vertical tick length above backbone for y-ion brackets
TICK_DN = 0.55     # vertical tick length below backbone for b-ion brackets
LBL_OFF = 0.28     # extra gap between tick end and ion label

# x-positions for residues and cleavage bonds
xs     = np.linspace(1.2, FIG_W - 1.2, N)
x_cuts = [(xs[i] + xs[i + 1]) / 2 for i in range(N - 1)]

# N-terminus and C-terminus horizontal extent
x_nterm = xs[0]  - 0.55   # left edge of the backbone line
x_cterm = xs[-1] + 0.55   # right edge of the backbone line

# Thin backbone line connecting all residues
Y_BACK_UP = SEQ_Y + 0.30   # where y-ion tick starts (just above letters)
Y_BACK_DN = SEQ_Y - 0.30   # where b-ion tick starts (just below letters)

fig_ladder, ax_l = plt.subplots(figsize=(FIG_W, 4.0))
ax_l.set_xlim(-1.5, FIG_W)
ax_l.set_ylim(-0.2, 5.4)
ax_l.axis("off")


# Residue letters and thin backbone dashes 
for aa, x in zip(SEQUENCE, xs):
    ax_l.text(x, SEQ_Y, aa, ha="center", va="center",
              fontsize=15, fontweight="bold", color="#111111",
              fontfamily="monospace", zorder=4)
for i in range(N - 1):
    ax_l.plot([xs[i] + 0.30, xs[i + 1] - 0.30], [SEQ_Y, SEQ_Y],
              color="#BBBBBB", lw=1.0, zorder=1)

# b-ion brackets (inverted-L, blue): vertical DOWN then horizontal LEFT
b_labels_shown = {1, 2, 3, 4, 10}
for i, x_cut in enumerate(x_cuts, start=1):
    y_top  = Y_BACK_DN            # start of tick (just below letters)
    y_bot  = y_top  - TICK_DN     # end of tick (bottom of bracket)
    # Vertical tick downward
    ax_l.plot([x_cut, x_cut], [y_top, y_bot], color=B_COLOR, lw=1.5, zorder=3)
    # Horizontal leg going LEFT to N-terminus
    ax_l.plot([x_nterm, x_cut], [y_bot, y_bot], color=B_COLOR, lw=1.5, zorder=3)
    # Label
    if i in b_labels_shown:
        ax_l.text(x_cut, y_bot - LBL_OFF, f"b{i}",
                  ha="center", va="top", fontsize=9.5,
                  color=B_COLOR, fontweight="bold")

# y-ion brackets (L-shape, red): vertical UP then horizontal RIGHT 
for j in range(1, N):
    # y_j cleavage is after the (N-j)-th residue
    x_cut  = x_cuts[N - 1 - j]
    y_bot2 = Y_BACK_UP             # start of tick (just above letters)
    y_top2 = y_bot2 + TICK_UP      # end of tick (top of bracket)
    # Vertical tick upward
    ax_l.plot([x_cut, x_cut], [y_bot2, y_top2], color=Y_COLOR, lw=1.5, zorder=3)
    # Horizontal leg going RIGHT to C-terminus
    ax_l.plot([x_cut, x_cterm], [y_top2, y_top2], color=Y_COLOR, lw=1.5, zorder=3)
    # Label
    ax_l.text(x_cut, y_top2 + LBL_OFF, f"y{j}",
              ha="center", va="bottom", fontsize=9.5,
              color=Y_COLOR, fontweight="bold")

# Legend
ax_l.text(-1.3, Y_BACK_UP + TICK_UP + LBL_OFF, "y-ions (C-terminal)",
          ha="left", va="bottom", fontsize=9, color=Y_COLOR, fontweight="bold")
ax_l.text(-1.3, Y_BACK_DN - TICK_DN - LBL_OFF, "b-ions (N-terminal)",
          ha="left", va="top", fontsize=9, color=B_COLOR, fontweight="bold")

ax_l.set_title("Peptide fragment ion nomenclature — b/y ladder for WNQLQAFWGTGK",
               fontsize=10, pad=6)
fig_ladder.tight_layout()
fig_to_st(
    fig_ladder,
    caption=(
        "Ladder diagram for the peptide WNQLQAFWGTGK. b-ions (blue, N-terminal) "
        "and y-ions (red, C-terminal) arise from backbone cleavage at each peptide bond. "
    ),
)

# =====================================
#   Annotated MS2 spectrum

# We simulate the MS2 spectrum for WNQLQAFWGTGK (z=2, MW~1409 Da)
# b-ion masses (singly charged): b1–b11 for WNQLQAFWGTGK
AA_MASS = {
    "G": 57.021, "A": 71.037, "V": 99.068, "L": 113.084, "I": 113.084,
    "P": 97.053, "F": 147.068, "W": 186.079, "M": 131.040, "S": 87.032,
    "T": 101.048, "C": 103.009, "Y": 163.063, "H": 137.059, "D": 115.027,
    "E": 129.043, "N": 114.043, "Q": 128.059, "K": 128.095, "R": 156.101,
}
H    = 1.00728
seq  = "WNQLQAFWGTGK"

b_masses = []
running = 0.0
for aa in seq[:-1]:
    running += AA_MASS[aa]
    b_masses.append(running + H)          # b-ion m/z (z=1)

y_masses = []
running = 0.0
for aa in reversed(seq[1:]):
    running += AA_MASS[aa]
    y_masses.append(running + H + 18.011) # y-ion m/z (z=1)
y_masses = y_masses[::-1]                 # y1 first

# Assign relative intensities (simulated, high for mid-series)
rng_sp = np.random.default_rng(17)
n_b = len(b_masses); n_y = len(y_masses)
b_ints = np.clip(rng_sp.normal(0.55, 0.22, n_b) + 0.12 * np.arange(n_b), 0.1, 1.0)
y_ints = np.clip(rng_sp.normal(0.60, 0.25, n_y) + 0.08 * np.arange(n_y)[::-1], 0.1, 1.0)
b_ints /= max(b_ints.max(), y_ints.max())
y_ints /= max(b_ints.max(), y_ints.max())

# Add noise peaks
n_noise = 35
noise_mz  = rng_sp.uniform(80, 1280, n_noise)
noise_int = rng_sp.exponential(0.08, n_noise)
noise_int  = np.clip(noise_int, 0.01, 0.22)

fig_spec, ax_sp = plt.subplots(figsize=(12, 4.2))

# Noise peaks
for mz, h in zip(noise_mz, noise_int):
    ax_sp.plot([mz, mz], [0, h], color="#AAAAAA", lw=1.0, zorder=1)

# b-ion peaks
for i, (mz, h) in enumerate(zip(b_masses, b_ints)):
    ax_sp.plot([mz, mz], [0, h], color=B_COLOR, lw=2.0, zorder=3)
    if h > 0.25:
        ax_sp.text(mz, h + 0.025, f"b{i+1}+", ha="center", va="bottom",
                   fontsize=7.5, color=B_COLOR, fontweight="bold")

# y-ion peaks
for i, (mz, h) in enumerate(zip(y_masses, y_ints)):
    ax_sp.plot([mz, mz], [0, h], color=Y_COLOR, lw=2.0, zorder=3)
    if h > 0.25:
        ax_sp.text(mz, h + 0.025, f"y{i+1}+", ha="center", va="bottom",
                   fontsize=7.5, color=Y_COLOR, fontweight="bold")

b_patch = mpatches.Patch(color=B_COLOR, label="b-ions (matched)")
y_patch = mpatches.Patch(color=Y_COLOR, label="y-ions (matched)")
n_patch = mpatches.Patch(color="#AAAAAA", label="Unmatched / noise")
ax_sp.legend(handles=[b_patch, y_patch, n_patch], fontsize=9, loc="upper right")
ax_sp.set_xlabel("m/z", fontsize=11)
ax_sp.set_ylabel("Relative Intensity", fontsize=11)
ax_sp.set_title("Simulated MS2 spectrum — WNQLQAFWGTGK (z = 2)", fontsize=11)
ax_sp.set_ylim(-0.03, 1.18)
ax_sp.set_xlim(50, 1350)
ax_sp.axhline(0, color="black", lw=0.8)
ax_sp.spines["top"].set_visible(False)
ax_sp.spines["right"].set_visible(False)
fig_spec.tight_layout()
fig_to_st(
    fig_spec,
    caption=(
        "Simulated MS2 spectrum for WNQLQAFWGTGK. Matched b-ions are shown in blue, "
        "y-ions in red; unmatched peaks (grey) represent co-fragmented peptides or "
        "chemical noise — common in real DIA data where multiple precursors are "
        "fragmented together. Identifying a peptide from this spectrum requires matching "
        "the observed ion pattern against a spectral library or database."
    ),
)







