from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pyopenms as poms
import pyopenms_viz  # noqa: F401  # registers plotting backends for pandas
import streamlit as st
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy
from scipy.spatial import cKDTree

from src.common.common import page_setup
from utils.dia_tutorial import (
    mz_extraction_windows,
    filter_spectrum,
    reduce_spectra,
    annotate_filtered_spectra,
    apply_sgolay,
    msexperiment_to_dataframe,
    bin_3d_trace_df,
    add_binned_intensity_trace,
    add_binned_annotation_traces,
)

page_setup()


def _fmt_num(x: float) -> str:
    """Format numeric values with up to 6 significant digits, handling NaN."""
    try:
        if np.isfinite(x):
            return f"{x:.6g}"
    except Exception:
        pass
    return "n/a"


def _range(df: pd.DataFrame, col: str) -> str:
    """Return a human-friendly range string for numeric column `col` in `df`."""
    if col not in df.columns:
        return "n/a"
    ser = pd.to_numeric(df[col], errors="coerce").dropna()
    if ser.empty:
        return "n/a"
    mn = float(ser.min())
    mx = float(ser.max())
    if np.isclose(mn, mx):
        return _fmt_num(mn)
    return f"{_fmt_num(mn)} – {_fmt_num(mx)}"


# ------------------------------------
# Static target peptide parameters

mz_file = "example-data/mzML/20200505_Evosep_200SPD_SG06-16_filtered_ms1_mz_660_700_rt_130_170_with_ms2.mzML.gz"
precursor_mz = 680.3561
precursor_charge = 2
product_mzs = [386.2034, 487.2623, 600.3464, 701.3941, 861.4247, 974.5088]
product_charges = [1, 1, 1, 1, 1, 1]
product_annotations = ["b4^1", "y5^1", "y6^1", "y7^1", "y8^1", "y9^1"]
prec_mz_tol = 15
prod_mz_tol = 20
target_im = 0.96756938061
im_tol = 0.08

# ------------------------------------
# Session state — one flag per stage, plus cached data objects
#
_state_defaults = {
    # Stage 1: raw data loaded
    "stage1_done": False,
    "exp": None,
    "exp_df": None,
    # Stage 2: targeted extraction
    "stage2_done": False,
    "filtered_exp": None,
    "exp_df_targeted": None,
    # Stage 3: ion mobility exploration
    "stage3_done": False,
    "ion_df": None,
    # Stage 4: ion mobility filtering
    "stage4_done": False,
    "exp_df_targeted_with_im": None,
}
for _k, _v in _state_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# --------------------------------
# Page content
#
st.title("Getting Started With Targeted Data Extraction")
st.markdown(
    """
This page will walk you through the typical concepts for DIA data analysis using peptide-centric targeted data extraction. We will use a small sample dataset from the [DIA PASEF Evosep dataset (PXD017703)](https://www.ebi.ac.uk/pride/archive/projects/PXD017703) published by Meier et al., 2020. This dataset contains DIA data acquired on a Bruker timsTOF Pro instrument using the Evosep One LC system. The sample file we will use is a small subset of the full experiment run (*20200505_Evosep_200SPD_SG06-16_MLHeLa_200ng_py8_S3-A1_1_2737*), containing spectra in the MS1 precursor m/z range 660-700 and RT range 130-170s. 
"""
)

# ------------------------------------
# Load mzML data + raw peakmaps
#
st.markdown("---")
st.subheader("Load mzML Data")

load_clicked = st.button(
    "▶ Load DIA Data",
    type="primary",
    disabled=st.session_state.stage1_done,
    help="Load the mzML file and display raw MS1/MS2 peakmaps.",
)

if load_clicked or st.session_state.stage1_done:
    try:
        if load_clicked and not st.session_state.stage1_done:
            with st.spinner("Loading mzML file..."):
                exp = poms.MSExperiment()
                poms.MzMLFile().load(mz_file, exp)
                exp_df = exp.to_df(long_format=True)
                st.session_state.exp = exp
                st.session_state.exp_df = exp_df
                st.session_state.stage1_done = True
        else:
            exp = st.session_state.exp
            exp_df = st.session_state.exp_df

        st.success("Data loaded successfully.")
        st.markdown(
            f"Experiment summary: `MSExperiment(num_spectra={exp.getNrSpectra()}, num_chromatograms={exp.getNrChromatograms()}), ms_levels={exp.getMSLevels()}, rt_range=({exp.getMinRT()}, {exp.getMaxRT()}), mz_range=({exp.getMinMZ()}, {exp.getMaxMZ()})`"
        )
        st.markdown(
            "You can see our sample experiment file has 122 spectra, and contains both MS1 and MS2 spectra as expected for DIA data. We can also see the RT range is from 130.2601 to 169.9141, which matches our expected filtered RT range of 130-170s. The m/z range is from 95.0050 to 1704.9105, which is all inclusive of both the MS1 and MS2 m/z data"
        )

        with st.expander("Code:"):
            st.code(
                """exp = poms.MSExperiment()
poms.MzMLFile().load(mz_file, exp)
exp_df = exp.to_df(long_format=True)
print(exp)
print(exp_df['ms_level'].drop_duplicates())
""",
                language="python",
            )

        st.markdown("---")
        st.subheader("Raw DIA Peakmaps")

        fig1 = exp_df.loc[exp_df["ms_level"] == 1].plot(
            kind="peakmap",
            x="rt",
            y="mz",
            z="intensity",
            title="MS1 Spectra",
            aggregate_duplicates=True,
            z_log_scale=True,
            num_x_bins=500,
            num_y_bins=500,
            show_plot=False,
            backend="ms_plotly",
        )
        fig2 = exp_df.loc[exp_df["ms_level"] == 2].plot(
            kind="peakmap",
            x="rt",
            y="mz",
            z="intensity",
            title="MS2 Spectra",
            aggregate_duplicates=True,
            z_log_scale=True,
            num_x_bins=500,
            num_y_bins=500,
            show_plot=False,
            backend="ms_plotly",
        )

        fig_all = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["MS1 Spectra", "MS2 Spectra"],
        )
        for trace in fig1.data:
            fig_all.add_trace(trace, row=1, col=1)
        for trace in fig2.data:
            fig_all.add_trace(trace, row=1, col=2)

        fig_all.update_xaxes(title_text="Retention Time (s)", row=1, col=1)
        fig_all.update_yaxes(title_text="m/z", row=1, col=1)
        fig_all.update_xaxes(title_text="Retention Time (s)", row=1, col=2)
        fig_all.update_yaxes(title_text="m/z", row=1, col=2)

        st.plotly_chart(fig_all, use_container_width=True)

        st.markdown(
            ":blue[**Note:** the peakmaps are z-log scaled and binned for easier and quicker visualization.]"
        )
        st.markdown(
            "Looking at the peakmaps, we can see the MS1 spectra on the left showing the detected precursor ions with an m/z range of 660 to 700. The MS2 spectra on the right show the fragment ions generated from DIA isolation windows. We can see that the MS2 spectra are more complex and contain many fragment ions across a wide m/z range, which is typical for DIA data since multiple precursors are fragmented together. The intensity of the MS2 spectra is generally lower than the MS1 spectra, which is also expected since the signal is distributed across many fragments. In the next sections, we will apply targeted extraction to pull out specific precursor and fragment ions of interest from this complex DIA data."
        )

        with st.expander("Code:"):
            st.code(
                """
fig1 = exp_df.loc[exp_df["ms_level"] == 1].plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="intensity",
    title="MS1 Spectra",
    aggregate_duplicates=True,
    z_log_scale=True,
    num_x_bins=500,
    num_y_bins=500,
    show_plot=False,
    backend="ms_plotly",
)
fig2 = exp_df.loc[exp_df["ms_level"] == 2].plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="intensity",
    title="MS2 Spectra",
    aggregate_duplicates=True,
    z_log_scale=True,
    num_x_bins=500,
    num_y_bins=500,
    show_plot=False,
    backend="ms_plotly",
)

fig_all = make_subplots(rows=1, cols=2, subplot_titles=["MS1 Spectra", "MS2 Spectra"])
for trace in fig1.data:
    fig_all.add_trace(trace, row=1, col=1)
for trace in fig2.data:
    fig_all.add_trace(trace, row=1, col=2)
fig_all.update_xaxes(title_text="Retention Time (s)", row=1, col=1)
fig_all.update_yaxes(title_text="m/z", row=1, col=1)
fig_all.update_xaxes(title_text="Retention Time (s)", row=1, col=2)
fig_all.update_yaxes(title_text="m/z", row=1, col=2)
            """,
                language="python",
            )

    except Exception as e:
        st.error(f"Failed to load data: {e}")

# ------------------------------------
# Targeted extraction + annotation + filtered peakmaps + XICs
#
if st.session_state.stage1_done:
    st.markdown("---")
    st.subheader("Targeted Extraction")

    extract_clicked = st.button(
        "▶ Perform Targeted Extraction",
        type="primary",
        disabled=st.session_state.stage2_done,
        help="Filter spectra to the target peptide m/z windows, annotate matched peaks, and visualise extracted ion chromatograms.",
    )

    if extract_clicked or st.session_state.stage2_done:
        try:
            if extract_clicked and not st.session_state.stage2_done:
                with st.spinner("Filtering and annotating spectra..."):
                    exp = st.session_state.exp
                    exp_df = st.session_state.exp_df

                    filtered_exp = reduce_spectra(
                        exp,
                        float(precursor_mz),
                        product_mzs,
                        float(prec_mz_tol),
                        float(prod_mz_tol),
                    )
                    filtered_exp.updateRanges()
                    exp_df_targeted = filtered_exp.to_df(long_format=True)
                    exp_df_targeted = annotate_filtered_spectra(
                        filtered_df=exp_df_targeted,
                        precursor_mz=float(precursor_mz),
                        precursor_charge=int(precursor_charge),
                        product_mzs=product_mzs,
                        product_charges=product_charges,
                        product_annotations=product_annotations,
                        prec_mz_tol=float(prec_mz_tol),
                        prod_mz_tol=float(prod_mz_tol),
                    )
                    st.session_state.filtered_exp = filtered_exp
                    st.session_state.exp_df_targeted = exp_df_targeted
                    st.session_state.stage2_done = True
            else:
                exp = st.session_state.exp
                exp_df = st.session_state.exp_df
                filtered_exp = st.session_state.filtered_exp
                exp_df_targeted = st.session_state.exp_df_targeted

            st.markdown("""
For our small example file, we will perform targeted data extraction for the peptide **NTGIIC(UniMod:4)TIGPASR** with charge state 2. (The full mzML file was filtered for this specific example in mind based on prior analysis).

To perform targeted extraction, we need some information about the precursor ion we want to extract (m/z and charge), as well as the expected fragment ions (m/z, charge, and annotation if available). 

For our example peptide, **NTGIIC(UniMod:4)TIGPASR** with charge state 2, we expect the following precursor and fragment ions:

**precursor m/z (charge 2)**: 680.3561

| Fragment Ion | m/z | Charge | Annotation |
|--------------|---------|--------|------------|
| 386.2034     | 386.2034| 1      | b4^1 |
| 487.2623     | 487.2623| 1      | y5^1 |
| 600.3464     | 600.3464| 1      | y6^1 |
| 701.3941     | 701.3941| 1      | y7^1 |
| 861.4247     | 861.4247| 1      | y8^1 |
| 974.5088     | 974.5088| 1      | y9^1 |
""")

            st.markdown(
                "We use these target m/z values as coordinates and extract peaks in spectra that fall within the specified m/z tolerance window around each target m/z. This is done for both the precursor ion in the MS1 spectra and the fragment ions in the MS2 spectra. The result is a filtered set of spectra that contain peaks matching our target precursor and fragment ions."
            )
            st.markdown(
                ":blue[**Note:** In a real analysis scenario, you would perform additional filtering of the spectra by retention time. You would typically have prior information of where the target peptide has eluted from the liquid chromatography column (from (pseudo)DDA experiments or from predictions). These are usually expected normalized retention time values, however, for this sampled experiment file, we already filtered the full mzML file around where we expect to see the peak eluting in retention time, so there is no need for additional retention time filtering.]"
            )

            st.markdown("---")
            st.markdown("#### Filtering & Annotation")

            st.write(
                f"Filtered experiment summary: `MSExperiment(num_spectra={filtered_exp.getNrSpectra()}, num_chromatograms={filtered_exp.getNrChromatograms()}), ms_levels={filtered_exp.getMSLevels()}, rt_range=({filtered_exp.getMinRT()}, {filtered_exp.getMaxRT()}), mz_range=({filtered_exp.getMinMZ()}, {filtered_exp.getMaxMZ()})`"
            )
            st.write(
                "We can see that the number of spectra in the filtered experiment has reduced from 122 to 81, which means we have extracted a subset of spectra that contain peaks matching our target precursor and fragment ions. The retention time ranges remained the same because we did not apply any RT filtering, but the m/z range has narrowed to 386.1987 to 974.5265, which is within the m/z values of our target precursor and fragment ions."
            )

            st.dataframe(
                pd.concat([exp_df_targeted.head(), exp_df_targeted.tail()]),
                use_container_width=True,
            )

            st.markdown(
                "We can see from the resulting filtered spectra dataframe that we have spectra with both MS1 and MS2 spectra that contain peaks matching our target precursor and fragment ions. However, at this point, the spectra are not yet annotated with which peaks correspond to which target ions. The next step is to annotate the filtered spectra with the precursor and fragment ion assignments based on which target m/z they matched. We can also calculate the mass error in ppm for each matched peak to see how close the observed m/z is to the expected m/z for each target ion."
            )

            st.dataframe(
                pd.concat([exp_df_targeted.head(), exp_df_targeted.tail()]),
                use_container_width=True,
            )

            with st.expander("Code:"):
                st.code(
                    """precursor_mz = 680.3561
precursor_charge = 2
product_mzs = [386.2034,487.2623,600.3464,701.3941,861.4247,974.5088]
product_charges = [1,1,1,1,1,1]
product_annotations = ["b4^1", "y5^1", "y6^1", "y7^1", "y8^1", "y9^1"]
prec_mz_tol = 15
prod_mz_tol = 20

filtered_exp  = reduce_spectra(exp, float(precursor_mz), product_mzs, float(prec_mz_tol), float(prod_mz_tol))
filtered_exp.updateRanges()
exp_df_targeted = filtered_exp.to_df(long_format=True)
exp_df_targeted = annotate_filtered_spectra(
    filtered_df=exp_df_targeted,
    precursor_mz=float(precursor_mz),
    precursor_charge=int(precursor_charge),
    product_mzs=product_mzs,
    product_charges=product_charges,
    product_annotations=product_annotations,
    prec_mz_tol=float(prec_mz_tol),
    prod_mz_tol=float(prod_mz_tol),
)""",
                    language="python",
                )

            st.markdown("---")
            st.markdown("#### Raw vs Filtered Peakmaps")
            fig_cmp, axes_cmp = plt.subplots(2, 2, figsize=(12, 8))
            axes_cmp = axes_cmp.flatten()

            exp_df.loc[exp_df["ms_level"] == 1].plot(
                kind="peakmap",
                x="rt",
                y="mz",
                z="intensity",
                canvas=axes_cmp[0],
                title="Raw MS1 Spectra",
                aggregate_duplicates=True,
                z_log_scale=True,
                num_x_bins=500,
                num_y_bins=500,
                show_plot=False,
                backend="ms_matplotlib",
            )
            exp_df.loc[exp_df["ms_level"] == 2].plot(
                kind="peakmap",
                x="rt",
                y="mz",
                z="intensity",
                canvas=axes_cmp[1],
                title="Raw MS2 Spectra",
                aggregate_duplicates=True,
                z_log_scale=True,
                num_x_bins=500,
                num_y_bins=500,
                show_plot=False,
                backend="ms_matplotlib",
            )
            exp_df_targeted.loc[exp_df_targeted["ms_level"] == 1].plot(
                kind="peakmap",
                x="rt",
                y="mz",
                z="intensity",
                canvas=axes_cmp[2],
                title="Filtered MS1 Spectra",
                aggregate_duplicates=True,
                z_log_scale=True,
                num_x_bins=500,
                num_y_bins=500,
                show_plot=False,
                backend="ms_matplotlib",
            )
            exp_df_targeted.loc[exp_df_targeted["ms_level"] == 2].plot(
                kind="peakmap",
                x="rt",
                y="mz",
                z="intensity",
                canvas=axes_cmp[3],
                title="Filtered MS2 Spectra",
                aggregate_duplicates=True,
                z_log_scale=True,
                num_x_bins=500,
                num_y_bins=500,
                show_plot=False,
                backend="ms_matplotlib",
            )
            axes_cmp[2].set_ylim(axes_cmp[0].get_ylim())
            axes_cmp[3].set_ylim(axes_cmp[1].get_ylim())
            fig_cmp.tight_layout()
            st.pyplot(fig_cmp, use_container_width=True)

            st.markdown(
                "Looking at the raw vs filtered peakmaps, we can see that the filtered spectra contain a small subset of peaks that match our target precursor and fragment ions. The raw MS1 spectra show many precursor ion signals across the m/z range of 660-700, while the filtered MS1 spectra show only the peaks around our target precursor m/z of 680.3561. Similarly, the raw MS2 spectra show many fragment ions across a wide m/z range, while the filtered MS2 spectra show only the peaks around our target fragment m/z values. This illustrates how targeted extraction can pull out specific signals of interest from multiplexed spectra in DIA data."
            )

            with st.expander("Code:"):
                st.code(
                    """fig_cmp, axes_cmp = plt.subplots(2, 2, figsize=(12, 8))
axes_cmp = axes_cmp.flatten()
exp_df.loc[exp_df["ms_level"] == 1].plot(
    kind="peakmap", x="rt", y="mz", z="intensity", canvas=axes_cmp[0],
    title="Raw MS1 Spectra", aggregate_duplicates=True, z_log_scale=True,
    num_x_bins=500, num_y_bins=500, show_plot=False, backend="ms_matplotlib",
)
exp_df.loc[exp_df["ms_level"] == 2].plot(
    kind="peakmap", x="rt", y="mz", z="intensity", canvas=axes_cmp[1],
    title="Raw MS2 Spectra", aggregate_duplicates=True, z_log_scale=True,
    num_x_bins=500, num_y_bins=500, show_plot=False, backend="ms_matplotlib",
)
exp_df_targeted.loc[exp_df_targeted["ms_level"] == 1].plot(
    kind="peakmap", x="rt", y="mz", z="intensity", canvas=axes_cmp[2],
    title="Filtered MS1 Spectra", aggregate_duplicates=True, z_log_scale=True,
    num_x_bins=500, num_y_bins=500, show_plot=False, backend="ms_matplotlib",
)
exp_df_targeted.loc[exp_df_targeted["ms_level"] == 2].plot(
    kind="peakmap", x="rt", y="mz", z="intensity", canvas=axes_cmp[3],
    title="Filtered MS2 Spectra", aggregate_duplicates=True, z_log_scale=True,
    num_x_bins=500, num_y_bins=500, show_plot=False, backend="ms_matplotlib",
)
axes_cmp[2].set_ylim(axes_cmp[0].get_ylim())
axes_cmp[3].set_ylim(axes_cmp[1].get_ylim())
fig_cmp.tight_layout()""",
                    language="python",
                )

            with st.spinner("Generating 3D peakmaps — this may take a moment..."):
                progress = st.progress(0)
                exp_df_copy = exp_df.copy()
                exp_df_copy["is_targeted"] = False
                if not exp_df_targeted.empty:
                    mz_tol = 0.01
                    for ms in exp_df_copy["ms_level"].unique():
                        mask_copy = exp_df_copy["ms_level"] == ms
                        mask_target = exp_df_targeted["ms_level"] == ms
                        if not mask_target.any():
                            continue
                        tgt_mz = exp_df_targeted.loc[mask_target, "mz"].to_numpy()
                        tgt_coords = (tgt_mz / mz_tol).reshape(-1, 1)
                        tree = cKDTree(tgt_coords)
                        copy_mz = exp_df_copy.loc[mask_copy, "mz"].to_numpy()
                        copy_coords = (copy_mz / mz_tol).reshape(-1, 1)
                        neighbors = tree.query_ball_point(copy_coords, r=1.0)
                        has_match = np.array([len(n) > 0 for n in neighbors])
                        exp_df_copy.loc[mask_copy, "is_targeted"] = has_match
                progress.progress(30)

                exp_df_copy["is_targeted_label"] = np.where(
                    exp_df_copy["is_targeted"], "Targeted", "Raw"
                )
                exp_df_targeted["ms_level_label"] = (
                    exp_df_targeted["ms_level"]
                    .map({1: "MS1", 2: "MS2"})
                    .fillna(exp_df_targeted["ms_level"].astype(str))
                )
                progress.progress(50)

                raw_3d = exp_df_copy.plot(
                    kind="peakmap",
                    x="rt",
                    y="mz",
                    z="intensity",
                    by="is_targeted_label",
                    title="Raw Spectra (MS1 + MS2)",
                    plot_3d=True,
                    aggregate_duplicates=False,
                    bin_peaks=True,
                    num_x_bins=500,
                    num_y_bins=500,
                    aggregation_method="sum",
                    show_plot=False,
                    width=800,
                    height=600,
                    legend_config=dict(title="Targeted"),
                    backend="ms_plotly",
                )
                progress.progress(70)

                filt_3d = exp_df_targeted.plot(
                    kind="peakmap",
                    x="rt",
                    y="mz",
                    z="intensity",
                    by="ms_level_label",
                    title="Filtered Spectra (MS1 + MS2)",
                    plot_3d=True,
                    aggregate_duplicates=False,
                    bin_peaks=True,
                    num_x_bins=500,
                    num_y_bins=500,
                    aggregation_method="sum",
                    show_plot=False,
                    width=800,
                    height=600,
                    legend_config=dict(title="MS Level"),
                    backend="ms_plotly",
                )
                progress.progress(90)

                fig_3d = make_subplots(
                    rows=1,
                    cols=2,
                    specs=[[{"type": "scene"}, {"type": "scene"}]],
                    subplot_titles=["Raw Spectra", "Filtered Spectra"],
                )
                for idx, panel in enumerate([raw_3d, filt_3d]):
                    for trace in panel.data:
                        fig_3d.add_trace(trace, row=1, col=idx + 1)
                    scene_name = "scene" if idx == 0 else f"scene{idx + 1}"
                    if getattr(panel.layout, "scene", None) is not None:
                        fig_3d.layout[scene_name].update(panel.layout.scene)

                fig_3d.update_layout(
                    scene=dict(domain=dict(x=[0.03, 0.48])),
                    scene2=dict(domain=dict(x=[0.52, 0.97])),
                    width=1200,
                    height=600,
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                fig_3d.layout.annotations[0].x = 0.25
                fig_3d.layout.annotations[1].x = 0.75
                progress.progress(100)
                st.plotly_chart(fig_3d, use_container_width=True)

            st.markdown(
                "We can look at the peakmaps in 3D to visually see topological features more easily. If we look at the raw spectra, which we have color coded the corresponding targeted spectra matching our charged peptide of interest (red), we can see how small the spectral peaks are for our target peptide in comparison to the rest of the spectral signal in the data. If we look at the filtered spectra, you can more clearly see matching spectral patterns of the chromatography features (i.e. the eluting chromatographic peak in retention time). You can also tell the more intense precursor ion signal (blue) apart from the much lower intensity fragment ions (red)."
            )
            st.markdown(
                ":orange[**Tip:** You can toggle the legend items to hide/show the targeted spectra or the MS1 vs MS2 spectra.]"
            )

            with st.expander("Code:"):
                st.code(
                    """exp_df_copy = exp_df.copy()
exp_df_copy["is_targeted"] = False
if not exp_df_targeted.empty:
    mz_tol = 0.01
    for ms in exp_df_copy["ms_level"].unique():
        mask_copy   = exp_df_copy["ms_level"] == ms
        mask_target = exp_df_targeted["ms_level"] == ms
        if not mask_target.any():
            continue
        tgt_mz     = exp_df_targeted.loc[mask_target, "mz"].to_numpy()
        tgt_coords = (tgt_mz / mz_tol).reshape(-1, 1)
        tree       = cKDTree(tgt_coords)
        copy_mz    = exp_df_copy.loc[mask_copy, "mz"].to_numpy()
        copy_coords = (copy_mz / mz_tol).reshape(-1, 1)
        neighbors  = tree.query_ball_point(copy_coords, r=1.0)
        has_match  = np.array([len(n) > 0 for n in neighbors])
        exp_df_copy.loc[mask_copy, "is_targeted"] = has_match

exp_df_copy["is_targeted_label"] = np.where(exp_df_copy["is_targeted"], "Targeted", "Raw")
exp_df_targeted["ms_level_label"] = (
    exp_df_targeted["ms_level"].map({1: "MS1", 2: "MS2"})
    .fillna(exp_df_targeted["ms_level"].astype(str))
)
raw_3d  = exp_df_copy.plot(kind="peakmap", ..., by="is_targeted_label", plot_3d=True, ...)
filt_3d = exp_df_targeted.plot(kind="peakmap", ..., by="ms_level_label", plot_3d=True, ...)
fig_3d  = make_subplots(rows=1, cols=2, specs=[[{"type":"scene"},{"type":"scene"}]])
for idx, panel in enumerate([raw_3d, filt_3d]):
    for trace in panel.data:
        fig_3d.add_trace(trace, row=1, col=idx+1)
fig_3d.update_layout(scene=dict(domain=dict(x=[0.03,0.48])), scene2=dict(domain=dict(x=[0.52,0.97])))
                    """,
                    language="python",
                )

            st.markdown("---")
            st.markdown("#### Extracted Ion Chromatograms")

            chrom_fig = exp_df_targeted.plot(
                kind="chromatogram",
                x="rt",
                y="intensity",
                by="annotation",
                title="Chromatogram",
                aggregate_duplicates=True,
                legend_config=dict(title="Transition"),
                backend="ms_plotly",
                show_plot=False,
            )

            group_cols = ["ms_level", "annotation", "rt"]
            integrate_col = "intensity"
            smoothed_chrom_fig = (
                exp_df_targeted.apply(
                    lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna(".")
                )
                .groupby(group_cols)[integrate_col]
                .sum()
                .reset_index()
                .groupby(["annotation", "ms_level"])[group_cols + [integrate_col]]
                .apply(apply_sgolay, window_length=9, polyorder=3)
                .reset_index(drop=True)
                .plot(
                    kind="chromatogram",
                    x="rt",
                    y="smoothed_int",
                    by="annotation",
                    title="Smoothed Chromatogram",
                    aggregate_duplicates=False,
                    legend_config=dict(title="Transition"),
                    backend="ms_plotly",
                    show_plot=False,
                )
            )

            chrom_subfig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["Raw Chromatogram", "Smoothed Chromatogram"],
            )
            for i, (t_raw, t_smooth) in enumerate(
                zip(chrom_fig.data, smoothed_chrom_fig.data)
            ):
                name = (
                    getattr(t_raw, "name", None)
                    or getattr(t_smooth, "name", None)
                    or f"trace_{i}"
                )
                t_raw.name = name
                t_raw.legendgroup = name
                t_raw.showlegend = True
                t_smooth.name = name
                t_smooth.legendgroup = name
                t_smooth.showlegend = False
                chrom_subfig.add_trace(t_raw, row=1, col=1)
                chrom_subfig.add_trace(t_smooth, row=1, col=2)
            if len(chrom_fig.data) > len(smoothed_chrom_fig.data):
                for extra in chrom_fig.data[len(smoothed_chrom_fig.data) :]:
                    chrom_subfig.add_trace(extra, row=1, col=1)
            elif len(smoothed_chrom_fig.data) > len(chrom_fig.data):
                for extra in smoothed_chrom_fig.data[len(chrom_fig.data) :]:
                    chrom_subfig.add_trace(extra, row=1, col=2)
            chrom_subfig.update_layout(
                width=1200, height=500, margin=dict(l=20, r=20, t=60, b=20)
            )
            st.plotly_chart(chrom_subfig, use_container_width=True)

            st.markdown(
                "The chromatograms show the intensity of the precursor and fragment ions over retention time. The raw chromatogram on the left shows the original signal, which can be a little noisy. The smoothed chromatogram on the right applies a Savitzky-Golay filter to reduce noise and make it easier to see the elution profiles of the ions, as well as making downstream peak-picking more stable."
            )
            st.markdown(
                ":orange[**Tip:** You can toggle the legend items to show/hide specific transitions and see how they co-elute, which is important for confirming the presence of the target peptide in DIA data.]"
            )

            with st.expander("Code:"):
                st.code(
                    """chrom_fig = exp_df_targeted.plot(
    kind="chromatogram", x="rt", y="intensity", by="annotation",
    title="Chromatogram", aggregate_duplicates=True,
    legend_config=dict(title="Transition"), backend="ms_plotly", show_plot=False,
)

group_cols=['ms_level', 'annotation', 'rt']
integrate_col = 'intensity'
smoothed_chrom_fig = (
    exp_df_targeted
    .apply(lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna("."))
    .groupby(group_cols)[integrate_col].sum().reset_index()
    .groupby(['annotation', 'ms_level'])[group_cols + [integrate_col]]
    .apply(apply_sgolay, window_length=9, polyorder=3)
    .reset_index(drop=True)
    .plot(
        kind="chromatogram", x="rt", y="smoothed_int", by="annotation",
        title="Smoothed Chromatogram", aggregate_duplicates=False,
        legend_config=dict(title="Transition"), backend="ms_plotly", show_plot=False,
    )
)""",
                    language="python",
                )

        except Exception as e:
            st.error(f"Failed to perform targeted extraction: {e}")

# ------------------------------------
# Explore the Ion Mobility dimension (raw 3D peakmaps)
#
if st.session_state.stage2_done:
    st.markdown("---")
    st.subheader("What About the Ion Mobility Dimension?")

    im_explore_clicked = st.button(
        "▶ Explore Ion Mobility Dimension",
        type="primary",
        disabled=st.session_state.stage3_done,
        help="Visualise the full dataset in 3D (RT × m/z × ion mobility).",
    )

    if im_explore_clicked or st.session_state.stage3_done:
        try:
            if im_explore_clicked and not st.session_state.stage3_done:
                exp = st.session_state.exp
                ion_df = msexperiment_to_dataframe(exp)
                st.session_state.ion_df = ion_df
                st.session_state.stage3_done = True
            else:
                exp = st.session_state.exp
                ion_df = st.session_state.ion_df

            ms1_ion_df = ion_df.loc[ion_df["ms_level"] == 1]
            ms2_ion_df = ion_df.loc[ion_df["ms_level"] == 2]

            st.write("MS1 Spectra")
            ms1_table = (
                "| Metric | Range |\n"
                "|---|---:|\n"
                f"| Retention time | {_range(ms1_ion_df, 'rt')} |\n"
                f"| m/z | {_range(ms1_ion_df, 'mz')} |\n"
                f"| Ion mobility | {_range(ms1_ion_df, 'ion_mobility')} |\n"
            )
            st.markdown(ms1_table)

            st.write("MS2 Spectra")
            ms2_table = (
                "| Metric | Range |\n"
                "|---|---:|\n"
                f"| Retention time | {_range(ms2_ion_df, 'rt')} |\n"
                f"| m/z | {_range(ms2_ion_df, 'mz')} |\n"
                f"| Ion mobility | {_range(ms2_ion_df, 'ion_mobility')} |\n"
            )
            st.markdown(ms2_table)

            st.markdown(
                """This example DIA dataset was acquired on a timsTOF instrument, which means there is an additional ion mobility dimension in the data. Our sampled dataset contains MS1 spectra containing inversed ion mobility in the range 0.60199 to 1.59998, and MS2 spectra contained inversed ion mobility in the range 0.89033 to 1.10950."""
            )

            with st.spinner(
                "Generating 3D ion-mobility peakmaps — this may take a moment..."
            ):
                progress = st.progress(0)

                ms1_binned = bin_3d_trace_df(
                    ms1_ion_df,
                    rt_col="rt",
                    mz_col="mz",
                    im_col="ion_mobility",
                    intensity_col="intensity",
                    bins=(100, 100, 50),
                    intensity_agg="mean",
                )
                progress.progress(30)

                ms2_binned = bin_3d_trace_df(
                    ms2_ion_df,
                    rt_col="rt",
                    mz_col="mz",
                    im_col="ion_mobility",
                    intensity_col="intensity",
                    bins=(100, 100, 50),
                    intensity_agg="mean",
                )
                progress.progress(60)

                fig = make_subplots(
                    rows=1,
                    cols=2,
                    specs=[[{"type": "scene"}, {"type": "scene"}]],
                    subplot_titles=["MS1 Spectra", "MS2 Spectra"],
                )
                progress.progress(70)

                all_color_vals = []
                if not ms1_binned.empty:
                    all_color_vals.append(ms1_binned["agg_value"].to_numpy(dtype=float))
                if not ms2_binned.empty:
                    all_color_vals.append(ms2_binned["agg_value"].to_numpy(dtype=float))

                if all_color_vals:
                    all_color_vals = np.concatenate(all_color_vals)
                    all_color_vals = np.log10(np.clip(all_color_vals, 0, None) + 1.0)
                    cmin, cmax = np.quantile(all_color_vals, [0.01, 0.99])
                else:
                    cmin, cmax = 0.0, 1.0

                add_binned_intensity_trace(
                    fig, ms1_binned, row=1, col=1, name="MS1", cmin=cmin, cmax=cmax
                )
                add_binned_intensity_trace(
                    fig, ms2_binned, row=1, col=2, name="MS2", cmin=cmin, cmax=cmax
                )
                progress.progress(90)

                fig.update_layout(
                    scene=dict(
                        xaxis_title="Retention Time (s)",
                        yaxis_title="m/z",
                        zaxis_title="Ion mobility",
                    ),
                    scene2=dict(
                        xaxis_title="Retention Time (s)",
                        yaxis_title="m/z",
                        zaxis_title="Ion mobility",
                    ),
                    showlegend=False,
                    margin=dict(l=0, r=0, t=50, b=0),
                )
                progress.progress(100)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """Looking at the spectra in 3D with the ion mobility dimension, we can see just how much more complicated the data looks like. It just looks like blobs of color"""
            )

            with st.expander("Code:"):
                st.code(
                    """ion_df = msexperiment_to_dataframe(exp)
ms1_ion_df = ion_df.loc[ion_df["ms_level"] == 1]
ms2_ion_df = ion_df.loc[ion_df["ms_level"] == 2]

ms1_binned = bin_3d_trace_df(ms1_ion_df, rt_col="rt", mz_col="mz", im_col="ion_mobility",
    intensity_col="intensity", bins=(100, 100, 50), intensity_agg="mean")
ms2_binned = bin_3d_trace_df(ms2_ion_df, rt_col="rt", mz_col="mz", im_col="ion_mobility",
    intensity_col="intensity", bins=(100, 100, 50), intensity_agg="mean")

fig = make_subplots(rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=["MS1 Spectra", "MS2 Spectra"])

all_color_vals = np.log10(np.clip(np.concatenate([
    ms1_binned["agg_value"].to_numpy(dtype=float),
    ms2_binned["agg_value"].to_numpy(dtype=float)]), 0, None) + 1.0)
cmin, cmax = np.quantile(all_color_vals, [0.01, 0.99])

add_binned_intensity_trace(fig, ms1_binned, row=1, col=1, name="MS1", cmin=cmin, cmax=cmax)
add_binned_intensity_trace(fig, ms2_binned, row=1, col=2, name="MS2", cmin=cmin, cmax=cmax)
fig.update_layout(scene=dict(xaxis_title="Retention Time (s)", yaxis_title="m/z", zaxis_title="Ion mobility"),
    scene2=dict(xaxis_title="Retention Time (s)", yaxis_title="m/z", zaxis_title="Ion mobility"))
                    """,
                    language="python",
                )

        except Exception as e:
            st.error(f"Failed to explore ion mobility: {e}")

# ------------------------------------
# Ion Mobility filtering + annotated 3D + chromatograms + mobilograms
#
if st.session_state.stage3_done:
    st.markdown("---")
    st.subheader("Filtering Spectra Including Ion Mobility")

    im_filter_clicked = st.button(
        "▶ Apply Ion Mobility Filtering",
        type="primary",
        disabled=st.session_state.stage4_done,
        help="Re-run targeted extraction with an additional ion mobility filter window, then visualise IM-filtered chromatograms and mobilograms.",
    )

    if im_filter_clicked or st.session_state.stage4_done:
        try:
            if im_filter_clicked and not st.session_state.stage4_done:
                with st.spinner("Applying ion mobility filter..."):
                    exp = st.session_state.exp
                    filtered_exp_with_im = reduce_spectra(
                        exp,
                        float(precursor_mz),
                        product_mzs,
                        float(prec_mz_tol),
                        float(prod_mz_tol),
                        float(target_im),
                        float(im_tol),
                    )
                    filtered_exp_with_im.updateRanges()
                    exp_df_targeted_with_im = msexperiment_to_dataframe(
                        filtered_exp_with_im
                    )
                    exp_df_targeted_with_im = annotate_filtered_spectra(
                        filtered_df=exp_df_targeted_with_im,
                        precursor_mz=float(precursor_mz),
                        precursor_charge=int(precursor_charge),
                        product_mzs=product_mzs,
                        product_charges=product_charges,
                        product_annotations=product_annotations,
                        prec_mz_tol=float(prec_mz_tol),
                        prod_mz_tol=float(prod_mz_tol),
                    )
                    st.session_state.exp_df_targeted_with_im = exp_df_targeted_with_im
                    st.session_state.stage4_done = True
            else:
                exp = st.session_state.exp
                ion_df = st.session_state.ion_df
                exp_df_targeted_with_im = st.session_state.exp_df_targeted_with_im

            st.markdown(
                """Like m/z and retention time filtering, we can further filter the spectra by ion mobility as well. This means we need to have another coordinate to extract and filter around a targeted ion mobility value. For our target peptide of interest, from prior analysis, the targeted ion mobility value is **0.96756938061**. We can include this target ion mobility in our filtering criteria using a tolerance window of 0.08."""
            )

            with st.spinner(
                "Generating 3D ion-mobility peakmaps — this may take a moment..."
            ):
                progress = st.progress(0)

                ion_df = st.session_state.ion_df
                raw_binned = bin_3d_trace_df(
                    ion_df,
                    rt_col="rt",
                    mz_col="mz",
                    im_col="ion_mobility",
                    intensity_col="intensity",
                    bins=(100, 100, 50),
                    intensity_agg="mean",
                )
                progress.progress(30)

                fig = make_subplots(
                    rows=1,
                    cols=2,
                    specs=[[{"type": "scene"}, {"type": "scene"}]],
                    subplot_titles=["Raw Spectra", "Filtered Spectra"],
                )

                raw_color = np.log10(
                    np.clip(raw_binned["agg_value"].to_numpy(dtype=float), 0, None)
                    + 1.0
                )
                if len(raw_color) > 0:
                    cmin, cmax = np.quantile(raw_color, [0.01, 0.99])
                else:
                    cmin, cmax = 0.0, 1.0

                fig.add_trace(
                    go.Scatter3d(
                        x=raw_binned["rt"],
                        y=raw_binned["mz"],
                        z=raw_binned["ion_mobility"],
                        mode="markers",
                        name="all",
                        showlegend=False,
                        marker=dict(
                            symbol="square",
                            size=4,
                            opacity=0.9,
                            color=raw_color,
                            colorscale="Viridis",
                            cmin=float(cmin),
                            cmax=float(cmax),
                            showscale=False,
                            line=dict(width=0),
                        ),
                        customdata=np.stack(
                            [
                                raw_binned["count"].to_numpy(dtype=float),
                                raw_binned["agg_value"].to_numpy(dtype=float),
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "rt: %{x:.4f}<br>mz: %{y:.4f}<br>"
                            "ion_mobility: %{z:.4f}<br>count: %{customdata[0]:.0f}<br>"
                            "mean(intensity): %{customdata[1]:.4f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )
                progress.progress(55)

                fig = add_binned_annotation_traces(
                    fig,
                    exp_df_targeted_with_im,
                    row=1,
                    col=2,
                    annotation_col="annotation",
                    rt_col="rt",
                    mz_col="mz",
                    im_col="ion_mobility",
                    intensity_col="intensity",
                    bins=(100, 100, 50),
                    intensity_agg="mean",
                    log_color=True,
                    color_quantile_clip=(0.01, 0.99),
                    marker_size=4,
                    marker_opacity=0.9,
                )
                progress.progress(85)

                fig.update_layout(
                    scene=dict(
                        xaxis_title="Retention Time (s)",
                        yaxis_title="m/z",
                        zaxis_title="Ion mobility",
                    ),
                    scene2=dict(
                        xaxis_title="Retention Time (s)",
                        yaxis_title="m/z",
                        zaxis_title="Ion mobility",
                    ),
                    legend=dict(
                        title="Annotation",
                        x=1.02,
                        y=1.0,
                        xanchor="left",
                        yanchor="top",
                        groupclick="togglegroup",
                    ),
                    margin=dict(l=0, r=0, t=50, b=0),
                )
                progress.progress(100)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                "From the filtered spectra, we get a much clearer picture of the spectral features corresponding to our target peptide, and how they are distributed in the ion mobility dimension. We can see that the targeted spectra (colored by annotation) cluster around the target ion mobility value, which is consistent with our expectation. This ion mobility filtering can help to further reduce interference from co-eluting peptides in DIA data, and improve the quality of extracted chromatograms and downstream quantification."
            )

            with st.expander("Code:"):
                st.code(
                    """target_im = 0.96756938061
im_tol = 0.08  
filtered_exp_with_im = reduce_spectra(exp, float(precursor_mz), product_mzs,
    float(prec_mz_tol), float(prod_mz_tol), float(target_im), float(im_tol))
filtered_exp_with_im.updateRanges()
exp_df_targeted_with_im = msexperiment_to_dataframe(filtered_exp_with_im)
exp_df_targeted_with_im = annotate_filtered_spectra(
    filtered_df=exp_df_targeted_with_im, precursor_mz=float(precursor_mz),
    precursor_charge=int(precursor_charge), product_mzs=product_mzs,
    product_charges=product_charges, product_annotations=product_annotations,
    prec_mz_tol=float(prec_mz_tol), prod_mz_tol=float(prod_mz_tol))""",
                    language="python",
                )

            st.markdown("---")
            st.markdown("#### Chromatograms With vs Without Ion Mobility Filtering")

            group_cols = ["ms_level", "annotation", "rt"]
            integrate_col = "intensity"

            chrom_fig = st.session_state.exp_df_targeted.plot(
                kind="chromatogram",
                x="rt",
                y="intensity",
                by="annotation",
                title="Chromatogram (no IM)",
                aggregate_duplicates=True,
                legend_config=dict(title="Transition"),
                backend="ms_plotly",
                show_plot=False,
            )
            smoothed_chrom_fig = (
                st.session_state.exp_df_targeted.apply(
                    lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna(".")
                )
                .groupby(group_cols)[integrate_col]
                .sum()
                .reset_index()
                .groupby(["annotation", "ms_level"])[group_cols + [integrate_col]]
                .apply(apply_sgolay, window_length=9, polyorder=3)
                .reset_index(drop=True)
                .plot(
                    kind="chromatogram",
                    x="rt",
                    y="smoothed_int",
                    by="annotation",
                    title="Smoothed Chromatogram (no IM)",
                    aggregate_duplicates=False,
                    legend_config=dict(title="Transition"),
                    backend="ms_plotly",
                    show_plot=False,
                )
            )
            chrom_fig_im = exp_df_targeted_with_im.plot(
                kind="chromatogram",
                x="rt",
                y="intensity",
                by="annotation",
                title="Chromatogram (with IM)",
                aggregate_duplicates=True,
                legend_config=dict(title="Transition"),
                backend="ms_plotly",
                show_plot=False,
            )
            smoothed_chrom_fig_im = (
                exp_df_targeted_with_im.apply(
                    lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna(".")
                )
                .groupby(group_cols)[integrate_col]
                .sum()
                .reset_index()
                .groupby(["annotation", "ms_level"])[group_cols + [integrate_col]]
                .apply(apply_sgolay, window_length=9, polyorder=3)
                .reset_index(drop=True)
                .plot(
                    kind="chromatogram",
                    x="rt",
                    y="smoothed_int",
                    by="annotation",
                    title="Smoothed Chromatogram (with IM)",
                    aggregate_duplicates=False,
                    legend_config=dict(title="Transition"),
                    backend="ms_plotly",
                    show_plot=False,
                )
            )

            chrom_subfig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    "Raw Chromatogram (no IM)",
                    "Smoothed Chromatogram (no IM)",
                    "Raw Chromatogram (with IM)",
                    "Smoothed Chromatogram (with IM)",
                ],
            )
            for i, (t_raw, t_smooth) in enumerate(
                zip(chrom_fig.data, smoothed_chrom_fig.data)
            ):
                name = getattr(t_raw, "name", None)
                if name is not None:
                    t_raw.name = name
                    t_raw.legendgroup = name
                    t_raw.showlegend = True
                    t_smooth.name = name
                    t_smooth.legendgroup = name
                    t_smooth.showlegend = False
                    chrom_subfig.add_trace(t_raw, row=1, col=1)
                    chrom_subfig.add_trace(t_smooth, row=1, col=2)
            for i, (t_raw, t_smooth) in enumerate(
                zip(chrom_fig_im.data, smoothed_chrom_fig_im.data)
            ):
                name = getattr(t_raw, "name", None)
                if name is not None:
                    t_raw.name = name
                    t_raw.legendgroup = name
                    t_raw.showlegend = True
                    t_smooth.name = name
                    t_smooth.legendgroup = name
                    t_smooth.showlegend = False
                    chrom_subfig.add_trace(t_raw, row=2, col=1)
                    chrom_subfig.add_trace(t_smooth, row=2, col=2)
            chrom_subfig.update_layout(width=1200, height=800)
            st.plotly_chart(chrom_subfig, use_container_width=True)

            st.markdown(
                "Comparing the chromatograms with and without ion mobility filtering, we can see that the IM-filtered chromatograms (bottom row) show cleaner and more distinct elution peaks. The additional ion mobility dimension helps to separate overlapping peptide signals that may co-elute in both RT and m/z."
            )

            with st.expander("Code:"):
                st.code(
                    """chrom_fig    = exp_df_targeted.plot(kind="chromatogram", x="rt", y="intensity", ...)
chrom_fig_im = exp_df_targeted_with_im.plot(kind="chromatogram", x="rt", y="intensity", ...)
# (smoothed versions via apply_sgolay — see earlier code block)
chrom_subfig = make_subplots(rows=2, cols=2, subplot_titles=[...])
# add traces with shared legendgroups for row 1 (no IM) and row 2 (with IM)
chrom_subfig.update_layout(width=1200, height=800)""",
                    language="python",
                )

            st.markdown("---")
            st.markdown("#### Extracted Ion Mobilogram")

            mobi_fig = exp_df_targeted_with_im.plot(
                kind="mobilogram",
                x="ion_mobility",
                y="intensity",
                by="annotation",
                title="Mobilogram",
                aggregate_duplicates=True,
                legend_config=dict(title="Transition"),
                backend="ms_plotly",
                show_plot=False,
            )

            group_cols_im = ["ms_level", "annotation", "ion_mobility"]
            smoothed_ombi_fig = (
                exp_df_targeted_with_im.apply(
                    lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna(".")
                )
                .groupby(group_cols_im)[integrate_col]
                .sum()
                .reset_index()
                .groupby(["annotation", "ms_level"])[group_cols_im + [integrate_col]]
                .apply(
                    apply_sgolay, along_col="ion_mobility", window_length=9, polyorder=3
                )
                .reset_index(drop=True)
                .plot(
                    kind="mobilogram",
                    x="ion_mobility",
                    y="smoothed_int",
                    by="annotation",
                    title="Smoothed Mobilogram",
                    aggregate_duplicates=False,
                    legend_config=dict(title="Transition"),
                    backend="ms_plotly",
                    show_plot=False,
                )
            )

            mobi_subfig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["Raw Mobilogram", "Smoothed Mobilogram"],
            )
            for i, (t_raw, t_smooth) in enumerate(
                zip(mobi_fig.data, smoothed_ombi_fig.data)
            ):
                name = getattr(t_raw, "name", None)
                if name is not None:
                    t_raw.name = name
                    t_raw.legendgroup = name
                    t_raw.showlegend = True
                    t_smooth.name = name
                    t_smooth.legendgroup = name
                    t_smooth.showlegend = False
                    mobi_subfig.add_trace(t_raw, row=1, col=1)
                    mobi_subfig.add_trace(t_smooth, row=1, col=2)
            mobi_subfig.update_layout(width=1200, height=800)
            st.plotly_chart(mobi_subfig, use_container_width=True)

            st.markdown(
                "The mobilogram shows the intensity of the ions over the ion mobility dimension. Similar to the chromatogram, we can see that the smoothed mobilogram on the right reduces noise and makes it easier to see the ion mobility profiles of the ions corresponding to our target peptide."
            )

            with st.expander("Code:"):
                st.code(
                    """mobi_fig = exp_df_targeted_with_im.plot(
    kind="mobilogram", x="ion_mobility", y="intensity", by="annotation",
    title="Mobilogram", aggregate_duplicates=True,
    legend_config=dict(title="Transition"), backend="ms_plotly", show_plot=False,
)

group_cols=['ms_level', 'annotation', 'ion_mobility']
integrate_col = 'intensity'
smoothed_ombi_fig = (
    exp_df_targeted_with_im
    .apply(lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna("."))
    .groupby(group_cols)[integrate_col].sum().reset_index()
    .groupby(['annotation', 'ms_level'])[group_cols + [integrate_col]]
    .apply(apply_sgolay, along_col="ion_mobility", window_length=9, polyorder=3)
    .reset_index(drop=True)
    .plot(kind="mobilogram", x="ion_mobility", y="smoothed_int", by="annotation",
          title="Smoothed Mobilogram", aggregate_duplicates=False,
          legend_config=dict(title="Transition"), backend="ms_plotly", show_plot=False)
)

mobi_subfig = make_subplots(rows=1, cols=2, subplot_titles=["Raw Mobilogram", "Smoothed Mobilogram"])
for i, (t_raw, t_smooth) in enumerate(zip(mobi_fig.data, smoothed_ombi_fig.data)):
    name = getattr(t_raw, "name", None)
    if name is not None:
        t_raw.name = name; t_raw.legendgroup = name; t_raw.showlegend = True
        t_smooth.name = name; t_smooth.legendgroup = name; t_smooth.showlegend = False
        mobi_subfig.add_trace(t_raw, row=1, col=1)
        mobi_subfig.add_trace(t_smooth, row=1, col=2)
mobi_subfig.update_layout(width=1200, height=800)
                    """,
                    language="python",
                )

        except Exception as e:
            st.error(f"Failed to apply ion mobility filtering: {e}")

# ------------------------------------
# Functions reference
#
st.markdown("---")
st.subheader("Imports & Functions Used in This Tutorial")
with st.expander("Click to view code for utility functions used in the tutorial"):
    st.code(
        f"""import numpy as np # v{np.__version__}
import pandas as pd # v{pd.__version__}
import matplotlib.pyplot as plt # v{matplotlib.__version__}
import pyopenms as poms # v{poms.__version__}
import pyopenms_viz # v{pyopenms_viz.__version__}, registers MS plotting methods for pandas dataframes
import plotly # v{plotly.__version__}
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
from scipy.spatial import cKDTree # v{scipy.__version__}
# Method imports below are local modules part of the OpenDIAKiosk
from utils.dia_tutorial import mz_extraction_windows, filter_spectrum, reduce_spectra, annotate_filtered_spectra, apply_sgolay, msexperiment_to_dataframe, bin_3d_trace_df, add_binned_intensity_trace, add_binned_annotation_traces""",
        language="python",
    )
    st.markdown("---")

    st.code(inspect.getsource(mz_extraction_windows), language="python")
    st.code(inspect.getsource(filter_spectrum), language="python")
    st.code(inspect.getsource(reduce_spectra), language="python")
    st.code(inspect.getsource(annotate_filtered_spectra), language="python")
    st.code(inspect.getsource(apply_sgolay), language="python")
    st.code(inspect.getsource(msexperiment_to_dataframe), language="python")
    st.code(inspect.getsource(bin_3d_trace_df), language="python")
    st.code(inspect.getsource(add_binned_intensity_trace), language="python")
    st.code(inspect.getsource(add_binned_annotation_traces), language="python")
