from __future__ import annotations

import inspect

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pyopenms as poms
import pyopenms_viz  # noqa: F401  # registers plotting backends for pandas

import streamlit as st
from src.common.common import page_setup
from utils.dia_tutorial import (
    reduce_spectra,
    annotate_filtered_spectra,
    apply_sgolay,
    msexperiment_to_dataframe,
)
from utils.dia_peak_picking import perform_xic_peak_picking

page_setup()


st.title("Extracted Ion Chromatogram (XIC) Peak Picking and Feature Scoring")
st.markdown(
    """
This page will walk you through the typical concepts for performing peak picking and feature scoring on extracted ion chromatograms (XICs) in DIA data.
We will use the same small sample dataset (as the previous section) from the [DIA PASEF Evosep dataset (PXD017703)](https://www.ebi.ac.uk/pride/archive/projects/PXD017703) published by Meier et al., 2020. This dataset contains DIA data acquired on a Bruker timsTOF Pro instrument using the Evosep One LC system. The sample file we will use is a small subset of the full experiment run (*20200505_Evosep_200SPD_SG06-16_MLHeLa_200ng_py8_S3-A1_1_2737*), containing spectra in the MS1 precursor m/z range 660-700 and RT range 130-170s. 
"""
)

st.markdown("---")
st.subheader("Load mzML Data")
mz_file = "example-data/mzML/20200505_Evosep_200SPD_SG06-16_filtered_ms1_mz_660_700_rt_130_170_with_ms2.mzML.gz"

peptide = "NTGIIC(UniMod:4)TIGPASR"
precursor_mz = 680.3561
precursor_charge = 2
product_mzs = [386.2034, 487.2623, 600.3464, 701.3941, 861.4247, 974.5088]
product_charges = [1, 1, 1, 1, 1, 1]
product_annotations = ["b4^1", "y5^1", "y6^1", "y7^1", "y8^1", "y9^1"]
prec_mz_tol = 15
prod_mz_tol = 20
target_im = 0.96756938061
im_tol = 0.08

# Session state variables to store the loaded data and prevent re-loading on every button click
if "exp_df_targeted" not in st.session_state:
    st.session_state.exp_df_targeted = None

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

load_clicked = st.button(
    "Load DIA Data and perform targeted extraction", type="primary"
)

if load_clicked or st.session_state.data_loaded:
    with st.spinner("Loading and extracting targeted XICs..."):
        progress = st.progress(0)
        if not st.session_state.data_loaded:
            exp = poms.MSExperiment()
            poms.MzMLFile().load(mz_file, exp)
            progress.progress(20)
            exp_df = exp.to_df(long_format=True)
            progress.progress(30)

            filtered_exp = reduce_spectra(
                exp,
                float(precursor_mz),
                product_mzs,
                float(prec_mz_tol),
                float(prod_mz_tol),
                float(target_im),
                float(im_tol),
            )
            progress.progress(55)

            filtered_exp.updateRanges()
            progress.progress(65)
            exp_df_targeted = filtered_exp.to_df(long_format=True)
            progress.progress(75)

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
            st.session_state.data_loaded = True
            st.session_state.exp_df_targeted = exp_df_targeted
            progress.progress(90)
        else:
            exp_df_targeted = st.session_state.exp_df_targeted

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
                title=f"Smoothed Chromatogram - {peptide} / {precursor_charge}+",
                aggregate_duplicates=False,
                legend_config=dict(title="Transition"),
                backend="ms_plotly",
                show_plot=False,
            )
        )

        progress.progress(100)
        st.plotly_chart(smoothed_chrom_fig, use_container_width=True)

    st.markdown("""
                From the previous section, we learned how to perform targeted data extraction to filter the spectra in our DIA dataset to only those relevant to our peptide of interest (NTGIIC(UniMod:4)TIGPASR) and its corresponding transitions, and we saw how we can visualize the extracted ion chromatograms (XICs) for each transition. However, we have not yet performed any peak picking or feature scoring on these XICs, which are critical steps for determining the quality of our detected features and for downstream quantification and statistical analysis.
                
                Visually, it's pretty easy to tell that this peptide has a good co-elution of its transitions, with a clear peak around 152s. However, when processing thousands of peptides and features in a typical DIA dataset, we need to be able to perform automated peak picking and feature scoring to determine which features are likely to be true positives and which are likely to be false positives. This is where peak picking algorithms and feature scoring metrics come into play.
                """)

    with st.expander("Code:"):
        st.code(
            """exp = poms.MSExperiment()
    poms.MzMLFile().load(mz_file, exp)
    progress.progress(20)
    exp_df = exp.to_df(long_format=True)
    progress.progress(30)

    peptide = "NTGIIC(UniMod:4)TIGPASR"
    precursor_mz = 680.3561
    precursor_charge = 2
    product_mzs = [386.2034, 487.2623, 600.3464, 701.3941, 861.4247, 974.5088]
    product_charges = [1, 1, 1, 1, 1, 1]
    product_annotations = ["b4^1", "y5^1", "y6^1", "y7^1", "y8^1", "y9^1"]
    prec_mz_tol = 15
    prod_mz_tol = 20
    target_im = 0.96756938061
    im_tol = 0.08

    filtered_exp = reduce_spectra(
        exp,
        float(precursor_mz),
        product_mzs,
        float(prec_mz_tol),
        float(prod_mz_tol),
        float(target_im),
        float(im_tol),
    )
    progress.progress(55)

    filtered_exp.updateRanges()
    progress.progress(65)
    exp_df_targeted = filtered_exp.to_df(long_format=True)
    progress.progress(75)

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
    progress.progress(90)

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
            title=f"Smoothed Chromatogram - {peptide} / {precursor_charge}+",
            aggregate_duplicates=False,
            legend_config=dict(title="Transition"),
            backend="ms_plotly",
            show_plot=False,
        )
    )
    """,
            language="python",
        )


st.markdown("---")
st.subheader("Peak Picking")

st.markdown(
    """
    In traditional DIA data processing pipelines, peak picking is often performed on the extracted ion chromatograms (XICs) for each transition to identify potential chromatographic peaks present in each transition. This results in a set of identified peak boundaries for each transition, which are then merged to generate concensus peak boundaries for each precursor feature. 
    """
)

# Plot each XIC trace on a single row in a subplot, instead of having all transitions plotted on the same axes, to better visualize the individual traces
# iterate over the df by the annotation column, and plot each transition in a separate subplot row, with the same x-axis (rt) and y-axis (intensity),
if (
    st.session_state.exp_df_targeted is not None
    and not st.session_state.exp_df_targeted.empty
):
    group_cols = ["ms_level", "annotation", "rt"]
    integrate_col = "intensity"

    # compute smoothed dataframe (used for both combined and per-transition plots)
    exp_df_targeted = st.session_state.exp_df_targeted
    smoothed_df = (
        exp_df_targeted.apply(
            lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna(".")
        )
        .groupby(group_cols)[integrate_col]
        .sum()
        .reset_index()
        .groupby(["annotation", "ms_level"])[group_cols + [integrate_col]]
        .apply(apply_sgolay, window_length=9, polyorder=3)
        .reset_index(drop=True)
    )

    # buttom to perform peak picking on the smoothed dataframe
    peak_picking_clicked = st.button(
        "Perform Peak Picking on Smoothed XICs", type="primary"
    )
    if peak_picking_clicked:
        with st.spinner("Performing peak picking and preparing plots..."):
            progress_pp = st.progress(0)

            picked_peaks_df = perform_xic_peak_picking(
                smoothed_df, intensity_col="smoothed_int"
            )
            progress_pp.progress(50)

            # per-transition subplot: one row per annotation, shared x-axis
            annotations = list(smoothed_df["annotation"].unique())
            fig_sub = make_subplots(
                rows=len(annotations), cols=1, shared_xaxes=True, vertical_spacing=0.02
            )
            for i, ann in enumerate(annotations, start=1):
                df_sub = smoothed_df[smoothed_df["annotation"] == ann]
                fig_sub.add_trace(
                    go.Scatter(
                        x=df_sub["rt"],
                        y=df_sub["smoothed_int"],
                        mode="lines",
                        name=str(ann),
                        showlegend=False,
                    ),
                    row=i,
                    col=1,
                )

                # add vertical lines for picked peaks
                peaks_sub = picked_peaks_df[picked_peaks_df["annotation"] == ann]
                for _, peak in peaks_sub.iterrows():
                    fig_sub.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color="red", width=1, dash="dash"),
                        row=i,
                        col=1,
                        annotation_text=f"FWHM: {peak['FWHM']:.2f}s | Int: {peak['IntegratedIntensity']:.0f}",
                        annotation_position="top left",
                        name=str(ann) + " Peak",
                    )

                    fig_sub.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color="red", width=1, dash="dash"),
                        row=i,
                        col=1,
                        showlegend=False,
                        name=str(ann) + " Peak",
                    )

                fig_sub.update_yaxes(title_text="Intensity", row=i, col=1)
            fig_sub.update_layout(
                height=100 * len(annotations), title_text="XICs per Transition"
            )
            progress_pp.progress(100)
            st.plotly_chart(fig_sub, use_container_width=True)
