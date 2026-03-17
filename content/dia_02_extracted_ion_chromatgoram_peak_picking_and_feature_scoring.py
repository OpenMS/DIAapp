from __future__ import annotations

import inspect

import pandas as pd
import numpy as np
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
from utils.dia_peak_picking import (
    smooth_chromatogram,
    create_concensus_chromatogram,
    perform_xic_peak_picking,
    merge_transition_peak_boundaries_to_consensus,
)
from utils.dia_scoring import (
    extract_traces_in_peak,
    build_xcorr_matrices,
    calc_xcorr_shape_score,
    calc_xcorr_shape_weighted,
    calc_xcorr_coelution_score,
    calc_xcorr_coelution_weighted,
    calc_nr_peaks,
    calc_log_sn_score,
    build_mi_matrix,
    calc_mi_score,
    calc_mi_weighted_score,
)

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
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "exp_df_targeted" not in st.session_state:
    st.session_state.exp_df_targeted = None
if "sgolay_frame_length" not in st.session_state:
    st.session_state.sgolay_frame_length = 9
if "sgolay_polynomial_order" not in st.session_state:
    st.session_state.sgolay_polynomial_order = 3
if "gaussian_peak_width" not in st.session_state:
    st.session_state.gaussian_peak_width = 50.0
if "picked_peaks_df" not in st.session_state:
    st.session_state.picked_peaks_df = None
if "consensus_df" not in st.session_state:
    st.session_state.consensus_df = None
if "members_df" not in st.session_state:
    st.session_state.members_df = None
if "smoothed_df" not in st.session_state:
    st.session_state.smoothed_df = None
if "scores_computed" not in st.session_state:
    st.session_state.scores_computed = False

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
    # buttom to perform peak picking on the smoothed dataframe
    peak_picking_clicked = st.button(
        "Perform Peak Picking on Smoothed XICs", type="primary"
    )

    picker = poms.PeakPickerChromatogram()
    picker_params = picker.getDefaults()
    # st.write(picker_params.to_dict())

    st.markdown("#### Peak Picking Parameters")
    smoothing_method = st.selectbox(
        "Smoothing method",
        options=["Savitzky-Golay", "Gaussian", "Raw"],
        index=0,
        key="smoothing_method",
        help="Select the smoothing method to apply to the XICs before peak picking. 'Raw' will perform peak picking on the unsmoothed data.",
    )

    if smoothing_method == "Savitzky-Golay":
        col1, col2 = st.columns(2)
        with col1:
            st.slider(
                "Savitzky-Golay Frame Length",
                min_value=3,
                max_value=21,
                step=2,
                value=9,
                key="sgolay_frame_length",
                help="The window length for the Savitzky-Golay filter. Must be an odd integer.",
            )
        with col2:
            st.slider(
                "Savitzky-Golay Polynomial Order",
                min_value=1,
                max_value=21,
                step=1,
                value=3,
                key="sgolay_polynomial_order",
                help="The polynomial order for the Savitzky-Golay filter. Must be less than the frame length.",
            )
    elif smoothing_method == "Gaussian":
        st.slider(
            "Gaussian Peak Width (s)",
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            value=50.0,
            key="gaussian_peak_width",
            help="The width of the Gaussian kernel for Gaussian smoothing.",
        )

    if smoothing_method in ["Savitzky-Golay", "Gaussian"]:
        picker_params.setValue("method", "corrected")
        if smoothing_method == "Savitzky-Golay":
            picker_params.setValue(
                "sgolay_frame_length", st.session_state.sgolay_frame_length
            )
            picker_params.setValue(
                "sgolay_polynomial_order", st.session_state.sgolay_polynomial_order
            )
        elif smoothing_method == "Gaussian":
            picker_params.setValue("use_gauss", "true")
            picker_params.setValue("gauss_width", st.session_state.gaussian_peak_width)
    else:
        picker_params.setValue("method", "legacy")

    picker.setParameters(picker_params)

    if peak_picking_clicked or st.session_state.picked_peaks_df is not None:
        with st.spinner("Performing peak picking and preparing plots..."):
            progress_pp = st.progress(0)

            try:
                summed_xic_df = (
                    exp_df_targeted.apply(
                        lambda x: x.fillna(0)
                        if x.dtype.kind in "biufc"
                        else x.fillna(".")
                    )
                    .groupby(group_cols)[integrate_col]
                    .sum()
                    .reset_index()
                    .groupby(["annotation", "ms_level"])[group_cols + [integrate_col]]
                    .apply(apply_sgolay, window_length=9, polyorder=3)
                    .reset_index(drop=True)
                )
                st.session_state.picked_peaks_df = perform_xic_peak_picking(
                    summed_xic_df, intensity_col="intensity", picker=picker
                )
                st.dataframe(st.session_state.picked_peaks_df)
            except Exception as e:
                st.error(f"Error performing peak picking: {e}")
                st.stop()
            progress_pp.progress(50)

            smoothed_chromatogram_df = smooth_chromatogram(
                summed_xic_df,
                smoothing_method=smoothing_method,
                sgolay_window=st.session_state.sgolay_frame_length,
                sgolay_polyorder=st.session_state.sgolay_polynomial_order,
                gauss_width=st.session_state.gaussian_peak_width,
            )

            # per-transition subplot: one row per annotation, shared x-axis
            annotations = list(smoothed_chromatogram_df["annotation"].unique())
            fig_sub = make_subplots(
                rows=len(annotations), cols=1, shared_xaxes=True, vertical_spacing=0.02
            )
            for i, ann in enumerate(annotations, start=1):
                df_sub = smoothed_chromatogram_df[
                    smoothed_chromatogram_df["annotation"] == ann
                ]
                fig_sub.add_trace(
                    go.Scatter(
                        x=df_sub["rt"],
                        y=df_sub["intensity"],
                        mode="lines",
                        name=str(ann),
                        showlegend=False,
                    ),
                    row=i,
                    col=1,
                )

                # add vertical lines for picked peaks
                peaks_sub = st.session_state.picked_peaks_df[
                    st.session_state.picked_peaks_df["annotation"] == ann
                ]

                multiple_peaks = len(peaks_sub) > 1
                colors = ["red", "blue", "green", "orange", "purple", "brown"]
                idx = 0
                for _, peak in peaks_sub.iterrows():
                    if multiple_peaks:
                        ann_text = f"Feature {idx}"
                    else:
                        ann_text = f"FWHM: {peak['FWHM']:.2f}s | Int: {peak['integrated_intensity']:.0f}"
                    fig_sub.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color=colors[idx], width=1, dash="dash"),
                        row=i,
                        col=1,
                        annotation_text=ann_text,
                        annotation_position="top left",
                        name=str(ann) + " Peak " + str(idx),
                    )

                    fig_sub.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color=colors[idx], width=1, dash="dash"),
                        row=i,
                        col=1,
                        showlegend=False,
                        name=str(ann) + " Peak " + str(idx),
                    )
                    idx += 1

                fig_sub.update_yaxes(title_text="Intensity", row=i, col=1)
            fig_sub.update_layout(
                height=100 * len(annotations), title_text="XICs per Transition"
            )
            progress_pp.progress(100)
            st.plotly_chart(fig_sub, use_container_width=True)

            if smoothing_method in ["Gaussian"]:
                st.markdown(
                    ":blue[**Note:** The XIC rendered is the raw data, peak picking was performed using the gaussian smoothed data internally.]"
                )

            st.markdown("""
            In the plot above, we show the XICs for each transition in separate subplots, with vertical dashed lines indicating the left and right peak boundaries identified by the peak picking algorithm. The annotation for each peak shows the FWHM and integrated intensity for that peak. Depending on the smoothing method used, the detected peaks and their boundaries may differ. We can also see that the individual peak boundaries may differ slightly across transitions. The next step is to merge these individual peak boundaries across transitions to generate consensus peak boundaries for the precursor feature.
                        """)

            # button to perform merging of transition-level peak boundaries into consensus features
            merge_clicked = st.button(
                "Merge Transition-Level Peak Boundaries into Consensus Features",
                type="primary",
            )
            if merge_clicked:
                consensus_df, members_df = (
                    merge_transition_peak_boundaries_to_consensus(
                        st.session_state.picked_peaks_df,
                        intensity_col="integrated_intensity_fda",
                        min_annotations=2,
                        keep_singletons=True,
                        boundary_mode="weighted_median",
                        min_overlap=0.05,
                        apex_tol_factor=0.4,
                        gap_factor=0.1,
                    )
                )

                # Store in session state for scoring section
                st.session_state.consensus_df = consensus_df
                st.session_state.members_df = members_df

                group_cols = ["ms_level", "annotation", "rt"]
                integrate_col = "intensity"

                # compute smoothed dataframe (used for both combined and per-transition plots)
                exp_df_targeted = st.session_state.exp_df_targeted
                smoothed_df = (
                    exp_df_targeted.apply(
                        lambda x: x.fillna(0)
                        if x.dtype.kind in "biufc"
                        else x.fillna(".")
                    )
                    .groupby(group_cols)[integrate_col]
                    .sum()
                    .reset_index()
                    .groupby(["annotation", "ms_level"])[group_cols + [integrate_col]]
                    .apply(apply_sgolay, window_length=9, polyorder=3)
                    .reset_index(drop=True)
                )

                # Store in session state for scoring section
                st.session_state.smoothed_df = smoothed_df
                st.session_state.scores_computed = (
                    False  # reset when new data is merged
                )

            # Display merged consensus results (persist even after rerun)
            if (
                st.session_state.consensus_df is not None
                and not st.session_state.consensus_df.empty
            ):
                st.subheader("Merged Consensus Features")
                st.dataframe(st.session_state.consensus_df)

                # Get max smoothed_int across all data
                max_smoothed_int = st.session_state.smoothed_df["smoothed_int"].max()
                st.session_state.consensus_df[["apexIntensity"]] = max_smoothed_int

                xic_plot = st.session_state.smoothed_df.plot(
                    kind="chromatogram",
                    x="rt",
                    y="smoothed_int",
                    by="annotation",
                    title=f"Smoothed Chromatogram with Consensus Features - {peptide} / {precursor_charge}+",
                    aggregate_duplicates=False,
                    annotation_data=st.session_state.consensus_df,
                    legend_config=dict(title="Transition"),
                    # feature_config=dict(),
                    backend="ms_plotly",
                    show_plot=False,
                )
                st.session_state.xic_plot = xic_plot

            # Display smoothed chromatogram with consensus (persist even after rerun)
            if st.session_state.consensus_df is not None and hasattr(
                st.session_state, "xic_plot"
            ):
                st.plotly_chart(st.session_state.xic_plot, use_container_width=True)

                st.markdown("""
                In the plot above, we show the smoothed XICs for each transition with the consensus peak boundaries plotted as vertical solid lines. 
                
                But why go through all the trouble of doing individual transition-level peak picking and then merging into consensus features? Why not just perform peak picking on the summed XICs across all transitions for a precursor? Let's try that next.
                """)

            if merge_clicked and st.session_state.consensus_df is not None:
                # Only compute these comparison plots when merge is clicked (expensive computation)
                exp_df_targeted = st.session_state.exp_df_targeted

                # =========================================
                #   Raw concensus with raw peak picking
                concensus_raw_chrom_df = create_concensus_chromatogram(exp_df_targeted)

                # perform peak picking
                picker = poms.PeakPickerChromatogram()
                picker_params = picker.getDefaults()
                picker_params.setValue("method", "legacy")
                picker.setParameters(picker_params)

                concensus_raw_pp_df = perform_xic_peak_picking(
                    concensus_raw_chrom_df, intensity_col="intensity", picker=picker
                )
                concensus_raw_chrom_df[["apexIntensity"]] = concensus_raw_chrom_df[
                    "intensity"
                ].max()

                concensus_raw_chrom_plot = concensus_raw_chrom_df.plot(
                    kind="chromatogram",
                    x="rt",
                    y="intensity",
                    title=f"Concensus Chromatogram with Peak Picking - {peptide} / {precursor_charge}+",
                    aggregate_duplicates=False,
                    annotation_data=concensus_raw_pp_df,
                    legend_config=dict(title="Transition"),
                    backend="ms_plotly",
                    show_plot=False,
                )

                # =========================================
                #   Raw concensus with smoothed peak picking
                picker = poms.PeakPickerChromatogram()
                picker_params = picker.getDefaults()
                picker_params.setValue("method", "corrected")
                picker_params.setValue("sgolay_frame_length", 9)
                picker_params.setValue("sgolay_polynomial_order", 3)
                picker.setParameters(picker_params)
                concensus_raw_pp_df_smooth = perform_xic_peak_picking(
                    concensus_raw_chrom_df, intensity_col="intensity", picker=picker
                )
                concensus_raw_chrom_df[["apexIntensity"]] = concensus_raw_chrom_df[
                    "intensity"
                ].max()
                concensus_raw_chrom_plot_smooth = concensus_raw_chrom_df.plot(
                    kind="chromatogram",
                    x="rt",
                    y="intensity",
                    title=f"Concensus Chromatogram with Smoothed Peak Picking - {peptide} / {precursor_charge}+",
                    aggregate_duplicates=False,
                    annotation_data=concensus_raw_pp_df_smooth,
                    legend_config=dict(title="Transition"),
                    backend="ms_plotly",
                    show_plot=False,
                )

                # =========================================
                #   smoothed concensus with smoothed peak picking
                group_cols = ["ms_level", "annotation", "rt"]
                integrate_col = "intensity"

                exp_df_targeted = st.session_state.exp_df_targeted
                smoothed_df = (
                    exp_df_targeted.apply(
                        lambda x: x.fillna(0)
                        if x.dtype.kind in "biufc"
                        else x.fillna(".")
                    )
                    .groupby(group_cols)[integrate_col]
                    .sum()
                    .reset_index()
                    .groupby(["annotation", "ms_level"])[group_cols + [integrate_col]]
                    .apply(apply_sgolay, window_length=9, polyorder=3)
                    .reset_index(drop=True)
                )

                picker = poms.PeakPickerChromatogram()
                picker_params = picker.getDefaults()
                picker_params.setValue("method", "corrected")
                picker_params.setValue("sgolay_frame_length", 9)
                picker_params.setValue("sgolay_polynomial_order", 3)
                picker.setParameters(picker_params)

                smoothed_for_consensus = smoothed_df[["rt", "annotation"]].copy()
                smoothed_for_consensus["intensity"] = smoothed_df["smoothed_int"]
                smoothed_concensus_chrom_df = create_concensus_chromatogram(
                    smoothed_for_consensus
                )
                smoothed_concensus_pp_df = perform_xic_peak_picking(
                    smoothed_concensus_chrom_df,
                    intensity_col="intensity",
                    picker=picker,
                )
                smoothed_concensus_chrom_df[["apexIntensity"]] = (
                    smoothed_concensus_chrom_df["intensity"].max()
                )

                smoothed_concensus_chrom_plot = smoothed_concensus_chrom_df.plot(
                    kind="chromatogram",
                    x="rt",
                    y="intensity",
                    title=f"Smoothed Concensus Chromatogram with Peak Picking - {peptide} / {precursor_charge}+",
                    aggregate_duplicates=False,
                    annotation_data=smoothed_concensus_pp_df,
                    legend_config=dict(title="Transition"),
                    backend="ms_plotly",
                    show_plot=False,
                )

                # ==================================================
                # Create a 3x2 layout: left column = concensus plots, right column = individual-transition XICs
                combined_concensus_plot = make_subplots(
                    rows=3,
                    cols=2,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    horizontal_spacing=0.08,
                    subplot_titles=[
                        "Raw Concensus Chromatogram (Raw PP)",
                        "Raw XICs (per transition)",
                        "Raw Concensus Chromatogram (Smoothed PP)",
                        "Raw XICs (per transition)",
                        "Smoothed Concensus Chromatogram (Smoothed PP)",
                        "Smoothed XICs (per transition)",
                    ],
                )

                # prepare per-transition raw and smoothed data
                exp_df_targeted = st.session_state.exp_df_targeted
                # raw per-transition XICs (unsmoothed)
                raw_transitions_df = (
                    exp_df_targeted.apply(
                        lambda x: x.fillna(0)
                        if x.dtype.kind in "biufc"
                        else x.fillna(".")
                    )
                    .groupby(["annotation", "rt"])["intensity"]
                    .sum()
                    .reset_index()
                )

                # smoothed per-transition XICs (smoothed_df was computed earlier)
                smoothed_transitions_df = smoothed_df.copy()

                # Row 1: Raw concensus with raw peak picking (left) and raw transitions (right)
                combined_concensus_plot.add_trace(
                    go.Scatter(
                        x=concensus_raw_chrom_df["rt"],
                        y=concensus_raw_chrom_df["intensity"],
                        mode="lines",
                        name="Raw Concensus Chromatogram",
                    ),
                    row=1,
                    col=1,
                )
                for _, peak in concensus_raw_pp_df.iterrows():
                    combined_concensus_plot.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color="red", width=1, dash="dash"),
                        row=1,
                        col=1,
                        annotation_text=f"FWHM: {peak['FWHM']:.2f}s | Int: {peak['integrated_intensity_fda']:.0f}",
                        annotation_position="top left",
                    )
                    combined_concensus_plot.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color="red", width=1, dash="dash"),
                        row=1,
                        col=1,
                    )

                # right column: raw per-transition traces (with legend grouping across all rows)
                annotations = list(raw_transitions_df["annotation"].unique())
                for ai, ann in enumerate(annotations):
                    df_sub = raw_transitions_df[raw_transitions_df["annotation"] == ann]
                    combined_concensus_plot.add_trace(
                        go.Scatter(
                            x=df_sub["rt"],
                            y=df_sub["intensity"],
                            mode="lines",
                            name=str(ann),
                            legendgroup=str(ann),
                            showlegend=True,
                        ),
                        row=1,
                        col=2,
                    )
                # overlay concensus boundaries on right column
                for _, peak in concensus_raw_pp_df.iterrows():
                    combined_concensus_plot.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color="red", width=1, dash="dash"),
                        row=1,
                        col=2,
                    )
                    combined_concensus_plot.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color="red", width=1, dash="dash"),
                        row=1,
                        col=2,
                    )

                # Row 2: Raw concensus with smoothed peak picking (left) and raw transitions (right)
                combined_concensus_plot.add_trace(
                    go.Scatter(
                        x=concensus_raw_chrom_df["rt"],
                        y=concensus_raw_chrom_df["intensity"],
                        mode="lines",
                        name="Raw Concensus Chromatogram",
                    ),
                    row=2,
                    col=1,
                )
                for _, peak in concensus_raw_pp_df_smooth.iterrows():
                    combined_concensus_plot.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color="green", width=1, dash="dash"),
                        row=2,
                        col=1,
                        annotation_text=f"FWHM: {peak['FWHM']:.2f}s | Int: {peak['integrated_intensity_fda']:.0f}",
                        annotation_position="top left",
                    )
                    combined_concensus_plot.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color="green", width=1, dash="dash"),
                        row=2,
                        col=1,
                    )

                # right column: raw per-transition traces (same groups; hide duplicate legend entries)
                for ai, ann in enumerate(annotations):
                    df_sub = raw_transitions_df[raw_transitions_df["annotation"] == ann]
                    combined_concensus_plot.add_trace(
                        go.Scatter(
                            x=df_sub["rt"],
                            y=df_sub["intensity"],
                            mode="lines",
                            name=str(ann),
                            legendgroup=str(ann),
                            showlegend=False,
                        ),
                        row=2,
                        col=2,
                    )
                for _, peak in concensus_raw_pp_df_smooth.iterrows():
                    combined_concensus_plot.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color="green", width=1, dash="dash"),
                        row=2,
                        col=2,
                    )
                    combined_concensus_plot.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color="green", width=1, dash="dash"),
                        row=2,
                        col=2,
                    )

                # Row 3: Smoothed concensus with smoothed peak picking (left) and smoothed transitions (right)
                combined_concensus_plot.add_trace(
                    go.Scatter(
                        x=smoothed_concensus_chrom_df["rt"],
                        y=smoothed_concensus_chrom_df["intensity"],
                        mode="lines",
                        name="Smoothed Concensus Chromatogram",
                    ),
                    row=3,
                    col=1,
                )
                for _, peak in smoothed_concensus_pp_df.iterrows():
                    combined_concensus_plot.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color="blue", width=1, dash="dash"),
                        row=3,
                        col=1,
                        annotation_text=f"FWHM: {peak['FWHM']:.2f}s | Int: {peak['integrated_intensity_fda']:.0f}",
                        annotation_position="top left",
                    )
                    combined_concensus_plot.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color="blue", width=1, dash="dash"),
                        row=3,
                        col=1,
                    )

                # right column: smoothed per-transition traces
                smoothed_annotations = list(
                    smoothed_transitions_df["annotation"].unique()
                )
                for ai, ann in enumerate(smoothed_annotations):
                    df_sub = smoothed_transitions_df[
                        smoothed_transitions_df["annotation"] == ann
                    ]
                    combined_concensus_plot.add_trace(
                        go.Scatter(
                            x=df_sub["rt"],
                            y=df_sub["smoothed_int"],
                            mode="lines",
                            name=str(ann),
                            legendgroup=str(ann),
                            showlegend=False,
                        ),
                        row=3,
                        col=2,
                    )
                for _, peak in smoothed_concensus_pp_df.iterrows():
                    combined_concensus_plot.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color="blue", width=1, dash="dash"),
                        row=3,
                        col=2,
                    )
                    combined_concensus_plot.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color="blue", width=1, dash="dash"),
                        row=3,
                        col=2,
                    )

                # axis labels and layout
                combined_concensus_plot.update_yaxes(
                    title_text="Intensity", row=1, col=1
                )
                combined_concensus_plot.update_yaxes(
                    title_text="Intensity", row=2, col=1
                )
                combined_concensus_plot.update_yaxes(
                    title_text="Intensity", row=3, col=1
                )
                combined_concensus_plot.update_layout(
                    height=900,
                    title_text="Comparison of Concensus Chromatograms and Individual Transition XICs",
                )
                # Store for persistent display
                st.session_state.combined_concensus_plot = combined_concensus_plot

            # Display combined consensus plot (persist even after rerun)
            if hasattr(st.session_state, "combined_concensus_plot"):
                st.plotly_chart(
                    st.session_state.combined_concensus_plot, use_container_width=True
                )

                st.markdown("""
            From the plots above, we can see that performing peak picking on the summed XICs across all transitions (concensus chromatogram) can yield reasonable peak boundaries when performing peak picking with internal smoothing. The raw concensus chromatogram with raw peak picking results in half of the peak being picked, which would result in inaccurate quantification. The raw concensus chromatogram and smooth concensus chromatogram with smoothed peak picking both yield pretty much the same peak boundaries. In this example we used just plain summation to generate the concensus chromatogram, but there are potentially other better methods for generating the concensus chromatogram that could further improve the peak picking results, such as median, weighting the transitions differently based on their intensity or other characteristics. Lastly, even though peak picking on the concensus chromatogram can yield reasonable results in this example, this may not be the case for 1000s of peptide-precursor extracted ion chromatograms. Large scale benchmarking experiments would need to be performed to compare the overall performance.
                """)

st.markdown("---")
st.subheader("Extracted Ion Chromatogram Peak-Group Scoring")


# Check if we have the necessary data to compute scores
if (
    st.session_state.consensus_df is not None
    and not st.session_state.consensus_df.empty
    and st.session_state.smoothed_df is not None
    and not st.session_state.smoothed_df.empty
):
    st.markdown("""
    Once consensus features have been identified through peak boundary merging, 
    we can compute chromatographic quality scores to assess the reliability of each feature.
    """)

    compute_clicked = st.button(
        "Compute Peak-Group Chromatographic Scores",
        type="primary",
    )

    if compute_clicked or st.session_state.scores_computed:
        with st.spinner("Computing chromatographic scores..."):
            try:
                consensus_df = st.session_state.consensus_df
                members_df = (
                    st.session_state.members_df
                    if st.session_state.members_df is not None
                    else pd.DataFrame()
                )
                smoothed_df = st.session_state.smoothed_df

                score_rows = []
                score_details = {}  # store matrices and traces for visualization

                for _, crow in consensus_df.iterrows():
                    cid = crow.get("consensus_feature_id", None)
                    # members for this consensus
                    if not members_df.empty:
                        memb = members_df[members_df["consensus_feature_id"] == cid]
                    else:
                        memb = pd.DataFrame()

                    if not memb.empty:
                        annotations_for_group = list(memb["annotation"].unique())
                    elif "annotations" in crow:
                        annotations_for_group = [
                            a for a in str(crow.get("annotations", "")).split(",") if a
                        ]
                    else:
                        annotations_for_group = list(smoothed_df["annotation"].unique())

                    # extract smoothed traces for this consensus window
                    try:
                        rt_grid, traces = extract_traces_in_peak(
                            smoothed_df,
                            crow,
                            annotation_col="annotation",
                            rt_col="rt",
                            intensity_col="smoothed_int",
                            annotations=annotations_for_group,
                            n_points=101,
                        )
                    except Exception:
                        # fallback to full set
                        rt_grid, traces = extract_traces_in_peak(
                            smoothed_df, crow, intensity_col="smoothed_int"
                        )

                    if len(traces) == 0:
                        continue

                    corr_max, lag_at_max, anns = build_xcorr_matrices(traces)
                    lib_intens = np.array(
                        [
                            float(
                                smoothed_df[smoothed_df["annotation"] == a][
                                    "smoothed_int"
                                ].max()
                            )
                            if not smoothed_df[smoothed_df["annotation"] == a].empty
                            else 0.0
                            for a in anns
                        ]
                    )

                    xcorr_shape = calc_xcorr_shape_score(corr_max)
                    xcorr_coelution = calc_xcorr_coelution_score(lag_at_max)
                    sn_score = calc_log_sn_score(traces)

                    mi_mat, _ = build_mi_matrix(traces)
                    mi_score = calc_mi_score(mi_mat)

                    row_scores = dict(
                        consensus_feature_id=cid,
                        apex_rt=float(crow.get("apex_rt", np.nan)),
                        leftWidth=float(crow.get("leftWidth", np.nan)),
                        rightWidth=float(crow.get("rightWidth", np.nan)),
                        n_members=int(crow.get("n_members", 0)),
                        n_annotations=int(crow.get("n_annotations", 0)),
                        VAR_XCORR_SHAPE=xcorr_shape,
                        VAR_XCORR_SHAPE_WEIGHTED=calc_xcorr_shape_weighted(
                            corr_max, lib_intens
                        ),
                        VAR_XCORR_COELUTION=xcorr_coelution,
                        VAR_XCORR_COELUTION_WEIGHTED=calc_xcorr_coelution_weighted(
                            lag_at_max, lib_intens
                        ),
                        NR_PEAKS=int(memb.shape[0]) if not memb.empty else 0,
                        VAR_LOG_SN_SCORE=sn_score,
                        VAR_MI_SCORE=mi_score,
                        VAR_MI_WEIGHTED_SCORE=calc_mi_weighted_score(
                            mi_mat, lib_intens
                        ),
                    )

                    score_rows.append(row_scores)

                    # Store details for visualization
                    score_details[cid] = {
                        "corr_max": corr_max,
                        "lag_at_max": lag_at_max,
                        "mi_mat": mi_mat,
                        "anns": anns,
                        "traces": traces,
                        "row_scores": row_scores,
                    }

                if score_rows:
                    scores_df = pd.DataFrame(score_rows)

                    st.subheader("Chromatographic Scores per Consensus Feature")
                    st.markdown("""
                    **Score Definitions:**
                    - **VAR_XCORR_SHAPE**: Cross-correlation shape score (mean of max correlations between transitions). Higher = better coelution.
                    - **VAR_XCORR_COELUTION**: Coelution score (mean lag + std of lags). Lower = better coelution.
                    - **VAR_LOG_SN_SCORE**: Log signal-to-noise ratio (mean of S/N per transition). Higher = better.
                    - **VAR_MI_SCORE**: Mutual information score (mean of MI matrix). Higher = more dependent transitions.
                    - **NR_PEAKS**: Number of member peaks in the consensus feature.
                    """)

                    # Display as compact score cards
                    st.markdown("#### Score Summary per Consensus Feature")
                    for idx, (cid, scores) in enumerate(score_details.items()):
                        with st.expander(
                            f"🔍 {cid} (RT: {scores['row_scores']['apex_rt']:.2f}s)"
                        ):
                            # Metric cards in columns
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(
                                    "Shape Score",
                                    f"{scores['row_scores']['VAR_XCORR_SHAPE']:.4f}",
                                    help="Cross-correlation shape (higher is better)",
                                )
                            with col2:
                                st.metric(
                                    "Coelution",
                                    f"{scores['row_scores']['VAR_XCORR_COELUTION']:.4f}",
                                    help="Coelution score (lower is better)",
                                )
                            with col3:
                                st.metric(
                                    "Log S/N",
                                    f"{scores['row_scores']['VAR_LOG_SN_SCORE']:.4f}",
                                    help="Log signal-to-noise (higher is better)",
                                )
                            with col4:
                                st.metric(
                                    "MI Score",
                                    f"{scores['row_scores']['VAR_MI_SCORE']:.4f}",
                                    help="Mutual information (higher = more dependent)",
                                )

                            # Quality summary interpretation
                            st.divider()
                            shape_score = scores["row_scores"]["VAR_XCORR_SHAPE"]
                            coelution_score = scores["row_scores"][
                                "VAR_XCORR_COELUTION"
                            ]
                            sn_score = scores["row_scores"]["VAR_LOG_SN_SCORE"]
                            mi_score = scores["row_scores"]["VAR_MI_SCORE"]

                            shape_status = (
                                "✓ Excellent"
                                if shape_score > 95
                                else ("⚠ Good" if shape_score > 85 else "✗ Fair")
                            )
                            coelution_status = (
                                "✓ Excellent"
                                if coelution_score < 1.0
                                else ("⚠ Good" if coelution_score < 2.0 else "✗ Fair")
                            )
                            sn_status = (
                                "✓ Excellent"
                                if sn_score > 2.0
                                else ("⚠ Good" if sn_score > 1.0 else "✗ Fair")
                            )

                            col_interp1, col_interp2, col_interp3 = st.columns(3)
                            with col_interp1:
                                st.caption(
                                    f"**Shape**: {shape_status} ({shape_score:.2f})"
                                )
                            with col_interp2:
                                st.caption(
                                    f"**Coelution**: {coelution_status} ({coelution_score:.2f})"
                                )
                            with col_interp3:
                                st.caption(f"**S/N**: {sn_status} ({sn_score:.2f})")

                            # Cross-correlation heatmap with text annotations and zoomed colorscale
                            st.write("**Cross-Correlation Matrix (Shape):**")
                            corr_max_arr = scores["corr_max"]
                            # Create text annotations
                            text_annotations = [
                                [f"{val:.2f}" for val in row] for row in corr_max_arr
                            ]
                            fig_xcorr = go.Figure(
                                data=go.Heatmap(
                                    z=corr_max_arr,
                                    x=scores["anns"],
                                    y=scores["anns"],
                                    text=text_annotations,
                                    texttemplate="%{text}",
                                    textfont={"size": 10},
                                    colorscale="Blues",
                                    zmin=np.percentile(
                                        corr_max_arr, 5
                                    ),  # Zoom to actual data range
                                    zmax=100,
                                    hovertemplate="<b>%{x} vs %{y}</b><br>Correlation: %{z:.2f}<extra></extra>",
                                )
                            )
                            fig_xcorr.update_layout(
                                height=400,
                                title="Pairwise Cross-Correlation (max values, higher = better coelution)",
                            )
                            st.plotly_chart(fig_xcorr, use_container_width=True)

                            # Lag-at-maximum heatmap (shows coelution visually)
                            st.write("**Coelution (Lag-at-Maximum):**")
                            lag_arr = np.abs(scores["lag_at_max"])  # Use absolute lags
                            text_lag = [
                                [f"{val:.1f}" for val in row] for row in lag_arr
                            ]
                            fig_lag = go.Figure(
                                data=go.Heatmap(
                                    z=lag_arr,
                                    x=scores["anns"],
                                    y=scores["anns"],
                                    text=text_lag,
                                    texttemplate="%{text}",
                                    textfont={"size": 10},
                                    colorscale="Reds",
                                    hovertemplate="<b>%{x} vs %{y}</b><br>Abs Lag: %{z:.1f} points<extra></extra>",
                                )
                            )
                            fig_lag.update_layout(
                                height=400,
                                title="Absolute Lag at Maximum (lower = better coelution)",
                            )
                            st.plotly_chart(fig_lag, use_container_width=True)

                            # Per-transition S/N bar chart
                            st.write("**Per-Transition Signal-to-Noise Scores:**")
                            sn_per_transition = []
                            trace_names = []
                            for ann, trace in scores["traces"].items():
                                # S/N = peak intensity / baseline noise (90th percentile as baseline)
                                peak_intensity = np.max(trace)
                                baseline = np.percentile(trace, 90)
                                sn_val = (
                                    np.log10(peak_intensity / (baseline + 1e-12))
                                    if baseline > 0
                                    else 0
                                )
                                sn_per_transition.append(sn_val)
                                trace_names.append(str(ann))

                            fig_sn = go.Figure(
                                data=go.Bar(
                                    x=trace_names,
                                    y=sn_per_transition,
                                    text=[f"{val:.2f}" for val in sn_per_transition],
                                    textposition="auto",
                                    marker=dict(
                                        color=sn_per_transition,
                                        colorscale="Greens",
                                        showscale=False,
                                        line=dict(color="darkgreen", width=1),
                                    ),
                                    hovertemplate="<b>%{x}</b><br>Log S/N: %{y:.2f}<extra></extra>",
                                )
                            )
                            fig_sn.update_layout(
                                height=300,
                                title="Signal-to-Noise per Transition (log scale, >1.0 = good)",
                                xaxis_title="Transition",
                                yaxis_title="Log(S/N)",
                                showlegend=False,
                            )
                            st.plotly_chart(fig_sn, use_container_width=True)

                            # Heatmap of MI matrix
                            st.write("**Mutual Information Matrix:**")
                            fig_mi = go.Figure(
                                data=go.Heatmap(
                                    z=scores["mi_mat"],
                                    x=scores["anns"],
                                    y=scores["anns"],
                                    text=[
                                        [f"{val:.3f}" for val in row]
                                        for row in scores["mi_mat"]
                                    ],
                                    texttemplate="%{text}",
                                    textfont={"size": 9},
                                    colorscale="Viridis",
                                    hovertemplate="<b>%{x} vs %{y}</b><br>MI: %{z:.3f}<extra></extra>",
                                )
                            )
                            fig_mi.update_layout(
                                height=400,
                                title="Pairwise Mutual Information (higher = more information dependency)",
                            )
                            st.plotly_chart(fig_mi, use_container_width=True)

                            # Traces visualization
                            st.write("**Standardized Traces in Peak Window:**")
                            fig_traces = go.Figure()
                            for ann, trace in scores["traces"].items():
                                # standardize for display
                                trace_std = (trace - np.mean(trace)) / (
                                    np.std(trace) + 1e-12
                                )
                                fig_traces.add_trace(
                                    go.Scatter(
                                        y=trace_std,
                                        mode="lines",
                                        name=str(ann),
                                        opacity=0.8,
                                    )
                                )
                            fig_traces.update_layout(
                                height=300,
                                title="Standardized Transition Traces (z-scored for visualization)",
                                xaxis_title="Sample Point",
                                yaxis_title="Standardized Intensity",
                            )
                            st.plotly_chart(fig_traces, use_container_width=True)

                    # Summary table
                    st.markdown("#### Detailed Score Table")
                    st.dataframe(scores_df)

                    st.session_state.scores_computed = True
                else:
                    st.info("No consensus features found to score.")

            except Exception as e:
                import traceback

                st.error(f"Error computing chromatographic scores: {e}")
                st.write(traceback.format_exc())
