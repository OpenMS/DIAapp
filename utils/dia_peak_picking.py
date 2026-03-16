from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import pyopenms as poms


def smooth_chromatogram(
    xic_df: pd.DataFrame,
    smoothing_method: str = "sgolay",
    sgolay_window: int = 9,
    sgolay_polyorder: int = 3,
    gauss_width: int = 50,
) -> pd.DataFrame:
    """Applies smoothing to the provided XIC data using the specified method.
    Args:
        xic_df (pd.DataFrame): A DataFrame containing the XIC data with columns 'rt', 'intensity', and 'annotation'.
        smoothing_method (str): The smoothing method to apply ('sgolay' or 'gaussian').
        sgolay_window (int): The window length for Savitzky-Golay smoothing (must be odd).
        sgolay_polyorder (int): The polynomial order for Savitzky-Golay smoothing.
        gauss_width (int): The width of the Gaussian kernel for Gaussian smoothing.
    Returns:
        pd.DataFrame: A DataFrame containing the smoothed XIC data.
    """
    if smoothing_method == "Savitzky-Golay":
        xic_df["intensity"] = xic_df.groupby("annotation")["intensity"].transform(
            lambda x: savgol_filter(
                x, window_length=sgolay_window, polyorder=sgolay_polyorder
            )
        )
    elif smoothing_method == "Gaussian":
        annotations = xic_df["annotation"].unique()
        smoothed_dfs = []
        for ann in annotations:
            df_sub = xic_df[xic_df["annotation"] == ann].copy()
            gauss_filter = poms.GaussFilter()
            gauss_params = gauss_filter.getDefaults()
            gauss_params.setValue("width", gauss_width)
            gauss_filter.setParameters(gauss_params)
            input_chrom = poms.MSChromatogram()
            input_chrom.set_peaks((df_sub["rt"].values, df_sub["intensity"].values))
            gauss_filter.filter(input_chrom)
            smoothed_dfs.append(input_chrom.to_df().assign(annotation=ann))
        xic_df = pd.concat(smoothed_dfs, ignore_index=True)
    elif smoothing_method == "Raw":
        return xic_df.copy()
    else:
        raise ValueError(f"Unsupported smoothing method: {smoothing_method}")

    return xic_df


def perform_xic_peak_picking(
    xic_df: pd.DataFrame,
    intensity_col: str = "intensity",
    picker: poms.PeakPickerChromatogram = None,
) -> pd.DataFrame:
    """Performs peak picking on the provided XIC data using the specified peak picker.
    Args:
        xic_df (pd.DataFrame): A DataFrame containing the XIC data with columns 'rt', 'intensity', and 'annotation'.
        intensity_col (str): The column name for the intensity values.
        picker (poms.PeakPickerChromatogram): An instance of a pyOpenMS PeakPickerChromatogram to use for peak picking.
    Returns:
        pd.DataFrame: A DataFrame containing the picked peak information, including FWHM, integrated intensity, and peak widths.
    """
    if picker is None:
        picker = poms.PeakPickerChromatogram()

    annotations = xic_df["annotation"].unique()
    picked_peaks = []
    picked_chroms_df = []
    for i, ann in enumerate(annotations, start=1):
        df_sub = xic_df[xic_df["annotation"] == ann]

        input_chrom = poms.MSChromatogram()
        input_chrom.set_peaks((df_sub["rt"].values, df_sub[intensity_col].values))
        input_chrom.setMetaValue("annotation", ann)

        picked_chrom = poms.MSChromatogram()
        picker.pickChromatogram(input_chrom, picked_chrom)

        fdas = picked_chrom.getFloatDataArrays()

        # Ensure that the expected number of data arrays are present
        if fdas[0].size() == 0:
            raise ValueError(
                f"No peaks were picked for XIC transition {ann}. Please check the input data and peak picking parameters."
            )

        peaks_apex_rt, peaks_apex_int = picked_chrom.get_peaks()

        for idx in range(fdas[0].size()):
            picked_peaks.append(
                {
                    "annotation": ann,
                    "feature_id": f"feat_{idx + 1}",
                    "apex_rt": peaks_apex_rt[idx],
                    "integrated_intensity": peaks_apex_int[idx],
                    "FWHM": fdas[0].get_data()[idx],
                    "leftWidth": fdas[2].get_data()[idx],
                    "rightWidth": fdas[3].get_data()[idx],
                    "integrated_intensity_fda": fdas[1].get_data()[idx],
                }
            )

    return pd.DataFrame(picked_peaks)
