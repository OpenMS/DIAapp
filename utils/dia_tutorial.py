from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import pyopenms as poms
import plotly.graph_objects as go

def mz_extraction_windows(
    target_mz: float, tol_ppm: float
):
    """Calculate m/z extraction window bounds in Dalton given target m/z and ppm tolerance."""
    # tol_ppm is parts-per-million; convert to Dalton: mz * (ppm / 1e6)
    tol_da = target_mz * tol_ppm / 1e6
    lower_bound = target_mz - tol_da
    upper_bound = target_mz + tol_da
    return lower_bound, upper_bound

def rt_extraction_windows(
    target_rt: float, tol_seconds: float
):
    """Calculate RT extraction window bounds in seconds given target RT and tolerance."""
    lower_bound = target_rt - tol_seconds / 2.0
    upper_bound = target_rt + tol_seconds / 2.0
    return lower_bound, upper_bound

def im_extraction_windows(
    target_im: float, tol: float
):
    """Calculate ion mobility extraction window bounds given target IM and tolerance."""
    lower_bound = target_im - tol / 2.0
    upper_bound = target_im + tol / 2.0
    return lower_bound, upper_bound

def filter_spectrum(
    spectrum: poms.MSSpectrum,
    target_precursor_mz: float,
    target_product_mzs: list[float],
    prec_mz_tol: float,
    prod_mz_tol: float,
    target_im: float | None = None,
    im_tol: float | None = None,
) -> poms.MSSpectrum:
    """Filter a spectrum for peaks around target precursor/product masses."""
    # Calculate m/z extraction windows
    prec_mz_low, prec_mz_high = mz_extraction_windows(target_precursor_mz, prec_mz_tol)
    prod_mz_windows = [mz_extraction_windows(mz, prod_mz_tol) for mz in target_product_mzs]

    # Calculate ion mobility extraction window if provided
    if target_im is not None and im_tol is not None:
        im_low, im_high = im_extraction_windows(target_im, im_tol)
    else:
        im_low, im_high = None, None

    mz_array = spectrum.get_mz_array()
    # build intensity array robustly
    try:
        int_array = spectrum.get_intensity_array()
    except Exception:
        int_array = np.array([spectrum.getIntensity(i) for i in range(spectrum.size())])

    # ion mobility array (if available)
    im_array = None
    if im_low is not None and im_high is not None:
        try:
            im_array = spectrum.get_drift_time_array()
        except Exception:
            # fallback: try to read first FloatDataArray named 'Ion Mobility'
            fda_list = spectrum.getFloatDataArrays()
            if fda_list:
                im_array = np.array(fda_list[0].get_data())

    # Handle MS1: keep peaks within precursor window (and IM if provided)
    if spectrum.getMSLevel() == 1:
        mz_mask = (mz_array >= prec_mz_low) & (mz_array <= prec_mz_high)
        if im_array is not None:
            im_mask = (im_array >= im_low) & (im_array <= im_high)
            mask = mz_mask & im_mask
        else:
            mask = mz_mask

        if not mask.any():
            return poms.MSSpectrum()

        filtered_mz = mz_array[mask]
        filtered_int = np.asarray(int_array)[mask]

        spec_out = poms.MSSpectrum()
        spec_out.set_peaks((filtered_mz, filtered_int))
        spec_out.setMSLevel(1)
        spec_out.setRT(spectrum.getRT())

        if im_array is not None:
            fda = poms.FloatDataArray()
            fda.set_data(np.asarray(im_array)[mask].astype(np.float32))
            fda.setName("Ion Mobility")
            spec_out.setFloatDataArrays([fda])

        return spec_out

    # Handle MS2: respect spectrum isolation window (SWATH) and product windows
    elif spectrum.getMSLevel() == 2:
        precs = spectrum.getPrecursors()
        if not precs:
            return poms.MSSpectrum()
        current_prec = precs[0]
        swath_lower = current_prec.getMZ() - current_prec.getIsolationWindowLowerOffset()
        swath_upper = current_prec.getMZ() + current_prec.getIsolationWindowUpperOffset()

        if not (target_precursor_mz >= swath_lower and target_precursor_mz <= swath_upper):
            return poms.MSSpectrum()

        # build product mask by OR-ing all product windows
        prod_mask = np.zeros_like(mz_array, dtype=bool)
        for low, high in prod_mz_windows:
            prod_mask |= (mz_array >= low) & (mz_array <= high)

        if im_array is not None:
            im_mask = (im_array >= im_low) & (im_array <= im_high)
            mask = prod_mask & im_mask
        else:
            mask = prod_mask

        if not mask.any():
            return poms.MSSpectrum()

        filtered_mz = mz_array[mask]
        filtered_int = np.asarray(int_array)[mask]

        spec_out = poms.MSSpectrum()
        spec_out.set_peaks((filtered_mz, filtered_int))
        spec_out.setMSLevel(2)
        spec_out.setRT(spectrum.getRT())

        if im_array is not None:
            fda = poms.FloatDataArray()
            fda.set_data(np.asarray(im_array)[mask].astype(np.float32))
            fda.setName("Ion Mobility")
            spec_out.setFloatDataArrays([fda])

        return spec_out

def reduce_spectra(
    exp: poms.MSExperiment,
    target_precursor_mz: float,
    target_product_mzs: list[float],
    prec_mz_tol: float,
    prod_mz_tol: float,
    target_im: float | None = None,
    im_tol: float | None = None,
    tartget_rt: float | None = None,
    rt_tol: float | None = None,
):
    """Filter spectra in an MSExperiment for peaks around target precursor/product masses."""
    filtered_exp = poms.MSExperiment()
    for spectrum in exp.getSpectra():
        if tartget_rt is not None and rt_tol is not None:
            rt_low, rt_high = rt_extraction_windows(tartget_rt, rt_tol)
            if not (rt_low <= spectrum.getRT() <= rt_high):
                continue  # skip spectra outside RT window
        filtered_spectrum = filter_spectrum(
            spectrum,
            target_precursor_mz,
            target_product_mzs,
            prec_mz_tol,
            prod_mz_tol,
            target_im,
            im_tol,
        )
        if filtered_spectrum.size() > 0:
            filtered_exp.addSpectrum(filtered_spectrum)
    return filtered_exp

def annotate_filtered_spectra(
    filtered_df: pd.DataFrame,
    precursor_mz: float,
    precursor_charge: int,
    product_mzs: list[float],
    product_charges: list[int],
    product_annotations: list[str],
    prec_mz_tol: float,
    prod_mz_tol: float,
) -> pd.DataFrame:
    """Annotate filtered spectra with precursor/fragment assignment and mass error."""
    out = filtered_df.copy()
    
    # Ensure an `ion_mobility` column exists in the output when the input
    # DataFrame contains it (or add a numeric column if missing). This makes
    # downstream code that expects an `ion_mobility` column robust.
    if "ion_mobility" in filtered_df.columns:
        out["ion_mobility"] = pd.to_numeric(out["ion_mobility"], errors="coerce")
    else:
        out["ion_mobility"] = np.nan
    out["annotation"] = pd.NA
    out["target_mz"] = np.nan
    out["target_charge"] = pd.NA
    out["mz_error_da"] = np.nan
    out["mz_error_ppm"] = np.nan

    product_targets = list(zip(product_mzs, product_charges, product_annotations))

    for idx, row in out.iterrows():
        obs_mz = row["mz"]
        ms_level = row["ms_level"]

        if ms_level == 1:
            tol_da = precursor_mz * prec_mz_tol / 1e6
            err_da = obs_mz - precursor_mz
            err_ppm = (err_da / precursor_mz) * 1e6

            if abs(err_da) <= tol_da:
                out.at[idx, "annotation"] = "Precursor"
                out.at[idx, "target_mz"] = precursor_mz
                out.at[idx, "target_charge"] = precursor_charge
                out.at[idx, "mz_error_da"] = err_da
                out.at[idx, "mz_error_ppm"] = err_ppm

        elif ms_level == 2:
            best_match = None
            best_abs_ppm = np.inf

            for target_mz, target_charge, annotation in product_targets:
                tol_da = target_mz * prod_mz_tol / 1e6
                err_da = obs_mz - target_mz

                if abs(err_da) <= tol_da:
                    err_ppm = (err_da / target_mz) * 1e6
                    abs_ppm = abs(err_ppm)

                    if abs_ppm < best_abs_ppm:
                        best_abs_ppm = abs_ppm
                        best_match = {
                            "annotation": annotation,
                            "target_mz": target_mz,
                            "target_charge": target_charge,
                            "mz_error_da": err_da,
                            "mz_error_ppm": err_ppm,
                        }

            if best_match is not None:
                out.at[idx, "annotation"] = best_match["annotation"]
                out.at[idx, "target_mz"] = best_match["target_mz"]
                out.at[idx, "target_charge"] = best_match["target_charge"]
                out.at[idx, "mz_error_da"] = best_match["mz_error_da"]
                out.at[idx, "mz_error_ppm"] = best_match["mz_error_ppm"]

    return out


def apply_sgolay(
    group: pd.DataFrame,
    along_col: str = "rt",
    window_length: int = 11,
    polyorder: int = 4,
) -> pd.DataFrame:
    """Apply Savitzky-Golay smoothing to intensity values inside each group."""
    group = group.sort_values(along_col)
    group["smoothed_int"] = savgol_filter(
        group["intensity"], window_length=window_length, polyorder=polyorder
    )
    return group

def msexperiment_to_dataframe(
    exp: poms.MSExperiment,
) -> pd.DataFrame:
    """Convert an MSExperiment to a long-format DataFrame with one row per peak."""
    rows = []
    for spectrum in exp.getSpectra():
        rt = spectrum.getRT()
        ms_level = spectrum.getMSLevel()

        mz_array = spectrum.get_mz_array()
        try:
            int_array = spectrum.get_intensity_array()
        except Exception:
            int_array = np.array([spectrum.getIntensity(i) for i in range(spectrum.size())])

        im_array = None
        try:
            im_array = spectrum.get_drift_time_array()
        except Exception:
            fda_list = spectrum.getFloatDataArrays()
            if fda_list:
                im_array = np.array(fda_list[0].get_data())

        for i in range(spectrum.size()):
            row = {
                "rt": rt,
                "mz": mz_array[i],
                "intensity": int_array[i],
                "ms_level": ms_level,
            }
            if im_array is not None and i < len(im_array):
                row["ion_mobility"] = im_array[i]
            else:
                row["ion_mobility"] = np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    return df

def plot_3d_binned_xic_scatter(
    df: pd.DataFrame,
    rt_col: str = "rt",
    mz_col: str = "mz",
    im_col: str = "ion_mobility",
    intensity_col: str = "intensity",
    bins: tuple[int, int, int] = (120, 120, 60),
    intensity_agg: str = "mean",   # "mean" | "sum" | "count"
    log_color: bool = True,
    color_quantile_clip: tuple[float, float] | None = (0.01, 0.99),
    marker_size: float = 4.0,
    marker_opacity: float = 0.9,
    title: str = "3D binned RT-m/z-ion mobility scatter",
):
    """
    Plot a dense RT-m/z-ion mobility dataframe as a 3D binned scatter plot.

    The data are voxelized in 3D:
      x = rt
      y = mz
      z = ion_mobility

    Each non-empty voxel is plotted at its bin center as a square marker.
    Marker color is based on an aggregated intensity value per voxel.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    rt_col, mz_col, im_col, intensity_col : str
        Column names.
    bins : tuple[int, int, int]
        Number of bins in (rt, mz, ion_mobility).
    intensity_agg : str
        How to aggregate intensity within each 3D bin:
        - "mean": average intensity of points in the voxel
        - "sum": total intensity in the voxel
        - "count": number of points in the voxel
    log_color : bool
        If True, color by log10(value + 1).
    color_quantile_clip : tuple[float, float] | None
        Optional color clipping by quantiles to reduce the effect of outliers.
    marker_size : float
        Marker size for plotted voxel centers.
    marker_opacity : float
        Marker opacity.
    title : str
        Figure title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure.
    binned_df : pd.DataFrame
        Dataframe of non-empty voxel centers and aggregated values.
    """
    required = [rt_col, mz_col, im_col, intensity_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[required].dropna().copy()

    x = data[rt_col].to_numpy(dtype=float)
    y = data[mz_col].to_numpy(dtype=float)
    z = data[im_col].to_numpy(dtype=float)
    w = data[intensity_col].to_numpy(dtype=float)

    coords = np.column_stack([x, y, z])

    # Count how many points land in each voxel
    counts, edges = np.histogramdd(coords, bins=bins)

    # Weighted sum of intensity per voxel
    intensity_sum, _ = np.histogramdd(coords, bins=edges, weights=w)

    if intensity_agg == "mean":
        agg = np.divide(
            intensity_sum,
            counts,
            out=np.zeros_like(intensity_sum, dtype=float),
            where=counts > 0,
        )
        agg_label = f"mean({intensity_col})"
    elif intensity_agg == "sum":
        agg = intensity_sum
        agg_label = f"sum({intensity_col})"
    elif intensity_agg == "count":
        agg = counts
        agg_label = "count"
    else:
        raise ValueError("intensity_agg must be one of: 'mean', 'sum', 'count'")

    # Bin centers
    x_centers = 0.5 * (edges[0][:-1] + edges[0][1:])
    y_centers = 0.5 * (edges[1][:-1] + edges[1][1:])
    z_centers = 0.5 * (edges[2][:-1] + edges[2][1:])

    # Indices of non-empty bins
    ix, iy, iz = np.where(counts > 0)

    binned_df = pd.DataFrame({
        rt_col: x_centers[ix],
        mz_col: y_centers[iy],
        im_col: z_centers[iz],
        "count": counts[ix, iy, iz],
        "agg_value": agg[ix, iy, iz],
    })

    color_values = binned_df["agg_value"].to_numpy(dtype=float)
    color_title = agg_label

    if log_color:
        color_values = np.log10(np.clip(color_values, a_min=0, a_max=None) + 1.0)
        color_title = f"log10({agg_label} + 1)"

    marker_kwargs = dict(
        size=marker_size,
        opacity=marker_opacity,
        symbol="square",
        color=color_values,
        colorscale="Viridis",
        colorbar=dict(title=color_title),
    )

    if color_quantile_clip is not None and len(color_values) > 0:
        q_low, q_high = color_quantile_clip
        cmin, cmax = np.quantile(color_values, [q_low, q_high])
        marker_kwargs["cmin"] = float(cmin)
        marker_kwargs["cmax"] = float(cmax)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=binned_df[rt_col],
                y=binned_df[mz_col],
                z=binned_df[im_col],
                mode="markers",
                marker=marker_kwargs,
                customdata=np.stack(
                    [
                        binned_df["count"].to_numpy(dtype=float),
                        binned_df["agg_value"].to_numpy(dtype=float),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    f"{rt_col}: %{{x:.4f}}<br>"
                    f"{mz_col}: %{{y:.4f}}<br>"
                    f"{im_col}: %{{z:.4f}}<br>"
                    "count: %{customdata[0]:.0f}<br>"
                    f"{agg_label}: %{{customdata[1]:.4f}}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        scene=dict(
            xaxis_title=rt_col,
            yaxis_title=mz_col,
            zaxis_title=im_col,
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig, binned_df