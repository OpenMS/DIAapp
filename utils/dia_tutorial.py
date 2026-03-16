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



def bin_3d_trace_df(
    df: pd.DataFrame,
    rt_col: str = "rt",
    mz_col: str = "mz",
    im_col: str = "ion_mobility",
    intensity_col: str = "intensity",
    bins: tuple[int, int, int] = (100, 100, 50),
    intensity_agg: str = "mean",
) -> pd.DataFrame:
    data = df[[rt_col, mz_col, im_col, intensity_col]].dropna().copy()
    if data.empty:
        return pd.DataFrame(columns=[rt_col, mz_col, im_col, "count", "agg_value"])

    coords = data[[rt_col, mz_col, im_col]].to_numpy(dtype=float)
    weights = data[intensity_col].to_numpy(dtype=float)

    counts, edges = np.histogramdd(coords, bins=bins)
    intensity_sum, _ = np.histogramdd(coords, bins=edges, weights=weights)

    if intensity_agg == "mean":
        agg = np.divide(
            intensity_sum,
            counts,
            out=np.zeros_like(intensity_sum, dtype=float),
            where=counts > 0,
        )
    elif intensity_agg == "sum":
        agg = intensity_sum
    elif intensity_agg == "count":
        agg = counts
    else:
        raise ValueError("intensity_agg must be one of: 'mean', 'sum', 'count'")

    x_centers = 0.5 * (edges[0][:-1] + edges[0][1:])
    y_centers = 0.5 * (edges[1][:-1] + edges[1][1:])
    z_centers = 0.5 * (edges[2][:-1] + edges[2][1:])

    ix, iy, iz = np.where(counts > 0)

    return pd.DataFrame({
        rt_col: x_centers[ix],
        mz_col: y_centers[iy],
        im_col: z_centers[iz],
        "count": counts[ix, iy, iz],
        "agg_value": agg[ix, iy, iz],
    })

def add_binned_intensity_trace(fig, binned_df, row, col, name, cmin, cmax):
    """
    Add a single intensity-colored trace to the figure from binned DataFrame.
    The binned DataFrame should have columns: 'rt', 'mz', 'ion_mobility', 'count', 'agg_value'.
    """
    if binned_df.empty:
        return

    color_vals = np.log10(
        np.clip(binned_df["agg_value"].to_numpy(dtype=float), 0, None) + 1.0
    )

    fig.add_trace(
        go.Scatter3d(
            x=binned_df["rt"],
            y=binned_df["mz"],
            z=binned_df["ion_mobility"],
            mode="markers",
            name=name,
            showlegend=False,
            marker=dict(
                symbol="square",
                size=4,
                opacity=0.9,
                color=color_vals,
                colorscale="Viridis",
                cmin=float(cmin),
                cmax=float(cmax),
                showscale=False,   # hide intensity colorbar
                line=dict(width=0),
            ),
            customdata=np.stack(
                [
                    binned_df["count"].to_numpy(dtype=float),
                    binned_df["agg_value"].to_numpy(dtype=float),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "rt: %{x:.4f}<br>"
                "mz: %{y:.4f}<br>"
                "ion_mobility: %{z:.4f}<br>"
                "count: %{customdata[0]:.0f}<br>"
                "mean(intensity): %{customdata[1]:.4f}"
                "<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )
    
def add_binned_annotation_traces(
    fig,
    df: pd.DataFrame,
    row: int,
    col: int,
    annotation_col: str = "annotation",
    rt_col: str = "rt",
    mz_col: str = "mz",
    im_col: str = "ion_mobility",
    intensity_col: str = "intensity",
    bins: tuple[int, int, int] = (100, 100, 50),
    intensity_agg: str = "mean",
    log_color: bool = True,
    color_quantile_clip: tuple[float, float] | None = (0.01, 0.99),
    marker_size: float = 4.0,
    marker_opacity: float = 0.9,
):
    """
    Add one real intensity-colored trace per annotation, plus one dummy legend trace
    per annotation so the legend toggles annotations without changing plot coloring.
    """
    if annotation_col not in df.columns:
        raise ValueError(f"Missing annotation column: {annotation_col}")

    grouped = []
    for ann, subdf in df.groupby(annotation_col, sort=True):
        binned = bin_3d_trace_df(
            subdf,
            rt_col=rt_col,
            mz_col=mz_col,
            im_col=im_col,
            intensity_col=intensity_col,
            bins=bins,
            intensity_agg=intensity_agg,
        )
        if not binned.empty:
            binned["_annotation"] = str(ann)
            grouped.append(binned)

    if not grouped:
        return fig

    all_binned = pd.concat(grouped, ignore_index=True)

    all_color = all_binned["agg_value"].to_numpy(dtype=float)
    if log_color:
        all_color = np.log10(np.clip(all_color, 0, None) + 1.0)

    if color_quantile_clip is not None and len(all_color) > 0:
        q_low, q_high = color_quantile_clip
        cmin, cmax = np.quantile(all_color, [q_low, q_high])
    else:
        cmin, cmax = float(np.min(all_color)), float(np.max(all_color))

    # real traces: intensity-colored, no legend
    annotations = list(all_binned["_annotation"].unique())
    for ann in annotations:
        sub = all_binned[all_binned["_annotation"] == ann].copy()

        color_values = sub["agg_value"].to_numpy(dtype=float)
        if log_color:
            color_values = np.log10(np.clip(color_values, 0, None) + 1.0)

        fig.add_trace(
            go.Scatter3d(
                x=sub[rt_col],
                y=sub[mz_col],
                z=sub[im_col],
                mode="markers",
                name=ann,
                legendgroup=ann,
                showlegend=False,
                marker=dict(
                    symbol="square",
                    size=marker_size,
                    opacity=marker_opacity,
                    color=color_values,
                    colorscale="Viridis",
                    cmin=float(cmin),
                    cmax=float(cmax),
                    showscale=False,
                    line=dict(width=0),
                ),
                customdata=np.stack(
                    [
                        sub["count"].to_numpy(dtype=float),
                        sub["agg_value"].to_numpy(dtype=float),
                        sub["_annotation"].to_numpy(dtype=object),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    f"{rt_col}: %{{x:.4f}}<br>"
                    f"{mz_col}: %{{y:.4f}}<br>"
                    f"{im_col}: %{{z:.4f}}<br>"
                    "count: %{customdata[0]:.0f}<br>"
                    f"{intensity_agg}({intensity_col}): %{{customdata[1]:.4f}}<br>"
                    "annotation: %{customdata[2]}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    # dummy traces: legend only
    dummy_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    for i, ann in enumerate(annotations):
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                name=ann,
                legendgroup=ann,
                showlegend=True,
                hoverinfo="skip",
                marker=dict(
                    symbol="square",
                    size=6,
                    color=dummy_colors[i % len(dummy_colors)],
                    opacity=1.0,
                ),
            ),
            row=row,
            col=col,
        )

    return fig