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


def create_concensus_chromatogram(xic_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a consensus chromatogram by summing intensities across all transitions for each retention time point.
    Args:
        xic_df (pd.DataFrame): A DataFrame containing the XIC data with columns 'rt', 'intensity', and 'annotation'.
    Returns:
        pd.DataFrame: A DataFrame containing the consensus chromatogram with columns 'rt' and 'intensity'.
    """
    consensus_df = xic_df.groupby("rt")["intensity"].sum().reset_index(name="intensity")
    return consensus_df


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


def _weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(values) == 0:
        return np.nan
    if len(values) == 1:
        return float(values[0])

    order = np.argsort(values)
    values = values[order]
    weights = np.clip(weights[order], 0.0, None)

    total = weights.sum()
    if total <= 0:
        return float(np.quantile(values, q))

    cdf = np.cumsum(weights) / total
    return float(np.interp(q, cdf, values))


def _interval_overlap_fraction(left1, right1, left2, right2):
    overlap = max(0.0, min(right1, right2) - max(left1, left2))
    width1 = max(right1 - left1, 1e-12)
    width2 = max(right2 - left2, 1e-12)
    return overlap / min(width1, width2)


def _interval_gap(left1, right1, left2, right2):
    if min(right1, right2) >= max(left1, left2):
        return 0.0
    return max(left1, left2) - min(right1, right2)


def _compute_consensus_window(
    members: pd.DataFrame,
    *,
    apex_col: str,
    left_col: str,
    right_col: str,
    weight_col: str,
    boundary_mode: str = "weighted_median",
    boundary_quantiles: tuple[float, float] = (0.25, 0.75),
):
    weights = np.clip(members[weight_col].to_numpy(dtype=float), 1e-12, None)

    apex_rt = _weighted_quantile(members[apex_col], weights, 0.5)

    if boundary_mode == "weighted_median":
        left = _weighted_quantile(members[left_col], weights, 0.5)
        right = _weighted_quantile(members[right_col], weights, 0.5)
    elif boundary_mode == "weighted_quantile":
        ql, qr = boundary_quantiles
        left = _weighted_quantile(members[left_col], weights, ql)
        right = _weighted_quantile(members[right_col], weights, qr)
    elif boundary_mode == "envelope":
        left = float(members[left_col].min())
        right = float(members[right_col].max())
    else:
        raise ValueError(
            "boundary_mode must be one of "
            "'weighted_median', 'weighted_quantile', 'envelope'"
        )

    if right < left:
        left, right = right, left

    return float(apex_rt), float(left), float(right)


def _pick_best_match_for_annotation(
    candidates: pd.DataFrame,
    *,
    consensus_apex: float,
    consensus_left: float,
    consensus_right: float,
    apex_col: str,
    left_col: str,
    right_col: str,
    intensity_col: str,
    min_overlap: float,
    apex_tol: float | None,
    apex_tol_factor: float,
    min_apex_tol: float,
    max_gap: float | None,
    gap_factor: float,
):
    """
    Choose at most one peak from one annotation for the current consensus group.
    """
    if candidates.empty:
        return None

    width = max(consensus_right - consensus_left, 1e-12)
    dyn_apex_tol = (
        apex_tol if apex_tol is not None else max(min_apex_tol, apex_tol_factor * width)
    )
    dyn_max_gap = max_gap if max_gap is not None else max(0.0, gap_factor * width)

    best_idx = None
    best_score = -np.inf

    for idx, row in candidates.iterrows():
        left = float(row[left_col])
        right = float(row[right_col])
        apex = float(row[apex_col])
        inten = float(row[intensity_col])

        overlap = _interval_overlap_fraction(
            left, right, consensus_left, consensus_right
        )
        gap = _interval_gap(left, right, consensus_left, consensus_right)
        apex_diff = abs(apex - consensus_apex)

        valid = (overlap >= min_overlap) or (
            (gap <= dyn_max_gap) and (apex_diff <= dyn_apex_tol)
        )
        if not valid:
            continue

        # Prefer overlap, then closer apex, then stronger intensity.
        score = (
            5.0 * overlap
            - 1.5 * (apex_diff / max(dyn_apex_tol, 1e-12))
            - 1.0 * (gap / max(dyn_max_gap, 1e-12) if dyn_max_gap > 0 else 0.0)
            + 0.05 * np.log1p(max(inten, 0.0))
        )

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def _grow_one_consensus_group(
    available_df: pd.DataFrame,
    seed_idx: int,
    *,
    annotation_col: str,
    apex_col: str,
    left_col: str,
    right_col: str,
    intensity_col: str,
    boundary_mode: str,
    boundary_quantiles: tuple[float, float],
    min_overlap: float,
    apex_tol: float | None,
    apex_tol_factor: float,
    min_apex_tol: float,
    max_gap: float | None,
    gap_factor: float,
    max_refine_iter: int,
):
    """
    Build a single consensus group starting from one seed peak.
    """
    seed = available_df.loc[seed_idx]

    selected = {seed_idx}
    consensus_apex = float(seed[apex_col])
    consensus_left = float(seed[left_col])
    consensus_right = float(seed[right_col])

    seed_annotation = seed[annotation_col]

    for _ in range(max_refine_iter):
        previous = selected.copy()
        selected = {seed_idx}

        for ann, ann_df in available_df.groupby(annotation_col, sort=False):
            if ann == seed_annotation:
                continue

            best_idx = _pick_best_match_for_annotation(
                ann_df,
                consensus_apex=consensus_apex,
                consensus_left=consensus_left,
                consensus_right=consensus_right,
                apex_col=apex_col,
                left_col=left_col,
                right_col=right_col,
                intensity_col=intensity_col,
                min_overlap=min_overlap,
                apex_tol=apex_tol,
                apex_tol_factor=apex_tol_factor,
                min_apex_tol=min_apex_tol,
                max_gap=max_gap,
                gap_factor=gap_factor,
            )
            if best_idx is not None:
                selected.add(best_idx)

        members = available_df.loc[sorted(selected)]
        consensus_apex, consensus_left, consensus_right = _compute_consensus_window(
            members,
            apex_col=apex_col,
            left_col=left_col,
            right_col=right_col,
            weight_col=intensity_col,
            boundary_mode=boundary_mode,
            boundary_quantiles=boundary_quantiles,
        )

        if selected == previous:
            break

    members = available_df.loc[sorted(selected)]
    consensus_apex, consensus_left, consensus_right = _compute_consensus_window(
        members,
        apex_col=apex_col,
        left_col=left_col,
        right_col=right_col,
        weight_col=intensity_col,
        boundary_mode=boundary_mode,
        boundary_quantiles=boundary_quantiles,
    )

    return sorted(selected), consensus_apex, consensus_left, consensus_right


def merge_transition_peak_boundaries_to_consensus(
    df: pd.DataFrame,
    *,
    annotation_col: str = "annotation",
    feature_id_col: str = "feature_id",
    apex_col: str = "apex_rt",
    left_col: str = "leftWidth",
    right_col: str = "rightWidth",
    intensity_col: str | None = None,
    min_annotations: int = 2,
    keep_singletons: bool = True,
    boundary_mode: str = "weighted_median",
    boundary_quantiles: tuple[float, float] = (0.25, 0.75),
    min_overlap: float = 0.05,
    apex_tol: float | None = None,
    apex_tol_factor: float = 0.5,
    min_apex_tol: float = 0.5,
    max_gap: float | None = None,
    gap_factor: float = 0.15,
    max_refine_iter: int = 3,
):
    """
    Merge pre-picked transition-level peak boundaries into all consensus features. This is in similar spirit to how MRMTransitionGroupPicker works in OpenSWATH.

    Parameters
    ----------
    df : pd.DataFrame
        One row per picked peak in one annotation/transition chromatogram.
        Expected columns include:
            annotation, feature_id, apex_rt, leftWidth, rightWidth,
            and an intensity column.
    annotation_col : str
        Column identifying the transition/annotation.
    feature_id_col : str
        Original per-annotation feature identifier.
    apex_col, left_col, right_col : str
        Columns describing the picked peak apex and boundaries.
    intensity_col : str or None
        Column used to rank seeds and weight consensus boundaries.
        If None, tries:
            - integrated_intensity_fda
            - integrated_intensity
    min_annotations : int
        Minimum number of distinct annotations required to accept a multi-trace
        consensus group.
    keep_singletons : bool
        If True, keep seed-only groups when they do not reach min_annotations.
    boundary_mode : str
        How to derive consensus boundaries:
            - weighted_median
            - weighted_quantile
            - envelope
    boundary_quantiles : tuple[float, float]
        Used only for boundary_mode='weighted_quantile'.
    min_overlap : float
        Minimum overlap fraction needed for a direct match.
    apex_tol : float or None
        Absolute apex tolerance. If None, uses apex_tol_factor * current width.
    apex_tol_factor : float
        Dynamic apex tolerance as a fraction of the current consensus width.
    min_apex_tol : float
        Lower bound on the dynamic apex tolerance.
    max_gap : float or None
        Absolute allowed gap between intervals. If None, uses gap_factor * width.
    gap_factor : float
        Dynamic allowed gap as a fraction of the current consensus width.
    max_refine_iter : int
        Number of matching/refinement rounds per seed.

    Returns
    -------
    consensus_df : pd.DataFrame
        One row per consensus feature.
    members_df : pd.DataFrame
        One row per original picked peak assigned to a consensus feature.
    """
    required = {annotation_col, feature_id_col, apex_col, left_col, right_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if intensity_col is None:
        if "integrated_intensity_fda" in df.columns:
            intensity_col = "integrated_intensity_fda"
        elif "integrated_intensity" in df.columns:
            intensity_col = "integrated_intensity"
        else:
            raise ValueError(
                "Could not infer intensity_col. "
                "Please provide intensity_col explicitly."
            )

    work = df.copy().reset_index(drop=False).rename(columns={"index": "_input_row"})
    for col in [apex_col, left_col, right_col, intensity_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(
        subset=[
            annotation_col,
            feature_id_col,
            apex_col,
            left_col,
            right_col,
            intensity_col,
        ]
    ).copy()
    work = work[work[right_col] > work[left_col]].copy()

    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    unassigned = set(work.index.tolist())
    consensus_rows = []
    member_tables = []
    consensus_num = 0

    while unassigned:
        available = work.loc[sorted(unassigned)].copy()

        # strongest remaining peak seeds the next consensus feature
        seed_idx = available[intensity_col].idxmax()

        selected_idxs, consensus_apex, consensus_left, consensus_right = (
            _grow_one_consensus_group(
                available,
                seed_idx,
                annotation_col=annotation_col,
                apex_col=apex_col,
                left_col=left_col,
                right_col=right_col,
                intensity_col=intensity_col,
                boundary_mode=boundary_mode,
                boundary_quantiles=boundary_quantiles,
                min_overlap=min_overlap,
                apex_tol=apex_tol,
                apex_tol_factor=apex_tol_factor,
                min_apex_tol=min_apex_tol,
                max_gap=max_gap,
                gap_factor=gap_factor,
                max_refine_iter=max_refine_iter,
            )
        )

        members = available.loc[selected_idxs].copy()
        n_annotations = int(members[annotation_col].nunique())

        # If group is too small and we do not keep singletons, discard only the seed.
        if n_annotations < min_annotations and not keep_singletons:
            unassigned.remove(seed_idx)
            continue

        consensus_num += 1
        consensus_id = f"consensus_{consensus_num}"

        seed_row = work.loc[seed_idx]
        total_intensity = float(members[intensity_col].sum())

        consensus_rows.append(
            {
                "consensus_feature_id": consensus_id,
                "seed_annotation": seed_row[annotation_col],
                "seed_feature_id": seed_row[feature_id_col],
                "apex_rt": consensus_apex,
                "leftWidth": consensus_left,
                "rightWidth": consensus_right,
                "consensus_width": consensus_right - consensus_left,
                "n_members": int(len(members)),
                "n_annotations": n_annotations,
                "total_intensity": total_intensity,
                "annotations": ",".join(
                    sorted(map(str, members[annotation_col].unique()))
                ),
            }
        )

        members["consensus_feature_id"] = consensus_id
        members["source_peak_id"] = (
            members[annotation_col].astype(str)
            + "::"
            + members[feature_id_col].astype(str)
            + "::row"
            + members["_input_row"].astype(str)
        )
        member_tables.append(members)

        # Remove all members of this consensus group, then continue.
        unassigned -= set(selected_idxs)

    consensus_df = pd.DataFrame(consensus_rows)
    members_df = (
        pd.concat(member_tables, ignore_index=True) if member_tables else pd.DataFrame()
    )

    if not consensus_df.empty:
        consensus_df = consensus_df.sort_values("apex_rt").reset_index(drop=True)

    if not members_df.empty:
        members_df = members_df.sort_values(
            ["consensus_feature_id", apex_col, annotation_col]
        ).reset_index(drop=True)

    return consensus_df, members_df
