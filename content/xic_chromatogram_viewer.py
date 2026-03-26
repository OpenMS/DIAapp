"""
xic_chromatogram_viewer.py  —  XIC Chromatogram Viewer

Upload one or more .xic Parquet files produced by OpenMS (pyopenms >= 3.6.0),
select a precursor of interest from the shared analyte list, and view aligned
chromatograms for every uploaded file side-by-side.

Requires pyopenms >= 3.6.0 for XICParquetFile.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from scipy.signal import savgol_filter

from src.common.common import page_setup
from src import fileupload

# -- pyopenms version guard ----------------------------------------------------
try:
    import pyopenms as poms
    import pyopenms_viz  # noqa: F401 – registers ms_plotly backend

    _poms_version = tuple(int(x) for x in poms.__version__.split(".")[:2])
    if _poms_version < (3, 6):
        st.error(
            f"**pyopenms {poms.__version__} detected** - Currently the XIC viewer only supports OpenMS XIC parquet files.`XICParquetFile` requires pyopenms ≥ 3.6.0. Please upgrade: `pip install pyopenms --upgrade`"
        )
        st.stop()
    POMS_OK = True
except ImportError:
    POMS_OK = False

page_setup()

# -----------------------------------------------------------------------------
# Page header

st.title("📊 XIC Chromatogram Viewer")
st.markdown(
    """
Upload one or more **`.xic` Parquet files** (OpenMS XIC format, requires
pyopenms ≥ 3.6.0), choose a peptide precursor from the shared analyte list,
and visualise the extracted ion chromatograms for every file simultaneously.
"""
)

with st.expander("ℹ️ About XIC Parquet files"):
    st.markdown(
        """
        `.xic` files are Parquet-format chromatogram files written by OpenMS
        (e.g. via `XICParquetFile`). Each file contains:
        - **Analyte table** — one row per precursor with sequence, charge, decoy flag,
          transition IDs, annotation, etc.
        - **Chromatogram data** — retention time and intensity arrays per transition.

        Multiple files can be loaded simultaneously (e.g. replicate runs or
        conditions). The viewer will attempt to locate the selected precursor in
        every file and render a chromatogram panel for each; a clear error is
        shown for any file where the precursor is absent.
        """
    )

if not POMS_OK:
    st.error("pyopenms is not installed. Install with `pip install pyopenms`.")
    st.stop()

st.markdown("---")

# -----------------------------------------------------------------------------
# Session state

_defaults: dict = {
    "xic_tmp_paths": [],  # list of (display_name, tmp_path) for uploaded files
    "shared_analytes": None,  # pd.DataFrame — union/intersection of analytes
    "file_analytes": {},  # {display_name: pd.DataFrame}
    "files_loaded": False,
    "smooth_toggle": True,  # Savitzky-Golay smoothing toggle
    "sgolay_order": 3,  # Polynomial order for Savitzky-Golay filter
    "sgolay_window": 9,  # Window length for Savitzky-Golay filter
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# -----------------------------------------------------------------------------
# Helper functions


def _load_analytes(tmp_path: str) -> pd.DataFrame:
    """Open an XIC parquet file and return its analyte dataframe (one row per precursor)."""
    xic_h = poms.XICParquetFile(tmp_path)
    return xic_h.get_analyte_df()


def _make_label(row: pd.Series) -> str:
    """Human-readable dropdown label for a precursor row."""
    seq = row.get("modified_sequence", "unknown")
    charge = row.get("precursor_charge", "?")
    decoy = row.get("precursor_decoy", 0)
    pid = row.get("precursor_id", "")
    tag = " [decoy]" if int(decoy) == 1 else ""
    return f"{seq}/{charge}{tag}  (id={pid})"


def _build_shared_analytes(
    file_analytes: dict[str, pd.DataFrame], mode: str = "intersection"
) -> pd.DataFrame:
    """
    Compute the shared analyte set across files.

    mode="intersection"  — only precursors present in ALL files
    mode="union"         — all precursors from ANY file (files that lack one will show an error)
    """
    if not file_analytes:
        return pd.DataFrame()

    # Use the first file's analyte table as the reference; keep unique precursors
    frames = list(file_analytes.values())

    if mode == "intersection":
        # Precursor IDs present in every file
        id_sets = [set(df["precursor_id"].tolist()) for df in frames]
        common = id_sets[0].intersection(*id_sets[1:])
        ref = frames[0][frames[0]["precursor_id"].isin(common)].copy()
    else:
        # Union — merge all, deduplicate by precursor_id
        ref = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["precursor_id"]
        )

    return ref.sort_values(
        ["modified_sequence", "precursor_charge", "precursor_decoy"]
    ).reset_index(drop=True)


def _plot_chromatogram(
    xic_h: "poms.XICParquetFile",
    precursor_id: int,
    title_suffix: str = "",
    smooth_enabled: bool = False,
    sgolay_order: int = 3,
    sgolay_window: int = 9,
) -> Optional[object]:
    """
    Query chromatograms for a single precursor from an open XICParquetFile.
    Optionally apply Savitzky-Golay smoothing to intensity values.
    Returns a Plotly figure or None on failure.
    """
    try:
        chrom_query = xic_h.query_chromatograms().filter_precursor_id(precursor_id)
        xic = chrom_query.to_df()
        if xic is None or xic.empty:
            return None

        # Apply Savitzky-Golay smoothing if enabled
        if smooth_enabled:
            xic["intensity"] = xic.groupby("annotation")["intensity"].transform(
                lambda x: savgol_filter(
                    x, window_length=sgolay_window, polyorder=sgolay_order
                )
            )

        sequence = xic["modified_sequence"].iloc[0]
        charge = xic["precursor_charge"].iloc[0]
        is_decoy = xic["precursor_decoy"].iloc[0]
        decoy_tag = " | decoy" if int(is_decoy) == 1 else ""
        title = f"{sequence}/{charge}{decoy_tag}"
        if title_suffix:
            title = f"{title}  ·  {title_suffix}"
        fig = xic.plot(
            kind="chromatogram",
            x="rt",
            y="intensity",
            by="annotation",
            title=title,
            backend="ms_plotly",
            show_plot=False,
        )
        return fig
    except Exception as exc:
        return exc  # caller distinguishes figure vs exception


# -----------------------------------------------------------------------------
# SECTION 1 — File upload

st.subheader("📂 Upload XIC Files")

col_up1, col_up2 = st.columns([3, 1])
with col_up1:
    uploaded_files = st.file_uploader(
        "Upload one or more `.xic` Parquet files",
        type=["xic", "parquet"],
        accept_multiple_files=True,
        help="OpenMS XIC Parquet files (produced by XICParquetFile, pyopenms ≥ 3.6.0).",
        key="xic_uploader",
    )
with col_up2:
    analyte_mode = st.radio(
        "Precursor list",
        options=["Intersection", "Union"],
        index=0,
        help=(
            "**Intersection** — only show precursors present in *all* files.\n\n"
            "**Union** — show every precursor from any file; "
            "files missing a precursor will display an error panel."
        ),
    )

load_btn = st.button(
    "▶ Load Files",
    type="primary",
    disabled=(not uploaded_files),
    help="Parse the uploaded XIC files and build the shared precursor list.",
)

# -- Handle load ---------------------------------------------------------------
if load_btn and uploaded_files:
    with st.spinner(
        "Saving uploaded XIC files to workspace and reading analyte tables…"
    ):
        progress = st.progress(0)
        tmp_paths: list[tuple[str, str]] = []
        file_analytes: dict[str, pd.DataFrame] = {}
        errors: list[str] = []

        # Save uploaded files into the workspace (reuses existing workspace logic)
        try:
            saved_files = fileupload.save_uploaded_xic(uploaded_files)
        except Exception as exc:
            st.error(f"Failed to save uploaded XIC files: {exc}")
            saved_files = []

        for i, (display, tmp_path) in enumerate(saved_files):
            try:
                analytes = _load_analytes(tmp_path)
                tmp_paths.append((display, tmp_path))
                file_analytes[display] = analytes
            except Exception as exc:
                errors.append(f"**{display}**: {exc}")

            progress.progress(int(100 * (i + 1) / max(1, len(saved_files))))

        if errors:
            for e in errors:
                st.error(e)

        if file_analytes:
            shared = _build_shared_analytes(file_analytes, mode=analyte_mode.lower())
            st.session_state.xic_tmp_paths = tmp_paths
            st.session_state.file_analytes = file_analytes
            st.session_state.shared_analytes = shared
            st.session_state.files_loaded = True
            progress.progress(100)
            st.success(
                f"Loaded **{len(file_analytes)}** file(s) — "
                f"**{len(shared)}** precursors available "
                f"({analyte_mode.lower()} of all files)."
            )
        else:
            st.error(
                "No files could be loaded. Check that the files are valid XIC Parquet files."
            )

# -----------------------------------------------------------------------------
# SECTION 2 — Precursor selection + file summary

if st.session_state.files_loaded and st.session_state.shared_analytes is not None:
    shared_df = st.session_state.shared_analytes

    if shared_df.empty:
        st.warning(
            "No shared precursors found. "
            "Try switching from **Intersection** to **Union** mode and re-loading."
        )
        st.stop()

    st.markdown("---")
    st.subheader("🔬 Precursor Selection")

    # -- Summary metrics -------------------------------------------------------
    n_files = len(st.session_state.file_analytes)
    n_prec = len(shared_df)
    n_targets = (shared_df["precursor_decoy"] == 0).sum()
    n_decoys = (shared_df["precursor_decoy"] == 1).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files loaded", str(n_files))
    c2.metric("Precursors", str(n_prec))
    c3.metric("Target precursors", str(n_targets))
    c4.metric("Decoy precursors", str(n_decoys))

    # -- Filter controls -------------------------------------------------------
    with st.expander("🔎 Filter precursor list", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            seq_filter = st.text_input(
                "Sequence contains",
                value="",
                placeholder="e.g. PEPTIDE or UniMod",
                help="Case-insensitive substring match on modified_sequence.",
                key="seq_filter",
            )
        with fc2:
            charge_opts = sorted(
                shared_df["precursor_charge"].dropna().unique().tolist()
            )
            charge_filter = st.multiselect(
                "Charge state(s)",
                options=charge_opts,
                default=[],
                key="charge_filter",
            )
        with fc3:
            decoy_filter = st.selectbox(
                "Decoy status",
                options=["All", "Targets only", "Decoys only"],
                index=0,
                key="decoy_filter",
            )

    # Apply filters
    display_df = shared_df.copy()
    if seq_filter.strip():
        display_df = display_df[
            display_df["modified_sequence"].str.contains(
                seq_filter.strip(), case=False, na=False
            )
        ]
    if charge_filter:
        display_df = display_df[display_df["precursor_charge"].isin(charge_filter)]
    if decoy_filter == "Targets only":
        display_df = display_df[display_df["precursor_decoy"] == 0]
    elif decoy_filter == "Decoys only":
        display_df = display_df[display_df["precursor_decoy"] == 1]

    if display_df.empty:
        st.warning(
            "No precursors match the current filters — adjust the filter settings above."
        )
        st.stop()

    # -- Dropdown --------------------------------------------------------------
    label_map: dict[str, int] = {
        _make_label(row): int(row["precursor_id"]) for _, row in display_df.iterrows()
    }
    label_list = list(label_map.keys())

    selected_label = st.selectbox(
        f"Select precursor  ({len(label_list)} matching)",
        options=label_list,
        index=0,
        key="precursor_sel",
        help="Choose the peptide precursor to visualise. Chromatograms for all loaded files will be rendered below.",
    )
    selected_pid = label_map[selected_label]

    # Show selected precursor detail
    sel_row = display_df[display_df["precursor_id"] == selected_pid].iloc[0]
    with st.expander("📋 Selected precursor details", expanded=False):
        detail = {
            "precursor_id": sel_row.get("precursor_id"),
            "modified_sequence": sel_row.get("modified_sequence"),
            "precursor_charge": sel_row.get("precursor_charge"),
            "precursor_decoy": sel_row.get("precursor_decoy"),
            "transition_ids": sel_row.get("transition_id"),
            "annotations": sel_row.get("annotation"),
            "transition_types": sel_row.get("transition_type"),
        }
        st.json(
            {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in detail.items()
                if v is not None
            }
        )

    # -- Plot options ----------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Chromatograms")

    po_col1, po_col2 = st.columns(2)
    with po_col1:
        n_cols = st.select_slider(
            "Columns per row",
            options=[1, 2, 3],
            value=min(2, n_files),
            key="plot_cols",
            help="Number of chromatogram panels per row.",
        )
    with po_col2:
        plot_height = st.slider(
            "Plot height (px)",
            min_value=250,
            max_value=800,
            value=380,
            step=50,
            key="plot_height",
        )

    # -- Smoothing options --
    st.markdown("")
    smooth_col1, smooth_col2, smooth_col3 = st.columns([2, 1, 1])
    with smooth_col1:
        smooth_enabled = st.toggle(
            "Apply Savitzky-Golay smoothing",
            value=st.session_state.smooth_toggle,
            key="smooth_toggle",
            help="Enable smoothing of chromatogram intensities using Savitzky-Golay filter.",
        )

    if smooth_enabled:
        with smooth_col2:
            sgolay_order = st.number_input(
                "Order",
                min_value=1,
                max_value=10,
                value=st.session_state.sgolay_order,
                key="sgolay_order",
                help="Polynomial order for Savitzky-Golay filter.",
            )
        with smooth_col3:
            sgolay_window = st.number_input(
                "Window length",
                min_value=3,
                max_value=51,
                value=st.session_state.sgolay_window,
                step=2,
                key="sgolay_window",
                help="Window length for Savitzky-Golay filter (must be odd).",
            )
    else:
        sgolay_order = st.session_state.sgolay_order
        sgolay_window = st.session_state.sgolay_window

    # Runs selection + clear button
    ctl_col1, ctl_col2 = st.columns([3, 1])
    tmp_path_map = dict(st.session_state.xic_tmp_paths)  # {display_name: tmp_path}
    all_file_names = list(tmp_path_map.keys())
    with ctl_col1:
        runs_sel = st.multiselect(
            "Select runs to display",
            options=all_file_names,
            default=all_file_names,
            key="runs_sel",
            help="Choose which uploaded runs should be rendered.",
        )
    with ctl_col2:
        clear_btn = st.button(
            "🗑️ Clear loaded XIC files",
            type="secondary",
            disabled=(not bool(all_file_names)),
            help="Remove all uploaded XIC files from the workspace and clear the viewer.",
        )
        if clear_btn:
            fileupload.remove_all_xic_files()
            st.rerun()

    # -------------------------------------------------------------------------
    # Render chromatogram panels

    # Filter file list by user selection
    tmp_path_map = dict(st.session_state.xic_tmp_paths)  # {display_name: tmp_path}
    file_names = [
        f
        for f in list(tmp_path_map.keys())
        if f in st.session_state.get("runs_sel", list(tmp_path_map.keys()))
    ]

    if not file_names:
        st.warning("No runs selected — pick runs from 'Select runs to display'.")
        st.stop()

    # Tile into rows of n_cols
    rows = [file_names[i : i + n_cols] for i in range(0, len(file_names), n_cols)]

    for row_files in rows:
        cols = st.columns(len(row_files))

        for col_obj, fname in zip(cols, row_files):
            with col_obj:
                tmp_path = tmp_path_map[fname]

                # Check whether this file has the precursor at all
                file_pids = set(
                    st.session_state.file_analytes[fname]["precursor_id"].tolist()
                )
                if selected_pid not in file_pids:
                    # Precursor absent in this file
                    st.markdown(f"**{fname}**")
                    st.error(
                        f"⚠️ Precursor **{selected_label}** "
                        f"(id={selected_pid}) not found in `{fname}`."
                    )
                    continue

                # Open and plot
                try:
                    xic_h = poms.XICParquetFile(tmp_path)
                    result = _plot_chromatogram(
                        xic_h,
                        selected_pid,
                        title_suffix=fname,
                        smooth_enabled=st.session_state.smooth_toggle,
                        sgolay_order=st.session_state.sgolay_order,
                        sgolay_window=st.session_state.sgolay_window,
                    )

                    if result is None:
                        st.markdown(f"**{fname}**")
                        st.warning(
                            f"No chromatogram data returned for precursor "
                            f"id={selected_pid} in `{fname}`."
                        )
                    elif isinstance(result, Exception):
                        st.markdown(f"**{fname}**")
                        st.error(
                            f"Error rendering chromatogram from `{fname}`:\n\n"
                            f"`{result}`"
                        )
                    else:
                        # Valid Plotly figure
                        result.update_layout(height=plot_height)
                        st.plotly_chart(result, use_container_width=True)

                except Exception as exc:
                    st.markdown(f"**{fname}**")
                    st.error(f"Failed to open `{fname}`:\n\n`{exc}`")

    # -- Analyte table (optional) ----------------------------------------------
    st.markdown("---")
    with st.expander("📊 Full analyte table (all loaded files)", expanded=False):
        tabs = st.tabs(list(st.session_state.file_analytes.keys()))
        for tab, (fname, analytes_df) in zip(
            tabs, st.session_state.file_analytes.items()
        ):
            with tab:
                st.caption(
                    f"{len(analytes_df)} precursors  |  "
                    f"targets: {(analytes_df['precursor_decoy'] == 0).sum()}  |  "
                    f"decoys: {(analytes_df['precursor_decoy'] == 1).sum()}"
                )
                st.dataframe(
                    analytes_df[
                        [
                            "precursor_id",
                            "modified_sequence",
                            "precursor_charge",
                            "precursor_decoy",
                            "transition_type",
                            "annotation",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
