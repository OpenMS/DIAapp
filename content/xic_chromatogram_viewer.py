"""
xic_chromatogram_viewer.py — XIC Chromatogram Viewer

Upload one or more .xic Parquet files produced by OpenMS (pyopenms >= 3.6.0),
select a precursor of interest from the shared analyte list, and view aligned
chromatograms for every uploaded file side-by-side.

Optional: upload an OpenSwath OSW SQLite file to overlay feature peak
boundaries and inspect peak-group level feature information.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter

from src.common.common import page_setup
from src import fileupload
from src.osw_utils import OSWFile

# -- pyopenms version guard ----------------------------------------------------
try:
    import pyopenms as poms
    import pyopenms_viz  # noqa: F401 - registers ms_plotly backend

    _poms_version = tuple(int(x) for x in poms.__version__.split(".")[:2])
    if _poms_version < (3, 6):
        st.error(
            f"**pyopenms {poms.__version__} detected** - Currently the XIC viewer only supports OpenMS XIC parquet files. "
            "`XICParquetFile` requires pyopenms >= 3.6.0. Please upgrade: `pip install pyopenms --upgrade`"
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
pyopenms >= 3.6.0), choose a peptide precursor from the shared analyte list,
and visualise the extracted ion chromatograms for every file simultaneously.

You can also upload an **OSW SQLite feature file** to overlay feature peak
boundaries from the `FEATURE` table and inspect the corresponding peak-group
feature information.
"""
)

with st.expander("ℹ️ About XIC Parquet files"):
    st.markdown(
        """
        `.xic` files are Parquet-format chromatogram files written by OpenMS
        (for example via `XICParquetFile`). Each file contains:
        - **Analyte table** - one row per precursor with sequence, charge, decoy flag,
          transition IDs, annotation, and related metadata.
        - **Chromatogram data** - retention time and intensity arrays per transition.

        Multiple files can be loaded simultaneously. The viewer will attempt to
        locate the selected precursor in every file and render a chromatogram
        panel for each file.
        """
    )

with st.expander("ℹ️ About OSW boundaries"):
    st.markdown(
        """
        When an OSW file is provided, the viewer can fetch peak boundaries from
        the `FEATURE` table by matching **precursor_id** and **run_id**.

        Boundary ranking is determined by:
        1. `SCORE_MS2.RANK = 1` when `SCORE_MS2` is present.
        2. Otherwise, the feature with the highest `FEATURE_MS2.AREA_INTENSITY`.

        You can choose to display **all feature boundaries** or **only the top feature**.
        """
    )

if not POMS_OK:
    st.error("pyopenms is not installed. Install with `pip install pyopenms`.")
    st.stop()

st.markdown("---")

# -----------------------------------------------------------------------------
# Session state

_defaults: dict = {
    "xic_tmp_paths": [],
    "shared_analytes": None,
    "file_analytes": {},
    "xic_run_metadata": {},
    "files_loaded": False,
    "smooth_toggle": True,
    "sgolay_order": 3,
    "sgolay_window": 9,
    "osw_file_path": None,
    "osw_file_name": None,
    "osw_handler": None,
    "osw_runs_df": None,
    "show_peak_boundaries": False,
    "show_top_boundary_only": True,
    "boundary_run_mapping": {},
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# -----------------------------------------------------------------------------
# Helper functions


def _load_analytes_and_run_metadata(tmp_path: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Open an XIC parquet file and return its analyte dataframe plus run metadata."""
    xic_h = poms.XICParquetFile(tmp_path)
    analytes = xic_h.get_analyte_df()

    run_meta: dict[str, object] = {
        "run_id": None,
        "source_file": None,
        "run_df": pd.DataFrame(),
    }

    try:
        run_df = xic_h.get_run_df()
        if run_df is None:
            run_df = pd.DataFrame()
        else:
            run_df = pd.DataFrame(run_df).copy()

        run_meta["run_df"] = run_df

        if not run_df.empty:
            if "run_id" in run_df.columns:
                run_id = pd.to_numeric(run_df["run_id"].iloc[0], errors="coerce")
                if pd.notna(run_id):
                    run_meta["run_id"] = int(run_id)
            if "source_file" in run_df.columns:
                source_file = run_df["source_file"].iloc[0]
                if pd.notna(source_file):
                    run_meta["source_file"] = str(source_file)
    except Exception:
        pass

    return analytes, run_meta



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

    mode="intersection"  - only precursors present in all files
    mode="union"         - all precursors from any file
    """
    if not file_analytes:
        return pd.DataFrame()

    frames = list(file_analytes.values())

    if mode == "intersection":
        id_sets = [set(df["precursor_id"].tolist()) for df in frames]
        common = id_sets[0].intersection(*id_sets[1:])
        ref = frames[0][frames[0]["precursor_id"].isin(common)].copy()
    else:
        ref = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["precursor_id"])

    return ref.sort_values(
        ["modified_sequence", "precursor_charge", "precursor_decoy"]
    ).reset_index(drop=True)



def _normalize_name_for_matching(name: str) -> str:
    """Normalize file names for XIC <-> OSW run matching."""
    base = Path(str(name)).name.lower()
    suffixes = [
        ".sqmass",
        ".osw",
        ".parquet",
        ".xic",
        ".mzml",
        ".chrom",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                changed = True
    return base



def _guess_osw_run_mapping(
    xic_display_names: list[str],
    xic_run_metadata: dict[str, dict[str, object]],
    osw_runs_df: pd.DataFrame,
) -> dict[str, Optional[int]]:
    """Best-effort mapping of uploaded XIC files to OSW RUN.ID values.

    Priority:
    1. Match XIC run_id to OSW RUN.ID
    2. Match XIC source_file to OSW RUN.FILENAME
    3. Match uploaded XIC display name to OSW RUN.FILENAME
    """
    mapping: dict[str, Optional[int]] = {}
    if osw_runs_df is None or osw_runs_df.empty:
        return {name: None for name in xic_display_names}

    runs_df = osw_runs_df.copy()
    if "ID" in runs_df.columns:
        runs_df["ID"] = pd.to_numeric(runs_df["ID"], errors="coerce").astype("Int64")

    run_records = []
    for _, row in runs_df.iterrows():
        filename = str(row["FILENAME"])
        run_records.append(
            {
                "run_id": None if pd.isna(row["ID"]) else int(row["ID"]),
                "filename": filename,
                "norm": _normalize_name_for_matching(filename),
            }
        )

    osw_run_id_map = {rec["run_id"]: rec for rec in run_records if rec["run_id"] is not None}

    for display_name in xic_display_names:
        meta = xic_run_metadata.get(display_name, {}) or {}
        xic_run_id = meta.get("run_id")
        if xic_run_id is not None:
            try:
                xic_run_id = int(xic_run_id)
            except Exception:
                xic_run_id = None

        if xic_run_id is not None and xic_run_id in osw_run_id_map:
            mapping[display_name] = xic_run_id
            continue

        candidate_names = []
        source_file = meta.get("source_file")
        if source_file:
            candidate_names.append(str(source_file))
        candidate_names.append(display_name)

        matched_run_id = None
        for candidate in candidate_names:
            candidate_norm = _normalize_name_for_matching(candidate)
            matches = [r for r in run_records if r["norm"] == candidate_norm]
            if not matches:
                matches = [
                    r for r in run_records if candidate_norm in r["norm"] or r["norm"] in candidate_norm
                ]
            if matches:
                matched_run_id = matches[0]["run_id"]
                break

        mapping[display_name] = matched_run_id

    return mapping


def _save_uploaded_osw_file(uploaded_osw) -> str:
    """Persist uploaded OSW file to a temporary path and return the path."""
    suffix = Path(uploaded_osw.name).suffix or ".osw"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_osw.getbuffer())
        return tmp.name



def _load_osw_file(uploaded_osw) -> None:
    """Load an uploaded OSW file into session state."""
    if st.session_state.get("osw_handler") is not None:
        try:
            st.session_state.osw_handler.close()
        except Exception:
            pass

    osw_tmp_path = _save_uploaded_osw_file(uploaded_osw)
    osw_handler = OSWFile(osw_tmp_path)
    osw_runs_df = osw_handler.list_runs()

    st.session_state.osw_file_path = osw_tmp_path
    st.session_state.osw_file_name = uploaded_osw.name
    st.session_state.osw_handler = osw_handler
    st.session_state.osw_runs_df = osw_runs_df

    xic_display_names = [name for name, _ in st.session_state.get("xic_tmp_paths", [])]
    st.session_state.boundary_run_mapping = _guess_osw_run_mapping(
        xic_display_names,
        st.session_state.get("xic_run_metadata", {}),
        osw_runs_df,
    )



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

        if smooth_enabled:
            xic["intensity"] = xic.groupby("annotation")["intensity"].transform(
                lambda x: savgol_filter(
                    x,
                    window_length=sgolay_window,
                    polyorder=sgolay_order,
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
        return exc



def _add_peak_boundaries_to_figure(fig: go.Figure, boundaries_df: pd.DataFrame) -> go.Figure:
    """Overlay OSW peak boundaries on a chromatogram figure."""
    if boundaries_df is None or boundaries_df.empty:
        return fig

    max_y = 1.0
    try:
        y_values = []
        for trace in fig.data:
            if getattr(trace, "y", None) is not None:
                y_values.extend([float(v) for v in trace.y if v is not None])
        if y_values:
            max_y = max(y_values)
    except Exception:
        max_y = 1.0

    boundary_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    for idx, (_, row) in enumerate(boundaries_df.iterrows()):
        label = f"Peak boundaries ({'top' if bool(row.get('TOP_FEATURE', False)) else 'feature'})"
        feature_id = row.get("FEATURE_ID", "?")
        rank_text = row.get("RANK")
        score_text = row.get("SCORE")
        hover_text = (
            f"feature_id={feature_id}<br>"
            f"rt={row.get('EXP_RT', 'NA')}<br>"
            f"left={row.get('LEFT_WIDTH', 'NA')}<br>"
            f"right={row.get('RIGHT_WIDTH', 'NA')}<br>"
            f"rank={rank_text}<br>"
            f"score={score_text}"
        )
        legend_group = "feature_boundaries"
        show_legend = idx == 0
        boundary_color = boundary_colors[idx % len(boundary_colors)]

        fig.add_trace(
            go.Scatter(
                x=[row["LEFT_WIDTH"], row["LEFT_WIDTH"]],
                y=[0, max_y],
                mode="lines",
                line={"dash": "dash", "width": 2, "color": boundary_color},
                name="Peak boundaries",
                legendgroup=legend_group,
                showlegend=show_legend,
                hovertemplate="Left boundary<br>%{text}<extra></extra>",
                text=[hover_text, hover_text],
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["RIGHT_WIDTH"], row["RIGHT_WIDTH"]],
                y=[0, max_y],
                mode="lines",
                line={"dash": "dash", "width": 2, "color": boundary_color},
                name="Peak boundaries",
                legendgroup=legend_group,
                showlegend=False,
                hovertemplate="Right boundary<br>%{text}<extra></extra>",
                text=[hover_text, hover_text],
            )
        )

    fig.update_layout(legend={"groupclick": "togglegroup"})
    return fig



def _get_mapped_osw_run_ids() -> list[int]:
    """Return unique mapped OSW run IDs for the currently loaded XIC files."""
    run_ids = []
    for run_id in st.session_state.get("boundary_run_mapping", {}).values():
        if run_id is None:
            continue
        try:
            run_ids.append(int(run_id))
        except Exception:
            continue
    return sorted(set(run_ids))


def _get_osw_precursor_qvalue_summary(osw_handler: Optional[OSWFile]) -> pd.DataFrame:
    """Fetch per-precursor OSW q-value summaries for the mapped runs."""
    if osw_handler is None or not osw_handler.has_table("SCORE_MS2"):
        return pd.DataFrame()

    run_ids = _get_mapped_osw_run_ids()
    return osw_handler.get_precursor_qvalue_summary(run_ids=run_ids if run_ids else None)


def _format_feature_table(boundaries_df: pd.DataFrame) -> pd.DataFrame:
    """Format the peak-group feature table for display."""
    if boundaries_df is None or boundaries_df.empty:
        return pd.DataFrame()

    display_cols = [
        "TOP_FEATURE",
        "TOP_SOURCE",
        "FEATURE_ID",
        "EXP_RT",
        "LEFT_WIDTH",
        "RIGHT_WIDTH",
        "BOUNDARY_WIDTH",
        "AREA_INTENSITY",
        "APEX_INTENSITY",
        "SCORE",
        "PVALUE",
        "QVALUE",
        "RANK",
        "PEP",
    ]
    keep_cols = [c for c in display_cols if c in boundaries_df.columns]
    table_df = boundaries_df[keep_cols].copy()

    for col in [
        "EXP_RT",
        "LEFT_WIDTH",
        "RIGHT_WIDTH",
        "BOUNDARY_WIDTH",
        "AREA_INTENSITY",
        "APEX_INTENSITY",
        "SCORE",
        "PVALUE",
        "QVALUE",
        "PEP",
    ]:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors="coerce")
            table_df[col] = table_df[col].round(4)

    if "RANK" in table_df.columns:
        table_df["RANK"] = pd.to_numeric(table_df["RANK"], errors="coerce").astype("Int64")
    if "FEATURE_ID" in table_df.columns:
        table_df["FEATURE_ID"] = pd.to_numeric(table_df["FEATURE_ID"], errors="coerce").astype("Int64")

    return table_df


# -----------------------------------------------------------------------------
# SECTION 1 — File upload

st.subheader("📂 Upload XIC Files")

col_up1, col_up2 = st.columns([3, 1])
with col_up1:
    uploaded_files = st.file_uploader(
        "Upload one or more `.xic` Parquet files",
        type=["xic", "parquet"],
        accept_multiple_files=True,
        help="OpenMS XIC Parquet files (produced by XICParquetFile, pyopenms >= 3.6.0).",
        key="xic_uploader",
    )
with col_up2:
    analyte_mode = st.radio(
        "Precursor list",
        options=["Intersection", "Union"],
        index=0,
        help=(
            "**Intersection** - only show precursors present in all files.\n\n"
            "**Union** - show every precursor from any file; files missing a precursor will display an error panel."
        ),
    )

load_btn = st.button(
    "▶ Load Files",
    type="primary",
    disabled=(not uploaded_files),
    help="Parse the uploaded XIC files and build the shared precursor list.",
)

if load_btn and uploaded_files:
    with st.spinner("Saving uploaded XIC files to workspace and reading analyte tables…"):
        progress = st.progress(0)
        tmp_paths: list[tuple[str, str]] = []
        file_analytes: dict[str, pd.DataFrame] = {}
        xic_run_metadata: dict[str, dict[str, object]] = {}
        errors: list[str] = []

        try:
            saved_files = fileupload.save_uploaded_xic(uploaded_files)
        except Exception as exc:
            st.error(f"Failed to save uploaded XIC files: {exc}")
            saved_files = []

        for i, (display, tmp_path) in enumerate(saved_files):
            try:
                analytes, run_meta = _load_analytes_and_run_metadata(tmp_path)
                tmp_paths.append((display, tmp_path))
                file_analytes[display] = analytes
                xic_run_metadata[display] = run_meta
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
            st.session_state.xic_run_metadata = xic_run_metadata
            st.session_state.shared_analytes = shared
            st.session_state.files_loaded = True

            if st.session_state.get("osw_runs_df") is not None:
                xic_display_names = [name for name, _ in tmp_paths]
                st.session_state.boundary_run_mapping = _guess_osw_run_mapping(
                    xic_display_names,
                    st.session_state.get("xic_run_metadata", {}),
                    st.session_state.osw_runs_df,
                )

            progress.progress(100)
            st.success(
                f"Loaded **{len(file_analytes)}** file(s) - "
                f"**{len(shared)}** precursors available "
                f"({analyte_mode.lower()} of all files)."
            )
        else:
            st.error("No files could be loaded. Check that the files are valid XIC Parquet files.")

# -----------------------------------------------------------------------------
# SECTION 1B — Optional OSW upload and run mapping

st.markdown("---")
st.subheader("🧭 Optional OSW feature file")

osw_col1, osw_col2 = st.columns([3, 1])
with osw_col1:
    uploaded_osw = st.file_uploader(
        "Upload an OSW SQLite file",
        type=["osw", "sqlite", "db"],
        accept_multiple_files=False,
        help="Optional OpenSwath OSW file used to fetch feature boundaries and peak-group information.",
        key="osw_uploader",
    )
with osw_col2:
    osw_load_btn = st.button(
        "Load OSW",
        type="secondary",
        disabled=(uploaded_osw is None),
        help="Load the OSW file and list its runs.",
    )

if osw_load_btn and uploaded_osw is not None:
    try:
        _load_osw_file(uploaded_osw)
        n_runs = 0 if st.session_state.osw_runs_df is None else len(st.session_state.osw_runs_df)
        st.success(f"Loaded OSW file **{uploaded_osw.name}** with **{n_runs}** run(s).")
    except Exception as exc:
        st.error(f"Failed to load OSW file: {exc}")

if st.session_state.get("osw_file_name"):
    st.caption(f"Loaded OSW file: {st.session_state.osw_file_name}")

if st.session_state.get("osw_runs_df") is not None and not st.session_state.osw_runs_df.empty:
    with st.expander("OSW run mapping", expanded=False):
        st.caption("Map each uploaded XIC file to the correct OSW RUN.ID. Auto-matching tries XIC run_id first, then XIC source_file, then the uploaded filename.")
        runs_df = st.session_state.osw_runs_df.copy()
        run_options = {"None": None}
        for _, row in runs_df.iterrows():
            label = f"{row['ID']} | {Path(str(row['FILENAME'])).name}"
            run_options[label] = int(row["ID"])

        for display_name, _ in st.session_state.get("xic_tmp_paths", []):
            meta = st.session_state.get("xic_run_metadata", {}).get(display_name, {}) or {}
            xic_run_id = meta.get("run_id")
            xic_source_file = meta.get("source_file")
            if xic_run_id is not None or xic_source_file:
                st.caption(
                    f"{display_name}: XIC run_id={xic_run_id if xic_run_id is not None else 'NA'}"
                    + (f" | source_file={xic_source_file}" if xic_source_file else "")
                )

            current_mapping = st.session_state.boundary_run_mapping.get(display_name)
            option_labels = list(run_options.keys())
            current_index = 0
            for idx, label in enumerate(option_labels):
                if run_options[label] == current_mapping:
                    current_index = idx
                    break
            selected_label = st.selectbox(
                f"OSW run for {display_name}",
                options=option_labels,
                index=current_index,
                key=f"osw_run_map_{display_name}",
            )
            st.session_state.boundary_run_mapping[display_name] = run_options[selected_label]

# -----------------------------------------------------------------------------
# SECTION 2 — Precursor selection + file summary

if st.session_state.files_loaded and st.session_state.shared_analytes is not None:
    shared_df = st.session_state.shared_analytes

    if shared_df.empty:
        st.warning(
            "No shared precursors found. Try switching from **Intersection** to **Union** mode and re-loading."
        )
        st.stop()

    st.markdown("---")
    st.subheader("🔬 Precursor Selection")

    n_files = len(st.session_state.file_analytes)
    n_prec = len(shared_df)
    n_targets = (shared_df["precursor_decoy"] == 0).sum()
    n_decoys = (shared_df["precursor_decoy"] == 1).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files loaded", str(n_files))
    c2.metric("Precursors", str(n_prec))
    c3.metric("Target precursors", str(n_targets))
    c4.metric("Decoy precursors", str(n_decoys))

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
            charge_opts = sorted(shared_df["precursor_charge"].dropna().unique().tolist())
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

        osw_handler_for_filter: Optional[OSWFile] = st.session_state.get("osw_handler")
        osw_qvalue_filter_enabled = False
        qvalue_range = (0.0, 1.0)
        osw_summary_df = pd.DataFrame()

        if osw_handler_for_filter is not None and osw_handler_for_filter.has_table("SCORE_MS2"):
            osw_summary_df = _get_osw_precursor_qvalue_summary(osw_handler_for_filter)
            if not osw_summary_df.empty:
                valid_qvals = pd.to_numeric(osw_summary_df["BEST_QVALUE"], errors="coerce").dropna()
                q_min_default = 0.0
                q_max_default = 1.0
                if not valid_qvals.empty:
                    q_min_default = max(0.0, float(valid_qvals.min()))
                    q_max_default = max(1.0, float(valid_qvals.max()))

                fq1, fq2 = st.columns([1, 2])
                with fq1:
                    osw_qvalue_filter_enabled = st.toggle(
                        "Filter by OSW q-value",
                        value=False,
                        key="osw_qvalue_filter_enabled",
                        help="Filter precursors by the best OSW feature q-value across the mapped runs.",
                    )
                with fq2:
                    qvalue_range = st.slider(
                        "OSW best q-value range",
                        min_value=0.0,
                        max_value=max(1.0, q_max_default),
                        value=(0.0, min(0.05, max(1.0, q_max_default))),
                        key="osw_qvalue_range",
                        disabled=(not osw_qvalue_filter_enabled),
                        help="Uses the minimum feature q-value per precursor across the mapped OSW runs.",
                    )
            else:
                st.caption("OSW file is loaded, but no SCORE_MS2 q-values were found for the currently mapped runs.")
        elif osw_handler_for_filter is not None:
            st.caption("OSW q-value precursor filtering is unavailable because SCORE_MS2 is not present.")

    display_df = shared_df.copy()
    if seq_filter.strip():
        display_df = display_df[
            display_df["modified_sequence"].str.contains(seq_filter.strip(), case=False, na=False)
        ]
    if charge_filter:
        display_df = display_df[display_df["precursor_charge"].isin(charge_filter)]
    if decoy_filter == "Targets only":
        display_df = display_df[display_df["precursor_decoy"] == 0]
    elif decoy_filter == "Decoys only":
        display_df = display_df[display_df["precursor_decoy"] == 1]

    if osw_qvalue_filter_enabled and not osw_summary_df.empty:
        osw_summary_df = osw_summary_df.copy()
        osw_summary_df["PRECURSOR_ID"] = pd.to_numeric(osw_summary_df["PRECURSOR_ID"], errors="coerce").astype("Int64")
        osw_summary_df["BEST_QVALUE"] = pd.to_numeric(osw_summary_df["BEST_QVALUE"], errors="coerce")
        keep_ids = set(
            osw_summary_df.loc[
                osw_summary_df["BEST_QVALUE"].between(float(qvalue_range[0]), float(qvalue_range[1]), inclusive="both"),
                "PRECURSOR_ID",
            ].dropna().astype(int).tolist()
        )
        display_df = display_df[display_df["precursor_id"].isin(keep_ids)]

    if display_df.empty:
        st.warning("No precursors match the current filters - adjust the filter settings above.")
        st.stop()

    label_map: dict[str, int] = {
        _make_label(row): int(row["precursor_id"]) for _, row in display_df.iterrows()
    }
    label_list = list(label_map.keys())

    selected_label = st.selectbox(
        f"Select precursor ({len(label_list)} matching)",
        options=label_list,
        index=0,
        key="precursor_sel",
        help="Choose the peptide precursor to visualise. Chromatograms for all loaded files will be rendered below.",
    )
    selected_pid = label_map[selected_label]

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

    runs_ctl_col1, runs_ctl_col2 = st.columns([3, 1])
    tmp_path_map = dict(st.session_state.xic_tmp_paths)
    all_file_names = list(tmp_path_map.keys())
    with runs_ctl_col1:
        runs_sel = st.multiselect(
            "Select runs to display",
            options=all_file_names,
            default=all_file_names,
            key="runs_sel",
            help="Choose which uploaded runs should be rendered.",
        )
    with runs_ctl_col2:
        clear_btn = st.button(
            "🗑️ Clear loaded XIC files",
            type="secondary",
            disabled=(not bool(all_file_names)),
            help="Remove all uploaded XIC files from the workspace and clear the viewer.",
        )
        if clear_btn:
            fileupload.remove_all_xic_files()
            st.rerun()

    boundary_mode_col1, boundary_mode_col2 = st.columns(2)
    with boundary_mode_col1:
        show_peak_boundaries = st.toggle(
            "Show OSW peak boundaries",
            value=st.session_state.show_peak_boundaries,
            key="show_peak_boundaries",
            disabled=(st.session_state.get("osw_handler") is None),
            help="Overlay peak boundaries from the OSW FEATURE table.",
        )
    with boundary_mode_col2:
        show_top_boundary_only = st.toggle(
            "Top feature only",
            value=st.session_state.show_top_boundary_only,
            key="show_top_boundary_only",
            disabled=(st.session_state.get("osw_handler") is None or not show_peak_boundaries),
            help="If enabled, only the top-ranked feature is shown. Otherwise all feature boundaries are shown.",
        )

    tmp_path_map = dict(st.session_state.xic_tmp_paths)
    file_names = [
        f for f in list(tmp_path_map.keys()) if f in st.session_state.get("runs_sel", list(tmp_path_map.keys()))
    ]

    if not file_names:
        st.warning("No runs selected - pick runs from 'Select runs to display'.")
        st.stop()

    rows = [file_names[i : i + n_cols] for i in range(0, len(file_names), n_cols)]
    osw_handler: Optional[OSWFile] = st.session_state.get("osw_handler")

    for row_files in rows:
        cols = st.columns(len(row_files))

        for col_obj, fname in zip(cols, row_files):
            with col_obj:
                tmp_path = tmp_path_map[fname]

                file_pids = set(st.session_state.file_analytes[fname]["precursor_id"].tolist())
                if selected_pid not in file_pids:
                    st.markdown(f"**{fname}**")
                    st.error(
                        f"⚠️ Precursor **{selected_label}** (id={selected_pid}) not found in `{fname}`."
                    )
                    continue

                boundaries_df = pd.DataFrame()
                boundary_table = pd.DataFrame()
                mapped_run_id = st.session_state.boundary_run_mapping.get(fname)
                xic_meta = st.session_state.get("xic_run_metadata", {}).get(fname, {}) or {}

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
                            f"No chromatogram data returned for precursor id={selected_pid} in `{fname}`."
                        )
                    elif isinstance(result, Exception):
                        st.markdown(f"**{fname}**")
                        st.error(f"Error rendering chromatogram from `{fname}`:\n\n`{result}`")
                    else:
                        if show_peak_boundaries and osw_handler is not None:
                            if mapped_run_id is None:
                                st.info("No OSW run is mapped to this XIC file, so peak boundaries are not shown.")
                            else:
                                boundaries_df = osw_handler.get_selected_peak_boundaries(
                                    precursor_id=selected_pid,
                                    run_id=int(mapped_run_id),
                                    top_only=show_top_boundary_only,
                                )
                                if boundaries_df.empty:
                                    st.info(
                                        f"No matching OSW features found for precursor_id={selected_pid} and run_id={mapped_run_id}."
                                    )
                                else:
                                    result = _add_peak_boundaries_to_figure(result, boundaries_df)
                                    boundary_table = _format_feature_table(boundaries_df)

                        result.update_layout(height=plot_height)
                        st.plotly_chart(result, use_container_width=True)

                        xic_run_id_caption = xic_meta.get("run_id")
                        if xic_run_id_caption is not None or mapped_run_id is not None:
                            caption_parts = []
                            if xic_run_id_caption is not None:
                                caption_parts.append(f"XIC run_id: {xic_run_id_caption}")
                            if mapped_run_id is not None:
                                caption_parts.append(f"OSW run_id: {mapped_run_id}")
                            st.caption(" | ".join(caption_parts))

                        if show_peak_boundaries and osw_handler is not None and mapped_run_id is not None:
                            st.dataframe(
                                boundary_table,
                                use_container_width=True,
                                hide_index=True,
                            )

                except Exception as exc:
                    st.markdown(f"**{fname}**")
                    st.error(f"Failed to open `{fname}`:\n\n`{exc}`")

    st.markdown("---")
    with st.expander("📊 Full analyte table (all loaded files)", expanded=False):
        tabs = st.tabs(list(st.session_state.file_analytes.keys()))
        for tab, (fname, analytes_df) in zip(tabs, st.session_state.file_analytes.items()):
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
