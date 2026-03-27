from pathlib import Path

import streamlit as st
import pandas as pd

from src.common.common import (
    page_setup,
    save_params,
    v_space,
    show_table,
    TK_AVAILABLE,
    tk_directory_dialog,
)
from src import fileupload

params = page_setup()

st.title("File Upload")

# Check if there are any files in the workspace
mzML_dir = Path(st.session_state.workspace, "mzML-files")
if not any(Path(mzML_dir).iterdir()):
    # No files present, load example data
    fileupload.load_example_mzML_files()

tabs = ["File Upload"]
if st.session_state.location == "local":
    tabs.append("Files from local folder")

tabs = st.tabs(tabs)

with tabs[0]:
    with st.form("mzML-upload", clear_on_submit=True):
        files = st.file_uploader(
            "mzML files",
            type=["mzML", "mzML.gz"],
            accept_multiple_files=(st.session_state.location == "local"),
            help="Upload your mzML files here. You can also upload files later or use the local folder option.",
        )
        # Optional FASTA and spectral library uploads
        fasta_file = st.file_uploader(
            "Optional FASTA file",
            type=["fasta", "fa", "faa"],
            accept_multiple_files=False,
            help="Upload an optional FASTA file for sequence lookup.",
            key="fasta_upload",
        )
        lib_file = st.file_uploader(
            "Optional spectral library / transition list",
            type=["tsv", "traML", "pqp"],
            accept_multiple_files=False,
            help="Upload an optional transition list or spectral library (TSV/CSV/TraML).",
            key="lib_upload",
        )
        cols = st.columns(3)
        if cols[1].form_submit_button("Add files to workspace", type="primary"):
            any_saved = False
            if files:
                fileupload.save_uploaded_mzML(files)
                any_saved = True

            # Save optional FASTA
            if fasta_file is not None:
                fasta_dir = Path(st.session_state.workspace, "input-files", "fasta")
                fasta_dir.mkdir(parents=True, exist_ok=True)
                fasta_path = Path(
                    fasta_dir, getattr(fasta_file, "name", "uploaded.fasta")
                )
                with open(fasta_path, "wb") as fh:
                    fh.write(fasta_file.getbuffer())
                st.success(f"Saved FASTA to workspace: {fasta_path.name}")
                any_saved = True

            # Save optional library/transition list
            if lib_file is not None:
                lib_dir = Path(st.session_state.workspace, "input-files", "libraries")
                lib_dir.mkdir(parents=True, exist_ok=True)
                lib_path = Path(lib_dir, getattr(lib_file, "name", "uploaded_lib.tsv"))
                with open(lib_path, "wb") as fh:
                    fh.write(lib_file.getbuffer())
                st.success(f"Saved library to workspace: {lib_path.name}")
                any_saved = True

            if not any_saved:
                st.warning("Select files first.")

# Local file upload option: via directory path
if st.session_state.location == "local":
    with tabs[1]:
        st_cols = st.columns([0.05, 0.95], gap="small")
        with st_cols[0]:
            st.write("\n")
            st.write("\n")
            dialog_button = st.button(
                "📁",
                key="local_browse",
                help="Browse for your local directory with MS data.",
                disabled=not TK_AVAILABLE,
            )
            if dialog_button:
                st.session_state["local_dir"] = tk_directory_dialog(
                    "Select directory with your MS data",
                    st.session_state["previous_dir"],
                )
                st.session_state["previous_dir"] = st.session_state["local_dir"]
        with st_cols[1]:
            # with st.form("local-file-upload"):
            local_mzML_dir = st.text_input(
                "path to folder with mzML files", value=st.session_state["local_dir"]
            )
        # raw string for file paths
        local_mzML_dir = rf"{local_mzML_dir}"
        cols = st.columns([0.65, 0.3, 0.4, 0.25], gap="small")
        copy_button = cols[1].button(
            "Copy files to workspace", type="primary", disabled=(local_mzML_dir == "")
        )
        use_copy = cols[2].checkbox(
            "Make a copy of files",
            key="local_browse-copy_files",
            value=True,
            help="Create a copy of files in workspace.",
        )
        if not use_copy:
            st.warning(
                "**Warning**: You have deselected the `Make a copy of files` option. "
                "This **_assumes you know what you are doing_**. "
                "This means that the original files will be used instead. "
            )
        if copy_button:
            fileupload.copy_local_mzML_files_from_directory(local_mzML_dir, use_copy)

if any(Path(mzML_dir).iterdir()):
    v_space(2)
    # Prepare lists
    mzml_files = [
        f.name for f in Path(mzML_dir).iterdir() if "external_files.txt" not in f.name
    ]
    # include external files
    external_files = Path(mzML_dir, "external_files.txt")
    external_list = []
    if external_files.exists():
        with open(external_files, "r") as f_handle:
            external_list = [f.strip() for f in f_handle.readlines()]

    mzml_display = mzml_files + external_list

    fasta_dir = Path(st.session_state.workspace, "input-files", "fasta")
    fasta_list = [p.name for p in fasta_dir.iterdir()] if fasta_dir.exists() else []

    lib_dir = Path(st.session_state.workspace, "input-files", "libraries")
    lib_list = [p.name for p in lib_dir.iterdir()] if lib_dir.exists() else []

    # Three-column overview
    col_mz, col_fa, col_lib = st.columns([3, 2, 3])

    with col_mz:
        st.markdown(f"##### mzML files ({len(mzml_display)})")
        show_table(pd.DataFrame({"file name": mzml_display}))
        to_remove_mz = st.multiselect(
            "Select mzML files to remove", options=mzml_display
        )
        rm_mz_c1, rm_mz_c2 = st.columns([1, 1])
        if rm_mz_c2.button("Remove selected mzML", disabled=not any(to_remove_mz)):
            params = fileupload.remove_selected_mzML_files(
                [Path(f).stem for f in to_remove_mz], params
            )
            save_params(params)
            st.rerun()
        if rm_mz_c1.button("Remove all mzML", disabled=not any(mzml_display)):
            params = fileupload.remove_all_mzML_files(params)
            save_params(params)
            st.rerun()

    with col_fa:
        st.markdown(f"##### FASTA files ({len(fasta_list)})")
        if fasta_list:
            show_table(pd.DataFrame({"file name": fasta_list}))
            to_remove_fasta = st.multiselect(
                "Select FASTA to remove", options=sorted(fasta_list)
            )
            fa_c1, fa_c2 = st.columns(2)
            if fa_c2.button("Remove selected FASTA", disabled=not any(to_remove_fasta)):
                for fn in to_remove_fasta:
                    try:
                        Path(fasta_dir, fn).unlink()
                    except Exception:
                        st.warning(f"Could not remove {fn}")
                st.success("Selected FASTA files removed")
                st.rerun()
            if fa_c1.button("Remove all FASTA", disabled=not any(fasta_list)):
                for p in fasta_dir.iterdir():
                    try:
                        if p.is_file():
                            p.unlink()
                    except Exception:
                        pass
                st.success("All FASTA files removed")
                st.rerun()
        else:
            st.info("No FASTA files in workspace")

    with col_lib:
        st.markdown(f"##### Library / Transition lists ({len(lib_list)})")
        if lib_list:
            show_table(pd.DataFrame({"file name": lib_list}))
            to_remove_lib = st.multiselect(
                "Select libraries to remove", options=sorted(lib_list)
            )
            lb_c1, lb_c2 = st.columns(2)
            if lb_c2.button(
                "Remove selected libraries", disabled=not any(to_remove_lib)
            ):
                for fn in to_remove_lib:
                    try:
                        Path(lib_dir, fn).unlink()
                    except Exception:
                        st.warning(f"Could not remove {fn}")
                st.success("Selected library files removed")
                st.rerun()
            if lb_c1.button("Remove all libraries", disabled=not any(lib_list)):
                for p in lib_dir.iterdir():
                    try:
                        if p.is_file():
                            p.unlink()
                    except Exception:
                        pass
                st.success("All library files removed")
                st.rerun()
        else:
            st.info("No library files in workspace")

save_params(params)
