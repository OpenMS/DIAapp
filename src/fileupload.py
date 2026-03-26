import shutil
from pathlib import Path

import streamlit as st

from src.common.common import reset_directory, OS_PLATFORM


@st.cache_data
def save_uploaded_mzML(uploaded_files: list[bytes]) -> None:
    """
    Saves uploaded mzML files to the mzML directory.

    Args:
        uploaded_files (List[bytes]): List of uploaded mzML files.

    Returns:
        None
    """
    mzML_dir = Path(st.session_state.workspace, "mzML-files")
    # A list of files is required, since online allows only single upload, create a list
    if st.session_state.location == "online":
        uploaded_files = [uploaded_files]
    # If no files are uploaded, exit early
    if not uploaded_files:
        st.warning("Upload some files first.")
        return
    # Write files from buffer to workspace mzML directory, add to selected files
    for f in uploaded_files:
        if f.name not in [f.name for f in mzML_dir.iterdir()] and f.name.endswith(
            "mzML"
        ):
            with open(Path(mzML_dir, f.name), "wb") as fh:
                fh.write(f.getbuffer())
    st.success("Successfully added uploaded files!")


def copy_local_mzML_files_from_directory(
    local_mzML_directory: str, make_copy: bool = True
) -> None:
    """
    Copies local mzML files from a specified directory to the mzML directory.

    Args:
        local_mzML_directory (str): Path to the directory containing the mzML files.
        make_copy (bool): Whether to make a copy of the files in the workspace. Default is True. If False, local file paths will be written to an external_files.txt file.

    Returns:
        None
    """
    mzML_dir = Path(st.session_state.workspace, "mzML-files")
    # Check if local directory contains mzML files, if not exit early
    if not any(Path(local_mzML_directory).glob("*.mzML")):
        st.warning("No mzML files found in specified folder.")
        return
    # Copy all mzML files to workspace mzML directory, add to selected files
    files = Path(local_mzML_directory).glob("*.mzML")
    for f in files:
        if make_copy:
            shutil.copy(f, Path(mzML_dir, f.name))
        else:
            # Create a temporary file to store the path to the local directories
            external_files = Path(mzML_dir, "external_files.txt")
            # Check if the file exists, if not create it
            if not external_files.exists():
                external_files.touch()
            # Write the path to the local directories to the file
            with open(external_files, "a") as f_handle:
                f_handle.write(f"{f}\n")

    st.success("Successfully added local files!")


def load_example_mzML_files() -> None:
    """
    Copies example mzML files to the mzML directory.

    On Linux, creates symlinks to example files instead of copying them.

    Args:
        None

    Returns:
        None
    """
    mzML_dir = Path(st.session_state.workspace, "mzML-files")
    # Copy or symlink files from example-data/mzML to workspace mzML directory
    for f in Path("example-data", "mzML").glob("*.mzML"):
        target = mzML_dir / f.name
        if OS_PLATFORM == "linux":
            if target.exists():
                target.unlink()
            target.symlink_to(f.resolve())
        else:
            shutil.copy(f, mzML_dir)
    st.success("Example mzML files loaded!")


def remove_selected_mzML_files(to_remove: list[str], params: dict) -> dict:
    """
    Removes selected mzML files from the mzML directory.

    Args:
        to_remove (List[str]): List of mzML files to remove.
        params (dict): Parameters.


    Returns:
        dict: parameters with updated mzML files
    """
    mzML_dir = Path(st.session_state.workspace, "mzML-files")
    # remove all given files from mzML workspace directory and selected files
    for f in to_remove:
        Path(mzML_dir, f + ".mzML").unlink()
    for k, v in params.items():
        if isinstance(v, list):
            if f in v:
                params[k].remove(f)
    st.success("Selected mzML files removed!")
    return params


def save_uploaded_xic(uploaded_files: list[bytes]) -> list[tuple[str, str]]:
    """
    Saves uploaded XIC (.xic / parquet) files to the workspace xic directory.

    Returns a list of tuples (display_name, absolute_path) for the saved files.
    """
    xic_dir = Path(st.session_state.workspace, "xic-files")
    xic_dir.mkdir(parents=True, exist_ok=True)

    # Online mode: single file may be passed as non-list
    if st.session_state.location == "online":
        uploaded_files = [uploaded_files]

    if not uploaded_files:
        st.warning("Upload some XIC files first.")
        return []

    saved: list[tuple[str, str]] = []
    existing_names = [p.name for p in xic_dir.iterdir() if p.is_file()]

    for f in uploaded_files:
        # If a file with same name already exists, overwrite to ensure latest
        target = Path(xic_dir, f.name)
        with open(target, "wb") as fh:
            fh.write(f.getbuffer())
        saved.append((f.name, str(target.resolve())))

    st.success("Successfully added uploaded XIC files!")
    return saved


def remove_all_xic_files() -> None:
    """
    Remove all XIC files stored in the current workspace `xic-files` directory.
    Also removes session-state references if present.
    """
    xic_dir = Path(st.session_state.workspace, "xic-files")
    if not xic_dir.exists():
        st.info("No XIC files present in workspace.")
        return

    removed = 0
    for p in xic_dir.iterdir():
        try:
            if p.is_file():
                p.unlink()
                removed += 1
        except Exception:
            pass

    # Clear session state keys related to XIC viewer if present
    for key in ["xic_tmp_paths", "file_analytes", "shared_analytes", "files_loaded"]:
        if key in st.session_state:
            del st.session_state[key]

    if removed:
        st.success(f"Removed {removed} XIC file(s) from workspace.")
    else:
        st.info("No XIC files were removed.")


def remove_all_mzML_files(params: dict) -> dict:
    """
    Removes all mzML files from the mzML directory.

    Args:
        params (dict): Parameters.

    Returns:
        dict: parameters with updated mzML files
    """
    mzML_dir = Path(st.session_state.workspace, "mzML-files")
    # reset (delete and re-create) mzML directory in workspace
    reset_directory(mzML_dir)
    # reset all parameter items which have mzML in key and are list
    for k, v in params.items():
        if "mzML" in k and isinstance(v, list):
            params[k] = []
    st.success("All mzML files removed!")
    return params
