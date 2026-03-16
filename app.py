import streamlit as st
from pathlib import Path
import json

# For some reason the windows version only works if this is imported here
import pyopenms

if "settings" not in st.session_state:
    with open("settings.json", "r") as f:
        st.session_state.settings = json.load(f)

if __name__ == "__main__":
    pages = {
        str(st.session_state.settings["app-name"]): [
            st.Page(Path("content", "quickstart.py"), title="Quickstart", icon="👋"),
            st.Page(
                Path("content", "documentation.py"), title="Documentation", icon="📖"
            ),
        ],
        "Getting started with DIA": [
            st.Page(
                Path("content", "dia_00_concepts.py"),
                title="Concepts",
                icon="📚",
            ),
            st.Page(
                Path("content", "dia_01_targeted_data_extraction.py"),
                title="Targeted Data Extraction",
                icon="🧪",
            ),
            st.Page(
                Path("content", "dia_02_extracted_ion_chromatgoram_peak_picking_and_feature_scoring.py"),
                title="Peak Detection and Feature Scoring",
                icon="📶",
            )
        ],
        "Spectral Library Generation": [
            st.Page(
                Path("content", "insilico_spectral_library_generation.py"),
                title="Predicted Library",
                icon="📚",
            ),
            st.Page(
                Path("content", "openswathassay_generation.py"),
                title="Filter and Optimize Library",
                icon="🔧",
            ),
            st.Page(
                Path("content", "openswathdecoy_generation.py"),
                title="Generate/Append Decoys",
                icon="🎭",
            ),
        ],
        "pyOpenMS Toolbox": [
            st.Page(Path("content", "digest.py"), title="In Silico Digest", icon="✂️"),
            st.Page(
                Path("content", "peptide_mz_calculator.py"),
                title="m/z Calculator",
                icon="⚖️",
            ),
            st.Page(
                Path("content", "isotope_pattern_generator.py"),
                title="Isotopic Pattern Calculator",
                icon="📶",
            ),
            st.Page(
                Path("content", "fragmentation.py"),
                title="Fragment Ion Generation",
                icon="💥",
            ),
        ],
        "pyOpenMS Workflow": [
            st.Page(Path("content", "file_upload.py"), title="File Upload", icon="📂"),
            st.Page(
                Path("content", "raw_data_viewer.py"), title="View MS data", icon="👀"
            ),
            st.Page(
                Path("content", "run_example_workflow.py"),
                title="Run Workflow",
                icon="⚙️",
            ),
            st.Page(
                Path("content", "download_section.py"),
                title="Download Results",
                icon="⬇️",
            ),
        ],
        "Others Topics": [
            st.Page(
                Path("content", "simple_workflow.py"), title="Simple Workflow", icon="⚙️"
            ),
            st.Page(
                Path("content", "run_subprocess.py"), title="Run Subprocess", icon="🖥️"
            ),
        ],
    }

    pg = st.navigation(pages)
    pg.run()
