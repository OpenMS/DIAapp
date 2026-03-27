import streamlit as st
from pathlib import Path
import json
import base64

# For some reason the windows version only works if this is imported here
import pyopenms


def get_base64_of_bin_file(png_file: str) -> str:
    with open(png_file, "rb") as f:
        return base64.b64encode(f.read()).decode()


@st.cache_resource
def build_markup_for_logo(png_file: str) -> str:
    binary_string = get_base64_of_bin_file(png_file)
    return f"""
            <style>
                [data-testid="stSidebarHeader"] {{
                    background-image: url("data:image/png;base64,{binary_string}");
                    background-repeat: no-repeat;
                    background-size: contain;
                    background-position: top center;
                }}
            </style>
            """


if "settings" not in st.session_state:
    with open("settings.json", "r") as f:
        st.session_state.settings = json.load(f)

if __name__ == "__main__":
    st.markdown(
        build_markup_for_logo(str(Path("assets", "OpenDIAKiosk_logo_portrait.png"))),
        unsafe_allow_html=True,
    )

    pages = {
        "Getting started with DIA": [
            st.Page(
                Path("content", "dia_00_concepts.py"),
                title="Concepts",
                icon="📚",
            ),
            st.Page(
                Path("content", "dia_01_targeted_data_extraction.py"),
                title="Targeted Data Extraction",
                icon="🎯",
            ),
            st.Page(
                Path(
                    "content",
                    "dia_02_extracted_ion_chromatgoram_peak_picking_and_feature_scoring.py",
                ),
                title="Peak Detection and Feature Scoring",
                icon="🔍",
            ),
            st.Page(
                Path("content", "dia_03_feature_scoring.py"),
                title="Feature Scoring",
                icon="⭐️",
            ),
            st.Page(
                Path("content", "dia_04_statistical_validation.py"),
                title="Statistical Validation",
                icon="📊",
            ),
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
        "OpenSwath": [
            st.Page(
                Path("content", "openswath_file_upload.py"),
                title="File Upload",
                icon="📁",
            ),
            st.Page(
                Path("content", "openswath_configuration.py"),
                title="Configuration",
                icon="⚖️",
            ),
            st.Page(
                Path("content", "xic_chromatogram_viewer.py"),
                title="XIC Chromatogram Viewer",
                icon="📊",
            ),
        ],
        "Others": [
            st.Page(
                Path("content", "log_viewer.py"),
                title="Log Viewer",
                icon="🧾",
            ),
            st.Page(
                Path("content", "workspace_viewer.py"),
                title="Workspace Viewer",
                icon="📁",
            ),
        ],
    }

    pg = st.navigation(pages)
    pg.run()
