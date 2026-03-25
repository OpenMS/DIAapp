import shutil
from pathlib import Path

import streamlit as st

from src.common.common import page_setup, save_params
from src.workflow.ParameterManager import ParameterManager
from src.workflow.StreamlitUI import StreamlitUI

# OpenSwath Configuration params


params = page_setup()

st.title("⚙️ OpenSwath Configuration")
st.markdown("""
Configure OpenSwathWorkflow parameters. Select a tool descriptor (CTD/INI)
and edit parameters before running OpenSwath.
""")

# Find available CTD/INI descriptors bundled in the app
assets_dir = Path("assets", "common-tool-descriptors", "openswathworkflow")
inis = []
if assets_dir.exists():
    inis = sorted([p.name for p in assets_dir.glob("*.ini")])

if not inis:
    st.error(
        "No OpenSwath descriptor INI files found in assets/common-tool-descriptors/openswathworkflow."
    )
    st.stop()

selected_ini = st.selectbox("Select OpenSwath descriptor (INI)", options=inis, index=0)

# Prepare workspace ini directory via ParameterManager
workspace_dir = Path(st.session_state.get("workspace", "."))
pm = ParameterManager(workspace_dir)
ini_target_dir = pm.ini_dir
ini_target_dir.mkdir(parents=True, exist_ok=True)

# Copy selected ini into workspace ini dir as OpenSwathWorkflow.ini
src_ini = assets_dir / selected_ini
dest_ini = ini_target_dir / "OpenSwathWorkflow.ini"
if not dest_ini.exists() or st.button(
    "Overwrite workspace INI with selected descriptor"
):
    try:
        shutil.copy2(src_ini, dest_ini)
        st.success(f"Copied {selected_ini} -> {dest_ini}")
    except Exception as e:
        st.error(f"Failed to copy INI: {e}")

# Instantiate a StreamlitUI helper and render TOPP params for OpenSwathWorkflow
ui = StreamlitUI(workspace_dir, logger=None, executor=None, parameter_manager=pm)

st.markdown("---")
st.subheader("OpenSwathWorkflow Parameters")
st.markdown("Toggle 'Advanced' in the sidebar to show advanced parameters.")

# `advanced` toggle is provided in the global Settings sidebar expander
if "advanced" not in st.session_state:
    st.session_state["advanced"] = False

# Provide custom widgets for priority file params and blacklist them
tool_name = "OpenSwathWorkflow"
tpref = pm.topp_param_prefix

# 'in' (multiple input data files)
in_key = f"{tpref}{tool_name}:1:in"
in_widget_key = f"{in_key}_uploader"
in_files = st.file_uploader(
    "Input files (mzML/sqMass)",
    accept_multiple_files=True,
    type=["mzML", "mzxml", "sqMass"],
    key=in_widget_key,
)
# Do not assign to the same key used by the widget; write uploaded filenames into the TOPP param key
if in_files is not None:
    st.session_state[in_key] = "\n".join([getattr(f, "name", str(f)) for f in in_files])

# 'tr' (transition file)
tr_key = f"{tpref}{tool_name}:1:tr"
tr_widget_key = f"{tr_key}_uploader"
tr_file = st.file_uploader(
    "Transition file (TraML/TSV/PQP)",
    accept_multiple_files=False,
    type=["traml", "tsv", "pqp"],
    key=tr_widget_key,
)
if tr_file is not None:
    st.session_state[tr_key] = getattr(tr_file, "name", str(tr_file))

# Output file paths
out_features_key = f"{tpref}{tool_name}:1:out_features"
st.text_input(
    "Output features file",
    value=st.session_state.get(out_features_key, ""),
    key=out_features_key,
)


# Exclude these keys from auto-generated UI
exclude_keys = [
    "in",
    "tr",
    "out_features",
]

ui.input_TOPP(
    "OpenSwathWorkflow",
    num_cols=3,
    display_tool_name=True,
    display_subsections=True,
    exclude_parameters=exclude_keys,
)

# Save parameters button
if st.button("Save OpenSwath parameters to workspace params.json"):
    pm.save_parameters()
    save_params(pm.get_parameters_from_json())
    st.success("Parameters saved to workspace params.json")

# --- PyProphet params link / info
st.markdown("---")
st.subheader("PyProphet Parameters")
st.markdown(
    "Refer to the PyProphet CLI for scoring and infer parameters; these are configured separately."
)
