import shutil
from pathlib import Path

import streamlit as st

from src.common.common import page_setup, save_params
from src.workflow.ParameterManager import ParameterManager
from src.workflow.StreamlitUI import StreamlitUI
from src.workflow.PyProphet import PyProphetCLI
import json

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

# instantiate UI helper before rendering upstream tool sections
ui = StreamlitUI(workspace_dir, logger=None, executor=None, parameter_manager=pm)
# placeholder for OpenSwathWorkflow transition param key (may be set from upstream tools)
tr_key = f"{pm.topp_param_prefix}OpenSwathWorkflow:1:tr"

# --- Spectral library / upstream tools
st.markdown("---")
st.subheader("Spectral Library parameters")

# locate FASTA and library directories inside workspace input-files
fasta_dir = workspace_dir / "input-files" / "fasta"
lib_dir = workspace_dir / "input-files" / "libraries"
fasta_list = (
    [p.name for p in fasta_dir.iterdir() if p.is_file()] if fasta_dir.exists() else []
)
lib_list = (
    [p.name for p in lib_dir.iterdir() if p.is_file()] if lib_dir.exists() else []
)

st.markdown("Select or confirm inputs for spectral library generation and refinement.")

selected_fasta = None
if fasta_list:
    selected_fasta = st.selectbox(
        "FASTA in workspace", options=["None"] + fasta_list, index=0
    )
    if selected_fasta and selected_fasta != "None":
        st.session_state[f"{pm.param_prefix}fasta"] = selected_fasta
else:
    st.info(
        "No FASTA uploaded to workspace/input-files/fasta. Add via Uploads to enable insilico library generation."
    )

# easypqp insilico config (use provided template)
easypqp_asset = Path(
    "assets", "common-tool-descriptors", "easypqp_insilico", "easypqp_insilico.json"
)
if easypqp_asset.exists():
    st.markdown("**EasyPQP (insilico) configuration**")
    st.caption(
        "Basic insilico options - a template will be copied to workspace when requested"
    )
    # allow a small set of common overrides
    gen_decoys = st.checkbox("Generate decoys in insilico library", value=False)
    max_var_mods = st.number_input("Max variable modifications", value=3, min_value=0)
    peptide_min_mass = st.number_input("Peptide min mass", value=500.0)
    peptide_max_mass = st.number_input("Peptide max mass", value=5000.0)

    if st.button("Create insilico config and copy to workspace libraries"):
        # ensure libraries dir exists
        lib_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(easypqp_asset, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["database"]["generate_decoys"] = gen_decoys
            cfg["database"]["max_variable_mods"] = int(max_var_mods)
            cfg["database"]["peptide_min_mass"] = float(peptide_min_mass)
            cfg["database"]["peptide_max_mass"] = float(peptide_max_mass)
            target = lib_dir / "easypqp_insilico.json"
            with open(target, "w", encoding="utf-8") as out:
                json.dump(cfg, out, indent=2)
            st.success(f"Wrote insilico config -> {target}")
            lib_list.append(target.name)
        except Exception as e:
            st.error(f"Failed to write insilico config: {e}")

# OpenSwathAssayGenerator UI
osag_asset_dir = Path("assets", "common-tool-descriptors", "openswathassaygenerator")
osag_ini = next(osag_asset_dir.glob("*.ini"), None) if osag_asset_dir.exists() else None
if osag_ini:
    st.markdown("---")
    st.subheader("OpenSwathAssayGenerator (assay refinement)")
    # copy ini if not present
    dest_osag = ini_target_dir / "OpenSwathAssayGenerator.ini"
    if not dest_osag.exists():
        try:
            shutil.copy2(osag_ini, dest_osag)
        except Exception:
            pass
    # choose input transition list (from libraries or generated insilico)
    if lib_list:
        sel_assay_in = st.selectbox(
            "Transition list for assay generator", options=[None] + lib_list
        )
        if sel_assay_in:
            st.session_state[f"{pm.topp_param_prefix}OpenSwathAssayGenerator:1:in"] = (
                sel_assay_in
            )
    else:
        st.info("No transition lists present in workspace/input-files/libraries.")
    ui.input_TOPP(
        "OpenSwathAssayGenerator",
        num_cols=3,
        display_tool_name=False,
        exclude_parameters=["in"],
    )

# OpenSwathDecoyGenerator UI
osdg_asset_dir = Path("assets", "common-tool-descriptors", "openswathdecoygenerator")
osdg_ini = next(osdg_asset_dir.glob("*.ini"), None) if osdg_asset_dir.exists() else None
if osdg_ini:
    st.markdown("---")
    st.subheader("OpenSwathDecoyGenerator")
    dest_osdg = ini_target_dir / "OpenSwathDecoyGenerator.ini"
    if not dest_osdg.exists():
        try:
            shutil.copy2(osdg_ini, dest_osdg)
        except Exception:
            pass
    # default input for decoy generator is output of assay generator if provided, else allow selection
    assay_out_key = f"{pm.topp_param_prefix}OpenSwathAssayGenerator:1:out"
    default_decoy_in = None
    if assay_out_key in st.session_state and st.session_state[assay_out_key]:
        default_decoy_in = st.session_state[assay_out_key]
    elif lib_list:
        default_decoy_in = None

    if lib_list:
        sel_index = 0
        if default_decoy_in and default_decoy_in in lib_list:
            sel_index = lib_list.index(default_decoy_in) + 1
        sel_decoy_in = st.selectbox(
            "Input to decoy generator (transition list)",
            options=[None] + lib_list,
            index=sel_index,
        )
        if sel_decoy_in:
            st.session_state[f"{pm.topp_param_prefix}OpenSwathDecoyGenerator:1:in"] = (
                sel_decoy_in
            )
    ui.input_TOPP(
        "OpenSwathDecoyGenerator",
        num_cols=3,
        display_tool_name=False,
        exclude_parameters=["in"],
    )

# After spectral tool sections, if decoy output is present, set OpenSwathWorkflow tr to it
decoy_out_key = f"{pm.topp_param_prefix}OpenSwathDecoyGenerator:1:out"
if decoy_out_key in st.session_state and st.session_state[decoy_out_key]:
    st.session_state[tr_key] = st.session_state[decoy_out_key]
else:
    # fallback: use any selected library if present
    if lib_list:
        # prefer an explicitly selected one for workflow tr if present
        if (
            f"{pm.topp_param_prefix}OpenSwathAssayGenerator:1:out" in st.session_state
            and st.session_state[f"{pm.topp_param_prefix}OpenSwathAssayGenerator:1:out"]
        ):
            st.session_state[tr_key] = st.session_state[
                f"{pm.topp_param_prefix}OpenSwathAssayGenerator:1:out"
            ]
        else:
            # use first library as default
            st.session_state[tr_key] = lib_list[0]

st.markdown("---")
st.subheader("OpenSwathWorkflow Parameters")
st.markdown("Toggle 'Advanced' in the sidebar to show advanced parameters.")

# `advanced` toggle is provided in the global Settings sidebar expander
if "advanced" not in st.session_state:
    st.session_state["advanced"] = False

# Provide custom widgets for priority file params and blacklist them
tool_name = "OpenSwathWorkflow"
tpref = pm.topp_param_prefix

# Derive input mzML files from workspace uploads (no redundant uploader)
in_key = f"{tpref}{tool_name}:1:in"
# collect mzML files from workspace mzML-files dir
mzml_dir = Path(st.session_state.get("workspace", ".")) / "mzML-files"
mzml_list = []
if mzml_dir.exists():
    mzml_list = [
        p.name
        for p in mzml_dir.iterdir()
        if p.is_file() and "external_files.txt" not in p.name
    ]
    external = mzml_dir / "external_files.txt"
    if external.exists():
        with open(external, "r") as fh:
            ext = [l.strip() for l in fh.read().splitlines() if l.strip()]
            # show only basename for external file entries
            mzml_list += [Path(p).name for p in ext]

if mzml_list:
    st.markdown(f"**Detected mzML files ({len(mzml_list)}):**")
    for m in mzml_list:
        st.write(f"- {m}")
    # set TOPP 'in' key to newline-separated file list (matches previous behavior)
    st.session_state[in_key] = "\n".join(mzml_list)
else:
    st.warning("No mzML files found in workspace mzML-files. Add files via Uploads.")

# 'tr' (transition file) will be derived from selected or generated libraries below; set placeholder key
tr_key = f"{tpref}{tool_name}:1:tr"

# Output features: hard-coded and uneditable
out_features_key = f"{tpref}{tool_name}:1:out_features"
st.text_input(
    "Output features file",
    value="openswath_results.osw",
    key=out_features_key,
    disabled=True,
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
# Integrate PyProphet UI (passed as None in config page; available during workflow)
py = PyProphetCLI(pm, workspace_dir, executor=None, logger=None)
py.ui()
