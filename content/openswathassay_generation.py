import json as _json
import time
import traceback
from pathlib import Path

import streamlit as st

from src.common.common import page_setup, save_params
from src.workflow.OpenSwathAssayGeneratorWorkflow import OpenSwathAssayGeneratorWorkflow

params = page_setup()

# --- Page Header -------------------------------------------------------------
st.title("🎯 OpenSwath Assay Generator")
st.markdown(
    """
Generate optimized **OpenSwath targeted assays** from spectral libraries using 
[OpenSwathAssayGenerator](https://abibuilder.cs.uni-tuebingen.de/archive/openms/Documentation/nightly/html/TOPP_OpenSwathAssayGenerator.html).
Configure the parameters below and generate targeted assays.
"""
)

# =============================================================================
# PARAMETER SECTIONS
# =============================================================================

st.markdown("---")

# --- Input File Upload ---
st.subheader("📥 Input Library File")

# Scan for existing EasyPQP results
default_workspace = Path(st.session_state.get("workspace", ".")).resolve()
easypqp_results_dir = Path(default_workspace, "insilico-spectral-library-generation", "results", "easypqp").resolve()
existing_libraries = []
if easypqp_results_dir.exists():
    existing_libraries = sorted([f.name for f in easypqp_results_dir.glob("*") if f.is_file()])

# Option 1: Select from existing EasyPQP libraries
if existing_libraries:
    st.write("**Option 1: Use existing EasyPQP library**")
    selected_lib = st.selectbox(
        "Select from existing libraries",
        options=[""] + existing_libraries,
        index=0,
        key="select_existing_library",
    )
    if selected_lib:
        selected_lib_path = Path(easypqp_results_dir, selected_lib)
        st.write(f"✅ Selected: {selected_lib} ({selected_lib_path.stat().st_size / (1024 * 1024):.1f} MB)")
else:
    selected_lib = None

# Option 2: Upload new file
st.write("**Option 2: Upload new library file**")
input_file = st.file_uploader(
    "Or upload spectral library (TraML, TSV, MRM, PQP, or oswpq)",
    type=["traml", "tsv", "mrm", "pqp", "oswpq"],
    key="input_library_file",
)
if input_file is not None:
    st.write(f"✅ Loaded: {input_file.name}")

# --- Output File Configuration ---
st.markdown("---")
st.subheader("📤 Output Configuration")
col1, col2 = st.columns(2)
with col1:
    output_file = st.text_input(
        "Output filename",
        value="openswath_target_assays.tsv",
        help="Name for the generated assay file. Format will be inferred from extension if set to 'auto'.",
    )
with col2:
    output_format = st.selectbox(
        "Output format",
        options=["auto", "tsv", "TraML", "pqp", "oswpq"],
        index=0,
        help="'auto' will infer format from filename extension. Otherwise explicitly sets the output format.",
    )

# --- Transition Parameters ---
st.markdown("---")
st.subheader("1️⃣ Transition Settings")
col1, col2 = st.columns(2)
with col1:
    min_transitions = st.number_input(
        "Min Transitions per Peptide",
        value=6,
        min_value=1,
        max_value=20,
        help="Minimum number of transitions to keep",
    )
with col2:
    max_transitions = st.number_input(
        "Max Transitions per Peptide",
        value=6,
        min_value=1,
        max_value=20,
        help="Maximum number of transitions to keep",
    )

# --- Fragment Settings ---
st.subheader("2️⃣ Fragment Ion Settings")
col1, col2 = st.columns(2)
with col1:
    allowed_fragment_types = st.text_input(
        "Allowed Fragment Types",
        value="b,y",
        help="Comma-separated fragment ion types (e.g., 'b,y')",
    )
with col2:
    allowed_fragment_charges = st.text_input(
        "Allowed Fragment Charges",
        value="1,2,3,4",
        help="Comma-separated charge states (e.g., '1,2,3,4')",
    )

col1, col2 = st.columns(2)
with col1:
    st.checkbox(
        "Enable detection-specific neutral losses",
        value=False,
        key="enable_detection_specific_losses",
    )
with col2:
    st.checkbox(
        "Enable detection-unspecific neutral losses",
        value=False,
        key="enable_detection_unspecific_losses",
    )

# --- Advanced Parameters (Collapsible) ---
with st.expander("⚙️ Advanced m/z Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Precursor m/z threshold (Thomson)",
            value=0.025,
            step=0.001,
            key="precursor_mz_threshold",
            help="M/z tolerance for precursor ion selection",
        )
    with col2:
        st.number_input(
            "Product m/z threshold (Thomson)",
            value=0.025,
            step=0.001,
            key="product_mz_threshold",
            help="M/z tolerance for fragment ion annotation",
        )
    
    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Precursor lower m/z limit",
            value=400.0,
            step=10.0,
            key="precursor_lower_mz_limit",
            help="Lower m/z boundary for precursor ions",
        )
    with col2:
        st.number_input(
            "Precursor upper m/z limit",
            value=1200.0,
            step=10.0,
            key="precursor_upper_mz_limit",
            help="Upper m/z boundary for precursor ions",
        )
    
    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Product lower m/z limit",
            value=350.0,
            step=10.0,
            key="product_lower_mz_limit",
            help="Lower m/z boundary for fragment ions",
        )
    with col2:
        st.number_input(
            "Product upper m/z limit",
            value=2000.0,
            step=10.0,
            key="product_upper_mz_limit",
            help="Upper m/z boundary for fragment ions",
        )

# --- IPF Settings ---
with st.expander("⚙️ Advanced IPF (Identification Transitions) Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        enable_ipf = st.checkbox(
            "Enable IPF (Identification Transitions)",
            value=False,
            key="enable_ipf",
            help="Generate identification-specific transitions for increased specificity",
        )
    with col2:
        if enable_ipf:
            st.number_input(
                "Max alternative localizations",
                value=10000,
                min_value=1,
                key="max_num_alternative_localizations",
                help="Maximum site-localization permutations for IPF",
            )
    
    if enable_ipf:
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox(
                "Disable MS2-level precursor ions for identification",
                value=False,
                key="disable_identification_ms2_precursors",
            )
        with col2:
            st.checkbox(
                "Disable identification-specific losses",
                value=False,
                key="disable_identification_specific_losses",
            )

# --- Optional Files ---
st.markdown("---")
st.subheader("3️⃣ Optional Files")
col1, col2 = st.columns(2)
with col1:
    unimod_file = st.file_uploader(
        "UniMod XML file (optional)",
        type=["xml"],
        key="unimod_file",
        help="Required if IPF is enabled",
    )
    if unimod_file is not None:
        st.write(f"✅ Loaded: {unimod_file.name}")

with col2:
    swath_windows_file = st.file_uploader(
        "SWATH windows file (optional, TSV)",
        type=["txt", "tsv"],
        key="swath_windows_file",
        help="Tab-separated file with SWATH window ranges",
    )
    if swath_windows_file is not None:
        st.write(f"✅ Loaded: {swath_windows_file.name}")

# --- Processing Options ---
st.markdown("---")
st.subheader("4️⃣ Processing Options")
threads = st.number_input(
    "Number of Threads",
    value=0,
    min_value=0,
    help="Number of parallel threads. `0` = use all available cores.",
)

st.markdown("---")

# =============================================================================
# RUN WORKFLOW
# =============================================================================
if st.button(
    "🚀 Run Assay Generation in Workspace", type="primary", use_container_width=True
):
    try:
        st.info("🔄 Initializing workflow...")

        # Instantiate workflow and prepare workspace input dirs
        wf = OpenSwathAssayGeneratorWorkflow()
        wf_dir = Path(wf.workflow_dir).resolve()  # Ensure absolute path
        st.write(f"✅ Workflow created: {wf_dir}")

        input_dest_dir = Path(wf_dir, "input-files", "lib")
        input_dest_dir.mkdir(parents=True, exist_ok=True)

        if input_file is None and not selected_lib:
            st.error("❌ Please either upload an input library file or select an existing one before running in workspace.")
        else:
            st.write("📤 Preparing files...")

            # === FILE UPLOADS ===
            # Copy input file - either from upload or from existing library
            if input_file is not None:
                input_name = getattr(input_file, "name", "library.tsv")
                input_path = Path(input_dest_dir, input_name)
                with open(input_path, "wb") as fh:
                    fh.write(input_file.getbuffer())
                st.write(f"✅ Input uploaded to: {input_path}")
            else:
                # Use selected existing library
                input_path = Path(easypqp_results_dir, selected_lib)
                st.write(f"✅ Using existing library: {input_path}")

            # Optional: copy unimod file
            unimod_path_str = None
            if unimod_file is not None:
                unimod_dest_dir = Path(wf_dir, "input-files", "unimod")
                unimod_dest_dir.mkdir(parents=True, exist_ok=True)
                unimod_path_str = str(Path(unimod_dest_dir, unimod_file.name))
                with open(unimod_path_str, "wb") as fh:
                    fh.write(unimod_file.getbuffer())
                st.write(f"✅ UniMod XML uploaded to: {unimod_path_str}")

            # Optional: copy SWATH windows file
            swath_windows_str = None
            if swath_windows_file is not None:
                swath_dest_dir = Path(wf_dir, "input-files", "swath")
                swath_dest_dir.mkdir(parents=True, exist_ok=True)
                swath_windows_str = str(Path(swath_dest_dir, swath_windows_file.name))
                with open(swath_windows_str, "wb") as fh:
                    fh.write(swath_windows_file.getbuffer())
                st.write(f"✅ SWATH windows file uploaded to: {swath_windows_str}")

            # === SET OUTPUT DIRECTORY ===
            results_dir = Path(wf_dir, "results", "openswath").resolve()  # Ensure absolute path
            results_dir.mkdir(parents=True, exist_ok=True)
            output_filename = (
                Path(output_file).name
                if output_file
                else "openswath_assays.tsv"
            )
            workspace_output_file = str(Path(results_dir, output_filename))
            st.write(f"✅ Output will go to: {workspace_output_file}")

            st.write("📝 Building configuration...")

            # === BUILD CONFIGURATION ===
            config_data = {
                "traml_input": str(input_path),
                "output_file": workspace_output_file,
                "min_transitions": int(min_transitions),
                "max_transitions": int(max_transitions),
                "allowed_fragment_types": str(allowed_fragment_types),
                "allowed_fragment_charges": str(allowed_fragment_charges),
                "precursor_mz_threshold": float(st.session_state.get("precursor_mz_threshold", 0.025)),
                "product_mz_threshold": float(st.session_state.get("product_mz_threshold", 0.025)),
                "precursor_lower_mz_limit": float(st.session_state.get("precursor_lower_mz_limit", 400.0)),
                "precursor_upper_mz_limit": float(st.session_state.get("precursor_upper_mz_limit", 1200.0)),
                "product_lower_mz_limit": float(st.session_state.get("product_lower_mz_limit", 350.0)),
                "product_upper_mz_limit": float(st.session_state.get("product_upper_mz_limit", 2000.0)),
                "enable_detection_specific_losses": bool(st.session_state.get("enable_detection_specific_losses", False)),
                "enable_detection_unspecific_losses": bool(st.session_state.get("enable_detection_unspecific_losses", False)),
                "enable_ipf": bool(st.session_state.get("enable_ipf", False)),
                "unimod_file": unimod_path_str,
                "swath_windows_file": swath_windows_str,
                "threads": int(threads),
            }

            # Add output format only if not 'auto' (auto infers from filename extension)
            if output_format != "auto":
                config_data["output_format"] = output_format

            # Add IPF settings if enabled
            if st.session_state.get("enable_ipf", False):
                config_data["max_num_alternative_localizations"] = int(st.session_state.get("max_num_alternative_localizations", 10000))
                config_data["disable_identification_ms2_precursors"] = bool(st.session_state.get("disable_identification_ms2_precursors", False))
                config_data["disable_identification_specific_losses"] = bool(st.session_state.get("disable_identification_specific_losses", False))

            st.write("✅ Configuration built")

            # === SAVE PARAMS ===
            params_to_write = {
                "traml": str(input_path),
                "output_file": workspace_output_file,
                "min_transitions": int(min_transitions),
                "max_transitions": int(max_transitions),
                "allowed_fragment_types": str(allowed_fragment_types),
                "allowed_fragment_charges": str(allowed_fragment_charges),
                "precursor_mz_threshold": float(st.session_state.get("precursor_mz_threshold", 0.025)),
                "product_mz_threshold": float(st.session_state.get("product_mz_threshold", 0.025)),
                "precursor_lower_mz_limit": float(st.session_state.get("precursor_lower_mz_limit", 400.0)),
                "precursor_upper_mz_limit": float(st.session_state.get("precursor_upper_mz_limit", 1200.0)),
                "product_lower_mz_limit": float(st.session_state.get("product_lower_mz_limit", 350.0)),
                "product_upper_mz_limit": float(st.session_state.get("product_upper_mz_limit", 2000.0)),
                "enable_detection_specific_losses": bool(st.session_state.get("enable_detection_specific_losses", False)),
                "enable_detection_unspecific_losses": bool(st.session_state.get("enable_detection_unspecific_losses", False)),
                "enable_ipf": bool(st.session_state.get("enable_ipf", False)),
                "unimod_file": unimod_path_str,
                "swath_windows_file": swath_windows_str,
                "threads": int(threads),
                "output_format": output_format,
            }

            # Ensure params file directory exists (use absolute path for consistency)
            params_file = Path(wf.parameter_manager.params_file).resolve()
            params_file.parent.mkdir(parents=True, exist_ok=True)
            with open(params_file, "w", encoding="utf-8") as f:
                _json.dump(params_to_write, f, indent=2)
            st.write(f"✅ Params saved to: {params_file}")

            st.write("🚀 Starting workflow process...")

            # === START WORKFLOW ===
            wf.start_workflow()
            st.write("✅ Workflow process spawned")

            # === DISPLAY SUCCESS MESSAGE & STATUS PANEL ===
            st.success("✅ Workflow submitted! Processing in background...")

            st.markdown("---")
            st.subheader("📊 Workflow Status")

            # Display status and links
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Workflow", wf.workflow_dir.name)
            with col2:
                st.metric("Status", "Running (check logs)")
            with col3:
                st.metric("Progress", "Monitor below ↓")

            # Show workspace info in expandable section
            with st.expander("📂 Workspace Info", expanded=False):
                st.write("**Workspace Location:**")
                st.code(str(wf_dir), language="bash")
                st.write("**Log Location:**")
                log_file = Path(wf_dir, "logs", "workflow.log")
                st.code(str(log_file), language="bash")

            # === AUTO-RERUN ON COMPLETION ===
            # Poll to check if workflow is still running
            # When done (PID dir removed), automatically rerun to show results
            pid_dir = wf.executor.pid_dir

            # Show a placeholder while waiting
            progress_placeholder = st.empty()

            # Poll for workflow completion (check every 2 seconds, timeout after 15 minutes)
            max_wait_time = 15 * 60  # 15 minutes
            start_time = time.time()
            poll_interval = 2  # seconds

            while time.time() - start_time < max_wait_time:
                if not pid_dir.exists():
                    # Workflow finished! PID directory was cleaned up
                    progress_placeholder.info("✅ Workflow completed! Refreshing to show results...")
                    time.sleep(1)  # Brief pause to ensure files are written
                    st.rerun()  # Automatically rerun to show download buttons
                    break

                # Still running - show progress
                elapsed = int(time.time() - start_time)
                progress_placeholder.info(f"⏳ Processing... ({elapsed}s elapsed)")
                time.sleep(poll_interval)
            else:
                # Timeout reached
                progress_placeholder.warning(
                    "⚠️ Workflow is taking longer than expected. "
                    "Check the logs or refresh the page to see results."
                )

    except Exception as e:
        st.error(f"❌ Failed to start workflow: {e}")
        st.text(traceback.format_exc())
        st.error("📋 Full traceback shown above - please check")

# =============================================================================
# ALWAYS CHECK FOR RESULTS (outside button handler, runs on every page render)
# =============================================================================
st.markdown("---")
st.subheader("📥 Download Results")

# Get the default workspace path
default_workspace = Path(st.session_state.get("workspace", ".")).resolve()
openswath_workspace = Path(default_workspace, "openswath-assay-generator").resolve()
results_dir = Path(openswath_workspace, "results", "openswath").resolve()

# Only show download section if the workspace exists
if openswath_workspace.exists() and results_dir.exists():
    # Scan results directory for generated files
    # Only keep files that actually exist (filter out any that were deleted or not yet created)
    all_assay_files = list(results_dir.glob("*"))
    assay_files = [f for f in all_assay_files if f.is_file() and f.exists()]

    if assay_files:
        st.success("✅ Results generated! Download below.")

        st.write("**📊 Assay Files:**")
        for assay_file in assay_files:
            try:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"📄 {assay_file.name}")
                with col2:
                    file_size = assay_file.stat().st_size / (1024 * 1024)  # MB
                    if file_size < 1:
                        st.write(f"({assay_file.stat().st_size / 1024:.1f} KB)")
                    else:
                        st.write(f"({file_size:.1f} MB)")
                with col3:
                    with open(assay_file, "rb") as f:
                        st.download_button(
                            label="⬇️ Download",
                            data=f.read(),
                            file_name=assay_file.name,
                            key=f"assay_{assay_file.name}",
                            use_container_width=True,
                        )
            except Exception as e:
                st.warning(f"⚠️ Could not access {assay_file.name}: {e}")

        # === CLEAR RESULTS BUTTON ===
        st.markdown("---")
        st.subheader("🗑️ Clear Results")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear Results", key="clear_results_btn", use_container_width=True):
                st.session_state.clear_results_confirm = True

        # Show confirmation modal if user clicked clear
        if st.session_state.get("clear_results_confirm", False):
            st.markdown("---")
            st.warning("⚠️ **WARNING: This will permanently delete the following files:**")

            # List all files that will be deleted
            for file in assay_files:
                st.write(f"  • `{file.name}` ({file.stat().st_size / (1024 * 1024) if file.stat().st_size > 1024 * 1024 else file.stat().st_size / 1024:.1f} {'MB' if file.stat().st_size > 1024 * 1024 else 'KB'})")

            st.markdown("")
            col1, col2, col3 = st.columns([2, 1, 1])

            with col2:
                if st.button("❌ Cancel", key="cancel_clear", use_container_width=True):
                    st.session_state.clear_results_confirm = False
                    st.rerun()

            with col3:
                if st.button("🗑️ Delete", key="confirm_clear", type="secondary", use_container_width=True):
                    try:
                        deleted_count = 0
                        for file in assay_files:
                            try:
                                file.unlink()
                                deleted_count += 1
                            except Exception as e:
                                st.warning(f"Could not delete {file.name}: {e}")

                        # Clear confirmation state
                        st.session_state.clear_results_confirm = False

                        # Show success message
                        st.success(f"✅ Successfully deleted {deleted_count} file(s)!")
                        st.info("The page will refresh to show the updated results section.")
                        time.sleep(1)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error clearing results: {e}")
    else:
        st.info(
            "⏳ Processing in progress... Refresh the page to check for newly generated files."
        )
else:
    st.info(
        "📂 No workspace found. Click '🚀 Run Assay Generation in Workspace' above to start."
    )

save_params(params)
