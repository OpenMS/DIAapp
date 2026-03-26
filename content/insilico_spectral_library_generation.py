import os
import json as _json
import time
import traceback
from pathlib import Path

import streamlit as st

from src.common.common import page_setup, save_params
from src.workflow.EasyPQPWorkflow import EasyPQPWorkflow

params = page_setup()

# --- Page Header -------------------------------------------------------------
st.title("🔬 In-Silico Predicted Spectral Library")
st.markdown(
    """
Generate **in-silico predicted spectral libraries** using
[EasyPQP](https://github.com/grosenberger/easypqp).
Configure the parameters below and run the library generation.
"""
)
with st.expander("📖 About EasyPQP In-Silico Library Generation"):
    st.markdown(
        """
        EasyPQP generates predicted spectral libraries by:
        - **Digesting** protein FASTA files with configurable enzyme parameters
        - **Predicting** MS2 fragment intensities, retention time, and ion mobility using deep-learning models

        For more information see the
        [EasyPQP documentation](https://github.com/grosenberger/easypqp?tab=readme-ov-file#generating-an-in-silico-library).
        """
    )

st.markdown("---")

# Set REDEEM_PRETRAINED_MODELS_DIR env variable to use shipped pretrained models data/pretrained_models
os.environ["REDEEM_PRETRAINED_MODELS_DIR"] = str(
    Path("data", "pretrained_models").resolve()
)

# =============================================================================
# SECTION 1: INPUT & OUTPUT
# =============================================================================
st.subheader("📂 Input & Output")
col1, col2 = st.columns([1, 1])
with col1:
    fasta_file = st.file_uploader(
        "FASTA File",
        type=["fasta", "fa", "faa", "fna"],
        help="FASTA file with protein sequences to digest and generate a library from.",
    )
with col2:
    output_file = st.text_input(
        "Output File",
        value="easypqp_insilico_library.tsv",
        help="Output path for the generated spectral library (TSV or Parquet).",
    )

st.markdown("---")

# =============================================================================
# SECTION 2: DATABASE SETTINGS
# =============================================================================
st.subheader("🗄️ Database Settings")

# Main database parameters
col1, col2 = st.columns([1, 2])
with col1:
    generate_decoys = st.toggle(
        "Generate Decoys",
        value=False,
        help="Generate a decoy library alongside the target library.",
    )
with col2:
    decoy_tag = st.text_input(
        "Decoy Tag",
        value="rev_",
        help="Prefix tag applied to decoy protein/peptide identifiers.",
        disabled=not generate_decoys,
    )

# Advanced database parameters in collapsible
with st.expander("⚙️ Advanced Database Settings", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        enzyme = st.selectbox(
            "Enzyme",
            ["Trypsin", "Lys-C", "Asp-N", "LysN", "GluC", "ArgC"],
            index=0,
            key="adv_enzyme",
            help="Enzyme used for protein digestion.",
        )
        missed_cleavages = st.number_input(
            "Missed Cleavages",
            min_value=0,
            max_value=5,
            value=0,
            key="adv_missed_cleavages",
            help="Number of allowed missed cleavages.",
        )
        cleave_at = st.text_input(
            "Cleave At",
            value="KR",
            max_chars=10,
            key="adv_cleave_at",
            help="Residues where the enzyme cleaves (e.g., 'KR' for Trypsin).",
        )
        restrict = st.text_input(
            "Restrict (before this residue)",
            value="P",
            max_chars=10,
            key="adv_restrict",
            help="Residues that prevent cleavage (e.g., 'P' for Proline with Trypsin).",
        )
    with col2:
        peptide_min_mass = st.number_input(
            "Min Peptide Mass (Da)",
            min_value=100.0,
            value=500.0,
            step=10.0,
            key="adv_min_mass",
            help="Minimum peptide mass threshold.",
        )
        peptide_max_mass = st.number_input(
            "Max Peptide Mass (Da)",
            min_value=1000.0,
            value=5000.0,
            step=100.0,
            key="adv_max_mass",
            help="Maximum peptide mass threshold.",
        )
        min_peptide_len = st.number_input(
            "Min Peptide Length",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            key="adv_min_pep_len",
            help="Minimum peptide length (0 = no restriction).",
        )
        max_peptide_len = st.number_input(
            "Max Peptide Length",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            key="adv_max_pep_len",
            help="Maximum peptide length (0 = no restriction).",
        )
    with col3:
        max_variable_mods = st.number_input(
            "Max Variable Modifications",
            min_value=0,
            max_value=10,
            value=3,
            key="adv_max_var_mods",
            help="Maximum number of variable modifications per peptide.",
        )
        semi_enzymatic = st.toggle(
            "Semi-Enzymatic Cleavage",
            value=False,
            key="adv_semi_enzymatic",
            help="Allow peptides with only one enzymatic end.",
        )
        c_terminal = st.toggle(
            "C-Terminal Specific",
            value=False,
            key="adv_c_terminal",
            help="Require C-terminal cleavage specificity.",
        )

    # --- Static Modifications ---
    st.divider()
    st.markdown(
        "**Static Modifications** — Fixed mass shifts applied to all instances of specified residues."
    )
    static_mods_col1, static_mods_col2 = st.columns([2, 1])

    # Initialize session state for static mods
    if "static_mods" not in st.session_state:
        st.session_state.static_mods = [
            ("C", 57.0215)
        ]  # Default: Carbamidomethyl-Cysteine

    with static_mods_col1:
        st.markdown("**Residue** → **Mass (Da)**")
        for i, (residue, mass) in enumerate(st.session_state.static_mods):
            col_a, col_b, col_c = st.columns([1, 2, 0.5])
            with col_a:
                new_residue = st.text_input(
                    f"Residue {i}",
                    value=residue,
                    max_chars=1,
                    key=f"static_res_{i}",
                )
            with col_b:
                new_mass = st.number_input(
                    f"Mass {i}",
                    value=mass,
                    format="%.4f",
                    key=f"static_mass_{i}",
                )
            with col_c:
                if st.button(
                    "✕", key=f"rm_static_{i}", help="Remove this modification"
                ):
                    st.session_state.static_mods.pop(i)
                    st.rerun()
            # Update session state with new values
            st.session_state.static_mods[i] = (new_residue, new_mass)

    with static_mods_col2:
        if st.button("➕ Add Static Mod", key="add_static_mod"):
            st.session_state.static_mods.append(("", 0.0))
            st.rerun()

    # --- Variable Modifications ---
    st.divider()
    st.markdown(
        """**Variable Modifications** — Optional mass shifts with position markers: 
\n
- `^X` = Modification to be applied to amino acid X if it appears at the N-terminus of a peptide
- `$X` = Modification to be applied to amino acid X if it appears at the C-terminus of a peptide
- `[X` = Modification to be applied to amino acid X if it appears at the N-terminus of a protein
- `]X` = Modification to be applied to amino acid X if it appears at the C-terminus of a protein
"""
    )
    var_mods_col1, var_mods_col2 = st.columns([2, 1])

    # Initialize session state for variable mods
    if "variable_mods" not in st.session_state:
        st.session_state.variable_mods = [
            ("[", [42.0]),  # N-terminal acetylation
            # ("$K", [8.014199]),  # Lysine
            ("M", [15.9949]),  # Methionine
            # ("$R", [10.008269]),  # Arginine
        ]

    with var_mods_col1:
        st.markdown("**Position Marker** → **Masses (comma-separated)**")
        for i, (position, masses) in enumerate(st.session_state.variable_mods):
            col_a, col_b, col_c = st.columns([1, 2, 0.5])
            with col_a:
                new_position = st.text_input(
                    f"Marker {i}",
                    value=position,
                    max_chars=3,
                    key=f"var_pos_{i}",
                    help="e.g., '[' (N-term), '$K' (Lys), 'M' (Met), '$R' (Arg), '' (C-term)",
                )
            with col_b:
                masses_str = ", ".join(str(m) for m in masses)
                new_masses_str = st.text_input(
                    f"Masses {i}",
                    value=masses_str,
                    key=f"var_masses_{i}",
                    help="Comma-separated mass values",
                )
                try:
                    new_masses = [
                        float(m.strip()) for m in new_masses_str.split(",") if m.strip()
                    ]
                except ValueError:
                    new_masses = masses
            with col_c:
                if st.button("✕", key=f"rm_var_{i}", help="Remove this modification"):
                    st.session_state.variable_mods.pop(i)
                    st.rerun()
            # Update session state with new values
            st.session_state.variable_mods[i] = (new_position, new_masses)

    with var_mods_col2:
        if st.button("➕ Add Var Mod", key="add_var_mod"):
            st.session_state.variable_mods.append(("", [0.0]))
            st.rerun()

st.markdown("---")

# =============================================================================
# SECTION 3: IN-SILICO SETTINGS
# =============================================================================
st.subheader("⚛️ In-Silico Library Generation")

# Main in-silico parameters
col1, col2, col3 = st.columns(3)
with col1:
    precursor_charge = st.text_input(
        "Precursor Charge(s)",
        value="2,3",
        help="Comma-separated precursor charge states (e.g. `2,3` or `1,2,3`).",
    )
with col2:
    max_fragment_charge = st.number_input(
        "Max Fragment Charge",
        min_value=1,
        max_value=8,
        value=2,
        step=1,
        help="Maximum charge state for fragment ions.",
    )
with col3:
    fragmentation_model = st.selectbox(
        "Fragmentation Model",
        options=[
            "hcd",
            "cid",
            "etd",
            "td_etd",
            "ethcd",
            "etcad",
            "eacid",
            "ead",
            "all",
            "none",
        ],
        index=0,
        help=(
            "Fragmentation model for theoretical fragment ion generation. "
            "See [FragmentationModel docs](https://docs.rs/rustyms/latest/rustyms/model/struct.FragmentationModel.html)."
        ),
    )

col1, col2 = st.columns(2)
with col1:
    min_transitions = st.number_input(
        "Min Transitions",
        min_value=1,
        value=6,
        step=1,
        help="Minimum number of fragment transitions retained per peptide.",
    )
with col2:
    max_transitions = st.number_input(
        "Max Transitions",
        min_value=1,
        value=6,
        step=1,
        help="Maximum number of fragment transitions retained per peptide.",
    )

allowed_fragment_types = st.text_input(
    "Allowed Fragment Types",
    value="b,y",
    help="Comma-separated ion types to include (e.g. `b,y`). Current MS2 prediction supports b and y ions only.",
)

# Advanced in-silico parameters in collapsible
with st.expander("⚙️ Advanced In-Silico Settings", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        rt_scale = st.number_input(
            "RT Scale Factor",
            min_value=0.0,
            value=100.0,
            step=1.0,
            key="adv_rt_scale",
            help="RT output scaling factor (e.g. `100.0` converts 0–1 range to 0–100).",
        )
        unimod_annotation = st.toggle(
            "UniMod Annotation",
            value=True,
            key="adv_unimod_ann",
            help=(
                "Re-annotate mass-bracket modifications (e.g. `[+57.0215]`) "
                "to UniMod notation (e.g. `(UniMod:4)`)."
            ),
        )
    with col2:
        max_delta_unimod = st.number_input(
            "Max Delta UniMod (Da)",
            min_value=0.0,
            value=0.02,
            step=0.001,
            format="%.4f",
            key="adv_max_delta_unimod",
            help="Maximum mass tolerance (Da) for matching modifications to UniMod entries.",
            disabled=not unimod_annotation,
        )
    with col3:
        enable_unannotated = st.toggle(
            "Keep Unannotated Modifications",
            value=True,
            key="adv_unannotated",
            help=(
                "Retain mass-bracket notation for modifications that cannot be matched to UniMod. "
                "If disabled, raises an error for unmatched modifications."
            ),
            disabled=not unimod_annotation,
        )

    unimod_xml_file = st.file_uploader(
        "Custom UniMod XML Database (optional)",
        type=["xml"],
        key="adv_unimod_file",
        help="Custom UniMod XML database file. Leave empty to use the built-in database.",
    )

st.markdown("---")

# =============================================================================
# SECTION 4: DEEP LEARNING PREDICTION
# =============================================================================
st.subheader("🎛️ Deep Learning Prediction")

# Main DL parameters
col1, col2, col3 = st.columns(3)
with col1:
    instrument = st.selectbox(
        "Instrument",
        options=["QE", "Lumos", "timsTOF", "SciexTOF", "ThermoTOF"],
        index=0,
        help="Mass spectrometer instrument type for MS2 intensity prediction.",
    )
with col2:
    nce = st.number_input(
        "NCE (Normalized Collision Energy)",
        min_value=1,
        max_value=100,
        value=20,
        step=1,
        help="Normalized collision energy used for MS2 intensity prediction.",
    )
with col3:
    batch_size = st.number_input(
        "Batch Size",
        min_value=1,
        value=10,
        step=1,
        help="Number of peptides processed per batch during inference.",
    )

# Fine-tuning section
st.markdown("**Fine-Tuning Options:**")
col1, col2 = st.columns([1, 2])
with col1:
    fine_tune = st.toggle(
        "Fine-tune Models",
        value=False,
        help="Fine-tune the prediction models using your own training data.",
    )
with col2:
    save_model = st.toggle(
        "Save Fine-tuned Model",
        value=False,
        disabled=not fine_tune,
        help="Save the fine-tuned model weights to disk after training.",
    )

train_data_file = None
if fine_tune:
    train_data_file = st.file_uploader(
        "Training Data (TSV)",
        type=["tsv"],
        help=(
            "TSV file with columns: `fullpeptidename`,	`precursor_mz`,	`precursor_charge`,	`rt`,	`ion_mobility`,	`fragment_type`,	`fragment_series_number`,	`product_charge`,	`intensity`."
        ),
    )

# Advanced DL parameters in collapsible
with st.expander("⚙️ Advanced Deep Learning Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        device = st.selectbox(
            "Compute Device",
            ["cpu", "cuda"],
            index=0,
            key="adv_device",
            help="Compute device for deep learning prediction.",
        )
    with col2:
        peptide_chunking = st.number_input(
            "Peptide Chunking",
            min_value=0,
            value=0,
            step=1000,
            key="adv_chunking",
            help="Number of peptides per chunk (0 = auto-calculate based on available memory).",
        )

    if fine_tune and train_data_file is not None:
        st.divider()
        st.markdown("**Fine-Tuning Advanced Settings:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.00001,
                max_value=0.1,
                value=0.001,
                format="%.5f",
                key="adv_lr",
                help="Learning rate for fine-tuning.",
            )
        with col2:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=100,
                value=10,
                key="adv_epochs",
                help="Number of training epochs for fine-tuning.",
            )
        with col3:
            ft_batch_size = st.number_input(
                "Training Batch Size",
                min_value=1,
                max_value=512,
                value=32,
                key="adv_ft_batch",
                help="Batch size for fine-tuning training.",
            )

st.markdown("---")

# =============================================================================
# SECTION 5: OUTPUT OPTIONS
# =============================================================================
st.subheader("📤 Output Options")
col1, col2, col3, col4 = st.columns(4)
with col1:
    write_report = st.toggle(
        "Write HTML Report",
        value=True,
        help="Generate an HTML quality report alongside the library.",
    )
with col2:
    parquet_output = st.toggle(
        "Parquet Output",
        value=False,
        help="Output library in Parquet format instead of TSV.",
    )
with col3:
    threads = st.number_input(
        "Threads",
        min_value=0,
        value=0,
        step=1,
        help="Number of parallel threads. `0` = use all available cores.",
    )

st.markdown("---")

# =============================================================================
# RUN WORKFLOW
# =============================================================================
if st.button(
    "🚀 Run Library Generation in Workspace", type="primary", use_container_width=True
):
    try:
        st.info("🔄 Initializing workflow...")

        # Instantiate workflow and prepare workspace input dirs
        wf = EasyPQPWorkflow()
        wf_dir = Path(wf.workflow_dir).resolve()  # Ensure absolute path
        st.write(f"✅ Workflow created: {wf_dir}")

        fasta_dest_dir = Path(wf_dir, "input-files", "fasta")
        fasta_dest_dir.mkdir(parents=True, exist_ok=True)

        if fasta_file is None:
            st.error("❌ Please upload a FASTA file before running in workspace.")
        else:
            st.write("📤 Uploading files...")

            # === FILE UPLOADS ===
            # Copy FASTA file
            fasta_name = getattr(fasta_file, "name", "uploaded.fasta")
            fasta_path = Path(fasta_dest_dir, fasta_name)
            with open(fasta_path, "wb") as fh:
                fh.write(fasta_file.getbuffer())
            st.write(f"✅ FASTA uploaded to: {fasta_path}")

            # Optional: copy unimod file
            unimod_path_str = None
            if unimod_xml_file is not None:
                unimod_dest_dir = Path(wf_dir, "input-files", "unimod")
                unimod_dest_dir.mkdir(parents=True, exist_ok=True)
                unimod_path_str = str(Path(unimod_dest_dir, unimod_xml_file.name))
                with open(unimod_path_str, "wb") as fh:
                    fh.write(unimod_xml_file.getbuffer())
                st.write(f"✅ UniMod XML uploaded to: {unimod_path_str}")

            # Optional: copy training data
            train_data_str = None
            if fine_tune and train_data_file is not None:
                train_dest_dir = Path(wf_dir, "input-files", "train-data")
                train_dest_dir.mkdir(parents=True, exist_ok=True)
                train_data_str = str(Path(train_dest_dir, train_data_file.name))
                with open(train_data_str, "wb") as fh:
                    fh.write(train_data_file.getbuffer())
                st.write(f"✅ Training data uploaded to: {train_data_str}")

            # === SET OUTPUT DIRECTORY ===
            results_dir = Path(
                wf_dir, "results", "insilico"
            ).resolve()  # Ensure absolute path
            results_dir.mkdir(parents=True, exist_ok=True)
            output_filename = (
                Path(output_file).name
                if output_file
                else "easypqp_insilico_library.tsv"
            )
            workspace_output_file = str(Path(results_dir, output_filename))
            st.write(f"✅ Output will go to: {workspace_output_file}")

            st.write("⚙️ Gathering advanced parameters...")

            # === GATHER PARAMETERS FROM SESSION STATE ===
            # Get advanced parameters from session state (with defaults if not set)
            enzyme = st.session_state.get("adv_enzyme", "Trypsin")
            missed_cleavages = st.session_state.get("adv_missed_cleavages", 0)
            min_peptide_len = st.session_state.get("adv_min_pep_len", 0)
            max_peptide_len = st.session_state.get("adv_max_pep_len", 0)
            cleave_at = st.session_state.get("adv_cleave_at", "KR")
            restrict = st.session_state.get("adv_restrict", "P")
            semi_enzymatic = st.session_state.get("adv_semi_enzymatic", False)
            c_terminal = st.session_state.get("adv_c_terminal", False)
            max_variable_mods = st.session_state.get("adv_max_var_mods", 3)
            peptide_min_mass = st.session_state.get("adv_min_mass", 500.0)
            peptide_max_mass = st.session_state.get("adv_max_mass", 5000.0)
            rt_scale = st.session_state.get("adv_rt_scale", 100.0)
            unimod_annotation = st.session_state.get("adv_unimod_ann", True)
            max_delta_unimod = st.session_state.get("adv_max_delta_unimod", 0.02)
            enable_unannotated = st.session_state.get("adv_unannotated", True)
            device = st.session_state.get("adv_device", "cpu")
            peptide_chunking = st.session_state.get("adv_chunking", 0)
            learning_rate = st.session_state.get("adv_lr", 0.001)
            epochs = st.session_state.get("adv_epochs", 10)
            ft_batch_size = st.session_state.get("adv_ft_batch", 32)

            st.write("📝 Building configuration...")

            # === BUILD MERGED JSON CONFIG (per JSON schema) ===
            # Prepare static mods dictionary (filter out empty entries)
            static_mods_dict = {
                res: mass for res, mass in st.session_state.static_mods if res.strip()
            }

            # Prepare variable mods dictionary (filter out empty entries)
            variable_mods_dict = {
                pos: masses
                for pos, masses in st.session_state.variable_mods
                if pos.strip() and masses
            }

            merged_config = {
                "database": {
                    "fasta": str(fasta_path),
                    "generate_decoys": bool(generate_decoys),
                    "decoy_tag": str(decoy_tag),
                    "enzyme": {
                        "missed_cleavages": int(missed_cleavages),
                        "min_len": int(min_peptide_len)
                        if min_peptide_len > 0
                        else None,
                        "max_len": int(max_peptide_len)
                        if max_peptide_len > 0
                        else None,
                        "cleave_at": str(cleave_at) if cleave_at.strip() else None,
                        "restrict": str(restrict) if restrict.strip() else None,
                        "c_terminal": bool(c_terminal) if c_terminal else None,
                        "semi_enzymatic": bool(semi_enzymatic)
                        if semi_enzymatic
                        else None,
                    },
                    "max_variable_mods": int(max_variable_mods),
                    "peptide_min_mass": float(peptide_min_mass),
                    "peptide_max_mass": float(peptide_max_mass),
                    "static_mods": static_mods_dict,
                    "variable_mods": variable_mods_dict,
                },
                "insilico_settings": {
                    "precursor_charge": [
                        int(c.strip())
                        for c in str(precursor_charge).split(",")
                        if c.strip()
                    ],
                    "max_fragment_charge": int(max_fragment_charge),
                    "min_transitions": int(min_transitions),
                    "max_transitions": int(max_transitions),
                    "fragmentation_model": str(fragmentation_model),
                    "allowed_fragment_types": [
                        t.strip()
                        for t in str(allowed_fragment_types).split(",")
                        if t.strip()
                    ],
                    "rt_scale": float(rt_scale),
                    "unimod_annotation": bool(unimod_annotation),
                    "max_delta_unimod": float(max_delta_unimod),
                    "enable_unannotated": bool(enable_unannotated),
                    "unimod_xml_path": unimod_path_str,
                },
                "dl_feature_generators": {
                    "device": str(device),
                    "instrument": str(instrument),
                    "nce": float(nce),
                    "batch_size": int(batch_size),
                },
                "peptide_chunking": int(peptide_chunking),
                "output_file": workspace_output_file,
                "write_report": bool(write_report),
                "parquet_output": bool(parquet_output),
            }

            # Add fine-tuning config if enabled
            if fine_tune and train_data_str:
                merged_config["dl_feature_generators"]["fine_tune_config"] = {
                    "fine_tune": True,
                    "train_data_path": train_data_str,
                    "save_model": bool(save_model),
                    "learning_rate": float(learning_rate),
                    "epochs": int(epochs),
                    "batch_size": int(ft_batch_size),
                }

            # Optional: add threads if specified
            if int(threads) > 0:
                merged_config["threads"] = int(threads)

            # === SAVE MERGED CONFIG TO WORKFLOW DIRECTORY ===
            config_path = Path(wf_dir, "easypqp_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                _json.dump(merged_config, f, indent=2)
            st.write(f"✅ Config saved to: {config_path}")

            # === SAVE PARAMS TO WORKFLOW PARAMETER MANAGER ===
            params_to_write = {
                "config_file": str(config_path),
                "output_file": workspace_output_file,
                "write_report": bool(write_report),
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

            # Show workspace info in expandable section (not the main focus)
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
                    progress_placeholder.info(
                        "✅ Workflow completed! Refreshing to show results..."
                    )
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
easypqp_workspace = Path(default_workspace, "easypqp-insilico").resolve()
results_dir = Path(easypqp_workspace, "results", "insilico").resolve()

# Only show download section if the workspace exists
if easypqp_workspace.exists() and results_dir.exists():
    # Scan results directory for generated files
    # Only keep files that actually exist (filter out any that were deleted or not yet created)
    all_tsv_files = list(results_dir.glob("*.tsv"))
    all_html_files = list(results_dir.glob("*.html"))
    all_parquet_files = list(results_dir.glob("*.parquet"))

    tsv_files = [f for f in all_tsv_files if f.exists()]
    html_files = [f for f in all_html_files if f.exists()]
    report_files = [f for f in all_parquet_files if f.exists()]

    if tsv_files or html_files or report_files:
        st.success("✅ Results generated! Download below.")

        # TSV Library files
        if tsv_files:
            st.write("**📊 TSV Spectral Libraries:**")
            for tsv_file in tsv_files:
                try:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"📄 {tsv_file.name}")
                    with col2:
                        file_size = tsv_file.stat().st_size / (1024 * 1024)  # MB
                        st.write(f"({file_size:.1f} MB)")
                    with col3:
                        with open(tsv_file, "rb") as f:
                            st.download_button(
                                label="⬇️ Download",
                                data=f.read(),
                                file_name=tsv_file.name,
                                key=f"tsv_{tsv_file.name}",
                                use_container_width=True,
                            )
                except Exception as e:
                    st.warning(f"⚠️ Could not access {tsv_file.name}: {e}")

        # HTML Report files
        if html_files:
            st.write("**📈 HTML Reports:**")
            for html_file in html_files:
                try:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"📊 {html_file.name}")
                    with col2:
                        file_size = html_file.stat().st_size / (1024 * 1024)  # MB
                        st.write(f"({file_size:.1f} MB)")
                    with col3:
                        with open(html_file, "rb") as f:
                            st.download_button(
                                label="⬇️ Download",
                                data=f.read(),
                                file_name=html_file.name,
                                key=f"html_{html_file.name}",
                                use_container_width=True,
                            )
                except Exception as e:
                    st.warning(f"⚠️ Could not access {html_file.name}: {e}")

        # Parquet files
        if report_files:
            st.write("**📋 Raw Data (Parquet):**")
            for parquet_file in report_files:
                try:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"📁 {parquet_file.name}")
                    with col2:
                        file_size = parquet_file.stat().st_size / (1024 * 1024)  # MB
                        st.write(f"({file_size:.1f} MB)")
                    with col3:
                        with open(parquet_file, "rb") as f:
                            st.download_button(
                                label="⬇️ Download",
                                data=f.read(),
                                file_name=parquet_file.name,
                                key=f"parquet_{parquet_file.name}",
                                use_container_width=True,
                            )
                except Exception as e:
                    st.warning(f"⚠️ Could not access {parquet_file.name}: {e}")

        # === CLEAR RESULTS BUTTON ===
        st.markdown("---")
        st.subheader("🗑️ Clear Results")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button(
                "🗑️ Clear Results", key="clear_results_btn", use_container_width=True
            ):
                st.session_state.clear_results_confirm = True

        # Show confirmation modal if user clicked clear
        if st.session_state.get("clear_results_confirm", False):
            st.markdown("---")
            st.warning(
                "⚠️ **WARNING: This will permanently delete the following files:**"
            )

            # List all files that will be deleted
            all_files = tsv_files + html_files + report_files
            for file in all_files:
                st.write(
                    f"  • `{file.name}` ({file.stat().st_size / (1024 * 1024):.1f} MB)"
                )

            st.markdown("")
            col1, col2, col3 = st.columns([2, 1, 1])

            with col2:
                if st.button("❌ Cancel", key="cancel_clear", use_container_width=True):
                    st.session_state.clear_results_confirm = False
                    st.rerun()

            with col3:
                if st.button(
                    "🗑️ Delete",
                    key="confirm_clear",
                    type="secondary",
                    use_container_width=True,
                ):
                    try:
                        deleted_count = 0
                        for file in all_files:
                            try:
                                file.unlink()
                                deleted_count += 1
                            except Exception as e:
                                st.warning(f"Could not delete {file.name}: {e}")

                        # Clear confirmation state
                        st.session_state.clear_results_confirm = False

                        # Show success message
                        st.success(f"✅ Successfully deleted {deleted_count} file(s)!")
                        st.info(
                            "The page will refresh to show the updated results section."
                        )
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
        "📂 No workspace found. Click '🚀 Run Library Generation in Workspace' above to start."
    )

save_params(params)
