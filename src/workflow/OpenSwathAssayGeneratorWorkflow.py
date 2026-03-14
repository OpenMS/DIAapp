import sys
import shutil
import streamlit as st
from pathlib import Path
from .WorkflowManager import WorkflowManager


class OpenSwathAssayGeneratorWorkflow(WorkflowManager):
    """Workflow wrapper for running OpenSwathAssayGenerator via the workflow system.

    This class provides upload/configure/execution/result fragments compatible
    with the existing `WorkflowManager` and `StreamlitUI` utilities.
    """

    def __init__(self) -> None:
        super().__init__("OpenSwath Assay Generator", st.session_state["workspace"])

    def upload(self) -> None:
        # Upload widgets: Input TraML/TSV file, optional UniMod XML
        t = st.tabs(["Files"])
        with t[0]:
            self.ui.upload_widget(key="traml", file_types="traml,tsv,mrm,pqp,oswpq", name="Input Library (TraML/TSV/etc)")
            self.ui.upload_widget(
                key="unimod", file_types="xml", name="UniMod XML (optional)"
            )

    @st.fragment
    def configure(self) -> None:
        # Select the uploaded files and set simple parameters
        self.ui.select_input_file("traml", multiple=False)
        self.ui.select_input_file("unimod", multiple=False)

        # Basic OpenSwathAssayGenerator parameters
        self.ui.input_widget(
            "output_file", "./openswath_assays.tsv", "Output File"
        )
        self.ui.input_widget("min_transitions", 6, "Min Transitions")
        self.ui.input_widget("max_transitions", 6, "Max Transitions")
        self.ui.input_widget("allowed_fragment_types", "b,y", "Fragment Types")
        self.ui.input_widget("allowed_fragment_charges", "1,2,3,4", "Fragment Charges")
        self.ui.input_widget("threads", 0, "Threads")

    def execution(self) -> bool:
        """Execute OpenSwathAssayGenerator.

        Preferred method: uses config file if available
        Fallback: builds command from individual parameters
        """
        self.logger.log("=" * 80)
        self.logger.log("OPENSWATH ASSAY GENERATOR EXECUTION STARTED")
        self.logger.log("=" * 80)

        # Find the OpenSwathAssayGenerator executable
        openswath_cmd = shutil.which("OpenSwathAssayGenerator")
        if not openswath_cmd:
            self.logger.log("❌ ERROR: 'OpenSwathAssayGenerator' command not found in PATH")
            self.logger.log(
                f"Searched in PATH. Current sys.prefix: {sys.prefix}"
            )
            return False

        self.logger.log(f"Found OpenSwathAssayGenerator at: {openswath_cmd}")

        # Validate required params
        self.logger.log("Loading parameters from JSON...")
        params = self.parameter_manager.get_parameters_from_json()
        self.logger.log(f"Parameters loaded. Keys: {list(params.keys())}")

        # Prepare results directory
        results_dir = Path(self.workflow_dir, "results", "openswath")
        results_dir.mkdir(parents=True, exist_ok=True)
        self.logger.log(f"Results directory ready: {results_dir}")

        # Get input file
        traml_param = params.get("traml")
        if not traml_param:
            self.logger.log("ERROR: No input file selected.")
            return False

        # Resolve input file path using FileManager
        in_files = self.file_manager.get_files(traml_param)
        if not in_files:
            self.logger.log(f"ERROR: Could not find input file: {traml_param}")
            return False
        
        input_file = in_files[0]
        self.logger.log(f"Input file: {input_file}")

        # Get output file
        output_file = params.get(
            "output_file", str(Path(results_dir, "openswath_assays.tsv"))
        )
        self.logger.log(f"Output file: {output_file}")

        # Build command using OpenSwathAssayGenerator executable
        cmd = [
            openswath_cmd,
            "-in", str(input_file),
            "-out", str(output_file),
        ]

        # Add output format only if not "auto" (auto will infer from filename extension)
        output_format = params.get("output_format", "auto")
        if output_format and output_format != "auto":
            cmd += ["-output_type", output_format]
            self.logger.log(f"Output format explicitly set to: {output_format}")
        else:
            self.logger.log("Output format: auto (inferred from filename extension)")

        # Add transition parameters
        if "min_transitions" in params:
            cmd += ["-min_transitions", str(params.get("min_transitions", 6))]
        
        if "max_transitions" in params:
            cmd += ["-max_transitions", str(params.get("max_transitions", 6))]

        # Fragment types
        if "allowed_fragment_types" in params:
            cmd += ["-allowed_fragment_types", str(params.get("allowed_fragment_types", "b,y"))]

        # Fragment charges
        if "allowed_fragment_charges" in params:
            cmd += ["-allowed_fragment_charges", str(params.get("allowed_fragment_charges", "1,2,3,4"))]

        # Optional: MZ thresholds
        if "precursor_mz_threshold" in params:
            cmd += ["-precursor_mz_threshold", str(params.get("precursor_mz_threshold"))]
        
        if "product_mz_threshold" in params:
            cmd += ["-product_mz_threshold", str(params.get("product_mz_threshold"))]

        # Optional: MZ limits
        if "precursor_lower_mz_limit" in params:
            cmd += ["-precursor_lower_mz_limit", str(params.get("precursor_lower_mz_limit"))]
        
        if "precursor_upper_mz_limit" in params:
            cmd += ["-precursor_upper_mz_limit", str(params.get("precursor_upper_mz_limit"))]
        
        if "product_lower_mz_limit" in params:
            cmd += ["-product_lower_mz_limit", str(params.get("product_lower_mz_limit"))]
        
        if "product_upper_mz_limit" in params:
            cmd += ["-product_upper_mz_limit", str(params.get("product_upper_mz_limit"))]

        # Optional: UniMod file
        if "unimod_file" in params and params.get("unimod_file"):
            cmd += ["-unimod_file", str(params.get("unimod_file"))]

        # Optional: IPF settings
        if params.get("enable_ipf"):
            cmd += ["-enable_ipf"]

        # Optional: SWATH windows file
        if "swath_windows_file" in params and params.get("swath_windows_file"):
            cmd += ["-swath_windows_file", str(params.get("swath_windows_file"))]

        # Optional: Detection losses
        if params.get("enable_detection_specific_losses"):
            cmd += ["-enable_detection_specific_losses"]
        
        if params.get("enable_detection_unspecific_losses"):
            cmd += ["-enable_detection_unspecific_losses"]

        # Threads (may be None; coerce to int with safe fallback to 1)
        threads_raw = params.get("threads", 1)
        try:
            threads = int(threads_raw) if threads_raw is not None else 1
        except (TypeError, ValueError):
            threads = 1
        if threads > 0:
            cmd += ["-threads", str(threads)]

        # Run command via executor
        cmd_str = " ".join(cmd)
        self.logger.log(f"Full command: {cmd_str}")
        self.logger.log("Spawning subprocess...")
        success = self.executor.run_command(cmd)

        if not success:
            self.logger.log("❌ OpenSwathAssayGenerator execution failed (non-zero exit code).")
            return False

        self.logger.log("✅ OpenSwathAssayGenerator execution finished successfully.")
        return True

    def results(self) -> None:
        # Display output file location and simple download if present
        output_candidates = list(
            Path(self.workflow_dir, "results", "openswath").glob("*.tsv")
        )
        if output_candidates:
            out = output_candidates[0]
            st.markdown(f"**Output:** {out}")
            with open(out, "rb") as f:
                st.download_button("⬇️ Download assays", data=f, file_name=out.name)
        else:
            st.info(
                "No assay output found yet. Run the workflow to generate outputs."
            )
