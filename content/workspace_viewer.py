import streamlit as st
from pathlib import Path
from typing import List

from src.common.common import page_setup

page_setup()

st.title("Workspace Viewer")

st.markdown(
    """View and manage files in your workspace. Select a workspace to browse its contents,
view file sizes, and download files.
"""
)


def discover_roots() -> List[Path]:
    """Discover available workspace roots"""
    roots: List[Path] = []
    try:
        settings = st.session_state.settings
        location = st.session_state.get("location", "local")
        if settings.get("workspaces_dir") and location == "local":
            default_ws = Path(
                settings["workspaces_dir"],
                "workspaces-" + settings.get("repository-name", ""),
            )
        else:
            default_ws = Path("..")
    except Exception:
        default_ws = Path("..")

    candidates = [
        default_ws,
        default_ws / "default",
        Path("example-data") / "workspaces",
    ]

    for c in candidates:
        if c.exists() and c.is_dir():
            roots.append(c)

    try:
        parent = default_ws.parent
        for p in parent.iterdir():
            if p.is_dir() and p.name.startswith("workspaces"):
                roots.append(p)
    except Exception:
        pass

    seen = set()
    uniq: List[Path] = []
    for p in roots:
        rp = p.resolve()
        if str(rp) not in seen:
            seen.add(str(rp))
            uniq.append(p)
    return uniq


def get_workspace_dirs(base: Path) -> List[Path]:
    """Find all workspace directories"""
    if not base.exists() or not base.is_dir():
        return []
    dirs = [d for d in base.iterdir() if d.is_dir()]
    return sorted(dirs, key=lambda p: p.name)


def format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def build_ascii_tree(
    path: Path,
    prefix: str = "",
    is_last: bool = True,
    max_depth: int = 10,
    current_depth: int = 0,
) -> str:
    """Build ASCII tree structure like the 'tree' command"""
    if current_depth > max_depth:
        return ""

    tree_str = ""

    try:
        items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        items = [item for item in items if not item.name.startswith(".")]

        for i, item in enumerate(items):
            is_last_item = i == len(items) - 1
            current_prefix = "└── " if is_last_item else "├── "

            # Format: icon + name + size/extra info
            if item.is_dir():
                tree_str += f"{prefix}{current_prefix}📁 {item.name}/\n"

                # Recursively add subdirectory contents
                next_prefix = prefix + ("    " if is_last_item else "│   ")
                tree_str += build_ascii_tree(
                    item, next_prefix, is_last_item, max_depth, current_depth + 1
                )
            else:
                size = format_size(item.stat().st_size)
                tree_str += f"{prefix}{current_prefix}📄 {item.name} ({size})\n"

    except PermissionError:
        tree_str += f"{prefix}[Permission Denied]\n"

    return tree_str


# Select workspace root
root_choices = discover_roots()
if root_choices:
    root_strs = [str(p) for p in root_choices]
    selected_root = st.selectbox("Workspaces root directory", root_strs)
    base_dir = Path(selected_root)
else:
    default_root = Path(st.session_state.settings.get("workspaces_dir", ".."))
    base_dir_input = st.text_input("Workspaces root directory", value=str(default_root))
    base_dir = Path(base_dir_input)

workspace_dirs = get_workspace_dirs(base_dir)

if not workspace_dirs:
    st.warning("No workspace directories found under the given root.")
    st.info("Create a workspace first or check the root directory path.")
    st.stop()

workspace_choice = st.selectbox("Select workspace", [p.name for p in workspace_dirs])
selected_workspace = next(
    (p for p in workspace_dirs if p.name == workspace_choice), workspace_dirs[0]
)

st.divider()

# Create tabs for different views
tab1, tab2 = st.tabs(["Tree View", "Files List"])

with tab1:
    st.markdown("**Directory Structure**")

    if not any(selected_workspace.iterdir()):
        st.info("Workspace appears to be empty")
    else:
        # Build and display ASCII tree
        tree_output = f"{selected_workspace.name}/\n"
        tree_output += build_ascii_tree(selected_workspace)

        # Display in code block with monospace font for proper alignment
        st.code(tree_output, language="plaintext")

with tab2:
    st.markdown("**All Files**")

    # Collect all files with metadata
    all_files = []
    for item in selected_workspace.rglob("*"):
        if item.is_file() and not any(part.startswith(".") for part in item.parts):
            rel_path = item.relative_to(selected_workspace)
            size = item.stat().st_size
            all_files.append((str(rel_path), size, item))

    if not all_files:
        st.info("No files found in workspace")
    else:
        # Create header columns
        col1, col2, col3, col4 = st.columns([3, 1, 1, 0.5])
        col1.markdown("**File Path**")
        col2.markdown("**Size**")
        col3.markdown("**Type**")
        col4.markdown("**Action**")

        st.divider()

        for file_path, size, full_path in sorted(all_files, key=lambda x: x[0]):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 0.5])

            # File path with icon
            file_ext = full_path.suffix.lower()
            if file_ext in [".log", ".txt"]:
                icon = "📄"
            elif file_ext in [".osw", ".parquet", ".tsv", ".csv"]:
                icon = "📊"
            elif file_ext in [".mzml", ".raw", ".d"]:
                icon = "🧬"
            elif file_ext in [".json", ".ini", ".yaml", ".yml"]:
                icon = "⚙️"
            else:
                icon = "📦"

            col1.text(f"{icon} {file_path}")
            col2.text(format_size(size))
            col3.text(file_ext or "file")

            # Download button
            try:
                with open(full_path, "rb") as f:
                    file_data = f.read()
                col4.download_button(
                    label="⬇️",
                    data=file_data,
                    file_name=full_path.name,
                    key=f"download_{full_path}",
                    help="Download file",
                )
            except Exception:
                col4.text("❌")

# Summary statistics
with st.expander("Workspace Statistics"):
    total_size = 0
    file_count = 0
    dir_count = 0

    for item in selected_workspace.rglob("*"):
        if not any(part.startswith(".") for part in item.parts):
            if item.is_file():
                file_count += 1
                total_size += item.stat().st_size
            elif item.is_dir():
                dir_count += 1

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Files", file_count)
    col2.metric("Total Directories", dir_count)
    col3.metric("Total Size", format_size(total_size))
    col4.metric("Workspace Path", selected_workspace.name)
