import os
from pathlib import Path

# File extension types to include as readable code/text
ALLOWED_EXTENSIONS = {
    ".py", ".pyw", ".txt", ".md", ".json", ".yaml", 
    ".yml", ".toml", ".ini", ".cfg", ".env.example", ".sh", ".sql"
}

# Directories and files to automatically exclude
EXCLUDED_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", 
    ".idea", ".vscode", "build", "dist", "node_modules", ".pytest_cache"
}

EXCLUDED_FILES = {
    "export_project.py",  # Exclude this script itself
    "project_context.txt", # Exclude the output file if placed in root
    ".DS_Store", "poetry.lock", "Pipfile.lock"
}

def is_excluded(path: Path, root_dir: Path) -> bool:
    """Check if a file or folder should be skipped."""
    rel_parts = path.relative_to(root_dir).parts
    if any(part in EXCLUDED_DIRS for part in rel_parts):
        return True
    if path.name in EXCLUDED_FILES:
        return True
    return False

def generate_tree(dir_path: Path, prefix: str = "") -> str:
    """Generates an ASCII representation of the file directory structure."""
    tree_str = ""
    contents = sorted(
        [p for p in dir_path.iterdir() if not is_excluded(p, dir_path.parent)],
        key=lambda x: (not x.is_dir(), x.name.lower())
    )

    for index, path in enumerate(contents):
        is_last = (index == len(contents) - 1)
        connector = "└── " if is_last else "├── "
        
        tree_str += f"{prefix}{connector}{path.name}\n"

        if path.is_dir():
            extension_prefix = "    " if is_last else "│   "
            tree_str += generate_tree(path, prefix + extension_prefix)

    return tree_str

def export_codebase(project_root: str, output_file: str):
    """Parses project tree and outputs file contents into a single text file."""
    root = Path(project_root).resolve()
    out_path = Path(output_file).resolve()

    with open(out_path, "w", encoding="utf-8") as out:
        # 1. Output Header & Architecture Tree
        out.write("=================================================================\n")
        out.write(f"PROJECT ARCHITECTURE: {root.name}\n")
        out.write("=================================================================\n\n")
        out.write(f"{root.name}/\n")
        out.write(generate_tree(root))
        out.write("\n\n")

        # 2. Output File Contents
        out.write("=================================================================\n")
        out.write("FILE CONTENTS\n")
        out.write("=================================================================\n\n")

        for path in sorted(root.rglob("*")):
            if path.is_file() and not is_excluded(path, root):
                if path.suffix in ALLOWED_EXTENSIONS or path.name in {"Dockerfile", "Makefile"}:
                    try:
                        relative_path = path.relative_to(root)
                        out.write(f"--- START FILE: {relative_path} ---\n")
                        with open(path, "r", encoding="utf-8", errors="replace") as f:
                            out.write(f.read())
                        out.write(f"\n--- END FILE: {relative_path} ---\n\n")
                    except Exception as e:
                        print(f"Skipping {path} due to error: {e}")

    print(f"Project successfully exported to: {out_path}")

if __name__ == "__main__":
    # Uses current working directory by default
    current_dir = os.getcwd()
    output_filename = "project_context.txt"
    
    export_codebase(current_dir, output_filename)