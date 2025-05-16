from pathlib import Path

def build_directory_tree(
    root: Path,
    prefix: str = "",
    excluded_dirs: set[str] = set(),
    excluded_extensions: set[str] = set()
) -> list[str]:

    lines = []
    entries = sorted(root.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    entries_dirs = [e for e in entries if e.is_dir() and e.name not in excluded_dirs]
    entries_files = [
        e for e in entries
        if e.is_file() and e.suffix not in excluded_extensions
    ]

    for i, entry in enumerate(entries_dirs + entries_files):
        connector = "└── " if i == len(entries_dirs + entries_files) - 1 else "├── "
        line = prefix + connector + entry.name
        lines.append(line)
        if entry.is_dir():
            extension = "    " if i == len(entries_dirs + entries_files) - 1 else "│   "
            lines.extend(build_directory_tree(entry, prefix + extension, excluded_dirs, excluded_extensions))
    return lines

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    output_file = project_root / "directory_map.txt"

    excluded_dirs = {"__pycache__", ".git", "venv", ".idea", ".conda", ".vscode"}
    excluded_extensions = {".parquet", ".log", ".tmp", ".ipynb"}  # add more as needed

    tree_lines = [project_root.name + "/"]
    tree_lines += build_directory_tree(
        project_root,
        excluded_dirs=excluded_dirs,
        excluded_extensions=excluded_extensions
    )

    for line in tree_lines:
        print(line)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tree_lines))

    print(f"\nDirectory map written to: {output_file}")

