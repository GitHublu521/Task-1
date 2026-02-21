"""Check that students have produced the bonus result files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BONUS_FILES = [
    "lr_curve.png",
]


def find_results_dirs():
    """Collect results/ and any results_*/ directories."""
    dirs = []
    default = ROOT / "results"
    if default.is_dir():
        dirs.append(default)
    for d in sorted(ROOT.glob("results_*")):
        if d.is_dir():
            dirs.append(d)
    return dirs


def main():
    dirs = find_results_dirs()
    if not dirs:
        print("FAILED: No results directory found")
        sys.exit(1)

    missing = []
    for name in BONUS_FILES:
        found = any((d / name).exists() for d in dirs)
        if not found:
            missing.append(name)

    if missing:
        print("FAILED: Missing bonus result files")
        for name in missing:
            print(f"  - {name}")
        sys.exit(1)

    print(f"PASSED: All {len(BONUS_FILES)} bonus result files found")


if __name__ == "__main__":
    main()
