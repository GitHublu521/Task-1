"""Check that all Python files have valid syntax."""

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# All .py files students might create or modify
REQUIRED = ["dataset.py", "model.py", "train.py", "run.py"]


def main():
    errors = []

    # Check required files exist
    for name in REQUIRED:
        path = ROOT / name
        if not path.exists():
            errors.append(f"MISSING: {name}")
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"SYNTAX ERROR in {name}: {e}")

    # Also check any extra .py files students may have added
    for path in sorted(ROOT.glob("*.py")):
        if path.name in REQUIRED:
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"SYNTAX ERROR in {path.name}: {e}")

    if errors:
        print("FAILED: Python syntax check")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"PASSED: All {len(REQUIRED)} required files exist and have valid syntax")


if __name__ == "__main__":
    main()
