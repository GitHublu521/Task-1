"""Check that students have produced the required result files and filled in summary.md."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

REQUIRED_FILES = [
    "comparison.csv",
    "all_results.json",
    "training_curves.png",
    "compare_activation.png",
    "compare_optimizer.png",
]

SUMMARY_FILE = ROOT / "summary.md"

# Placeholder marker in summary.md — students must replace all TODO comments
PLACEHOLDER = "<!-- TODO"


def check_result_files():
    errors = []
    if not RESULTS_DIR.is_dir():
        return ["results/ directory does not exist. Have you run `python run.py` yet?"]

    for name in REQUIRED_FILES:
        if not (RESULTS_DIR / name).exists():
            errors.append(f"Missing results/{name}")
    return errors


def check_summary():
    if not SUMMARY_FILE.exists():
        return ["summary.md does not exist"]

    content = SUMMARY_FILE.read_text()
    # Split on a standalone "---" line (the bonus divider), not table separators like |---|---|
    required_section = content
    for part in content.split("\n---\n"):
        required_section = part
        break
    remaining = required_section.count(PLACEHOLDER)
    if remaining > 0:
        return [f"summary.md still has {remaining} unfilled TODO placeholder(s) — please write your findings"]
    return []


def main():
    errors = check_result_files() + check_summary()

    if errors:
        print("FAILED:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"PASSED: All {len(REQUIRED_FILES)} required result files found and summary.md is filled in")


if __name__ == "__main__":
    main()
