from __future__ import annotations

import json
from pathlib import Path

BAD_VALUES = {"null", "none", "n/a", "na", "not found", "unknown"}


def find_bad_values(obj, prefix=""):
    issues = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            child_prefix = f"{prefix}.{k}" if prefix else k
            issues.extend(find_bad_values(v, child_prefix))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            child_prefix = f"{prefix}[{i}]"
            issues.extend(find_bad_values(v, child_prefix))
    elif isinstance(obj, str):
        if obj.strip().lower() in BAD_VALUES:
            issues.append((prefix, obj))

    return issues


def main():
    gold_dir = Path("assets/dataset/gold_labels")
    files = sorted(gold_dir.glob("*.json"))

    total_issues = 0
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        issues = find_bad_values(data)
        if issues:
            print(f"\n{path.name}")
            for loc, value in issues:
                print(f"  {loc}: {value!r}")
            total_issues += len(issues)

    if total_issues == 0:
        print("No null-like string issues found.")
    else:
        print(f"\nTotal issues found: {total_issues}")


if __name__ == "__main__":
    main()
