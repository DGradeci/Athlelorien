"""Replace workstation-specific repository paths in public figure metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
PUBLIC_ROOTS = (
    REPO_ROOT / "analysis" / "levy_paper" / "figures" / "source_data",
    REPO_ROOT / "analysis" / "levy_paper" / "figures" / "supplementary_material",
)


def portable_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    root = REPO_ROOT.as_posix()
    if normalized.lower().startswith(root.lower() + "/"):
        return normalized[len(root) + 1 :]
    return value


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return portable_path(value)
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            portable_path(str(key)): sanitize_json_value(item)
            for key, item in value.items()
        }
    return value


def sanitize_json(path: Path) -> bool:
    original = json.loads(path.read_text(encoding="utf-8"))
    sanitized = sanitize_json_value(original)
    if sanitized == original:
        return False
    path.write_text(json.dumps(sanitized, indent=2) + "\n", encoding="utf-8")
    return True


def sanitize_csv(path: Path) -> bool:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    sanitized = [[portable_path(value) for value in row] for row in rows]
    if sanitized == rows:
        return False
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(sanitized)
    return True


def main() -> None:
    changed: list[Path] = []
    for root in PUBLIC_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() == ".json" and sanitize_json(path):
                changed.append(path)
            elif path.suffix.lower() == ".csv" and sanitize_csv(path):
                changed.append(path)
    for path in changed:
        print(path.relative_to(REPO_ROOT).as_posix())
    print(f"sanitized_files={len(changed)}")


if __name__ == "__main__":
    main()
