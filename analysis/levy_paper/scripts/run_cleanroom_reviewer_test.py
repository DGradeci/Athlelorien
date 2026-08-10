from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[3]
LEVY_DIR = ROOT / "analysis" / "levy_paper"
REPORT_DIR = LEVY_DIR / "reports"
NOTEBOOK_DIR = Path("analysis") / "levy_paper" / "notebooks_publication"
CACHE_MANIFEST = LEVY_DIR / "data" / "cache_bundle_manifest.csv"


def _git_visible_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _copy_visible_files(root: Path, cleanroom: Path) -> dict[str, int]:
    copied = 0
    missing = 0
    for rel_path in _git_visible_files(root):
        src = root / rel_path
        if not src.is_file():
            missing += 1
            continue
        dest = cleanroom / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    return {"copied_files": copied, "missing_deleted_paths": missing}


def _copy_cache_bundle(root: Path, cleanroom: Path) -> list[dict]:
    if not CACHE_MANIFEST.exists():
        return []
    manifest = pd.read_csv(CACHE_MANIFEST)
    rows: list[dict] = []
    for rel_path in manifest["relative_path"].dropna().astype(str):
        src = root / rel_path
        dest = cleanroom / rel_path
        row = {
            "relative_path": rel_path,
            "source_exists": bool(src.exists()),
            "copied": False,
            "bytes": None,
        }
        if src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            row["copied"] = True
            row["bytes"] = int(src.stat().st_size)
        rows.append(row)
    return rows


def _run_reproducibility_check(cleanroom: Path) -> dict:
    script = cleanroom / "analysis" / "levy_paper" / "scripts" / "check_publication_reproducibility.py"
    result = subprocess.run(
        [sys.executable, str(script.relative_to(cleanroom))],
        cwd=cleanroom,
        text=True,
        capture_output=True,
    )
    payload = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "summary": None,
    }
    try:
        payload["summary"] = json.loads(result.stdout)
    except Exception:
        payload["summary"] = None
    return payload


def _patch_notebook_flags(cleanroom: Path, *, data_mode: str) -> list[dict]:
    rows: list[dict] = []
    nb_dir = cleanroom / NOTEBOOK_DIR
    for nb_path in sorted(nb_dir.glob("*.ipynb")):
        nb = nbformat.read(nb_path, as_version=4)
        replacements = 0
        for cell in nb.cells:
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", "")
            new_src = src
            new_src = new_src.replace('DATA_MODE = "auto"', f'DATA_MODE = "{data_mode}"')
            new_src = new_src.replace("REBUILD_CACHE_FROM_AWS = True", "REBUILD_CACHE_FROM_AWS = False")
            if new_src != src:
                cell["source"] = new_src
                replacements += 1
        nbformat.write(nb, nb_path)
        rows.append({
            "notebook": str(nb_path.relative_to(cleanroom)).replace("\\", "/"),
            "flag_replacements": replacements,
            "joint_renderer_already_covered": False,
        })
    return rows


def _execute_publication_notebooks(cleanroom: Path, timeout: int) -> list[dict]:
    rows: list[dict] = []
    nb_dir = cleanroom / NOTEBOOK_DIR
    for nb_path in sorted(nb_dir.glob("*.ipynb")):
        started = time.perf_counter()
        client = None
        row = {
            "notebook": str(nb_path.relative_to(cleanroom)).replace("\\", "/"),
            "ok": False,
            "seconds": None,
            "image_outputs": 0,
            "expected_image_outputs": 0,
            "figure_output_ok": False,
            "error": "",
        }
        try:
            nb = nbformat.read(nb_path, as_version=4)
            client = NotebookClient(
                nb,
                timeout=timeout,
                kernel_name="python3",
                resources={"metadata": {"path": str(cleanroom)}},
            )
            client.execute()
            image_outputs = sum(
                "image/png" in output.get("data", {})
                for cell in nb.cells
                for output in cell.get("outputs", [])
            )
            expected_image_outputs = 0
            if not nb_path.name.startswith("00_"):
                expected_image_outputs = 2 if nb_path.name.startswith("06_") else 1
            row["image_outputs"] = image_outputs
            row["expected_image_outputs"] = expected_image_outputs
            row["figure_output_ok"] = image_outputs >= expected_image_outputs
            row["ok"] = row["figure_output_ok"]
            if not row["ok"]:
                row["error"] = (
                    f"Expected at least {expected_image_outputs} image/png outputs; "
                    f"found {image_outputs}."
                )
        except Exception as exc:
            row["error"] = f"{exc.__class__.__name__}: {exc}"[:2000]
        finally:
            if client is not None:
                try:
                    client._cleanup_kernel()
                except Exception:
                    pass
            row["seconds"] = round(time.perf_counter() - started, 2)
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clean-room smoke tests for publication notebooks."
    )
    parser.add_argument("--keep-cleanroom", action="store_true")
    parser.add_argument("--execute-notebooks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mode",
        choices=["reviewer", "local-cache"],
        default="reviewer",
        help="reviewer uses Git-visible files only; local-cache also copies ignored processed/model caches.",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    cleanroom = Path(tempfile.mkdtemp(prefix="athlelorien_cleanroom_"))
    generated_at = datetime.now().isoformat(timespec="seconds")

    copy_summary = _copy_visible_files(ROOT, cleanroom)
    cache_copy_rows = _copy_cache_bundle(ROOT, cleanroom) if args.mode == "local-cache" else []
    flag_patch_rows = _patch_notebook_flags(
        cleanroom,
        data_mode="cache" if args.mode == "local-cache" else "reviewer",
    )
    forbidden = [
        ".env",
        ".venv",
    ]
    forbidden_status = {
        path: (cleanroom / path).exists()
        for path in forbidden
    }
    heavy_extensions = {".parquet", ".mp4", ".mov", ".avi", ".zip", ".h5", ".db"}
    allowed_heavy_files = {
        row["relative_path"].replace("\\", "/")
        for row in cache_copy_rows
        if row["copied"]
    }
    leaked_heavy_files = [
        str(path.relative_to(cleanroom)).replace("\\", "/")
        for path in cleanroom.rglob("*")
        if path.is_file()
        and path.suffix.lower() in heavy_extensions
        and str(path.relative_to(cleanroom)).replace("\\", "/") not in allowed_heavy_files
    ]

    check = _run_reproducibility_check(cleanroom)
    notebook_rows = _execute_publication_notebooks(cleanroom, args.timeout) if args.execute_notebooks else []
    notebook_csv = args.report_dir / f"cleanroom_{args.mode}_notebook_execution.csv"
    _write_csv(notebook_csv, notebook_rows)

    summary = {
        "generated_at": generated_at,
        "mode": args.mode,
        "cleanroom": str(cleanroom),
        "copy_summary": copy_summary,
        "cache_files_expected": len(cache_copy_rows),
        "cache_files_copied": int(sum(bool(row["copied"]) for row in cache_copy_rows)),
        "cache_files_missing": int(sum(not bool(row["copied"]) for row in cache_copy_rows)),
        "notebook_flag_patch_rows": flag_patch_rows,
        "forbidden_status": forbidden_status,
        "leaked_heavy_files": leaked_heavy_files,
        "reproducibility_check_returncode": check["returncode"],
        "reproducibility_check_summary": check["summary"],
        "notebooks_executed": bool(args.execute_notebooks),
        "all_publication_notebooks_executed": all(row["ok"] for row in notebook_rows) if notebook_rows else None,
        "notebook_execution_csv": str(notebook_csv),
        "kept_cleanroom": bool(args.keep_cleanroom),
    }

    json_path = args.report_dir / f"cleanroom_{args.mode}_check.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not args.keep_cleanroom:
        shutil.rmtree(cleanroom, ignore_errors=True)

    ok = (
        check["returncode"] == 0
        and bool(check["summary"] and check["summary"].get("github_only_reviewer_ready"))
        and (args.mode != "local-cache" or bool(check["summary"].get("local_cache_regeneration_ready")))
        and (args.mode != "local-cache" or all(row["copied"] for row in cache_copy_rows))
        and not any(forbidden_status.values())
        and not leaked_heavy_files
        and (not args.execute_notebooks or all(row["ok"] for row in notebook_rows))
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
