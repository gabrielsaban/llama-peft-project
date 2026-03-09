#!/usr/bin/env python3
"""Aggregate per-run comparison artifacts into a single table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="outputs",
        help="root directory containing run outputs (default: outputs)",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/reports_index",
        help="directory for aggregate outputs (default: outputs/reports_index)",
    )
    return parser.parse_args()


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("reports/comparison_row.json")):
        if "reports_index" in path.parts:
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                row = json.load(f)
        except Exception as err:
            print(f"[warn] skipping unreadable file: {path} ({err})")
            continue
        row["_source_path"] = str(path)
        rows.append(row)
    return rows


def all_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    priority = [
        "run_id",
        "experiment_name",
        "mode",
        "base_model",
        "use_4bit",
        "gradient_checkpointing",
        "seed",
        "max_steps",
        "effective_batch_size_sequences",
        "max_seq_length",
        "baseline_ppl_all",
        "best_ppl_all",
        "train_runtime_s",
        "train_steps_per_second",
        "status",
        "_source_path",
    ]
    for key in priority:
        if any(key in row for row in rows):
            names.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                names.append(key)
                seen.add(key)
    return names


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = all_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    summary = {
        "num_rows": len(rows),
        "num_completed": sum(1 for r in rows if r.get("status") == "completed"),
        "num_oom": sum(1 for r in rows if r.get("status") == "oom"),
        "num_failed": sum(1 for r in rows if r.get("status") not in {"completed", "oom"}),
        "experiments": sorted({str(r.get("experiment_name")) for r in rows if r.get("experiment_name")}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)

    rows = load_rows(root)
    if not rows:
        print(f"[warn] no comparison_row.json files found under: {root}")
        return

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            str(r.get("experiment_name", "")),
            str(r.get("run_id", "")),
            str(r.get("_source_path", "")),
        ),
    )

    write_csv(out_dir / "all_runs_comparison.csv", rows_sorted)
    write_json(out_dir / "all_runs_comparison.json", rows_sorted)
    write_summary(out_dir / "all_runs_summary.json", rows_sorted)
    print(f"[info] wrote {len(rows_sorted)} rows to: {out_dir}")


if __name__ == "__main__":
    main()
