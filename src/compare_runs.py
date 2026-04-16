#!/usr/bin/env python3
"""Aggregate per-run comparison artifacts into raw, summary, and plot-ready tables."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


COMPARISON_PRIORITY = [
    "run_id",
    "experiment_name",
    "variant",
    "mode",
    "protocol_version",
    "logging_schema_version",
    "base_model",
    "use_4bit",
    "gradient_checkpointing",
    "seed",
    "rank",
    "alpha",
    "max_steps",
    "effective_batch_size_sequences",
    "max_seq_length",
    "baseline_ppl_all",
    "best_ppl_all",
    "final_ppl_all",
    "train_runtime_s",
    "train_steps_per_second",
    "status",
    "_source_path",
]

CONDITION_GROUP_FIELDS = [
    "protocol_version",
    "logging_schema_version",
    "mode",
    "variant",
    "base_model",
    "use_4bit",
    "gradient_checkpointing",
    "rank",
    "alpha",
    "lora_dropout",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size_sequences",
    "max_steps",
    "max_seq_length",
    "learning_rate",
    "warmup_ratio",
    "weight_decay",
    "logging_steps",
    "eval_steps",
    "save_steps",
]

SUMMARY_NUMERIC_EXCLUDE_FIELDS = {
    "seed",
    "rank",
    "alpha",
    "lora_dropout",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size_sequences",
    "max_steps",
    "max_seq_length",
    "learning_rate",
    "warmup_ratio",
    "weight_decay",
    "logging_steps",
    "eval_steps",
    "save_steps",
}


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


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_csv_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            parsed = float(text)
            if math.isfinite(parsed):
                return parsed
            return None
        return int(text)
    except ValueError:
        return text


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("reports/comparison_row.json")):
        if "reports_index" in path.parts:
            continue
        try:
            row = _read_json(path)
        except Exception as err:
            print(f"[warn] skipping unreadable file: {path} ({err})")
            continue
        if not isinstance(row, dict):
            print(f"[warn] skipping non-dict comparison row: {path}")
            continue
        row["_source_path"] = str(path)
        rows.append(row)
    return rows


def all_fieldnames(rows: list[dict[str, Any]], priority: list[str] | None = None) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for key in priority or []:
        if any(key in row for row in rows):
            names.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                names.append(key)
                seen.add(key)
    return names


def write_csv(path: Path, rows: list[dict[str, Any]], priority: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = all_fieldnames(rows, priority=priority)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _numeric_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _summary_numeric_fields(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    excluded = set(CONDITION_GROUP_FIELDS) | SUMMARY_NUMERIC_EXCLUDE_FIELDS | {
        "run_id",
        "experiment_name",
        "status",
        "_source_path",
    }
    for row in rows:
        for key, value in row.items():
            if key in excluded:
                continue
            if key not in seen and _is_finite_number(value):
                names.append(key)
                seen.add(key)
    return names


def build_condition_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in CONDITION_GROUP_FIELDS)
        groups.setdefault(key, []).append(row)

    numeric_fields = _summary_numeric_fields(rows)
    summaries: list[dict[str, Any]] = []
    for key, grouped_rows in groups.items():
        sample = grouped_rows[0]
        summary: dict[str, Any] = {field: sample.get(field) for field in CONDITION_GROUP_FIELDS}
        seeds = sorted({int(r["seed"]) for r in grouped_rows if isinstance(r.get("seed"), int)})
        statuses = sorted({str(r.get("status")) for r in grouped_rows if r.get("status") is not None})
        experiments = sorted(
            {str(r.get("experiment_name")) for r in grouped_rows if r.get("experiment_name") is not None}
        )
        summary.update(
            {
                "n_runs": len(grouped_rows),
                "num_unique_seeds": len(seeds),
                "seed_values": ",".join(str(seed) for seed in seeds),
                "status_values": ",".join(statuses),
                "experiment_names": ",".join(experiments),
            }
        )
        for field in numeric_fields:
            values = [float(r[field]) for r in grouped_rows if _is_finite_number(r.get(field))]
            if not values:
                continue
            stats = _numeric_summary(values)
            for stat_name, stat_value in stats.items():
                summary[f"{field}_{stat_name}"] = stat_value
        summaries.append(summary)

    return sorted(
        summaries,
        key=lambda row: (
            str(row.get("protocol_version", "")),
            str(row.get("mode", "")),
            str(row.get("variant", "")),
            str(row.get("rank", "")),
            str(row.get("alpha", "")),
        ),
    )


def _artifact_path_from_row(row: dict[str, Any], artifact_name: str) -> Path:
    return Path(str(row["_source_path"])).parent / artifact_name


def build_eval_curve_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_rows: list[dict[str, Any]] = []
    for row in rows:
        path = _artifact_path_from_row(row, "eval_summary.csv")
        if not path.exists():
            print(f"[warn] missing eval summary for run: {path}")
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for eval_row in reader:
                    merged_row = dict(row)
                    merged_row["_eval_summary_path"] = str(path)
                    merged_row.update({key: _coerce_csv_value(value) for key, value in eval_row.items()})
                    merged_rows.append(merged_row)
        except Exception as err:
            print(f"[warn] skipping unreadable eval summary: {path} ({err})")
    return merged_rows


def _flatten_history_row(item: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "history_step": item.get("step"),
        "history_epoch": item.get("epoch"),
        "history_event_type": item.get("event_type"),
        "history_timestamp_utc": item.get("timestamp_utc"),
    }
    metrics = item.get("metrics")
    if isinstance(metrics, str):
        try:
            parsed = ast.literal_eval(metrics)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            metrics = parsed
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            flat[f"metric_{key}"] = value
    return flat


def build_metrics_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_rows: list[dict[str, Any]] = []
    for row in rows:
        path = _artifact_path_from_row(row, "metrics_history.jsonl")
        if not path.exists():
            print(f"[warn] missing metrics history for run: {path}")
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    item = json.loads(text)
                    if not isinstance(item, dict):
                        continue
                    merged_row = dict(row)
                    merged_row["_metrics_history_path"] = str(path)
                    merged_row.update(_flatten_history_row(item))
                    merged_rows.append(merged_row)
        except Exception as err:
            print(f"[warn] skipping unreadable metrics history: {path} ({err})")
    return merged_rows


def write_summary(path: Path, rows: list[dict[str, Any]], condition_rows: list[dict[str, Any]]) -> None:
    summary = {
        "num_rows": len(rows),
        "num_conditions": len(condition_rows),
        "num_completed": sum(1 for r in rows if r.get("status") == "completed"),
        "num_oom": sum(1 for r in rows if r.get("status") == "oom"),
        "num_failed": sum(1 for r in rows if r.get("status") not in {"completed", "oom"}),
        "experiments": sorted({str(r.get("experiment_name")) for r in rows if r.get("experiment_name")}),
        "protocol_versions": sorted({str(r.get("protocol_version")) for r in rows if r.get("protocol_version")}),
        "variants": sorted({str(r.get("variant")) for r in rows if r.get("variant")}),
        "modes": sorted({str(r.get("mode")) for r in rows if r.get("mode")}),
        "condition_group_fields": CONDITION_GROUP_FIELDS,
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
            str(r.get("protocol_version", "")),
            str(r.get("experiment_name", "")),
            str(r.get("run_id", "")),
            str(r.get("_source_path", "")),
        ),
    )
    condition_rows = build_condition_summaries(rows_sorted)
    eval_curve_rows = build_eval_curve_rows(rows_sorted)
    metrics_history_rows = build_metrics_history_rows(rows_sorted)

    write_csv(out_dir / "all_runs_comparison.csv", rows_sorted, priority=COMPARISON_PRIORITY)
    write_json(out_dir / "all_runs_comparison.json", rows_sorted)
    write_csv(out_dir / "condition_summary.csv", condition_rows, priority=CONDITION_GROUP_FIELDS)
    write_json(out_dir / "condition_summary.json", condition_rows)
    write_csv(
        out_dir / "all_eval_curves.csv",
        eval_curve_rows,
        priority=COMPARISON_PRIORITY
        + [
            "_eval_summary_path",
            "step",
            "phase",
            "eval_loss",
            "eval_perplexity",
            "eval_layer_a_loss",
            "eval_layer_a_perplexity",
            "eval_layer_b_loss",
            "eval_layer_b_perplexity",
        ],
    )
    write_json(out_dir / "all_eval_curves.json", eval_curve_rows)
    write_csv(
        out_dir / "all_metrics_history.csv",
        metrics_history_rows,
        priority=COMPARISON_PRIORITY
        + [
            "_metrics_history_path",
            "history_step",
            "history_epoch",
            "history_event_type",
            "history_timestamp_utc",
            "metric_loss",
        ],
    )
    write_json(out_dir / "all_metrics_history.json", metrics_history_rows)
    write_summary(out_dir / "all_runs_summary.json", rows_sorted, condition_rows)
    print(f"[info] wrote {len(rows_sorted)} comparison rows to: {out_dir}")
    print(f"[info] wrote {len(condition_rows)} condition summary rows")
    print(f"[info] wrote {len(eval_curve_rows)} merged eval-curve rows")
    print(f"[info] wrote {len(metrics_history_rows)} merged metrics-history rows")


if __name__ == "__main__":
    main()
