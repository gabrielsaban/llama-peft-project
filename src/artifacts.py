from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


def json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, int):
            safe[key] = value
        elif isinstance(value, float):
            safe[key] = value if math.isfinite(value) else None
        elif isinstance(value, str):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_text_artifact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")


def write_yaml_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe_metrics(row), sort_keys=True) + "\n")


def write_eval_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "step",
        "phase",
        "eval_loss",
        "eval_perplexity",
        "eval_layer_a_loss",
        "eval_layer_a_perplexity",
        "eval_layer_b_loss",
        "eval_layer_b_perplexity",
        "train_step_time_mean_s_window",
        "train_step_time_p50_s_window",
        "train_step_time_p95_s_window",
        "train_step_time_std_s_window",
        "train_step_time_num_samples_window",
        "eval_peak_vram_allocated_gb",
        "eval_peak_vram_reserved_gb",
        "eval_peak_vram_allocated_gb_since_last_reset",
        "eval_peak_vram_reserved_gb_since_last_reset",
        "eval_peak_vram_allocated_gb_eval_only",
        "eval_peak_vram_reserved_gb_eval_only",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
