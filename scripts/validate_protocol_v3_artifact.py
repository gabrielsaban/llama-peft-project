#!/usr/bin/env python3
"""Validate the compact, reportable protocol_v3 research artefact."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*protocol_v3*.yaml"))
EXPERIMENTS = ROOT / "outputs" / "experiments" / "protocol_v3"
REPORTS = ROOT / "outputs" / "reports_index" / "protocol_v3"
EXPECTED = {
    ("lora", 16): (7.267048809916087, 15.125940799713135, 1.9927118843427991),
    ("qlora", 16): (7.412493751322766, 7.437320232391357, 4.957653210964982),
    ("qlora", 32): (7.383221464997856, 7.589663982391357, 4.983221184616171),
    ("qlora", 64): (7.383747490805507, 7.894351482391357, 4.987969786495641),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"[invalid] {message}")


def main() -> None:
    require(len(CONFIGS) == 12, f"expected 12 protocol_v3 configs, found {len(CONFIGS)}")
    require((ROOT / "paper/gabriel-saban-dissertation.pdf").is_file(), "dissertation PDF is missing")
    require((ROOT / "docs/assets/protocol-v3-tradeoff.png").is_file(), "trade-off figure is missing")
    run_reports = sorted(EXPERIMENTS.glob("*/reports/comparison_row.json"))
    require(len(run_reports) == 12, f"expected 12 comparison rows, found {len(run_reports)}")

    for path in sorted(REPORTS.glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            json.load(stream)

    with (REPORTS / "condition_summary.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 4, f"expected 4 conditions, found {len(rows)}")

    observed = set()
    for row in rows:
        key = (row["variant"], int(row["rank"]))
        observed.add(key)
        require(key in EXPECTED, f"unexpected condition {key}")
        require(int(row["n_runs"]) == 3, f"{key} does not contain 3 runs")
        require(row["seed_values"] == "42,43,44", f"{key} has unexpected seeds")
        require(row["status_values"] == "completed", f"{key} is not complete")
        actual = (
            float(row["best_ppl_all_mean"]),
            float(row["train_peak_vram_allocated_gb_mean"]),
            float(row["step_time_mean_s_mean"]),
        )
        for value, expected in zip(actual, EXPECTED[key], strict=True):
            require(math.isclose(value, expected, rel_tol=0, abs_tol=1e-10),
                    f"{key} aggregate value does not match the frozen result")
        for field in (
            "num_inf_loss_events_mean", "num_nan_loss_events_mean",
            "num_nonfinite_events_mean", "num_nonfinite_grad_norm_events_mean",
            "num_oom_events_mean", "optimizer_reset_count_mean",
        ):
            require(float(row[field]) == 0.0, f"{key} records a {field} event")
    require(observed == set(EXPECTED), "condition matrix is incomplete")
    print("[valid] protocol_v3: 12 runs, 4 complete conditions, frozen aggregates verified")


if __name__ == "__main__":
    main()
