#!/usr/bin/env python3
"""Generate the README trade-off figure from the frozen condition summary."""

from __future__ import annotations

import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/reports_index/protocol_v3/condition_summary.csv"
OUTPUT = ROOT / "docs/assets/protocol-v3-tradeoff.png"
COLOURS = {("lora", 16): "#2563EB", ("qlora", 16): "#DC2626",
           ("qlora", 32): "#D97706", ("qlora", 64): "#7C3AED"}


def load_rows() -> list[dict[str, float | str | int]]:
    with SOURCE.open(newline="", encoding="utf-8-sig") as stream:
        rows = []
        for raw in csv.DictReader(stream):
            variant, rank = raw["variant"], int(raw["rank"])
            rows.append({"variant": variant, "rank": rank,
                         "label": f"{'LoRA' if variant == 'lora' else 'QLoRA'} r{rank}",
                         "perplexity": float(raw["best_ppl_all_mean"]),
                         "vram": float(raw["train_peak_vram_allocated_gb_mean"]),
                         "step_time": float(raw["step_time_mean_s_mean"])})
    return rows


def main() -> None:
    rows = load_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    offsets = {
        "vram": {("lora", 16): (8, 8), ("qlora", 16): (8, 8),
                 ("qlora", 32): (8, -16), ("qlora", 64): (8, 8)},
        "step_time": {("lora", 16): (8, 8), ("qlora", 16): (-8, 9),
                      ("qlora", 32): (-8, -16), ("qlora", 64): (-8, 8)},
    }
    panels = ((axes[0], "vram", "Peak allocated training VRAM (GB)", "Memory trade-off"),
              (axes[1], "step_time", "Mean training step time (s)", "Runtime trade-off"))
    for axis, x_key, x_label, title in panels:
        axis.grid(True, color="#E5E7EB", linewidth=0.8, zorder=0)
        for row in rows:
            key = (str(row["variant"]), int(row["rank"]))
            axis.scatter(row[x_key], row["perplexity"], s=90, color=COLOURS[key],
                         edgecolor="white", linewidth=1.2, zorder=3)
            x_offset, y_offset = offsets[x_key][key]
            axis.annotate(str(row["label"]), (row[x_key], row["perplexity"]),
                          xytext=(x_offset, y_offset), textcoords="offset points",
                          ha="left" if x_offset > 0 else "right", fontsize=9,
                          color="#111827", weight="semibold")
        axis.set_title(title, pad=12)
        axis.set_xlabel(x_label, labelpad=8)
        axis.set_ylim(7.245, 7.43)
    axes[0].set_ylabel("Best overall validation perplexity (lower is better)", labelpad=9)
    figure.suptitle("Quality-resource trade-offs under protocol_v3",
                    fontsize=15, fontweight="bold", y=1.01)
    figure.text(0.5, -0.01, "Condition means across seeds 42, 43, and 44 on NVIDIA L40S",
                ha="center", color="#4B5563", fontsize=9)
    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
