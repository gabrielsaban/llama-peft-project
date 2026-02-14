#!/usr/bin/env python3
"""Create train/val splits from corpus_manifest with stratification.

Stratifies by:
1) layer (A/B)
2) token-length quartile within each layer
"""

import argparse
import json
import math
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stratified train/val splits from corpus_manifest.json"
    )
    parser.add_argument(
        "--manifest",
        default="data/domain_corpus/corpus_manifest.json",
        help="Path to corpus manifest JSON",
    )
    parser.add_argument(
        "--output",
        default="data/splits/intrinsic_splits.json",
        help="Output split JSON path",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio per stratum (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def assign_quartiles(entries: list[dict]) -> list[dict]:
    """Assign quartile labels 1..4 by token-rank bins (equal-count buckets)."""
    sorted_entries = sorted(entries, key=lambda x: (x["tokens"], x["filename"]))
    n = len(sorted_entries)
    if n == 0:
        return []

    out = []
    for i, item in enumerate(sorted_entries):
        quartile = (i * 4) // n + 1
        annotated = dict(item)
        annotated["quartile"] = quartile
        out.append(annotated)
    return out


def allocate_val_counts(group_sizes: list[int], total_val: int) -> list[int]:
    """Allocate exact per-layer val count across quartile groups."""
    total = sum(group_sizes)
    if total == 0:
        return [0 for _ in group_sizes]

    raw = [n * total_val / total for n in group_sizes]
    alloc = [math.floor(x) for x in raw]

    total_alloc = sum(alloc)
    remainder = total_val - total_alloc

    # Largest-remainder method with capacity checks.
    order = sorted(
        range(len(group_sizes)),
        key=lambda i: (raw[i] - alloc[i], group_sizes[i]),
        reverse=True,
    )
    i = 0
    while remainder > 0 and i < len(order):
        idx = order[i]
        if alloc[idx] < group_sizes[idx]:
            alloc[idx] += 1
            remainder -= 1
        i += 1

    # If still short due capacity limits, fill any available slots.
    if remainder > 0:
        for idx in range(len(group_sizes)):
            if remainder == 0:
                break
            room = group_sizes[idx] - alloc[idx]
            if room > 0:
                step = min(room, remainder)
                alloc[idx] += step
                remainder -= step

    return alloc


def main() -> None:
    args = parse_args()
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError(f"--val-ratio must be in (0,1), got {args.val_ratio}")

    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    layer_docs: dict[str, list[dict]] = {
        "layer_a": manifest["layer_a"]["selected"],
        "layer_b": manifest["layer_b"]["selected"],
    }

    rng = random.Random(args.seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    allocation_debug: dict[str, dict[str, list[int] | int]] = {}

    for layer, rows in layer_docs.items():
        with_quartiles = assign_quartiles(rows)
        quartile_groups = [[r for r in with_quartiles if r["quartile"] == q] for q in range(1, 5)]
        group_sizes = [len(g) for g in quartile_groups]
        total = sum(group_sizes)
        target_val = int(round(total * args.val_ratio))
        if total > 1:
            target_val = max(1, min(total - 1, target_val))
        else:
            target_val = 0
        val_allocs = allocate_val_counts(group_sizes, target_val)
        allocation_debug[layer] = {
            "group_sizes": group_sizes,
            "target_val": target_val,
            "val_allocs": val_allocs,
        }

        for q in range(1, 5):
            group = quartile_groups[q - 1]
            if not group:
                continue
            rng.shuffle(group)
            n_val = val_allocs[q - 1]
            val_group = group[:n_val]
            train_group = group[n_val:]

            for item in train_group:
                train_rows.append(
                    {
                        "layer": layer,
                        "filename": item["filename"],
                        "relpath": f"{layer}/{item['filename']}",
                        "tokens": item["tokens"],
                        "quartile": item["quartile"],
                        "split": "train",
                    }
                )
            for item in val_group:
                val_rows.append(
                    {
                        "layer": layer,
                        "filename": item["filename"],
                        "relpath": f"{layer}/{item['filename']}",
                        "tokens": item["tokens"],
                        "quartile": item["quartile"],
                        "split": "val",
                    }
                )

    all_rows = train_rows + val_rows

    by_relpath: dict[str, str] = {}
    for row in all_rows:
        relpath = row["relpath"]
        if relpath in by_relpath:
            raise ValueError(f"Duplicate relpath in split assignment: {relpath}")
        by_relpath[relpath] = row["split"]

    counts: dict[str, dict[str, int]] = {
        "layer_a": {"train": 0, "val": 0, "total": 0},
        "layer_b": {"train": 0, "val": 0, "total": 0},
    }
    counts_by_quartile: dict[str, dict[str, dict[str, int]]] = {
        "layer_a": {str(q): {"train": 0, "val": 0, "total": 0} for q in range(1, 5)},
        "layer_b": {str(q): {"train": 0, "val": 0, "total": 0} for q in range(1, 5)},
    }
    tokens_by_split: dict[str, dict[str, int]] = {
        "layer_a": {"train": 0, "val": 0, "total": 0},
        "layer_b": {"train": 0, "val": 0, "total": 0},
        "overall": {"train": 0, "val": 0, "total": 0},
    }

    for row in all_rows:
        layer = row["layer"]
        split = row["split"]
        q = str(row["quartile"])
        counts[layer][split] += 1
        counts[layer]["total"] += 1
        counts_by_quartile[layer][q][split] += 1
        counts_by_quartile[layer][q]["total"] += 1
        tokens_by_split[layer][split] += row["tokens"]
        tokens_by_split[layer]["total"] += row["tokens"]
        tokens_by_split["overall"][split] += row["tokens"]
        tokens_by_split["overall"]["total"] += row["tokens"]

    payload = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "stratification": ["layer", "token_quartile"],
        "quartile_definition": "token-rank bins (equal-count buckets within each layer)",
        "manifest": str(manifest_path),
        "counts": counts,
        "counts_by_quartile": counts_by_quartile,
        "tokens_by_split": tokens_by_split,
        "allocation_debug": allocation_debug,
        "splits": {"train": train_rows, "val": val_rows},
        "by_relpath": by_relpath,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[saved] {output_path}")
    print(f"[summary] layer_a train={counts['layer_a']['train']} val={counts['layer_a']['val']}")
    print(f"[summary] layer_b train={counts['layer_b']['train']} val={counts['layer_b']['val']}")
    print(
        "[summary] tokens train="
        f"{tokens_by_split['overall']['train']} val={tokens_by_split['overall']['val']}"
    )


if __name__ == "__main__":
    main()
