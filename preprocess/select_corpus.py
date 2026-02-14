#!/usr/bin/env python3
"""select layer A + layer B documents into a fixed-budget training corpus."""

import argparse
import json
import random
import subprocess
import shutil
from datetime import datetime, timezone
from pathlib import Path


def get_git_hash() -> str:
    """return short git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def as_repo_relative(path: Path, repo_root: Path) -> str:
    """Return a portable path relative to repo root when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(
        description="select documents into a fixed-token-budget training corpus"
    )
    parser.add_argument(
        "--pilot-json",
        default="../data/domain_corpus/pilot_token_stats.json",
        help="path to pilot token stats (layer A)",
    )
    parser.add_argument(
        "--layer-b-dir",
        default="../data/domain_corpus/layer_b_processed",
        help="directory containing layer B .txt files",
    )
    parser.add_argument(
        "--layer-b-json",
        default="../data/domain_corpus/layer_b_token_stats.json",
        help="path to layer B token stats",
    )
    parser.add_argument(
        "--reflow-dir",
        default="../data/domain_corpus/raw_txt_reflow",
        help="directory containing post-reflow layer A .txt files",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/domain_corpus/corpus_final",
        help="directory to populate with selected files (symlinks)",
    )
    parser.add_argument(
        "--manifest",
        default="../data/domain_corpus/corpus_manifest.json",
        help="path for output manifest",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=1_800_000,
        help="total corpus token budget (layer A + layer B)",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=5000,
        help="allowed overshoot above target (tokens)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for deterministic shuffle",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="minimum per-doc tokens to be eligible (0 = no filter)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="copy files instead of symlinking",
    )
    args = parser.parse_args()

    # load layer B stats
    layer_b_json = Path(args.layer_b_json)
    if not layer_b_json.exists():
        raise FileNotFoundError(f"layer B stats not found: {layer_b_json}")
    with open(layer_b_json) as f:
        lb_data = json.load(f)
    layer_b_tokens = lb_data["overall"]["total_tokens"]
    layer_b_files = lb_data["per_file"]

    # load layer A pilot stats
    pilot_json = Path(args.pilot_json)
    if not pilot_json.exists():
        raise FileNotFoundError(f"pilot stats not found: {pilot_json}")
    with open(pilot_json) as f:
        pilot_data = json.load(f)

    all_docs = pilot_data["per_file"]
    print(f"[info] layer A candidates: {len(all_docs)}")
    print(f"[info] layer B fixed: {layer_b_tokens:,} tokens ({len(layer_b_files)} files)")

    # filter eligible docs
    if args.min_tokens > 0:
        eligible = [d for d in all_docs if d["tokens"] >= args.min_tokens]
        print(f"[info] eligible after min_tokens={args.min_tokens}: {len(eligible)}")
    else:
        eligible = list(all_docs)

    # layer A budget = total target minus layer B
    layer_a_budget = args.target_tokens - layer_b_tokens
    print(f"[info] target total: {args.target_tokens:,}")
    print(f"[info] layer A budget: {layer_a_budget:,} (target - layer B)")

    # deterministic shuffle
    rng = random.Random(args.seed)
    rng.shuffle(eligible)

    # greedy selection
    selected = []
    cumulative = 0
    skipped = []

    for doc in eligible:
        tok = doc["tokens"]
        if cumulative + tok <= layer_a_budget:
            selected.append(doc)
            cumulative += tok
        elif cumulative + tok <= layer_a_budget + args.tolerance:
            # within tolerance band — accept
            selected.append(doc)
            cumulative += tok
            break
        else:
            # too large, skip and try next (might find a smaller one that fits)
            skipped.append(doc)
            continue

        # stop if we've hit the budget
        if cumulative >= layer_a_budget:
            break

    # if still under budget after first pass, try skipped docs smallest-first
    if cumulative < layer_a_budget:
        skipped.sort(key=lambda d: d["tokens"])
        for doc in skipped:
            tok = doc["tokens"]
            if cumulative + tok <= layer_a_budget + args.tolerance:
                selected.append(doc)
                cumulative += tok
                if cumulative >= layer_a_budget:
                    break

    total_with_lb = cumulative + layer_b_tokens

    print(f"\n[result] selected {len(selected)} layer A docs")
    print(f"[result] layer A tokens: {cumulative:,}")
    print(f"[result] layer B tokens: {layer_b_tokens:,}")
    print(f"[result] total corpus:   {total_with_lb:,}")
    print(f"[result] vs target:      {total_with_lb - args.target_tokens:+,}")

    # build output directory
    output_dir = Path(args.output_dir)
    reflow_dir = Path(args.reflow_dir).resolve()
    layer_b_dir = Path(args.layer_b_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]

    # clear and recreate
    if output_dir.exists():
        shutil.rmtree(output_dir)

    layer_a_out = output_dir / "layer_a"
    layer_b_out = output_dir / "layer_b"
    layer_a_out.mkdir(parents=True, exist_ok=True)
    layer_b_out.mkdir(parents=True, exist_ok=True)

    link_fn = shutil.copy2 if args.copy else None

    # symlink/copy layer A
    for doc in selected:
        src = reflow_dir / doc["filename"]
        dst = layer_a_out / doc["filename"]
        if args.copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src)

    # symlink/copy layer B
    for lb_file in layer_b_files:
        src = layer_b_dir / lb_file["filename"]
        dst = layer_b_out / lb_file["filename"]
        if src.exists():
            if args.copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src)

    # write manifest
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_hash(),
        "seed": args.seed,
        "target_tokens": args.target_tokens,
        "tolerance": args.tolerance,
        "layer_a": {
            "files": len(selected),
            "tokens": cumulative,
            "source_dir": as_repo_relative(reflow_dir, repo_root),
            "selected": sorted(
                [{"filename": d["filename"], "tokens": d["tokens"]} for d in selected],
                key=lambda x: x["filename"],
            ),
        },
        "layer_b": {
            "files": len(layer_b_files),
            "tokens": layer_b_tokens,
            "source_dir": as_repo_relative(layer_b_dir, repo_root),
            "selected": sorted(
                [{"filename": d["filename"], "tokens": d["tokens"]} for d in layer_b_files],
                key=lambda x: x["filename"],
            ),
        },
        "total_tokens": total_with_lb,
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[saved] manifest: {manifest_path}")
    print(f"[saved] corpus:   {output_dir}/ (layer_a/ + layer_b/)")


if __name__ == "__main__":
    main()
