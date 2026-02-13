#!/usr/bin/env python3
"""tokenise post-reflow Layer A tribunal texts and report pilot statistics."""

import argparse
import json
import csv
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer


def compute_stats(values: list) -> dict:
    """return summary stats for a list of numeric values."""
    arr = np.array(values)
    return {
        "n": len(arr),
        "total": int(arr.sum()),
        "mean": round(float(arr.mean()), 1),
        "median": round(float(np.median(arr)), 1),
        "std": round(float(arr.std()), 1),
        "min": int(arr.min()),
        "q1": round(float(np.percentile(arr, 25)), 1),
        "q3": round(float(np.percentile(arr, 75)), 1),
        "max": int(arr.max()),
        "iqr": round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 1),
    }


def detect_outliers(filenames: list, token_counts: list, factor: float = 1.5) -> list:
    """flag outliers using IQR method."""
    arr = np.array(token_counts)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    outliers = []
    for fname, tc in zip(filenames, token_counts):
        if tc < lower or tc > upper:
            outliers.append({
                "filename": fname,
                "tokens": int(tc),
                "direction": "low" if tc < lower else "high",
            })
    return sorted(outliers, key=lambda x: x["tokens"])


def main():
    parser = argparse.ArgumentParser(
        description="tokenise post-reflow tribunal texts and compute pilot stats"
    )
    parser.add_argument(
        "--input-dir",
        default="../data/domain_corpus/raw_txt_reflow",
        help="directory of reflowed .txt files",
    )
    parser.add_argument(
        "--output-json",
        default="../data/domain_corpus/pilot_token_stats.json",
        help="path for output JSON stats",
    )
    parser.add_argument(
        "--output-csv",
        default="../data/domain_corpus/pilot_token_stats.csv",
        help="path for per-file CSV",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model name for tokenizer",
    )
    parser.add_argument(
        "--layer-b-tokens",
        type=int,
        default=200237,
        help="known Layer B token count (for budget calculation)",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=1800000,
        help="target total corpus tokens (Layer A + Layer B)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"no .txt files in {input_dir}")

    print(f"loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"  vocab size: {len(tokenizer):,}\n")

    print(f"tokenising {len(txt_files)} files from {input_dir} ...")
    file_stats = []
    filenames = []
    token_counts = []

    for fp in txt_files:
        text = fp.read_text(encoding="utf-8")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tc = len(tokens)
        chars = len(text)
        words = len(text.split())
        lines = text.count("\n") + 1

        file_stats.append({
            "filename": fp.name,
            "chars": chars,
            "words": words,
            "lines": lines,
            "tokens": tc,
            "tokens_per_1k_chars": round(tc / chars * 1000, 2) if chars else 0,
        })
        filenames.append(fp.name)
        token_counts.append(tc)

    # aggregate stats
    stats = compute_stats(token_counts)
    outliers = detect_outliers(filenames, token_counts)

    # budget estimation
    layer_a_target = args.target_total - args.layer_b_tokens
    median_tok = stats["median"]
    # estimate survival rate from PDFs→kept files
    # (caller can override by editing json, this is just a reasonable guess)
    pdf_count_guess = len(txt_files)  # approximation if 1:1 survival
    docs_needed = int(layer_a_target / median_tok) if median_tok > 0 else 0

    budget = {
        "layer_b_tokens": args.layer_b_tokens,
        "target_total_tokens": args.target_total,
        "layer_a_target_tokens": layer_a_target,
        "median_tokens_per_decision": median_tok,
        "estimated_decisions_needed": docs_needed,
        "pilot_kept_files": len(txt_files),
        "note": "adjust survival_rate after checking dropped/ count to refine scraping target",
    }

    # assemble output
    output = {
        "summary": stats,
        "budget_estimate": budget,
        "outliers": outliers,
        "per_file": file_stats,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nsaved: {output_json}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(file_stats[0].keys()))
        writer.writeheader()
        writer.writerows(file_stats)
    print(f"saved: {output_csv}")

    # console report
    print("\n" + "=" * 65)
    print("  LAYER A PILOT TOKENISATION")
    print("=" * 65)
    print(f"  files:      {stats['n']}")
    print(f"  total tok:  {stats['total']:,}")
    print(f"  mean:       {stats['mean']:,.1f}")
    print(f"  median:     {stats['median']:,.1f}")
    print(f"  std:        {stats['std']:,.1f}")
    print(f"  min:        {stats['min']:,}")
    print(f"  Q1:         {stats['q1']:,.1f}")
    print(f"  Q3:         {stats['q3']:,.1f}")
    print(f"  max:        {stats['max']:,}")
    print(f"  IQR:        {stats['iqr']:,.1f}")

    if outliers:
        print(f"\n  outliers ({len(outliers)}):")
        for o in outliers:
            tag = "LOW " if o["direction"] == "low" else "HIGH"
            print(f"    [{tag}] {o['tokens']:>7,} tok  {o['filename'][:55]}")

    print(f"\n  --- budget estimate ---")
    print(f"  layer B (fixed):        {budget['layer_b_tokens']:>10,} tok")
    print(f"  layer A target:         {budget['layer_a_target_tokens']:>10,} tok")
    print(f"  median tok/decision:    {budget['median_tokens_per_decision']:>10,.1f}")
    print(f"  decisions needed:       {budget['estimated_decisions_needed']:>10,}")
    print(f"  pilot kept files:       {budget['pilot_kept_files']:>10,}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
