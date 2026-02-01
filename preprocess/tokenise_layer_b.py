#!/usr/bin/env python3
"""
tokenise Layer B corpus using LLaMA-3 tokenizer and compute token statistics.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean, median, stdev
from transformers import AutoTokenizer


def main():
    # paths
    layer_b_dir = Path("../data/domain_corpus/layer_b_raw_extracted")
    output_json = Path("../data/domain_corpus/layer_b_token_stats.json")
    output_csv = Path("../data/domain_corpus/layer_b_token_stats.csv")
    
    # check if cleaned directory exists, fallback to raw_extracted
    if not layer_b_dir.exists():
        print(f"⚠️  {layer_b_dir} not found, falling back to layer_b_raw_extracted")
        layer_b_dir = Path("data/domain_corpus/layer_b_raw_extracted")
        if not layer_b_dir.exists():
            raise FileNotFoundError(f"Neither layer_b_cleaned nor layer_b_raw_extracted found")
    
    print(f"📂 Using corpus directory: {layer_b_dir}")
    
    # load LLaMA-3 tokenizer
    print("🔧 Loading LLaMA-3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    print(f"✅ Tokenizer loaded: {len(tokenizer)} vocab size\n")
    
    # find all .txt files
    txt_files = sorted(layer_b_dir.glob("*.txt"))
    print(f"📄 Found {len(txt_files)} text files\n")
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {layer_b_dir}")
    
    # tokenize each file and collect stats
    file_stats = []
    source_type_data = defaultdict(list)
    
    print("🔢 Tokenizing files...")
    for filepath in txt_files:
        # read file
        text = filepath.read_text(encoding="utf-8")
        
        # count characters
        char_count = len(text)
        line_count = text.count('\n') + 1
        
        # tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(tokens)
        
        # derived metrics
        tokens_per_1k_chars = (token_count / char_count * 1000) if char_count > 0 else 0
        tokens_per_line = token_count / line_count if line_count > 0 else 0
        
        # extract source type from filename (everything before first __)
        filename = filepath.name
        if "__" in filename:
            source_type = filename.split("__")[0]
        else:
            source_type = "unknown"
        
        # store per-file stats
        file_stat = {
            "filename": filename,
            "source_type": source_type,
            "chars": char_count,
            "lines": line_count,
            "tokens": token_count,
            "tokens_per_1k_chars": round(tokens_per_1k_chars, 2),
            "tokens_per_line": round(tokens_per_line, 2)
        }
        file_stats.append(file_stat)
        
        # group by source type
        source_type_data[source_type].append(token_count)
    
    print(f"✅ Tokenized {len(file_stats)} files\n")
    
    # compute per-type aggregates
    source_type_stats = {}
    for source_type, token_counts in sorted(source_type_data.items()):
        source_type_stats[source_type] = {
            "file_count": len(token_counts),
            "total_tokens": sum(token_counts),
            "mean_tokens_per_file": round(mean(token_counts), 2),
            "median_tokens_per_file": round(median(token_counts), 2),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "std_tokens": round(stdev(token_counts), 2) if len(token_counts) > 1 else 0.0
        }
    
    #  overall totals
    total_tokens = sum(stat["tokens"] for stat in file_stats)
    total_chars = sum(stat["chars"] for stat in file_stats)
    
    overall_stats = {
        "total_files": len(file_stats),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "mean_tokens_per_file": round(mean([s["tokens"] for s in file_stats]), 2),
        "median_tokens_per_file": round(median([s["tokens"] for s in file_stats]), 2)
    }
    
    # prepare output
    output_data = {
        "overall": overall_stats,
        "by_source_type": source_type_stats,
        "per_file": file_stats
    }
    
    # write JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"💾 Saved JSON: {output_json}")
    
    # write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "source_type", "chars", "lines", "tokens", 
                     "tokens_per_1k_chars", "tokens_per_line"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(file_stats)
    print(f"💾 Saved CSV: {output_csv}")
    
    # print console summary
    print("\n" + "=" * 70)
    print("📊 LAYER B TOKENIZATION SUMMARY")
    print("=" * 70)
    
    print(f"\n📈 Overall Statistics:")
    print(f"  Total files:       {overall_stats['total_files']}")
    print(f"  Total tokens:      {overall_stats['total_tokens']:,}")
    print(f"  Total characters:  {overall_stats['total_chars']:,}")
    print(f"  Mean tokens/file:  {overall_stats['mean_tokens_per_file']:,}")
    print(f"  Median tokens/file: {overall_stats['median_tokens_per_file']:,}")
    
    print(f"\n📚 By Source Type:")
    for source_type, stats in sorted(source_type_stats.items()):
        print(f"\n  {source_type}:")
        print(f"    Files:       {stats['file_count']}")
        print(f"    Total:       {stats['total_tokens']:,} tokens")
        print(f"    Mean/file:   {stats['mean_tokens_per_file']:,} tokens")
        print(f"    Median/file: {stats['median_tokens_per_file']:,} tokens")
        print(f"    Range:       {stats['min_tokens']:,} - {stats['max_tokens']:,} tokens")
        print(f"    Std dev:     {stats['std_tokens']:,} tokens")
    
    # ranked list of biggest files
    print(f"\n🏆 Top 10 Largest Files (by token count):")
    ranked_files = sorted(file_stats, key=lambda x: x["tokens"], reverse=True)[:10]
    for i, fstat in enumerate(ranked_files, 1):
        print(f"  {i:2d}. {fstat['filename'][:60]:<60} {fstat['tokens']:>7,} tokens")
    
    print("\n" + "=" * 70)
    print("✅ Tokenization complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
