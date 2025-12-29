#!/usr/bin/env python
import argparse
import re
import sys
from pathlib import Path

import pdfplumber


# deterministic boilerplate patterns to remove (case-insensitive)
BOILERPLATE_PATTERNS = [
    # ACAS branding
    r"acas\s+helpline",
    r"www\.acas\.org\.uk",
    r"telephone:\s*0300\s*123\s*11\s*00",
    r"textphone:\s*18001\s*0300\s*123\s*11\s*00",
    
    # copyright notices
    r"©\s*crown\s+copyright",
    r"©\s*acas",
    r"this\s+publication\s+may\s+be\s+reproduced",
    
    # gov.uk standard footers
    r"print\s+this\s+page",
    r"is\s+this\s+page\s+useful",
    r"report\s+a\s+problem\s+with\s+this\s+page",
    
    # navigation elements
    r"back\s+to\s+top",
    r"skip\s+to\s+main\s+content",
]

# page number patterns (standalone or with prefix)
PAGE_NUMBER_PATTERNS = [
    r"^page\s+\d+\s*$",
    r"^\d+\s*$",  # lone numbers on a line (conservative - only if isolated)
    r"^page\s+\d+\s+of\s+\d+\s*$",
]

# header/footer indicators (lines that suggest non-content)
HEADER_FOOTER_INDICATORS = [
    r"^acas\s*$",
    r"^guidance\s*$",
    r"^code\s+of\s+practice\s*$",
]


def looks_like_page_number(line: str) -> bool:
    """Check if a line is likely just a page number."""
    stripped = line.strip()
    if not stripped:
        return False
    
    # exact digit only if short
    if stripped.isdigit() and len(stripped) <= 3:
        return True
    
    # "page X" variants
    for pattern in PAGE_NUMBER_PATTERNS:
        if re.match(pattern, stripped, re.IGNORECASE):
            return True
    
    return False


def looks_like_header_footer(line: str) -> bool:
    """Check if a line is likely a header/footer (not content)."""
    stripped = line.strip()
    if not stripped:
        return False
    
    for pattern in HEADER_FOOTER_INDICATORS:
        if re.match(pattern, stripped, re.IGNORECASE):
            return True
    
    return False


def contains_boilerplate(text: str) -> bool:
    """Check if text contains deterministic boilerplate patterns."""
    lower = text.lower()
    for pattern in BOILERPLATE_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def remove_boilerplate_lines(lines: list[str]) -> list[str]:
    """Remove lines that match deterministic boilerplate patterns."""
    cleaned = []
    for line in lines:
        stripped = line.strip()
        
        # skip empty
        if not stripped:
            cleaned.append(line)
            continue
        
        # skip page numbers
        if looks_like_page_number(stripped):
            continue
        
        # skip headers/footers
        if looks_like_header_footer(stripped):
            continue
        
        # skip lines with boilerplate
        if contains_boilerplate(stripped):
            continue
        
        cleaned.append(line)
    
    return cleaned


def fix_hyphenation(text: str) -> str:
    """Join words split by hyphen + newline: 'employ-\nment' -> 'employment'."""
    return re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace:
    - Strip trailing spaces from lines
    - Collapse >2 consecutive blank lines to 2
    - Strip leading/trailing blank lines
    """
    lines = [line.rstrip() for line in text.splitlines()]
    
    # collapse excessive blank lines
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
        else:
            blank_count = 0
        
        # allow up to 2 consecutive blank lines
        if blank_count <= 2:
            cleaned_lines.append(line)
    
    # strip leading/trailing blank lines
    while cleaned_lines and cleaned_lines[0].strip() == "":
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1].strip() == "":
        cleaned_lines.pop()
    
    return "\n".join(cleaned_lines)


def remove_contents_section(text: str) -> str:
    """
    Attempt to remove table of contents section.
    Heuristic: if we see "contents" heading followed by lines with dots or page refs,
    cut from "contents" to first substantial paragraph.
    """
    lines = text.splitlines()
    
    # find "contents" heading (case-insensitive, may have leading numbers)
    contents_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        # match "contents", "table of contents", "1. contents", etc.
        if re.match(r"^(\d+\.?\s*)?contents?$", stripped):
            contents_idx = i
            break
    
    if contents_idx is None:
        return text
    
    # scan forward to find end of contents (heuristic: first line without dots/page numbers)
    end_idx = None
    for i in range(contents_idx + 1, min(contents_idx + 50, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        
        # contents lines typically have dots or page numbers
        has_dots = "..." in line or "…" in line
        has_page_ref = re.search(r"\d+$", line)
        
        # if line doesn't look like contents, assume contents section ended
        if not has_dots and not has_page_ref and len(line) > 20:
            end_idx = i
            break
    
    if end_idx is not None:
        # remove contents section
        lines = lines[:contents_idx] + lines[end_idx:]
    
    return "\n".join(lines)


def extract_and_clean(pdf_path: Path, verbose: bool = True) -> str | None:
    """
    Extract text from PDF and apply deterministic cleaning.
    Returns cleaned text or None if extraction fails or output is too short.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_texts = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    page_texts.append(text)
        
        if not page_texts:
            if verbose:
                print(f"  [warn] no text extracted from {pdf_path.name}", file=sys.stderr)
            return None
        
        # join pages
        joined = "\n\n".join(page_texts)
        
        # fix hyphenation first (before line-by-line processing)
        joined = fix_hyphenation(joined)
        
        # remove deterministic boilerplate lines
        lines = joined.splitlines()
        lines = remove_boilerplate_lines(lines)
        cleaned = "\n".join(lines)
        
        # remove contents section (if present)
        cleaned = remove_contents_section(cleaned)
        
        # normalize whitespace
        cleaned = normalize_whitespace(cleaned)
        
        # filter very short outputs
        if len(cleaned) < 500:
            if verbose:
                print(f"  [skip] {pdf_path.name} produced short text ({len(cleaned)} chars)", file=sys.stderr)
            return None
        
        return cleaned
    
    except Exception as e:
        print(f"  [error] failed on {pdf_path.name}: {e}", file=sys.stderr)
        return None


def get_source_type(pdf_path: Path, layer_b_root: Path) -> str:
    """Determine source type based on subdirectory."""
    try:
        relative = pdf_path.relative_to(layer_b_root)
        # first part of path is the source type
        source = relative.parts[0]
        return source
    except ValueError:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Extract text from Layer B PDFs with deterministic cleaning")
    parser.add_argument(
        "--source-dir",
        default="data/domain_corpus/layer_b_raw",
        help="root directory containing layer B subdirectories (acas_guides, codes, doctrine, govuk)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/domain_corpus/layer_b_raw_extracted",
        help="output directory for extracted .txt files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing .txt files",
    )
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    
    if not source_dir.exists():
        print(f"[fatal] source-dir {source_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # find all PDFs recursively
    pdf_paths = sorted(source_dir.rglob("*.pdf"))
    
    if not pdf_paths:
        print(f"[warn] no PDF files found under {source_dir}", file=sys.stderr)
        sys.exit(0)
    
    print(f"[info] found {len(pdf_paths)} PDF files under {source_dir}")
    
    # track stats by source type
    stats = {}
    
    for i, pdf_path in enumerate(pdf_paths, start=1):
        source_type = get_source_type(pdf_path, source_dir)
        
        # create output path (flatten structure, prefix with source type)
        out_filename = f"{source_type}__{pdf_path.stem}.txt"
        out_path = out_dir / out_filename
        
        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(pdf_paths)}] [skip] {out_filename} already exists")
            continue
        
        print(f"[{i}/{len(pdf_paths)}] [{source_type}] {pdf_path.name}")
        
        cleaned_text = extract_and_clean(pdf_path, verbose=True)
        
        if cleaned_text is None:
            stats[source_type] = stats.get(source_type, {"success": 0, "failed": 0})
            stats[source_type]["failed"] += 1
            continue
        
        # write output
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        
        print(f"  [ok] wrote {len(cleaned_text)} chars, {len(cleaned_text.splitlines())} lines")
        
        stats[source_type] = stats.get(source_type, {"success": 0, "failed": 0})
        stats[source_type]["success"] += 1
    
    # summary
    print("\n" + "="*60)
    print("Summary by source type:")
    for source_type in sorted(stats.keys()):
        s = stats[source_type]["success"]
        f = stats[source_type]["failed"]
        print(f"  {source_type:20s}: {s} success, {f} failed")
    
    total_success = sum(s["success"] for s in stats.values())
    total_failed = sum(s["failed"] for s in stats.values())
    print(f"\n  Total: {total_success} extracted, {total_failed} failed/skipped")


if __name__ == "__main__":
    main()
