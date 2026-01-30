#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path
import re

import pdfplumber


# indicate the start of actual judgment content
JUDGMENT_START_KEYWORDS = [
    "JUDGMENT",
    "DECISION ON A PRELIMINARY HEARING",
    "RESERVED JUDGMENT", 
    "ORDER",
    "JUDGMENT AT AN OPEN PRELIMINARY HEARING",
    "PRELIMINARY HEARING JUDGMENT"
]

BOILERPLATE_PATTERNS = [
    # interest notice page
    "the employment tribunals (interest) order 1990",
    "article 12",
    # public access + recording/transcription footer stuff
    "public access to employment tribunal decisions",
    "recording and transcription",
    "joint presidential practice direction on the recording and transcription of hearings",
]

TRAILING_LINE_PATTERNS = [
    # typical admin tails
    "judgment sent to the parties",
    "judgment was sent to the parties",
    "for the tribunal office",
]

RULE_FOOTER_SNIPPETS = [
    "judgment - rule",
    "judgment – rule",  # en dash
    "rule 61 march 2017",
]

TRUNCATION_PATTERNS = [
    "judgment sent to the parties on",
    "judgment was sent to the parties on",
    "judgment sent to the parties",
]


def looks_like_boilerplate(page_text: str) -> bool:
    """heuristically decide if a page is pure boilerplate we want to drop."""
    t = page_text.lower()
    return any(pat in t for pat in BOILERPLATE_PATTERNS)


def extract_page_with_margins(page, top_margin_pct: float = 0.08, bottom_margin_pct: float = 0.08) -> str:
    """
    Extract text from page while cropping top/bottom margins to remove headers/footers.
    
    Args:
        page: pdfplumber page object
        top_margin_pct: percentage of page height to crop from top (default 8%)
        bottom_margin_pct: percentage of page height to crop from bottom (default 8%)
    
    Returns:
        Extracted text with margins removed
    """
    bbox = page.bbox
    x0, y0, x1, y1 = bbox
    height = y1 - y0
    
    # Calculate cropping coordinates
    crop_top = y0 + (height * top_margin_pct)
    crop_bottom = y1 - (height * bottom_margin_pct)
    
    # Crop the page
    cropped = page.crop((x0, crop_top, x1, crop_bottom))
    
    # Extract text with improved settings
    text = cropped.extract_text(layout=True, x_tolerance=1, y_tolerance=3) or ""
    
    return text


def fix_hyphenation(text: str) -> str:
    """
    join words split by hyphen + newline:
      'discrimi-\nnation' -> 'discrimination'
    """
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def strip_trailing_admin(text: str) -> str:
    """
    remove trailing admin boilerplate:
      - 'JUDGMENT SENT TO THE PARTIES...'
      - 'FOR THE TRIBUNAL OFFICE'
      - standard 'Note:' block about written reasons / rule 61
      - 'Judgment - rule 61 March 2017' footer lines
    """
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return text

    # look only in the last ~40 lines for admin tails
    window_start = max(0, n - 40)
    cutoff = n

    for i in range(window_start, n):
        raw = lines[i]
        low = raw.strip().lower()

        # direct trailing lines
        if any(low.startswith(pat) for pat in TRAILING_LINE_PATTERNS):
            cutoff = i
            break

        # rule 61 footer
        if any(snippet in low for snippet in RULE_FOOTER_SNIPPETS):
            cutoff = i
            break

        # "Note:" block – if we see a "note:" line and the next couple of lines
        # mention written reasons / rule 61, treat as admin block
        if low.startswith("note"):
            neighbourhood = " ".join(l.lower() for l in lines[i : min(n, i + 4)])
            if ("written reasons" in neighbourhood) or ("rule 61" in neighbourhood):
                cutoff = i
                break

    # don't aggressively chop tiny docs
    if cutoff < n and cutoff > 10:
        lines = lines[:cutoff]

    # drop residual rule-footer lines anywhere
    cleaned = []
    for raw in lines:
        low = raw.strip().lower()
        if any(snippet in low for snippet in RULE_FOOTER_SNIPPETS):
            continue
        cleaned.append(raw)

    return "\n".join(cleaned)


def truncate_admin_tail(text: str) -> str:
    """
    final hard cut: if any of the truncation patterns appear,
    chop everything from the earliest occurrence onwards.
    """
    lower = text.lower()
    cut_pos = None
    for pat in TRUNCATION_PATTERNS:
        idx = lower.find(pat)
        if idx != -1:
            cut_pos = idx if cut_pos is None else min(cut_pos, idx)
    if cut_pos is None:
        return text
    return text[:cut_pos].rstrip()


def clean_text(text: str) -> str:
    """
    basic normalisation:
      - strip trailing spaces
      - collapse >1 blank line
      - trim leading/trailing blank lines
    """
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
        else:
            blank_count = 0
        # allow up to 1 consecutive blank line
        if blank_count <= 1:
            cleaned_lines.append(line)

    # strip leading/trailing empty lines
    while cleaned_lines and cleaned_lines[0].strip() == "":
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1].strip() == "":
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


def chop_front_matter(text: str) -> str:
    """
    remove everything before the first occurrence of judgment keywords.
    """
    lines = text.splitlines()
    
    for i, line in enumerate(lines):
        line_upper = line.strip().upper()
        
        # check for exact match or keyword at start of line
        for keyword in JUDGMENT_START_KEYWORDS:
            if line_upper == keyword or line_upper.startswith(keyword):
                # return text from this line onwards
                return "\n".join(lines[i:])
    
    # o keyword found, return original text
    return text


def score_document_quality(text: str) -> dict:
    """
    score a document based on features indicating it contains actual reasoning.
    """
    score = 0
    reasons = []
    
    lower_text = text.lower()
    lines = text.splitlines()
    
    # check first 500 chars for 'reasons' - strong signal
    first_chunk = lower_text[:500]
    if 'reasons' in first_chunk:
        score += 50
        reasons.append("'reasons' appears near top")
    
    # Check for section headings (case-insensitive, allowing variations)
    section_headings = [
        ('introduction', 10),
        ('the issues', 15),
        ('issues', 10),
        ('finding of fact', 20),
        ('findings of fact', 20),
        ('facts', 10),
        ('discussion', 15),
        ('analysis', 12),
        ('conclusion', 15),
        ('conclusions', 15),
        ('law', 8),
        ('relevant law', 12),
    ]
    
    for heading, points in section_headings:
        # Look for heading as standalone line or at start of line
        pattern = r'(?:^|\n)\s*' + re.escape(heading) + r'\s*(?:\n|$)'
        if re.search(pattern, lower_text, re.MULTILINE | re.IGNORECASE):
            score += points
            reasons.append(f"section heading: '{heading}'")
    
    # Additional quality indicators
    if 'judgment' in lower_text:
        score += 5
        
    if 'tribunal' in lower_text:
        score += 3
    
    # Decision quality: has paragraph numbering
    if re.search(r'\n\s*\d{1,3}\.\s+\w', text):
        score += 10
        reasons.append("contains numbered paragraphs")
    
    # Keep if score >= 30 or contains 'reasons' anywhere
    keep = score >= 30 or ('reasons' in lower_text)
    
    return {
        'score': score,
        'keep': keep,
        'reasons': reasons
    }


def convert_pdf(pdf_path: Path, out_path: Path, overwrite: bool = False, verbose: bool = True) -> None:
    if out_path.exists() and not overwrite:
        if verbose:
            print(f"  [skip] {out_path.name} already exists")
        return

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_texts = []
            for i, page in enumerate(pdf.pages):
                # Use new extraction method with margin cropping
                text = extract_page_with_margins(page)
                
                if not text.strip():
                    continue

                if looks_like_boilerplate(text):
                    if verbose:
                        print(f"    [drop] page {i+1} looks like boilerplate in {pdf_path.name}")
                    continue

                page_texts.append(text)

        if not page_texts:
            print(f"  [warn] no non-boilerplate text extracted from {pdf_path.name}", file=sys.stderr)
            return

        # Join pages, then apply all cleaning steps
        joined = "\n\n".join(page_texts)
        
        # Fix hyphenation first
        joined = fix_hyphenation(joined)
        
        # Chop front
        joined = chop_front_matter(joined)
        
        # Strip trailing admin text
        stripped = strip_trailing_admin(joined)
        stripped = truncate_admin_tail(stripped)
        
        # Normalize formatting
        cleaned = clean_text(stripped)
        
        # Quality scoring
        quality = score_document_quality(cleaned)
        
        # Count words (not characters)
        word_count = len(cleaned.split())
        has_reasons = 'reasons' in cleaned.lower()
        
        # Filter: keep if 400+ words OR has 'reasons'
        if word_count < 400 and not has_reasons:
            if out_path.exists() and overwrite:
                try:
                    out_path.unlink()
                except Exception as e:
                    print(
                        f"  [warn] could not remove existing output for short doc {out_path.name}: {e}",
                        file=sys.stderr,
                    )
            print(
                f"  [skip] {pdf_path.name} too short ({word_count} words < 400, no 'reasons')",
                file=sys.stderr,
            )
            return
        
        # Additional quality filter
        if not quality['keep']:
            print(
                f"  [skip] {pdf_path.name} low quality score ({quality['score']})",
                file=sys.stderr,
            )
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        if verbose:
            quality_info = f"score={quality['score']}"
            print(f"  [ok] wrote {out_path.name} ({word_count} words, {quality_info})")

    except Exception as e:
        print(f"  [error] failed on {pdf_path.name}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf-dir",
        default="../data/domain_corpus/raw_pdfs",
        help="directory containing tribunal PDFs",
    )
    parser.add_argument(
        "--out-dir",
        default="../data/domain_corpus/raw_txt",
        help="directory to write cleaned .txt files",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="optional cap for testing (e.g. 150)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing .txt files",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)

    if not pdf_dir.exists():
        print(f"[fatal] pdf-dir {pdf_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    pdf_paths = sorted(p for p in pdf_dir.glob("*.pdf"))
    if args.max_files is not None:
        pdf_paths = pdf_paths[: args.max_files]

    print(f"[info] found {len(pdf_paths)} pdf files under {pdf_dir}")

    for i, pdf_path in enumerate(pdf_paths, start=1):
        out_path = out_dir / (pdf_path.stem + ".txt")
        print(f"[{i}/{len(pdf_paths)}] {pdf_path.name}")
        convert_pdf(pdf_path, out_path, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
