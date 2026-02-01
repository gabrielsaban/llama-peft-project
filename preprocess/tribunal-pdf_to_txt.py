#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path
import re

import json

import pdfplumber


# indicate the start of actual judgment content (must be mostly uppercase)
JUDGMENT_START_KEYWORDS = [
    "JUDGMENT",
    "REASONS",
    "WRITTEN REASONS",
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


def truncate_boilerplate_in_page(page_text: str) -> str:
    """Cut boilerplate from bottom of page."""
    t_lower = page_text.lower()
    text_len = len(page_text)
    
    # find first boilerplate pattern
    earliest_pos = None
    for pat in BOILERPLATE_PATTERNS:
        pos = t_lower.find(pat)
        if pos != -1:
            earliest_pos = pos if earliest_pos is None else min(earliest_pos, pos)
    
    # if boilerplate found, cut everything from that point onwards
    # only if it appears near the bottom of the page (footer-like)
    if earliest_pos is not None:
        footer_threshold = int(text_len * 0.6) if text_len else 0
        if earliest_pos >= footer_threshold or text_len < 1200:
            return page_text[:earliest_pos].rstrip()
    
    return page_text


def extract_page_with_margins(page, top_margin_pct: float = 0.05, bottom_margin_pct: float = 0.05) -> str:
    """Extract text with top/bottom margins cropped to remove headers/footers."""
    bbox = page.bbox
    x0, y0, x1, y1 = bbox
    height = y1 - y0
    
    # calculate cropping coordinates
    crop_top = y0 + (height * top_margin_pct)
    crop_bottom = y1 - (height * bottom_margin_pct)
    
    # crop the page
    cropped = page.crop((x0, crop_top, x1, crop_bottom))
    
    # extract text with improved settings
    text = cropped.extract_text(layout=True, x_tolerance=1, y_tolerance=3) or ""
    
    return text


def fix_hyphenation(text: str) -> str:
    """Join words split by hyphen + newline."""
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def remove_header_footer_artifacts(text: str) -> tuple:
    """Remove case numbers, page numbers, separator lines that survived margin crop."""
    lines = text.splitlines()
    cleaned = []
    removed_count = 0
    
    for line in lines:
        stripped = line.strip()
        
        # skip 'Page 1 of 99' / 'page 1 of 99'
        if re.match(r'^\s*page\s+\d+\s+of\s+\d+\s*$', stripped, re.IGNORECASE):
            removed_count += 1
            continue
        
        # skip 'Case No. 1234567/2023' or 'Case No 1234567/2023, 2345678/2023'
        if re.match(r'^\s*case\s+no\.?\s*:?\s*\d{5,7}/\d{4}(\s*(?:,|&|and)\s*\d{5,7}/\d{4})*\s*$', stripped, re.IGNORECASE):
            removed_count += 1
            continue
        
        # skip 'Case Number: 1234567/2023' or 'Case Numbers: 1234567/2023 & 2345678/2023'
        if re.match(r'^\s*case\s+number(s)?\s*:?\s*\d{5,7}/\d{4}(\s*(?:,|&|and)\s*\d{5,7}/\d{4})*\s*$', stripped, re.IGNORECASE):
            removed_count += 1
            continue

        # skip 'Claim number. 2223380/2024'
        if re.match(r'^\s*claim\s+number\.?\s*\d{5,7}/\d{4}\s*$', stripped, re.IGNORECASE):
            removed_count += 1
            continue
        
        # skip standalone 'Case Number' or 'Case Numbers' lines
        if re.match(r'^\s*case\s+number(s)?\s*$', stripped, re.IGNORECASE):
            removed_count += 1
            continue
            
        # skip standalone case ID: '1234567/2023'
        if re.match(r'^\s*\d{5,7}/\d{4}\s*$', line):
            removed_count += 1
            continue
            
        # skip page numbers (standalone digits)
        if re.match(r'^\s*\d+\s*$', line):
            removed_count += 1
            continue

        # skip page markers like '- 1 -' / '- 2 -'
        if re.match(r'^\s*-\s*\d+\s*-\s*$', line):
            removed_count += 1
            continue

        # skip page markers like '3 of 12'
        if re.match(r'^\s*\d+\s+of\s+\d+\s*$', stripped, re.IGNORECASE):
            removed_count += 1
            continue
        
        # skip 'Approved by' lines that leak through
        if re.match(r'^\s*approved\s+by\s*$', stripped, re.IGNORECASE):
            removed_count += 1
            continue
            
        # skip separator lines (5+ underscores or dashes)
        if re.match(r'^\s*_{5,}\s*$', line) or re.match(r'^\s*-{5,}\s*$', line):
            removed_count += 1
            continue
            
        cleaned.append(line)
    
    return "\n".join(cleaned), removed_count


def strip_trailing_admin(text: str) -> tuple:
    """Remove trailing admin boilerplate and rule 61 footers."""
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return text, None, None

    # look only in the last ~40 lines for admin tails
    window_start = max(0, n - 40)
    cutoff = n

    reason = None
    for i in range(window_start, n):
        raw = lines[i]
        low = raw.strip().lower()

        # direct trailing lines
        if any(low.startswith(pat) for pat in TRAILING_LINE_PATTERNS):
            cutoff = i
            reason = "trailing_line"
            break

        # rule 61 footer
        if any(snippet in low for snippet in RULE_FOOTER_SNIPPETS):
            cutoff = i
            reason = "rule_footer"
            break

        # "Note:" block – if we see a "note:" line and the next couple of lines
        # mention written reasons / rule 61, treat as admin block
        if low.startswith("note"):
            neighbourhood = " ".join(l.lower() for l in lines[i : min(n, i + 4)])
            if ("written reasons" in neighbourhood) or ("rule 61" in neighbourhood):
                cutoff = i
                reason = "note_block"
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

    return "\n".join(cleaned), reason, cutoff if reason else None


def truncate_admin_tail(text: str) -> tuple:
    """Hard cut at earliest truncation pattern."""
    lower = text.lower()
    cut_pos = None
    matched_pattern = None
    for pat in TRUNCATION_PATTERNS:
        idx = lower.find(pat)
        if idx != -1 and (cut_pos is None or idx < cut_pos):
            cut_pos = idx
            matched_pattern = pat
    if cut_pos is None:
        return text, None, None
    return text[:cut_pos].rstrip(), matched_pattern, cut_pos


def clean_text(text: str) -> str:
    """Strip indentation, collapse blank lines, trim edges."""
    lines = [line.lstrip().rstrip() for line in text.splitlines()]
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


def chop_front_matter(text: str) -> tuple:
    """Remove everything before first uppercase judgment keyword."""
    lines = text.splitlines()
    
    for i, line in enumerate(lines):
        # Normalize whitespace to handle multiple spaces (e.g., 'RESERVED  JUDGMENT')
        norm = re.sub(r"\s+", " ", line.strip())
        
        # Count uppercase vs total letters to ensure it's a heading
        letters = [c for c in norm if c.isalpha()]
        if not letters:
            continue
        uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        
        # Must be at least 70% uppercase to be considered a heading
        if uppercase_ratio < 0.7:
            continue
        
        # Check for exact match or keyword at start of line
        norm_upper = norm.upper()
        for keyword in JUDGMENT_START_KEYWORDS:
            if norm_upper == keyword or norm_upper.startswith(keyword + " ") or norm_upper.startswith(keyword + ":"):
                # Return text from this line onwards
                return "\n".join(lines[i:]), i
    
    # No keyword found, return original text
    return text, None


def final_safety_trim(text: str) -> tuple:
    """Check last 20% of document for admin patterns at line starts."""
    lines = text.splitlines()
    n = len(lines)
    
    if n == 0:
        return text, None, None
    
    # Admin patterns to detect at line start (case insensitive)
    admin_patterns = [
        'employment judge',
        'approved by',
        'order sent to the parties',
        'judgment sent to the parties',
        'sent to the parties',
        'for the tribunal office',
    ]
    
    cutoff = n
    
    # Check last 20% of document (minimum 10 lines, maximum 50 lines)
    check_lines = max(10, min(50, int(n * 0.2)))
    check_start = max(0, n - check_lines)
    
    matched_pattern = None
    for i in range(check_start, n):
        line_stripped = lines[i].strip().lower()
        for pattern in admin_patterns:
            # Check if line starts with the pattern (not just contains it)
            if line_stripped.startswith(pattern):
                cutoff = min(cutoff, i)
                matched_pattern = pattern
                break
        if cutoff < n:
            break
    
    if cutoff < n:
        return "\n".join(lines[:cutoff]), matched_pattern, cutoff
    
    return text, None, None


def write_jsonl_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def score_document_quality(text: str) -> dict:
    """Score document based on features indicating actual reasoning."""
    score = 0
    reasons_list = []
    
    lower_text = text.lower()
    lines = text.splitlines()
    
    # Check first 500 chars for 'REASONS' as heading (uppercase)
    first_lines = lines[:10] if len(lines) >= 10 else lines
    for line in first_lines:
        norm = re.sub(r"\s+", " ", line.strip())
        if 'REASONS' in norm.upper():
            # Check if mostly uppercase (heading)
            letters = [c for c in norm if c.isalpha()]
            if letters:
                uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
                if uppercase_ratio >= 0.6:
                    score += 50
                    reasons_list.append("'REASONS' heading appears near top")
                    break
    
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
            reasons_list.append(f"section heading: '{heading}'")
    
    # Additional quality indicators
    if 'judgment' in lower_text:
        score += 5
        
    if 'tribunal' in lower_text:
        score += 3
    
    # Decision quality: has paragraph numbering (required for keep)
    paragraph_numbering_pattern = (
        r'(?:^|\n)\s*'
        r'(?:'
        r'\d{1,3}\.\s+\w'
        r'|\d+\)\s+\w'
        r'|\(\d+\)\s+\w'
        r'|\[\d+\]\s+\w'
        r'|\d{1,3}\.\d+\s+\w'
        r')'
    )
    has_paragraph_numbering = bool(re.search(paragraph_numbering_pattern, text))
    if has_paragraph_numbering:
        score += 10
        reasons_list.append("contains numbered paragraphs")
    
    # Keep criteria: must have paragraph numbering
    keep = has_paragraph_numbering
    
    return {
        'score': score,
        'keep': keep,
        'has_paragraph_numbering': has_paragraph_numbering,
        'reasons': reasons_list
    }


def convert_pdf(pdf_path: Path, out_path: Path, overwrite: bool = False, verbose: bool = True) -> None:
    if out_path.exists() and not overwrite:
        if verbose:
            print(f"  [skip] {out_path.name} already exists")
        return

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_texts = []
            page_char_counts = []
            for i, page in enumerate(pdf.pages):
                # Use new extraction method with margin cropping
                text = extract_page_with_margins(page)
                page_char_counts.append(len(text))
                
                if not text.strip():
                    continue

                # Truncate boilerplate within page instead of dropping whole page
                text = truncate_boilerplate_in_page(text)
                
                if not text.strip():
                    continue

                page_texts.append(text)

        if not page_texts:
            print(f"  [warn] no non-boilerplate text extracted from {pdf_path.name}", file=sys.stderr)
            return

        # Join pages, then apply all cleaning steps in proper order
        joined = "\n\n".join(page_texts)
        
        # 1. Fix hyphenation
        joined = fix_hyphenation(joined)
        
        # 2. Remove header/footer artifacts (case numbers, page numbers, separator lines)
        joined, removed_header_footer_lines = remove_header_footer_artifacts(joined)
        
        # 3. Chop front matter (before judgment keywords)
        joined, chop_front_index = chop_front_matter(joined)
        
        # 4. Strip trailing admin text (old patterns for compatibility)
        stripped, strip_reason, strip_cutoff = strip_trailing_admin(joined)
        stripped, trunc_pattern, trunc_pos = truncate_admin_tail(stripped)
        
        # 6. Normalize formatting (lstrip, collapse blanks)
        cleaned = clean_text(stripped)
        
        # 7. Final safety trim for leaked 'Approved By' and 'Employment Judge'
        cleaned, final_pattern, final_cutoff = final_safety_trim(cleaned)
        
        # 8. Quality scoring
        quality = score_document_quality(cleaned)
        
        # Count words (not characters)
        word_count = len(cleaned.split())
        
        # Filter: reject if word_count < 500 OR no paragraph numbering
        if word_count < 500:
            if out_path.exists() and overwrite:
                try:
                    out_path.unlink()
                except Exception as e:
                    print(
                        f"  [warn] could not remove existing output for short doc {out_path.name}: {e}",
                        file=sys.stderr,
                    )
            print(
                f"  [skip] {pdf_path.name} too short ({word_count} words < 500)",
                file=sys.stderr,
            )
            return
        
        # Quality filter: must have paragraph numbering
        if not quality['keep']:
            print(
                f"  [skip] {pdf_path.name} no paragraph numbering (quality score: {quality['score']})",
                file=sys.stderr,
            )
            return

        if verbose:
            log_path = Path(__file__).resolve().parent.parent / "logs" / "pdf_to_txt_metrics.jsonl"
            write_jsonl_log(
                log_path,
                {
                    "pdf": pdf_path.name,
                    "page_char_counts": page_char_counts,
                    "removed_header_footer_lines": removed_header_footer_lines,
                    "chop_front_matter_triggered": chop_front_index is not None,
                    "chop_front_matter_index": chop_front_index,
                    "strip_trailing_admin_triggered": strip_reason is not None,
                    "strip_trailing_admin_reason": strip_reason,
                    "strip_trailing_admin_cutoff": strip_cutoff,
                    "truncate_admin_tail_triggered": trunc_pattern is not None,
                    "truncate_admin_tail_pattern": trunc_pattern,
                    "truncate_admin_tail_pos": trunc_pos,
                    "final_safety_trim_triggered": final_pattern is not None,
                    "final_safety_trim_pattern": final_pattern,
                    "final_safety_trim_cutoff": final_cutoff,
                    "word_count": word_count,
                    "has_paragraph_numbering": quality["has_paragraph_numbering"],
                    "quality_score": quality["score"],
                },
            )

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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="write per-file metrics to logs",
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
        convert_pdf(pdf_path, out_path, overwrite=args.overwrite, verbose=args.verbose)


if __name__ == "__main__":
    main()
