#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path
import re

import pdfplumber


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


def convert_pdf(pdf_path: Path, out_path: Path, overwrite: bool = False, verbose: bool = True) -> None:
    if out_path.exists() and not overwrite:
        if verbose:
            print(f"  [skip] {out_path.name} already exists")
        return

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_texts = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
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

        # join pages, then fix hyphenation, then strip tails, then normalise
        joined = "\n\n".join(page_texts)
        joined = fix_hyphenation(joined)
        stripped = strip_trailing_admin(joined)
        stripped = truncate_admin_tail(stripped)
        cleaned = clean_text(stripped)

        if len(cleaned) < 200:
            print(f"  [note] {pdf_path.name} produced very short text ({len(cleaned)} chars)", file=sys.stderr)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        if verbose:
            print(f"  [ok] wrote {out_path.name} ({len(cleaned.splitlines())} lines)")

    except Exception as e:
        print(f"  [error] failed on {pdf_path.name}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf-dir",
        default="data/domain_corpus/raw_pdfs",
        help="directory containing tribunal PDFs",
    )
    parser.add_argument(
        "--out-dir",
        default="data/domain_corpus/raw_txt",
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
