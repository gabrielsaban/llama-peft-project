#!/usr/bin/env python
"""
Reflow tribunal text to remove PDF layout artifacts while preserving legal structure.
"""
import argparse
import json
import re
import sys
from pathlib import Path


def normalize_whitespace(line: str) -> str:
    """Collapse multiple spaces to single space, strip leading/trailing."""
    return re.sub(r'\s+', ' ', line.strip())


def parse_marker(line: str) -> tuple:
    """
    Single source of truth for marker parsing.
    Returns (type, depth_rank, marker_text, span_end) or (None, None, None, None) if no marker.
    
    Types: 'number', 'decimal', 'alpha', 'roman', 'dash', 'bullet'
    Depth ranks: total ordering across marker kinds.
    - top-level numbered paragraphs: 0
    - decimal subsections: dot_count (1.3 => 1, 1.3.1 => 2)
    - dash: 9
    - bullet: 9
    - alpha: 10
    - roman: 11
    
    span_end: index where marker + following whitespace ends.
    
    Handles:
    - Decimal subsections: 1.3, 1.3.1 (with or without trailing dot)
    - Simple numbered: 70., 71), (72), [73]
    - Alphabetic: a., (b), c) - but excludes legal cross-refs like "(b) ERA"
    - Roman numerals: iv., (v), vi) AND bare forms like "iv " when standalone
    - Dash bullets: - item
    - Bullet dots: • item
    """
    line = line.strip()
    if not line:
        return (None, None, None, None)
    
    # check for decimal subsections (1.3, 1.3.1, etc) with optional trailing dot
    # capture marker span separately so span_end stops at marker+dot+spaces
    decimal_match = re.match(r'^(\d+(?:\.\d+)+)(\.?)\s+(.+)', line)
    if decimal_match:
        marker = decimal_match.group(1)
        rest = decimal_match.group(3)
        
        # treat as marker unless rest looks like a unit/measurement (e.g., "5.6 weeks")
        if rest:
            unit_like = re.match(
                r'^(?:%|£|\$|€|\d+\s*(?:%|°c|°f)|weeks?|months?|days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|years?|year|week|month|day|hour|minute|second|kg|kgs|g|mg|lb|lbs|km|m|cm|mm|mi|mph|kph)\b',
                rest,
                re.IGNORECASE
            )
            if not unit_like:
                depth_rank = marker.count('.')
                span_end = decimal_match.start(3)
                return ('decimal', depth_rank, marker, span_end)
    
    # check for simple numbered markers (70., 71., etc)
    simple_num_match = re.match(r'^(\d+)\.\s+', line)
    if simple_num_match:
        number_text = simple_num_match.group(1)
        # avoid treating 4-digit years as numbered paragraphs (e.g., 2019.)
        if len(number_text) == 4:
            return (None, None, None, None)
        return ('number', 0, number_text + '.', simple_num_match.end())
    
    # check for numbered with parens: 1), (2), [3]
    paren_match = re.match(r'^(\d+\)|\(\d+\)|\[\d+\])\s+', line)
    if paren_match:
        number_text = re.sub(r'[^0-9]', '', paren_match.group(1))
        if len(number_text) == 4:
            return (None, None, None, None)
        return ('number', 0, paren_match.group(1), paren_match.end())

    # check for dash bullets: - item
    dash_match = re.match(r'^(-)\s+', line)
    if dash_match:
        return ('dash', 9, '-', dash_match.end())

    # check for bullet dots: • item
    bullet_match = re.match(r'^(•)\s+', line)
    if bullet_match:
        return ('bullet', 9, '•', bullet_match.end())
    
    # check for alphabetic (a., b., (a), a)) but exclude cross-references
    alpha_match = re.match(r'^([a-zA-Z]\.|\([a-zA-Z]\)|[a-zA-Z]\))\s+', line)
    if alpha_match:
        # check if this is a legal cross-reference like "(b) ERA" or "(a) ICR"
        # pattern: (letter) followed by 2-5 uppercase letters (common acronyms)
        cross_ref = re.match(r'^\([a-zA-Z]\)\s+[A-Z]{2,5}(?:\s|$)', line)
        if cross_ref:
            # this is a cross-reference, not a list marker
            return (None, None, None, None)
        return ('alpha', 10, alpha_match.group(1), alpha_match.end())
    
    # check for roman numerals with punctuation: i., (ii), iii)
    # pattern: roman numeral + dot/paren/close-paren + whitespace
    roman_with_punct = re.match(
        r'^((?:i{1,3}v?|vi{0,3}|i?x|I{1,3}V?|VI{0,3}|I?X)[.\)]|\((?:i{1,3}v?|vi{0,3}|i?x|I{1,3}V?|VI{0,3}|I?X)\))\s+',
        line
    )
    if roman_with_punct:
        return ('roman', 11, roman_with_punct.group(1), roman_with_punct.end())
    
    # check for bare roman numerals (no punctuation) - be careful not to match words
    # only match if it's a standalone roman numeral followed by whitespace
    bare_roman_match = re.match(
        r'^(i{1,3}v?|vi{0,3}|i?x|I{1,3}V?|VI{0,3}|I?X)\s+',
        line
    )
    if bare_roman_match:
        roman_part = bare_roman_match.group(1)
        rest = line[bare_roman_match.end():].lstrip()
        next_char = rest[0] if rest else ''
        
        # safety check: next char should be uppercase (start of sentence)
        # this prevents matching words like "ivory" or "ivf"
        if next_char.isupper() or next_char.isdigit():
            # normalize to dot format: "iv" -> "iv."
            return ('roman', 11, roman_part.lower() + '.', bare_roman_match.end())
    
    return (None, None, None, None)


def has_paragraph_marker(line: str) -> bool:
    """Check if line starts with a recognized paragraph/listing marker."""
    marker_type, _, _, _ = parse_marker(line)
    return marker_type is not None


def normalize_paragraph_marker(line: str) -> str:
    """Normalize spacing after paragraph markers: ensure exactly one space."""
    line = line.strip()
    
    marker_type, _, marker_text, span_end = parse_marker(line)
    
    if marker_type is None:
        return line
    
    rest = line[span_end:].strip() if span_end is not None else ''
    return f"{marker_text} {rest}".strip()


def is_heading(line: str, prev_blank: bool, next_blank: bool, next_is_numbered: bool, in_list_block: bool) -> bool:
    """
    Check if line is a heading based on isolation and uppercase ratio.
    """
    normalized = normalize_whitespace(line)
    
    if not normalized:
        return False
    
    # exclusion: starts with a paragraph/list marker (not a heading)
    if has_paragraph_marker(normalized):
        return False
    
    # exclusion: ends with sentence punctuation
    if normalized.endswith(('.', '?', '!', ';')):
        return False
    
    # exclusion: too long to be a heading
    if len(normalized) > 60:
        return False
    
    # rule 1: genuine isolation-based detection
    # must be surrounded by blank lines OR followed by blank + numbered paragraph
    # avoid using numbered lookahead when inside a list block
    if prev_blank and (next_blank or (next_is_numbered and not in_list_block)):
        if in_list_block:
            # only treat as heading if it's clearly uppercase in list contexts
            letters = [c for c in normalized if c.isalpha()]
            if letters:
                uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
                if uppercase_ratio >= 0.7:
                    return True
            return False
        return True
    
    # rule 2: short AND mostly uppercase (≥70%)
    if len(normalized) <= 60:
        letters = [c for c in normalized if c.isalpha()]
        
        if letters:
            uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            
            # must be mostly uppercase to be a heading
            if uppercase_ratio >= 0.7:
                return True
    
    return False


def should_add_blank_before_marker(prev_marker_info, current_marker_info, prev_was_prose, in_list_block):
    """
    Return whether to add blank line before marker based on depth transitions.
    """
    if prev_was_prose or not prev_marker_info or prev_marker_info == (None, None, None, None):
        # coming from prose or start of doc
        return True
    
    prev_type, prev_depth, _, _ = prev_marker_info
    curr_type, curr_depth, _, _ = current_marker_info
    
    if not prev_type or not curr_type:
        return True
    
    # always blank before top-level numbered paragraphs (unless previous was also top-level number)
    if curr_depth == 0:
        return True
    
    # inside list blocks, avoid blanks between nearby depths
    if in_list_block:
        if prev_depth > 0 and curr_depth > 0:
            if abs(curr_depth - prev_depth) in (0, 1):
                return False
    
    return True


def update_list_stack(list_stack, depth_rank):
    """Update list depth stack based on current marker depth."""
    if depth_rank == 0:
        return []
    
    if not list_stack:
        return [depth_rank]
    
    if depth_rank > list_stack[-1]:
        list_stack.append(depth_rank)
        return list_stack
    
    if depth_rank == list_stack[-1]:
        list_stack[-1] = depth_rank
        return list_stack
    
    # shallower: pop until we find a parent depth
    while list_stack and list_stack[-1] > depth_rank:
        list_stack.pop()
    
    if list_stack and list_stack[-1] == depth_rank:
        list_stack[-1] = depth_rank
    else:
        list_stack.append(depth_rank)
    
    return list_stack


def preprocess_blank_lines(lines):
    """
    Remove excess blank lines inside lists while preserving structural blanks.
    """
    result = []
    n = len(lines)
    
    for i, line in enumerate(lines):
        if line:
            result.append(line)
            continue
        
        # find previous non-blank and next non-blank lines
        prev_idx = i - 1
        while prev_idx >= 0 and not lines[prev_idx]:
            prev_idx -= 1
        next_idx = i + 1
        while next_idx < n and not lines[next_idx]:
            next_idx += 1
        
        prev_line = lines[prev_idx] if prev_idx >= 0 else ''
        next_line = lines[next_idx] if next_idx < n else ''
        
        prev_marker = parse_marker(prev_line) if prev_line else (None, None, None, None)
        next_marker = parse_marker(next_line) if next_line else (None, None, None, None)
        
        prev_depth = prev_marker[1]
        next_depth = next_marker[1]
        
        # preserve blank before headings
        if next_line:
            next_is_marker = next_marker[0] is not None
            next_next_blank = (next_idx + 1 >= n) or not lines[next_idx + 1]
            if is_heading(next_line, True, next_next_blank, next_is_marker, False):
                result.append(line)
                continue
        
        # preserve blank between top-level numbered paragraphs
        if prev_depth == 0 and next_depth == 0:
            result.append(line)
            continue
        
        # remove blanks between list markers (depth > 0)
        if prev_depth and next_depth and prev_depth > 0 and next_depth > 0:
            continue
        
        result.append(line)
    
    return result


def reflow_text(text: str) -> str:
    """
    Reflow text by merging lines within paragraphs while preserving structure.
    """
    lines = text.splitlines()
    
    # first pass: normalize whitespace and identify blank lines
    normalized_lines = []
    for line in lines:
        norm = normalize_whitespace(line)
        normalized_lines.append(norm)
    
    # preprocess: remove PDF-artefact blank lines inside list blocks
    normalized_lines = preprocess_blank_lines(normalized_lines)
    
    # second pass: identify structure and build output
    output = []
    i = 0
    n = len(normalized_lines)
    
    # track the marker info of the last paragraph added (to determine blank lines)
    last_marker_info = (None, None, None, None)
    # track current list context using a stack of depth ranks
    list_stack = []
    
    while i < n:
        current = normalized_lines[i]
        
        # skip blank lines (we'll add them back strategically)
        if not current:
            i += 1
            continue
        
        # look ahead
        prev_blank = (i == 0) or (i > 0 and not normalized_lines[i-1])
        in_list_block = len(list_stack) > 0
        
        next_blank = False
        next_is_numbered = False
        if i + 1 < n:
            next_blank = not normalized_lines[i + 1]
            if i + 1 < n and normalized_lines[i + 1]:
                next_is_numbered = has_paragraph_marker(normalized_lines[i + 1])
        
        # check if current line is a heading
        if is_heading(current, prev_blank, next_blank, next_is_numbered, in_list_block):
            # add blank line before heading (if not at start)
            if output and output[-1] != '':
                output.append('')
            
            output.append(current)
            
            # add blank line after heading
            output.append('')
            # reset marker and list tracking after heading
            last_marker_info = (None, None, None, None)
            list_stack = []
            i += 1
            continue
        
        # get marker info for current line (before any modifications)
        current_marker_info = parse_marker(current)
        
        # check if starts with paragraph marker
        if has_paragraph_marker(current):
            # determine if we should add blank line before this marker
            prev_was_prose = (last_marker_info == (None, None, None, None))
            should_add_blank = should_add_blank_before_marker(
                last_marker_info, current_marker_info, prev_was_prose, in_list_block
            )
            
            # add blank line before paragraph (if needed)
            if should_add_blank and output and output[-1] != '':
                output.append('')
            
            # update list context
            current_depth = current_marker_info[1]
            list_stack = update_list_stack(list_stack, current_depth)
            
            # normalize the marker spacing
            current = normalize_paragraph_marker(current)
        else:
            # not a marker - reset list context
            list_stack = []
        
        # start building paragraph
        paragraph = [current]
        i += 1
        
        # try to merge following lines into this paragraph
        while i < n:
            next_line = normalized_lines[i]
            
            # hit a blank line - look ahead to decide
            if not next_line:
                # find next non-blank line
                j = i + 1
                while j < n and not normalized_lines[j]:
                    j += 1
                
                if j >= n:
                    # no more content, end paragraph
                    i = j
                    break
                
                next_non_blank = normalized_lines[j]
                # parse marker once - single source of truth
                next_marker = parse_marker(next_non_blank)
                next_depth = next_marker[1]
                
                # swallow blank lines inside list blocks
                if list_stack and next_marker[0] is not None:
                    base_depth = list_stack[0]
                    if next_depth is not None and next_depth >= base_depth and next_depth > 0:
                        i = j
                        break
                else:
                    # check if next is structural (heading or new marker)
                    real_prev_blank = not normalized_lines[j - 1] if j > 0 else True
                    next_next_blank = (j + 1 >= n) or not normalized_lines[j + 1]
                    next_has_marker = (next_marker[0] is not None)
                    
                    is_structural = (
                        is_heading(next_non_blank, real_prev_blank, next_next_blank, next_has_marker, list_stack != []) or
                        next_has_marker
                    )
                    
                    if is_structural:
                        # end paragraph here
                        i = j
                        break
                    else:
                        # skip blank lines and continue merging
                        i = j
                        next_line = normalized_lines[i]
                        # fall through to process next_line
            
            # now next_line is non-blank
            if not next_line:
                break
            
            # check real prev_blank context for this line
            real_prev_blank_for_next = (i > 0) and not normalized_lines[i - 1]
            
            # check lookahead for next line after this one
            next_next_blank = (i + 1 >= n) or not normalized_lines[i + 1]
            next_next_is_num = False
            if i + 1 < n and normalized_lines[i + 1]:
                next_next_is_num = has_paragraph_marker(normalized_lines[i + 1])
            
            # check if next line is a heading (with real context)
            if is_heading(next_line, real_prev_blank_for_next, next_next_blank, next_next_is_num, list_stack != []):
                # Don't merge, end paragraph here
                break
            
            # check if next line starts new paragraph
            if has_paragraph_marker(next_line):
                # Don't merge, end paragraph here
                break
            
            # check if should merge with next line
            # aggressive: merge by default unless next is structural
            paragraph.append(next_line)
            i += 1
        
        # join paragraph into single line
        paragraph_text = ' '.join(paragraph)
        output.append(paragraph_text)
        
        # update last marker info only if current paragraph started with a marker
        if current_marker_info != (None, None, None, None):
            last_marker_info = current_marker_info
        else:
            last_marker_info = (None, None, None, None)
    
    # final pass: collapse multiple blank lines and clean edges
    final = []
    blank_count = 0
    
    for line in output:
        if not line:
            blank_count += 1
            if blank_count <= 1:
                final.append(line)
        else:
            blank_count = 0
            final.append(line)
    
    # strip leading/trailing blank lines
    while final and not final[0]:
        final.pop(0)
    while final and not final[-1]:
        final.pop()
    
    return '\n'.join(final)


def truncate_at_paragraph_boundary(text: str, max_tokens: int, tokenizer) -> tuple:
    """Truncate text at last paragraph boundary before max_tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    original_count = len(tokens)

    if original_count <= max_tokens:
        return text, original_count, original_count, False

    # decode the first max_tokens back to text, then find last paragraph break
    truncated_text = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)

    # find last paragraph boundary (double newline)
    last_break = truncated_text.rfind("\n\n")
    if last_break > len(truncated_text) * 0.5:
        # only snap to boundary if it's in the latter half — avoids catastrophic loss
        truncated_text = truncated_text[:last_break].rstrip()
    else:
        # fall back to last single newline (sentence-ish boundary)
        last_nl = truncated_text.rfind("\n")
        if last_nl > len(truncated_text) * 0.8:
            truncated_text = truncated_text[:last_nl].rstrip()

    kept_count = len(tokenizer.encode(truncated_text, add_special_tokens=False))
    return truncated_text, original_count, kept_count, True


def process_file(input_path: Path, output_path: Path, overwrite: bool = False,
                 tokenizer=None, max_tokens: int = None) -> dict:
    """Process a single tribunal text file."""
    if output_path.exists() and not overwrite:
        print(f"  [skip] {output_path.name} already exists")
        return None
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not text.strip():
            print(f"  [skip] {input_path.name} is empty", file=sys.stderr)
            return None
        
        # marker count for logging (based on normalized lines)
        normalized_lines = [normalize_whitespace(line) for line in text.splitlines()]
        marker_count = sum(1 for line in normalized_lines if parse_marker(line)[0] is not None)
        reflowed = reflow_text(text)

        # apply token cap if requested
        truncated = False
        original_tokens = None
        kept_tokens = None
        if tokenizer is not None and max_tokens is not None:
            reflowed, original_tokens, kept_tokens, truncated = (
                truncate_at_paragraph_boundary(reflowed, max_tokens, tokenizer)
            )
            if truncated:
                print(f"  [cap] {input_path.name} truncated {original_tokens:,} -> {kept_tokens:,} tokens")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(reflowed)
        
        print(f"  [ok] {output_path.name}")
        return {
            "marker_count": marker_count,
            "input_lines": len(normalized_lines),
            "output_lines": len(reflowed.splitlines()) if text.strip() else 0,
            "truncated": truncated,
            "original_tokens": original_tokens,
            "kept_tokens": kept_tokens,
        }
    
    except Exception as e:
        print(f"  [error] failed on {input_path.name}: {e}", file=sys.stderr)
        return None


def write_jsonl_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Reflow tribunal text to remove PDF layout artifacts'
    )
    parser.add_argument(
        '--input-dir',
        default='../data/domain_corpus/raw_txt',
        help='directory containing extracted .txt files'
    )
    parser.add_argument(
        '--output-dir',
        default='../data/domain_corpus/raw_txt_reflow',
        help='directory to write reflowed .txt files'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='optional cap for testing'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='overwrite existing files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='write per-file metrics to logs'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=None,
        help='cap per-document token count (truncates at paragraph boundary)',
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"[fatal] input-dir {input_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    txt_files = sorted(input_dir.glob('*.txt'))
    if args.max_files:
        txt_files = txt_files[:args.max_files]
    
    print(f"[info] found {len(txt_files)} text files in {input_dir}")

    # load tokenizer if capping is requested
    tokenizer = None
    if args.max_tokens is not None:
        from transformers import AutoTokenizer
        print(f"[info] loading tokenizer for --max-tokens {args.max_tokens:,}")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        print(f"[info] tokenizer ready (vocab {len(tokenizer):,})")

    for i, input_path in enumerate(txt_files, start=1):
        output_path = output_dir / input_path.name
        print(f"[{i}/{len(txt_files)}] {input_path.name}")
        stats = process_file(
            input_path, output_path,
            overwrite=args.overwrite,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
        )
        if args.verbose and stats is not None:
            log_path = Path(__file__).resolve().parent.parent / "logs" / "reflow_metrics.jsonl"
            record = {
                "input": input_path.name,
                "output": output_path.name,
                "marker_count": stats["marker_count"],
                "input_lines": stats["input_lines"],
                "output_lines": stats["output_lines"],
            }
            if stats["truncated"]:
                record["truncated"] = True
                record["original_tokens"] = stats["original_tokens"]
                record["kept_tokens"] = stats["kept_tokens"]
            write_jsonl_log(log_path, record)


if __name__ == '__main__':
    main()
