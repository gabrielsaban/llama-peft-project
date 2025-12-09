#!/usr/bin/env python
import argparse
import os
import time
import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, urljoin

import requests
from bs4 import BeautifulSoup


def build_page_url(base_url: str, page: int) -> str:
    """inject or replace the `page` query parameter."""
    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query)
    qs["page"] = [str(page)]
    new_query = urlencode(qs, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def iter_decision_links(session: requests.Session, base_url: str, max_pages: int | None = None):
    page = 1
    while True:
        if max_pages is not None and page > max_pages:
            break

        page_url = build_page_url(base_url, page)
        print(f"[page {page}] fetching {page_url}")
        resp = session.get(page_url, timeout=15)
        if resp.status_code != 200:
            print(f"  ! got status {resp.status_code}, stopping")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # all decision links live in the results section and look like:
        # <a href="/employment-tribunal-decisions/..." >Mr X v Y: 1234/2024</a>
        anchors = soup.select('a[href^="/employment-tribunal-decisions/"]')

        decisions = []
        for a in anchors:
            title = a.get_text(strip=True)
            href = a.get("href")

            # cheap heuristic to avoid the top "employment tribunal decisions" nav link etc.
            if not href or " v " not in title:
                continue

            full_url = urljoin("https://www.gov.uk", href)
            decisions.append((title, full_url))

        if not decisions:
            print("  no decision links found on this page, stopping pagination.")
            break

        print(f"  found {len(decisions)} decisions")
        for title, url in decisions:
            yield title, url

        # be polite
        page += 1
        time.sleep(1.0)


def find_pdf_url(session: requests.Session, decision_url: str) -> str | None:
    """open a decision page and extract the pdf link."""
    resp = session.get(decision_url, timeout=15)
    if resp.status_code != 200:
        print(f"    ! decision page {decision_url} → status {resp.status_code}")
        return None

        # pattern on decision pages:
        # "Read the full decision in <a href='https://assets.publishing.service.gov.uk/...pdf'>..."
    soup = BeautifulSoup(resp.text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "assets.publishing.service.gov.uk" in href and href.lower().endswith(".pdf"):
            return href

    print(f"    ! no pdf link found on {decision_url}")
    return None


def safe_filename(base: str, ext: str = ".pdf", max_len: int = 150) -> str:
    """
    clean a base name and ensure the extension is preserved, even when truncated.
    """
    bad = '<>:"/\\|?*'
    for ch in bad:
        base = base.replace(ch, "_")

    # normalise whitespace/underscores
    base = " ".join(base.split())
    base = base.replace(" ", "_")
    base = base.strip("._")

    # ensure there's always room for the extension
    max_base_len = max_len - len(ext)
    if len(base) > max_base_len:
        base = base[:max_base_len]

    return base + ext


def extract_case_id(title: str) -> str | None:
    """
    pull out things like 1603397/2025 → 1603397_2025
    """
    m = re.search(r"(\d{5,7})/(\d{4})", title)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"


def slugify_title(title: str, max_len: int = 80) -> str:
    """
    "Ms N Pona v Yo Sushi: 1603397/2025" -> "Ms_N_Pona_v_Yo_Sushi"
    shortened to avoid ridiculous filenames.
    """
    if ":" in title:
        title = title.split(":", 1)[0]
    title = title.strip()
    out = []
    for ch in title:
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_.":
            out.append("_")
        # else drop punctuation
    slug = "".join(out)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "case"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        required=True,
        help="filtered employment-tribunal-decisions url (with all jurisdiction + date filters, no page param needed)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/domain_corpus/raw_pdfs",
        help="where to store downloaded pdfs",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="optional limit for testing (e.g. 5). omit to crawl all pages.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="sleep between downloads (seconds)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "employment-law-research-bot/0.1 (student project; contact via uni email)",
        }
    )
    session.max_redirects = 10

    seen = 0
    skipped = 0

    for title, decision_url in iter_decision_links(session, args.base_url, args.max_pages):
        pdf_url = find_pdf_url(session, decision_url)
        if not pdf_url:
            skipped += 1
            continue

        # we only use the extension from the pdf name
        pdf_name_part = os.path.basename(urlparse(pdf_url).path)
        _, ext = os.path.splitext(pdf_name_part)
        if not ext:
            ext = ".pdf"

        case_id = extract_case_id(title) or "unknown_caseid"
        slug = slugify_title(title)

        # simple, non-duplicated naming scheme
        base = f"{case_id}__{slug}"
        fname = safe_filename(base, ext=ext)
        out_path = os.path.join(args.out_dir, fname)

        if os.path.exists(out_path):
            print(f"    already have {fname}, skipping")
            continue

        print(f"    downloading pdf → {fname}")
        try:
            r = session.get(pdf_url, timeout=60, allow_redirects=True)
            r.raise_for_status()
            content = r.content

            # check pdf header before writing
            if not content.startswith(b"%PDF-"):
                print(f"    ! not a valid PDF (no %PDF- header), skipping: {pdf_url}")
                skipped += 1
                time.sleep(args.delay)
                continue

            with open(out_path, "wb") as f:
                f.write(content)

            seen += 1
        except Exception as e:
            print(f"    ! error downloading {pdf_url}: {e}")
            skipped += 1

        time.sleep(args.delay)

    print(f"\nfinished. downloaded {seen} pdfs, skipped {skipped}.")


if __name__ == "__main__":
    main()
