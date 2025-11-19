import re
import json
from pathlib import Path

RAW = Path("../data/domain_corpus/raw")
PROC = Path("../data/domain_corpus/processed")
PROC.mkdir(exist_ok=True, parents=True)

def clean_text(t: str) -> str:
    t = t.replace("\r\n", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()

def main():
    docs = []
    for path in RAW.glob("*.txt"):
        raw = path.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        if not cleaned:
            continue
        docs.append({
            "id": path.stem,
            "text": cleaned,
            "source": "education_law_raw"
        })

    out_path = PROC / "corpus.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # summary stats
    stats = {
        "num_docs": len(docs),
        "avg_chars": sum(len(d["text"]) for d in docs) / max(len(docs), 1)
    }
    (PROC / "corpus_stats.json").write_text(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
