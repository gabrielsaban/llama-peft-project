import json
import random
from pathlib import Path

PROC = Path("../data/domain_corpus/processed")
SPLITS = Path("../data/splits")
SPLITS.mkdir(parents=True, exist_ok=True)

def main(seed=42, val_ratio=0.1, test_ratio=0.1):
    docs = []
    with open(PROC / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    ids = [d["id"] for d in docs]
    random.Random(seed).shuffle(ids)

    n = len(ids)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test+n_val]
    train_ids = ids[n_test+n_val:]

    splits = {
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids,
    }

    (SPLITS / "intrinsic_splits.json").write_text(
        json.dumps(splits, indent=2),
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
