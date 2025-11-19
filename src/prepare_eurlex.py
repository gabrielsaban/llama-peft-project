from datasets import load_dataset
import json
from pathlib import Path

OUT_DIR = Path("data/lexglue")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ds = load_dataset("coastalcph/lex_glue", "eurlex")

    split_ids = {
        "train": list(range(len(ds["train"]))),
        "validation": list(range(len(ds["validation"]))),
        "test": list(range(len(ds["test"]))),
    }

    (OUT_DIR / "eurlex_splits.json").write_text(
        json.dumps(split_ids, indent=2),
        encoding="utf-8",
    )

    # store first sample of each split for inspection
    sample = {split: ds[split][0] for split in split_ids.keys()}
    (OUT_DIR / "eurlex_sample.json").write_text(
        json.dumps(sample, indent=2),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
