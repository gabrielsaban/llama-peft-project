from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import json

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, PreTrainedTokenizerBase


@dataclass
class LMDataConfig:
    dataset_name: str  # e.g. "eurlex_text_lm" or "domain_corpus_lm"
    max_seq_length: int = 1024
    train_subset: Optional[int] = None
    val_subset: Optional[int] = None
    corpus_dir: Optional[str] = None
    split_json: Optional[str] = None
    seed: int = 42


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # llama tokenizer usually has no pad token; use eos as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tokenize_and_chunk_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    desc_prefix: str,
) -> Dataset:
    if not texts:
        raise ValueError(f"{desc_prefix}: no documents provided")

    tokenized = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
    )
    doc_token_ids = tokenized["input_ids"]

    concatenated: list[int] = []
    for ids in doc_token_ids:
        concatenated.extend(ids)

    total_length = (len(concatenated) // max_seq_length) * max_seq_length
    if total_length == 0:
        raise ValueError(
            f"{desc_prefix}: token stream shorter than one chunk of {max_seq_length}"
        )

    concatenated = concatenated[:total_length]
    input_ids = [
        concatenated[i : i + max_seq_length]
        for i in range(0, total_length, max_seq_length)
    ]
    attention_mask = [[1] * max_seq_length for _ in range(len(input_ids))]
    return Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask})


def _load_domain_split_rows(split_json_path: Path) -> tuple[list[dict], list[dict]]:
    with split_json_path.open("r", encoding="utf-8") as f:
        split_obj = json.load(f)

    if (
        "splits" in split_obj
        and "train" in split_obj["splits"]
        and ("val" in split_obj["splits"] or "validation" in split_obj["splits"])
    ):
        train_rows = split_obj["splits"]["train"]
        val_rows = split_obj["splits"].get("val", split_obj["splits"]["validation"])
        return train_rows, val_rows

    # Backward compatibility with legacy mapping-only files.
    if "by_relpath" in split_obj:
        train_rows = []
        val_rows = []
        for relpath, split in split_obj["by_relpath"].items():
            layer = relpath.split("/", 1)[0]
            row = {"relpath": relpath, "layer": layer}
            split_norm = str(split).lower()
            if split_norm == "train":
                train_rows.append(row)
            elif split_norm in {"val", "validation"}:
                val_rows.append(row)
        return train_rows, val_rows

    raise ValueError(f"Unsupported split JSON schema in: {split_json_path}")


def _row_relpath(row: dict) -> str:
    relpath = row.get("relpath")
    if relpath:
        return relpath
    layer = row.get("layer")
    filename = row.get("filename")
    if layer and filename:
        return f"{layer}/{filename}"
    raise ValueError(f"Split row missing relpath and layer/filename: {row}")


def _row_layer(row: dict) -> str:
    layer = row.get("layer")
    if layer:
        return layer
    relpath = _row_relpath(row)
    return relpath.split("/", 1)[0]


def get_domain_corpus_lm_datasets(
    tokenizer: PreTrainedTokenizerBase,
    cfg: LMDataConfig,
) -> Tuple[Dataset, Dataset, Dataset, Dataset, DataCollatorForLanguageModeling]:
    """
    Build LM datasets from corpus_final + intrinsic split JSON.

    Returns:
        lm_train, lm_val_all, lm_val_layer_a, lm_val_layer_b, collator
    """
    if not cfg.corpus_dir or not cfg.split_json:
        raise ValueError("domain_corpus_lm requires cfg.corpus_dir and cfg.split_json")

    corpus_dir = Path(cfg.corpus_dir)
    split_json_path = Path(cfg.split_json)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"corpus_dir not found: {corpus_dir}")
    if not split_json_path.exists():
        raise FileNotFoundError(f"split_json not found: {split_json_path}")

    train_rows, val_rows = _load_domain_split_rows(split_json_path)
    train_rows = sorted(train_rows, key=_row_relpath)
    val_rows = sorted(val_rows, key=_row_relpath)

    if cfg.train_subset is not None:
        train_rows = train_rows[: min(cfg.train_subset, len(train_rows))]
    if cfg.val_subset is not None:
        val_rows = val_rows[: min(cfg.val_subset, len(val_rows))]

    val_a_rows = [r for r in val_rows if _row_layer(r) == "layer_a"]
    val_b_rows = [r for r in val_rows if _row_layer(r) == "layer_b"]

    def read_texts(rows: list[dict]) -> list[str]:
        texts = []
        missing = []
        for row in rows:
            relpath = _row_relpath(row)
            path = corpus_dir / relpath
            if not path.exists():
                missing.append(str(path))
                continue
            texts.append(path.read_text(encoding="utf-8", errors="replace"))

        if missing:
            preview = "\n".join(missing[:5])
            raise FileNotFoundError(
                f"Missing {len(missing)} files referenced by split JSON under {corpus_dir}. "
                f"Examples:\n{preview}"
            )
        return texts

    train_texts = read_texts(train_rows)
    val_texts = read_texts(val_rows)
    val_a_texts = read_texts(val_a_rows)
    val_b_texts = read_texts(val_b_rows)

    lm_train = _tokenize_and_chunk_texts(
        train_texts, tokenizer, cfg.max_seq_length, desc_prefix="domain train"
    )
    lm_val_all = _tokenize_and_chunk_texts(
        val_texts, tokenizer, cfg.max_seq_length, desc_prefix="domain val (all)"
    )
    lm_val_layer_a = _tokenize_and_chunk_texts(
        val_a_texts, tokenizer, cfg.max_seq_length, desc_prefix="domain val (layer_a)"
    )
    lm_val_layer_b = _tokenize_and_chunk_texts(
        val_b_texts, tokenizer, cfg.max_seq_length, desc_prefix="domain val (layer_b)"
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return lm_train, lm_val_all, lm_val_layer_a, lm_val_layer_b, collator


def get_eurlex_text_lm_datasets(
    tokenizer: PreTrainedTokenizerBase,
    cfg: LMDataConfig,
) -> Tuple[Dataset, Dataset, DataCollatorForLanguageModeling]:
    """
    prepare eur-lex as a causal language modelling dataset:

    - load 'coastalcph/lex_glue', config 'eurlex'
    - use 'text' field
    - tokenize
    - concatenate and chunk into fixed-length sequences
    """

    raw = load_dataset("coastalcph/lex_glue", "eurlex")

    train_ds = raw["train"]
    val_ds = raw["validation"]

    if cfg.train_subset is not None:
        train_ds = train_ds.select(range(min(cfg.train_subset, len(train_ds))))
    if cfg.val_subset is not None:
        val_ds = val_ds.select(range(min(cfg.val_subset, len(val_ds))))

    # 1) tokenize without truncation so we can pack efficiently
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            add_special_tokens=True,
            truncation=False,
        )

    tokenized_train = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="tokenising train",
    )
    tokenized_val = val_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=val_ds.column_names,
        desc="tokenising val",
    )

    # 2) group texts into chunks of max_seq_length
    def group_texts(examples):
        # concatenate all tokens in the batch
        concatenated = sum(examples["input_ids"], [])
        total_length = len(concatenated)
        # drop the remainder to avoid ragged batches
        total_length = (total_length // cfg.max_seq_length) * cfg.max_seq_length
        if total_length == 0:
            return {"input_ids": [], "attention_mask": []}

        concatenated = concatenated[:total_length]

        # split into chunks
        input_ids = [
            concatenated[i : i + cfg.max_seq_length]
            for i in range(0, total_length, cfg.max_seq_length)
        ]
        attention_mask = [
            [1] * cfg.max_seq_length for _ in range(len(input_ids))
        ]

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    lm_train = tokenized_train.map(
        group_texts,
        batched=True,
        desc="grouping train into chunks",
    )
    lm_val = tokenized_val.map(
        group_texts,
        batched=True,
        desc="grouping val into chunks",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return lm_train, lm_val, collator
