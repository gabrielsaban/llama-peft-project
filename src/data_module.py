# src/data_module.py

from dataclasses import dataclass
from typing import Optional, Tuple

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling,
)


@dataclass
class LMDataConfig:
    dataset_name: str               # e.g. "eurlex_text_lm"
    max_seq_length: int = 1024
    train_subset: Optional[int] = None
    val_subset: Optional[int] = None
    seed: int = 42


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # llama tokenizer usually has no pad token; use eos as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal lm, not mlm
    )

    return lm_train, lm_val, collator
