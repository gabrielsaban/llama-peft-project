# src/train_lora.py

import argparse
import math
import yaml
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .data_module import (
    LMDataConfig,
    get_domain_corpus_lm_datasets,
    get_eurlex_text_lm_datasets,
    load_tokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to yaml config file",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_torch_dtype(dtype_str: str):
    if dtype_str.lower() in ["bfloat16", "bf16"]:
        return torch.bfloat16
    if dtype_str.lower() in ["float16", "fp16"]:
        return torch.float16
    if dtype_str.lower() in ["float32", "fp32"]:
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype_str}")


def load_base_model(
    model_name: str,
    dtype: torch.dtype,
    use_4bit: bool,
):
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    return model


def apply_lora(model, lora_cfg_dict: dict):
    lora_cfg = LoraConfig(
        r=lora_cfg_dict["r"],
        lora_alpha=lora_cfg_dict["alpha"],
        lora_dropout=lora_cfg_dict["dropout"],
        bias="none",
        target_modules=lora_cfg_dict["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


class StratifiedEvalTrainer(Trainer):
    def __init__(
        self,
        *args,
        eval_dataset_layer_a=None,
        eval_dataset_layer_b=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_dataset_layer_a = eval_dataset_layer_a
        self.eval_dataset_layer_b = eval_dataset_layer_b

    @staticmethod
    def _add_perplexity(metrics: dict, prefix: str) -> None:
        key = f"{prefix}_loss"
        if key in metrics:
            loss = float(metrics[key])
            if not math.isfinite(loss):
                metrics[f"{prefix}_perplexity"] = float("nan")
                return
            try:
                metrics[f"{prefix}_perplexity"] = math.exp(loss)
            except OverflowError:
                metrics[f"{prefix}_perplexity"] = float("inf")

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        metrics = Trainer.evaluate(
            self,
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self._add_perplexity(metrics, metric_key_prefix)

        if self.eval_dataset_layer_a is not None:
            metrics_a = Trainer.evaluate(
                self,
                eval_dataset=self.eval_dataset_layer_a,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}_layer_a",
            )
            self._add_perplexity(metrics_a, f"{metric_key_prefix}_layer_a")
            metrics.update(metrics_a)

        if self.eval_dataset_layer_b is not None:
            metrics_b = Trainer.evaluate(
                self,
                eval_dataset=self.eval_dataset_layer_b,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}_layer_b",
            )
            self._add_perplexity(metrics_b, f"{metric_key_prefix}_layer_b")
            metrics.update(metrics_b)

        return metrics


def get_datasets_and_collator(
    data_cfg: dict,
    model_name: str,
    seed: int,
):
    lm_cfg = LMDataConfig(
        dataset_name=data_cfg["type"],
        max_seq_length=data_cfg["max_seq_length"],
        train_subset=data_cfg.get("train_subset"),
        val_subset=data_cfg.get("val_subset"),
        corpus_dir=data_cfg.get("corpus_dir"),
        split_json=data_cfg.get("split_json"),
        seed=seed,
    )

    tokenizer = load_tokenizer(model_name)
    val_a_ds: Optional[torch.utils.data.Dataset] = None
    val_b_ds: Optional[torch.utils.data.Dataset] = None

    if data_cfg["type"] == "eurlex_text_lm":
        train_ds, val_ds, collator = get_eurlex_text_lm_datasets(tokenizer, lm_cfg)
    elif data_cfg["type"] == "domain_corpus_lm":
        train_ds, val_ds, val_a_ds, val_b_ds, collator = get_domain_corpus_lm_datasets(
            tokenizer, lm_cfg
        )
    else:
        raise ValueError(f"unknown data.type: {data_cfg['type']}")

    return tokenizer, train_ds, val_ds, val_a_ds, val_b_ds, collator


def main():
    args = parse_args()
    cfg = load_config(args.config)

    exp_name = cfg["experiment_name"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    hw_cfg = cfg["hardware"]

    set_seed(train_cfg["seed"])

    base_model_name = model_cfg["base_model"]
    dtype = get_torch_dtype(model_cfg["dtype"])
    use_4bit = hw_cfg.get("use_4bit", False)

    # data
    tokenizer, train_ds, val_ds, val_a_ds, val_b_ds, collator = get_datasets_and_collator(
        data_cfg,
        base_model_name,
        seed=train_cfg["seed"],
    )

    # model
    model = load_base_model(
        model_name=base_model_name,
        dtype=dtype,
        use_4bit=use_4bit,
    )
    model = apply_lora(model, model_cfg["lora"])

    # gradient checkpointing, if requested
    if hw_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # training args
    lr = float(train_cfg["learning_rate"])
    warmup_ratio = float(train_cfg["warmup_ratio"])
    weight_decay = float(train_cfg["weight_decay"])
    max_steps = train_cfg.get("max_steps")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=exp_name,
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1.0)),
        max_steps=int(max_steps) if max_steps is not None else -1,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        evaluation_strategy="steps",
        eval_steps=int(train_cfg["eval_steps"]),
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to=[],
        log_level="info",
    )


    trainer = StratifiedEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        eval_dataset_layer_a=val_a_ds,
        eval_dataset_layer_b=val_b_ds,
        data_collator=collator,
    )

    trainer.train()

    # final save
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))


if __name__ == "__main__":
    main()
