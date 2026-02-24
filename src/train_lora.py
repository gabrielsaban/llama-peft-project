# src/train_lora.py

import argparse
import gc
import json
import math
import time
import yaml
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
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
    attn_implementation: Optional[str] = None,
):
    common_kwargs: dict[str, Any] = {"device_map": "auto"}
    if attn_implementation:
        common_kwargs["attn_implementation"] = attn_implementation

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
            **common_kwargs,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            **common_kwargs,
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


def _json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, int):
            safe[key] = value
        elif isinstance(value, float):
            safe[key] = value if math.isfinite(value) else None
        elif isinstance(value, str):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def _write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_text_artifact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")


def _cuda_memory_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return snapshot

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)

    snapshot.update(
        {
            "device_index": int(device_idx),
            "device_name": torch.cuda.get_device_name(device_idx),
            "total_bytes": int(total_bytes),
            "free_bytes": int(free_bytes),
            "used_bytes_driver_view": int(total_bytes - free_bytes),
            "allocated_bytes": int(torch.cuda.memory_allocated(device_idx)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device_idx)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device_idx)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device_idx)),
            "device_total_memory_bytes": int(props.total_memory),
        }
    )
    return snapshot


def _log_cuda_memory_snapshot(tag: str, output_dir: Path) -> None:
    snapshot = _cuda_memory_snapshot()
    snapshot["tag"] = tag
    print(f"[cuda-mem] {json.dumps(snapshot, sort_keys=True)}")
    _write_json_artifact(output_dir / f"cuda_memory_{tag}.json", snapshot)


def _configure_sdp_backends(hw_cfg: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        print("[info] CUDA not available; skipping explicit SDPA backend configuration")
        return

    # Make backend selection explicit for reproducibility/debugging.
    flash_sdp = bool(hw_cfg.get("flash_sdp", True))
    mem_efficient_sdp = bool(hw_cfg.get("mem_efficient_sdp", False))
    math_sdp = bool(hw_cfg.get("math_sdp", False))

    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        print("[warn] torch.backends.cuda unavailable; cannot configure SDPA backends")
        return

    if hasattr(cuda_backends, "enable_flash_sdp"):
        cuda_backends.enable_flash_sdp(flash_sdp)
    if hasattr(cuda_backends, "enable_mem_efficient_sdp"):
        cuda_backends.enable_mem_efficient_sdp(mem_efficient_sdp)
    if hasattr(cuda_backends, "enable_math_sdp"):
        cuda_backends.enable_math_sdp(math_sdp)

    flash_state = (
        cuda_backends.flash_sdp_enabled() if hasattr(cuda_backends, "flash_sdp_enabled") else None
    )
    mem_state = (
        cuda_backends.mem_efficient_sdp_enabled()
        if hasattr(cuda_backends, "mem_efficient_sdp_enabled")
        else None
    )
    math_state = (
        cuda_backends.math_sdp_enabled() if hasattr(cuda_backends, "math_sdp_enabled") else None
    )
    print(
        "[info] SDPA backends configured: "
        f"flash={flash_state}, mem_efficient={mem_state}, math={math_state}"
    )


class TrainStepTimeTracker:
    def __init__(self) -> None:
        self._step_start_t: Optional[float] = None
        self._interval_step_durations_s: list[float] = []

    def reset(self) -> None:
        self._step_start_t = None
        self._interval_step_durations_s.clear()

    def on_step_begin(self) -> None:
        self._step_start_t = time.perf_counter()

    def on_step_end(self) -> None:
        if self._step_start_t is None:
            return
        self._interval_step_durations_s.append(time.perf_counter() - self._step_start_t)
        self._step_start_t = None

    def consume_mean_interval_step_time_s(self) -> float:
        if not self._interval_step_durations_s:
            return float("nan")
        mean_s = sum(self._interval_step_durations_s) / len(self._interval_step_durations_s)
        self._interval_step_durations_s.clear()
        return mean_s


class TrainStepTimeCallback(TrainerCallback):
    def __init__(self, tracker: TrainStepTimeTracker) -> None:
        self.tracker = tracker

    def on_step_begin(self, args, state, control, **kwargs):
        self.tracker.on_step_begin()

    def on_step_end(self, args, state, control, **kwargs):
        self.tracker.on_step_end()


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
        self._train_step_time_tracker = TrainStepTimeTracker()
        self.add_callback(TrainStepTimeCallback(self._train_step_time_tracker))

    def reset_protocol_trackers(self) -> None:
        self._train_step_time_tracker.reset()
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device_idx)

    @staticmethod
    def _peak_vram_metrics(metric_key_prefix: str) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {}

        peak_allocated = 0
        peak_reserved = 0
        for device_idx in range(torch.cuda.device_count()):
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device_idx))
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device_idx))

        gb = 1024**3
        return {
            f"{metric_key_prefix}_peak_vram_allocated_gb": peak_allocated / gb,
            f"{metric_key_prefix}_peak_vram_reserved_gb": peak_reserved / gb,
        }

    def _interval_step_time_metrics(self, metric_key_prefix: str) -> dict[str, float]:
        return {
            f"{metric_key_prefix}_mean_train_step_time_s": self._train_step_time_tracker.consume_mean_interval_step_time_s()
        }

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

        metrics.update(self._interval_step_time_metrics(metric_key_prefix))
        metrics.update(self._peak_vram_metrics(metric_key_prefix))
        # Trainer.evaluate logs each sub-eval separately; emit a combined log with
        # derived metrics so they are persisted in trainer_state log history.
        self.log(metrics)
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
    attn_implementation = hw_cfg.get("attn_implementation")

    _configure_sdp_backends(hw_cfg)
    if attn_implementation:
        print(f"[info] attn_implementation requested: {attn_implementation}")

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
        attn_implementation=attn_implementation,
    )
    model = apply_lora(model, model_cfg["lora"])

    # KV cache is useful for generation, not training; disable to reduce memory spikes.
    if hasattr(model, "config"):
        model.config.use_cache = False
        print("[info] forced model.config.use_cache=False for training")

    # gradient checkpointing, if requested
    if hw_cfg.get("gradient_checkpointing", False):
        gc_kwargs = {"use_reentrant": bool(hw_cfg.get("gc_use_reentrant", False))}
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        except TypeError:
            # Older transformers versions may not accept gradient_checkpointing_kwargs.
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            print("[info] enabled input grads for LoRA + gradient checkpointing")
        else:
            print("[warn] model has no enable_input_require_grads(); GC may fail with frozen base")

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # training args
    lr = float(train_cfg["learning_rate"])
    warmup_ratio = float(train_cfg["warmup_ratio"])
    weight_decay = float(train_cfg["weight_decay"])
    max_steps = train_cfg.get("max_steps")
    save_steps = int(train_cfg["save_steps"])
    eval_steps = int(train_cfg["eval_steps"])
    if save_steps % eval_steps != 0:
        raise ValueError(
            "Protocol checkpointing requires save_steps to be a multiple of eval_steps "
            f"(got save_steps={save_steps}, eval_steps={eval_steps})."
        )

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
        save_strategy="steps",
        save_steps=save_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_perplexity",
        greater_is_better=False,
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

    # Baseline (pre-training) eval artifact for protocol zero-shot reference.
    baseline_metrics = trainer.evaluate(metric_key_prefix="baseline")
    _write_json_artifact(
        output_dir / "baseline_eval_metrics.json",
        {
            "experiment_name": exp_name,
            "global_step": int(trainer.state.global_step),
            "metrics": _json_safe_metrics(baseline_metrics),
        },
    )
    _log_cuda_memory_snapshot("post_baseline_eval_preclear", output_dir)

    # Baseline eval can leave allocator cache populated; clear unused blocks before training.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_cuda_memory_snapshot("pre_train", output_dir)

    # Reset interval timers + CUDA peak memory so run metrics reflect training/eval run only.
    trainer.reset_protocol_trackers()
    try:
        train_result = trainer.train()
    except torch.cuda.OutOfMemoryError:
        _log_cuda_memory_snapshot("oom_exception", output_dir)
        if torch.cuda.is_available():
            try:
                oom_mem_summary = torch.cuda.memory_summary(abbreviated=True)
                print("[cuda-mem-summary][oom]\n" + oom_mem_summary)
                _write_text_artifact(output_dir / "cuda_memory_oom_summary.txt", oom_mem_summary)
            except Exception as mem_summary_err:
                print(f"[warn] failed to capture cuda memory_summary after OOM: {mem_summary_err}")
        raise

    _write_json_artifact(
        output_dir / "training_summary.json",
        {
            "experiment_name": exp_name,
            "global_step": int(trainer.state.global_step),
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": (
                float(trainer.state.best_metric)
                if trainer.state.best_metric is not None
                else None
            ),
            "metric_for_best_model": training_args.metric_for_best_model,
            "train_metrics": _json_safe_metrics(getattr(train_result, "metrics", {})),
        },
    )

    # final save
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))


if __name__ == "__main__":
    main()
