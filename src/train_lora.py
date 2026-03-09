# src/train_lora.py

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .artifacts import (
    json_safe_metrics,
    write_eval_summary_csv,
    write_json_artifact,
    write_jsonl_rows,
    write_text_artifact,
    write_yaml_artifact,
)
from .callbacks_metrics import (
    TrainStepTimeCallback,
    TrainStepTimeTracker,
    build_timing_summary,
    collect_finite,
)
from .callbacks_stability import build_stability_artifacts
from .data_module import (
    LMDataConfig,
    get_domain_corpus_lm_datasets,
    get_eurlex_text_lm_datasets,
    load_tokenizer,
)
from .provenance import build_environment_snapshot, iso_utc_now

_RUN_CONTEXT: dict[str, Any] = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to yaml config file")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="run held-out baseline evaluation and exit without training",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_torch_dtype(dtype_str: str):
    dtype_norm = dtype_str.lower()
    if dtype_norm in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_norm in {"float16", "fp16"}:
        return torch.float16
    if dtype_norm in {"float32", "fp32"}:
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


def _log_cuda_memory_snapshot(tag: str, output_dir: Path) -> dict[str, Any]:
    snapshot = _cuda_memory_snapshot()
    snapshot["tag"] = tag
    print(f"[cuda-mem] {json.dumps(snapshot, sort_keys=True)}")
    write_json_artifact(output_dir / f"cuda_memory_{tag}.json", snapshot)
    return snapshot


def _configure_sdp_backends(hw_cfg: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        print("[info] CUDA not available; skipping explicit SDPA backend configuration")
        return

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
            f"{metric_key_prefix}_peak_vram_allocated_gb_since_last_reset": peak_allocated / gb,
            f"{metric_key_prefix}_peak_vram_reserved_gb_since_last_reset": peak_reserved / gb,
        }

    def _interval_step_time_metrics(self, metric_key_prefix: str) -> dict[str, Any]:
        stats = self._train_step_time_tracker.consume_interval_stats()
        return {
            f"{metric_key_prefix}_mean_train_step_time_s": stats["mean_s"],
            f"{metric_key_prefix}_p50_train_step_time_s": stats["p50_s"],
            f"{metric_key_prefix}_p95_train_step_time_s": stats["p95_s"],
            f"{metric_key_prefix}_std_train_step_time_s": stats["std_s"],
            f"{metric_key_prefix}_num_train_step_time_samples": stats["num_samples"],
        }

    def train_peak_vram_metrics(self) -> dict[str, Optional[float]]:
        return self._train_step_time_tracker.train_peak_vram_gb()

    @staticmethod
    def _add_perplexity(metrics: dict[str, Any], prefix: str) -> None:
        key = f"{prefix}_loss"
        if key not in metrics:
            return
        loss = float(metrics[key])
        if not math.isfinite(loss):
            metrics[f"{prefix}_perplexity"] = float("nan")
            return
        try:
            metrics[f"{prefix}_perplexity"] = math.exp(loss)
        except OverflowError:
            metrics[f"{prefix}_perplexity"] = float("inf")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
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
        self.log(metrics)
        return metrics


def get_datasets_and_collator(
    data_cfg: dict[str, Any],
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


def _extract_doc_counts_from_split_json(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}

    split_path = Path(path)
    if not split_path.exists():
        return {}

    try:
        with split_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}

    out: dict[str, Any] = {}
    out["split_seed"] = obj.get("seed")
    out["stratification"] = obj.get("stratification")
    out["val_ratio"] = obj.get("val_ratio")
    out["tokens_by_split"] = obj.get("tokens_by_split")
    counts = obj.get("counts")
    if isinstance(counts, dict):
        out["counts"] = counts
    return out


def _build_dataset_summary(data_cfg: dict[str, Any], train_ds, val_ds, val_a_ds, val_b_ds) -> dict[str, Any]:
    split_meta = _extract_doc_counts_from_split_json(data_cfg.get("split_json"))
    counts = split_meta.get("counts", {})
    return {
        "data_type": data_cfg.get("type"),
        "corpus_dir": data_cfg.get("corpus_dir"),
        "split_json": data_cfg.get("split_json"),
        "max_seq_length": data_cfg.get("max_seq_length"),
        "train_chunks_total": len(train_ds) if train_ds is not None else None,
        "val_chunks_total": len(val_ds) if val_ds is not None else None,
        "val_chunks_layer_a": len(val_a_ds) if val_a_ds is not None else None,
        "val_chunks_layer_b": len(val_b_ds) if val_b_ds is not None else None,
        "train_docs_total": (
            (counts.get("layer_a", {}).get("train", 0) + counts.get("layer_b", {}).get("train", 0))
            if counts
            else None
        ),
        "val_docs_total": (
            (counts.get("layer_a", {}).get("val", 0) + counts.get("layer_b", {}).get("val", 0))
            if counts
            else None
        ),
        "val_docs_layer_a": counts.get("layer_a", {}).get("val") if counts else None,
        "val_docs_layer_b": counts.get("layer_b", {}).get("val") if counts else None,
        "split_seed": split_meta.get("split_seed"),
        "stratification": split_meta.get("stratification"),
        "val_ratio": split_meta.get("val_ratio"),
        "tokens_by_split": split_meta.get("tokens_by_split"),
    }


def _build_budget_summary(train_cfg: dict[str, Any], data_cfg: dict[str, Any], max_steps_completed: int) -> dict[str, Any]:
    per_device_train_batch_size = int(train_cfg["per_device_train_batch_size"])
    gradient_accumulation_steps = int(train_cfg["gradient_accumulation_steps"])
    effective_batch_size_sequences = per_device_train_batch_size * gradient_accumulation_steps
    max_seq_length = int(data_cfg["max_seq_length"])
    tokens_per_optimizer_step = effective_batch_size_sequences * max_seq_length
    max_steps_planned = int(train_cfg["max_steps"]) if train_cfg.get("max_steps") is not None else None
    planned_tokens_processed = (
        max_steps_planned * tokens_per_optimizer_step if max_steps_planned is not None else None
    )
    realized_tokens_processed = max_steps_completed * tokens_per_optimizer_step

    budget_basis = {
        "max_steps_planned": max_steps_planned,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_seq_length": max_seq_length,
        "eval_steps": int(train_cfg["eval_steps"]),
        "save_steps": int(train_cfg["save_steps"]),
    }
    budget_match_key = hashlib.sha256(
        json.dumps(budget_basis, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    return {
        "max_steps_planned": max_steps_planned,
        "max_steps_completed": max_steps_completed,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size_sequences": effective_batch_size_sequences,
        "max_seq_length": max_seq_length,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "planned_tokens_processed": planned_tokens_processed,
        "realized_tokens_processed": realized_tokens_processed,
        "eval_steps": int(train_cfg["eval_steps"]),
        "save_steps": int(train_cfg["save_steps"]),
        "seed": int(train_cfg["seed"]),
        "budget_match_key": budget_match_key,
        "budget_basis": budget_basis,
    }


def _event_type_from_history_item(item: dict[str, Any]) -> str:
    keys = set(item.keys())
    if any(k.startswith("baseline_") for k in keys):
        return "baseline_eval"
    if any(k.startswith("eval_") for k in keys):
        return "eval"
    if "loss" in keys or "grad_norm" in keys:
        return "train_log"
    return "system"


def _build_metrics_history_rows(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in log_history:
        rows.append(
            {
                "step": item.get("step"),
                "epoch": item.get("epoch"),
                "event_type": _event_type_from_history_item(item),
                "timestamp_utc": iso_utc_now(),
                "metrics": json_safe_metrics(item),
            }
        )
    return rows


def _extract_eval_rows(run_id: str, baseline_metrics: dict[str, Any], log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "run_id": run_id,
            "step": 0,
            "phase": "baseline",
            "eval_loss": baseline_metrics.get("baseline_loss"),
            "eval_perplexity": baseline_metrics.get("baseline_perplexity"),
            "eval_layer_a_loss": baseline_metrics.get("baseline_layer_a_loss"),
            "eval_layer_a_perplexity": baseline_metrics.get("baseline_layer_a_perplexity"),
            "eval_layer_b_loss": baseline_metrics.get("baseline_layer_b_loss"),
            "eval_layer_b_perplexity": baseline_metrics.get("baseline_layer_b_perplexity"),
            "train_step_time_mean_s_window": baseline_metrics.get("baseline_mean_train_step_time_s"),
            "train_step_time_p50_s_window": None,
            "train_step_time_p95_s_window": None,
            "train_step_time_std_s_window": None,
            "train_step_time_num_samples_window": None,
            "eval_peak_vram_allocated_gb": baseline_metrics.get("baseline_peak_vram_allocated_gb"),
            "eval_peak_vram_reserved_gb": baseline_metrics.get("baseline_peak_vram_reserved_gb"),
            "eval_peak_vram_allocated_gb_since_last_reset": baseline_metrics.get(
                "baseline_peak_vram_allocated_gb_since_last_reset",
                baseline_metrics.get("baseline_peak_vram_allocated_gb"),
            ),
            "eval_peak_vram_reserved_gb_since_last_reset": baseline_metrics.get(
                "baseline_peak_vram_reserved_gb_since_last_reset",
                baseline_metrics.get("baseline_peak_vram_reserved_gb"),
            ),
        }
    ]

    for item in log_history:
        if "eval_perplexity" not in item:
            continue
        rows.append(
            {
                "run_id": run_id,
                "step": item.get("step"),
                "phase": "eval",
                "eval_loss": item.get("eval_loss"),
                "eval_perplexity": item.get("eval_perplexity"),
                "eval_layer_a_loss": item.get("eval_layer_a_loss"),
                "eval_layer_a_perplexity": item.get("eval_layer_a_perplexity"),
                "eval_layer_b_loss": item.get("eval_layer_b_loss"),
                "eval_layer_b_perplexity": item.get("eval_layer_b_perplexity"),
                "train_step_time_mean_s_window": item.get("eval_mean_train_step_time_s"),
                "train_step_time_p50_s_window": item.get("eval_p50_train_step_time_s"),
                "train_step_time_p95_s_window": item.get("eval_p95_train_step_time_s"),
                "train_step_time_std_s_window": item.get("eval_std_train_step_time_s"),
                "train_step_time_num_samples_window": item.get("eval_num_train_step_time_samples"),
                "eval_peak_vram_allocated_gb": item.get("eval_peak_vram_allocated_gb"),
                "eval_peak_vram_reserved_gb": item.get("eval_peak_vram_reserved_gb"),
                "eval_peak_vram_allocated_gb_since_last_reset": item.get(
                    "eval_peak_vram_allocated_gb_since_last_reset",
                    item.get("eval_peak_vram_allocated_gb"),
                ),
                "eval_peak_vram_reserved_gb_since_last_reset": item.get(
                    "eval_peak_vram_reserved_gb_since_last_reset",
                    item.get("eval_peak_vram_reserved_gb"),
                ),
            }
        )
    return rows


def _best_eval_row_from_eval_rows(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    eval_rows = [r for r in rows if r.get("phase") == "eval" and r.get("eval_perplexity") is not None]
    if not eval_rows:
        return None
    return min(eval_rows, key=lambda r: float(r["eval_perplexity"]))


def _build_memory_summary(
    baseline_metrics: dict[str, Any],
    eval_rows: list[dict[str, Any]],
    run_peak_snapshot: Optional[dict[str, Any]],
    train_peak_metrics: Optional[dict[str, Optional[float]]],
) -> dict[str, Any]:
    eval_peak_alloc = collect_finite(
        [
            r.get("eval_peak_vram_allocated_gb_since_last_reset", r.get("eval_peak_vram_allocated_gb"))
            for r in eval_rows
            if r.get("phase") == "eval"
        ]
    )
    eval_peak_resv = collect_finite(
        [
            r.get("eval_peak_vram_reserved_gb_since_last_reset", r.get("eval_peak_vram_reserved_gb"))
            for r in eval_rows
            if r.get("phase") == "eval"
        ]
    )

    run_peak_alloc_gb = None
    run_peak_resv_gb = None
    if run_peak_snapshot and run_peak_snapshot.get("cuda_available"):
        gb = 1024**3
        max_alloc_bytes = run_peak_snapshot.get("max_allocated_bytes")
        max_resv_bytes = run_peak_snapshot.get("max_reserved_bytes")
        if isinstance(max_alloc_bytes, int):
            run_peak_alloc_gb = max_alloc_bytes / gb
        if isinstance(max_resv_bytes, int):
            run_peak_resv_gb = max_resv_bytes / gb

    train_peak_alloc = None
    train_peak_resv = None
    if train_peak_metrics:
        train_peak_alloc = train_peak_metrics.get("train_peak_vram_allocated_gb")
        train_peak_resv = train_peak_metrics.get("train_peak_vram_reserved_gb")

    return {
        "baseline_eval_peak_vram_allocated_gb": baseline_metrics.get("baseline_peak_vram_allocated_gb"),
        "baseline_eval_peak_vram_reserved_gb": baseline_metrics.get("baseline_peak_vram_reserved_gb"),
        "baseline_eval_peak_vram_allocated_gb_since_last_reset": baseline_metrics.get(
            "baseline_peak_vram_allocated_gb_since_last_reset",
            baseline_metrics.get("baseline_peak_vram_allocated_gb"),
        ),
        "baseline_eval_peak_vram_reserved_gb_since_last_reset": baseline_metrics.get(
            "baseline_peak_vram_reserved_gb_since_last_reset",
            baseline_metrics.get("baseline_peak_vram_reserved_gb"),
        ),
        "train_peak_vram_allocated_gb": train_peak_alloc,
        "train_peak_vram_reserved_gb": train_peak_resv,
        "max_eval_peak_vram_allocated_gb_since_last_reset": max(eval_peak_alloc) if eval_peak_alloc else None,
        "max_eval_peak_vram_reserved_gb_since_last_reset": max(eval_peak_resv) if eval_peak_resv else None,
        "run_peak_vram_allocated_gb": run_peak_alloc_gb,
        "run_peak_vram_reserved_gb": run_peak_resv_gb,
        "memory_measurement_notes": (
            "train_peak tracked from per-step allocator snapshots; eval peaks are since-last-reset metrics."
        ),
    }


def _build_run_manifest(
    run_id: str,
    exp_name: str,
    status: str,
    baseline_only: bool,
    variant: str,
    protocol_version: str,
    logging_schema_version: str,
    start_time_utc: str,
    end_time_utc: str,
    duration_s: float,
    paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_name": exp_name,
        "variant": variant,
        "status": status,
        "baseline_only": baseline_only,
        "protocol_version": protocol_version,
        "logging_schema_version": logging_schema_version,
        "start_time_utc": start_time_utc,
        "end_time_utc": end_time_utc,
        "duration_s": duration_s,
        "paths": paths,
    }


def _write_failed_manifest_from_context(exc: Exception) -> None:
    ctx = _RUN_CONTEXT
    if not ctx:
        return
    if ctx.get("status") in {"completed", "oom"}:
        return
    reports_dir = ctx.get("reports_dir")
    run_id = ctx.get("run_id")
    exp_name = ctx.get("exp_name")
    if not reports_dir or not run_id or not exp_name:
        return

    duration_s = max(0.0, time.time() - float(ctx.get("run_start_wall", time.time())))
    termination_reason = f"failed:{type(exc).__name__}"
    try:
        write_json_artifact(
            Path(reports_dir) / "run_manifest.json",
            _build_run_manifest(
                run_id=run_id,
                exp_name=exp_name,
                status="failed",
                baseline_only=bool(ctx.get("baseline_only", False)),
                variant=str(ctx.get("variant", "unknown")),
                protocol_version=str(ctx.get("protocol_version", "protocol_v1")),
                logging_schema_version=str(ctx.get("logging_schema_version", "logging_v1")),
                start_time_utc=str(ctx.get("start_time_utc", iso_utc_now())),
                end_time_utc=iso_utc_now(),
                duration_s=duration_s,
                paths=dict(ctx.get("paths_common", {})),
            ),
        )
        write_json_artifact(
            Path(reports_dir) / "stability_summary.json",
            {
                "num_nan_loss_events": 0,
                "num_inf_loss_events": 0,
                "num_nonfinite_grad_norm_events": 0,
                "num_oom_events": 0,
                "optimizer_reset_count": 0,
                "terminated_early": True,
                "termination_reason": termination_reason,
                "last_completed_step": 0,
                "max_grad_norm": None,
                "min_grad_norm": None,
                "stability_rule_config": {
                    "finite_checks": ["loss", "grad_norm"],
                    "event_scope": [
                        "nan_loss",
                        "inf_loss",
                        "nonfinite_grad_norm",
                        "oom",
                        "early_termination",
                    ],
                },
            },
        )
    except Exception as manifest_err:
        print(f"[warn] failed to write failed run manifest: {manifest_err}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_start_ts = iso_utc_now()
    run_start_wall = time.time()
    _RUN_CONTEXT.clear()
    _RUN_CONTEXT.update(
        {
            "status": "starting",
            "start_time_utc": run_start_ts,
            "run_start_wall": run_start_wall,
            "baseline_only": bool(args.baseline_only),
        }
    )

    exp_name = cfg["experiment_name"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    hw_cfg = cfg["hardware"]
    exp_meta = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}

    set_seed(int(train_cfg["seed"]))

    base_model_name = model_cfg["base_model"]
    dtype = get_torch_dtype(model_cfg["dtype"])
    use_4bit = bool(hw_cfg.get("use_4bit", False))
    attn_implementation = hw_cfg.get("attn_implementation")
    variant = exp_meta.get("variant") or ("qlora" if use_4bit else "lora")
    protocol_version = exp_meta.get("protocol_version", "protocol_v1")
    logging_schema_version = exp_meta.get("logging_schema_version", "logging_v1")

    _configure_sdp_backends(hw_cfg)
    if attn_implementation:
        print(f"[info] attn_implementation requested: {attn_implementation}")

    tokenizer, train_ds, val_ds, val_a_ds, val_b_ds, collator = get_datasets_and_collator(
        data_cfg,
        base_model_name,
        seed=int(train_cfg["seed"]),
    )

    model = load_base_model(
        model_name=base_model_name,
        dtype=dtype,
        use_4bit=use_4bit,
        attn_implementation=attn_implementation,
    )
    if args.baseline_only:
        print("[info] baseline-only mode enabled: evaluating unadapted base model")

    if hasattr(model, "config"):
        model.config.use_cache = False
        print("[info] forced model.config.use_cache=False for training")

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{exp_name}__{int(time.time())}"

    paths_common = {
        "output_dir": str(output_dir),
        "reports_dir": str(reports_dir),
        "config_path": str(args.config),
        "run_log": str(output_dir / "raw" / "run.log"),
        "baseline_eval_metrics": str(output_dir / "baseline_eval_metrics.json"),
        "training_summary": str(output_dir / "training_summary.json"),
        "final_adapter_dir": str(output_dir / "final_adapter"),
    }
    _RUN_CONTEXT.update(
        {
            "status": "running",
            "run_id": run_id,
            "exp_name": exp_name,
            "variant": variant,
            "protocol_version": protocol_version,
            "logging_schema_version": logging_schema_version,
            "reports_dir": str(reports_dir),
            "paths_common": paths_common,
        }
    )

    write_yaml_artifact(reports_dir / "resolved_config.yaml", cfg)
    write_json_artifact(reports_dir / "environment.json", build_environment_snapshot())
    write_json_artifact(
        reports_dir / "dataset_summary.json",
        _build_dataset_summary(data_cfg, train_ds, val_ds, val_a_ds, val_b_ds),
    )
    write_json_artifact(
        reports_dir / "run_manifest.json",
        _build_run_manifest(
            run_id=run_id,
            exp_name=exp_name,
            status="running",
            baseline_only=bool(args.baseline_only),
            variant=variant,
            protocol_version=protocol_version,
            logging_schema_version=logging_schema_version,
            start_time_utc=run_start_ts,
            end_time_utc=run_start_ts,
            duration_s=0.0,
            paths=paths_common,
        ),
    )

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

    baseline_trainer = StratifiedEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        eval_dataset_layer_a=val_a_ds,
        eval_dataset_layer_b=val_b_ds,
        data_collator=collator,
    )

    baseline_metrics = baseline_trainer.evaluate(metric_key_prefix="baseline")
    write_json_artifact(
        output_dir / "baseline_eval_metrics.json",
        {
            "experiment_name": exp_name,
            "run_id": run_id,
            "variant": variant,
            "global_step": int(baseline_trainer.state.global_step),
            "baseline_only": bool(args.baseline_only),
            "metrics": json_safe_metrics(baseline_metrics),
        },
    )
    baseline_run_peak_snapshot = _log_cuda_memory_snapshot("post_baseline_eval_preclear", output_dir)

    if args.baseline_only:
        eval_rows = _extract_eval_rows(run_id, baseline_metrics, baseline_trainer.state.log_history)
        write_eval_summary_csv(reports_dir / "eval_summary.csv", eval_rows)
        write_jsonl_rows(
            reports_dir / "metrics_history.jsonl",
            _build_metrics_history_rows(baseline_trainer.state.log_history),
        )
        write_json_artifact(
            reports_dir / "budget_summary.json",
            _build_budget_summary(train_cfg, data_cfg, max_steps_completed=0),
        )
        write_json_artifact(
            reports_dir / "timing_summary.json",
            build_timing_summary(train_metrics={}, eval_rows=eval_rows, baseline_only=True),
        )
        write_json_artifact(
            reports_dir / "memory_summary.json",
            _build_memory_summary(
                baseline_metrics=baseline_metrics,
                eval_rows=eval_rows,
                run_peak_snapshot=baseline_run_peak_snapshot,
                train_peak_metrics=None,
            ),
        )
        stability_summary, stability_events = build_stability_artifacts(baseline_trainer.state.log_history)
        write_json_artifact(reports_dir / "stability_summary.json", stability_summary)
        write_jsonl_rows(reports_dir / "stability_events.jsonl", stability_events)

        write_json_artifact(
            output_dir / "baseline_only_summary.json",
            {
                "experiment_name": exp_name,
                "run_id": run_id,
                "variant": variant,
                "global_step": int(baseline_trainer.state.global_step),
                "mode": "baseline_only",
                "metrics": json_safe_metrics(baseline_metrics),
            },
        )
        write_json_artifact(
            reports_dir / "final_metrics.json",
            {
                "run_id": run_id,
                "variant": variant,
                "mode": "baseline_only",
                "selection_metric": "eval_perplexity",
                "baseline": {
                    "perplexity_all": baseline_metrics.get("baseline_perplexity"),
                    "perplexity_layer_a": baseline_metrics.get("baseline_layer_a_perplexity"),
                    "perplexity_layer_b": baseline_metrics.get("baseline_layer_b_perplexity"),
                },
            },
        )
        write_json_artifact(
            reports_dir / "comparison_row.json",
            {
                "run_id": run_id,
                "experiment_name": exp_name,
                "variant": variant,
                "mode": "baseline_only",
                "base_model": base_model_name,
                "use_4bit": bool(use_4bit),
                "seed": int(train_cfg["seed"]),
                "baseline_ppl_all": baseline_metrics.get("baseline_perplexity"),
                "baseline_ppl_layer_a": baseline_metrics.get("baseline_layer_a_perplexity"),
                "baseline_ppl_layer_b": baseline_metrics.get("baseline_layer_b_perplexity"),
                "num_nonfinite_events": (
                    stability_summary.get("num_nan_loss_events", 0)
                    + stability_summary.get("num_inf_loss_events", 0)
                    + stability_summary.get("num_nonfinite_grad_norm_events", 0)
                ),
                "status": "completed",
            },
        )

        run_end_ts = iso_utc_now()
        write_json_artifact(
            reports_dir / "run_manifest.json",
            _build_run_manifest(
                run_id=run_id,
                exp_name=exp_name,
                status="completed",
                baseline_only=True,
                variant=variant,
                protocol_version=protocol_version,
                logging_schema_version=logging_schema_version,
                start_time_utc=run_start_ts,
                end_time_utc=run_end_ts,
                duration_s=time.time() - run_start_wall,
                paths={
                    **paths_common,
                    "baseline_only_summary": str(output_dir / "baseline_only_summary.json"),
                    "eval_summary_csv": str(reports_dir / "eval_summary.csv"),
                    "comparison_row": str(reports_dir / "comparison_row.json"),
                },
            ),
        )
        _RUN_CONTEXT["status"] = "completed"
        print("[info] baseline-only run complete; skipped training and adapter save")
        return

    model = apply_lora(model, model_cfg["lora"])

    if bool(hw_cfg.get("gradient_checkpointing", False)):
        gc_kwargs = {"use_reentrant": bool(hw_cfg.get("gc_use_reentrant", False))}
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            print("[info] enabled input grads for LoRA + gradient checkpointing")
        else:
            print("[warn] model has no enable_input_require_grads(); GC may fail with frozen base")

    trainer = StratifiedEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        eval_dataset_layer_a=val_a_ds,
        eval_dataset_layer_b=val_b_ds,
        data_collator=collator,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_cuda_memory_snapshot("pre_train", output_dir)

    trainer.reset_protocol_trackers()
    try:
        train_result = trainer.train()
    except torch.cuda.OutOfMemoryError:
        _log_cuda_memory_snapshot("oom_exception", output_dir)
        if torch.cuda.is_available():
            try:
                oom_mem_summary = torch.cuda.memory_summary(abbreviated=True)
                print("[cuda-mem-summary][oom]\n" + oom_mem_summary)
                write_text_artifact(output_dir / "cuda_memory_oom_summary.txt", oom_mem_summary)
            except Exception as mem_summary_err:
                print(f"[warn] failed to capture cuda memory_summary after OOM: {mem_summary_err}")

        stability_summary, stability_events = build_stability_artifacts(trainer.state.log_history)
        stability_summary["num_oom_events"] = 1
        stability_summary["terminated_early"] = True
        stability_summary["termination_reason"] = "oom"
        write_json_artifact(reports_dir / "stability_summary.json", stability_summary)
        write_jsonl_rows(reports_dir / "stability_events.jsonl", stability_events)

        write_json_artifact(
            reports_dir / "run_manifest.json",
            _build_run_manifest(
                run_id=run_id,
                exp_name=exp_name,
                status="oom",
                baseline_only=False,
                variant=variant,
                protocol_version=protocol_version,
                logging_schema_version=logging_schema_version,
                start_time_utc=run_start_ts,
                end_time_utc=iso_utc_now(),
                duration_s=time.time() - run_start_wall,
                paths={
                    **paths_common,
                    "oom_snapshot": str(output_dir / "cuda_memory_oom_exception.json"),
                    "oom_summary": str(output_dir / "cuda_memory_oom_summary.txt"),
                },
            ),
        )
        _RUN_CONTEXT["status"] = "oom"
        raise

    write_json_artifact(
        output_dir / "training_summary.json",
        {
            "experiment_name": exp_name,
            "run_id": run_id,
            "variant": variant,
            "global_step": int(trainer.state.global_step),
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": (
                float(trainer.state.best_metric) if trainer.state.best_metric is not None else None
            ),
            "metric_for_best_model": training_args.metric_for_best_model,
            "train_metrics": json_safe_metrics(getattr(train_result, "metrics", {})),
        },
    )

    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))
    post_train_peak_snapshot = _log_cuda_memory_snapshot("post_train", output_dir)

    eval_rows = _extract_eval_rows(run_id, baseline_metrics, trainer.state.log_history)
    write_eval_summary_csv(reports_dir / "eval_summary.csv", eval_rows)
    write_jsonl_rows(reports_dir / "metrics_history.jsonl", _build_metrics_history_rows(trainer.state.log_history))
    max_steps_completed = int(trainer.state.global_step)
    write_json_artifact(
        reports_dir / "budget_summary.json",
        _build_budget_summary(train_cfg, data_cfg, max_steps_completed=max_steps_completed),
    )
    write_json_artifact(
        reports_dir / "timing_summary.json",
        build_timing_summary(
            train_metrics=getattr(train_result, "metrics", {}),
            eval_rows=eval_rows,
            baseline_only=False,
        ),
    )
    write_json_artifact(
        reports_dir / "memory_summary.json",
        _build_memory_summary(
            baseline_metrics=baseline_metrics,
            eval_rows=eval_rows,
            run_peak_snapshot=post_train_peak_snapshot,
            train_peak_metrics=trainer.train_peak_vram_metrics(),
        ),
    )
    stability_summary, stability_events = build_stability_artifacts(trainer.state.log_history)
    write_json_artifact(reports_dir / "stability_summary.json", stability_summary)
    write_jsonl_rows(reports_dir / "stability_events.jsonl", stability_events)

    best_eval_row = _best_eval_row_from_eval_rows(eval_rows)
    final_eval_row = eval_rows[-1] if eval_rows else None
    write_json_artifact(
        reports_dir / "final_metrics.json",
        {
            "run_id": run_id,
            "variant": variant,
            "selection_metric": "eval_perplexity",
            "baseline": {
                "perplexity_all": baseline_metrics.get("baseline_perplexity"),
                "perplexity_layer_a": baseline_metrics.get("baseline_layer_a_perplexity"),
                "perplexity_layer_b": baseline_metrics.get("baseline_layer_b_perplexity"),
            },
            "best_eval": best_eval_row,
            "final_eval": final_eval_row,
            "deltas_from_baseline": (
                {
                    "delta_ppl_all": (
                        best_eval_row.get("eval_perplexity") - baseline_metrics.get("baseline_perplexity")
                    )
                    if best_eval_row and baseline_metrics.get("baseline_perplexity") is not None
                    else None,
                    "delta_ppl_layer_a": (
                        best_eval_row.get("eval_layer_a_perplexity")
                        - baseline_metrics.get("baseline_layer_a_perplexity")
                    )
                    if best_eval_row and baseline_metrics.get("baseline_layer_a_perplexity") is not None
                    else None,
                    "delta_ppl_layer_b": (
                        best_eval_row.get("eval_layer_b_perplexity")
                        - baseline_metrics.get("baseline_layer_b_perplexity")
                    )
                    if best_eval_row and baseline_metrics.get("baseline_layer_b_perplexity") is not None
                    else None,
                }
                if best_eval_row
                else None
            ),
        },
    )

    write_json_artifact(
        reports_dir / "comparison_row.json",
        {
            "run_id": run_id,
            "experiment_name": exp_name,
            "variant": variant,
            "base_model": base_model_name,
            "use_4bit": bool(use_4bit),
            "gradient_checkpointing": bool(hw_cfg.get("gradient_checkpointing", False)),
            "seed": int(train_cfg["seed"]),
            "max_steps": int(train_cfg["max_steps"]) if train_cfg.get("max_steps") is not None else None,
            "effective_batch_size_sequences": int(train_cfg["per_device_train_batch_size"])
            * int(train_cfg["gradient_accumulation_steps"]),
            "max_seq_length": int(data_cfg["max_seq_length"]),
            "baseline_ppl_all": baseline_metrics.get("baseline_perplexity"),
            "baseline_ppl_layer_a": baseline_metrics.get("baseline_layer_a_perplexity"),
            "baseline_ppl_layer_b": baseline_metrics.get("baseline_layer_b_perplexity"),
            "best_ppl_all": best_eval_row.get("eval_perplexity") if best_eval_row else None,
            "best_ppl_layer_a": best_eval_row.get("eval_layer_a_perplexity") if best_eval_row else None,
            "best_ppl_layer_b": best_eval_row.get("eval_layer_b_perplexity") if best_eval_row else None,
            "train_runtime_s": getattr(train_result, "metrics", {}).get("train_runtime"),
            "train_steps_per_second": getattr(train_result, "metrics", {}).get("train_steps_per_second"),
            "num_nonfinite_events": (
                stability_summary.get("num_nan_loss_events", 0)
                + stability_summary.get("num_inf_loss_events", 0)
                + stability_summary.get("num_nonfinite_grad_norm_events", 0)
            ),
            "status": "completed",
        },
    )

    write_json_artifact(
        reports_dir / "run_manifest.json",
        _build_run_manifest(
            run_id=run_id,
            exp_name=exp_name,
            status="completed",
            baseline_only=False,
            variant=variant,
            protocol_version=protocol_version,
            logging_schema_version=logging_schema_version,
            start_time_utc=run_start_ts,
            end_time_utc=iso_utc_now(),
            duration_s=time.time() - run_start_wall,
            paths={
                **paths_common,
                "eval_summary_csv": str(reports_dir / "eval_summary.csv"),
                "comparison_row": str(reports_dir / "comparison_row.json"),
            },
        ),
    )
    _RUN_CONTEXT["status"] = "completed"


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as err:
        _write_failed_manifest_from_context(err)
        raise
