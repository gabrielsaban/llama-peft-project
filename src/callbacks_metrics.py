from __future__ import annotations

import math
import statistics
import time
from typing import Any, Optional

import torch
from transformers import TrainerCallback


def collect_finite(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def build_timing_summary(
    train_metrics: dict[str, Any],
    eval_rows: list[dict[str, Any]],
    baseline_only: bool,
) -> dict[str, Any]:
    step_times = collect_finite(
        [r.get("train_step_time_mean_s_window") for r in eval_rows if r.get("phase") == "eval"]
    )
    step_times_sorted = sorted(step_times)
    return {
        "mode": "baseline_only" if baseline_only else "train_and_eval",
        "train_runtime_s": train_metrics.get("train_runtime"),
        "train_steps_per_second": train_metrics.get("train_steps_per_second"),
        "optimizer_step_time_mean_s": (
            statistics.fmean(step_times_sorted) if step_times_sorted else None
        ),
        "optimizer_step_time_p50_s": percentile(step_times_sorted, 0.5),
        "optimizer_step_time_p95_s": percentile(step_times_sorted, 0.95),
        "optimizer_step_time_std_s": (
            statistics.pstdev(step_times_sorted)
            if len(step_times_sorted) > 1
            else 0.0 if step_times_sorted else None
        ),
        "num_step_time_samples": len(step_times_sorted),
        "timing_window_definition": "windowed mean optimizer step time emitted at each eval interval",
    }


class TrainStepTimeTracker:
    def __init__(self) -> None:
        self._step_start_t: Optional[float] = None
        self._interval_step_durations_s: list[float] = []
        self._train_peak_allocated_bytes: int = 0
        self._train_peak_reserved_bytes: int = 0

    def reset(self) -> None:
        self._step_start_t = None
        self._interval_step_durations_s.clear()
        self._train_peak_allocated_bytes = 0
        self._train_peak_reserved_bytes = 0

    def on_step_begin(self) -> None:
        self._step_start_t = time.perf_counter()

    def on_step_end(self) -> None:
        if self._step_start_t is not None:
            self._interval_step_durations_s.append(time.perf_counter() - self._step_start_t)
            self._step_start_t = None

        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            self._train_peak_allocated_bytes = max(
                self._train_peak_allocated_bytes,
                int(torch.cuda.memory_allocated(device_idx)),
            )
            self._train_peak_reserved_bytes = max(
                self._train_peak_reserved_bytes,
                int(torch.cuda.memory_reserved(device_idx)),
            )

    def consume_interval_stats(self) -> dict[str, Optional[float]]:
        samples = sorted(self._interval_step_durations_s)
        self._interval_step_durations_s.clear()
        if not samples:
            return {
                "mean_s": None,
                "p50_s": None,
                "p95_s": None,
                "std_s": None,
                "num_samples": 0,
            }
        return {
            "mean_s": statistics.fmean(samples),
            "p50_s": percentile(samples, 0.5),
            "p95_s": percentile(samples, 0.95),
            "std_s": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
            "num_samples": len(samples),
        }

    def train_peak_vram_gb(self) -> dict[str, Optional[float]]:
        if not torch.cuda.is_available():
            return {
                "train_peak_vram_allocated_gb": None,
                "train_peak_vram_reserved_gb": None,
            }
        gb = 1024**3
        return {
            "train_peak_vram_allocated_gb": self._train_peak_allocated_bytes / gb,
            "train_peak_vram_reserved_gb": self._train_peak_reserved_bytes / gb,
        }


class TrainStepTimeCallback(TrainerCallback):
    def __init__(self, tracker: TrainStepTimeTracker) -> None:
        self.tracker = tracker

    def on_step_begin(self, args, state, control, **kwargs):
        self.tracker.on_step_begin()

    def on_step_end(self, args, state, control, **kwargs):
        self.tracker.on_step_end()
