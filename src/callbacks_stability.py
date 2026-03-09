from __future__ import annotations

import math
from typing import Any

from .provenance import iso_utc_now


def build_stability_artifacts(log_history: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    loss_entries: list[tuple[int, float]] = []
    nan_loss_events = 0
    inf_loss_events = 0
    nonfinite_grad_events = 0
    divergence_events = 0

    for item in log_history:
        step = int(item.get("step", -1))
        if "loss" in item:
            try:
                loss = float(item["loss"])
            except (TypeError, ValueError):
                loss = float("nan")
            if math.isnan(loss):
                nan_loss_events += 1
                divergence_events += 1
                events.append(
                    {
                        "step": step,
                        "event_type": "nan_loss",
                        "severity": "error",
                        "value": None,
                        "threshold": "finite",
                        "message": "Observed NaN train loss.",
                        "timestamp_utc": iso_utc_now(),
                    }
                )
            elif math.isinf(loss):
                inf_loss_events += 1
                divergence_events += 1
                events.append(
                    {
                        "step": step,
                        "event_type": "inf_loss",
                        "severity": "error",
                        "value": None,
                        "threshold": "finite",
                        "message": "Observed Inf train loss.",
                        "timestamp_utc": iso_utc_now(),
                    }
                )
            else:
                loss_entries.append((step, loss))

        if "grad_norm" in item:
            try:
                grad_norm = float(item["grad_norm"])
            except (TypeError, ValueError):
                grad_norm = float("nan")
            if not math.isfinite(grad_norm):
                nonfinite_grad_events += 1
                divergence_events += 1
                events.append(
                    {
                        "step": step,
                        "event_type": "nonfinite_grad_norm",
                        "severity": "error",
                        "value": None,
                        "threshold": "finite",
                        "message": "Observed non-finite grad_norm.",
                        "timestamp_utc": iso_utc_now(),
                    }
                )

    loss_spike_events = 0
    spike_ratio_threshold = 1.5
    for idx in range(1, len(loss_entries)):
        prev_step, prev_loss = loss_entries[idx - 1]
        curr_step, curr_loss = loss_entries[idx]
        if prev_loss > 0 and curr_loss > prev_loss * spike_ratio_threshold:
            loss_spike_events += 1
            events.append(
                {
                    "step": curr_step,
                    "event_type": "loss_spike",
                    "severity": "warn",
                    "value": curr_loss,
                    "threshold": f">{spike_ratio_threshold}x previous loss",
                    "message": (
                        f"Loss spiked from {prev_loss:.6f} at step {prev_step} to {curr_loss:.6f}."
                    ),
                    "timestamp_utc": iso_utc_now(),
                }
            )

    summary = {
        "num_loss_spike_events": loss_spike_events,
        "num_divergence_events": divergence_events,
        "num_nan_loss_events": nan_loss_events,
        "num_inf_loss_events": inf_loss_events,
        "num_nonfinite_grad_norm_events": nonfinite_grad_events,
        "num_oom_events": 0,
        "optimizer_reset_count": 0,
        "terminated_early": False,
        "termination_reason": "completed",
        "last_completed_step": max((int(item.get("step", 0)) for item in log_history), default=0),
        "stability_rule_config": {
            "loss_spike_rule": f"curr_loss > {spike_ratio_threshold} * prev_loss",
            "finite_checks": ["loss", "grad_norm"],
        },
    }
    return summary, events
