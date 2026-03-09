from __future__ import annotations

import math
from typing import Any

from .provenance import iso_utc_now


def build_stability_artifacts(log_history: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    finite_grad_norms: list[float] = []
    nan_loss_events = 0
    inf_loss_events = 0
    nonfinite_grad_events = 0

    for item in log_history:
        step = int(item.get("step", -1))
        if "loss" in item:
            try:
                loss = float(item["loss"])
            except (TypeError, ValueError):
                loss = float("nan")
            if math.isnan(loss):
                nan_loss_events += 1
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

        if "grad_norm" in item:
            try:
                grad_norm = float(item["grad_norm"])
            except (TypeError, ValueError):
                grad_norm = float("nan")
            if not math.isfinite(grad_norm):
                nonfinite_grad_events += 1
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
            else:
                finite_grad_norms.append(grad_norm)

    summary = {
        "num_nan_loss_events": nan_loss_events,
        "num_inf_loss_events": inf_loss_events,
        "num_nonfinite_grad_norm_events": nonfinite_grad_events,
        "num_oom_events": 0,
        "optimizer_reset_count": 0,
        "terminated_early": False,
        "termination_reason": "completed",
        "last_completed_step": max((int(item.get("step", 0)) for item in log_history), default=0),
        "max_grad_norm": max(finite_grad_norms) if finite_grad_norms else None,
        "min_grad_norm": min(finite_grad_norms) if finite_grad_norms else None,
        "stability_rule_config": {
            "finite_checks": ["loss", "grad_norm"],
            "event_scope": ["nan_loss", "inf_loss", "nonfinite_grad_norm", "oom", "early_termination"],
        },
    }
    return summary, events
