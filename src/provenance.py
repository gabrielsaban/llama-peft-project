from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Optional

import torch


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_pkg_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def git_rev_parse_short_head() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip() or None
    except Exception:
        return None


def git_is_dirty() -> Optional[bool]:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return bool(proc.stdout.strip())
    except Exception:
        return None


def build_environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
        "git_commit": git_rev_parse_short_head(),
        "git_dirty": git_is_dirty(),
        "torch_version": get_pkg_version("torch"),
        "transformers_version": get_pkg_version("transformers"),
        "peft_version": get_pkg_version("peft"),
        "datasets_version": get_pkg_version("datasets"),
        "bitsandbytes_version": get_pkg_version("bitsandbytes"),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }

    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        snapshot.update(
            {
                "cuda_device_index": int(device_idx),
                "cuda_device_name": torch.cuda.get_device_name(device_idx),
                "cuda_total_memory_bytes": int(props.total_memory),
            }
        )
    return snapshot
