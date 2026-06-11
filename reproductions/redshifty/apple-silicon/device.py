"""Torch device selection for the Apple Silicon (MPS) port of redshifty.

Prefers Metal Performance Shaders (Apple Silicon GPU), then CUDA, then CPU. Setting
PYTORCH_ENABLE_MPS_FALLBACK=1 routes any op without an MPS kernel to the CPU instead
of erroring — set here (before torch initializes the backend) and also exported in
the run commands as a belt-and-suspenders measure.

This file is copied verbatim into the patched source as `src-redshifty/nersc/_mps.py`
so the (sys.path-on-nersc) trainers can `from _mps import pick_device, pin_ok`.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def pick_device(prefer_gpu_index=None):
    """Return the best available torch device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{prefer_gpu_index or 0}")
    return torch.device("cpu")


def pin_ok(device) -> bool:
    """pin_memory only helps CUDA; MPS unified memory has no pinned host memory."""
    return device.type == "cuda"
