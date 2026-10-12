"""
Central device resolution: CUDA (Linux/Windows GPU boxes) > MPS (Apple
Silicon, e.g. the M-series MacBook Air this project is tuned for) > CPU.

Import `resolve_device()` everywhere instead of hardcoding
`torch.cuda.is_available()` checks, so the SAME codebase runs unmodified on
Colab (CUDA), a Mac (MPS), or plain CPU.

Also sets PYTORCH_ENABLE_MPS_FALLBACK=1 on import: a handful of ops used in
this codebase (e.g. torch.fft.rfft inside the TF-C baseline, some
reduction ops) have historically lagged behind on the MPS backend in some
PyTorch versions; this environment variable makes PyTorch silently fall
back to CPU for just that op instead of raising, with no code changes
needed. It is a no-op on non-Mac machines.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def resolve_device(verbose: bool = False) -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if verbose:
        print(f"[device] using '{device}'"
              + (" (Apple Silicon GPU)" if device == "mps" else ""))
    return device
