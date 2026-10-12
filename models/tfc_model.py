"""
Inference-time embedding wrapper for TF-C: a single nn.Module whose
forward(x) -> (B, 2d) returns the concatenation of the time-domain and
frequency-domain branch representations, so downstream code (evaluate.py,
plots.py, retrieval/*) can treat it exactly like any other encoder's
forward(x) -> (B, d) contract, without special-casing TF-C.
"""
import torch
import torch.nn as nn

from losses.tfc_loss import to_freq_magnitude


class TFCEmbeddingWrapper(nn.Module):
    def __init__(self, time_encoder: nn.Module, freq_encoder: nn.Module):
        super().__init__()
        self.time_encoder = time_encoder
        self.freq_encoder = freq_encoder

    def forward(self, x: torch.Tensor, return_seq: bool = False) -> torch.Tensor:
        time_repr = self.time_encoder(x)
        freq_mag = to_freq_magnitude(x)
        freq_repr = self.freq_encoder(freq_mag)
        return torch.cat([time_repr, freq_repr], dim=1)
