"""
Encoder f_theta: R^{1xT} -> R^d  (Section 3.2)

Architecture: 1-D convolutional tokeniser (downsamples the raw signal into
a sequence of local patch tokens) followed by a standard Transformer
encoder (Vaswani et al., 2017), with mean pooling over the token sequence
to obtain a single fixed-size embedding per recording.

A lightweight 1-D ResNet-18-style backbone (used as an ablation / cheaper
alternative, per Section 4.4 "Implementation Details (Tentative)") is also
provided.
"""
import math
import torch
import torch.nn as nn


class ConvTokenizer(nn.Module):
    """Strided 1-D convolutions turning (B,1,T) into a token sequence
    (B, L, C)."""

    def __init__(self, in_channels=1, channels=64, n_layers=3, kernel_size=7):
        super().__init__()
        layers = []
        c_in = in_channels
        for i in range(n_layers):
            layers += [
                nn.Conv1d(c_in, channels, kernel_size=kernel_size, stride=2,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            ]
            c_in = channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, T) -> (B, C, L) -> (B, L, C)
        z = self.net(x)
        return z.transpose(1, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerECGEncoder(nn.Module):
    """Conv tokenizer + Transformer encoder -> pooled d-dim embedding."""

    def __init__(self, embed_dim=128, conv_channels=64, n_layers=4, n_heads=4,
                 ff_dim=256, dropout=0.1, in_channels=1):
        super().__init__()
        self.tokenizer = ConvTokenizer(in_channels, conv_channels, n_layers=3)
        self.input_proj = nn.Linear(conv_channels, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.embed_dim = embed_dim

    def forward(self, x, return_seq: bool = False):
        # x: (B, 1, T)
        tokens = self.tokenizer(x)              # (B, L, C)
        tokens = self.input_proj(tokens)         # (B, L, d)
        tokens = self.pos_enc(tokens)
        h = self.transformer(tokens)             # (B, L, d)
        if return_seq:
            return h                             # per-timestamp representations (used by TS2Vec)
        pooled = h.mean(dim=1)                   # (B, d)  mean pooling
        return pooled


class ResNet1DBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 7, stride=stride, padding=3)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 7, padding=3)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.GELU()
        self.downsample = None
        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride), nn.BatchNorm1d(c_out)
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class ResNet18_1D(nn.Module):
    """Lightweight 1-D ResNet-18-style backbone (cheaper alternative to the
    Transformer encoder, per Section 4.4)."""

    def __init__(self, embed_dim=128, base_channels=64, in_channels=1):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c, 15, stride=2, padding=7), nn.BatchNorm1d(c), nn.GELU()
        )
        self.layer1 = nn.Sequential(ResNet1DBlock(c, c), ResNet1DBlock(c, c))
        self.layer2 = nn.Sequential(ResNet1DBlock(c, 2 * c, stride=2), ResNet1DBlock(2 * c, 2 * c))
        self.layer3 = nn.Sequential(ResNet1DBlock(2 * c, 4 * c, stride=2), ResNet1DBlock(4 * c, 4 * c))
        self.layer4 = nn.Sequential(ResNet1DBlock(4 * c, 8 * c, stride=2), ResNet1DBlock(8 * c, 8 * c))
        self.proj = nn.Linear(8 * c, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x, return_seq: bool = False):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)          # (B, C, L)
        if return_seq:
            seq = h.transpose(1, 2)         # (B, L, C)
            return self.proj(seq)           # (B, L, d) per-position representations
        pooled = h.mean(dim=-1)     # global average pool
        return self.proj(pooled)


class FCNEncoder(nn.Module):
    """Classic 4-layer Fully Convolutional Network backbone (Wang, Yan &
    Oates, 2017), the same architecture used in the authors' prior
    semi-supervised time-series-classification work (Goyo, Nanculef &
    Valle, "Improving negative generation for self-supervised
    triplet-based time series classification"; and the companion
    InverseTime paper): four 1-D convolutions with 16, 32, 32, 64 filters
    and kernel sizes 8, 5, 3, 3, each followed by BatchNorm + ReLU, then
    global average pooling over time. This is deliberately much smaller
    and lower-capacity than TransformerECGEncoder / ResNet18_1D above
    (roughly 20-40x fewer parameters depending on cfg.embed_dim), so any
    residual encoder capacity that could otherwise absorb/launder a
    training-loss-level mechanism (Component A/B) is minimised -- the aim
    is to isolate the loss/pretext-task's own effect from the encoder's.
    The paper's original 64-dim conv output is projected to cfg.embed_dim
    only if that differs from 64 (identity otherwise, to stay faithful to
    the original architecture when embed_dim=64 is used, as recommended).
    """

    FILTERS = (16, 32, 32, 64)
    KERNELS = (8, 5, 3, 3)

    def __init__(self, embed_dim=64, in_channels=1):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out, k in zip(self.FILTERS, self.KERNELS):
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = self.FILTERS[-1]
        self.proj = None if embed_dim == self.out_channels else nn.Linear(self.out_channels, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x, return_seq: bool = False):
        # x: (B, 1, T)
        h = self.net(x)                      # (B, 64, T) -- 'same' padding, stride 1 throughout
        if return_seq:
            seq = h.transpose(1, 2)           # (B, T, 64)
            return self.proj(seq) if self.proj is not None else seq
        pooled = h.mean(dim=-1)               # (B, 64) global average pool
        return self.proj(pooled) if self.proj is not None else pooled


def build_encoder(name: str, cfg) -> nn.Module:
    if name == "transformer":
        return TransformerECGEncoder(
            embed_dim=cfg.embed_dim, conv_channels=cfg.conv_channels,
            n_layers=cfg.n_transformer_layers, n_heads=cfg.n_heads,
            ff_dim=cfg.ff_dim, dropout=cfg.dropout,
            in_channels=getattr(cfg, "in_channels", 1),
        )
    elif name == "resnet18":
        return ResNet18_1D(embed_dim=cfg.embed_dim, base_channels=cfg.conv_channels,
                          in_channels=getattr(cfg, "in_channels", 1))
    elif name == "fcn":
        return FCNEncoder(embed_dim=cfg.embed_dim, in_channels=getattr(cfg, "in_channels", 1))
    else:
        raise ValueError(f"Unknown encoder '{name}'")
