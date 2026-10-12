import torch.nn as nn


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head g_phi (Section 3.2), following the
    standard SimCLR convention."""

    def __init__(self, in_dim=128, hidden_dim=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LinearProbe(nn.Module):
    """Frozen-encoder linear classifier used for the downstream AUROC
    metric (Section 4.2, item 4)."""

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class PhysioRegressionHead(nn.Module):
    """Component B (DA-ECG-Ret v2): small MLP predicting the unsupervised
    physiological/morphology descriptors (data/physiological_features.py)
    from the encoder's representation, used as an auxiliary self-supervised
    regularizer that anchors the embedding to physiologically meaningful
    axes rather than only augmentation-invariant nuisance directions."""

    def __init__(self, in_dim, hidden_dim, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x):
        return self.net(x)
