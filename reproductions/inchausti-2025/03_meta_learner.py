#!/usr/bin/env python3
"""
03_meta_learner.py

The feature-weighted-stacking meta-learner of Inchausti+2025 (arXiv:2508.20087,
§3.2.3) that fuses the shielded-ResNet and EfficientNetV2 base-model
probabilities into one ensemble probability.

What the paper says (§3.2.3, verbatim where quoted):
  "To combine these predictions, we employ feature weighted stacking
   (Coscrato et al. 2020), which consists of a meta-learner that aggregates the
   probabilities produced by the two base models. This meta-learner is
   implemented as a simple one-layer neural network with 300 nodes, designed to
   take the probabilities of the base models as its input and output a single
   probability." (meta AUC 0.9989 == simple-average AUC 0.9989; the correlated
   bases mean stacking ~ averaging, but the meta-learner "optimizes the weighting
   ... a more systematic approach than manual averaging.")

Coscrato, Inacio & Izbicki 2020 (arXiv:1906.09735, "NN-Stacking: Feature
weighted linear stacking through neural networks") is the cited method. Full FWLS
makes the combination weights depend on input features; Inchausti's stated input
is ONLY the two scalar base probabilities, so the faithful minimal reconstruction
is a one-hidden-layer (300-unit) MLP over the 2 probabilities -> single logit,
trained with binary cross-entropy. A parameterless `simple_average` baseline
reproduces the paper's "stacking ~ averaging" finding.

Feature order convention (used everywhere downstream): column 0 = shielded
ResNet probability, column 1 = EfficientNetV2 probability.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

HIDDEN = 300
FEATURE_ORDER = ("shielded194k", "efficientnet")  # meta-input column order


class MetaLearner(nn.Module):
    """One-hidden-layer FWLS meta-learner: [p_resnet, p_effnet] -> ensemble logit.

    forward(p) -> (B,) logit; probability = sigmoid(logit). BCEWithLogitsLoss.
    """

    def __init__(self, n_base: int = 2, hidden: int = HIDDEN):
        super().__init__()
        self.n_base = n_base
        self.net = nn.Sequential(
            nn.Linear(n_base, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p).squeeze(-1)


def simple_average(p: np.ndarray) -> np.ndarray:
    """Parameterless baseline: mean of base-model probabilities over the last axis."""
    return np.asarray(p, dtype=np.float64).mean(axis=-1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def smoke_test() -> None:
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)

    model = MetaLearner()
    n = count_params(model)
    expected = (2 * HIDDEN + HIDDEN) + (HIDDEN + 1)  # fc1 (600+300) + fc2 (300+1) = 1201
    print(f"[smoke] MetaLearner params: {n:,}  hidden={HIDDEN}")
    assert n == expected, f"unexpected param count {n} (expected {expected})"
    assert model.net[0].out_features == 300, "expected a 300-node hidden layer"

    # Synthetic correlated base probabilities -> reproduce "stacking ~ averaging".
    N = 4000
    y = rng.integers(0, 2, size=N)
    signal = 0.4 * (2 * y - 1)              # +-0.4 by class
    shared = 0.15 * rng.normal(0, 1, N)     # correlation between the two bases
    p_res = np.clip(0.5 + signal + shared + 0.10 * rng.normal(0, 1, N), 0, 1)
    p_eff = np.clip(0.5 + signal + shared + 0.10 * rng.normal(0, 1, N), 0, 1)
    P = np.stack([p_res, p_eff], axis=1).astype(np.float32)

    avg_auc = float(roc_auc_score(y, simple_average(P)))

    Xt = torch.from_numpy(P)
    yt = torch.from_numpy(y.astype(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(400):
        opt.zero_grad()
        lossf(model(Xt), yt).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        meta_auc = float(roc_auc_score(y, torch.sigmoid(model(Xt)).numpy()))

    print(f"[smoke] avg_auc={avg_auc:.4f}  meta_auc={meta_auc:.4f}  "
          f"|delta|={abs(meta_auc - avg_auc):.4f}  (paper: equal at 0.9989)")
    assert abs(meta_auc - avg_auc) < 0.02, "meta should ~ average on correlated bases"
    print("[smoke] OK")


if __name__ == "__main__":
    smoke_test()
