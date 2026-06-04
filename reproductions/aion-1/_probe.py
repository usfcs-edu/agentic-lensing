"""
Lightweight probing heads + training loops for AION-1 downstream tasks.

The heads mirror the ones the paper uses (and that ship on the AION
``posttraining`` branch, aion/posttraining/models.py):

  * ``MLPHead``       -- mean-pooled features -> 2-layer MLP  (morphology, task 4)
  * ``LinearHead``    -- mean-pooled features -> linear        (quick baselines)
  * ``CrossAttnHead`` -- learned queries -> NormCrossAttention -> per-target
                         linear heads = the paper's "attentive pooling"
                         (galaxy/stellar property estimation, tasks 1,2).

These operate on *precomputed* frozen-encoder features (``(N,D)`` mean-pooled,
or ``(N,T,D)`` full token embeddings for the attentive head), which decouples
the expensive backbone pass from cheap head training/sweeps. Reuses
``aion.fourm.fm_utils.NormCrossAttention`` (aion/fourm/fm_utils.py:345).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import _config as C


class LinearHead(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out)

    def forward(self, x):  # x: (B, D)
        return self.linear(x)


class MLPHead(nn.Module):
    def __init__(self, dim: int, dim_out: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim_out)
        )

    def forward(self, x):  # x: (B, D)
        return self.net(x)


class SegHead(nn.Module):
    """Lightweight convolutional upsampler for galaxy-structure segmentation
    (task 5). AION tokenises a 96x96 image to a 24x24 grid (576 tokens); we
    reshape the frozen token embeddings back to that grid and upsample to a
    per-pixel mask. Matches the paper's 'mean-pool spatial tokens + lightweight
    conv upsampler' recipe (here we keep the full spatial grid, not a mean)."""

    def __init__(self, dim: int, n_classes: int, grid: int = 24):
        super().__init__()
        self.grid = grid
        self.proj = nn.Conv2d(dim, 128, 1)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),  # 24 -> 48
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),   # 48 -> 96
            nn.Conv2d(32, n_classes, 1),
        )

    def forward(self, x):  # x: (B, grid*grid, D)
        B, T, D = x.shape
        g = self.grid
        x = x[:, : g * g].transpose(1, 2).reshape(B, D, g, g)
        return self.up(self.proj(x))  # (B, n_classes, 96, 96)


class CrossAttnHead(nn.Module):
    """Attentive pooling: ``dim_out`` learned queries cross-attend to the token
    embeddings, then a per-target linear maps each query to a scalar. Matches
    aion.posttraining.models.CrossAttentionProbing (minus the frozen backbone,
    which we run separately)."""

    def __init__(self, dim: int, dim_out: int, num_heads: int):
        super().__init__()
        from aion.fourm.fm_utils import NormCrossAttention

        self.query = nn.Parameter(torch.randn(1, dim_out, dim))
        self.attention = NormCrossAttention(dim, num_heads=num_heads, proj_bias=False)
        self.decoders = nn.ModuleList([nn.Linear(dim, 1) for _ in range(dim_out)])

    def forward(self, x):  # x: (B, T, D)
        q = self.query.expand(x.size(0), -1, -1)
        out = self.attention(q, x)  # (B, dim_out, D)
        return torch.cat([dec(out[:, i]) for i, dec in enumerate(self.decoders)], dim=-1)


# --- training loops ----------------------------------------------------------
def _loader(X, Y, batch_size, shuffle, device):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(C.SEED)
        rng.shuffle(idx)
    for s in range(0, n, batch_size):
        b = idx[s : s + batch_size]
        xb = torch.as_tensor(np.asarray(X[b], dtype=np.float32), device=device)
        yb = torch.as_tensor(np.asarray(Y[b]), device=device)
        yield xb, yb


def train_regression(X_tr, Y_tr, X_te, Y_te, head_factory, *, epochs=100, lr=1e-3,
                     weight_decay=1e-4, batch_size=256, patience=12, device="cuda",
                     standardize_x=True, verbose=False):
    """Train a regression head; returns (preds_test, r2_per_target, best_state).

    Y is standardized per-target for stable optimisation and de-standardized
    before scoring (R^2 is scale-invariant)."""
    Y_tr = np.atleast_2d(np.asarray(Y_tr, dtype=np.float64))
    Y_te = np.atleast_2d(np.asarray(Y_te, dtype=np.float64))
    if Y_tr.shape[0] != len(X_tr):
        Y_tr, Y_te = Y_tr.T, Y_te.T
    ymu, ysd = Y_tr.mean(0), Y_tr.std(0) + 1e-8

    # Big token-level arrays (N,T,D) stay fp16 and are cast per-batch in _loader,
    # to avoid a float32 blow-up (xlarge phot_image_spec is ~71 GB fp16; float32
    # would be ~142 GB and OOM the host).
    if standardize_x and np.ndim(X_tr) == 2:
        Xtr = np.asarray(X_tr, dtype=np.float32)
        Xte = np.asarray(X_te, dtype=np.float32)
        xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - xmu) / xsd, (Xte - xmu) / xsd
    else:
        Xtr, Xte = X_tr, X_te

    Ytr_s = ((Y_tr - ymu) / ysd).astype(np.float32)
    dim = np.shape(Xtr)[-1]
    # token-aware eval batch (1024 OOMs the GPU for big (N,T,D) image embeddings)
    eval_bs = 1024 if np.ndim(Xtr) == 2 else int(np.clip(2.5e7 / np.prod(np.shape(Xtr)[1:]), 8, 256))
    head = head_factory(dim, Y_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss()

    best_r2, best_preds, best_state, bad = -1e9, None, None, 0
    for ep in range(epochs):
        head.train()
        for xb, yb in _loader(Xtr, Ytr_s, batch_size, True, device):
            opt.zero_grad()
            loss = lossf(head(xb), yb.float())
            loss.backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            preds = []
            for xb, _ in _loader(Xte, np.zeros((len(Xte), 1)), eval_bs, False, device):
                preds.append(head(xb).cpu().numpy())
        preds = np.concatenate(preds, 0) * ysd + ymu
        from _metrics import r2_per_target

        r2s = r2_per_target(Y_te, preds)
        mr2 = float(np.mean(r2s))
        if verbose:
            print(f"  ep{ep:03d} meanR2={mr2:.4f} {['%.3f'%v for v in r2s]}")
        if mr2 > best_r2:
            best_r2, best_preds, bad = mr2, preds, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    from _metrics import r2_per_target

    return best_preds, r2_per_target(Y_te, best_preds), best_state


def train_classification(X_tr, Y_tr, X_te, Y_te, head_factory, n_classes, *, epochs=120,
                         lr=1e-3, weight_decay=1e-4, batch_size=256, patience=15,
                         device="cuda", standardize_x=True, verbose=False):
    """Train a classification head; returns (preds_test, accuracy, best_state)."""
    if standardize_x and np.ndim(X_tr) == 2:
        Xtr = np.asarray(X_tr, dtype=np.float32)
        Xte = np.asarray(X_te, dtype=np.float32)
        xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - xmu) / xsd, (Xte - xmu) / xsd
    else:
        Xtr, Xte = X_tr, X_te
    Ytr = np.asarray(Y_tr).astype(np.int64)
    Yte = np.asarray(Y_te).astype(np.int64)
    dim = np.shape(Xtr)[-1]
    eval_bs = 1024 if np.ndim(Xtr) == 2 else int(np.clip(2.5e7 / np.prod(np.shape(Xtr)[1:]), 8, 256))
    head = head_factory(dim, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()

    best_acc, best_preds, best_state, bad = -1.0, None, None, 0
    for ep in range(epochs):
        head.train()
        for xb, yb in _loader(Xtr, Ytr, batch_size, True, device):
            opt.zero_grad()
            loss = lossf(head(xb), yb.long())
            loss.backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            logits = []
            for xb, _ in _loader(Xte, np.zeros(len(Xte)), eval_bs, False, device):
                logits.append(head(xb).cpu().numpy())
        preds = np.concatenate(logits, 0).argmax(1)
        acc = float(np.mean(preds == Yte))
        if verbose:
            print(f"  ep{ep:03d} acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_preds, bad = acc, preds, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    return best_preds, best_acc, best_state
