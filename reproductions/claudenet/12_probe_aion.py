#!/usr/bin/env python3
"""12_probe_aion.py — train a light probe on frozen AION-1 embeddings and emit
lens-class PROBABILITIES for the held-out eval sets (claudenet venv).

The head mirrors _probe.MLPHead (Linear->GELU->Dropout->Linear, hidden=256) but we
keep a probability output (softmax[:,1]) and a held-out-val early stop, which the
matched-FPR gate needs. Standardisation is fit on the train pool only.

Outputs:
  data/ckpt/aion_probe_<variant>.pt        head state + xmu/xsd (reused as Phase-1 member)
  data/scores_aion_gate.parquet            [split,row_id,label,fits_path,p_aion]

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      /home2/benson/.venvs/claudenet/bin/python 12_probe_aion.py --variant base
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import _clib as C

EMB = C.EMB


class MLPProbe(nn.Module):
    """mirror of _probe.MLPHead with a 2-class output."""

    def __init__(self, dim, hidden=256, k=2, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


def load_split(sp, variant):
    emb = np.load(EMB / f"aion_emb_{sp}_{variant}.npy")
    man = pd.read_parquet(EMB / f"aion_in_{sp}_manifest.parquet")
    assert len(emb) == len(man), f"{sp}: emb {len(emb)} != manifest {len(man)}"
    return emb.astype(np.float32), man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr_all, man_tr = load_split("trainpool", args.variant)
    ytr_all = man_tr["label"].to_numpy().astype(np.int64)

    # standardise on train pool
    xmu, xsd = Xtr_all.mean(0), Xtr_all.std(0) + 1e-6
    Xn = (Xtr_all - xmu) / xsd

    # 80/20 stratified internal split for early stopping on val AUC
    rng = np.random.default_rng(C.SEED)
    va = np.zeros(len(Xn), bool)
    for lab in (0, 1):
        idx = np.where(ytr_all == lab)[0]
        va[rng.choice(idx, size=int(0.2 * len(idx)), replace=False)] = True
    Xt, yt = torch.from_numpy(Xn[~va]).to(device), torch.from_numpy(ytr_all[~va]).to(device)
    Xv = torch.from_numpy(Xn[va]).to(device)

    # mild class weighting (1:~5.8 imbalance) so the rare class isn't ignored
    cw = torch.tensor([1.0, float((ytr_all == 0).sum() / max((ytr_all == 1).sum(), 1))],
                      device=device)
    head = MLPProbe(Xn.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(weight=cw)

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(Xt), device=device)
        for s in range(0, len(Xt), 256):
            b = perm[s:s + 256]
            opt.zero_grad(); lossf(head(Xt[b]), yt[b]).backward(); opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            pv = torch.softmax(head(Xv), 1)[:, 1].cpu().numpy()
        auc = roc_auc_score(ytr_all[va], pv)
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= 25:
                break
    head.load_state_dict(best_state); head.eval()
    print(f"[probe] AION-{args.variant} probe val AUC={best_auc:.4f} (dim={Xn.shape[1]})")

    torch.save({"state_dict": best_state, "xmu": xmu, "xsd": xsd,
                "dim": int(Xn.shape[1]), "variant": args.variant, "val_auc": best_auc},
               C.CKPT / f"aion_probe_{args.variant}.pt")

    # score held-out eval splits
    out = []
    for sp in ("testneg", "storfer", "inchausti"):
        X, man = load_split(sp, args.variant)
        Xs = torch.from_numpy(((X - xmu) / xsd).astype(np.float32)).to(device)
        with torch.no_grad():
            p = torch.softmax(head(Xs), 1)[:, 1].cpu().numpy()
        d = man[["row_id", "label", "fits_path"]].copy()
        d["split"] = sp; d["p_aion"] = p
        out.append(d)
        print(f"[probe] scored {sp:10s} n={len(d)} mean_p={p.mean():.3f}")
    pd.concat(out, ignore_index=True).to_parquet(C.DATA / "scores_aion_gate.parquet", index=False)
    print("[12] wrote scores_aion_gate.parquet")


if __name__ == "__main__":
    main()
