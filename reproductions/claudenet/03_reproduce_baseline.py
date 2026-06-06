#!/usr/bin/env python3
"""03_reproduce_baseline.py — SANITY GATE.

Re-derive the Stage-D (staged) matched-FPR recovery numbers from the synced
checkpoints + cutouts using ClaudeNet's own evaluation path (_ensemble.recovery_at_fpr),
and assert they reproduce the on-disk meta_metrics_staged.json. This proves the
data + checkpoints + eval harness are wired correctly, so any later ClaudeNet
delta is real. No ClaudeNet improvement is claimed until this passes.

Mirrors inchausti-2025/24_train_staged.recovery_at_fpr exactly: threshold on the
held-out TEST negatives (cutouts_fits_neg_dr9), recovery of the published
Storfer/Inchausti candidates; meta = sigmoid(MetaLearner([p_shielded, p_effnet])).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      /home2/benson/.venvs/claudenet/bin/python 03_reproduce_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _clib as C
import _scorelib as SL
import _ensemble as E

M = C.models()
MetaLearner = M["MetaLearner"]


def remap(fits_dir: str) -> Path:
    """Rewrite an inchausti absolute fits_dir to the synced claudenet copy."""
    return C.DATA / Path(str(fits_dir)).name


def load_meta(device):
    ck = torch.load(str(C.DATA / "checkpoint_best_meta_staged.pt"),
                    map_location="cpu", weights_only=False)
    meta = MetaLearner().to(device)
    meta.load_state_dict(ck["state_dict"])
    meta.eval()
    return meta


def meta_prob(meta, p_sh, p_ef, device):
    P = np.stack([p_sh, p_ef], 1).astype(np.float32)
    ok = np.isfinite(P).all(1)
    out = np.full(len(P), np.nan, np.float32)
    with torch.no_grad():
        out[ok] = torch.sigmoid(meta(torch.from_numpy(P[ok]).to(device))).cpu().numpy()
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sh, _, m_sh, s_sh, _ = SL.load_checkpoint_model(C.DATA / "checkpoint_best_shielded194k_staged.pt", device)
    ef, _, m_ef, s_ef, _ = SL.load_checkpoint_model(C.DATA / "checkpoint_best_efficientnet_staged.pt", device)
    meta = load_meta(device)

    # held-out test negatives from the staged split
    split = pd.read_parquet(C.DATA / "training_split_staged.parquet")
    neg = split[(split.split == "test") & (split.label == 0)]
    neg_paths = [remap(r.fits_dir) / f"{r.row_id}.fits" for r in neg.itertuples()]
    print(f"[base] {len(neg_paths)} held-out test negatives")
    nr = SL.score_paths(neg_paths, sh, "shielded", m_sh, s_sh, device)
    ne = SL.score_paths(neg_paths, ef, "efficientnet", m_ef, s_ef, device)
    nm = meta_prob(meta, nr, ne, device)
    negs = {"resnet": nr, "effnet": ne, "meta": nm}

    rows = []
    for key, cut, csv in (("storfer", "cutouts_fits_candidates_storfer", "storfer2024_published_catalog.csv"),
                          ("inchausti", "cutouts_fits_candidates_inchausti", "inchausti2025_published_catalog.csv")):
        cat = pd.read_csv(C.DATA / csv)
        cp = [C.DATA / cut / f"{n}.fits" for n in cat["name"]]
        cr = SL.score_paths(cp, sh, "shielded", m_sh, s_sh, device)
        ce = SL.score_paths(cp, ef, "efficientnet", m_ef, s_ef, device)
        cm = meta_prob(meta, cr, ce, device)
        cands = {"resnet": cr, "effnet": ce, "meta": cm}
        for mdl in ("resnet", "effnet", "meta"):
            rec = E.recovery_at_fpr(negs[mdl], cands[mdl], fprs=C.TARGET_FPR)
            for fpr in C.TARGET_FPR:
                rows.append({"catalog": key, "model": mdl, "fpr": fpr,
                             "threshold": rec[fpr]["threshold"],
                             "recovery": rec[fpr]["recovery"]})
    got = pd.DataFrame(rows)
    got.to_csv(C.DATA / "baseline_reproduced.csv", index=False)

    # compare against stored meta_metrics_staged.json
    ref = json.load(open(C.DATA / "meta_metrics_staged.json"))["recovery_at_fpr"]
    print(f"\n{'catalog':>10} {'model':>7} {'fpr':>6} {'ours':>8} {'stored':>8} {'dAUC':>7}")
    worst = 0.0
    checks = []
    for _, r in got.iterrows():
        k = f"{r.catalog}|{r.model}|{r.fpr}"
        stored = ref.get(k, {}).get("recovery", float("nan"))
        d = abs(r.recovery - stored) if np.isfinite(stored) else float("nan")
        if np.isfinite(d):
            worst = max(worst, d)
            # gate on the headline cells: effnet & meta @1% FPR
            if r.model in ("effnet", "meta") and r.fpr == 0.01:
                checks.append((k, d))
        print(f"{r.catalog:>10} {r.model:>7} {r.fpr:>6.3f} {r.recovery:>8.3f} {stored:>8.3f} {d:>7.3f}")

    tol = 0.02
    bad = [(k, d) for k, d in checks if d > tol]
    print(f"\n[gate] worst |delta| over all cells = {worst:.3f}; headline cells within {tol}: "
          f"{'PASS' if not bad else 'FAIL ' + str(bad)}")
    if bad:
        print("[gate] SANITY GATE FAILED — do not trust later ClaudeNet deltas until fixed.")
        return 1
    print("[gate] SANITY GATE PASSED — baseline reproduced; harness is trustworthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
