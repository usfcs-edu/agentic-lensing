#!/usr/bin/env python3
"""40_label_efficiency.py — Phase 3: how many labeled lenses does each approach need?

Compares, at positive-label fractions {5,10,25,50,100}% (negatives fixed), the
matched-FPR recovery of:
  * AION-1 frozen-embedding MLP probe  (foundation features -> label-efficient?)
  * Shielded-ResNet trained from scratch on the same cutouts (supervised baseline)
on IDENTICAL training objects (the AION trainpool: 1373 pos + 8000 neg) and the
same held-out Storfer/Inchausti positives vs staged test-negatives.

Deliverable: recovery@1%FPR vs #labeled-positives (the label-efficiency curve) and
the #labels each needs to reach a target -> the labor-saving factor.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      /home2/benson/.venvs/claudenet/bin/python 40_label_efficiency.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import _clib as C
import _ensemble as E
import _minelib as ML

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.0]
EMB = C.EMB


class MLPProbe(nn.Module):
    def __init__(self, dim, hidden=256, k=2, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(p), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


def train_probe(Xtr, ytr, device, epochs=200):
    xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xn = (Xtr - xmu) / xsd
    rng = np.random.default_rng(C.SEED)
    va = np.zeros(len(Xn), bool)
    for lab in (0, 1):
        idx = np.where(ytr == lab)[0]
        if len(idx) >= 5:
            va[rng.choice(idx, max(1, int(0.2 * len(idx))), replace=False)] = True
    head = MLPProbe(Xn.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    cw = torch.tensor([1.0, float((ytr == 0).sum() / max((ytr == 1).sum(), 1))], device=device)
    lossf = nn.CrossEntropyLoss(weight=cw)
    Xt = torch.from_numpy(Xn[~va].astype(np.float32)).to(device)
    yt = torch.from_numpy(ytr[~va]).to(device)
    Xv = torch.from_numpy(Xn[va].astype(np.float32)).to(device)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xt), device=device)
        for s in range(0, len(Xt), 256):
            b = perm[s:s + 256]
            opt.zero_grad(); lossf(head(Xt[b]), yt[b]).backward(); opt.step()
        head.eval()
        with torch.no_grad():
            pv = torch.softmax(head(Xv), 1)[:, 1].cpu().numpy()
        auc = roc_auc_score(ytr[va], pv) if len(np.unique(ytr[va])) > 1 else 0.5
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= 30:
                break
    head.load_state_dict(best_state); head.eval()
    return head, xmu, xsd


def probe_scores(head, xmu, xsd, X, device):
    with torch.no_grad():
        return torch.softmax(head(torch.from_numpy(((X - xmu) / xsd).astype(np.float32)).to(device)), 1)[:, 1].cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(C.SEED)

    # --- AION arm: reuse trainpool + eval embeddings ---
    man = pd.read_parquet(EMB / "aion_in_trainpool_manifest.parquet")
    Xtr = np.load(EMB / "aion_emb_trainpool_base.npy").astype(np.float32)
    y = man["label"].to_numpy()
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    Xev = {sp: np.load(EMB / f"aion_emb_{sp}_base.npy").astype(np.float32)
           for sp in ("testneg", "storfer", "inchausti")}

    # --- shielded arm: cache the same trainpool cutouts + eval sets ---
    man = man.copy()
    man["fits_dir"] = man["fits_path"].apply(lambda p: str(Path(p).parent))
    pos_rows = man[man.label == 1][["row_id", "label", "fits_dir"]].reset_index(drop=True)
    neg_rows = man[man.label == 0][["row_id", "label", "fits_dir"]].reset_index(drop=True)
    val = pd.read_parquet(C.DATA / "eval_val.parquet")[["row_id", "label", "fits_dir"]]
    val["fits_dir"] = val["fits_dir"].apply(lambda p: str(C.DATA / Path(str(p)).name))
    ev_rows = {}
    for sp, f in (("testneg", "eval_testneg.parquet"), ("storfer", "eval_storfer.parquet"),
                  ("inchausti", "eval_inchausti.parquet")):
        d = pd.read_parquet(C.DATA / f)[["row_id", "label", "fits_dir"]]
        if "cutouts" not in str(d.fits_dir.iloc[0]):
            d["fits_dir"] = d["fits_dir"].apply(lambda p: str(C.DATA / Path(str(p)).name))
        ev_rows[sp] = d
    print("[le] caching shielded cutouts ...")
    cache = ML.load_cache(pos_rows, neg_rows, val, ev_rows["testneg"], ev_rows["storfer"], ev_rows["inchausti"])

    rows = []
    for f in FRACTIONS:
        npos = max(5, int(round(f * len(pos_idx))))
        sel = rng.choice(len(pos_idx), npos, replace=False)   # positions into pos_idx / pos_rows
        sub_pos = pos_idx[sel]
        # AION probe
        Xa = np.concatenate([Xtr[sub_pos], Xtr[neg_idx]], 0)
        ya = np.concatenate([np.ones(npos), np.zeros(len(neg_idx))]).astype(np.int64)
        head, xmu, xsd = train_probe(Xa, ya, device)
        neg_s = probe_scores(head, xmu, xsd, Xev["testneg"], device)
        for cat in ("storfer", "inchausti"):
            cs = probe_scores(head, xmu, xsd, Xev[cat], device)
            r = E.recovery_at_fpr(neg_s, cs)
            rows.append({"method": "aion_probe", "frac": f, "n_pos": npos, "cat": cat,
                         "rec_1": r[0.01]["recovery"], "rec_01": r[0.001]["recovery"]})
        # shielded from scratch on the same objects
        sub_pos_rows = pos_rows.iloc[sel]
        tr = pd.concat([sub_pos_rows, neg_rows], ignore_index=True)
        m, mean, std, _ = ML.train_shielded(tr, val, cache, device, epochs=30, aug_seed=800 + int(f * 100))
        neg_s = ML.score(m, ev_rows["testneg"], cache, mean, std, device)
        for cat in ("storfer", "inchausti"):
            cs = ML.score(m, ev_rows[cat], cache, mean, std, device)
            r = E.recovery_at_fpr(neg_s, cs)
            rows.append({"method": "shielded", "frac": f, "n_pos": npos, "cat": cat,
                         "rec_1": r[0.01]["recovery"], "rec_01": r[0.001]["recovery"]})
        print(f"[le] frac={f:.2f} ({npos} pos) done")

    df = pd.DataFrame(rows)
    df.to_csv(C.DATA / "label_efficiency.csv", index=False)
    piv = df[df.cat == "storfer"].pivot(index="n_pos", columns="method", values="rec_1")
    print("\nStorfer recovery@1%FPR vs #labeled positives:")
    print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
    (C.DATA / "label_efficiency_summary.json").write_text(json.dumps(
        df.to_dict("records"), indent=2))
    print("[40] wrote label_efficiency.csv")


if __name__ == "__main__":
    raise SystemExit(main())
