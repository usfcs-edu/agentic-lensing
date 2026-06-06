#!/usr/bin/env python3
"""60_domain_adapt.py — Phase 5: unsupervised domain adaptation for the DECaLS
north<->south shift (the repo handles this ad-hoc with north negatives).

SOURCE (labeled) = SOUTH positives + negatives. TARGET (unlabeled) = NORTH
negatives. We train a shielded net (a) source-only and (b) with an MMD penalty
aligning source/target negative FEATURES (pooled penultimate activations), then
compare recovery on held-out NORTH vs SOUTH Storfer lenses. If MMD lifts NORTH
(target) recovery without hurting SOUTH, principled UDA beats ad-hoc transfer.

Real domain shift, real data (no sims). Caveat: only ~80 held-out north lenses ->
north recovery is noisy; we report it honestly.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      /home2/benson/.venvs/claudenet/bin/python 60_domain_adapt.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import _clib as C
import _ensemble as E
import _minelib as ML

DEC_SPLIT = 32.375
EPOCHS = 20
LAMBDA_MMD = 1.0


def feat_logit(model, x):
    x = model.stem_conv(x); x = F.elu(model.stem_bn(x))
    x = model.body(x); x = F.elu(model.bn_final(x))
    feat = model.pool(x).flatten(1)
    return feat, model.fc(feat).squeeze(-1)


def mmd_rbf(X, Y, sigmas=(1.0, 2.0, 4.0, 8.0)):
    def d2(A, B):
        return (A * A).sum(1, keepdim=True) - 2 * A @ B.t() + (B * B).sum(1).unsqueeze(0)
    def k(D2):
        return sum(torch.exp(-D2 / (2 * s * s)) for s in sigmas)
    return k(d2(X, X)).mean() + k(d2(Y, Y)).mean() - 2 * k(d2(X, Y)).mean()


def prep():
    """Return labeled south df, unlabeled north-neg df, and eval dfs (n/s storfer, n/s testneg)."""
    split = pd.read_parquet(C.DATA / "training_split_staged.parquet").copy()
    split["fits_dir"] = split["fits_dir"].apply(lambda p: str(C.DATA / Path(str(p)).name))
    foot = pd.read_parquet(C.DATA / "negatives_extra.parquet")[["row_id", "footprint"]]
    fmap = dict(zip(foot.row_id.astype(str), foot.footprint))
    cur = pd.read_parquet(C.DATA / "positives_curated.parquet")
    posdec = dict(zip(cur["name"].astype(str), cur["DEC"]))

    def is_north_row(r):
        if r.label == 1:
            d = posdec.get(str(r.row_id), np.nan)
        else:
            return fmap.get(str(r.row_id), "south") == "north"
        return d > DEC_SPLIT
    split["north"] = split.apply(is_north_row, axis=1)

    tr = split[split.split == "train"]
    south_pos = tr[(tr.label == 1) & (~tr.north)]
    south_neg = tr[(tr.label == 0) & (~tr.north)]
    north_neg = tr[(tr.label == 0) & (tr.north)]
    rng = np.random.default_rng(C.SEED)
    south_neg = south_neg.iloc[rng.choice(len(south_neg), min(15000, len(south_neg)), replace=False)]
    north_neg = north_neg.iloc[rng.choice(len(north_neg), min(8000, len(north_neg)), replace=False)]

    val = pd.read_parquet(C.DATA / "eval_val.parquet").copy()
    val["fits_dir"] = val["fits_dir"].apply(lambda p: str(C.DATA / Path(str(p)).name))

    testneg = split[(split.split == "test") & (split.label == 0)].copy()
    # storfer candidates with dec
    cat = pd.read_csv(C.DATA / "storfer2024_published_catalog.csv")
    st = pd.DataFrame({"row_id": cat["name"], "label": 1, "DEC": cat["DEC"]})
    st["fits_dir"] = str(C.DATA / "cutouts_fits_candidates_storfer")
    st["north"] = st["DEC"] > DEC_SPLIT
    return south_pos, south_neg, north_neg, val, testneg, st


def train_da(south_df, north_neg, val, cache, device, lam, seed=900):
    torch.manual_seed(seed); np.random.seed(seed)
    mean, std = ML.band_stats(south_df, cache)
    model = C.models()["ShieldedDeepLens"](in_channels=3, **C.CFG194).to(device)
    tr = torch.utils.data.DataLoader(ML.CachedDS(south_df, cache, mean, std, True),
                                     batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    nn_neg = torch.utils.data.DataLoader(ML.CachedDS(north_neg, cache, mean, std, True),
                                         batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, 8, 0.1)
    bce = nn.BCEWithLogitsLoss()
    yv = val["label"].to_numpy()
    best, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        nit = iter(nn_neg)
        for x, y in tr:
            x = x.to(device)
            feat_s, logit_s = feat_logit(model, x)
            loss = bce(logit_s, y.to(device))
            if lam > 0:
                try:
                    xn, _ = next(nit)
                except StopIteration:
                    nit = iter(nn_neg); xn, _ = next(nit)
                # align SOURCE negatives vs TARGET negatives in feature space
                neg_mask = (y == 0)
                if neg_mask.sum() > 4:
                    feat_tn, _ = feat_logit(model, xn.to(device))
                    loss = loss + lam * mmd_rbf(feat_s[neg_mask], feat_tn)
            opt.zero_grad(); loss.backward(); opt.step()
        # val auc
        model.eval(); ps = []
        va = torch.utils.data.DataLoader(ML.CachedDS(val, cache, mean, std, False), batch_size=512, num_workers=0)
        with torch.no_grad():
            for xv, _ in va:
                ps.append(torch.sigmoid(model(xv.to(device))).cpu().numpy())
        auc = roc_auc_score(yv, np.concatenate(ps))
        if auc > best:
            best, best_state = auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        sched.step()
    model.load_state_dict(best_state); model.eval()
    return model, mean, std


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    south_pos, south_neg, north_neg, val, testneg, st = prep()
    south_df = pd.concat([south_pos[["row_id", "label", "fits_dir"]],
                          south_neg[["row_id", "label", "fits_dir"]]], ignore_index=True)
    print(f"[da] source south: {len(south_pos)} pos + {len(south_neg)} neg; "
          f"target north-neg: {len(north_neg)}")
    print(f"[da] eval storfer north={int(st.north.sum())} south={int((~st.north).sum())}")

    tn_s = testneg[~testneg.north][["row_id", "label", "fits_dir"]]
    tn_n = testneg[testneg.north][["row_id", "label", "fits_dir"]]
    st_s = st[~st.north][["row_id", "label", "fits_dir"]]
    st_n = st[st.north][["row_id", "label", "fits_dir"]]
    print(f"[da] test negs north={len(tn_n)} south={len(tn_s)}")

    cache = ML.load_cache(south_df, north_neg[["row_id", "label", "fits_dir"]], val,
                          tn_s, tn_n, st_s, st_n)

    def evalmodel(model, mean, std):
        def rec(neg_df, cand_df):
            n = ML.score(model, neg_df, cache, mean, std, device)
            c = ML.score(model, cand_df, cache, mean, std, device)
            return E.recovery_at_fpr(n, c, fprs=(0.01,))[0.01]["recovery"]
        return {"south": rec(tn_s, st_s), "north": rec(tn_n, st_n)}

    res = {}
    for tag, lam in (("source_only", 0.0), ("mmd_da", LAMBDA_MMD)):
        print(f"\n[da] training {tag} (lambda={lam}) ...")
        m, mean, std = train_da(south_df, north_neg[["row_id", "label", "fits_dir"]], val, cache, device, lam)
        res[tag] = evalmodel(m, mean, std)
        print(f"[da] {tag}: south@1%={res[tag]['south']:.3f}  north@1%={res[tag]['north']:.3f}")

    d_north = res["mmd_da"]["north"] - res["source_only"]["north"]
    d_south = res["mmd_da"]["south"] - res["source_only"]["south"]
    res["mmd_minus_source_north"] = d_north
    res["mmd_minus_source_south"] = d_south
    res["verdict"] = "DA-HELPS" if d_north > 0.005 else "NO-EFFECT"
    (C.DATA / "domain_adapt.json").write_text(json.dumps(res, indent=2))
    print(f"\n[da] MMD - source: NORTH(target) {d_north:+.3f}, SOUTH {d_south:+.3f}")
    print(f"[da] VERDICT: {res['verdict']}  (north lenses n={len(st_n)} -> noisy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
