#!/usr/bin/env python3
"""321_lensvmimic_head.py — ClaudeNet v3 A3: the lens-vs-MIMIC discriminator head.

A6 showed the equal-weight `average` combiner is suboptimal for mimic separation (dropping the
weak l18 member by hand — v3_noR46 — jumped seed mimic-recovery 0.235->0.528). A learned combiner
should find the optimal member weighting automatically. This trains a logistic head on the member
pc scores as features, positives = confirmed lenses (storfer+inchausti), negatives = the lens-mimic
bank, and evaluates recovery@matched-mimic-FPR LEAKAGE-FREE via stratified k-fold out-of-fold
prediction (train on 4/5 of lenses+mimics, predict the held-out 1/5; recovery is computed on OOF
scores only). A full-data fit is then applied to the OOD seed bank. As a stage-2 survivor re-ranker
it lifts mimic discrimination at NO stage-1 recall cost (it only re-orders survivors). Compared to
the A6 average-combiner rosters; the learned member weights are reported (does it down-weight the
l18 member, recovering v3_noR46 automatically and optimally?).

    /home2/benson/.venvs/claudenet/bin/python 321_lensvmimic_head.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C
import _ensemble as E
mm = C._load("cn_321_mm", C.ROOT / "301_mimic_metric.py")
s317 = C._load("cn_321_317", C.ROOT / "317_seed_compare.py")

V2, V3 = C.DATA / "v2", C.DATA / "v3"
POS = ("storfer", "inchausti")
FEATS = ["effnet_B", "zoobot_N", "effnet_S2_hard", "effnet_S2_b50",
         "effnet_B3_hard", "effnet_B3_b50", "resnet46_C_hard", "resnet46_C_b50"]
POS_PARQ = {"effnet_B": "data/scores_member_effnet_B.parquet",
            "zoobot_N": "data/v2/scores_member_zoobot_N.parquet",
            "effnet_S2_hard": "data/v2/scores_member_effnet_S2_hard.parquet",
            "effnet_B3_hard": "data/v2/scores_member_effnet_B3_hard.parquet",
            "resnet46_C_hard": "data/v2/scores_member_resnet46_C_hard.parquet",
            "effnet_S2_b50": "data/v2/scores_member_effnet_S2_b50.parquet",
            "effnet_B3_b50": "data/v2/scores_member_effnet_B3_b50.parquet",
            "resnet46_C_b50": "data/v2/scores_member_resnet46_C_b50.parquet"}
STORED_SEED = {"effnet_B", "zoobot_N", "effnet_S2_hard", "effnet_B3_hard", "resnet46_C_hard"}
B50_BASE = {"effnet_S2_b50": "effnet_S2", "effnet_B3_b50": "effnet_B3", "resnet46_C_b50": "resnet46_C"}


def lens_feats(sp):
    mats = None
    for m in FEATS:
        d = pd.read_parquet(C.ROOT / POS_PARQ[m])
        g = d[d["split"] == sp][["row_id", "pc"]].rename(columns={"pc": m}).astype({"row_id": str})
        mats = g if mats is None else mats.merge(g, on="row_id", how="inner")
    return mats[FEATS].to_numpy(float)


def dr10_feats():
    mats = None
    for m in FEATS:
        d = pd.read_parquet(V3 / f"scores_member_{m}_mimiceval.parquet")[["row_id", "pc"]]
        g = d.rename(columns={"pc": m}).astype({"row_id": str})
        mats = g if mats is None else mats.merge(g, on="row_id", how="inner")
    return mats[FEATS].to_numpy(float)


def seed_feats(seed, device):
    cols = {}
    for m in FEATS:
        if m in STORED_SEED:
            cal = s317.iso_from_val(POS_PARQ[m])
            cols[m] = cal.transform(seed[f"member_{m}"].to_numpy(float))
        else:
            ck = V2 / "ckpt" / f"member_{B50_BASE[m]}_b50.pt"
            cols[m] = s317.score_seed(ck, POS_PARQ[m], seed, device)
    return np.column_stack([cols[m] for m in FEATS])


def recov(mim_s, pos_s):
    r = mm.recovery_at_mimic_fpr(mim_s, pos_s)
    return {p: r[k]["recovery"] for p, k in (("0.05", 0.05), ("0.01", 0.01))}


def main() -> int:
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(V3 / "a3_lensvmimic_head.json"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xs = {sp: lens_feats(sp) for sp in POS}                    # lens features per split
    Xm = dr10_feats()                                          # DR10 mimic features (negatives)
    seed = pd.read_parquet(V3 / "mimic_bank_seed.parquet").astype({"row_id": str})
    Xseed = seed_feats(seed, device)
    print(f"[321] lenses storfer={len(Xs['storfer'])} inchausti={len(Xs['inchausti'])}; "
          f"dr10 mimics={len(Xm)}; seed mimics={len(Xseed)}; features={len(FEATS)}")

    # logit-space features for the linear head
    def lg(x): return np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    Xs_l = {sp: lg(Xs[sp]) for sp in POS}; Xm_l = lg(Xm); Xseed_l = lg(Xseed)

    # combined lens set, tagged by split, for joint k-fold with the mimics
    Xpos = np.vstack([Xs_l[sp] for sp in POS])
    pos_tag = np.concatenate([[sp] * len(Xs_l[sp]) for sp in POS])
    X = np.vstack([Xpos, Xm_l]); y = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xm_l))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    oof = np.full(len(X), np.nan)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    oof_pos, oof_mim = oof[:len(Xpos)], oof[len(Xpos):]

    rep = {"features": FEATS, "splits": {}}
    print("\n[321] HEAD recovery@mimic-FPR (leakage-free OOF) vs A6 average-combiner baselines")
    base = json.loads((V3 / "a6_ensemble_refit.json").read_text())["rosters"]
    for sp in POS:
        hs = oof_pos[pos_tag == sp]                            # OOF head scores for this lens split
        head_dr10 = recov(oof_mim, hs)
        rep["splits"][sp] = {"head_dr10": head_dr10,
                             "v3blend8_dr10": base["v3blend8"]["splits"][sp]["dr10"],
                             "v3_noR46_dr10": base["v3_noR46"]["splits"][sp]["dr10"]}
        print(f"  {sp:10s} dr10@0.05 head={head_dr10['0.05']:.3f}  "
              f"(v3blend8={base['v3blend8']['splits'][sp]['dr10']['0.05']:.3f}, "
              f"v3_noR46={base['v3_noR46']['splits'][sp]['dr10']['0.05']:.3f})   "
              f"@0.01 head={head_dr10['0.01']:.3f} "
              f"(noR46={base['v3_noR46']['splits'][sp]['dr10']['0.01']:.3f})")

    # full-fit -> OOD seed (head trained on ALL lenses+dr10mimics, applied to unseen seed)
    clf_full = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(X, y)
    seed_score = clf_full.decision_function(Xseed_l)
    print("\n[321] HEAD on OOD seed (full-fit, applied to unseen 601 seed):")
    for sp in POS:
        # use held-out OOF lens scores re-scored by the full model for an honest pos ref? Use full-fit
        pos_full = clf_full.decision_function(Xs_l[sp])
        hs = recov(seed_score, pos_full)
        rep["splits"][sp]["head_seed"] = hs
        rep["splits"][sp]["v3_noR46_seed"] = base["v3_noR46"]["splits"][sp]["seed"]
        print(f"  {sp:10s} seed@0.05 head={hs['0.05']:.3f} @0.01={hs['0.01']:.3f}  "
              f"(v3_noR46 seed={base['v3_noR46']['splits'][sp]['seed']['0.05']:.3f}/"
              f"{base['v3_noR46']['splits'][sp]['seed']['0.01']:.3f}, "
              f"v2lean={base['v2lean']['splits'][sp]['seed']['0.05']:.3f})")

    w = dict(zip(FEATS, clf_full.coef_[0]))
    rep["learned_weights"] = w
    print("\n[321] learned head weights (does it down-weight the l18 resnet?):")
    for m, c in sorted(w.items(), key=lambda kv: -kv[1]):
        print(f"     {m:18s} {c:+.3f}")
    Path(args.out).write_text(json.dumps(rep, indent=2, default=float))
    print(f"\n[321] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
