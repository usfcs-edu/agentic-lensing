#!/usr/bin/env python3
"""367_model_progression.py — does v3 beat the ORIGINAL (Storfer/Inchausti-class) models on
candidate QUALITY at DECaLS resolution (no Euclid)?

The original published lens finders are single CNNs / ensembles trained on RANDOM negatives.
ClaudeNet's lineage: effnet_B (a v1 member, random negatives) ≈ an original-class EfficientNet;
v2-lean adds CNN-high hard-negative mining; v3 adds contaminant-aware (typed-mimic) mining + a
learned lens-vs-mimic head. The right "quality without Euclid" metric is recovery@matched-mimic-FPR
(how many real lenses you keep at a fixed mimic-contamination level) — purely a DECaLS-resolution
measure. We also report grade-selectivity on the INDEPENDENT Euclid Q1 sample: AUC separating
Euclid grade-A lenses from grade-C among the EDF galaxies (can the model rank real lenses above
ambiguous ones?).

Models compared on the SAME held-out Storfer/Inchausti positives + the same mimic sets:
  orig (effnet_B, random neg)  ->  v2lean (5, hard-mined)  ->  v3blend8 (8)  ->  v3head (learned).

    /home2/benson/.venvs/claudenet/bin/python 367_model_progression.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import _clib as C
import _ensemble as E
mm = C._load("cn_367_mm", C.ROOT / "301_mimic_metric.py")
s317 = C._load("cn_367_317", C.ROOT / "321_lensvmimic_head.py")  # FEATS, lens_feats, dr10_feats
si317 = C._load("cn_367_si", C.ROOT / "317_seed_compare.py")     # iso_from_val, score_seed-not-needed
V3 = C.DATA / "v3"; REPRO = C.ROOT.parent
EUCLID_CAT = REPRO / "euclid-q1" / "data" / "raw" / "q1_discovery_engine_lens_catalog.csv"
POS = ("storfer", "inchausti")
V2 = ["effnet_B", "effnet_S2_hard", "effnet_B3_hard", "resnet46_C_hard", "zoobot_N"]
V3M = s317.FEATS  # 8
POS_PARQ = {"effnet_B": "data/scores_member_effnet_B.parquet",
            "zoobot_N": "data/v2/scores_member_zoobot_N.parquet",
            "effnet_S2_hard": "data/v2/scores_member_effnet_S2_hard.parquet",
            "effnet_B3_hard": "data/v2/scores_member_effnet_B3_hard.parquet",
            "resnet46_C_hard": "data/v2/scores_member_resnet46_C_hard.parquet",
            "effnet_S2_b50": "data/v2/scores_member_effnet_S2_b50.parquet",
            "effnet_B3_b50": "data/v2/scores_member_effnet_B3_b50.parquet",
            "resnet46_C_b50": "data/v2/scores_member_resnet46_C_b50.parquet"}
STORED_SEED = {"effnet_B", "zoobot_N", "effnet_S2_hard", "effnet_B3_hard", "resnet46_C_hard"}
_lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))


def pos_pc(member, sp):
    d = pd.read_parquet(C.ROOT / POS_PARQ[member]); g = d[d.split == sp][["row_id", "pc"]]
    return g.rename(columns={"pc": member}).astype({"row_id": str})

def ens_pos(members, sp):
    mats = None
    for m in members:
        g = pos_pc(m, sp); mats = g if mats is None else mats.merge(g, on="row_id", how="inner")
    return mats[members].to_numpy(float)

def eval_pc(members):  # DR10 held-out mimic-eval
    mats = None
    for m in members:
        d = pd.read_parquet(V3 / f"scores_member_{m}_mimiceval.parquet")[["row_id", "pc"]]
        g = d.rename(columns={"pc": m}).astype({"row_id": str})
        mats = g if mats is None else mats.merge(g, on="row_id", how="inner")
    return mats[members].to_numpy(float)

def combine(X, kind, clf=None):
    if kind == "mean":
        return X.mean(axis=1)
    return clf.decision_function(_lg(X))   # head


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import torch
    seed = pd.read_parquet(V3 / "mimic_bank_seed.parquet").astype({"row_id": str})
    device = torch.device("cpu")

    # fit the head on lenses vs DR10 mimic bank (logit features), like 321/363
    Xpos = np.vstack([s317.lens_feats(sp) for sp in POS]); Xm = s317.dr10_feats()
    head = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(
        _lg(np.vstack([Xpos, Xm])), np.r_[np.ones(len(Xpos)), np.zeros(len(Xm))])

    # seed member pc: stored for v2/frozen, re-score b50 (317.score_seed)
    seedcols = {}
    for m in V3M:
        if m in STORED_SEED:
            seedcols[m] = si317.iso_from_val(POS_PARQ[m]).transform(seed[f"member_{m}"].to_numpy(float))
        else:
            ck = C.DATA / "v2" / "ckpt" / f"member_{m.replace('_b50','')}_b50.pt"
            seedcols[m] = si317.score_seed(ck, POS_PARQ[m], seed, device)

    MODELS = {"orig (effnet_B, random-neg)": (["effnet_B"], "mean"),
              "v2lean (hard-mined, 5)": (V2, "mean"),
              "v3blend8 (8)": (V3M, "mean"),
              "v3head (learned)": (V3M, "head")}

    # ---- recovery@mimic-FPR(0.05) on seed + DR10-eval ----
    print("recovery@mimic-FPR(0.05)  [higher = more real lenses kept at fixed mimic contamination]")
    print(f"  {'model':28s} | {'seed storfer/inch':>20s} | {'DR10eval storfer/inch':>22s}")
    rep = {}
    for name, (mem, kind) in MODELS.items():
        row = {}
        # seed
        Xs = np.column_stack([seedcols[m] for m in mem]); ssc = combine(Xs, kind, head)
        # dr10 eval
        Xe = eval_pc(mem); esc = combine(Xe, kind, head)
        out = []
        for label, sc in (("seed", ssc), ("dr10", esc)):
            for sp in POS:
                Xp = ens_pos(mem, sp); psc = combine(Xp, kind, head)
                r = mm.recovery_at_mimic_fpr(sc, psc)[0.05]["recovery"]
                row[f"{label}_{sp}"] = r
        rep[name] = row
        print(f"  {name:28s} | {row['seed_storfer']:.3f} / {row['seed_inchausti']:.3f}      "
              f"| {row['dr10_storfer']:.3f} / {row['dr10_inchausti']:.3f}")

    # ---- Euclid grade-selectivity: AUC(grade-A vs grade-C) on EDF galaxies (independent sample) ----
    base = pd.read_parquet(V3 / "cv3_edf_parent_ids.parquet").astype({"row_id": str})
    df = base[["row_id", "RA", "DEC"]].copy()
    for m in V3M:
        s = pd.read_parquet(V3 / f"scores_member_{m}_edf.parquet")[["row_id", "pc"]]
        df = df.merge(s.rename(columns={"pc": m}).astype({"row_id": str}), on="row_id", how="inner")
    q = pd.read_csv(EUCLID_CAT)
    cq = SkyCoord(q.right_ascension.values * u.deg, q.declination.values * u.deg)
    cd = SkyCoord(df.RA.values * u.deg, df.DEC.values * u.deg)
    iq, sep, _ = cd.match_to_catalog_sky(cq)
    df["egrade"] = np.where(sep.arcsec < 3, q.grade.values[iq], None)
    AC = df[df.egrade.isin(["A", "C"])].copy(); y = (AC.egrade == "A").astype(int).to_numpy()
    print(f"\nEuclid grade-selectivity AUC(A vs C) on EDF  [n_A={int(y.sum())}, n_C={int((1-y).sum())};"
          " higher = ranks real Euclid lenses above ambiguous ones]")
    for name, (mem, kind) in MODELS.items():
        Xa = AC[mem].to_numpy(float); sc = combine(Xa, kind, head)
        auc = roc_auc_score(y, sc); rep[name]["euclid_AvsC_auc"] = float(auc)
        print(f"  {name:28s}  AUC = {auc:.3f}")
    (V3 / "model_progression.json").write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {V3/'model_progression.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
