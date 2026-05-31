#!/usr/bin/env python3
"""
21_train_stagec.py — Phase-5 Stage C (negative scale-up + FPR).

Lift the negative:positive ratio from Stage-B's ~2.5:1 toward the papers'
~33:1 (Storfer) by adding the ~45K brick-sliced DR9 random-galaxy negatives
(20), and quantify the effect on the FALSE-POSITIVE RATE and operating threshold
— the variable the small Stage-A/B negative set could not constrain.

Steps:
  1. Build the enlarged split: positives = 949 + 1,012 literature (same as Stage B);
     negatives = 5,000 original + ~45K extra (per-row fits_dir). 70/20/10 split;
     the test-split negatives are the held-out FPR evaluation set.
  2. FPR(before): score the held-out negatives with the EXISTING Stage-B models
     (no retrain) — the current false-positive rate on realistic negatives.
  3. Retrain shielded + EfficientNetV2 + meta on the enlarged set (reduced epochs:
     the ~9x larger set needs fewer passes; documented vs Stage A/B's 130/60).
  4. FPR(after) + operating threshold (negative-score percentile) + recovery of
     the published catalogues with the Stage-C models.

Outputs (suffix _stagec): checkpoints, meta_metrics_stagec.json, and a printed
Stage-B-vs-Stage-C table (test AUC, FPR@0.5/0.9, threshold@1%/0.1% FPR, recovery).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import _trainlib as TL
import _scorelib as SL


def _imp(name, fn):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


S19 = _imp("stageb", "19_train_stageb.py")          # reuse train_base, band_stats, CFG194
_meta = _imp("meta_learner", "03_meta_learner.py")
MetaLearner, simple_average = _meta.MetaLearner, _meta.simple_average

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def build_stagec_split(target_pos=1961):
    pos = pd.read_parquet(DATA / "positives_huang2020.parquet")
    rows = [{"row_id": r["Name"], "label": 1, "RA": r["RA"], "DEC": r["DEC"],
             "fits_dir": str(DATA / "cutouts_fits_dr9")} for _, r in pos.iterrows()]
    mf = pd.read_csv(DATA / "litpos_cutout_manifest.csv")
    good = mf[(mf["status"].isin(["ok", "skip"])) & (mf["flux_std"] > 1e-3)].reset_index(drop=True)
    n_take = max(0, target_pos - len(pos))
    if len(good) > n_take:
        good = good.sample(n=n_take, random_state=TL.SEED).reset_index(drop=True)
    for _, r in good.iterrows():
        rows.append({"row_id": r["Name"], "label": 1, "RA": r["RA"], "DEC": r["DEC"],
                     "fits_dir": str(DATA / "cutouts_fits_litpos_dr9")})
    # negatives: original 5,000 + extra brick negatives
    neg = pd.read_parquet(DATA / "negatives.parquet")
    for _, r in neg.iterrows():
        rows.append({"row_id": r["row_id"], "label": 0, "RA": r["RA"], "DEC": r["DEC"],
                     "fits_dir": str(DATA / "cutouts_fits_dr9")})
    extra = pd.read_parquet(DATA / "negatives_extra.parquet")
    for _, r in extra.iterrows():
        rows.append({"row_id": r["row_id"], "label": 0, "RA": r["RA"], "DEC": r["DEC"],
                     "fits_dir": str(DATA / "cutouts_fits_neg_dr9")})
    df = pd.DataFrame(rows)
    df["ok"] = df.apply(lambda r: (Path(r["fits_dir"]) / f"{r['row_id']}.fits").exists(), axis=1)
    df = df[df["ok"]].drop(columns="ok").drop_duplicates("row_id").reset_index(drop=True)
    rng = np.random.default_rng(TL.SEED)
    parts = []
    for lab in (0, 1):
        idx = df.index[df.label == lab].to_numpy().copy(); rng.shuffle(idx)
        n = len(idx); ntr, nv = int(round(.7 * n)), int(round(.2 * n))
        b = np.array(["test"] * n, dtype=object); b[:ntr] = "train"; b[ntr:ntr + nv] = "val"
        parts.append(pd.DataFrame({"index": idx, "split": b}))
    a = pd.concat(parts).sort_values("index"); df["split"] = a["split"].values
    npos = int(df.label.sum()); nneg = int((df.label == 0).sum())
    print(f"[split] {len(df)} rows: {npos} pos : {nneg} neg  (ratio 1:{nneg/max(npos,1):.0f})")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    return df


def fpr_stats(probs, thresholds=(0.5, 0.9)):
    p = probs[np.isfinite(probs)]
    out = {f"fpr@{t}": float((p >= t).mean()) for t in thresholds}
    out["thresh@1%fpr"] = float(np.quantile(p, 0.99))
    out["thresh@0.1%fpr"] = float(np.quantile(p, 0.999))
    return out


def score_test_negs(df, ckpt, device):
    neg = df[(df.split == "test") & (df.label == 0)]
    paths = [Path(r.fits_dir) / f"{r.row_id}.fits" for r in neg.itertuples()]
    m, arch, mean, std, _ = SL.load_checkpoint_model(DATA / ckpt, device)
    return SL.score_paths(paths, m, arch, mean, std, device), m, arch, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shielded-epochs", type=int, default=50)
    ap.add_argument("--effnet-epochs", type=int, default=30)
    args = ap.parse_args()
    torch.manual_seed(TL.SEED); np.random.seed(TL.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = build_stagec_split()
    df.to_parquet(DATA / "training_split_stagec.parquet", index=False)
    n_test_neg = int(((df.split == "test") & (df.label == 0)).sum())

    # --- FPR BEFORE: existing Stage-B models on the held-out negatives ---
    print(f"\n[FPR before] Stage-B models on {n_test_neg} held-out negatives")
    fpr_before = {}
    if (DATA / "checkpoint_best_shielded194k_stageb.pt").exists():
        prB, *_ = score_test_negs(df, "checkpoint_best_shielded194k_stageb.pt", device)
        peB, *_ = score_test_negs(df, "checkpoint_best_efficientnet_stageb.pt", device)
        ckm = torch.load(str(DATA / "checkpoint_best_meta_stageb.pt"), map_location="cpu", weights_only=False)
        mB = MetaLearner().to(device); mB.load_state_dict(ckm["state_dict"]); mB.eval()
        Pb = np.stack([prB, peB], 1).astype(np.float32); ok = np.isfinite(Pb).all(1)
        pmB = np.full(len(Pb), np.nan, np.float32)
        with torch.no_grad():
            pmB[ok] = torch.sigmoid(mB(torch.from_numpy(Pb[ok]).to(device))).cpu().numpy()
        fpr_before = {"shielded": fpr_stats(prB), "effnet": fpr_stats(peB), "meta": fpr_stats(pmB)}
        for k, v in fpr_before.items():
            print(f"  {k:9s} fpr@0.5={v['fpr@0.5']:.3f} fpr@0.9={v['fpr@0.9']:.3f} "
                  f"thr@1%fpr={v['thresh@1%fpr']:.3f}")

    # --- retrain Stage C ---
    mean, std = S19.band_stats(df[df.split == "train"])
    print(f"\n[train] shielded194k_stagec ({args.shielded_epochs} ep)")
    sh, sh_v, sh_t, _, _ = S19.train_base("shielded", df, mean, std, device,
                                          args.shielded_epochs, 128, 1e-3, max(10, args.shielded_epochs//3))
    torch.save({"state_dict": sh.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                "arch": "shielded", "shielded_cfg": S19.CFG194, "val_auc": sh_v},
               DATA / "checkpoint_best_shielded194k_stagec.pt")
    print(f"  shielded_stagec val={sh_v:.4f} test={sh_t:.4f}")
    print(f"\n[train] efficientnet_stagec ({args.effnet_epochs} ep)")
    ef, ef_v, ef_t, _, _ = S19.train_base("efficientnet", df, mean, std, device,
                                          args.effnet_epochs, 128, 1e-3, max(10, args.effnet_epochs//3), accum=2)
    torch.save({"state_dict": ef.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                "arch": "efficientnet", "variant": ef.variant, "head_dim": ef.head_dim,
                "num_classes": ef.num_classes, "val_auc": ef_v},
               DATA / "checkpoint_best_efficientnet_stagec.pt")
    print(f"  effnet_stagec val={ef_v:.4f} test={ef_t:.4f}")

    # meta on stage-C probs
    paths = [Path(r.fits_dir) / f"{r.row_id}.fits" for r in df.itertuples()]
    pr = SL.score_paths(paths, sh, "shielded", mean, std, device)
    pe = SL.score_paths(paths, ef, "efficientnet", mean, std, device)
    P = np.stack([pr, pe], 1).astype(np.float32); y = df.label.to_numpy(np.float32)
    tr, va, te = (df.split == "train").values, (df.split == "val").values, (df.split == "test").values
    meta = MetaLearner().to(device); opt = torch.optim.Adam(meta.parameters(), lr=1e-2)
    Xtr, ytr = torch.from_numpy(P[tr]).to(device), torch.from_numpy(y[tr]).to(device)
    bestv, bst = -1, None
    for _ in range(400):
        meta.train(); opt.zero_grad(); nn.BCEWithLogitsLoss()(meta(Xtr), ytr).backward(); opt.step()
        meta.eval()
        with torch.no_grad():
            pv = torch.sigmoid(meta(torch.from_numpy(P[va]).to(device))).cpu().numpy()
        av = roc_auc_score(y[va], pv)
        if av > bestv: bestv, bst = av, {k: v.cpu().clone() for k, v in meta.state_dict().items()}
    meta.load_state_dict(bst); meta.eval()
    torch.save({"state_dict": meta.state_dict(), "arch": "meta"}, DATA / "checkpoint_best_meta_stagec.pt")
    with torch.no_grad():
        pm = torch.sigmoid(meta(torch.from_numpy(P).to(device))).cpu().numpy()
    meta_t = roc_auc_score(y[te], pm[te])

    # --- FPR AFTER (Stage-C) on the SAME held-out negatives ---
    negmask = te & (y == 0)
    fpr_after = {"shielded": fpr_stats(pr[negmask]), "effnet": fpr_stats(pe[negmask]),
                 "meta": fpr_stats(pm[negmask])}

    # --- recovery of published catalogues with Stage-C models ---
    rec = {}
    for key, cut, csv in (("storfer", "cutouts_fits_candidates_storfer", "storfer2024_published_catalog.csv"),
                          ("inchausti", "cutouts_fits_candidates_inchausti", "inchausti2025_published_catalog.csv")):
        cat = pd.read_csv(DATA / csv)
        cp = [DATA / cut / f"{n}.fits" for n in cat["name"]]
        rr = SL.score_paths(cp, sh, "shielded", mean, std, device)
        re_ = SL.score_paths(cp, ef, "efficientnet", mean, std, device)
        Pc = np.stack([rr, re_], 1).astype(np.float32); okc = np.isfinite(Pc).all(1)
        pmc = np.full(len(Pc), np.nan, np.float32)
        with torch.no_grad():
            pmc[okc] = torch.sigmoid(meta(torch.from_numpy(Pc[okc]).to(device))).cpu().numpy()
        rec[key] = float(np.nanmean(pmc[okc] >= 0.5))

    json.dump({"n_neg_test": n_test_neg, "fpr_before": fpr_before, "fpr_after": fpr_after,
               "shielded_test_auc": sh_t, "effnet_test_auc": ef_t, "meta_test_auc": meta_t,
               "recovery": rec, "ratio": f"1:{int((df.label==0).sum()/max(df.label.sum(),1))}"},
              open(DATA / "meta_metrics_stagec.json", "w"), indent=2)

    print("\n================ Stage B vs Stage C (negative scale-up) ================")
    print(f"neg:pos ratio                      ~2.5:1      1:{int((df.label==0).sum()/max(df.label.sum(),1))}")
    print(f"held-out negatives evaluated       —           {n_test_neg}")
    print(f"{'meta test AUC':32s} {0.9881:>10.4f} {meta_t:>10.4f}")
    if fpr_before:
        print(f"{'meta FPR @0.5  (before -> after)':32s} {fpr_before['meta']['fpr@0.5']:>10.3f} {fpr_after['meta']['fpr@0.5']:>10.3f}")
        print(f"{'meta FPR @0.9  (before -> after)':32s} {fpr_before['meta']['fpr@0.9']:>10.3f} {fpr_after['meta']['fpr@0.9']:>10.3f}")
        print(f"{'meta thresh @1% FPR (before/after)':32s} {fpr_before['meta']['thresh@1%fpr']:>10.3f} {fpr_after['meta']['thresh@1%fpr']:>10.3f}")
    print(f"{'storfer recovery@0.5':32s} {0.935:>10.3f} {rec['storfer']:>10.3f}")
    print(f"{'inchausti recovery@0.5':32s} {0.969:>10.3f} {rec['inchausti']:>10.3f}")
    print("[done] Stage-C checkpoints + meta_metrics_stagec.json written")


if __name__ == "__main__":
    main()
