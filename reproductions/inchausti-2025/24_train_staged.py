#!/usr/bin/env python3
"""
24_train_staged.py — Phase-5 Stage D (closest-achievable faithful run).

Train on the most paper-faithful dataset we can assemble without the papers'
unpublished object-level curation:
  positives = the grade-curated union (23_, 1,961 = Storfer scale; spectroscopic
              SILO + known Stein lenses + grade-A candidates)
  negatives = 65,000 random DR9 galaxies (20_, Storfer's ~33:1 ratio)
Retrain shielded + EfficientNetV2 + meta, then report AUC, false-positive rate,
and recovery at a MATCHED FPR vs Stages B/C (the honest operating point).

Output (suffix _staged): checkpoints, meta_metrics_staged.json, and an
A/B/C/D recovery@1%FPR table.
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


S19 = _imp("stageb", "19_train_stageb.py")
_meta = _imp("meta_learner", "03_meta_learner.py")
MetaLearner, simple_average = _meta.MetaLearner, _meta.simple_average

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def build_staged_split():
    pos = pd.read_parquet(DATA / "positives_curated.parquet")  # name, RA, DEC, ...
    neg = pd.read_parquet(DATA / "negatives_extra.parquet")    # row_id, RA, DEC, ...
    rows = []
    for _, r in pos.iterrows():
        rows.append({"row_id": r["name"], "label": 1, "RA": r["RA"], "DEC": r["DEC"],
                     "fits_dir": str(DATA / "cutouts_fits_curated_dr9")})
    for _, r in neg.iterrows():
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
    npos, nneg = int(df.label.sum()), int((df.label == 0).sum())
    print(f"[split] {len(df)} rows: {npos} curated pos : {nneg} neg (ratio 1:{nneg/max(npos,1):.0f})")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    return df


def recovery_at_fpr(sh, ef, meta, mean, std, device, neg_paths, fprs=(0.01, 0.001)):
    nr = SL.score_paths(neg_paths, sh, "shielded", mean, std, device)
    ne = SL.score_paths(neg_paths, ef, "efficientnet", mean, std, device)
    Pn = np.stack([nr, ne], 1).astype(np.float32); okn = np.isfinite(Pn).all(1)
    with torch.no_grad():
        nm = np.full(len(Pn), np.nan, np.float32)
        nm[okn] = torch.sigmoid(meta(torch.from_numpy(Pn[okn]).to(device))).cpu().numpy()
    negs = {"resnet": nr, "effnet": ne, "meta": nm}
    out = {}
    for key, cut, csv in (("storfer", "cutouts_fits_candidates_storfer", "storfer2024_published_catalog.csv"),
                          ("inchausti", "cutouts_fits_candidates_inchausti", "inchausti2025_published_catalog.csv")):
        cat = pd.read_csv(DATA / csv)
        cp = [DATA / cut / f"{n}.fits" for n in cat["name"]]
        cr = SL.score_paths(cp, sh, "shielded", mean, std, device)
        ce = SL.score_paths(cp, ef, "efficientnet", mean, std, device)
        Pc = np.stack([cr, ce], 1).astype(np.float32); okc = np.isfinite(Pc).all(1)
        cm = np.full(len(Pc), np.nan, np.float32)
        with torch.no_grad():
            cm[okc] = torch.sigmoid(meta(torch.from_numpy(Pc[okc]).to(device))).cpu().numpy()
        cand = {"resnet": cr, "effnet": ce, "meta": cm}
        for fpr in fprs:
            for mdl in ("resnet", "effnet", "meta"):
                ns = negs[mdl][np.isfinite(negs[mdl])]
                thr = float(np.quantile(ns, 1 - fpr))
                cs = cand[mdl][np.isfinite(cand[mdl])]
                out[(key, mdl, fpr)] = (float((cs >= thr).mean()), thr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shielded-epochs", type=int, default=45)
    ap.add_argument("--effnet-epochs", type=int, default=25)
    args = ap.parse_args()
    torch.manual_seed(TL.SEED); np.random.seed(TL.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = build_staged_split()
    df.to_parquet(DATA / "training_split_staged.parquet", index=False)
    mean, std = S19.band_stats(df[df.split == "train"])

    print(f"\n[train] shielded194k_staged ({args.shielded_epochs} ep)")
    sh, sh_v, sh_t, _, _ = S19.train_base("shielded", df, mean, std, device,
                                          args.shielded_epochs, 128, 1e-3, max(10, args.shielded_epochs // 3))
    torch.save({"state_dict": sh.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                "arch": "shielded", "shielded_cfg": S19.CFG194, "val_auc": sh_v},
               DATA / "checkpoint_best_shielded194k_staged.pt")
    print(f"  shielded_staged val={sh_v:.4f} test={sh_t:.4f}")
    print(f"\n[train] efficientnet_staged ({args.effnet_epochs} ep)")
    ef, ef_v, ef_t, _, _ = S19.train_base("efficientnet", df, mean, std, device,
                                          args.effnet_epochs, 128, 1e-3, max(10, args.effnet_epochs // 3), accum=2)
    torch.save({"state_dict": ef.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                "arch": "efficientnet", "variant": ef.variant, "head_dim": ef.head_dim,
                "num_classes": ef.num_classes, "val_auc": ef_v}, DATA / "checkpoint_best_efficientnet_staged.pt")
    print(f"  effnet_staged val={ef_v:.4f} test={ef_t:.4f}")

    # meta on staged probs
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
    torch.save({"state_dict": meta.state_dict(), "arch": "meta"}, DATA / "checkpoint_best_meta_staged.pt")
    with torch.no_grad():
        pm = torch.sigmoid(meta(torch.from_numpy(P).to(device))).cpu().numpy()
    meta_t = roc_auc_score(y[te], pm[te])

    # recovery at matched FPR using held-out test negatives
    negmask = te & (y == 0)
    neg_paths = [paths[i] for i in np.where(negmask)[0]]
    rec = recovery_at_fpr(sh, ef, meta, mean, std, device, neg_paths)

    json.dump({"ratio": f"1:{int((df.label==0).sum()/max(df.label.sum(),1))}",
               "n_curated_pos": int(df.label.sum()), "n_neg": int((df.label == 0).sum()),
               "shielded_test_auc": sh_t, "effnet_test_auc": ef_t, "meta_test_auc": meta_t,
               "recovery_at_fpr": {f"{k[0]}|{k[1]}|{k[2]}": {"recovery": v[0], "thresh": v[1]}
                                   for k, v in rec.items()}},
              open(DATA / "meta_metrics_staged.json", "w"), indent=2)

    # A/B/C/D recovery@1%FPR (meta) — B/C from operating_point.csv if present
    print("\n================ Stage D (curated pos + 33:1 neg) ================")
    print(f"meta test AUC = {meta_t:.4f}   ratio 1:{int((df.label==0).sum()/max(df.label.sum(),1))}")
    print(f"\nRecovery at matched FPR (Stage D):")
    print(f"{'model':>7} {'FPR':>6} {'thresh':>8} {'storfer':>9} {'inchausti':>10}")
    for fpr in (0.01, 0.001):
        for mdl in ("resnet", "effnet", "meta"):
            rs, th = rec[("storfer", mdl, fpr)]; ri, _ = rec[("inchausti", mdl, fpr)]
            print(f"{mdl:>7} {fpr:>6.3f} {th:>8.3f} {rs:>9.3f} {ri:>10.3f}")
    op = DATA / "operating_point.csv"
    if op.exists():
        prev = pd.read_csv(op)
        print("\n[meta recovery@1% FPR across stages]")
        for st in ("B", "C"):
            row = prev[(prev.stage == st) & (prev.model == "meta") & (prev.target_fpr == 0.01)]
            if len(row):
                print(f"  Stage {st}: storfer {row.recovery_storfer.iloc[0]:.3f}  "
                      f"inchausti {row.recovery_inchausti.iloc[0]:.3f}")
        print(f"  Stage D: storfer {rec[('storfer','meta',0.01)][0]:.3f}  "
              f"inchausti {rec[('inchausti','meta',0.01)][0]:.3f}")
    print("[done] Stage-D checkpoints + meta_metrics_staged.json written")


if __name__ == "__main__":
    main()
