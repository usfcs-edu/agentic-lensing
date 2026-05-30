#!/usr/bin/env python3
"""
19_train_stageb.py — Phase-5 Stage B retrain + re-measure.

Retrain the two base models + meta-learner on an ENLARGED positive set (the 949
Huang-2020/2021 positives PLUS the in-DR9-footprint literature positions
assembled by 17/18), and report how the controlled AUC and the published-catalog
recovery SHIFT versus Stage A. Self-contained (does not touch the Stage-A
scripts 05/06/07); reuses the shared Dataset (per-row fits_dir) and model files.

Enlarged positives:
  - existing 949  -> data/cutouts_fits_dr9/<row_id>.fits   (label 1)
  - new literature (non-blank cutout, flux_std>1e-3) -> cutouts_fits_litpos_dr9/
Negatives: the same 5,000 -> cutouts_fits_dr9. SEED 2026, 70/20/10 split.

Outputs (suffix _stageb): checkpoints, test_result JSONs, meta_metrics_stageb.json,
candidate_scores_<cat>_stageb.csv, and a printed Stage-A-vs-B table.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import _trainlib as TL
import _scorelib as SL


def _imp(name, fn):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


ShieldedDeepLens = _imp("shielded_resnet", "01b_shielded_resnet.py").ShieldedDeepLens
EfficientNetV2Lens = _imp("efficientnet", "02_efficientnet.py").EfficientNetV2Lens
_meta = _imp("meta_learner", "03_meta_learner.py")
MetaLearner, simple_average = _meta.MetaLearner, _meta.simple_average

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CFG194 = dict(stage_out=52, stage_mid=32, shield_ch=12, final_out=24)


def build_enlarged_split(target_pos: int = 1961) -> pd.DataFrame:
    pos = pd.read_parquet(DATA / "positives_huang2020.parquet")
    neg = pd.read_parquet(DATA / "negatives.parquet")
    rows = []
    for _, r in pos.iterrows():
        rows.append({"row_id": r["Name"], "label": 1, "RA": r["RA"], "DEC": r["DEC"],
                     "fits_dir": str(DATA / "cutouts_fits_dr9")})
    # literature positives that actually have a non-blank DR9 cutout. We have
    # ~5,561 in-footprint, far more than the papers' curated set — sample (seeded)
    # to the Storfer training-set scale so the enlarged pool ≈ target_pos and the
    # class ratio with the 5,000 negatives stays comparable to Stage A.
    mf = pd.read_csv(DATA / "litpos_cutout_manifest.csv")
    good = mf[(mf["status"].isin(["ok", "skip"])) & (mf["flux_std"] > 1e-3)].reset_index(drop=True)
    n_take = max(0, target_pos - len(pos))
    if len(good) > n_take:
        good = good.sample(n=n_take, random_state=TL.SEED).reset_index(drop=True)
    print(f"[enlarge] {len(pos)} existing + {len(good)} literature (of "
          f"{int((mf['flux_std']>1e-3).sum())} in-footprint) -> ~{len(pos)+len(good)} positives")
    for _, r in good.iterrows():
        rows.append({"row_id": r["Name"], "label": 1, "RA": r["RA"], "DEC": r["DEC"],
                     "fits_dir": str(DATA / "cutouts_fits_litpos_dr9")})
    for _, r in neg.iterrows():
        rows.append({"row_id": r["row_id"], "label": 0, "RA": r["RA"], "DEC": r["DEC"],
                     "fits_dir": str(DATA / "cutouts_fits_dr9")})
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
    print(f"[split] enlarged: {len(df)} rows, {int(df.label.sum())} positives "
          f"({int(good['flux_std'].gt(1e-3).sum())} literature in-footprint)")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    return df


def band_stats(df_train):
    rng = np.random.default_rng(TL.SEED)
    s = df_train.iloc[rng.choice(len(df_train), size=min(500, len(df_train)), replace=False)]
    cubes = []
    for _, r in s.iterrows():
        p = Path(r["fits_dir"]) / f"{r['row_id']}.fits"
        if p.exists():
            cubes.append(TL.load_fits_cube(p))
    c = np.stack(cubes); return c.mean((0, 2, 3)), c.std((0, 2, 3)) + 1e-8


def train_base(arch, df, mean, std, device, epochs, batch, lr, decay_ep, accum=1):
    dls = {s: DataLoader(TL.LensDataset(df[df.split == s], DATA, mean, std, s == "train"),
                         batch_size=batch, shuffle=(s == "train"), num_workers=4,
                         drop_last=(s == "train"), pin_memory=True) for s in ("train", "val", "test")}
    if arch == "shielded":
        model = ShieldedDeepLens(in_channels=3, **CFG194).to(device); lossf = nn.BCEWithLogitsLoss()
    else:
        model = EfficientNetV2Lens(pretrained=True).to(device); lossf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_ep, gamma=0.1)
    best_auc, best_state = -1, None
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        for i, (x, y) in enumerate(dls["train"]):
            x = x.to(device)
            if arch == "shielded":
                loss = lossf(model(x), y.to(device)) / accum
            else:
                loss = lossf(model(x), y.long().to(device)) / accum
            loss.backward()
            if (i + 1) % accum == 0:
                opt.step(); opt.zero_grad()
        # val
        model.eval(); ps, ys = [], []
        with torch.no_grad():
            for x, y in dls["val"]:
                pr = TL.model_prob(model, x.to(device), arch).cpu().numpy()
                ps.append(pr); ys.append(y.numpy())
        auc = roc_auc_score(np.concatenate(ys), np.concatenate(ps))
        if auc > best_auc:
            best_auc = auc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        sched.step()
    model.load_state_dict(best_state); model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in dls["test"]:
            ps.append(TL.model_prob(model, x.to(device), arch).cpu().numpy()); ys.append(y.numpy())
    test_auc = roc_auc_score(np.concatenate(ys), np.concatenate(ps))
    return model, float(best_auc), float(test_auc), mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shielded-epochs", type=int, default=130)
    ap.add_argument("--effnet-epochs", type=int, default=60)
    args = ap.parse_args()
    torch.manual_seed(TL.SEED); np.random.seed(TL.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = build_enlarged_split()
    df.to_parquet(DATA / "training_split_stageb.parquet", index=False)
    mean, std = band_stats(df[df.split == "train"])
    print(f"[stat] mean={mean.tolist()} std={std.tolist()}")

    print("\n[train] shielded194k_stageb"); t0 = time.time()
    sh, sh_v, sh_t, _, _ = train_base("shielded", df, mean, std, device,
                                      args.shielded_epochs, 128, 1e-3, 40)
    torch.save({"state_dict": sh.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                "arch": "shielded", "shielded_cfg": CFG194, "val_auc": sh_v},
               DATA / "checkpoint_best_shielded194k_stageb.pt")
    print(f"  shielded_stageb val={sh_v:.4f} test={sh_t:.4f} ({(time.time()-t0)/60:.1f}m)")

    print("\n[train] efficientnet_stageb"); t0 = time.time()
    ef, ef_v, ef_t, _, _ = train_base("efficientnet", df, mean, std, device,
                                      args.effnet_epochs, 128, 1e-3, 30, accum=2)
    torch.save({"state_dict": ef.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                "arch": "efficientnet", "variant": ef.variant, "head_dim": ef.head_dim,
                "num_classes": ef.num_classes, "val_auc": ef_v},
               DATA / "checkpoint_best_efficientnet_stageb.pt")
    print(f"  effnet_stageb val={ef_v:.4f} test={ef_t:.4f} ({(time.time()-t0)/60:.1f}m)")

    # meta-learner on stage-B base probs over the enlarged split
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
    with torch.no_grad():
        pm = torch.sigmoid(meta(torch.from_numpy(P).to(device))).cpu().numpy()
    pa = simple_average(P)
    meta_t = roc_auc_score(y[te], pm[te]); avg_t = roc_auc_score(y[te], pa[te])
    torch.save({"state_dict": meta.state_dict(), "arch": "meta"},
               DATA / "checkpoint_best_meta_stageb.pt")
    json.dump({"shielded": {"val": sh_v, "test": sh_t}, "effnet": {"val": ef_v, "test": ef_t},
               "meta": {"val": bestv, "test": meta_t}, "avg": {"test": avg_t},
               "n_train_pos": int(df[(df.split == 'train') & (df.label == 1)].shape[0])},
              open(DATA / "meta_metrics_stageb.json", "w"), indent=2)

    # recovery shift: score published candidates with stage-B models
    print("\n[recovery] scoring published candidates with Stage-B models")
    rec = {}
    for key, cut in (("storfer", "cutouts_fits_candidates_storfer"),
                     ("inchausti", "cutouts_fits_candidates_inchausti")):
        cat = pd.read_csv(DATA / f"{key}2024_published_catalog.csv"
                          if key == "storfer" else DATA / "inchausti2025_published_catalog.csv")
        cp = [DATA / cut / f"{n}.fits" for n in cat["name"]]
        rr = SL.score_paths(cp, sh, "shielded", mean, std, device)
        re_ = SL.score_paths(cp, ef, "efficientnet", mean, std, device)
        Pc = np.stack([rr, re_], 1).astype(np.float32); ok = np.isfinite(Pc).all(1)
        pmc = np.full(len(Pc), np.nan, np.float32)
        with torch.no_grad():
            pmc[ok] = torch.sigmoid(meta(torch.from_numpy(Pc[ok]).to(device))).cpu().numpy()
        rec[key] = float(np.nanmean(pmc[ok] >= 0.5))

    # Stage-A reference numbers
    sa = json.load(open(DATA / "meta_metrics.json"))
    print("\n================ Stage A vs Stage B ================")
    print(f"{'metric':32s} {'Stage A':>10s} {'Stage B':>10s}")
    print(f"{'train positives':32s} {929:>10d} {int(df[(df.split=='train')&(df.label==1)].shape[0]):>10d}")
    print(f"{'shielded test AUC':32s} {0.9992:>10.4f} {sh_t:>10.4f}")
    print(f"{'effnet test AUC':32s} {0.9995:>10.4f} {ef_t:>10.4f}")
    print(f"{'meta test AUC':32s} {sa['test']['p_meta']:>10.4f} {meta_t:>10.4f}")
    print(f"{'storfer meta recovery@0.5':32s} {0.907:>10.3f} {rec['storfer']:>10.3f}")
    print(f"{'inchausti meta recovery@0.5':32s} {0.932:>10.3f} {rec['inchausti']:>10.3f}")
    print("[done] Stage-B checkpoints + meta_metrics_stageb.json written")


if __name__ == "__main__":
    main()
