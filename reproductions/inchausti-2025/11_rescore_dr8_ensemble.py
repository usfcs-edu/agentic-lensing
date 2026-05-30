#!/usr/bin/env python3
"""
11_rescore_dr8_ensemble.py — Phase-5 targeted-recovery, track (i).

Re-score the DR8 candidate pool we ALREADY have on disk (the ~319K cutouts kept
at p>=0.1 by the Phase-4 brick sweep, data/cutouts_fits_dr8/) with the two
Inchausti base models (shielded-194K + EfficientNetV2), combine them through the
meta-learner + simple average, and merge with the existing L18 / shielded-60K DR8
scores. This produces an ensemble re-ranking of the DR8 footprint with ZERO new
downloads (no full survey sweep — out of scope).

Note: this pool is biased — it is only the cutouts the Phase-4 (northaug) models
already flagged at p>=0.1, not the full 17.3M parent sample (those bricks were
discarded after Phase 4). So track (i) re-ranks the existing DR8 candidate pool;
the honest published-catalog recovery is track (ii) (12/13/14).

Reads each cutout ONCE and scores both base models. Outputs:
  data/inference_scores_ensemble_dr8.parquet         (row_id,ra,dec,p_*)
  data/inference_scores_ensemble_dr8_p_ge_0.5.parquet (slim, committable)
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _trainlib as TL
import _scorelib as SL

_spec = importlib.util.spec_from_file_location(
    "meta_learner", str(Path(__file__).resolve().parent / "03_meta_learner.py"))
_mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mm)
MetaLearner = _mm.MetaLearner
simple_average = _mm.simple_average

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CUTOUTS = DATA / "cutouts_fits_dr8"


@torch.no_grad()
def score_both(paths, m_res, m_eff, device, batch, workers):
    """Read each FITS once; score shielded-194K and EfficientNetV2 together."""
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __init__(self, paths): self.paths = paths
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            try:
                arr = TL.load_fits_cube(self.paths[i])
                if arr.shape != (3, 101, 101):
                    raise ValueError
                return i, torch.from_numpy(arr)
            except Exception:
                return i, torch.zeros(3, 101, 101)

    res_model, _, r_mean, r_std, _ = m_res
    eff_model, _, e_mean, e_std, _ = m_eff
    rmt, rst = torch.from_numpy(r_mean).to(device), torch.from_numpy(r_std).to(device)
    emt, est = torch.from_numpy(e_mean).to(device), torch.from_numpy(e_std).to(device)
    p_res = np.full(len(paths), np.nan, np.float32)
    p_eff = np.full(len(paths), np.nan, np.float32)
    dl = DataLoader(_DS(paths), batch_size=batch, num_workers=workers)
    n = 0
    for idx, x in dl:
        x = x.to(device, non_blocking=True)
        xr = torch.clamp((x - rmt) / rst, -250, 250)
        xe = torch.clamp((x - emt) / est, -250, 250)
        pr = torch.sigmoid(res_model(xr)).cpu().numpy()
        pe = torch.softmax(eff_model(xe), dim=1)[:, 1].cpu().numpy()
        ii = idx.numpy()
        p_res[ii] = pr
        p_eff[ii] = pe
        n += len(ii)
        if n % (batch * 50) == 0:
            print(f"  scored {n:,}/{len(paths):,}")
    return p_res, p_eff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = sorted(CUTOUTS.glob("*.fits"))
    if args.limit:
        paths = paths[: args.limit]
    row_ids = [p.stem for p in paths]
    print(f"[init] {len(paths):,} DR8 cutouts on disk")

    m_res = SL.load_checkpoint_model(DATA / "checkpoint_best_shielded194k.pt", device)
    m_eff = SL.load_checkpoint_model(DATA / "checkpoint_best_efficientnet.pt", device)
    print(f"[init] shielded194k val_auc={m_res[4]:.4f}  effnet val_auc={m_eff[4]:.4f}")

    p_res, p_eff = score_both(paths, m_res, m_eff, device, args.batch, args.workers)

    ck = torch.load(str(DATA / "checkpoint_best_meta.pt"), map_location="cpu", weights_only=False)
    meta = MetaLearner().to(device); meta.load_state_dict(ck["state_dict"]); meta.eval()
    P = np.stack([p_res, p_eff], axis=1).astype(np.float32)
    ok = np.isfinite(P).all(axis=1)
    p_meta = np.full(len(P), np.nan, np.float32)
    with torch.no_grad():
        p_meta[ok] = torch.sigmoid(meta(torch.from_numpy(P[ok]).to(device))).cpu().numpy()
    p_avg = simple_average(P)

    df = pd.DataFrame({"row_id": row_ids, "p_shielded194k": p_res,
                       "p_effnet": p_eff, "p_meta": p_meta, "p_avg": p_avg})
    # bring in ra/dec + the existing two-model scores by row_id
    l18 = pd.read_parquet(DATA / "inference_scores_l18_dr8.parquet").rename(
        columns={"score": "p_l18"})
    sh60 = pd.read_parquet(DATA / "inference_scores_shielded_dr8.parquet")[["row_id", "score"]].rename(
        columns={"score": "p_shielded60k"})
    df = df.merge(l18[["row_id", "ra", "dec", "p_l18"]], on="row_id", how="left")
    df = df.merge(sh60, on="row_id", how="left")

    out = DATA / "inference_scores_ensemble_dr8.parquet"
    df.to_parquet(out, index=False)
    slim = df[df["p_meta"] >= 0.5]
    slim.to_parquet(DATA / "inference_scores_ensemble_dr8_p_ge_0.5.parquet", index=False)

    print(f"\n[done] wrote {out.name} ({len(df):,} rows)")
    for c in ("p_l18", "p_shielded60k", "p_shielded194k", "p_effnet", "p_meta", "p_avg"):
        v = df[c].dropna()
        print(f"   {c:>16s}: median {v.median():.4f}  >=0.5 {int((v>=0.5).sum()):>7,}  "
              f">=0.9 {int((v>=0.9).sum()):>7,}")


if __name__ == "__main__":
    main()
