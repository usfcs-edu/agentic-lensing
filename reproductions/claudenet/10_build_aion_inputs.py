#!/usr/bin/env python3
"""10_build_aion_inputs.py — build AION LegacySurveyImage inputs (M,4,160,160) for
the Phase-0 decorrelation gate, from the synced grz finder FITS.

AION's LegacySurveyImage codec expects 4-band griz at 160x160 (num_tokens=576;
GZ10 used native 160px). Our finder cutouts are 3-band grz at 101x101 in the same
LS nanomaggie units. For the GATE (fast, no rate-limited LS fetching) we:
  * bilinear-resize 101 -> 160,
  * synthesize the i band as i = 0.5*(r + z)  (i is between r and z; documented
    i_synth=True).
This is a conservative, self-contained smoke input: if AION embeddings decorrelate
from EfficientNet even when scale-mismatched, the flagship is worth building with a
native-griz fetch in Phase 1. Runs under the claudenet venv (needs torch).

Splits built (manifest + flux .npy under data/emb/):
  trainpool : staged-train positives (all) + N negatives (probe training set)
  testneg   : staged-test negatives (threshold + correlation set)
  storfer   : published Storfer candidates (held-out positives)
  inchausti : published Inchausti candidates (held-out positives)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import _clib as C
import _trainlib as TL


def to_griz160(grz_batch: np.ndarray) -> np.ndarray:
    """(B,3,101,101) grz -> (B,4,160,160) griz, bilinear resize + i=0.5*(r+z)."""
    x = torch.from_numpy(grz_batch)                       # (B,3,101,101)
    x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
    g, r, z = x[:, 0], x[:, 1], x[:, 2]
    i = 0.5 * (r + z)
    out = torch.stack([g, r, i, z], dim=1)                # g,r,i,z order
    return out.numpy().astype(np.float32)


def build(rows: pd.DataFrame, name: str, batch=256):
    """rows: columns row_id,label,fits_path. Writes flux .npy + manifest parquet."""
    fluxes, keep = [], []
    buf, buf_idx = [], []

    def flush():
        if not buf:
            return
        grz = np.stack(buf).astype(np.float32)
        fluxes.append(to_griz160(grz))
        keep.extend(buf_idx)
        buf.clear(); buf_idx.clear()

    for j, rr in enumerate(rows.itertuples()):
        try:
            arr = TL.load_fits_cube(Path(rr.fits_path))
        except Exception:
            continue
        if arr.shape != (3, 101, 101):
            continue
        buf.append(arr); buf_idx.append(j)
        if len(buf) >= batch:
            flush()
    flush()
    flux = np.concatenate(fluxes, axis=0) if fluxes else np.zeros((0, 4, 160, 160), np.float32)
    man = rows.iloc[keep].reset_index(drop=True)
    man["i_synth"] = True
    np.save(C.EMB / f"aion_in_{name}.npy", flux)
    man.to_parquet(C.EMB / f"aion_in_{name}_manifest.parquet", index=False)
    print(f"[build] {name:10s} {flux.shape}  ({man['label'].sum()} pos / "
          f"{(man['label']==0).sum()} neg)")
    return flux.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-neg", type=int, default=8000)
    args = ap.parse_args()
    rng = np.random.default_rng(C.SEED)

    split = pd.read_parquet(C.DATA / "training_split_staged.parquet")
    split["fits_path"] = split.apply(
        lambda r: str(C.DATA / Path(str(r.fits_dir)).name / f"{r.row_id}.fits"), axis=1)

    # trainpool: all train positives + sampled train negatives
    tr = split[split.split == "train"]
    tr_pos = tr[tr.label == 1]
    tr_neg = tr[tr.label == 0]
    if len(tr_neg) > args.n_train_neg:
        tr_neg = tr_neg.iloc[rng.choice(len(tr_neg), args.n_train_neg, replace=False)]
    trainpool = pd.concat([tr_pos, tr_neg])[["row_id", "label", "fits_path"]]
    build(trainpool, "trainpool")

    # testneg: held-out staged test negatives
    te_neg = split[(split.split == "test") & (split.label == 0)][["row_id", "label", "fits_path"]]
    build(te_neg, "testneg")

    # held-out published positives
    for name, cut, csv in (("storfer", "cutouts_fits_candidates_storfer", "storfer2024_published_catalog.csv"),
                           ("inchausti", "cutouts_fits_candidates_inchausti", "inchausti2025_published_catalog.csv")):
        cat = pd.read_csv(C.DATA / csv)
        rows = pd.DataFrame({"row_id": cat["name"], "label": 1})
        rows["fits_path"] = rows["row_id"].apply(lambda n: str(C.DATA / cut / f"{n}.fits"))
        build(rows, name)

    print("[10] AION inputs built (griz 160, i=0.5*(r+z) synthetic)")


if __name__ == "__main__":
    main()
