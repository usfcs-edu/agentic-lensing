#!/usr/bin/env python3
"""149_make_teacher.py — Phase 150 prep: assemble the distillation teacher
targets (LOCAL, 1 GPU).

Scores the v1 staged-train rows + the locally-extracted mined rows with every
member of the v2 roster (112's checkpoint loader + _scorelib's exact file-path
math), then applies the persisted 145 fits (calibrators + combiner) to produce
p_teacher = the requested combiner's calibrated output.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
      /home2/benson/.venvs/claudenet/bin/python 149_make_teacher.py \\
          --combiner rf --out data/v2/teacher_targets.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _clib as C
import _scorelib as SL

V2 = C.DATA / "v2"


def member_rows() -> pd.DataFrame:
    """Staged train rows (host-remapped) + mined hard/random rows (label 0)."""
    split = pd.read_parquet(C.DATA / "training_split_staged.parquet")
    split["fits_dir"] = split["fits_dir"].apply(lambda p: str(C.DATA / Path(str(p)).name))
    tr = split[split.split == "train"][["row_id", "label", "fits_dir"]].copy()
    parts = [tr]
    for v in ("hard", "random"):
        f = V2 / f"mined_{v}_fits_manifest.parquet"
        if f.exists():
            m = pd.read_parquet(f)[["row_id", "label", "fits_dir"]]
            parts.append(m)
    df = pd.concat(parts, ignore_index=True).drop_duplicates("row_id", keep="first")
    df["row_id"] = df["row_id"].astype(str)
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roster", default=str(V2 / "roster_v2.json"))
    ap.add_argument("--fits", default=str(V2 / "ensemble_v2_fits"),
                    help="145 fits dir (ensemble_fits.joblib + meta.json)")
    ap.add_argument("--combiner", default="rf", choices=("average", "logistic", "rf"),
                    help="which calibrated combiner becomes p_teacher")
    ap.add_argument("--out", default=str(V2 / "teacher_targets.parquet"))
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    t0 = time.time()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "149 needs a GPU (CUDA init failed?)"

    df = member_rows()
    print(f"[149] {len(df):,} rows (train + mined); device={device}")
    paths = [Path(r.fits_dir) / f"{r.row_id}.fits" for r in df.itertuples()]

    M112 = C._load("cn_149_m112", C.ROOT / "112_score_pool.py")
    roster = json.load(open(args.roster))
    cols = {}
    for m in roster:
        name, col = m["name"], m["pool_column"]
        ck = None
        for d in (V2 / "ckpt", C.DATA / "ckpt"):
            p = d / f"member_{name}.pt"
            if p.exists():
                ck = p
                break
        assert ck is not None, f"checkpoint member_{name}.pt not found"
        model, score_arch, mean, std = M112.load_member_checkpoint(
            ck, device, M112.member_variant(name, C.DATA))
        probs = SL.score_paths(paths, model, score_arch, mean, std, device,
                               batch=args.batch)
        cols[col] = probs
        n = int(np.isfinite(probs).sum())
        print(f"[149] {name:16s} ({ck.name}) scored {n:,}/{len(df):,}")
        del model
        torch.cuda.empty_cache()

    raw = pd.DataFrame({"row_id": df.row_id, **cols})
    raw_path = V2 / "teacher_member_raw.parquet"
    raw.to_parquet(raw_path, index=False)

    # apply the persisted 145 fits -> calibrated combiner column
    M145 = C._load("cn_149_m145", C.ROOT / "145_refit_ensemble_v2.py")
    fits, combs = M145.load_fits(Path(args.fits))
    applied = M145.apply_fits(fits, combs, raw)
    tcol = f"{fits['tag']}_{args.combiner}"
    assert tcol in applied.columns, f"{tcol} not in applied fit columns {list(applied.columns)}"
    out = pd.DataFrame({"row_id": raw.row_id, "p_teacher": applied[tcol].astype(np.float32)})
    keep = np.isfinite(out.p_teacher.to_numpy())
    if (~keep).any():
        print(f"[149] dropping {(~keep).sum():,} rows with non-finite teacher scores")
        out = out[keep]
    out.to_parquet(args.out, index=False)
    print(f"[149] wrote {args.out} ({len(out):,} rows, combiner={args.combiner}, "
          f"{(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
