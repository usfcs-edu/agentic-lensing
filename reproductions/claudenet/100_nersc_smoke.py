#!/usr/bin/env python3
"""100_nersc_smoke.py — Phase 100 harness-identity smoke (v2 gate).

Proves the Perlmutter A100 + pytorch-module stack scores cutouts identically
(|Δp| < 1e-4) to the local TITAN harness, so every later Perlmutter number is
comparable. TF32 is disabled explicitly: A100 would otherwise run convs in
TF32 (TITAN has no TF32 units) and break bitwise-class parity.

Modes:
  --score   score a manifest with a checkpoint -> parquet   (run on BOTH hosts)
      python 100_nersc_smoke.py --score --manifest data/v2/smoke_manifest.parquet \
          --fits-dir <dir-with-fits> --ckpt <checkpoint.pt> --out <scores.parquet>
  --compare compare two score parquets -> gate verdict      (run locally)
      python 100_nersc_smoke.py --compare local.parquet nersc.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TOL = 1e-4


def do_score(args):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _scorelib as SL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(args.manifest)
    paths = [Path(args.fits_dir) / f"{r}.fits" for r in df["row_id"]]
    model, arch, mean, std, _ = SL.load_checkpoint_model(args.ckpt, device)
    probs = SL.score_paths(paths, model, arch, mean, std, device)
    out = pd.DataFrame({"row_id": df["row_id"], "p": probs})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    n_ok = int(np.isfinite(probs).sum())
    print(f"[smoke:score] device={device} arch={arch} scored {n_ok}/{len(df)} "
          f"-> {args.out}")
    return 0 if n_ok == len(df) else 1


def do_compare(a_path, b_path):
    a = pd.read_parquet(a_path).set_index("row_id")["p"]
    b = pd.read_parquet(b_path).set_index("row_id")["p"]
    j = a.to_frame("pa").join(b.to_frame("pb"), how="inner").dropna()
    d = (j.pa - j.pb).abs()
    res = {"n": int(len(j)), "max_abs_diff": float(d.max()),
           "mean_abs_diff": float(d.mean()), "tol": TOL,
           "verdict": "PASS" if (len(j) > 0 and d.max() < TOL) else "FAIL"}
    out = Path(__file__).resolve().parent / "data" / "v2" / "smoke_parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[smoke:compare] n={res['n']} max|Δp|={res['max_abs_diff']:.2e} "
          f"mean={res['mean_abs_diff']:.2e} -> {res['verdict']}")
    return 0 if res["verdict"] == "PASS" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--fits-dir")
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("LOCAL", "NERSC"))
    args = ap.parse_args()
    if args.score:
        return do_score(args)
    if args.compare:
        return do_compare(*args.compare)
    ap.error("need --score or --compare")


if __name__ == "__main__":
    raise SystemExit(main())
