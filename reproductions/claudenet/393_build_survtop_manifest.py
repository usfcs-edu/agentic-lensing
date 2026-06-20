#!/usr/bin/env python3
"""393_build_survtop_manifest.py — top-N DR11-south survivors by mean (for the fine-tune deploy
re-score, task D). survivors_dr11s_mean.parquet already carries RA/DEC/footprint/brick, so this
just selects the top-N by mean5 and writes the 111 input. Runs ON PERLMUTTER.
"""
from __future__ import annotations
import argparse
import pandas as pd

BASE = "/pscratch/sd/g/gdbenson/claudenet/sweep_dr11/south"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--survivors", default=f"{BASE}/survivors_dr11s_mean.parquet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = pd.read_parquet(a.survivors).sort_values("mean5", ascending=False).head(a.n).copy()
    s[["row_id", "RA", "DEC", "footprint", "brick"]].to_parquet(a.out, index=False)
    print(f"[393] top-{len(s):,} survivors by mean5 (range {s.mean5.min():.3f}-{s.mean5.max():.3f}) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
