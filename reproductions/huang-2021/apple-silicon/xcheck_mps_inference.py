#!/usr/bin/env python3
"""xcheck_mps_inference.py — MPS inference fidelity, two-model (Huang-2021).

Compares the bounded MPS smoke scores (11b run with the *phoenix* checkpoints on a
few hundred bricks) against phoenix's full DR8 score parquets, per row_id, for BOTH
the L18 and shielded models. Same weights + same brick pixels => scores must agree to
float precision, isolating MPS-vs-CUDA numerical drift from training stochasticity.

The shielded net (~60K params, four 1x1 "shield" convs) is the most MPS-sensitive
case, so its delta is reported separately. Writes data/mps_xcheck.json
{l18:{...}, shielded:{...}}, consumed by verify_against_reference.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def compare(smoke_p: str, ref_p: str) -> dict:
    sp, rp = Path(smoke_p), Path(ref_p)
    if not sp.exists() or not rp.exists():
        return {"present": False, "smoke": str(sp), "ref": str(rp)}
    s = pd.read_parquet(sp)[["row_id", "score"]].rename(columns={"score": "m"})
    r = pd.read_parquet(rp)[["row_id", "score"]].rename(columns={"score": "p"})
    # A few galaxies (~0.02%) appear twice in the DR8 parent — the south/north
    # footprint overlap scores the same (BRICKID,OBJID) from two different coadds,
    # so their two scores legitimately differ by up to ~1.0. Matching on row_id
    # alone would create spurious cross-coadd comparisons that dominate max|Δ|.
    # Restrict the fidelity check to unambiguous (singly-occurring) row_ids.
    n_dup = int(r["row_id"].duplicated(keep=False).sum())
    s = s[~s["row_id"].duplicated(keep=False)]
    r = r[~r["row_id"].duplicated(keep=False)]
    j = s.merge(r, on="row_id")
    if len(j) == 0:
        return {"present": True, "n_overlap": 0, "max_abs_delta": None}
    d = np.abs(j["m"].to_numpy() - j["p"].to_numpy())
    return {"present": True, "n_overlap": int(len(j)), "n_dup_ref_dropped": n_dup,
            "max_abs_delta": float(d.max()), "median_abs_delta": float(np.median(d)),
            "p99_9_abs_delta": float(np.percentile(d, 99.9))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-l18", default=str(DATA / "inference_scores_l18_shard0.parquet"))
    ap.add_argument("--smoke-shielded", default=str(DATA / "inference_scores_shielded_shard0.parquet"))
    ap.add_argument("--ref-l18", default=str(DATA / "ref" / "inference_scores_l18_dr8.parquet"))
    ap.add_argument("--ref-shielded", default=str(DATA / "ref" / "inference_scores_shielded_dr8.parquet"))
    a = ap.parse_args()
    out = {"l18": compare(a.smoke_l18, a.ref_l18),
           "shielded": compare(a.smoke_shielded, a.ref_shielded)}
    (DATA / "mps_xcheck.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    for k, v in out.items():
        if not v.get("present"):
            print(f"[xcheck] {k}: inputs missing"); continue
        mad = v.get("max_abs_delta")
        print(f"[xcheck] {k}: overlap={v['n_overlap']:,}  "
              f"max|Δ|={('%.3e' % mad) if mad is not None else 'n/a'}")


if __name__ == "__main__":
    main()
