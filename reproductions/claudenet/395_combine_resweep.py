#!/usr/bin/env python3
"""395_combine_resweep.py — combine the full DR11-south re-sweep (task D): mean of the v4
ensemble [effnet_B, zoobot_N (from the original stage-1) + the 3 DR11 fine-tuned members (new
scores)] over all 53.8M parent rows, select the top --budget by mean, and measure known-lens
recall by position-crossmatch (vs the union-95k and mean-150k baselines). Runs ON PERLMUTTER.

  module load python    # (pandas/pyarrow/scipy/astropy; no torch needed here)
  python 395_combine_resweep.py --budget 150000
Outputs -> <BASE>/survivors_dr11s_v4.parquet + resweep_v4_summary.json
"""
from __future__ import annotations
import argparse, glob, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

BASE = "/pscratch/sd/g/gdbenson/claudenet/sweep_dr11/south"
NEW = "/pscratch/sd/g/gdbenson/claudenet/scores/dr11s_resweep"
ANCHORS = ["member_effnet_B", "member_zoobot_N"]                       # from original stage-1
DR11 = ["member_effnet_S2_b50_dr11", "member_effnet_B3_b50_dr11", "member_resnet46_C_b50_dr11"]


def xyz(ra, dec):
    ra = np.radians(np.asarray(ra, float)); dec = np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=150_000)
    a = ap.parse_args()
    keep = []
    total = 0
    for nn in range(32):
        tag = f"{nn:02d}"
        of = glob.glob(f"{BASE}/stage1/stage1_part{tag}_lean_*.parquet")
        nf = f"{NEW}/stage1_part{tag}_dr11.parquet"
        assert of, f"missing original stage1 for part {tag}"
        o = pd.read_parquet(of[0], columns=["row_id"] + ANCHORS)
        n = pd.read_parquet(nf, columns=["row_id"] + DR11)
        m = o.merge(n, on="row_id", how="inner")
        cols = ANCHORS + DR11
        m["mean5"] = m[cols].to_numpy(np.float64).mean(axis=1)
        keep.append(m.nlargest(a.budget, "mean5")[["row_id", "mean5"]])
        total += len(m)
        print(f"[395] part {tag} merged={len(m):,} cum={total:,}", flush=True)
    allk = pd.concat(keep, ignore_index=True)
    sel = allk.nlargest(a.budget, "mean5").reset_index(drop=True)
    sel["rank"] = np.arange(1, len(sel) + 1)
    thr = float(sel.mean5.min()); thr95 = float(sel.mean5.nlargest(95_104).min())
    print(f"[395] parent merged={total:,} selected top-{a.budget:,} thr={thr:.5f} thr@95k={thr95:.5f}")

    # attach RA/DEC
    selset = set(sel.row_id)
    mrows = [pd.read_parquet(p, columns=["row_id", "RA", "DEC", "footprint", "brick"]).query("row_id in @selset")
             for p in glob.glob(f"{BASE}/sweep_manifest_part*of32.parquet")]
    sel = sel.merge(pd.concat(mrows, ignore_index=True), on="row_id", how="left")
    sel.to_parquet(f"{BASE}/survivors_dr11s_v4.parquet", index=False)

    # known-lens recall by position-crossmatch to the v4 survivors (reuse 380 diag knownlens)
    k = pd.read_parquet(f"{BASE}/diag/dr11s_diag_knownlens.parquet")
    tree = cKDTree(xyz(sel.RA, sel.DEC))
    d, _ = tree.query(xyz(k.ra, k.dec), k=1)
    k["in_v4"] = (np.degrees(2 * np.arcsin(np.clip(d / 2, 0, 1))) * 3600) < 5.0
    rep = {"budget": a.budget, "thr": thr, "thr95k": thr95, "parent": total, "recall": {}}
    for src, gr in [("inchausti", "A"), ("inchausti", "B"), ("storfer", "A")]:
        sub = k[(k.src == src) & (k.grade == gr)]
        inp = int(sub.in_parent.sum()); rec = int((sub.in_v4 & sub.in_parent).sum())
        rep["recall"][f"{src}_{gr}"] = dict(in_parent=inp, recovered_v4=rec,
                                            recall=rec / inp if inp else None)
        print(f"[395] {src}_{gr}: recovered_v4 {rec}/{inp} = {(rec/inp if inp else 0):.3f}")

    # vs the old union-95k and mean-150k survivor sets
    for name, f in [("union95k", f"{BASE}/survivors_dr10_recal.parquet"),
                    ("mean150k", f"{BASE}/survivors_dr11s_mean.parquet")]:
        try:
            old = set(pd.read_parquet(f).row_id.astype(str)); new = set(sel.row_id.astype(str))
            rep[f"vs_{name}"] = dict(n=len(old), retained=len(old & new), added=len(new - old), dropped=len(old - new))
            print(f"[395] vs {name}: retained {len(old & new):,} added {len(new - old):,} dropped {len(old - new):,}")
        except Exception as e:
            print(f"[395] vs {name}: skip ({e})")
    json.dump(rep, open(f"{BASE}/resweep_v4_summary.json", "w"), indent=2, default=float)
    print(f"[395] wrote survivors_dr11s_v4.parquet + resweep_v4_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
