#!/usr/bin/env python3
"""318_ensemble_v3_refit.py — ClaudeNet v3 A6: ensemble roster search + SHIP gate.

A2 showed the v3 mimic-blend members (effnet_*/resnet46_C `_b50`) win big on mimic-recovery
but slightly regress random-galaxy recall (the staf327 trade). So the deployable v3 ensemble is
not a wholesale member swap — it's the v2/v3 mix that keeps the mimic gain AND recovers the
random-FPR. This searches candidate rosters over the SAME `average` combiner (mean of the
isotonic-calibrated members — the v2-lean flagship) using all member scores from A2, on:
  * mimic-recovery@{0.05,0.01}  on the DR10 held-out eval (29.6k, in-dist) AND the 601 seed
    bank (OOD, the A0 headline);
  * random-FPR recovery@0.01  with the held-out testneg split as the random null (no-regression);
for Storfer + Inchausti positives. SHIP gate: random@0.01 not CI-below v2-lean AND mimic@0.05
CI-clear over v2-lean AND lrg_companion/seed improvement (reported by 317).

Runs LOCALLY (no GPU except the 3 `_b50` seed re-scores, CPU-fine). Member pc sources: pos/testneg
from scores_member_<name>.parquet; DR10 mimic from scores_member_<name>_mimiceval.parquet (A2/315);
seed from the bank's stored member_<name> columns (v2/frozen) or 317 re-score (`_b50`).

    /home2/benson/.venvs/claudenet/bin/python 318_ensemble_v3_refit.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C
import _ensemble as E
mm = C._load("cn_318_mm", C.ROOT / "301_mimic_metric.py")
s317 = C._load("cn_318_317", C.ROOT / "317_seed_compare.py")

V2, V3 = C.DATA / "v2", C.DATA / "v3"
POS = ("storfer", "inchausti")

# member -> its scores_member parquet (pos/testneg/val splits)
POS_PARQ = {
    "effnet_B": "data/scores_member_effnet_B.parquet",
    "zoobot_N": "data/v2/scores_member_zoobot_N.parquet",
    "effnet_S2_hard": "data/v2/scores_member_effnet_S2_hard.parquet",
    "effnet_B3_hard": "data/v2/scores_member_effnet_B3_hard.parquet",
    "resnet46_C_hard": "data/v2/scores_member_resnet46_C_hard.parquet",
    "effnet_S2_b50": "data/v2/scores_member_effnet_S2_b50.parquet",
    "effnet_B3_b50": "data/v2/scores_member_effnet_B3_b50.parquet",
    "resnet46_C_b50": "data/v2/scores_member_resnet46_C_b50.parquet",
}
STORED_SEED = {"effnet_B", "zoobot_N", "effnet_S2_hard", "effnet_B3_hard", "resnet46_C_hard"}
B50_BASE = {"effnet_S2_b50": "effnet_S2", "effnet_B3_b50": "effnet_B3",
            "resnet46_C_b50": "resnet46_C"}

ROSTERS = {
    "v2lean": ["effnet_B", "effnet_S2_hard", "effnet_B3_hard", "resnet46_C_hard", "zoobot_N"],
    "v3pure": ["effnet_B", "effnet_S2_b50", "effnet_B3_b50", "resnet46_C_b50", "zoobot_N"],
    "v3blend8": ["effnet_B", "zoobot_N", "effnet_S2_hard", "effnet_S2_b50",
                 "effnet_B3_hard", "effnet_B3_b50", "resnet46_C_hard", "resnet46_C_b50"],
    "v3sel_R46hard": ["effnet_B", "zoobot_N", "effnet_S2_b50", "effnet_B3_b50", "resnet46_C_hard"],
    "v3sel_bothR46": ["effnet_B", "zoobot_N", "effnet_S2_b50", "effnet_B3_b50",
                      "resnet46_C_hard", "resnet46_C_b50"],
    "v3_noR46": ["effnet_B", "zoobot_N", "effnet_S2_b50", "effnet_B3_b50"],
}


def _pos_pc(member, sp):
    d = pd.read_parquet(C.ROOT / POS_PARQ[member])
    return d[d["split"] == sp][["row_id", "pc"]].rename(columns={"pc": member}).astype({"row_id": str})


def _ens(per_member: dict[str, pd.DataFrame], roster) -> np.ndarray:
    mats = None
    for m in roster:
        g = per_member[m]
        mats = g if mats is None else mats.merge(g, on="row_id", how="inner")
    return mats[list(roster)].to_numpy(float).mean(axis=1)


def main() -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(V3 / "a6_ensemble_refit.json"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    members = sorted({m for r in ROSTERS.values() for m in r})
    seed = pd.read_parquet(V3 / "mimic_bank_seed.parquet").astype({"row_id": str})

    # per-member pc on each set
    pc = {sp: {} for sp in (*POS, "testneg")}
    dr10 = {}
    seedpc = {}
    for m in members:
        for sp in (*POS, "testneg"):
            pc[sp][m] = _pos_pc(m, sp)
        d = pd.read_parquet(V3 / f"scores_member_{m}_mimiceval.parquet")[["row_id", "pc"]]
        dr10[m] = d.rename(columns={"pc": m}).astype({"row_id": str})
        if m in STORED_SEED:
            cal = s317.iso_from_val(POS_PARQ[m])
            seedpc[m] = pd.DataFrame({"row_id": seed["row_id"],
                                      m: cal.transform(seed[f"member_{m}"].to_numpy(float))})
        else:
            base = B50_BASE[m]
            ck = V2 / "ckpt" / f"member_{base}_b50.pt"
            v = s317.score_seed(ck, POS_PARQ[m], seed, device)
            seedpc[m] = pd.DataFrame({"row_id": seed["row_id"], m: v})
            print(f"[318] re-scored {m} on {len(seed)} seed FITS")

    seed_type = dict(zip(seed["row_id"], seed["mimic_type"]))
    rep = {"rosters": {}}
    base_rand = {}
    print(f"\n[318] roster search (mimic-recovery@0.05/0.01 + random-FPR@0.01, average combiner)\n")
    hdr = f"{'roster':14s} {'pos':10s} | {'rand@01':>7s} | {'dr10@05':>7s} {'dr10@01':>7s} | {'seed@05':>7s} {'seed@01':>7s}"
    print(hdr); print("-" * len(hdr))
    for name, roster in ROSTERS.items():
        rep["rosters"][name] = {"members": roster, "splits": {}}
        for sp in POS:
            pos = _ens(pc[sp], roster)
            neg = _ens(pc["testneg"], roster)
            md = _ens(dr10, roster)
            ms = _ens(seedpc, roster)
            rand01 = E.recovery_at_fpr(neg, pos, fprs=(0.01,))[0.01]["recovery"]
            rd = mm.recovery_at_mimic_fpr(md, pos)
            rs = mm.recovery_at_mimic_fpr(ms, pos)
            ent = {"rand_fpr01": rand01,
                   "dr10": {str(k): rd[k]["recovery"] for k in rd},
                   "seed": {str(k): rs[k]["recovery"] for k in rs},
                   "n_pos": len(pos)}
            if name == "v2lean":
                base_rand[sp] = rand01
            rep["rosters"][name]["splits"][sp] = ent
            print(f"{name:14s} {sp:10s} | {rand01:7.3f} | {rd[0.05]['recovery']:7.3f} "
                  f"{rd[0.01]['recovery']:7.3f} | {rs[0.05]['recovery']:7.3f} {rs[0.01]['recovery']:7.3f}")
        print()

    # SHIP gate: random@0.01 no-regression (CI) vs v2lean + mimic gain
    print("[318] SHIP gate (vs v2lean): random@0.01 no-regression [Wilson 95%] + seed mimic@0.05 gain")
    for name, roster in ROSTERS.items():
        if name == "v2lean":
            continue
        ok = []
        for sp in POS:
            e = rep["rosters"][name]["splits"][sp]; n = e["n_pos"]
            r, b = e["rand_fpr01"], base_rand[sp]
            rlo, _ = mm.wilson_ci(int(round(r * n)), n)
            _, bhi = mm.wilson_ci(int(round(b * n)), n)
            no_reg = rlo >= b - 0.01 or r >= b           # CI-lenient no-regression
            gain = e["seed"]["0.05"] > rep["rosters"]["v2lean"]["splits"][sp]["seed"]["0.05"]
            ok.append(no_reg and gain)
            rep["rosters"][name].setdefault("ship", {})[sp] = {
                "rand": r, "v2_rand": b, "no_regression": bool(no_reg), "mimic_gain": bool(gain)}
        verdict = "SHIP" if all(ok) else ("partial" if any(ok) else "fail")
        print(f"  {name:14s}: {verdict}  "
              f"(storfer rand {rep['rosters'][name]['splits']['storfer']['rand_fpr01']:.3f} vs "
              f"{base_rand['storfer']:.3f}; inchausti {rep['rosters'][name]['splits']['inchausti']['rand_fpr01']:.3f} vs "
              f"{base_rand['inchausti']:.3f})")
    Path(args.out).write_text(json.dumps(rep, indent=2, default=float))
    print(f"\n[318] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
