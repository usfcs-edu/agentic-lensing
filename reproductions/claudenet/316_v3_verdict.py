#!/usr/bin/env python3
"""316_v3_verdict.py — ClaudeNet v3 A2: the v2-vs-v3 head-to-head on the mimic metric.

Computes, on the SAME frozen held-out mimic-eval and the SAME held-out Storfer/Inchausti
positives, recovery@matched-mimic-FPR for the v2-lean ensemble and the v3 ensemble (the 3
`hard3` members blended with the mimic bank swapped in for their v2 `_hard` counterparts;
effnet_B + zoobot_N frozen). Reports:

  * ENSEMBLE headline — recovery@mimic-FPR(0.05/0.01), v2 vs v3, per positive split + per
    contaminant type (the number the program is built to move: v2 = 0.168/0.307 @ 0.05).
  * NO-REGRESSION — recovery@random-FPR(0.01) with the held-out testneg split as the random
    null (the v2 benchmark must not regress).
  * PER-MEMBER G2 gate — each retrained member vs its v2 `_hard` counterpart (single-member
    mimic-recovery), the admission signal.
  * FRACTION SWEEP — effnet_S2 single-member mimic-recovery at blend f∈{0.30,0.50,0.70}.
  * INTEGRITY — recomputed v2 ensemble mimic-eval score vs the bank's stored p_final.

Member pc on the mimic-eval comes from 315 (scores_member_<name>_mimiceval.parquet); pc on
positives/testneg from each member's scores parquet. All ensemble scores are the mean of
isotonic-calibrated members (the `average` combiner) — one scale for positives and mimics.

    python 316_v3_verdict.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C
import _ensemble as E
mm = C._load("cn_316_mm", C.ROOT / "301_mimic_metric.py")

V2, V3 = C.DATA / "v2", C.DATA / "v3"
POS_SPLITS = ("storfer", "inchausti")

FROZEN = [{"name": "effnet_B", "pos": "data/scores_member_effnet_B.parquet"},
          {"name": "zoobot_N", "pos": "data/v2/scores_member_zoobot_N.parquet"}]
V2_SWAP = [{"name": "effnet_S2_hard", "pos": "data/v2/scores_member_effnet_S2_hard.parquet"},
           {"name": "effnet_B3_hard", "pos": "data/v2/scores_member_effnet_B3_hard.parquet"},
           {"name": "resnet46_C_hard", "pos": "data/v2/scores_member_resnet46_C_hard.parquet"}]
V2BASE = {"effnet_S2": "effnet_S2_hard", "effnet_B3": "effnet_B3_hard",
          "resnet46_C": "resnet46_C_hard"}


def _v3_swap(tag: str):
    return [{"name": f"{m}_{tag}", "pos": f"data/v2/scores_member_{m}_{tag}.parquet"}
            for m in ("effnet_S2", "effnet_B3", "resnet46_C")]


def mim_pc(name: str) -> pd.DataFrame:
    df = pd.read_parquet(V3 / f"scores_member_{name}_mimiceval.parquet")[["row_id", "pc"]]
    return df.rename(columns={"pc": name}).astype({"row_id": str})


def pos_pc(member: dict, sp: str) -> pd.DataFrame:
    df = pd.read_parquet(C.ROOT / member["pos"])
    g = df[df["split"] == sp][["row_id", "pc"]].rename(columns={"pc": member["name"]})
    return g.astype({"row_id": str})


def ensemble_mim(members) -> tuple[np.ndarray, list]:
    mats = None
    for m in members:
        g = mim_pc(m["name"])
        mats = g if mats is None else mats.merge(g, on="row_id", how="inner")
    cols = [m["name"] for m in members]
    return mats[cols].to_numpy(float).mean(axis=1), mats["row_id"].tolist()


def ensemble_pos(members, sp) -> np.ndarray:
    mats = None
    for m in members:
        g = pos_pc(m, sp)
        mats = g if mats is None else mats.merge(g, on="row_id", how="inner")
    cols = [m["name"] for m in members]
    return mats[cols].to_numpy(float).mean(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="b50", help="v3 blend tag for the headline ensemble")
    ap.add_argument("--bank-eval", default=str(V3 / "mimic_bank_eval.parquet"))
    ap.add_argument("--out", default=str(V3 / "a2_v3_verdict.json"))
    args = ap.parse_args()

    bank = pd.read_parquet(args.bank_eval).astype({"row_id": str})
    type_of = dict(zip(bank["row_id"], bank["mimic_type"]))
    p_final_v2_stored = dict(zip(bank["row_id"], bank["p_final"].astype(float)))

    v2 = FROZEN + V2_SWAP
    v3 = FROZEN + _v3_swap(args.tag)
    rep = {"tag": args.tag, "ensemble": {}, "per_member": {}, "fraction_sweep": {},
           "no_regression": {}, "integrity": {}}

    mim_v2, ids_v2 = ensemble_mim(v2)
    mim_v3, ids_v3 = ensemble_mim(v3)
    # integrity: recomputed v2 ensemble mimic score vs the stored bank p_final
    stored = np.array([p_final_v2_stored[r] for r in ids_v2])
    rep["integrity"] = {"n": len(ids_v2),
                        "max_abs_diff": float(np.max(np.abs(mim_v2 - stored))),
                        "corr": float(np.corrcoef(mim_v2, stored)[0, 1])}
    print(f"[316] integrity v2 recomputed-vs-stored p_final: "
          f"max|d|={rep['integrity']['max_abs_diff']:.3e} corr={rep['integrity']['corr']:.5f} "
          f"(n={len(ids_v2):,})")

    df_v3_types = pd.DataFrame({"row_id": ids_v3, "score": mim_v3})
    df_v3_types["mimic_type"] = df_v3_types["row_id"].map(type_of)

    print(f"\n[316] ===== ENSEMBLE recovery@mimic-FPR  (v2-lean -> v3/{args.tag}) =====")
    for sp in POS_SPLITS:
        pos_v2 = ensemble_pos(v2, sp)
        pos_v3 = ensemble_pos(v3, sp)
        r2 = mm.recovery_at_mimic_fpr(mim_v2, pos_v2)
        r3 = mm.recovery_at_mimic_fpr(mim_v3, pos_v3)
        pt = mm.per_type_recovery(df_v3_types, pos_v3, "score")
        rep["ensemble"][sp] = {
            "v2": {str(p): r2[p]["recovery"] for p in r2},
            "v3": {str(p): r3[p]["recovery"] for p in r3},
            "v3_per_type": pt, "n_pos": len(pos_v3)}
        print(f"  {sp:10s} @0.05: v2={r2[0.05]['recovery']:.3f} -> v3={r3[0.05]['recovery']:.3f}"
              f"   @0.01: v2={r2[0.01]['recovery']:.3f} -> v3={r3[0.01]['recovery']:.3f}"
              f"   (n_pos={len(pos_v3)}, n_mim v2={len(mim_v2)}/v3={len(mim_v3)})")
        worst = sorted(pt.items(), key=lambda kv: kv[1]["recovery"].get("0.05", 9))[:4]
        for t, d in worst:
            print(f"      v3 per-type {t:16s} n={d['n_mimic']:5d}  "
                  f"@0.05={d['recovery'].get('0.05', float('nan')):.3f}")

    # NO-REGRESSION: recovery@random-FPR(0.01), testneg as the random null
    print(f"\n[316] ===== NO-REGRESSION recovery@random-FPR(0.01) [testneg null] =====")
    neg_v2 = ensemble_pos(v2, "testneg"); neg_v3 = ensemble_pos(v3, "testneg")
    for sp in POS_SPLITS:
        pos_v2 = ensemble_pos(v2, sp); pos_v3 = ensemble_pos(v3, sp)
        a2 = E.recovery_at_fpr(neg_v2, pos_v2, fprs=(0.01,))[0.01]["recovery"]
        a3 = E.recovery_at_fpr(neg_v3, pos_v3, fprs=(0.01,))[0.01]["recovery"]
        rep["no_regression"][sp] = {"v2": a2, "v3": a3}
        print(f"  {sp:10s} v2={a2:.3f} -> v3={a3:.3f}  ({'OK' if a3 >= a2 - 0.02 else 'REGRESSION'})")

    # PER-MEMBER G2 + FRACTION SWEEP (single-member mimic-recovery)
    print(f"\n[316] ===== PER-MEMBER G2 (v2 _hard -> v3 {args.tag}) =====")
    for member, v2name in V2BASE.items():
        v3name = f"{member}_{args.tag}"
        try:
            e2, _ = ensemble_mim([{"name": v2name}]); e3, _ = ensemble_mim([{"name": v3name}])
        except FileNotFoundError as ex:
            print(f"  {member}: missing eval scores ({ex}); skip"); continue
        out = {}
        for sp in POS_SPLITS:
            p2 = ensemble_pos([{"name": v2name, "pos": f"data/v2/scores_member_{v2name}.parquet"}], sp)
            p3 = ensemble_pos([{"name": v3name, "pos": f"data/v2/scores_member_{v3name}.parquet"}], sp)
            r2 = mm.recovery_at_mimic_fpr(e2, p2)[0.05]["recovery"]
            r3 = mm.recovery_at_mimic_fpr(e3, p3)[0.05]["recovery"]
            out[sp] = {"v2": r2, "v3": r3}
            print(f"  {member:12s} {sp:10s} @0.05: v2={r2:.3f} -> v3={r3:.3f}")
        rep["per_member"][member] = out

    print(f"\n[316] ===== FRACTION SWEEP effnet_S2 (single-member @0.05) =====")
    for tag in ("b30", "b50", "b70"):
        name = f"effnet_S2_{tag}"
        try:
            e, _ = ensemble_mim([{"name": name}])
        except FileNotFoundError:
            print(f"  {tag}: not scored; skip"); continue
        row = {}
        for sp in POS_SPLITS:
            p = ensemble_pos([{"name": name, "pos": f"data/v2/scores_member_{name}.parquet"}], sp)
            row[sp] = mm.recovery_at_mimic_fpr(e, p)[0.05]["recovery"]
        rep["fraction_sweep"][tag] = row
        print(f"  {tag}: storfer={row['storfer']:.3f}  inchausti={row['inchausti']:.3f}")

    Path(args.out).write_text(json.dumps(rep, indent=2, default=float))
    print(f"\n[316] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
