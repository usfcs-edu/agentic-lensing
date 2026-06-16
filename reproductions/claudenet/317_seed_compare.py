#!/usr/bin/env python3
"""317_seed_compare.py — ClaudeNet v3 A2: v2-vs-v3 on the 601-row curated seed bank.

The A2 eval job (315/316) compares v2 and v3 on the DR10 held-out mimic-eval (29.6k) — but
those auto-typed DR10 survivors are IN-DISTRIBUTION for v3 (it trained on the DR10 mimic pool)
and EASIER than the curated seed (v2 already does ~0.71/0.86 there). The 601 DR9 campaign
rejects (mimic_bank_seed.parquet) are the hand-confirmed hardest mimics AND were EXCLUDED from
v3's train pool (313 restricted to dr10_sweep) — so they are a clean OOD generalization test
for BOTH v2 and v3, and the A0 headline denominator (v2 = 0.168/0.307 @ 0.05).

Only the 3 retrained members must be re-scored: effnet_B + zoobot_N are unchanged in v3, so
their stored member_* columns in the seed bank ARE the v3 frozen scores. So:
  v2 seed ens = mean isotonic_m(member_*_raw)        for the 5 v2-lean members  (all stored)
  v3 seed ens = mean isotonic_m(...)                  effnet_B + zoobot_N stored,
                                                       effnet_S2/B3/resnet46_C re-scored on the
                                                       601 local FITS with the _b50 ckpts.
Runs LOCALLY (601 FITS, CPU fine). Pull the 3 _b50 ckpts + their scores_member_*_b50.parquet
(val split for the isotonic) from Perlmutter first.

    /home2/benson/.venvs/claudenet/bin/python 317_seed_compare.py --tag b50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C
import _ensemble as E
M112 = C._load("cn_317_112", C.ROOT / "112_score_pool.py")
mm = C._load("cn_317_mm", C.ROOT / "301_mimic_metric.py")
import _train as T

V2 = C.DATA / "v2"
POS = ("storfer", "inchausti")
FROZEN = ["effnet_B", "zoobot_N"]                       # unchanged v3 (stored seed columns)
SWAP = {"effnet_S2": "effnet_S2_hard", "effnet_B3": "effnet_B3_hard",
        "resnet46_C": "resnet46_C_hard"}               # member -> v2 stored column name
POS_PARQ = {"effnet_B": "data/scores_member_effnet_B.parquet",
            "zoobot_N": "data/v2/scores_member_zoobot_N.parquet",
            "effnet_S2_hard": "data/v2/scores_member_effnet_S2_hard.parquet",
            "effnet_B3_hard": "data/v2/scores_member_effnet_B3_hard.parquet",
            "resnet46_C_hard": "data/v2/scores_member_resnet46_C_hard.parquet"}


def iso_from_val(scores_parquet: str):
    v = pd.read_parquet(C.ROOT / scores_parquet)
    v = v[v["split"] == "val"]
    return E.IsotonicCalibrator().fit(v["p"].to_numpy(float), v["label"].to_numpy(int))


def pos_pc(scores_parquet: str, sp: str) -> pd.DataFrame:
    d = pd.read_parquet(C.ROOT / scores_parquet)
    return d[d["split"] == sp][["row_id", "pc"]].astype({"row_id": str})


def ens_pos(members_parq, sp) -> np.ndarray:
    mats = None
    for i, p in enumerate(members_parq):
        g = pos_pc(p, sp).rename(columns={"pc": f"m{i}"})
        mats = g if mats is None else mats.merge(g, on="row_id", how="inner")
    cols = [c for c in mats.columns if c.startswith("m")]
    return mats[cols].to_numpy(float).mean(axis=1)


def score_seed(ckpt: Path, val_parq: str, seed: pd.DataFrame, device) -> np.ndarray:
    """raw p -> isotonic(val) pc for one member on the 601 seed FITS."""
    model, score_arch, mean, std = M112.load_member_checkpoint(ckpt, device, None)
    d = seed[["row_id", "fits_path"]].copy()
    d["fits_dir"] = d["fits_path"].apply(lambda p: str(Path(p).parent))
    d["row_id"] = d["row_id"].astype(str)
    p = T.score_df(model, score_arch, d, mean, std, device)
    return iso_from_val(val_parq).transform(np.asarray(p, float))


def main() -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="b50")
    ap.add_argument("--seed-bank", default=str(C.DATA / "v3" / "mimic_bank_seed.parquet"))
    ap.add_argument("--ckpt-dir", default=str(V2 / "ckpt"))
    ap.add_argument("--out", default=str(C.DATA / "v3" / "a2_seed_compare.json"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = pd.read_parquet(args.seed_bank).astype({"row_id": str})
    n = len(seed)

    # v2 seed ensemble = mean of isotonic-calibrated stored member columns
    def iso_stored(col, member_name):
        cal = iso_from_val(POS_PARQ[member_name])
        return cal.transform(seed[col].to_numpy(float))
    v2_members = ["effnet_B", "effnet_B3_hard", "effnet_S2_hard", "resnet46_C_hard", "zoobot_N"]
    v2_mat = np.column_stack([iso_stored(f"member_{m}", m) for m in v2_members])
    v2_seed = v2_mat.mean(axis=1)

    # v3 seed ensemble: frozen (stored) + 3 retrained (re-scored on the 601 FITS)
    cols = [iso_stored("member_effnet_B", "effnet_B"),
            iso_stored("member_zoobot_N", "zoobot_N")]
    for member in SWAP:
        ck = Path(args.ckpt_dir) / f"member_{member}_{args.tag}.pt"
        valp = f"data/v2/scores_member_{member}_{args.tag}.parquet"
        if not ck.exists():
            print(f"[317] MISSING {ck} — pull the v3 ckpts from Perlmutter first"); return 2
        cols.append(score_seed(ck, valp, seed, device))
        print(f"[317] re-scored {member}_{args.tag} on {n} seed FITS")
    v3_seed = np.column_stack(cols).mean(axis=1)

    rep = {"tag": args.tag, "n_seed": n, "splits": {}}
    print(f"\n[317] ===== SEED-BANK recovery@mimic-FPR (601 curated, OOD)  v2 -> v3/{args.tag} =====")
    for sp in POS:
        pos_v2 = ens_pos([POS_PARQ[m] for m in v2_members], sp)
        pos_v3 = ens_pos([POS_PARQ["effnet_B"], POS_PARQ["zoobot_N"]]
                         + [f"data/v2/scores_member_{m}_{args.tag}.parquet" for m in SWAP], sp)
        r2 = mm.recovery_at_mimic_fpr(v2_seed, pos_v2)
        r3 = mm.recovery_at_mimic_fpr(v3_seed, pos_v3)
        seed_df_v3 = seed[["mimic_type"]].copy(); seed_df_v3["score"] = v3_seed
        pt = mm.per_type_recovery(seed_df_v3, pos_v3, "score")
        rep["splits"][sp] = {"v2": {str(k): r2[k]["recovery"] for k in r2},
                             "v3": {str(k): r3[k]["recovery"] for k in r3},
                             "v3_per_type": pt}
        print(f"  {sp:10s} @0.05: v2={r2[0.05]['recovery']:.3f} -> v3={r3[0.05]['recovery']:.3f}"
              f"   @0.01: v2={r2[0.01]['recovery']:.3f} -> v3={r3[0.01]['recovery']:.3f}")
        for t, d in sorted(pt.items(), key=lambda kv: kv[1]["recovery"].get("0.05", 9))[:5]:
            print(f"      v3 {t:16s} n={d['n_mimic']:4d} @0.05={d['recovery'].get('0.05', float('nan')):.3f}")
    Path(args.out).write_text(json.dumps(rep, indent=2, default=float))
    print(f"\n[317] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
