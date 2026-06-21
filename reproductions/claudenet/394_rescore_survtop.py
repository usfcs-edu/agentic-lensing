#!/usr/bin/env python3
"""394_rescore_survtop.py — deploy the DR11-native fine-tune on the survivor pool (task D):
score the top-N survivors with the DR11 fine-tuned ensemble, re-rank, and emit a refreshed
DR11-south candidate shortlist + the baseline-vs-finetune comparison (how the top-N changes,
known-lens recovery). Local; reuses 112's member loader (handles all archs) + _train.score_df.

  CUDA_VISIBLE_DEVICES=0 /home2/benson/.venvs/claudenet/bin/python 394_rescore_survtop.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import _clib as C

D = C.DATA; V2 = D / "v2"
LEAN = {"effnet_B": "ckpt_lean/member_effnet_B.pt", "zoobot_N": "ckpt_lean/member_zoobot_N.pt",
        "effnet_S2_hard": "ckpt_lean/member_effnet_S2_hard.pt",
        "effnet_B3_hard": "ckpt_lean/member_effnet_B3_hard.pt",
        "resnet46_C_hard": "ckpt_lean/member_resnet46_C_hard.pt"}
FT = {"effnet_B": "ckpt_lean/member_effnet_B.pt", "zoobot_N": "ckpt_lean/member_zoobot_N.pt",
      "effnet_S2": "ckpt/member_effnet_S2_b50_dr11.pt", "effnet_B3": "ckpt/member_effnet_B3_b50_dr11.pt",
      "resnet46_C": "ckpt/member_resnet46_C_b50_dr11.pt"}
VHINT = {"effnet_B": "tf_efficientnetv2_s", "zoobot_N": "convnext_nano",
         "effnet_S2_hard": "tf_efficientnetv2_s", "effnet_B3_hard": "tf_efficientnet_b3",
         "resnet46_C_hard": None, "effnet_S2": "tf_efficientnetv2_s",
         "effnet_B3": "tf_efficientnet_b3", "resnet46_C": None}


def xyz(ra, dec):
    ra = np.radians(np.asarray(ra, float)); dec = np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def main():
    import torch
    import _train as T
    M112 = C._load("c394_112", C.ROOT / "112_score_pool.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}

    def score(key, rel, df):
        if rel not in cache:
            cache[rel] = M112.load_member_checkpoint(V2 / rel, device, VHINT.get(key))
        model, sa, mean, std = cache[rel]
        return T.score_df(model, sa, df, mean, std, device)

    def ens(members, df):
        return np.mean(np.column_stack([score(k, p, df) for k, p in members.items()]), axis=1)

    # survtop table (extracted-ok) + old mean5 from the mean-selection
    ok = pd.read_parquet(D / "_dr11shards/survtop/index.parquet").query("ok").row_id.astype(str)
    sm = pd.read_parquet(D / "v3" / "survivors_dr11s_mean.parquet").astype({"row_id": str})
    df = sm[sm.row_id.isin(set(ok))][["row_id", "RA", "DEC", "mean5"]].copy()
    df["fits_dir"] = str(D / "cutouts_fits_dr11_survtop")
    print(f"[394] scoring {len(df):,} survivors with baseline + finetune ensembles")
    df["base_mean"] = ens(LEAN, df)
    df["ft_mean"] = ens(FT, df)

    # known-lens crossmatch (expanded pool incl. held-out + the 24 DR11-confirmed)
    known = [pd.read_parquet(D / "harvest" / "expanded_positives_dr11s.parquet")[["ra", "dec"]]
             .rename(columns={"ra": "RA", "dec": "DEC"})]
    cf = C.ROOT.parent / "dr11-campaign" / "data" / "dr11s_confirmed_AB.csv"
    if cf.exists():
        c = pd.read_csv(cf); rc = {k.lower(): k for k in c.columns}
        known.append(c[[rc["ra"], rc["dec"]]].rename(columns={rc["ra"]: "RA", rc["dec"]: "DEC"}))
    kn = pd.concat(known, ignore_index=True).dropna()
    tree = cKDTree(xyz(kn.RA, kn.DEC))
    d, _ = tree.query(xyz(df.RA, df.DEC), k=1)
    df["known"] = (np.degrees(2 * np.arcsin(np.clip(d / 2, 0, 1))) * 3600) < 5.0

    df = df.sort_values("ft_mean", ascending=False).reset_index(drop=True)
    df["ft_rank"] = np.arange(1, len(df) + 1)
    out = D / "v3" / "cv3_dr11s_finetune_candidates.csv"
    df.to_csv(out, index=False)

    def topN(col, n):
        t = df.nlargest(n, col)
        return set(t.row_id), int(t.known.sum())
    rep = {"n_scored": len(df)}
    for n in (300, 500, 1000):
        ft_ids, ft_known = topN("ft_mean", n)
        bs_ids, bs_known = topN("base_mean", n)
        rep[f"top{n}"] = {
            "ft_known": ft_known, "ft_new": n - ft_known,
            "base_known": bs_known, "base_new": n - bs_known,
            "overlap": len(ft_ids & bs_ids), "ft_only": len(ft_ids - bs_ids)}
    json.dump(rep, open(D / "v3" / "cv3_dr11s_finetune_summary.json", "w"), indent=2)

    print(f"[394] refreshed candidate list -> {out}")
    for n in (300, 500, 1000):
        r = rep[f"top{n}"]
        print(f"[394] top-{n}: finetune known/new {r['ft_known']}/{r['ft_new']}  "
              f"baseline known/new {r['base_known']}/{r['base_new']}  "
              f"overlap {r['overlap']}/{n} (finetune-only {r['ft_only']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
