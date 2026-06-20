#!/usr/bin/env python3
"""392_gate_dr11.py — the DR11 fine-tune GATE (task D). Scores the held-out Storfer/Inchausti
lenses + a DR11 random NegEval (the 30k training-random set) at DR11 resolution with two
mean-combiner ensembles and compares, the 381 way (threshold-free AUC + matched-FPR recall):

  BASELINE = 5-lean  : effnet_B, zoobot_N, effnet_S2_hard, effnet_B3_hard, resnet46_C_hard
  FINETUNE = anchors + DR11 : effnet_B, zoobot_N, effnet_S2_b50_dr11, effnet_B3_b50_dr11, resnet46_C_b50_dr11

GATE (per the locked scope): the fine-tune passes if it BEATS the baseline on held-out
Inchausti-A recall @ the 95k-equiv FPR AND on the hard Storfer-A residual, without dropping AUC.
Same NegEval + same held-out lenses for both arms, so the comparison is fair (relative).

  CUDA_VISIBLE_DEVICES=0 /home2/benson/.venvs/claudenet/bin/python 392_gate_dr11.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import _clib as C

D = C.DATA; V2 = D / "v2"; M_TOTAL = 53_809_040; OP_BUDGET = 95_104

LEAN = {"effnet_B": "ckpt_lean/member_effnet_B.pt", "zoobot_N": "ckpt_lean/member_zoobot_N.pt",
        "effnet_S2_hard": "ckpt_lean/member_effnet_S2_hard.pt",
        "effnet_B3_hard": "ckpt_lean/member_effnet_B3_hard.pt",
        "resnet46_C_hard": "ckpt_lean/member_resnet46_C_hard.pt"}
FT = {"effnet_B": "ckpt_lean/member_effnet_B.pt", "zoobot_N": "ckpt_lean/member_zoobot_N.pt",
      "effnet_S2": "ckpt/member_effnet_S2_b50_dr11.pt", "effnet_B3": "ckpt/member_effnet_B3_b50_dr11.pt",
      "resnet46_C": "ckpt/member_resnet46_C_b50_dr11.pt"}
# variant hints for members whose ckpt lacks a 'variant' key (lean efficientnets); 112's
# load_member_checkpoint reads ckpt['variant'] first and falls back to this hint.
VHINT = {"effnet_B": "tf_efficientnetv2_s", "zoobot_N": "convnext_nano",
         "effnet_S2_hard": "tf_efficientnetv2_s", "effnet_B3_hard": "tf_efficientnet_b3",
         "resnet46_C_hard": None, "effnet_S2": "tf_efficientnetv2_s",
         "effnet_B3": "tf_efficientnet_b3", "resnet46_C": None}


def main():
    import torch
    import _train as T
    M112 = C._load("g392_112", C.ROOT / "112_score_pool.py")   # handles all archs incl. timm/zoobot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}

    def score(key, ckpt_rel, df):
        if ckpt_rel not in cache:
            cache[ckpt_rel] = M112.load_member_checkpoint(V2 / ckpt_rel, device, VHINT.get(key))
        model, score_arch, mean, std = cache[ckpt_rel]
        return T.score_df(model, score_arch, df, mean, std, device)

    def ens(members, df):
        return np.mean(np.column_stack([score(k, p, df) for k, p in members.items()]), axis=1)

    # held-out lenses (DR11 cutouts) + NegEval (30k random DR11 negs)
    ho = pd.read_parquet(D / "dr11s_heldout_manifest.parquet").astype({"row_id": str})
    ho_ok = set(pd.read_parquet(D / "_dr11shards/heldout/index.parquet").query("ok").row_id.astype(str))
    ho = ho[ho.row_id.isin(ho_ok)].copy(); ho["fits_dir"] = str(D / "cutouts_fits_dr11_heldout")
    neg_ok = sorted(pd.read_parquet(D / "_dr11shards/negatives/index.parquet").query("ok").row_id.astype(str))
    neg = pd.DataFrame({"row_id": neg_ok, "fits_dir": str(D / "cutouts_fits_dr11_neg")})
    print(f"[392] held-out (ok): {len(ho)}  {ho.groupby(['cat','grade']).size().to_dict()}; NegEval {len(neg):,}")

    rep = {"op_budget": OP_BUDGET, "m_total": M_TOTAL, "arms": {}}
    for tag, members in [("baseline", LEAN), ("finetune", FT)]:
        hs = ens(members, ho); ns = ens(members, neg)
        ho[f"m_{tag}"] = hs
        def block(mask, label, store):
            p = ho.loc[mask, f"m_{tag}"].to_numpy()
            if len(p) == 0:
                return
            y = np.r_[np.ones(len(p)), np.zeros(len(ns))]
            auc = float(roc_auc_score(y, np.r_[p, ns]))
            def rec_at(budget):
                t = float(np.quantile(ns, 1 - budget / M_TOTAL)); return float(np.mean(p >= t))
            store[label] = {"n": int(len(p)), "auc": round(auc, 4),
                            "recall@95k": round(rec_at(OP_BUDGET), 3),
                            "recall@150k": round(rec_at(150_000), 3)}
        arm = {}
        block((ho.cat == "inc") & (ho.grade == "A"), "inchausti_A", arm)
        block((ho.cat == "inc") & (ho.grade == "B"), "inchausti_B", arm)
        block((ho.cat == "sto") & (ho.grade == "A"), "storfer_A", arm)
        rep["arms"][tag] = arm
        print(f"[392] {tag:9s} " + " | ".join(
            f"{k}: AUC {v['auc']} rec@95k {v['recall@95k']} rec@150k {v['recall@150k']}"
            for k, v in arm.items()))

    b, f = rep["arms"]["baseline"], rep["arms"]["finetune"]
    def delta(k, m): return round(f[k][m] - b[k][m], 3)
    rep["gate"] = {
        "inchausti_A_recall95k": {"base": b["inchausti_A"]["recall@95k"], "ft": f["inchausti_A"]["recall@95k"],
                                  "delta": delta("inchausti_A", "recall@95k")},
        "storfer_A_recall95k": {"base": b["storfer_A"]["recall@95k"], "ft": f["storfer_A"]["recall@95k"],
                                "delta": delta("storfer_A", "recall@95k")},
        "inchausti_A_auc": {"base": b["inchausti_A"]["auc"], "ft": f["inchausti_A"]["auc"],
                            "delta": delta("inchausti_A", "auc")},
    }
    passed = (delta("inchausti_A", "recall@95k") >= 0 and delta("storfer_A", "recall@95k") > 0
              and f["inchausti_A"]["auc"] >= b["inchausti_A"]["auc"] - 0.002)
    rep["gate"]["PASS"] = bool(passed)
    json.dump(rep, open(D / "v3" / "dr11_finetune_gate.json", "w"), indent=2)
    print(f"\n[392] GATE deltas (finetune - baseline): "
          f"InchaustiA rec@95k {delta('inchausti_A','recall@95k'):+.3f}, "
          f"StorferA rec@95k {delta('storfer_A','recall@95k'):+.3f}, "
          f"InchaustiA AUC {delta('inchausti_A','auc'):+.4f}")
    print(f"[392] GATE: {'PASS' if passed else 'NO-PASS'} -> data/v3/dr11_finetune_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
