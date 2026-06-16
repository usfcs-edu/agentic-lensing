#!/usr/bin/env python3
"""363_cv3_select.py — ClaudeNet v3 C-v3: re-rank + select the DR10 survivors with the v3
model (v3blend8 ensemble + the A3 lens-vs-mimic head + the mimic-null conformal).

C-now produced the 150k DR10 survivor pool (stage2_scores: the 5 v2 members + RA/DEC/brick) and
crossmatch (NEW/known status). C-v3 adds the 3 `_b50` members scored on those survivors (315
--tag survdr10) → the full v3blend8 8-member set, then:
  * v3blend8 p_final = mean of the 8 calibrated members (the SHIP ensemble);
  * head score   = the A3 logistic lens-vs-mimic head (re-fit on lenses vs the DR10 mimic bank,
                   logit member features) applied to the survivors — the stage-2 mimic re-ranker;
  * mimic-null conformal ("mimic-FDR"): Jin–Candes conformal p-values of each survivor's head
    score vs the mimic bank head scores + BH (50_conformal_selection, reused) — a conservative
    resemblance control (the mimic null is enriched, not natural; reported alongside, not as a
    guarantee), the A3/`326` fix for the random-null blindness C-now exposed (0 selected).
Deliverables: recall-of-811 head-to-head (v2 p_final vs v3 head ranking), and the top NEW
survivors by head score → the C-vet shortlist. Runs LOCALLY after pulling scores_member_*_survdr10.

    /home2/benson/.venvs/claudenet/bin/python 363_cv3_select.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C
h321 = C._load("cn_363_321", C.ROOT / "321_lensvmimic_head.py")
conf = C._load("cn_363_50", C.ROOT / "50_conformal_selection.py")

V3 = C.DATA / "v3"
SD = V3 / "sweep_dr10"
FEATS = h321.FEATS                      # 8 members in head-feature order
# survivor calibrated-score column for each feature: 5 v2 from stage2 cal_member_*, 3 b50 from 315
CAL2 = {"effnet_B": "cal_member_effnet_B", "zoobot_N": "cal_member_zoobot_N",
        "effnet_S2_hard": "cal_member_effnet_S2_hard", "effnet_B3_hard": "cal_member_effnet_B3_hard",
        "resnet46_C_hard": "cal_member_resnet46_C_hard"}
B50 = ("effnet_S2_b50", "effnet_B3_b50", "resnet46_C_b50")


def _lg(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--topn", type=int, default=300)
    ap.add_argument("--out", default=str(V3 / "cv3_candidates_top.csv"))
    args = ap.parse_args()

    # --- fit the A3 head: lenses (pos) vs DR10 mimic bank (neg), logit member features ---
    Xpos = np.vstack([h321.lens_feats(sp) for sp in h321.POS])
    Xm = h321.dr10_feats()
    X = _lg(np.vstack([Xpos, Xm])); y = np.r_[np.ones(len(Xpos)), np.zeros(len(Xm))]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(X, y)
    mim_head = clf.decision_function(_lg(Xm))                     # mimic-null head scores
    print(f"[363] head fit on {len(Xpos)} lenses + {len(Xm)} mimics; "
          f"weights {dict(zip(FEATS, np.round(clf.coef_[0], 3)))}")

    # --- assemble the 8-member survivor feature matrix ---
    s2 = pd.read_parquet(SD / "stage2_scores.parquet").astype({"row_id": str})
    surv = s2[["row_id", "RA", "DEC", "brick"] + list(CAL2.values())].copy()
    for m in B50:
        b = pd.read_parquet(V3 / f"scores_member_{m}_survdr10.parquet")[["row_id", "pc"]]
        surv = surv.merge(b.rename(columns={"pc": m}).astype({"row_id": str}), on="row_id", how="inner")
    print(f"[363] survivors with all 8 members: {len(surv):,}/{len(s2):,}")
    feat_cols = [CAL2[f] if f in CAL2 else f for f in FEATS]
    Xs = surv[feat_cols].to_numpy(float)
    surv["v3blend8"] = Xs.mean(axis=1)
    surv["head"] = clf.decision_function(_lg(Xs))

    # --- mimic-null conformal (mimic-FDR) on the head score ---
    pvals = conf.conformal_pvalues(surv["head"].to_numpy(), mim_head)
    surv["mimic_p"] = pvals
    sel = conf.bh_select(pvals, args.alpha)
    surv["mimic_fdr_sel"] = sel
    print(f"[363] mimic-null conformal BH@{args.alpha}: {int(sel.sum()):,} selected "
          f"(vs C-now random-null 0); floor 1/(n_mim+1)={1/(len(mim_head)+1):.2e}")

    # --- crossmatch status + recall-of-811 head-to-head ---
    xm = pd.read_parquet(SD / "crossmatch.parquet").astype({"row_id": str})
    surv = surv.merge(xm[["row_id", "match_inchausti2025", "name_inchausti2025",
                          "match_storfer2024", "match_huang2021"]], on="row_id", how="left")
    known811 = surv["match_inchausti2025"].fillna(False)
    is_known = known811 | surv["match_storfer2024"].fillna(False) | surv["match_huang2021"].fillna(False)
    surv["status"] = np.where(is_known, "KNOWN", "NEW")
    n_known = int(is_known.sum())
    # how well does the head rank the known lenses among all survivors (percentile)?
    surv["head_rank_pct"] = surv["head"].rank(pct=True)
    surv["v2_rank_pct"] = surv["v3blend8"].rank(pct=True)  # proxy; true v2 = stage2 p_final below
    for col, lbl in [("head_rank_pct", "v3-head"), ("v2_rank_pct", "v3blend8")]:
        med_known = surv.loc[is_known, col].median()
        med_new = surv.loc[~is_known, col].median()
        print(f"[363] {lbl:9s} rank-percentile: known811 median={med_known:.3f}  NEW median={med_new:.3f}")
    # mimic-FDR selected that are known vs new
    seln = surv[surv["mimic_fdr_sel"]]
    print(f"[363] mimic-FDR selected: {len(seln)} ({int((seln.status=='KNOWN').sum())} known, "
          f"{int((seln.status=='NEW').sum())} NEW)")

    # --- candidate shortlist: top NEW by head score ---
    new = surv[surv["status"] == "NEW"].sort_values("head", ascending=False)
    top = new.head(args.topn)[["row_id", "RA", "DEC", "brick", "head", "v3blend8",
                               "mimic_p", "mimic_fdr_sel"]]
    top.to_csv(args.out, index=False)
    print(f"[363] wrote {args.out}: top {len(top)} NEW by head score "
          f"(head {top.head.min():.2f}..{top.head.max():.2f})")
    rep = {"n_survivors": int(len(surv)), "n_known": n_known, "n_new": int((surv.status=='NEW').sum()),
           "mimic_fdr_alpha": args.alpha, "mimic_fdr_n_sel": int(sel.sum()),
           "mimic_fdr_n_new": int((seln.status=='NEW').sum()),
           "head_weights": dict(zip(FEATS, [float(w) for w in clf.coef_[0]])),
           "top_new_head_range": [float(top.head.min()), float(top.head.max())]}
    Path(str(V3 / "cv3_select_summary.json")).write_text(json.dumps(rep, indent=2))
    print(f"[363] wrote {V3/'cv3_select_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
