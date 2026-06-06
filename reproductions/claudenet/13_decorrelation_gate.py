#!/usr/bin/env python3
"""13_decorrelation_gate.py — the Phase-0 KILL/VALIDATE gate for the flagship.

Question: are AION-1 frozen-embedding scores decorrelated enough from the
EfficientNetV2 baseline to make an engineered-diversity ensemble worthwhile?

Procedure (matched-FPR throughout, via _ensemble.recovery_at_fpr):
  1. p_aion: from 12_probe_aion (held-out testneg/storfer/inchausti).
  2. p_effnet: score the SAME objects (same manifest fits_path/order) with the
     staged EfficientNetV2 checkpoint.
  3. correlation: Pearson + Spearman between p_aion and p_effnet on the test
     negatives (and on the candidate positives).
  4. combine: rank-normalise each member to the test-negative ECDF (puts both on a
     common matched-FPR scale, removing raw-scale domination), then average; report
     recovery@1%/0.1% FPR for {aion, effnet, rank-avg}.

Verdict:
  KILL     if r_spearman > 0.9  AND  rank-avg does NOT beat the best single member.
  VALIDATE if r_spearman <= 0.9 AND  rank-avg already beats the best single member.
  else AMBIGUOUS -> proceed to Phase 1 but treat the calibrated nonlinear combiner
       (not the average) as decisive, and add the native-griz AION member.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      /home2/benson/.venvs/claudenet/bin/python 13_decorrelation_gate.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

import _clib as C
import _scorelib as SL
import _ensemble as E


def rank_normalize(scores, ref_neg):
    """Map scores to the empirical CDF of ref_neg (the test negatives), giving a
    common [0,1] matched-FPR scale across members."""
    ref = np.sort(np.asarray(ref_neg, dtype=np.float64))
    return np.searchsorted(ref, np.asarray(scores, dtype=np.float64), side="right") / max(len(ref), 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aion = pd.read_parquet(C.DATA / "scores_aion_gate.parquet")

    # score the SAME objects with the staged EfficientNet (aligned by manifest order)
    ef, _, m_ef, s_ef, _ = SL.load_checkpoint_model(C.DATA / "checkpoint_best_efficientnet_staged.pt", device)
    aion["p_effnet"] = np.nan
    for sp in ("testneg", "storfer", "inchausti"):
        m = aion.split == sp
        paths = aion.loc[m, "fits_path"].tolist()
        aion.loc[m, "p_effnet"] = SL.score_paths(paths, ef, "efficientnet", m_ef, s_ef, device)

    aion = aion[np.isfinite(aion["p_aion"]) & np.isfinite(aion["p_effnet"])].copy()
    aion.to_parquet(C.DATA / "scores_gate_merged.parquet", index=False)
    neg = aion[aion.split == "testneg"]
    pa_neg, pe_neg = neg["p_aion"].to_numpy(), neg["p_effnet"].to_numpy()

    # --- correlation (decorrelation measure) ---
    r_p = float(pearsonr(pa_neg, pe_neg)[0])
    r_s = float(spearmanr(pa_neg, pe_neg)[0])

    # --- rank-normalised average member ---
    aion["q_aion"] = rank_normalize(aion["p_aion"], pa_neg)
    aion["q_effnet"] = rank_normalize(aion["p_effnet"], pe_neg)
    aion["q_avg"] = 0.5 * (aion["q_aion"] + aion["q_effnet"])

    res = {"correlation": {"pearson": r_p, "spearman": r_s}, "recovery": {}}
    members = {"aion": "p_aion", "effnet": "p_effnet", "rank_avg": "q_avg"}
    negm = aion[aion.split == "testneg"]
    print(f"\n{'member':>9} {'cat':>10} {'rec@1%':>8} {'rec@0.1%':>9}")
    rec_at_1 = {}
    for mname, col in members.items():
        ncol = negm[col].to_numpy()
        for cat in ("storfer", "inchausti"):
            cs = aion[aion.split == cat][col].to_numpy()
            rec = E.recovery_at_fpr(ncol, cs, fprs=C.TARGET_FPR)
            res["recovery"][f"{mname}|{cat}"] = {f"{fpr}": rec[fpr]["recovery"] for fpr in C.TARGET_FPR}
            rec_at_1[(mname, cat)] = rec[0.01]["recovery"]
            print(f"{mname:>9} {cat:>10} {rec[0.01]['recovery']:>8.3f} {rec[0.001]['recovery']:>9.3f}")

    # --- learned combiner (5-fold OOF logistic+RF on [p_aion,p_effnet]) ---
    # A naive average is dragged down by the weaker member; a trained combiner can
    # exploit the decorrelation. This is the decisive test for the AMBIGUOUS case.
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    lab = aion[aion.split.isin(["testneg", "storfer", "inchausti"])]
    yl = (lab.split != "testneg").astype(int).to_numpy()
    Xl = lab[["p_aion", "p_effnet"]].to_numpy()
    spl = lab.split.to_numpy()
    oof = {"logistic": np.zeros(len(yl)), "rf": np.zeros(len(yl))}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(Xl, yl):
        oof["logistic"][te] = LogisticRegression(max_iter=2000).fit(Xl[tr], yl[tr]).predict_proba(Xl[te])[:, 1]
        oof["rf"][te] = RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=25,
                                               random_state=C.SEED, n_jobs=-1).fit(Xl[tr], yl[tr]).predict_proba(Xl[te])[:, 1]
    comb_rec = {}
    for cname, s in oof.items():
        thr = float(np.quantile(s[spl == "testneg"], 0.99))
        comb_rec[cname] = float((s[spl == "storfer"] >= thr).mean())
    best_combiner = max(comb_rec.values())
    res["learned_combiner_storfer@1%"] = comb_rec
    print(f"[gate] learned combiner storfer@1%FPR: logistic={comb_rec['logistic']:.3f} "
          f"rf={comb_rec['rf']:.3f}")

    # --- verdict (headline catalog = storfer) ---
    best_single = max(rec_at_1[("aion", "storfer")], rec_at_1[("effnet", "storfer")])
    avg_beats = rec_at_1[("rank_avg", "storfer")] > best_single + 1e-9
    combiner_beats = best_combiner > best_single + 1e-9
    res["combiner_beats_best_single"] = bool(combiner_beats)
    if r_s > 0.9 and not (avg_beats or combiner_beats):
        verdict = "KILL"
    elif r_s <= 0.9 and (avg_beats or combiner_beats):
        verdict = "VALIDATE"
    else:
        verdict = "AMBIGUOUS"
    res["best_single_storfer@1%"] = best_single
    res["rank_avg_storfer@1%"] = rec_at_1[("rank_avg", "storfer")]
    res["avg_beats_best_single"] = bool(avg_beats)
    res["verdict"] = verdict

    (C.DATA / "gate_phase0.json").write_text(json.dumps(res, indent=2))
    print(f"\n[gate] Spearman r(aion,effnet) on test-neg = {r_s:.3f}  (Pearson {r_p:.3f})")
    print(f"[gate] storfer@1%FPR: best single = {best_single:.3f}, rank-avg = "
          f"{rec_at_1[('rank_avg','storfer')]:.3f}  -> avg_beats_best={avg_beats}")
    print(f"[gate] VERDICT: {verdict}")
    print("  KILL = redundant w/ effnet; VALIDATE = decorrelated & additive; "
          "AMBIGUOUS = proceed to Phase 1, decide on the calibrated combiner.")
    return 0


if __name__ == "__main__":
    main()
