#!/usr/bin/env python3
"""finetune/build_distill_v2.py — Phase 4 v2: continuous-p_lens distillation + AUC checkpoint selection.

v1 distillation used grade-BUCKETED target p_lens and selected by token-acc -> it overfit shortcuts and
INVERTED on held-out objects (detection AUC 0.42). v2 fixes both levers:
  (1) CONTINUOUS targets = Claude's actual p_lens/grade/criteria/rationale (true soft distillation);
  (2) a 3-way-disjoint VAL-SELECT detection set so checkpoints are chosen by detection AUC, not token-acc.

Two stages:
  --stage manifest : emit Claude-labeling manifest for TRAIN (lens + random + mimic) and a VAL-SELECT
                     detection set, both DISJOINT from the 270-row eval manifest and from each other.
                     -> outputs/sft_v2/{label_train.csv, valsel.csv}
  --stage sft --train-preds <claude_train.parquet> : render + write the SFT set with CONTINUOUS targets
                     (-> sft_v2/sft_train.jsonl) and the val-select eval set (-> sft_v2/valsel.jsonl +
                     valsel_labels.csv) for eval_checkpoints.py.

Between the stages, grade TRAIN with the Claude teacher:
  python -m lensjudge.imaging.run_batch --mode direct --manifest outputs/sft_v2/label_train.csv \
         --out outputs/sft_v2/claude_train.parquet   # anthropic backend; ~$0.012/cand
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import io  # noqa: E402
from lensjudge.common.schemas import ImageGrade  # noqa: E402
from lensjudge.imaging.grader_direct import _DEFAULT_VIEWS  # noqa: E402
from lensjudge.finetune.build_sft_data import _render_and_write, _target_json  # noqa: E402

SEED = 2026


def _eval_names() -> set:
    f = config.OUT / "lensbench_manifest.csv"
    return set(pd.read_csv(f)["name"].astype(str)) if f.exists() else set()


def _stage_manifest(out_dir: Path, n_train_lens, n_train_neg, n_val_lens, n_val_neg):
    excl = _eval_names()
    rng = np.random.RandomState(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    # lenses (human A/B/C), shuffle then split train / valsel (disjoint)
    cand = io.load_candidates("both")
    cand = cand[cand["grade"].isin(["A", "B", "C"]) & ~cand["name"].astype(str).isin(excl)]
    cand = cand.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    tr_lens = cand.iloc[:n_train_lens]
    va_lens = cand.iloc[n_train_lens:n_train_lens + n_val_lens]

    # negatives: random + mimic pools, shuffled, split train/valsel
    def _neg_pool():
        rows = []
        negp = config.INCH_DATA / "negatives_extra.parquet"
        if negp.exists():
            for _, r in pd.read_parquet(negp).iterrows():
                rows.append(dict(name=str(r["row_id"]), ra=r.get("RA"), dec=r.get("DEC"),
                                 survey_key="storfer", source="random_neg"))
        mimp = config.REPRO / "claudenet" / "data" / "v3" / "mimic_bank_seed.parquet"
        if mimp.exists():
            for _, r in pd.read_parquet(mimp).iterrows():
                rows.append(dict(name=str(r["row_id"]), ra=r.get("RA"), dec=r.get("DEC"),
                                 survey_key="claudenet", source="mimic_neg"))
        df = pd.DataFrame(rows)
        df = df[~df["name"].astype(str).isin(excl)]
        return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    neg = _neg_pool()
    tr_neg = neg.iloc[:n_train_neg]
    va_neg = neg.iloc[n_train_neg:n_train_neg + n_val_neg]

    # TRAIN labeling manifest (Claude grades these): lens + neg, grade_truth is a placeholder
    tr = []
    for _, r in tr_lens.iterrows():
        tr.append(dict(name=str(r["name"]), ra=r.get("ra"), dec=r.get("dec"),
                       survey_key=r["catalog"], catalog=r["catalog"], grade_truth=r["grade"],
                       binary_label="lens"))
    for _, r in tr_neg.iterrows():
        tr.append(dict(name=r["name"], ra=r.get("ra"), dec=r.get("dec"),
                       survey_key=r["survey_key"], catalog=r["survey_key"], grade_truth="D",
                       binary_label="nonlens"))
    pd.DataFrame(tr).to_csv(out_dir / "label_train.csv", index=False)

    # VAL-SELECT detection set (true labels; NO Claude needed — used to pick the best checkpoint by AUC)
    va = []
    for _, r in va_lens.iterrows():
        va.append(dict(name=str(r["name"]), ra=r.get("ra"), dec=r.get("dec"),
                       survey_key=r["catalog"], catalog=r["catalog"], grade_truth=r["grade"],
                       binary_label="lens", source="graded"))
    for _, r in va_neg.iterrows():
        va.append(dict(name=r["name"], ra=r.get("ra"), dec=r.get("dec"),
                       survey_key=r["survey_key"], catalog=r["survey_key"], grade_truth="D",
                       binary_label="nonlens", source=r["source"]))
    pd.DataFrame(va).to_csv(out_dir / "valsel.csv", index=False)
    print(f"[v2 manifest] train: {len(tr)} ({len(tr_lens)} lens / {len(tr_neg)} neg) -> label_train.csv")
    print(f"[v2 manifest] valsel: {len(va)} ({len(va_lens)} lens / {len(va_neg)} neg) -> valsel.csv")
    print(f"[v2 manifest] disjoint from {len(excl)} eval names + train/val disjoint by slicing")


def _continuous_target(r: dict) -> str:
    """Build an ImageGrade JSON from Claude's CONTINUOUS prediction row (true soft distillation)."""
    crit = {k: int(r.get(f"crit_{k}", 0) or 0) for k in
            ("blue_source", "low_surface_brightness", "curvature", "counter_images", "arc_morphology")}
    g = str(r.get("grade_pred")) if r.get("grade_pred") in ("A", "B", "C", "D") else "D"
    return _target_json(g, contaminant=r.get("contaminant"), rationale=str(r.get("rationale") or "")[:300],
                        crit=crit, p_lens=float(r.get("p_lens") or 0.0),
                        confidence=float(r.get("confidence") or 0.6))


def _stage_sft(out_dir: Path, train_preds: Path):
    # TRAIN: Claude preds -> continuous targets. Join to label_train for cutout resolution.
    lab = pd.read_csv(out_dir / "label_train.csv").set_index("name")
    pr = pd.read_parquet(train_preds)
    pr = pr[pr["parse_ok"] == True]  # noqa: E712
    rows = []
    for _, r in pr.iterrows():
        nm = str(r["name"])
        if nm not in lab.index:
            continue
        meta = lab.loc[nm]
        rows.append(dict(name=nm, ra=meta.get("ra"), dec=meta.get("dec"),
                         survey_key=meta.get("survey_key"),
                         label=meta.get("binary_label"), target=_continuous_target(r.to_dict())))
    recs = _render_and_write(rows, out_dir / "images", _DEFAULT_VIEWS)
    import json
    with open(out_dir / "sft_train.jsonl", "w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")
    npos = sum(1 for r in recs if r["label"] == "lens")
    print(f"[v2 sft] train: {len(recs)} ({npos} lens / {len(recs)-npos} neg) -> sft_train.jsonl")

    # VAL-SELECT: render with a placeholder target (swift infer ignores assistant); keep true labels.
    va = pd.read_csv(out_dir / "valsel.csv")
    vrows = [dict(name=str(r["name"]), ra=r.get("ra"), dec=r.get("dec"), survey_key=r["survey_key"],
                  label=r["binary_label"], target=_target_json("D", rationale="")) for _, r in va.iterrows()]
    vrecs = _render_and_write(vrows, out_dir / "images_valsel", _DEFAULT_VIEWS)
    with open(out_dir / "valsel.jsonl", "w") as fh:
        for rec in vrecs:
            fh.write(json.dumps(rec) + "\n")
    # name<-order alignment: store labels in the SAME order as valsel.jsonl rows
    lab_rows = [{"idx": i, "label": rec["label"],
                 "name": rec["images"][0].split("/")[-1].rsplit("_", 1)[0]} for i, rec in enumerate(vrecs)]
    pd.DataFrame(lab_rows).to_csv(out_dir / "valsel_labels.csv", index=False)
    print(f"[v2 sft] valsel: {len(vrecs)} -> valsel.jsonl + valsel_labels.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("manifest", "sft"))
    ap.add_argument("--out-dir", default=str(config.OUT / "sft_v2"))
    ap.add_argument("--train-preds", default=None, help="(sft) Claude train preds parquet")
    ap.add_argument("--n-train-lens", type=int, default=700)
    ap.add_argument("--n-train-neg", type=int, default=300)
    ap.add_argument("--n-val-lens", type=int, default=60)
    ap.add_argument("--n-val-neg", type=int, default=60)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.stage == "manifest":
        _stage_manifest(out_dir, args.n_train_lens, args.n_train_neg, args.n_val_lens, args.n_val_neg)
    else:
        if not args.train_preds:
            raise SystemExit("--train-preds required for --stage sft")
        _stage_sft(out_dir, Path(args.train_preds))


if __name__ == "__main__":
    main()
