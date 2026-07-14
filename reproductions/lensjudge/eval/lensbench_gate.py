#!/usr/bin/env python3
"""eval/lensbench_gate.py — LensBench-VI gate for the v4 open-weight grader.

Compares an OPEN-weight grader's predictions against the Claude ORACLE on the frozen,
hash-pinned LensBench-VI manifest (eval/build_eval_set.py), and emits a PASS/FAIL verdict.

Unlike eval/score.py (which lumps all negatives into one "non-lens" class), this gate joins
the preds back to the manifest's ``source`` column and reports the two AUCs the v4 plan gates on
separately:
  - DETECTION  AUC = lens (A/B/C) vs RANDOM galaxies        (source=random_neg)
  - LENS-vs-MIMIC AUC = lens vs hard CNN mimics             (source=mimic_neg)
plus recovery@1%FPR (vs random), parse rate, and cost/throughput.

Produce both preds parquets with the SAME manifest + render settings, only the backend differs:
  # Claude oracle (default backend):
  python -m lensjudge.imaging.run_batch --mode direct --manifest outputs/lensbench_manifest.csv \
         --out outputs/preds_oracle.parquet
  # open-weight (server up; see serving/README.md):
  LENSJUDGE_BACKEND=openai LENSJUDGE_BASE_URL=http://localhost:8000/v1 \
  LENSJUDGE_MODEL_GRADER=Qwen/Qwen3-VL-8B-Instruct \
  python -m lensjudge.imaging.run_batch --mode direct --manifest outputs/lensbench_manifest.csv \
         --out outputs/preds_qwen8b.parquet
  # gate:
  python -m lensjudge.eval.lensbench_gate --open outputs/preds_qwen8b.parquet \
         --oracle outputs/preds_oracle.parquet --manifest outputs/lensbench_manifest.csv

Optional INT4-vs-FP16 grade-flip safety check:
  python -m lensjudge.eval.lensbench_gate --flip-a outputs/preds_int4.parquet \
         --flip-b outputs/preds_fp16.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.eval.score import recovery_at_fpr  # noqa: E402

LENS_GRADES = ("A", "B", "C")


def merge_manifest(preds: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Join preds to the manifest's source/binary_label by name (manifest is authoritative)."""
    man = manifest[["name", "source", "binary_label"]].copy()
    m = preds.merge(man, on="name", how="left", suffixes=("", "_man"))
    m = m[m["binary_label"].notna()].copy()
    m["is_lens"] = (m["binary_label"] == "lens")
    return m


def _auc(merged: pd.DataFrame, neg_source: str | None) -> tuple:
    """ROC-AUC of p_lens for lens vs a chosen negative pool. neg_source=None -> all non-lens.
    Parse failures (p_lens NaN) count as p_lens=0 (no lens evidence), matching score.py."""
    from sklearn.metrics import roc_auc_score
    pos = merged[merged["is_lens"]]
    if neg_source is None:
        neg = merged[(~merged["is_lens"]) & (merged["binary_label"] == "nonlens")]
    else:
        neg = merged[merged["source"] == neg_source]
    sub = pd.concat([pos, neg])
    if len(pos) == 0 or len(neg) == 0 or sub["is_lens"].nunique() < 2:
        return None, len(pos), len(neg)
    y = sub["is_lens"].astype(int)
    s = sub["p_lens"].fillna(0.0)
    return round(float(roc_auc_score(y, s)), 4), len(pos), len(neg)


def gate_metrics(merged: pd.DataFrame) -> dict:
    """Per-backend LensBench-VI metrics with negative-source splits."""
    n = len(merged)
    ok = merged[merged["parse_ok"] == True]  # noqa: E712
    det_auc, n_pos, n_rand = _auc(merged, "random_neg")
    mim_auc, _, n_mim = _auc(merged, "mimic_neg")
    gd_auc, _, n_gd = _auc(merged, "graded_D")
    all_auc, _, n_neg = _auc(merged, None)

    # recovery @1% FPR vs random negatives
    sub = merged[merged["is_lens"] | (merged["source"] == "random_neg")]
    rec, _ = recovery_at_fpr(sub["is_lens"].astype(int), sub["p_lens"].fillna(0.0), 0.01) \
        if len(sub) and sub["is_lens"].nunique() > 1 else (float("nan"), float("nan"))

    out = {
        "n": int(n),
        "parse_rate": round(len(ok) / max(1, n), 4),
        "n_pos": int(n_pos), "n_random": int(n_rand), "n_mimic": int(n_mim),
        "n_gradeD": int(n_gd), "n_nonlens": int(n_neg),
        "detection_auc": det_auc,            # lens vs random  (the Stage-1 headline)
        "mimic_auc": mim_auc,                # lens vs hard mimic (informational at Stage 1)
        "gradeD_auc": gd_auc,                # lens vs human Grade-D rejects
        "overall_auc": all_auc,              # lens vs all non-lens
        "recovery@0.01FPR": None if (rec is None or np.isnan(rec)) else round(float(rec), 4),
    }
    if "p_lens" in merged and (merged["grade_truth"] == "A").any():
        out["mean_p_lens_A"] = round(float(
            merged[merged["grade_truth"] == "A"]["p_lens"].mean()), 3)
    for c, key in (("cost_usd", "mean_cost_usd"), ("wall_s", "mean_wall_s")):
        if c in merged and merged[c].notna().any():
            out[key] = round(float(merged[c].mean()), 4)
    return out


def grade_flip_rate(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Fraction of shared candidates whose grade_pred / lens-call differ between two runs
    (e.g. INT4 vs FP16) — a quantization-safety check."""
    j = a[["name", "grade_pred", "p_lens"]].merge(
        b[["name", "grade_pred", "p_lens"]], on="name", suffixes=("_a", "_b"))
    if not len(j):
        return {"n_shared": 0}
    grade_flip = (j["grade_pred_a"] != j["grade_pred_b"]).mean()
    lens_a = j["grade_pred_a"].isin(LENS_GRADES)
    lens_b = j["grade_pred_b"].isin(LENS_GRADES)
    lens_flip = (lens_a != lens_b).mean()
    dp = (j["p_lens_a"].fillna(0) - j["p_lens_b"].fillna(0)).abs()
    return {"n_shared": int(len(j)), "grade_flip_rate": round(float(grade_flip), 4),
            "lens_call_flip_rate": round(float(lens_flip), 4),
            "mean_abs_p_lens_delta": round(float(dp.mean()), 4)}


def verdict(open_m: dict, oracle_m: dict, tol_auc: float, min_parse: float) -> dict:
    """PASS iff the open grader matches the oracle on DETECTION AUC within tol AND parses cleanly.
    (Mimic AUC and recovery are reported but not Stage-1-critical — mimic is the Phase-3 agentic
    job; here we gate the cheap detection workhorse.)"""
    checks = []

    def _chk(name, cond, detail):
        checks.append({"check": name, "pass": bool(cond), "detail": detail})

    od, rd = open_m.get("detection_auc"), oracle_m.get("detection_auc")
    if od is not None and rd is not None:
        _chk("detection_auc_within_tol", od >= rd - tol_auc,
             f"open {od} vs oracle {rd} (tol {tol_auc}; Δ={round(od - rd, 4)})")
    else:
        _chk("detection_auc_within_tol", False, f"missing AUC (open={od}, oracle={rd})")
    _chk("parse_rate_ok", open_m.get("parse_rate", 0) >= min_parse,
         f"open parse_rate {open_m.get('parse_rate')} >= {min_parse}")
    passed = all(c["pass"] for c in checks)
    return {"passed": passed, "checks": checks}


def _fmt(open_m, oracle_m, v) -> str:
    L = ["# LensBench-VI gate (open-weight vs Claude oracle)", ""]
    keys = ["n", "parse_rate", "detection_auc", "mimic_auc", "gradeD_auc", "overall_auc",
            "recovery@0.01FPR", "mean_p_lens_A", "mean_cost_usd", "mean_wall_s",
            "n_pos", "n_random", "n_mimic"]
    L.append(f"| metric | open | oracle |")
    L.append("|---|---|---|")
    for k in keys:
        L.append(f"| {k} | {open_m.get(k)} | {oracle_m.get(k) if oracle_m else '—'} |")
    if v:
        L += ["", f"## Verdict: {'PASS ✅' if v['passed'] else 'FAIL ❌'}", ""]
        for c in v["checks"]:
            L.append(f"- {'✅' if c['pass'] else '❌'} **{c['check']}** — {c['detail']}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", dest="open_preds", help="open-weight grader preds parquet")
    ap.add_argument("--oracle", default=None, help="Claude-oracle preds parquet (for the verdict)")
    ap.add_argument("--manifest", default=str(config.OUT / "lensbench_manifest.csv"))
    ap.add_argument("--tol-auc", type=float, default=0.03,
                    help="max allowed detection-AUC shortfall vs the oracle")
    ap.add_argument("--min-parse", type=float, default=0.95)
    ap.add_argument("--report", default=None, help="write the markdown report here")
    # optional standalone quant-safety check
    ap.add_argument("--flip-a", default=None)
    ap.add_argument("--flip-b", default=None)
    args = ap.parse_args()

    if args.flip_a and args.flip_b:
        fr = grade_flip_rate(pd.read_parquet(args.flip_a), pd.read_parquet(args.flip_b))
        print("[flip-rate]", json.dumps(fr, indent=2))
        return

    if not args.open_preds:
        raise SystemExit("--open is required (or use --flip-a/--flip-b for the quant check)")
    manifest = pd.read_csv(args.manifest)
    open_m = gate_metrics(merge_manifest(pd.read_parquet(args.open_preds), manifest))
    oracle_m = (gate_metrics(merge_manifest(pd.read_parquet(args.oracle), manifest))
                if args.oracle else {})
    v = verdict(open_m, oracle_m, args.tol_auc, args.min_parse) if args.oracle else None
    rep = _fmt(open_m, oracle_m, v)
    print(rep)
    if args.report:
        Path(args.report).write_text(rep)
        print(f"\n[written] {args.report}")
    if v and not v["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
