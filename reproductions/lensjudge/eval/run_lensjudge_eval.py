#!/usr/bin/env python3
"""eval/run_lensjudge_eval.py — LensJudge v2: honest held-out calibration + the regression gate.

The README's central finding is that the judge's hard number (lens vs Grade-D human-reject at
DECaLS 1.3" seeing) is resolution-limited (AUC ~0.5), while detection (lens vs random galaxy)
is strong. v2 must therefore measure DETECTION and GRADING *separately* and report recovery,
not agreement. This harness does exactly that on the frozen LensBench manifest, reusing
eval/score.py wholesale — the only new logic is partitioning the negative class by the
manifest's `source` tag:

  Benchmark A (DETECTION): positives = graded A/B/C   vs  negatives = random galaxies (source=random_neg)
  Benchmark B (GRADING)  : positives = graded A/B/C   vs  negatives = Grade-D human-rejects (source=graded_D)

It reports recovery@1%/0.1%FPR + AUC for each, and appends a one-line regression record keyed by
(label, mode, manifest_sha). The **$100 evidence gate**: a v2 config change (escalate / rubric /
exemplars) must beat the pinned v1-lean baseline on Benchmark-A recovery BEFORE any bulk grading.

SPENDS NOTHING by default — scores an existing preds parquet. `--grade` runs imaging/run_batch
on the frozen manifest first (the only $-spending path), and ABORTS unless a pre-run cost estimate
(rows x --est-per-cand) is <= --max-usd, so the budget cannot be blown by accident.

  # score an existing preds parquet (free):
  python eval/run_lensjudge_eval.py --preds outputs/preds_v1lean.parquet \
      --manifest outputs/lensbench_manifest.csv --label v1-lean --log outputs/eval_regression_log.jsonl
  # grade a small batch then score (gated by --max-usd):
  python eval/run_lensjudge_eval.py --grade --mode lean --model opus \
      --manifest outputs/lensbench_manifest.csv --max-usd 20 --label v1-lean
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.eval import score as S  # noqa: E402

POS_SRC = {"graded"}
NEG_SRC = {"A": "random_neg", "B": "graded_D"}   # benchmark -> negative source tag
HERE = Path(__file__).resolve().parent


def _manifest_sha(path: Path) -> str:
    shap = Path(str(path) + ".sha")
    if shap.exists():
        return shap.read_text().strip()
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _estimate_and_grade(args, manifest: Path) -> Path:
    """Run imaging/run_batch on the frozen manifest, but ONLY after a cost-estimate gate."""
    man = pd.read_csv(manifest)
    n = len(man)
    est = n * args.est_per_cand
    print(f"[gate] {n} rows x ${args.est_per_cand:.3f}/cand ~= ${est:.2f} "
          f"(cap ${args.max_usd:.2f})")
    if est > args.max_usd:
        raise SystemExit(f"[gate] ABORT: estimate ${est:.2f} > --max-usd ${args.max_usd:.2f}. "
                         f"Shrink the manifest or raise --max-usd (mind the $100 cap).")
    out = Path(args.out or (config.OUT / f"preds_{args.label}.parquet"))
    cmd = [sys.executable, str(HERE.parent / "imaging" / "run_batch.py"),
           "--manifest", str(manifest), "--mode", args.mode, "--out", str(out)]
    if args.model:
        cmd += ["--model", args.model]
    if args.rubric:
        cmd += ["--rubric", args.rubric]
    if args.trace_tag:
        cmd += ["--trace-tag", args.trace_tag]
    print(f"[grade] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out


def evaluate(preds: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    """Partition by manifest source -> Benchmark A/B -> score.score() each."""
    src = manifest[["name", "source", "binary_label"]].copy()
    src["name"] = src["name"].astype(str)
    p = preds.copy()
    p["name"] = p["name"].astype(str)
    p = p.merge(src, on="name", how="left", suffixes=("", "_man"))
    report = {"n_preds": len(p), "parse_rate": round(float(p["parse_ok"].mean()), 4),
              "mean_cost_usd": round(float(p["cost_usd"].mean()), 4),
              "total_cost_usd": round(float(p["cost_usd"].sum()), 4)}
    for bench, negsrc in NEG_SRC.items():
        sub = p[p["source"].isin(POS_SRC | {negsrc})].copy()
        if sub["source"].isin(POS_SRC).sum() == 0 or (sub["source"] == negsrc).sum() == 0:
            report[f"benchmark_{bench}"] = {"skipped": "missing pos or neg rows",
                                            "n_pos": int(sub["source"].isin(POS_SRC).sum()),
                                            "n_neg": int((sub["source"] == negsrc).sum())}
            continue
        s = S.score(sub)
        report[f"benchmark_{bench}"] = {
            "name": ("detection: lens-vs-random" if bench == "A" else "grading: lens-vs-hardreject"),
            "n_pos": int(sub["source"].isin(POS_SRC).sum()),
            "n_neg": int((sub["source"] == negsrc).sum()),
            "binary_auc": s.get("binary_auc"),
            "recovery@0.01FPR": s.get("recovery@0.01FPR"),
            "recovery@0.001FPR": s.get("recovery@0.001FPR"),
            "recovery_by_grade": s.get("recovery_by_grade"),
            "qwk_vs_consensus": s.get("qwk_vs_consensus"),
            "ece_p_lens": s.get("ece_p_lens"),
            "escalation_rate": s.get("escalation_rate"),
        }
    return report


def regression_check(report: dict, baseline: dict, tol: float, min_parse: float) -> tuple[bool, list]:
    msgs = []
    ok = True
    if report["parse_rate"] < min_parse:
        ok = False; msgs.append(f"parse_rate {report['parse_rate']} < {min_parse}")
    bA = report.get("benchmark_A", {}).get("recovery@0.01FPR")
    base_bA = baseline.get("benchmark_A", {}).get("recovery@0.01FPR")
    if bA is not None and base_bA is not None:
        if bA < base_bA - tol:
            ok = False
            msgs.append(f"Benchmark-A recovery@0.01FPR {bA} < baseline {base_bA} - {tol}")
        else:
            msgs.append(f"Benchmark-A recovery@0.01FPR {bA} vs baseline {base_bA} (OK, tol {tol})")
    return ok, msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None, help="existing preds parquet to score (free)")
    ap.add_argument("--manifest", required=True, help="frozen LensBench manifest CSV (has `source`)")
    ap.add_argument("--label", required=True, help="config name, e.g. v1-lean / v2-escalate")
    # --grade path (the only $-spending one)
    ap.add_argument("--grade", action="store_true", help="run run_batch on the manifest first")
    ap.add_argument("--mode", default="lean")
    ap.add_argument("--model", default=None)
    ap.add_argument("--rubric", default=None)
    ap.add_argument("--trace-tag", default=None)
    ap.add_argument("--max-usd", type=float, default=20.0, help="hard cap for the --grade estimate")
    ap.add_argument("--est-per-cand", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    # regression gate
    ap.add_argument("--baseline", default=None, help="baseline report JSON to regress against")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--min-parse", type=float, default=0.97)
    ap.add_argument("--check-regression", action="store_true", help="nonzero exit on regression")
    ap.add_argument("--report", default=None)
    ap.add_argument("--log", default=str(config.OUT / "eval_regression_log.jsonl"))
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if args.grade:
        preds_path = _estimate_and_grade(args, manifest)
    else:
        if not args.preds:
            raise SystemExit("provide --preds <parquet> or --grade")
        preds_path = Path(args.preds)
    preds = pd.read_parquet(preds_path)
    man = pd.read_csv(manifest)

    report = evaluate(preds, man)
    report.update({"label": args.label, "mode": args.mode,
                   "manifest_sha": _manifest_sha(manifest),
                   "preds": str(preds_path), "ts": int(time.time())})

    # human-readable
    print(f"\n=== LensJudge v2 eval — {args.label} (mode={args.mode}) ===")
    print(f"n={report['n_preds']}  parse_rate={report['parse_rate']}  "
          f"total_cost=${report['total_cost_usd']}  manifest_sha={report['manifest_sha']}")
    for b in ("A", "B"):
        d = report.get(f"benchmark_{b}", {})
        if "skipped" in d:
            print(f"  Benchmark {b}: SKIPPED ({d['skipped']}; n_pos={d['n_pos']} n_neg={d['n_neg']})")
            continue
        print(f"  Benchmark {b} [{d['name']}] n_pos={d['n_pos']} n_neg={d['n_neg']}: "
              f"AUC={d['binary_auc']} recovery@1%FPR={d['recovery@0.01FPR']} "
              f"@0.1%FPR={d['recovery@0.001FPR']} escalation={d.get('escalation_rate')}")

    # regression gate
    if args.baseline and Path(args.baseline).exists():
        base = json.loads(Path(args.baseline).read_text())
        ok, msgs = regression_check(report, base, args.tol, args.min_parse)
        report["regression_ok"] = ok
        print(f"  regression vs {Path(args.baseline).name}: {'PASS' if ok else 'FAIL'}")
        for m in msgs:
            print(f"    - {m}")
    # append regression log
    logp = Path(args.log)
    logp.parent.mkdir(parents=True, exist_ok=True)
    with logp.open("a") as f:
        f.write(json.dumps(report, default=float) + "\n")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=float))
        print(f"[written] {args.report}")
    print(f"[logged] {logp}")

    if args.check_regression and report.get("regression_ok") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
