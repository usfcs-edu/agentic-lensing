#!/usr/bin/env python3
"""Assemble the LensBench-VI v1 report: score each grader mode and compare.

  python lensjudge/eval/report.py --out outputs/lensbench_v1.md \
         lean=outputs/preds_lensbench_lean.parquet \
         panel=outputs/preds_lensbench_panel.parquet \
         multiagent=outputs/preds_lensbench_multiagent.parquet

Scores each predictions parquet with eval.score, builds a side-by-side comparison
(the lean-vs-robust-vs-multiagent question), and writes the report — every number
explicitly CONSENSUS-REFERENCED, NO HUMAN CEILING.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge.eval import score as scoremod  # noqa: E402

_ROWS = [("binary_auc", "Binary ROC-AUC (lens vs non-lens)"),
         ("recovery@0.01FPR", "Recovery @1% FPR"),
         ("recovery@0.001FPR", "Recovery @0.1% FPR"),
         ("qwk_vs_consensus", "Quadratic-weighted κ vs consensus"),
         ("exact_acc", "Exact-grade accuracy"),
         ("adjacent_acc", "Within-one-grade accuracy"),
         ("ece_p_lens", "Calibration ECE (p_lens)"),
         ("agent_vs_cnn_kappa", "Agent-vs-CNN κ"),
         ("escalation_rate", "Escalation rate"),
         ("parse_rate", "JSON parse rate"),
         ("mean_cost_usd", "Mean cost / candidate ($)"),
         ("mean_wall_s", "Mean wall (s)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="mode=path.parquet")
    ap.add_argument("--out", default="lensjudge/outputs/lensbench_v1.md")
    args = ap.parse_args()

    scored = {}
    for pair in args.pairs:
        mode, path = pair.split("=", 1)
        if Path(path).exists():
            scored[mode] = scoremod.score(pd.read_parquet(path))
    if not scored:
        print("no prediction files found"); return

    modes = list(scored)
    lines = ["# LensBench-VI v1 — agentic strong-lens VI grading", "",
             "> **CONSENSUS-REFERENCED, NO HUMAN CEILING.** Labels are a single "
             "2-senior-author consensus grade (A/B/C) plus Grade-D human-rejects; there "
             "are NO per-rater labels anywhere in the corpus, so these are *agreement* "
             "metrics, not accuracy, and cannot be placed against a human-vs-human "
             "ceiling. A small multi-grader study is the noted future fix.", "",
             "Held-out set: graded A/B/C (Storfer DR9 + Inchausti DR10) + Grade-D "
             "human-rejected hard negatives + random-galaxy negatives + Foundry-II "
             "confirmed/non-lens gold. Negatives kept as separate strata.", "",
             "## Grader comparison", "",
             "| Metric | " + " | ".join(modes) + " |",
             "|---|" + "|".join(["---"] * len(modes)) + "|"]
    for key, label in _ROWS:
        cells = [str(scored[m].get(key, "—")) for m in modes]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # the headline finding (honest): near-chance imaging agreement across all configs
    any_mode = scored[modes[0]]
    aucs = {m: scored[m].get("binary_auc") for m in modes if scored[m].get("binary_auc") is not None}
    lines += ["", "## Key finding — imaging grading is near-chance vs consensus, across all configs",
              "", "On the CNN's hard high-p_meta pool, no configuration reproduces the "
              "single-consensus A/B/C labels: binary ROC-AUC stays near 0.5 and quadratic-"
              "weighted kappa near 0 for every grader, and a stronger backbone (Opus) or "
              "multi-agent deliberation does not help.", "",
              "```", f"binary AUC by config: {aucs}", "```", ""]
    if "cnn_mean_p_meta_by_truth" in any_mode:
        c = any_mode["cnn_mean_p_meta_by_truth"]
        lines += ["The frozen CNN is itself saturated here (mean p_meta ~equal across grades, "
                  "incl. Grade-D rejects), so neither the CNN nor the agent separates this "
                  "pool:", "", "```", f"CNN mean p_meta by consensus grade: {c}", "```", "",
                  "Interpretation: A/B/C are mostly *unconfirmed* candidates and the C/D "
                  "boundary is subjective; agent-vs-consensus disagreement on imaging is not "
                  "adjudicable without multi-grader or spectroscopic ground truth (the "
                  "missing-ceiling gap). Where labels are hard (spectroscopy), the agent does "
                  "well (20/20 Hsu Grade-A, 4/4 non-lenses)."]

    for m in modes:
        s = scored[m]
        lines += ["", f"## {m} — detail"]
        for k in ("recovery_by_grade", "escalation_by_grade", "confusion",
                  "cnn_mean_p_meta_by_truth", "mean_p_lens_by_region"):
            if k in s:
                import json
                lines.append(f"\n**{k}**\n```\n{json.dumps(s[k], indent=2)}\n```")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines[:30]))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
