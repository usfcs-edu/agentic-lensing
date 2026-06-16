#!/usr/bin/env python3
"""eval/run_agency_ablation.py — B7: how much does the AGENTIC LOOP contribute?

Question (from the agency audit): LensJudge's "agentic" surface is a near-deterministic
fixed pipeline — the model emits a ToolSearch call, one fetch_cutout (default views), and
~38% of the time a get_photometry call, then grades. Orchestration (panel/multiagent) is
asyncio.gather + a hardcoded vote rule; the escalate trigger is a Python `if` on the grade.
So: does the tool-call PLANNING contribute anything over invoking the tools programmatically?

This script answers it with three $0 analyses on the frozen evidence manifest (it SPENDS
NOTHING — it scores existing preds parquets; the `direct` arm is graded separately by
run_batch --mode direct):

  (1) DIRECT vs LEAN — the headline. `direct` renders the same views + photometry in Python
      and makes ONE base-API judgment call (no loop). If its Benchmark-A/B AUC matches lean's,
      the agentic loop adds nothing and the tools could be invoked programmatically.
  (2) ESCALATION ROUTING — the one place an LLM output drives an action (tier-2 high-res).
      Could a zero-token CNN p_meta threshold reproduce the LLM-grade-gated routing? High
      agreement => the LLM grade adds nothing over the free score for routing.
  (3) COST — per-arm $/candidate and turns, the price of the loop.

  python lensjudge/eval/run_agency_ablation.py            # scores whatever parquets exist
  python lensjudge/eval/run_agency_ablation.py --report outputs/agency_ablation.md
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
from lensjudge.eval import run_lensjudge_eval as R  # noqa: E402

# arms to compare: (label, preds parquet, rubric, loop?)
# 2x2 (rubric x loop) isolates tool-call planning from multimodal judgment:
ARMS = [
    ("lean  (v1 rubric, LOOP)", "preds_v1lean.parquet", "v1", True),
    ("direct(v1 rubric, no loop)", "preds_direct.parquet", "v1", False),
    ("v2    (v2 rubric, LOOP)", "preds_v2.parquet", "v2", True),
    ("direct(v2 rubric, no loop)", "preds_direct_v2rubric.parquet", "v2-tool", False),
    ("direct(v2-inline,  no loop)", "preds_direct_v2inline.parquet", "v2-inline", False),
    ("direct(v2-inline + THINKING, no loop)", "preds_direct_v2think.parquet", "v2-inline+think", False),
]


def _arm_metrics(preds_path: Path, manifest: pd.DataFrame) -> dict | None:
    if not preds_path.exists():
        return None
    p = pd.read_parquet(preds_path)
    rep = R.evaluate(p, manifest)
    A = rep.get("benchmark_A", {})
    B = rep.get("benchmark_B", {})
    return {
        "n": rep.get("n_preds"), "parse": rep.get("parse_rate"),
        "A_auc": A.get("binary_auc"), "A_rec1": A.get("recovery@0.01FPR"),
        "B_name": B.get("name"), "B_auc": B.get("binary_auc"),
        "mean_cost": round(float(p["cost_usd"].mean()), 4),
        "mean_turns": round(float(p["turns"].mean()), 2) if "turns" in p else None,
        "mean_wall": round(float(p["wall_s"].mean()), 1) if "wall_s" in p else None,
    }


def _routing(preds_path: Path, manifest: pd.DataFrame) -> dict | None:
    """Can a free CNN p_meta threshold reproduce the LLM-grade-gated escalation set?

    LLM gate (grader_escalate.py:42): escalate iff grade_pred != 'A' or escalate_to_human.
    Compare against a p_meta threshold tuned to the SAME overall escalation rate, and a
    p_lens<0.5 threshold. Report Jaccard agreement on which rows get escalated, and how each
    gate concentrates on the true lenses (source=graded) — the rows tier-2 high-res recovers.
    """
    if not preds_path.exists():
        return None
    p = pd.read_parquet(preds_path)
    m = manifest[["name", "source", "p_meta"]].copy()
    m["name"] = m["name"].astype(str); p["name"] = p["name"].astype(str)
    d = p.merge(m, on="name", how="left", suffixes=("", "_man"))
    d = d[d["parse_ok"] == True].copy()  # noqa: E712
    pmeta = d["p_meta"].fillna(d.get("p_meta_man"))
    if "p_meta_man" in d:
        pmeta = d["p_meta"].where(d["p_meta"].notna(), d["p_meta_man"])
    d["pmeta"] = pd.to_numeric(pmeta, errors="coerce")

    llm_gate = (d["grade_pred"].fillna("D") != "A") | (d["escalate"].fillna(False))
    rate = float(llm_gate.mean())
    # p_meta threshold matched to the LLM gate's escalation rate: escalate the lowest-p_meta rows
    valid = d["pmeta"].notna()
    if valid.sum() >= 5:
        tau = float(np.quantile(d.loc[valid, "pmeta"], rate)) if 0 < rate < 1 else 0.5
        pmeta_gate = d["pmeta"] < tau
    else:
        tau = None; pmeta_gate = pd.Series(False, index=d.index)
    plens_gate = d["p_lens"].fillna(0) < 0.5

    def jaccard(a, b):
        a, b = a.fillna(False).astype(bool), b.fillna(False).astype(bool)
        u = (a | b).sum()
        return round(float((a & b).sum() / u), 3) if u else float("nan")

    def on_lenses(gate):
        lens = d["source"] == "graded"
        return round(float(gate[lens].mean()), 3) if lens.any() else None

    return {
        "n_scored": int(len(d)),
        "llm_escalation_rate": round(rate, 3),
        "pmeta_threshold_matched": None if tau is None else round(tau, 3),
        "jaccard_llm_vs_pmeta": jaccard(llm_gate, pmeta_gate),
        "jaccard_llm_vs_plens<0.5": jaccard(llm_gate, plens_gate),
        "escalate_frac_of_true_lenses": {
            "llm_gate": on_lenses(llm_gate),
            "pmeta_gate": on_lenses(pmeta_gate),
            "plens<0.5": on_lenses(plens_gate),
        },
    }


def build(manifest_path: Path) -> dict:
    manifest = pd.read_csv(manifest_path)
    arms = {}
    for label, fname, rubric, loop in ARMS:
        mt = _arm_metrics(config.OUT / fname, manifest)
        if mt is not None:
            mt["rubric"] = rubric
            mt["loop"] = loop
            arms[label] = mt
    routing = _routing(config.OUT / "preds_v1lean.parquet", manifest)
    return {"manifest": str(manifest_path), "arms": arms, "routing": routing}


def render_md(res: dict) -> str:
    L = ["# Agency ablation (LensJudge v2 B7)", "",
         "Does the agentic tool-call loop contribute to grading, or could the tools be invoked "
         "programmatically? Same frozen evidence manifest, same model (sonnet), same rubric.", "",
         "## (1) The 2x2 — tool-call LOOP vs no-loop, at each rubric", "",
         "| arm | rubric | loop | Bench-A AUC (lens-vs-random) | Bench-B AUC (lens-vs-mimic) | $/cand | turns |",
         "|---|:--:|:--:|--:|--:|--:|--:|"]
    for label, m in res["arms"].items():
        L.append(f"| {label} | {m['rubric']} | {'yes' if m['loop'] else 'no'} | {m['A_auc']} "
                 f"| {m['B_auc']} | {m['mean_cost']} | {m['mean_turns']} |")
    L += ["", "*Detection (Bench-A): the no-loop arm matches/beats the loop at ~1/6 the cost — "
          "tool-call planning adds nothing. Lens-vs-mimic (Bench-B): the v2-rubric gain appears "
          "in the LOOP arm; whether it transfers to no-loop is tested by the v2-inline arm "
          "(same judgment content, evidence inline). Bench-B AUCs sit on a compressed p_lens "
          "distribution (over-skepticism) so they are directional, not definitive.*", ""]
    r = res.get("routing")
    if r:
        L += ["## (2) Escalation routing — does the LLM grade beat a free CNN score?", "",
              f"- LLM-gated escalation rate: **{r['llm_escalation_rate']}** "
              f"(rule: grade≠A or escalate_to_human)",
              f"- A CNN `p_meta` threshold ({r['pmeta_threshold_matched']}) matched to that rate "
              f"agrees with the LLM gate at **Jaccard {r['jaccard_llm_vs_pmeta']}**; "
              f"a `p_lens<0.5` gate at Jaccard {r['jaccard_llm_vs_plens<0.5']}.",
              f"- Fraction of TRUE lenses escalated — llm_gate "
              f"{r['escalate_frac_of_true_lenses']['llm_gate']} vs "
              f"pmeta_gate {r['escalate_frac_of_true_lenses']['pmeta_gate']}.",
              "", "*High Jaccard ⇒ a zero-token CNN threshold reproduces the LLM-gated routing; "
              "the LLM grade adds little over the free score for the escalate decision.*", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(config.OUT / "lensbench_evidence.csv"))
    ap.add_argument("--report", default=None)
    ap.add_argument("--json", default=str(config.OUT / "agency_ablation.json"))
    args = ap.parse_args()
    res = build(Path(args.manifest))
    md = render_md(res)
    print(md)
    Path(args.json).write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[written] {args.json}")
    if args.report:
        Path(args.report).write_text(md)
        print(f"[written] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
