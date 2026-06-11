#!/usr/bin/env python3
"""Run LensJudge on the Euclid Q1 high-res strong-lens benchmark.

Tests the central LensJudge question with BETTER DATA (resolution, not labels): when the
arcs are resolved at Euclid's 0.1"/px (~13x finer than DESI's 1.3" seeing) and the labels
come from ~10 independent expert votes, does the vision grader's p_lens recover the lens
grade -- i.e. does it beat the ~0.5 / p_lens~0.02 wall it hits on the hard DESI pool?

Two experiments:
  --mode rank   : grade a stratified sample of the 539 Euclid cutouts; report
                  Spearman(p_lens, expert_score) and ROC-AUC for expert grade A vs C.
  --mode paired : for the DESI candidates that fall in Euclid Q1, grade the SAME object at
                  BOTH DESI 1.3" and Euclid 0.1"; compare p_lens (within-object control).

  python lensjudge/eval/run_euclid.py --mode rank   --n 90 --concurrency 6
  python lensjudge/eval/run_euclid.py --mode paired --concurrency 6
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from claude_agent_sdk import ClaudeAgentOptions  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import parse  # noqa: E402
from lensjudge.common.schemas import ImageGrade  # noqa: E402
from lensjudge.common.euclid import obj_dir, SUBSETS, EUCLID_ROOT  # noqa: E402
from lensjudge.imaging.grader_lean import _collect, grade_candidate  # noqa: E402
from lensjudge.tools import server  # noqa: E402

RUBRIC = (config.HERE / "prompts" / "rubric_imaging.md").read_text()
CAT = EUCLID_ROOT / "raw" / "q1_discovery_engine_lens_catalog.csv"

EUCLID_NOTE = """

# IMPORTANT — these are EUCLID Q1 cutouts, not DESI grz
Call **fetch_euclid_cutout(id_str=...)** (NOT fetch_cutout) to see the images.
- Resolution is 0.1"/px, ~13x sharper than DESI's ~1.3" ground seeing: tangential arcs and
  Einstein rings that are blurred into the lens-galaxy light at ground resolution are cleanly
  resolved here. Judge the morphology you actually see.
- Bands: VIS (sharp broad-optical luminance) + NIR Y/J/H. In color views the old red lens
  galaxy is red/orange and a lensed background source is blue. The 'vis', 'vis_zoom' and
  'vis_sub' (lens-light-subtracted) views are where thin arcs/rings show best.
- Apply the SAME 5-criterion rubric and the SAME A/B/C/D scale. There are no CNN scores here.
"""
EUCLID_SYS = RUBRIC + EUCLID_NOTE


def build_manifest() -> pd.DataFrame:
    cat = pd.read_csv(CAT).set_index("id_str")
    rows = []
    for sub in SUBSETS:
        d = EUCLID_ROOT / sub
        if not d.exists():
            continue
        for idd in sorted(p.name for p in d.iterdir() if p.is_dir()):
            if idd not in cat.index:
                continue
            r = cat.loc[idd]
            rows.append({"id_str": idd, "subset": sub, "grade": r["grade"],
                         "expert_score": float(r["expert_score"]),
                         "votes": int(r["expert_total_votes"]),
                         "ra": float(r["right_ascension"]), "dec": float(r["declination"])})
    return pd.DataFrame(rows)


def _euclid_user_message(obj: dict) -> str:
    return (f"Grade this strong-lens candidate. id_str={obj['id_str']!r} (Euclid Q1).\n"
            f"Call fetch_euclid_cutout with that id_str, inspect the views, then respond "
            f"with ONLY the ImageGrade JSON object.")


async def grade_euclid(obj: dict, model: str | None = None) -> dict:
    mcp_servers, allowed = server.build(["fetch_euclid_cutout"])
    opts = ClaudeAgentOptions(
        model=model or config.MODELS["grader"],
        system_prompt=EUCLID_SYS,
        mcp_servers=mcp_servers, allowed_tools=allowed,
        permission_mode="bypassPermissions",
        max_turns=config.MAX_TURNS, max_budget_usd=config.MAX_BUDGET_USD,
        setting_sources=None,
    )
    t0 = time.time()
    try:
        raw, cost, turns, _ = await _collect(_euclid_user_message(obj), opts)
    except Exception as e:
        return {**obj, "error": f"{type(e).__name__}: {e}", "p_lens": np.nan,
                "agent_grade": None, "cost_usd": 0.0}
    g = parse.parse_model(raw, ImageGrade)
    return {**obj, "p_lens": (g.p_lens if g else np.nan),
            "agent_grade": (g.grade if g else None),
            "confidence": (g.confidence if g else np.nan),
            "contaminant": (g.contaminant if g else None),
            "cost_usd": cost, "wall_s": round(time.time() - t0, 1),
            "rationale": (g.rationale[:300] if g else raw[:200])}


async def _bounded(coro_fns, concurrency):
    sem = asyncio.Semaphore(concurrency)
    async def run(fn):
        async with sem:
            return await fn()
    return await asyncio.gather(*[run(f) for f in coro_fns])


def _auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, np.r_[pos, neg])


def score_rank(df: pd.DataFrame):
    from scipy.stats import spearmanr
    ok = df[df.p_lens.notna()]
    print(f"\n=== RANK eval: {len(ok)}/{len(df)} graded ok ===")
    print("agent grade dist:", ok.agent_grade.value_counts().to_dict())
    print("mean p_lens by EXPERT grade:")
    print(ok.groupby("grade").p_lens.agg(["count", "mean", "median"]).to_string())
    rho, p = spearmanr(ok.expert_score, ok.p_lens)
    print(f"\nSpearman(p_lens, expert_score) = {rho:.3f}  (p={p:.1e}, n={len(ok)})")
    for hi, lo, name in [("A", "C", "A vs C"), ("A", "B", "A vs B"),
                         (("A", "B"), "C", "A/B vs C")]:
        pos = ok[ok.grade.isin([hi] if isinstance(hi, str) else list(hi))].p_lens
        neg = ok[ok.grade == lo].p_lens
        print(f"ROC-AUC p_lens, expert {name}: {_auc(pos, neg):.3f}  "
              f"({len(pos)} vs {len(neg)})")
    # agent grade vs expert grade agreement
    g = ok.dropna(subset=["agent_grade"])
    if len(g):
        print("\nconfusion (rows=expert grade, cols=agent grade):")
        print(pd.crosstab(g.grade, g.agent_grade).to_string())


async def run_rank(args):
    man = build_manifest()
    print(f"manifest: {len(man)} Euclid objects with FITS; "
          f"grade {man.grade.value_counts().to_dict()}")
    # stratified sample: keep ALL grade C (rare), sample A/B
    keep = [man[man.grade == "C"]]
    for g in ["A", "B"]:
        sub = man[man.grade == g]
        n = min(len(sub), max(1, (args.n - (man.grade == "C").sum()) // 2))
        keep.append(sub.sample(n, random_state=0))
    samp = pd.concat(keep).reset_index(drop=True)
    print(f"grading {len(samp)} objects ({samp.grade.value_counts().to_dict()}) "
          f"model={config.MODELS['grader']}")
    recs = await _bounded([lambda o=o: grade_euclid(o, args.model)
                           for o in samp.to_dict("records")], args.concurrency)
    df = pd.DataFrame(recs)
    out = config.OUT / "euclid_rank_preds.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {out}  (total ${df.cost_usd.sum():.2f})")
    score_rank(df)


async def run_paired(args):
    xm = pd.read_csv(config.OUT / "xmatch_euclid_q1.csv")
    have = set()
    for sub in SUBSETS:
        d = EUCLID_ROOT / sub
        if d.exists():
            have |= {p.name for p in d.iterdir() if p.is_dir()}
    xm = xm[xm.euclid_id.isin(have)].reset_index(drop=True)
    print(f"paired: {len(xm)} matched DESI<->Euclid candidates with local Euclid FITS")
    surv = {"storfer": "storfer", "inchausti": "inchausti",
            "huang2021": "ls-dr9", "huang2020": "ls-dr9"}

    async def one(r):
        eobj = {"id_str": r["euclid_id"], "grade": r["euclid_grade"],
                "expert_score": r["euclid_score"], "name": r["name"]}
        e = await grade_euclid(eobj, args.model)
        cand = {"name": r["name"], "survey_key": surv.get(r["source"], "ls-dr9"),
                "ra": r["ra"], "dec": r["dec"]}
        gd = await grade_candidate(cand, model=args.model)
        return {"name": r["name"], "source": r["source"],
                "desi_grade": r["grade"], "euclid_grade": r["euclid_grade"],
                "euclid_score": r["euclid_score"],
                "p_lens_desi": (gd.grade.p_lens if gd.grade else np.nan),
                "agent_grade_desi": (gd.grade.grade if gd.grade else None),
                "p_lens_euclid": e["p_lens"], "agent_grade_euclid": e["agent_grade"],
                "cost_usd": (gd.cost_usd + e["cost_usd"])}

    recs = await _bounded([lambda r=r: one(r) for _, r in xm.iterrows()], args.concurrency)
    df = pd.DataFrame(recs)
    out = config.OUT / "euclid_paired_preds.parquet"
    df.to_parquet(out, index=False)
    print(f"\nsaved {out}  (total ${df.cost_usd.sum():.2f})\n")
    cols = ["name", "desi_grade", "euclid_grade", "p_lens_desi", "p_lens_euclid",
            "agent_grade_desi", "agent_grade_euclid"]
    print(df[cols].to_string(index=False))
    d = df.dropna(subset=["p_lens_desi", "p_lens_euclid"])
    if len(d):
        print(f"\nmean p_lens: DESI 1.3\" = {d.p_lens_desi.mean():.3f}   "
              f"Euclid 0.1\" = {d.p_lens_euclid.mean():.3f}")
        print(f"p_lens increased on Euclid for {(d.p_lens_euclid > d.p_lens_desi).sum()}/{len(d)} objects")
        cflip = d[d.desi_grade.str.upper() == "C"]
        if len(cflip):
            print(f"DESI grade-C subset: mean p_lens DESI={cflip.p_lens_desi.mean():.3f} "
                  f"-> Euclid={cflip.p_lens_euclid.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rank", "paired"], default="rank")
    ap.add_argument("--n", type=int, default=90, help="rank mode: total objects to grade")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--model", default=None, help="override grader model")
    args = ap.parse_args()
    asyncio.run(run_rank(args) if args.mode == "rank" else run_paired(args))


if __name__ == "__main__":
    main()
