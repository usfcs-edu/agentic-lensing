#!/usr/bin/env python3
"""Grade a batch of candidates with bounded concurrency; resumable -> predictions parquet.

  python lensjudge/imaging/run_batch.py --which both --sample-per-grade 4 \
         --include-grade-d 4 --concurrency 6 --out outputs/preds_smoke.parquet

Reads graded A/B/C candidates (+ optional Grade-D human-reject negatives) and grades
each with the lean single-agent grader, writing one row per candidate. Re-running
skips names already present in the output (resumable). Prints a confusion summary.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# bootstrap: put reproductions/ on the path so `import lensjudge` works when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import io  # noqa: E402
from lensjudge.imaging import grader_lean  # noqa: E402


def _grader(mode):
    if mode == "panel":
        from lensjudge.imaging import judges
        return judges
    if mode == "multiagent":
        from lensjudge.imaging import orchestrator
        return orchestrator
    if mode == "escalate":
        from lensjudge.imaging import grader_escalate
        return grader_escalate
    if mode == "direct":
        from lensjudge.imaging import grader_direct
        return grader_direct
    return grader_lean


def _row_dict(g, cand):
    base = {"name": cand["name"], "grade_truth": cand.get("grade"),
            "catalog": cand.get("catalog"), "region": cand.get("region"),
            "p_meta": cand.get("p_meta"), "parse_ok": g.parse_ok,
            "cost_usd": g.cost_usd, "turns": g.num_turns,
            "wall_s": g.meta.get("wall_s"), "error": g.error,
            "n_thinking_blocks": g.meta.get("n_thinking_blocks"),
            "thinking_chars": g.meta.get("thinking_chars"),
            # escalate-mode provenance (None for non-escalate modes)
            "tier": g.meta.get("tier"), "escalated": g.meta.get("escalated"),
            "highres_survey": g.meta.get("highres_survey"),
            "p_lens_tier1": g.meta.get("p_lens_tier1"),
            "p_lens_tier2": g.meta.get("p_lens_tier2")}
    if g.grade is not None:
        c = g.grade.criteria.model_dump()
        base.update({"grade_pred": g.grade.grade, "p_lens": g.grade.p_lens,
                     "confidence": g.grade.confidence, "contaminant": g.grade.contaminant,
                     "escalate": g.grade.escalate_to_human, "rationale": g.grade.rationale,
                     **{f"crit_{k}": v for k, v in c.items()}})
    else:
        base.update({"grade_pred": None, "p_lens": None, "confidence": None,
                     "rationale": (g.raw or "")[:500]})
    return base


def build_set(args) -> pd.DataFrame:
    if args.manifest:
        df = pd.read_csv(args.manifest).rename(columns={"grade_truth": "grade"})
        if "catalog" not in df.columns:
            df["catalog"] = df.get("survey_key", "storfer")
        return df
    cand = io.load_candidates(args.which)
    cand["survey_key"] = cand["catalog"]
    if args.sample_per_grade:
        # sample per grade WITHOUT groupby.apply (pandas 2.x drops the grouping column)
        cand = pd.concat(
            [g.sample(min(len(g), args.sample_per_grade), random_state=2026)
             for _, g in cand.groupby("grade")],
            ignore_index=True)
    parts = [cand]
    if args.include_grade_d:
        gd = io.load_grade_d(args.which)
        gd = gd.sample(min(len(gd), args.include_grade_d), random_state=2026)
        gd["survey_key"] = gd["catalog"]
        gd["region"] = gd["survey"]
        gd["tractor_type"] = "?"
        parts.append(gd)
    return pd.concat(parts, ignore_index=True)


async def run(df: pd.DataFrame, out: Path, concurrency: int, model: str | None, mode: str,
              modelability: bool = False, representations: bool = False,
              trace_tag: str | None = None, rubric: str | None = None):
    grader = _grader(mode)
    done = set()
    if out.exists():
        done = set(pd.read_parquet(out)["name"].tolist())
    todo = [r.to_dict() for _, r in df.iterrows() if r["name"] not in done]
    flags = ("+modelability" if modelability else "") + ("+representations" if representations else "")
    print(f"[batch] mode={mode}{flags}: "
          f"{len(todo)} to grade ({len(done)} already done) -> {out}")
    sem = asyncio.Semaphore(concurrency)
    rows, lock = [], asyncio.Lock()
    trace_dir = config.OUT / (f"traces_{mode}_{trace_tag}" if trace_tag else f"traces_{mode}")
    trace_dir.mkdir(parents=True, exist_ok=True)
    # extra tools for the lean/panel grader: Foundry-I GIGA-Lens fit and/or the
    # engineered-representation feature+view tool.
    extra = {}
    if mode in ("lean", "panel", "escalate"):
        tools = ["fetch_cutout", "get_photometry"]
        if modelability:
            tools.append("quick_lensmodel")
        if representations:
            tools.append("lens_representations")
        if len(tools) > 2:
            extra["tools"] = tuple(tools)
    if rubric:
        if mode not in ("lean", "escalate", "direct"):
            raise SystemExit("--rubric is only supported with --mode lean, escalate, or direct")
        extra["system_prompt"] = Path(rubric).read_text()

    async def one(cand):
        async with sem:
            res = await grader.grade_candidate(
                cand, model=model, trace_path=str(trace_dir / f"{cand['name']}.jsonl"),
                **extra)
        async with lock:
            rows.append(_row_dict(res, cand))
            if len(rows) % 5 == 0 or len(rows) == len(todo):
                _flush(rows, out, done)

    await asyncio.gather(*(one(cand) for cand in todo))
    _flush(rows, out, done)
    return out


def _flush(rows, out, done):
    new = pd.DataFrame(rows)
    if out.exists():
        prev = pd.read_parquet(out)
        new = pd.concat([prev, new[~new["name"].isin(prev["name"])]], ignore_index=True)
    new.drop_duplicates("name", keep="last").to_parquet(out, index=False)


def summarize(out: Path):
    df = pd.read_parquet(out)
    ok = df[df["parse_ok"]]
    print(f"\n[summary] {len(df)} graded | parse_ok {len(ok)}/{len(df)} "
          f"({100*len(ok)/max(1,len(df)):.0f}%) | mean cost ${df['cost_usd'].mean():.3f} "
          f"| mean wall {df['wall_s'].mean():.1f}s")
    if len(ok):
        ct = pd.crosstab(ok["grade_truth"], ok["grade_pred"])
        print("\nconfusion (rows=consensus truth, cols=agent grade):")
        print(ct)
        # binary lens(A/B/C) vs non-lens(D) view of p_lens
        ok = ok.copy(); ok["is_lens"] = ok["grade_truth"].isin(["A", "B", "C"]).astype(int)
        if ok["p_lens"].notna().any() and ok["is_lens"].nunique() > 1:
            for g in ["A", "B", "C", "D"]:
                sub = ok[ok["grade_truth"] == g]
                if len(sub):
                    print(f"  truth {g}: n={len(sub)} mean p_lens={sub['p_lens'].mean():.2f} "
                          f"escalate={sub['escalate'].mean():.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("lean", "panel", "multiagent", "escalate", "direct"),
                    default="lean")
    ap.add_argument("--modelability", action="store_true",
                    help="give the lean/panel grader the Foundry-I quick_lensmodel tool")
    ap.add_argument("--representations", action="store_true",
                    help="give the lean/panel grader the lens_representations tool")
    ap.add_argument("--manifest", default=None, help="frozen LensBench manifest CSV (overrides sampling)")
    ap.add_argument("--which", choices=("storfer", "inchausti", "both"), default="both")
    ap.add_argument("--sample-per-grade", type=int, default=0, help="0 = all graded A/B/C")
    ap.add_argument("--include-grade-d", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=str(config.OUT / "preds.parquet"))
    ap.add_argument("--thinking", choices=("off", "adaptive"), default=None)
    ap.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default=None)
    ap.add_argument("--trace-tag", default=None, help="trace dir suffix: traces_{mode}_{tag}")
    ap.add_argument("--rubric", default=None,
                    help="path to an alternate system-prompt rubric (lean mode only)")
    args = ap.parse_args()
    import os
    if args.thinking:
        os.environ["LENSJUDGE_THINKING"] = args.thinking
    if args.effort:
        os.environ["LENSJUDGE_EFFORT"] = args.effort
    out = Path(args.out)
    df = build_set(args)
    print(f"[set] {len(df)} candidates: {df['grade'].value_counts().to_dict()}")
    asyncio.run(run(df, out, args.concurrency, args.model, args.mode, args.modelability,
                    args.representations, args.trace_tag, args.rubric))
    summarize(out)


if __name__ == "__main__":
    main()
