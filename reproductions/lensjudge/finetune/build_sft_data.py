#!/usr/bin/env python3
"""finetune/build_sft_data.py — assemble a vision-SFT set to distill/fine-tune the open grader.

Phase 4 of LensJudge v4: off-the-shelf Qwen3-VL fails the detection gate (best 0.56 vs Claude 0.66),
so we fine-tune the student to grade well. This builds a LEAKAGE-FREE, BALANCED SFT corpus disjoint
from the 270-row LensBench-VI eval manifest:

  POSITIVES  human-consensus Storfer/Inchausti A/B/C lenses (the ground truth), grade-mapped p_lens.
  NEGATIVES  random-galaxy negatives + CNN-high lens MIMICS + the Claude-graded claudenet_737 campaign
             rejects (true teacher distillation for the hard-negative frontier).

Each example renders the SAME views grader_direct feeds at inference (full/zoom/residual + photometry),
and the target is one ImageGrade JSON. Output is ms-swift / messages JSONL + PNGs under outputs/sft/.

  python -m lensjudge.finetune.build_sft_data --max-pos 900 --max-neg 900 --val-frac 0.1
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
from lensjudge.common import fetch, io, render  # noqa: E402
from lensjudge.common.schemas import ImageGrade  # noqa: E402
from lensjudge.imaging.grader_direct import _candidate_text, _DEFAULT_VIEWS  # noqa: E402
from lensjudge.imaging.grader_lean import _RUBRIC  # noqa: E402

SEED = 2026
# grade -> target p_lens (well-separated so the student's p_lens drives a high detection AUC)
P_BY_GRADE = {"A": 0.95, "B": 0.85, "C": 0.65, "D": 0.05}
# grade -> heuristic criteria (0-10) so targets are non-degenerate (criteria are not gated, but
# all-zero targets would teach the model to ignore them)
CRIT_BY_GRADE = {
    "A": dict(blue_source=8, low_surface_brightness=7, curvature=8, counter_images=5, arc_morphology=8),
    "B": dict(blue_source=6, low_surface_brightness=6, curvature=6, counter_images=3, arc_morphology=6),
    "C": dict(blue_source=4, low_surface_brightness=4, curvature=4, counter_images=2, arc_morphology=4),
    "D": dict(blue_source=1, low_surface_brightness=1, curvature=0, counter_images=0, arc_morphology=1),
}


def _excluded_names() -> set:
    man = config.OUT / "lensbench_manifest.csv"
    if man.exists():
        return set(pd.read_csv(man)["name"].astype(str))
    return set()


def _target_json(grade: str, contaminant=None, rationale: str = "", crit=None, p_lens=None,
                 confidence: float = 0.7) -> str:
    if not (isinstance(contaminant, str) and contaminant.strip()):
        contaminant = None        # coerce NaN / float / empty -> None for the pydantic validator
    if not isinstance(rationale, str):
        rationale = ""
    g = ImageGrade(
        grade=grade,
        criteria=(crit or CRIT_BY_GRADE.get(grade, CRIT_BY_GRADE["D"])),
        p_lens=(P_BY_GRADE[grade] if p_lens is None else float(p_lens)),
        confidence=confidence,
        contaminant=contaminant,
        escalate_to_human=False,
        rationale=rationale,
    )
    return g.model_dump_json()


def _rows_from_human(excl: set, max_pos: int) -> list[dict]:
    """Human-graded A/B/C lenses (ground-truth positives), stratified by grade, disjoint from eval."""
    cand = io.load_candidates("both")
    cand = cand[~cand["name"].astype(str).isin(excl)]
    cand = cand[cand["grade"].isin(["A", "B", "C"])]
    parts = []
    for g, sub in cand.groupby("grade"):
        parts.append(sub.sample(min(len(sub), max_pos), random_state=SEED))
    pool = pd.concat(parts, ignore_index=True)
    if len(pool) > max_pos:
        pool = pool.sample(max_pos, random_state=SEED).reset_index(drop=True)
    out = []
    for _, r in pool.iterrows():
        g = r["grade"]
        out.append(dict(name=str(r["name"]), ra=r.get("ra"), dec=r.get("dec"),
                        survey_key=r["catalog"], label="lens",
                        target=_target_json(g, rationale=f"Human-consensus grade-{g} strong-lens "
                                            "candidate: blue source offset from the red lens galaxy.")))
    return out


def _rows_from_negatives(excl: set, n_random: int, n_mimic: int) -> list[dict]:
    out = []
    negp = config.INCH_DATA / "negatives_extra.parquet"
    if negp.exists() and n_random > 0:
        neg = pd.read_parquet(negp)
        neg = neg[~neg["row_id"].astype(str).isin(excl)].sample(
            min(n_random, len(neg)), random_state=SEED)
        for _, r in neg.iterrows():
            out.append(dict(name=str(r["row_id"]), ra=r.get("RA"), dec=r.get("DEC"),
                            survey_key="storfer", label="nonlens",
                            target=_target_json("D", rationale="Ordinary galaxy; no lensing features.")))
    mimp = config.REPRO / "claudenet" / "data" / "v3" / "mimic_bank_seed.parquet"
    if mimp.exists() and n_mimic > 0:
        mm = pd.read_parquet(mimp)
        mm = mm[~mm["row_id"].astype(str).isin(excl)].sample(min(n_mimic, len(mm)), random_state=SEED)
        for _, r in mm.iterrows():
            mt = str(r.get("mimic_type", "contaminant"))
            out.append(dict(name=str(r["row_id"]), ra=r.get("RA"), dec=r.get("DEC"),
                            survey_key="claudenet", label="nonlens",
                            target=_target_json("D", contaminant=mt,
                                                 rationale=f"CNN-high lens MIMIC ({mt}); not a real lens.")))
    return out


def _rows_from_claude(excl: set, max_n: int) -> list[dict]:
    """Claude-graded claudenet_737 campaign candidates — true teacher distillation on the hard
    CNN-selected frontier (uses Claude's grade/p_lens/criteria/contaminant/rationale verbatim)."""
    f = config.OUT / "claudenet_737_lean.parquet"
    if not f.exists():
        return []
    d = pd.read_parquet(f)
    d = d[(d["parse_ok"] == True) & (~d["name"].astype(str).isin(excl))]  # noqa: E712
    if len(d) > max_n:
        d = d.sample(max_n, random_state=SEED)
    out = []
    for _, r in d.iterrows():
        crit = {k: int(r.get(f"crit_{k}", 0) or 0) for k in
                ("blue_source", "low_surface_brightness", "curvature", "counter_images", "arc_morphology")}
        g = str(r["grade_pred"]) if r.get("grade_pred") in ("A", "B", "C", "D") else "D"
        out.append(dict(name=str(r["name"]), ra=None, dec=None,
                        survey_key=str(r.get("catalog") or "claudenet"),
                        label=("lens" if g in ("A", "B", "C") else "nonlens"),
                        target=_target_json(g, contaminant=(r.get("contaminant") or None),
                                            rationale=str(r.get("rationale") or "")[:300],
                                            crit=crit, p_lens=float(r["p_lens"]),
                                            confidence=float(r.get("confidence") or 0.6))))
    return out


def _render_and_write(rows: list[dict], img_dir: Path, views) -> list[dict]:
    """Render each row's views to PNG, return ms-swift messages records (skip rows with no cutout)."""
    img_dir.mkdir(parents=True, exist_ok=True)
    recs, miss = [], 0
    for i, r in enumerate(rows):
        cube = fetch.get_cube(name=r["name"], ra=r.get("ra"), dec=r.get("dec"), survey=r["survey_key"])
        if cube is None:
            miss += 1
            continue
        imgs = render.render_views(cube, views=[v for v in views if v in render.VIEWS])
        if not imgs:
            miss += 1
            continue
        safe = str(r["name"]).replace("/", "_").replace(" ", "_")
        paths, tags = [], []
        for v, img in imgs.items():
            p = img_dir / f"{safe}_{v}.png"
            render.save_png(img, p)
            paths.append(str(p))
            tags.append("<image>")
        cand = {"name": r["name"], "survey_key": r["survey_key"], "ra": r.get("ra"), "dec": r.get("dec")}
        user = "".join(tags) + "\n" + _candidate_text(cand)
        recs.append({"messages": [
            {"role": "system", "content": _RUBRIC},
            {"role": "user", "content": user},
            {"role": "assistant", "content": r["target"]},
        ], "images": paths, "label": r["label"]})
        if (i + 1) % 200 == 0:
            print(f"  rendered {i + 1}/{len(rows)} (missing {miss})")
    print(f"  rendered {len(recs)} examples ({miss} skipped for missing cutout)")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pos", type=int, default=900)
    ap.add_argument("--max-neg", type=int, default=900, help="split across random+mimic+claude")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out-dir", default=str(config.OUT / "sft"))
    args = ap.parse_args()

    excl = _excluded_names()
    print(f"[sft] excluding {len(excl)} eval-manifest names (no leakage)")
    pos = _rows_from_human(excl, args.max_pos)
    n3 = args.max_neg // 3
    neg = _rows_from_negatives(excl, n_random=n3, n_mimic=n3) + _rows_from_claude(excl, n3)
    print(f"[sft] candidate rows: {len(pos)} pos, {len(neg)} neg (pre-render)")

    out_dir = Path(args.out_dir)
    recs = _render_and_write(pos + neg, out_dir / "images", _DEFAULT_VIEWS)

    rng = np.random.RandomState(SEED)
    rng.shuffle(recs)
    n_val = int(len(recs) * args.val_frac)
    val, train = recs[:n_val], recs[n_val:]
    for split, data in (("train", train), ("val", val)):
        p = out_dir / f"sft_{split}.jsonl"
        with open(p, "w") as fh:
            for rec in data:
                fh.write(json.dumps(rec) + "\n")
        npos = sum(1 for r in data if r["label"] == "lens")
        print(f"[sft] {split}: {len(data)} examples ({npos} lens / {len(data) - npos} nonlens) -> {p}")


if __name__ == "__main__":
    main()
