#!/usr/bin/env python3
"""finetune/distill_euclid.py — tier-2 (Euclid) distillation PoC: distill Claude's high-res grades
into the open student (Qwen3-VL-8B) and gate whether it fixes the student's over-skeptical collapse.

Off-the-shelf open VLMs collapse at tier-2 (Qwen3-VL-8B 86/89-D at Euclid; GLM-4.6V-106B 25/25-D at
HSC) while Claude recovers known lenses. Unlike DESI tier-1 (weak teacher -> distillation failed),
at Euclid 0.1" Claude's teacher signal is meaningful, so this tests whether LoRA distillation of
Claude's DIRECT Euclid grades transfers Claude's calibration to the student.

DIRECT paradigm (no tool loop, matches the deployable cheap grader + the existing v2 pipeline):
render the 4 Euclid views inline -> one grade. Same path for teacher (anthropic) and student (openai),
via grader_direct.grade_candidate(content=...).

Stages:
  manifest : split staged Euclid objects into train / valsel / test (disjoint, grade-stratified).
             python -m lensjudge.finetune.distill_euclid manifest
  label    : batch DIRECT-grade a manifest with the CURRENT backend (Claude teacher, or student gate).
             LENSJUDGE_BACKEND=anthropic python -m ...distill_euclid label --manifest euclid_train.csv \
               --out euclid_train_claude.parquet --concurrency 6
  sft      : build ms-swift SFT (Euclid PNGs + Claude JSON targets) from the train labels.
             python -m ...distill_euclid sft --train-preds euclid_train_claude.parquet
  gate     : compare a student preds parquet vs the Claude reference on the test set.
             python -m ...distill_euclid gate --student euclid_test_student.parquet \
               --ref euclid_test_claude.parquet [--baseline euclid_test_offshelf.parquet]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import euclid, render  # noqa: E402
from lensjudge.imaging import grader_direct  # noqa: E402
from lensjudge.imaging.grader_lean import _RUBRIC  # noqa: E402
from lensjudge.tools.euclid_cutout import VIEW_DESC as EUCLID_VIEW_DESC  # noqa: E402

SEED = 2026
OUT = config.OUT / "distill_euclid"
VIEWS = ("full", "vis", "zoom", "vis_sub")
CAT = euclid.EUCLID_ROOT / "raw" / "q1_discovery_engine_lens_catalog.csv"

# DIRECT Euclid system prompt: the base rubric + a note that the views are rendered INLINE (no tool).
_DIRECT_NOTE = """

# IMPORTANT — these are EUCLID Q1 cutouts (0.1"/px VIS+NIR), rendered INLINE below (NOT DESI grz, no tool)
- Resolution 0.1"/px (~13x sharper than DESI's ~1.3" seeing): arcs/rings blurred at ground resolution
  are cleanly resolved here. Judge the morphology you actually see.
- Bands: VIS (sharp broad-optical luminance) + NIR Y/J/H. Old red lens galaxy is red/orange; a lensed
  background source is blue. The vis and vis_sub (lens-light-subtracted) views show thin arcs best.
- Apply the SAME 5-criterion rubric and the SAME A/B/C/D scale. There are no CNN scores.
"""
DIRECT_SYS = _RUBRIC + _DIRECT_NOTE


def _euclid_imgs(id_str: str):
    bands = euclid.load_euclid(id_str)
    if bands is None:
        return None
    return euclid.render_euclid_views(bands, views=[v for v in VIEWS if v in euclid.VIEWS])


def euclid_content(id_str: str):
    """Pre-rendered Euclid evidence blocks for grader_direct.grade_candidate(content=...)."""
    imgs = _euclid_imgs(id_str)
    if not imgs:
        return None
    content = [{"type": "text", "text":
                f"Grade this Euclid Q1 strong-lens candidate (id_str={id_str}). "
                f"Rendered 0.1\"/px VIS+NIR views:"}]
    for v, img in imgs.items():
        content.append({"type": "text", "text": f"[{v}] {EUCLID_VIEW_DESC.get(v, '')}"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": render.png_b64(img)}})
    content.append({"type": "text", "text": "Respond with ONLY the JSON object for the required schema."})
    return content


# ------------------------------------------------------------------ manifest
def stage_manifest(args):
    cat = pd.read_csv(CAT).set_index("id_str")
    rows = []
    for sub in euclid.SUBSETS:
        d = euclid.EUCLID_ROOT / sub
        if not d.exists():
            continue
        for idd in sorted(p.name for p in d.iterdir() if p.is_dir()):
            if idd not in cat.index or euclid.obj_dir(idd) is None:
                continue
            r = cat.loc[idd]
            rows.append({"id_str": idd, "subset": sub, "grade": r["grade"],
                         "expert_score": float(r["expert_score"])})
    df = pd.DataFrame(rows)
    print(f"[manifest] {len(df)} staged Euclid objects; grade {df.grade.value_counts().to_dict()}")
    # grade-stratified split: test (n_test) + valsel (n_val) held out disjoint, rest = train
    rng = np.random.RandomState(SEED)
    test, val, train = [], [], []
    for g, sub in df.groupby("grade"):
        idx = sub.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        nt = max(1, int(round(len(idx) * args.test_frac)))
        nv = max(1, int(round(len(idx) * args.val_frac)))
        test.append(idx.iloc[:nt]); val.append(idx.iloc[nt:nt + nv]); train.append(idx.iloc[nt + nv:])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, parts in (("train", train), ("valsel", val), ("test", test)):
        d = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        p = OUT / f"euclid_{name}.csv"
        d.to_csv(p, index=False)
        print(f"[manifest] {name}: {len(d)} ({d.grade.value_counts().to_dict()}) -> {p}")


# --------------------------------------------------------------------- label
async def _bounded(fns, n):
    sem = asyncio.Semaphore(n)

    async def run(f):
        async with sem:
            return await f()
    return await asyncio.gather(*[run(f) for f in fns])


def stage_label(args):
    from lensjudge.common import llm_client
    man = pd.read_csv(args.manifest)
    backend = llm_client.get_backend()
    model = args.model
    print(f"[label] {len(man)} objects | backend={backend} | model={model or config.MODELS['grader']}")

    async def grade_one(r):
        idd = str(r["id_str"])
        content = euclid_content(idd)
        if content is None:
            return {"id_str": idd, "grade": r.get("grade"), "p_lens": np.nan, "agent_grade": None,
                    "target_json": None, "error": "no cutout"}
        g = await grader_direct.grade_candidate(
            {"name": idd}, model=model, system_prompt=DIRECT_SYS, content=content)
        return {"id_str": idd, "grade": r.get("grade"), "expert_score": r.get("expert_score"),
                "p_lens": (g.grade.p_lens if g.grade else np.nan),
                "agent_grade": (g.grade.grade if g.grade else None),
                "confidence": (g.grade.confidence if g.grade else np.nan),
                "contaminant": (g.grade.contaminant if g.grade else None),
                "target_json": (g.grade.model_dump_json() if g.grade else None),
                "cost_usd": g.cost_usd, "error": g.error}
    recs = asyncio.run(_bounded([lambda r=r: grade_one(r) for _, r in man.iterrows()], args.concurrency))
    df = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    outp = args.out or str(OUT / "euclid_labels.parquet")
    df.to_parquet(outp, index=False)
    ok = df[df.agent_grade.notna()]
    print(f"[label] parsed {len(ok)}/{len(df)} | grades {ok.agent_grade.value_counts().to_dict()} | "
          f"mean p_lens {ok.p_lens.mean():.3f} | total ${df.cost_usd.sum():.2f} -> {outp}")


# ----------------------------------------------------------------------- sft
def stage_sft(args):
    preds = pd.read_parquet(args.train_preds)
    preds = preds[preds.target_json.notna()].reset_index(drop=True)
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    recs, miss = [], 0
    for _, r in preds.iterrows():
        idd = str(r["id_str"])
        imgs = _euclid_imgs(idd)
        if not imgs:
            miss += 1
            continue
        paths, tags = [], []
        for v, img in imgs.items():
            p = img_dir / f"{idd}_{v}.png"
            render.save_png(img, p)
            paths.append(str(p)); tags.append("<image>")
        user = "".join(tags) + f"\nGrade this Euclid Q1 strong-lens candidate (id_str={idd}). " \
               "Rendered 0.1\"/px VIS+NIR views (full, vis, zoom, vis_sub). Respond with ONLY the JSON."
        recs.append({"messages": [
            {"role": "system", "content": DIRECT_SYS},
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(r["target_json"])},
        ], "images": paths, "label": ("lens" if str(r.get("agent_grade")) in ("A", "B", "C") else "nonlens")})
    rng = np.random.RandomState(SEED)
    rng.shuffle(recs)
    n_val = int(len(recs) * args.val_frac)
    OUT.mkdir(parents=True, exist_ok=True)
    for split, data in (("train", recs[n_val:]), ("val", recs[:n_val])):
        p = OUT / f"sft_{split}.jsonl"
        with open(p, "w") as fh:
            for rec in data:
                fh.write(json.dumps(rec) + "\n")
        npos = sum(1 for r in data if r["label"] == "lens")
        print(f"[sft] {split}: {len(data)} ({npos} lens / {len(data) - npos} nonlens) -> {p}")
    print(f"[sft] {miss} skipped for missing cutout")


# ---------------------------------------------------------------------- gate
def _auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, np.r_[pos, neg])


def _score(student, ref, tag):
    from scipy.stats import spearmanr
    j = ref[["id_str", "p_lens", "agent_grade"]].rename(
        columns={"p_lens": "cp", "agent_grade": "cg"}).merge(
        student[["id_str", "p_lens", "agent_grade"]], on="id_str")
    j = j.dropna(subset=["cp", "p_lens"])
    rho = spearmanr(j.cp, j.p_lens).correlation if len(j) > 2 else float("nan")
    # teacher-defined pos/neg: Claude A/B vs Claude D
    pos = j[j.cg.isin(["A", "B"])].p_lens
    neg = j[j.cg == "D"].p_lens
    rec = j[j.cg.isin(["A", "B"])].agent_grade.isin(["A", "B"]).mean() if (j.cg.isin(["A", "B"])).any() else float("nan")
    print(f"  {tag}: n={len(j)} | Spearman(student,Claude p_lens)={rho:.3f} | "
          f"AUC(Claude A/B vs D)={_auc(pos, neg):.3f} ({len(pos)} vs {len(neg)}) | "
          f"recovery of Claude-A/B={rec:.0%} | mean p_lens={j.p_lens.mean():.3f}")


def stage_gate(args):
    ref = pd.read_parquet(args.ref)
    print("=== Euclid tier-2 distillation gate (vs Claude teacher on held-out test) ===")
    print(f"  Claude ref: grades {ref[ref.agent_grade.notna()].agent_grade.value_counts().to_dict()} | "
          f"mean p_lens {ref.p_lens.mean():.3f}")
    _score(pd.read_parquet(args.student), ref, "DISTILLED student")
    if args.baseline:
        _score(pd.read_parquet(args.baseline), ref, "off-the-shelf ")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    m = sub.add_parser("manifest"); m.add_argument("--test-frac", type=float, default=0.15)
    m.add_argument("--val-frac", type=float, default=0.12)
    lb = sub.add_parser("label"); lb.add_argument("--manifest", required=True)
    lb.add_argument("--out", default=None); lb.add_argument("--model", default=None)
    lb.add_argument("--concurrency", type=int, default=6)
    s = sub.add_parser("sft"); s.add_argument("--train-preds", required=True)
    s.add_argument("--val-frac", type=float, default=0.1)
    g = sub.add_parser("gate"); g.add_argument("--student", required=True)
    g.add_argument("--ref", required=True); g.add_argument("--baseline", default=None)
    args = ap.parse_args()
    {"manifest": stage_manifest, "label": stage_label, "sft": stage_sft, "gate": stage_gate}[args.stage](args)


if __name__ == "__main__":
    main()
