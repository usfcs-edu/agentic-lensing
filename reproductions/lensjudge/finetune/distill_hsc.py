#!/usr/bin/env python3
"""finetune/distill_hsc.py — HSC tier-2 SUPERVISED set + REAL-LENS gate (productionizing the Euclid PoC).

The Euclid PoC fixed the collapse but the student over-called (Euclid = all lens candidates, no true
negatives). This trains on REAL HSC labels: SuGOHI-confirmed lenses (positives) + lens MIMICS / random
galaxies (true hard negatives), and gates on HELD-OUT real confirmed lenses vs real non-lenses — not
teacher agreement. DIRECT paradigm (inline HSC views, no tool loop), reusing grader_direct.

Prereqs: SuGOHI HSC cutouts (eval/stage_hsc from xmatch_sugohi) + negatives fetched (neg_positions.csv);
merge both into ONE cache and point LENSJUDGE_HSC_CACHE at it.

Stages:
  manifest : split HSC-covered SuGOHI positives + covered negatives into train/test (disjoint).
  sft      : GROUND-TRUTH SFT (pos->A, neg->D) + HSC PNGs -> ms-swift JSONL (combine w/ Euclid for scale).
  label    : DIRECT HSC-grade a manifest (current backend) -> Claude ref on the test set for the gate.
  gate     : real-lens metrics (recovery of confirmed lenses, rejection of non-lenses, lens-vs-nonlens
             AUC) for the distilled student vs off-the-shelf vs Claude.
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
from lensjudge.common import hsc, hsc_fetch, render  # noqa: E402
from lensjudge.imaging import grader_direct  # noqa: E402
from lensjudge.finetune.build_sft_data import _target_json  # noqa: E402
from lensjudge.tools.hsc_cutout import VIEW_DESC as HSC_VIEW_DESC  # noqa: E402

SEED = 2026
OUT = config.OUT / "distill_hsc"
VIEWS = ("full", "lum", "zoom", "lum_sub")
RUBRIC_V2 = (config.HERE / "prompts" / "rubric_imaging_v2.md").read_text()

_DIRECT_NOTE = """

# IMPORTANT — these are HSC-SSP PDR3 cutouts (0.168"/px grizy), rendered INLINE below (NOT DESI grz, no tool)
- Resolution 0.168"/px (~0.6" seeing), ~8x sharper than DESI: tangential arcs / Einstein rings blurred at
  ground resolution are resolved here. Judge the morphology you actually see.
- Color views: R=i, G=r, B=g (old red lens galaxy red/orange; lensed background source blue). The 'lum'
  (sharp i-band) and 'lum_sub' (lens-light-subtracted) views show thin arcs/rings best.
- Apply the SAME A/B/C/D scale + v2 rubric (rule out LRG+companion / ring / spiral / merger via colour +
  radial geometry). There are no CNN scores.
"""
DIRECT_SYS = RUBRIC_V2 + _DIRECT_NOTE


def _hsc_imgs(ra, dec):
    bands = hsc.load_hsc(float(ra), float(dec))
    if bands is None:
        return None
    return hsc.render_hsc_views(bands, views=[v for v in VIEWS if v in hsc.VIEWS])


def hsc_content(ra, dec):
    imgs = _hsc_imgs(ra, dec)
    if not imgs:
        return None
    content = [{"type": "text", "text":
                f"Grade this strong-lens candidate at HSC-SSP PDR3 0.168\"/px "
                f"(ra={float(ra):.6f}, dec={float(dec):.6f}). Rendered grizy views:"}]
    for v, img in imgs.items():
        content.append({"type": "text", "text": f"[{v}] {HSC_VIEW_DESC.get(v, '')}"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": render.png_b64(img)}})
    content.append({"type": "text", "text": "Respond with ONLY the JSON object for the required schema."})
    return content


# ------------------------------------------------------------------ manifest
def stage_manifest(args):
    su = pd.read_csv(config.OUT / "xmatch_sugohi.csv")
    pos = [{"name": str(r["name"]), "ra": float(r["ra"]), "dec": float(r["dec"]),
            "label": "lens", "kind": "sugohi"}
           for _, r in su.iterrows() if hsc_fetch.cached(float(r["ra"]), float(r["dec"]))]
    neg_src = pd.read_csv(OUT / "neg_positions.csv")
    neg = [{"name": str(r["name"]), "ra": float(r["ra"]), "dec": float(r["dec"]),
            "label": "nonlens", "kind": str(r["kind"])}
           for _, r in neg_src.iterrows() if hsc_fetch.cached(float(r["ra"]), float(r["dec"]))]
    print(f"[manifest] HSC-covered: {len(pos)} positives (SuGOHI), {len(neg)} negatives "
          f"({pd.Series([n['kind'] for n in neg]).value_counts().to_dict()})")

    def split(rows):
        idx = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        nt = max(1, int(round(len(idx) * args.test_frac)))
        return idx.iloc[nt:], idx.iloc[:nt]
    ptr, pte = split(pos)
    ntr, nte = split(neg)
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.concat([ptr, ntr]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    te = pd.concat([pte, nte]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    tr.to_csv(OUT / "hsc_train.csv", index=False)
    te.to_csv(OUT / "hsc_test.csv", index=False)
    print(f"[manifest] train {len(tr)} ({tr.label.value_counts().to_dict()}) -> hsc_train.csv")
    print(f"[manifest] test  {len(te)} ({te.label.value_counts().to_dict()}) -> hsc_test.csv")


# ----------------------------------------------------------------------- sft
def stage_sft(args):
    tr = pd.read_csv(args.manifest)
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    recs, miss = [], 0
    for _, r in tr.iterrows():
        imgs = _hsc_imgs(r["ra"], r["dec"])
        if not imgs:
            miss += 1
            continue
        paths, tags = [], []
        for v, img in imgs.items():
            p = img_dir / f"{r['name']}_{v}.png"
            render.save_png(img, p)
            paths.append(str(p)); tags.append("<image>")
        if r["label"] == "lens":  # GROUND TRUTH: SuGOHI-confirmed strong lens
            target = _target_json("A", rationale="Known SuGOHI-confirmed strong lens: resolved tangential "
                                  "arc / Einstein ring around the red lens galaxy at HSC resolution.")
        else:                     # GROUND TRUTH: true non-lens (mimic / random galaxy)
            kind = str(r.get("kind", "contaminant"))
            target = _target_json("D", contaminant=(kind if kind != "random" else None),
                                  rationale=f"Known non-lens ({kind}); no genuine tangential arc, "
                                            "counter-image, or lensing geometry.")
        user = "".join(tags) + f"\nGrade this strong-lens candidate at HSC 0.168\"/px. " \
               "Rendered grizy views (full, lum, zoom, lum_sub). Respond with ONLY the JSON."
        recs.append({"messages": [{"role": "system", "content": DIRECT_SYS},
                                  {"role": "user", "content": user},
                                  {"role": "assistant", "content": target}],
                     "images": paths, "label": r["label"]})
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
    print(f"[label] {len(man)} objects | backend={llm_client.get_backend()} | "
          f"model={args.model or config.MODELS['grader']}")

    async def grade_one(r):
        content = hsc_content(r["ra"], r["dec"])
        if content is None:
            return {"name": str(r["name"]), "label": r.get("label"), "p_lens": np.nan,
                    "agent_grade": None, "error": "no cutout"}
        g = await grader_direct.grade_candidate({"name": str(r["name"])}, model=args.model,
                                                system_prompt=DIRECT_SYS, content=content)
        return {"name": str(r["name"]), "label": r.get("label"), "kind": r.get("kind"),
                "p_lens": (g.grade.p_lens if g.grade else np.nan),
                "agent_grade": (g.grade.grade if g.grade else None),
                "cost_usd": g.cost_usd, "error": g.error}
    recs = asyncio.run(_bounded([lambda r=r: grade_one(r) for _, r in man.iterrows()], args.concurrency))
    df = pd.DataFrame(recs)
    outp = args.out or str(OUT / "hsc_labels.parquet")
    df.to_parquet(outp, index=False)
    ok = df[df.agent_grade.notna()]
    print(f"[label] parsed {len(ok)}/{len(df)} | grades {ok.agent_grade.value_counts().to_dict()} | "
          f"${df.cost_usd.sum():.2f} -> {outp}")


# ---------------------------------------------------------------------- gate
def _auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    pos = np.asarray(pos, float)[~np.isnan(np.asarray(pos, float))]
    neg = np.asarray(neg, float)[~np.isnan(np.asarray(neg, float))]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def _score(preds, truth, tag):
    j = truth.merge(preds[["name", "p_lens", "agent_grade"]], on="name")
    pos, neg = j[j.label == "lens"], j[j.label == "nonlens"]
    rec = pos.agent_grade.isin(["A", "B"]).mean() if len(pos) else float("nan")
    rej = (~neg.agent_grade.isin(["A", "B"])).mean() if len(neg) else float("nan")
    auc = _auc(pos.p_lens, neg.p_lens)
    print(f"  {tag}: n={len(j)} | REAL-lens AUC={auc:.3f} ({len(pos)} lens vs {len(neg)} non) | "
          f"recovery(lens->A/B)={rec:.0%} | rejection(non->not A/B)={rej:.0%} | "
          f"mean p_lens L={pos.p_lens.mean():.2f}/N={neg.p_lens.mean():.2f}")


def stage_gate(args):
    truth = pd.read_csv(args.manifest)[["name", "label"]]
    print("=== HSC tier-2 REAL-lens gate (held-out SuGOHI lenses vs true non-lenses) ===")
    _score(pd.read_parquet(args.student), truth, "DISTILLED student")
    if args.baseline:
        _score(pd.read_parquet(args.baseline), truth, "off-the-shelf  ")
    if args.claude:
        _score(pd.read_parquet(args.claude), truth, "Claude (oracle) ")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    m = sub.add_parser("manifest"); m.add_argument("--test-frac", type=float, default=0.35)
    s = sub.add_parser("sft"); s.add_argument("--manifest", required=True)
    s.add_argument("--val-frac", type=float, default=0.1)
    lb = sub.add_parser("label"); lb.add_argument("--manifest", required=True)
    lb.add_argument("--out", default=None); lb.add_argument("--model", default=None)
    lb.add_argument("--concurrency", type=int, default=6)
    g = sub.add_parser("gate"); g.add_argument("--manifest", required=True)
    g.add_argument("--student", required=True); g.add_argument("--baseline", default=None)
    g.add_argument("--claude", default=None)
    args = ap.parse_args()
    {"manifest": stage_manifest, "sft": stage_sft, "label": stage_label,
     "gate": stage_gate}[args.stage](args)


if __name__ == "__main__":
    main()
