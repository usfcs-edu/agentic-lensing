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
    soft = None
    if getattr(args, "claude_preds", None):   # SOFT distillation: use Claude's actual grades as targets
        cp = pd.read_parquet(args.claude_preds)
        soft = {str(n): tj for n, tj in zip(cp["name"], cp["target_json"]) if isinstance(tj, str)}
        print(f"[sft] SOFT distillation: {len(soft)} Claude targets loaded (calibrated, not ground-truth)")
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    recs, miss = [], 0
    for _, r in tr.iterrows():
        if soft is not None and str(r["name"]) not in soft:
            miss += 1
            continue
        imgs = _hsc_imgs(r["ra"], r["dec"])
        if not imgs:
            miss += 1
            continue
        paths, tags = [], []
        for v, img in imgs.items():
            p = img_dir / f"{r['name']}_{v}.png"
            render.save_png(img, p)
            paths.append(str(p)); tags.append("<image>")
        if soft is not None:      # SOFT: Claude's calibrated grade (captures visual difficulty)
            target = soft[str(r["name"])]
        elif r["label"] == "lens":  # GROUND TRUTH: SuGOHI-confirmed strong lens
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
                "p_lens_logprob": g.meta.get("p_lens_logprob"),
                "target_json": (g.grade.model_dump_json() if g.grade else None),
                "cost_usd": g.cost_usd, "error": g.error}
    recs = asyncio.run(_bounded([lambda r=r: grade_one(r) for _, r in man.iterrows()], args.concurrency))
    df = pd.DataFrame(recs)
    outp = args.out or str(OUT / "hsc_labels.parquet")
    df.to_parquet(outp, index=False)
    ok = df[df.agent_grade.notna()]
    print(f"[label] parsed {len(ok)}/{len(df)} | grades {ok.agent_grade.value_counts().to_dict()} | "
          f"${df.cost_usd.sum():.2f} -> {outp}")


# -------------------------------------------------------------------- sft-v5
_V5_GRADE = {"lens": None, "nonlens": "D", "softmid": "C"}   # lens keeps its catalog grade (A/B)


def _hash01(name: str, salt: str) -> float:
    """Deterministic per-example pseudo-random float in [0,1) — reproducible target jitter."""
    import hashlib
    return int(hashlib.md5(f"{salt}:{name}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _v5_target(r) -> str:
    """Per-example UNIQUE ImageGrade target. v5 collapse lesson (mode-collapse run 2026-07-05):
    class-constant templates let the student minimize loss by emitting the class prior and IGNORING
    the images (valsel AUC -> 0.50 while train loss -> 0.02). The Euclid PoC worked because Claude's
    targets were per-example unique — replicate that property from CATALOG FACTS + deterministic
    hash jitter: unique short rationales (coords/theta_E/score), soft-correlated jittered criteria,
    de-constanted p_lens. Loss can then only fall by conditioning on the pixels."""
    name = str(r.name)
    soft = float(r.soft)
    h = lambda s: _hash01(name, s)                      # noqa: E731
    # p_lens: keep continuous committee scores exactly; de-constant letter-grade mappings (+-0.04)
    if getattr(r, "src", "") == "sugohi" or r.label == "softmid":
        p = round(min(0.99, max(0.01, soft + (h("p") - 0.5) * 0.08)), 2)
    else:
        p = round(min(0.99, max(0.01, soft)), 2)
    # criteria: correlated with soft, per-example jitter, per-criterion offsets
    crit = {}
    for i, k in enumerate(("blue_source", "low_surface_brightness", "curvature",
                           "counter_images", "arc_morphology")):
        base = 10.0 * soft * (0.75 + 0.5 * h(f"c{i}"))
        crit[k] = int(min(10, max(0, round(base + (h(f"o{i}") - 0.5) * 2))))
    conf = round(min(0.95, max(0.2, 0.4 + abs(soft - 0.5) + (h("cf") - 0.5) * 0.2)), 2)
    # short, fact-bearing, per-example rationale (facts vary; phrasing varies by hash)
    facts = [f"({r.ra:.4f},{r.dec:.4f})"]
    te = getattr(r, "theta_e", None)
    if te is not None and pd.notna(te) and float(te) > 0:
        facts.append(f"thetaE~{float(te):.2f}\"")
    if str(getattr(r, "src", "")).startswith("holismokes"):
        facts.append(f"committee score {soft * 3:.2f}/3")
    fact_s = " ".join(facts)
    if r.label == "lens":
        stems = [f"Grade-{r.grade} committee lens {fact_s}: arc structure around the deflector.",
                 f"Committee-confirmed grade-{r.grade} candidate {fact_s}; lensing morphology present.",
                 f"{fact_s}: graded {r.grade} by the expert committee; consistent arc geometry."]
    elif r.label == "nonlens":
        stems = [f"Expert-rejected candidate {fact_s}: arc-like feature is a contaminant.",
                 f"{fact_s}: committee rejected; no genuine lensing geometry.",
                 f"Rejected on inspection {fact_s}; feature not consistent with lensing."]
    else:
        stems = [f"Ambiguous grade-C candidate {fact_s}: suggestive but unconvincing features.",
                 f"{fact_s}: committee grade C; lens-like hints below the confidence bar.",
                 f"Borderline candidate {fact_s}, graded C by the committee."]
    rationale = stems[int(h("r") * len(stems))]
    grade = _V5_GRADE[r.label] or r.grade
    return _target_json(grade, p_lens=p, rationale=rationale, crit=crit, confidence=conf,
                        contaminant=("contaminant" if r.label == "nonlens" else None))


def stage_sft_v5(args):
    """Build the v5 SFT set from finetune/build_corpus_v5 output: HSC PNGs + HUMAN-SOFT ImageGrade
    targets (catalog grade + committee-score p_lens), per-example UNIQUE (see _v5_target)."""
    tr = pd.read_csv(args.manifest)
    # join SuGOHI theta_e for fact-bearing rationales (unique targets); NaN-safe
    sug = config.HERE / "cache" / "v5_catalogs" / "sugohi" / "sugohi_normalized.csv"
    if sug.exists():
        s = pd.read_csv(sug)[["name", "theta_e"]]
        tr = tr.merge(s, on="name", how="left")
    else:
        tr["theta_e"] = np.nan
    img_dir = OUT / "images_v5"
    img_dir.mkdir(parents=True, exist_ok=True)
    recs, miss = [], 0
    for r in tr.itertuples():
        imgs = _hsc_imgs(r.ra, r.dec)
        if not imgs:
            miss += 1
            continue
        safe = str(r.name).replace("/", "_").replace(" ", "_")
        paths, tags = [], []
        for v, img in imgs.items():
            p = img_dir / f"{safe}_{v}.png"
            if not p.exists():
                render.save_png(img, p)
            paths.append(str(p)); tags.append("<image>")
        target = _v5_target(r)
        user = "".join(tags) + "\nGrade this strong-lens candidate at HSC 0.168\"/px. " \
               "Rendered grizy views (full, lum, zoom, lum_sub). Respond with ONLY the JSON."
        recs.append({"messages": [{"role": "system", "content": DIRECT_SYS},
                                  {"role": "user", "content": user},
                                  {"role": "assistant", "content": target}],
                     "images": paths, "label": r.label})
        if len(recs) % 500 == 0:
            print(f"  rendered {len(recs)} (missing {miss})")
    rng = np.random.RandomState(SEED)
    rng.shuffle(recs)
    n_val = int(len(recs) * args.val_frac)
    for split, data in (("train", recs[n_val:]), ("val", recs[:n_val])):
        p = OUT / f"sft_v5_{split}.jsonl"
        with open(p, "w") as fh:
            for rec in data:
                fh.write(json.dumps(rec) + "\n")
        counts = pd.Series([x["label"] for x in data]).value_counts().to_dict()
        print(f"[sft-v5] {split}: {len(data)} ({counts}) -> {p}")
    print(f"[sft-v5] {miss} skipped for missing cutout")


# ------------------------------------------------------------------- evalset
def stage_evalset(args):
    """Build an ms-swift infer valset (HSC images, no assistant) + labels CSV from a manifest,
    using its GROUND-TRUTH lens/nonlens labels — for HSC-only AUC checkpoint selection."""
    man = pd.read_csv(args.manifest)
    img_dir = OUT / "eval_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in man.iterrows():
        imgs = _hsc_imgs(r["ra"], r["dec"])
        if not imgs:
            continue
        paths, tags = [], []
        for v, img in imgs.items():
            p = img_dir / f"{r['name']}_{v}.png"
            render.save_png(img, p)
            paths.append(str(p)); tags.append("<image>")
        user = "".join(tags) + "\nGrade this strong-lens candidate at HSC 0.168\"/px. " \
               "Rendered grizy views (full, lum, zoom, lum_sub). Respond with ONLY the JSON."
        rows.append({"rec": {"messages": [{"role": "system", "content": DIRECT_SYS},
                                          {"role": "user", "content": user}], "images": paths},
                     "label": r["label"], "name": str(r["name"])})
    OUT.mkdir(parents=True, exist_ok=True)
    js, lab = OUT / f"{args.out_prefix}.jsonl", OUT / f"{args.out_prefix}_labels.csv"
    with open(js, "w") as fh:
        for x in rows:
            fh.write(json.dumps(x["rec"]) + "\n")
    pd.DataFrame([{"idx": i, "label": x["label"], "name": x["name"]}
                 for i, x in enumerate(rows)]).to_csv(lab, index=False)
    npos = sum(1 for x in rows if x["label"] == "lens")
    print(f"[evalset] {len(rows)} rows ({npos} lens / {len(rows) - npos} nonlens) -> {js} + {lab}")


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
    s.add_argument("--claude-preds", default=None, help="parquet w/ name,target_json -> SOFT distillation")
    ev = sub.add_parser("evalset"); ev.add_argument("--manifest", required=True)
    ev.add_argument("--out-prefix", required=True)
    s5 = sub.add_parser("sft-v5"); s5.add_argument("--manifest", required=True)
    s5.add_argument("--val-frac", type=float, default=0.03)
    lb = sub.add_parser("label"); lb.add_argument("--manifest", required=True)
    lb.add_argument("--out", default=None); lb.add_argument("--model", default=None)
    lb.add_argument("--concurrency", type=int, default=6)
    g = sub.add_parser("gate"); g.add_argument("--manifest", required=True)
    g.add_argument("--student", required=True); g.add_argument("--baseline", default=None)
    g.add_argument("--claude", default=None)
    args = ap.parse_args()
    {"manifest": stage_manifest, "sft": stage_sft, "label": stage_label,
     "evalset": stage_evalset, "sft-v5": stage_sft_v5, "gate": stage_gate}[args.stage](args)


if __name__ == "__main__":
    main()
