#!/usr/bin/env python3
"""finetune/build_corpus_desi.py — Phase C2: DESI-resolution SFT corpus from the parity train pool.

Transplants the v5 HSC recipe (build_corpus_v5 + distill_hsc sft-v5 + augment_sft_v5) to
DESI-resolution cutouts. Targets are HUMAN grades + soft p_lens (the pool's soft_target column) —
NOT Claude distillation, which failed at DESI resolution in v4. The ONLY data source is
outputs/parity_train_pool.csv (SHA-pinned, firewalled against the parity bench by name AND
position); split=gate rows are the frozen evidence gate and NEVER enter the corpus.

Composition (mirror v5 proportions — don't drown the graded signal in random negatives):
  train  : ALL graded A/B/C + ALL graded_D + a seed-2026 sample of 2,500 random_neg
  valsel : ALL graded + ALL graded_D + a proportional random_neg sample (same rf:graded ratio)

Targets are per-example UNIQUE (the v5 mode-collapse lesson, distill_hsc._v5_target): the pool's
soft_target values are letter-grade CONSTANTS (A .95 / B .80 / C .45 / D .05 / random .02), so
every p_lens is de-constanted with +-0.04 deterministic hash jitter, criteria are soft-correlated
with per-criterion jitter, and rationales are short fact-bearing stems (coords + published CNN
prior, 3 hash-selected phrasings). random_neg rows use the augment_sft_v5._rf_target analogue
(truthful "ordinary field galaxy" stems — no human-inspection claim). Uniqueness is asserted.

Run-3 augmentation is built in (--no-augment to skip): ONE deterministic dihedral copy per train
row, same target. DEVIATION from augment_sft_v5 (which transposed the saved PNGs): the DESI
'residual' view is a g|r|z montage built from a model fit, not a square per-pixel render, so we
transform the CUBE and re-render all views — panels keep their layout and the residual model
re-fits the transformed pixels. Random-field dilution needs no extra pass here: the random_neg
sample is already in the composition.

Outputs (finetune/corpus_desi/, gitignored):
  sft_desi_train.jsonl / sft_desi_val.jsonl  ms-swift SFT rows (val = 3% loss-monitor carve)
  sft_desi_train_aug.jsonl                   run-3 train file (originals + dihedral copies)
  valsel_desi.jsonl + valsel_desi_labels.csv swift-infer evalset for AUC checkpoint selection
  valsel_manifest.csv                        manifest for the merge->serve->grade selection path
  corpus_manifest.csv                        provenance (every corpus row + split/label)

  python -m lensjudge.finetune.build_corpus_desi build
  python -m lensjudge.finetune.build_corpus_desi label --split gate --out outputs/parity_gate_27b.parquet
  python -m lensjudge.finetune.build_corpus_desi gate  --preds outputs/parity_gate_27b.parquet
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
from lensjudge.common import fetch, render  # noqa: E402
from lensjudge.finetune.build_sft_data import _target_json  # noqa: E402
from lensjudge.finetune.distill_hsc import _hash01  # noqa: E402
from lensjudge.tools.cutout import VIEW_DESC  # noqa: E402

SEED = 2026
POOL = config.OUT / "parity_train_pool.csv"
OUT = Path(__file__).resolve().parent / "corpus_desi"
VIEWS = ("full", "zoom", "residual", "highcontrast")   # the standard DESI render.py views
N_RF_TRAIN = 2500          # random_neg dilution in train (valsel gets the proportional share)
VAL_FRAC = 0.03            # loss-monitor carve, matches distill_hsc sft-v5
CRIT_KEYS = ("blue_source", "low_surface_brightness", "curvature", "counter_images",
             "arc_morphology")
_LABEL = {"A": "lens", "B": "lens", "C": "softmid", "D": "nonlens"}

RUBRIC_V2 = (config.HERE / "prompts" / "rubric_imaging_v2.md").read_text()
# the DESI analogue of distill_hsc._DIRECT_NOTE: same rubric, views INLINE, no tool loop.
# residual description tracks the active residual convention (LENSJUDGE_RESIDUAL_VERSION).
_DIRECT_NOTE = f"""

# IMPORTANT — DESI Legacy Survey grz cutouts (0.262\"/px, 26.5\" field), rendered INLINE below (no tools)
- The four standard views are attached in order: full, zoom, residual, highcontrast. Do NOT call
  fetch_cutout or get_photometry — judge from these images alone. There are NO CNN scores.
- [full] {VIEW_DESC['full']}
- [zoom] {VIEW_DESC['zoom']}
- [residual] {render.residual_view_desc()}
- [highcontrast] {VIEW_DESC['highcontrast']}
- Apply the SAME A/B/C/D scale + v2 rubric (rule out LRG+companion / ring / spiral / merger via
  colour + radial geometry).
"""
DIRECT_SYS = RUBRIC_V2 + _DIRECT_NOTE
USER_MSG = ("<image><image><image><image>\nGrade this strong-lens candidate at DESI 0.262\"/px. "
            "Rendered grz views (full, zoom, residual, highcontrast). Respond with ONLY the JSON.")


def _desi_target(r) -> str:
    """Per-example UNIQUE ImageGrade target (v5 collapse lesson, distill_hsc._v5_target):
    class-constant templates let the student emit the class prior and ignore the pixels, so
    every numeric field carries deterministic hash jitter and every rationale carries row facts."""
    name = str(r.name)
    h = lambda s: _hash01(name, s)                      # noqa: E731
    if r.label_source == "random_neg":
        # augment_sft_v5._rf_target analogue: truthful, no human-inspection claim
        p = round(0.01 + h("p") * 0.03, 2)
        crit = {k: int(h(f"c{i}") * 2) for i, k in enumerate(CRIT_KEYS)}
        conf = round(0.75 + h("cf") * 0.2, 2)
        stems = [f"Ordinary field galaxy at ({r.ra:.4f},{r.dec:.4f}); no arc-like or lensed features.",
                 f"({r.ra:.4f},{r.dec:.4f}): unremarkable field scene, nothing consistent with lensing.",
                 f"Field position ({r.ra:.4f},{r.dec:.4f}) shows no deflector-arc geometry."]
        return _target_json("D", p_lens=p, rationale=stems[int(h('r') * len(stems))], crit=crit,
                            confidence=conf, contaminant=None)
    soft = float(r.soft_target)
    # p_lens: ALL pool soft targets are letter-grade constants -> de-constant (+-0.04, v5 rule)
    p = round(min(0.99, max(0.01, soft + (h("p") - 0.5) * 0.08)), 2)
    crit = {}
    for i, k in enumerate(CRIT_KEYS):
        base = 10.0 * soft * (0.75 + 0.5 * h(f"c{i}"))
        crit[k] = int(min(10, max(0, round(base + (h(f"o{i}") - 0.5) * 2))))
    conf = round(min(0.95, max(0.2, 0.4 + abs(soft - 0.5) + (h("cf") - 0.5) * 0.2)), 2)
    # short, fact-bearing, per-example rationale (coords + published CNN prior; phrasing by hash)
    fact_s = f"({r.ra:.4f},{r.dec:.4f})"
    if pd.notna(r.p_pub):
        fact_s += f" CNN p={float(r.p_pub):.2f}"
    lab = _LABEL[str(r.grade)]
    if lab == "lens":
        stems = [f"Human-graded {r.grade} DESI candidate {fact_s}: arc structure around the deflector.",
                 f"VI-campaign grade-{r.grade} candidate {fact_s}; lensing morphology present.",
                 f"{fact_s}: graded {r.grade} by the human inspectors; consistent arc geometry."]
    elif lab == "nonlens":
        stems = [f"Human-rejected candidate {fact_s}: arc-like feature is a contaminant.",
                 f"{fact_s}: graded D on visual inspection; no genuine lensing geometry.",
                 f"Rejected on inspection {fact_s}; feature not consistent with lensing."]
    else:
        stems = [f"Ambiguous grade-C candidate {fact_s}: suggestive but unconvincing features.",
                 f"{fact_s}: human grade C; lens-like hints below the confidence bar.",
                 f"Borderline candidate {fact_s}, graded C on visual inspection."]
    rationale = stems[int(h("r") * len(stems))]
    return _target_json(str(r.grade), p_lens=p, rationale=rationale, crit=crit, confidence=conf,
                        contaminant=("contaminant" if lab == "nonlens" else None))


def _dihedral_cube(cube: np.ndarray, t: int) -> np.ndarray:
    """Non-identity dihedral transform t in 0..6 applied to a (3,H,W) cube."""
    ops = (lambda c: c[:, :, ::-1], lambda c: c[:, ::-1, :],
           lambda c: np.rot90(c, 1, (1, 2)), lambda c: np.rot90(c, 2, (1, 2)),
           lambda c: np.rot90(c, 3, (1, 2)), lambda c: np.transpose(c, (0, 2, 1)),
           lambda c: np.rot90(np.transpose(c, (0, 2, 1)), 2, (1, 2)))
    return np.ascontiguousarray(ops[t](cube))


def _render_row(r, img_dir: Path, cube: np.ndarray | None = None, suffix: str = ""):
    """Fetch (cache-backed) + render the standard views; returns (cube, paths) or (None, None)."""
    if cube is None:
        cube = fetch.get_cube(name=str(r.name), ra=float(r.ra), dec=float(r.dec),
                              survey=str(r.survey_key))
    if cube is None:
        return None, None
    imgs = render.render_views(cube, views=VIEWS)
    if len(imgs) != len(VIEWS):
        return None, None
    safe = str(r.name).replace("/", "_").replace(" ", "_")
    paths = []
    for v in VIEWS:
        p = img_dir / f"{safe}_{v}{suffix}.png"
        if not p.exists():
            render.save_png(imgs[v], p)
        paths.append(str(p))
    return cube, paths


def _select(pool: pd.DataFrame, n_rf_train: int) -> pd.DataFrame:
    n_graded = {s: int(((pool.split == s) & (pool.label_source == "graded")).sum())
                for s in ("train", "valsel")}
    parts = []
    for split in ("train", "valsel"):
        s = pool[pool.split == split]
        n_rf = (n_rf_train if split == "train"
                else int(round(n_rf_train * n_graded["valsel"] / n_graded["train"])))
        parts.append(pd.concat([s[s.label_source.isin(["graded", "graded_D"])],
                                s[s.label_source == "random_neg"].sample(n_rf, random_state=SEED)]))
    sel = pd.concat(parts).reset_index(drop=True)
    sel["label"] = [("nonlens" if r.label_source == "random_neg" else _LABEL[str(r.grade)])
                    for r in sel.itertuples()]
    return sel


def _leakage_check(sel: pd.DataFrame):
    """Self-check THE RULE: no corpus row appears in either parity bench manifest, by name or
    within 2\" by position (the pool is firewalled upstream — assert it stayed that way)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    bench = pd.concat([pd.read_csv(config.OUT / f"parity_bench_arm{i}.csv")[["name", "ra", "dec"]]
                       for i in (1, 2)], ignore_index=True)
    hits = set(sel.name.astype(str)) & set(bench.name.astype(str))
    assert not hits, f"NAME leakage vs parity bench: {sorted(hits)[:5]}"
    idx, sep, _ = SkyCoord(sel.ra.values * u.deg, sel.dec.values * u.deg).match_to_catalog_sky(
        SkyCoord(bench.ra.values * u.deg, bench.dec.values * u.deg))
    n_pos = int((sep.arcsec < 2.0).sum())
    assert n_pos == 0, f"POSITION leakage: {n_pos} corpus rows within 2\" of a bench row"
    assert not (sel.get("split") == "gate").any(), "gate rows in the corpus"
    print(f"[leakage] {len(sel)} corpus rows vs {len(bench)} bench rows: "
          f"0 name hits, 0 within 2\", 0 gate rows -> PASS")


# ----------------------------------------------------------------------- build
def stage_build(args):
    pool = pd.read_csv(POOL, dtype={"grade": str})
    sel = _select(pool, args.n_rf_train)
    if args.limit:   # smoke-test only: first N rows per (split, label_source) after selection
        sel = sel.groupby(["split", "label_source"], group_keys=False).head(args.limit)
        sel = sel.reset_index(drop=True)
    _leakage_check(sel)
    print(f"[corpus] selected {len(sel)}: "
          f"{sel.groupby(['split', 'label_source']).size().to_dict()}")

    img_dir, aug_dir = OUT / "images", OUT / "images_aug"
    img_dir.mkdir(parents=True, exist_ok=True)

    recs = {"train": [], "valsel": []}
    vs_labels, miss, n_done = [], 0, 0
    for r in sel.itertuples():
        cube, paths = _render_row(r, img_dir)
        if paths is None:
            miss += 1
            continue
        if r.split == "train":
            recs["train"].append(
                {"messages": [{"role": "system", "content": DIRECT_SYS},
                              {"role": "user", "content": USER_MSG},
                              {"role": "assistant", "content": _desi_target(r)}],
                 "images": paths, "label": r.label, "name": str(r.name)})
        else:   # valsel: evalset row (NO assistant) + label row for AUC selection
            recs["valsel"].append(
                {"messages": [{"role": "system", "content": DIRECT_SYS},
                              {"role": "user", "content": USER_MSG}], "images": paths})
            vs_labels.append({"idx": len(vs_labels), "label": r.label, "name": str(r.name),
                              "grade": (None if r.label_source == "random_neg" else str(r.grade)),
                              "label_source": r.label_source})
        n_done += 1
        if n_done % 500 == 0:
            print(f"  rendered {n_done}/{len(sel)} (missing {miss})")

    # per-example-unique-target self-check (v5 standing rule: verify before every training run)
    targets = [x["messages"][2]["content"] for x in recs["train"]]
    assert len(set(targets)) == len(targets), "target collision — per-example-unique rule violated"
    print(f"[targets] {len(targets)} train targets, all unique -> PASS")

    rng = np.random.RandomState(SEED)
    rng.shuffle(recs["train"])
    n_val = int(len(recs["train"]) * args.val_frac)
    splits = {"sft_desi_val": recs["train"][:n_val], "sft_desi_train": recs["train"][n_val:]}
    for stem, data in splits.items():
        with open(OUT / f"{stem}.jsonl", "w") as fh:
            for rec in data:
                fh.write(json.dumps(rec) + "\n")
        counts = pd.Series([x["label"] for x in data]).value_counts().to_dict()
        print(f"[sft] {stem}: {len(data)} ({counts})")

    # run-3 dihedral augmentation: ONE hash-picked copy per train row, same target
    if not args.no_augment:
        aug_dir.mkdir(parents=True, exist_ok=True)
        by_name = {str(r.name): r for r in sel.itertuples()}
        out = list(splits["sft_desi_train"])
        for n, rec in enumerate(splits["sft_desi_train"]):
            stem0 = Path(rec["images"][0]).stem
            t = int(_hash01(stem0, "dihedral") * 7)
            r = by_name[rec["name"]]
            cube = fetch.get_cube(name=str(r.name), ra=float(r.ra), dec=float(r.dec),
                                  survey=str(r.survey_key))
            _, paths = _render_row(r, aug_dir, cube=_dihedral_cube(cube, t), suffix=f"_d{t}")
            out.append({**rec, "images": paths})
            if (n + 1) % 500 == 0:
                print(f"  dihedral {n + 1}/{len(splits['sft_desi_train'])}")
        rng2 = np.random.RandomState(SEED)
        rng2.shuffle(out)
        with open(OUT / "sft_desi_train_aug.jsonl", "w") as fh:
            for rec in out:
                fh.write(json.dumps(rec) + "\n")
        labs = pd.Series([x["label"] for x in out]).value_counts().to_dict()
        tgts = len({x["messages"][2]["content"] for x in out})
        print(f"[aug] sft_desi_train_aug: {len(out)} rows ({labs}) | unique targets {tgts}")

    with open(OUT / "valsel_desi.jsonl", "w") as fh:
        for rec in recs["valsel"]:
            fh.write(json.dumps(rec) + "\n")
    pd.DataFrame(vs_labels).to_csv(OUT / "valsel_desi_labels.csv", index=False)
    vs = sel[sel.split == "valsel"][["name", "ra", "dec", "survey_key", "grade",
                                     "label_source", "label"]]
    vs.to_csv(OUT / "valsel_manifest.csv", index=False)
    counts = pd.DataFrame(vs_labels).label.value_counts().to_dict()
    print(f"[valsel] {len(recs['valsel'])} evalset rows ({counts}) + labels + manifest")

    sel.to_csv(OUT / "corpus_manifest.csv", index=False)
    print(f"[corpus] {miss} skipped for missing cutout | manifest + jsonls -> {OUT}")
    if splits["sft_desi_train"]:
        s = json.dumps(splits["sft_desi_train"][0])
        print(f"[sample] {s[:220]} ... {s[-260:]}")


# ----------------------------------------------------------------------- label
def desi_content(r):
    """Inline DESI evidence blocks for the DIRECT grader — the eval-time counterpart of the
    corpus rows (same views, same system prompt; mirrors distill_hsc.hsc_content)."""
    cube = fetch.get_cube(name=str(r.name), ra=float(r.ra), dec=float(r.dec),
                          survey=str(r.survey_key))
    if cube is None:
        return None
    imgs = render.render_views(cube, views=VIEWS)
    if len(imgs) != len(VIEWS):
        return None
    content = [{"type": "text", "text":
                f"Grade this strong-lens candidate at DESI 0.262\"/px "
                f"(ra={float(r.ra):.6f}, dec={float(r.dec):.6f}). Rendered grz views:"}]
    for v, img in imgs.items():
        desc = render.residual_view_desc() if v == "residual" else VIEW_DESC[v]
        content.append({"type": "text", "text": f"[{v}] {desc}"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": render.png_b64(img)}})
    content.append({"type": "text", "text": "Respond with ONLY the JSON object for the required schema."})
    return content


async def _bounded(fns, n):
    sem = asyncio.Semaphore(n)

    async def run(f):
        async with sem:
            return await f()
    return await asyncio.gather(*[run(f) for f in fns])


def stage_label(args):
    """DIRECT-grade a manifest (or a pool split) with the corpus prompt; logprobs default ON."""
    from lensjudge.common import llm_client
    from lensjudge.imaging import grader_direct
    if args.manifest:
        man = pd.read_csv(args.manifest)
    else:
        pool = pd.read_csv(POOL, dtype={"grade": str})
        man = pool[pool.split == args.split].reset_index(drop=True)
    print(f"[label] {len(man)} objects ({args.manifest or f'pool split={args.split}'}) | "
          f"backend={llm_client.get_backend()} | model={args.model or config.MODELS['grader']}")

    async def grade_one(r):
        content = desi_content(r)
        if content is None:
            return {"name": str(r.name), "error": "no cutout"}
        g = await grader_direct.grade_candidate({"name": str(r.name)}, model=args.model,
                                                system_prompt=DIRECT_SYS, content=content)
        gp = g.meta.get("grade_probs") or {}
        return {"name": str(r.name),
                "p_lens": (g.grade.p_lens if g.grade else np.nan),
                "agent_grade": (g.grade.grade if g.grade else None),
                "p_lens_logprob": g.meta.get("p_lens_logprob"),
                **{f"gp_{k}": gp.get(k) for k in "ABCD"},
                "target_json": (g.grade.model_dump_json() if g.grade else None),
                "cost_usd": g.cost_usd, "error": g.error}
    recs = asyncio.run(_bounded([lambda r=r: grade_one(r) for r in man.itertuples()],
                                args.concurrency))
    df = pd.DataFrame(recs)
    df.to_parquet(args.out, index=False)
    ok = df[df.agent_grade.notna()]
    print(f"[label] parsed {len(ok)}/{len(df)} | grades {ok.agent_grade.value_counts().to_dict()} "
          f"-> {args.out}")


# ------------------------------------------------------------------------ gate
def stage_gate(args):
    """Grade-token-logprob metrics vs the pool's human labels (AUC on lens A/B vs nonlens
    D+random; softmid C rows are ambiguous by construction and excluded, per v5)."""
    from sklearn.metrics import roc_auc_score
    pool = pd.read_csv(POOL, dtype={"grade": str})
    truth = pool[pool.split == args.split].copy()
    truth["label"] = [("nonlens" if r.label_source == "random_neg" else _LABEL[str(r.grade)])
                      for r in truth.itertuples()]
    j = truth.merge(pd.read_parquet(args.preds), on="name")
    score = "p_lens_logprob" if j["p_lens_logprob"].notna().any() else "p_lens"
    pos, neg = j[j.label == "lens"], j[j.label == "nonlens"]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = pd.concat([pos[score], neg[score]]).astype(float).values
    auc = roc_auc_score(y[~np.isnan(s)], s[~np.isnan(s)])
    rec = pos.agent_grade.isin(["A", "B"]).mean()
    rej = (~neg.agent_grade.isin(["A", "B"])).mean()
    print(f"=== {args.split} split, scorer={score} ===")
    print(f"  n={len(j)} ({len(pos)} lens / {len(neg)} nonlens / {(j.label == 'softmid').sum()} "
          f"softmid excluded) | AUC={auc:.3f} | recovery(lens->A/B)={rec:.0%} | "
          f"rejection(non->not A/B)={rej:.0%}")
    for src, sub in j[j.label == "nonlens"].groupby("label_source"):
        print(f"  rejection[{src}]: {(~sub.agent_grade.isin(['A', 'B'])).mean():.0%} (n={len(sub)})")
    for g, sub in pos.groupby("grade"):
        print(f"  recovery[grade {g}]: {sub.agent_grade.isin(['A', 'B']).mean():.0%} (n={len(sub)})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n-rf-train", type=int, default=N_RF_TRAIN)
    b.add_argument("--val-frac", type=float, default=VAL_FRAC)
    b.add_argument("--no-augment", action="store_true")
    b.add_argument("--limit", type=int, default=0,
                   help="smoke test: keep only N rows per (split x label_source)")
    lb = sub.add_parser("label")
    lb.add_argument("--manifest", default=None, help="CSV with name,ra,dec,survey_key")
    lb.add_argument("--split", default="gate", choices=("gate", "valsel"),
                    help="pool split to grade when no --manifest is given")
    lb.add_argument("--out", required=True)
    lb.add_argument("--model", default=None)
    lb.add_argument("--concurrency", type=int, default=8)
    g = sub.add_parser("gate")
    g.add_argument("--preds", required=True)
    g.add_argument("--split", default="gate", choices=("gate", "valsel"))
    args = ap.parse_args()
    {"build": stage_build, "label": stage_label, "gate": stage_gate}[args.stage](args)


if __name__ == "__main__":
    main()
