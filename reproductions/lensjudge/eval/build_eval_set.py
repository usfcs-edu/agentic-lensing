#!/usr/bin/env python3
"""Build a frozen, hash-pinned LensBench-VI manifest (leak-aware, stratified).

Assembles the evaluation set from on-disk data into one manifest CSV that the
grader consumes and the scorer reads:

  positives : graded A/B/C candidates (Storfer DR9 + Inchausti DR10), cutouts on disk
  negatives : Grade-D human-rejects (the hard negatives; cutouts fetched on demand)
              + a sample of random-galaxy negatives (cutouts_fits_neg_dr9, on disk)
  gold      : Foundry-II confirmed (blind tier) — recorded, flagged source='gold'

Stratified by (grade x region) with a fixed seed so the split is reproducible; a
content hash of the manifest is written alongside. NOTE on leakage: the LLM grader
never trained on any cutout, so all rows are valid held-out for it; the leak flag
(NeuraLens training positives) matters only for the frozen-CNN p_meta baseline, and
is recorded per row for that caveat.

  python lensjudge/eval/build_eval_set.py --n-graded 150 --n-grade-d 60 --n-random 60
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import io  # noqa: E402

SEED = 2026


def _strat_sample(df, n, by, seed=SEED):
    if n <= 0 or len(df) <= n:
        return df.copy()
    frac = n / len(df)
    parts = [g.sample(max(1, round(len(g) * frac)), random_state=seed) for _, g in df.groupby(by)]
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n, random_state=seed).reset_index(drop=True)
    return out


def build(n_graded, n_grade_d, n_random, which="both", n_mimic=0):
    rows = []

    # positives — graded A/B/C, stratified by grade x region
    cand = io.load_candidates(which)
    cand["stratum"] = cand["grade"].astype(str) + "|" + cand["region"].astype(str)
    pos = _strat_sample(cand, n_graded, "stratum")
    for _, r in pos.iterrows():
        rows.append(dict(name=r["name"], ra=r["ra"], dec=r["dec"], survey_key=r["catalog"],
                         grade_truth=r["grade"], binary_label="lens", source="graded",
                         region=r["region"], tractor_type=r["tractor_type"],
                         p_meta=r["p_meta"], leak="unknown"))

    # hard negatives — Grade-D human-rejects
    gd = io.load_grade_d(which)
    gd = _strat_sample(gd, n_grade_d, "catalog")
    for _, r in gd.iterrows():
        rows.append(dict(name=r["name"], ra=r["ra"], dec=r["dec"], survey_key=r["catalog"],
                         grade_truth="D", binary_label="nonlens", source="graded_D",
                         region=r["survey"], tractor_type="?", p_meta=r.get("p_meta"),
                         leak="no"))

    # random-galaxy negatives — on disk (cutouts_fits_neg_dr9)
    negp = config.INCH_DATA / "negatives_extra.parquet"
    if negp.exists() and n_random > 0:
        neg = pd.read_parquet(negp).sample(min(n_random, len(pd.read_parquet(negp))),
                                           random_state=SEED)
        for _, r in neg.iterrows():
            rows.append(dict(name=str(r["row_id"]), ra=r.get("RA"), dec=r.get("DEC"),
                             survey_key="storfer", grade_truth="D", binary_label="nonlens",
                             source="random_neg", region=r.get("footprint", "?"),
                             tractor_type="?", p_meta=None, leak="no"))

    # lens-MIMIC negatives — the 601 ClaudeNet-v3 campaign rejects (CNN-high, dual-agent-
    # confirmed non-lenses), typed by contaminant. Cutouts resolve via CUTOUT_DIRS['claudenet'].
    # This makes Benchmark B = lens-vs-MIMIC (the v3 thesis), not just lens-vs-random.
    mimp = config.REPRO / "claudenet" / "data" / "v3" / "mimic_bank_seed.parquet"
    if mimp.exists() and n_mimic > 0:
        mm = pd.read_parquet(mimp)
        mm["mt"] = mm["mimic_type"].astype(str)
        mm = _strat_sample(mm, n_mimic, "mt")
        for _, r in mm.iterrows():
            rows.append(dict(name=str(r["row_id"]), ra=r.get("RA"), dec=r.get("DEC"),
                             survey_key="claudenet", grade_truth="D", binary_label="nonlens",
                             source="mimic_neg", region="dr9_sweep",
                             tractor_type=str(r.get("mimic_type", "?")),
                             p_meta=r.get("p_final"), leak="no"))

    # gold (blind) — Foundry-II spectroscopically confirmed (20) + known (1) as lens,
    # + the 4 confirmed NON-lenses; the 48 pending systems are EXCLUDED (not ground truth).
    g = io.load_foundry_ii_gold()
    if len(g) and "section" in g.columns:
        lens_sec, non_sec = {"confirmed", "known"}, {"nonlens"}
        gg = g[g["section"].isin(lens_sec | non_sec)]
        for _, r in gg.iterrows():
            is_non = r["section"] in non_sec
            rows.append(dict(name=str(r["name"]).replace(" ", "_"),
                             ra=r["ra_deg"], dec=r["dec_deg"], survey_key="inchausti",
                             grade_truth=("D" if is_non else "A"),
                             binary_label=("nonlens" if is_non else "lens"),
                             source="gold", region="foundry_ii", tractor_type="?",
                             p_meta=None, leak="no"))

    man = pd.DataFrame(rows).drop_duplicates("name").reset_index(drop=True)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-graded", type=int, default=150)
    ap.add_argument("--n-grade-d", type=int, default=60)
    ap.add_argument("--n-random", type=int, default=60)
    ap.add_argument("--n-mimic", type=int, default=0,
                    help="ClaudeNet-v3 lens-mimic negatives (Benchmark B = lens-vs-mimic)")
    ap.add_argument("--which", default="both")
    ap.add_argument("--out", default=str(config.OUT / "lensbench_manifest.csv"))
    args = ap.parse_args()
    man = build(args.n_graded, args.n_grade_d, args.n_random, args.which, args.n_mimic)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(args.out, index=False)
    h = hashlib.sha256(man.to_csv(index=False).encode()).hexdigest()[:16]
    Path(args.out + ".sha").write_text(h + "\n")
    print(f"[manifest] {len(man)} rows -> {args.out} (sha {h})")
    print("  by source:", man["source"].value_counts().to_dict())
    print("  by grade_truth:", man["grade_truth"].value_counts().to_dict())
    print("  by binary:", man["binary_label"].value_counts().to_dict())
    print("  by region:", man["region"].value_counts().to_dict())


if __name__ == "__main__":
    main()
