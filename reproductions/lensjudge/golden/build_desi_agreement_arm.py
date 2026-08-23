#!/usr/bin/env python3
"""golden/build_desi_agreement_arm.py — the DESI "golden-by-agreement" arm (zero PI time).

Paper II (Huang+ 2021, parity/data/vizier_huang2021_cand.csv) is the only table in the
program with a per-grader signal: `Score` is the mean of two integer 1-4 scores and `delSc`
their absolute difference, so `delSc == 0` rows are candidates where BOTH graders gave the
SAME score. Those rows are as close to a single-expert label as DESI resolution offers
without a new campaign: 726 rows = 100 A ({4,4}) + 165 B ({3,3}) + 461 C ({2,2}). Consensus
grade-D rejects (common/io.load_grade_d, Storfer 2024 + Inchausti 2025 lists) supply the
negatives, score 1 / letter D.

Caveats carried on every output (see the plan, Phase 5):
  * TRUNCATED AT ACCEPTANCE — the catalog holds Score >= 2.0 only; {1,1} agreements and all
    disagreeing pairs that averaged below 2.0 are absent, so D rows come from a different list.
  * AGREEMENT-FILTERED — these are the unambiguous cases by construction; any reliability or
    accuracy number on them is an upper bound on the full pool.
  * ASSUMES XH WAS ONE OF THE TWO PAPER II GRADERS (parity/fetch_vizier_paper2.py:5 says XH + AD);
    the pair is unordered, so "expert score" here means "the score both graders gave".

Firewall: THE RULE (parity/build_train_splits) — nothing that trains or calibrates a grader
may touch the parity bench. Every row is FLAGGED (`bench_overlap`, by name OR within 2" of
parity_bench_arm{1,2}) rather than dropped from the manifest, and carries the
parity_train_pool split when its name is in the pool. The few-shot pick and the SFT mix-in
EXCLUDE bench_overlap rows and pool_split == "gate" rows.

Outputs (golden/desi_agreement/, all pinned):
  agreement_manifest.csv      name,ra,dec,survey_key,score_1_4,grade_letter,label_source,
                              bench_overlap,pool_split
  fewshot_manifest_desi.csv   the grader_direct LENSJUDGE_FEWSHOT_MANIFEST columns (name,ra,dec,
                              survey_key,label,grade,note) + grade_source (WP-E header hook):
                              3 A / 3 B / 3 C / 6 D, deterministic hash pick
  sft_desi_agreement.jsonl    build_corpus_desi-shaped mix-in rows, ONLY for rows whose four
                              view PNGs already exist in finetune/corpus_desi/images/ (no network
                              by default; --render opts in to cache-backed fetch + render).
                              Written to finetune/corpus_golden/ (gitignored; ~9 KB per row, the
                              system prompt is repeated) — golden/ is tracked. Also excludes
                              pool_split in {gate, valsel}: the DESI valsel rows are the student
                              SELECTION set of corpus_desi and must never be trained on.

survey_key: Paper II names are NOT staged in any config.CUTOUT_DIRS (0/726 in the storfer or
inchausti dirs) and "huang2021" is not in config.SURVEY_LAYER, so fetch.get_cube would fall
through to the ls-dr10 endpoint. Paper II imaging was DECaLS DR8 (south); the nearest public
layer is ls-dr9 (eval/run_euclid maps huang2021 -> ls-dr9), so survey_key="ls-dr9" is
written here. GOTCHA: fetch.get_cube serves cache/cubes/<name>.fits by NAME regardless of
layer, and 73 of the 726 are already cached there as DR10 cubes (fetched by the parity bench
with survey_key="huang2021"); those rows would render DR10 pixels unless the cache is bypassed.

  python lensjudge/golden/build_desi_agreement_arm.py --out golden/desi_agreement/
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
from lensjudge.common import io  # noqa: E402
from lensjudge.golden import _util  # noqa: E402
from lensjudge.golden.build_corpus_golden import (  # noqa: E402
    _LABEL, BANNED_WORDS, golden_confidence, golden_criteria, golden_p)
from lensjudge.finetune.build_sft_data import _target_json  # noqa: E402
from lensjudge.finetune import build_corpus_desi as desi  # noqa: E402

VIZIER = config.HERE / "parity" / "data" / "vizier_huang2021_cand.csv"
LOCAL_CATALOG = config.REPRO / "huang-2021" / "data" / "huang2021_published_catalog.csv"
OUT = _util.HERE / "desi_agreement"
IMG_DIR = desi.OUT / "images"                  # finetune/corpus_desi/images
SURVEY_KEY_PAPER2 = "ls-dr9"                   # see module docstring
EXPECTED_COUNTS = {"A": 100, "B": 165, "C": 461}   # delSc==0 & pair_ok -> 726
FIREWALL_ARCSEC = 2.0
N_FEWSHOT = {"A": 3, "B": 3, "C": 3, "D": 6}
SFT_OUT = config.HERE / "finetune" / "corpus_golden" / "sft_desi_agreement.jsonl"
SFT_EXCLUDE_POOL_SPLITS = ("gate", "valsel")   # corpus_desi's frozen gate + selection rows
FEWSHOT_LABEL = {"A": "LENS", "B": "LENS", "C": "POSSIBLE LENS", "D": "NON-LENS"}
GRADE_SOURCE_AGREE = "expert (Paper II, both graders agreed)"
GRADE_SOURCE_D = "expert VI reject (Storfer 2024 / Inchausti 2025 grade-D lists)"
LABEL_AGREE, LABEL_D = "paper2_agree", "gradeD_consensus"
# agreement rows are unanimous -> H. Grade-D lists are consensus rejects with no recorded
# per-grader split; H keeps their p_lens (0.05 +- 0.02) on the corpus_desi graded_D scale.
CONF_AGREE, CONF_D = "H", "H"


# ------------------------------------------------------------------ sources
def select_agreement(viz: pd.DataFrame, expected: dict | None = EXPECTED_COUNTS) -> pd.DataFrame:
    """pair_ok & delSc == 0 -> name (VizieR Name), score_1_4 (= Score, integral by
    construction), grade_letter (= Q, checked against score_to_letter). Asserts the published
    counts when `expected` is given."""
    ok = viz["pair_ok"].astype(str).str.lower().isin(["true", "1"])
    sel = viz[ok & (pd.to_numeric(viz["delSc"], errors="coerce") == 0)].copy()
    score = pd.to_numeric(sel["Score"], errors="coerce")
    assert np.isclose(score, score.round()).all(), "delSc==0 rows must have integer Score"
    sel["score_1_4"] = score.round().astype(int)
    sel["grade_letter"] = sel["Q"].astype(str).str.strip().str.upper()
    letters = sel.score_1_4.map(_util.score_to_letter)
    assert (letters == sel.grade_letter).all(), "Q disagrees with score_to_letter(Score)"
    sel = sel.rename(columns={"Name": "name"})[["name", "score_1_4", "grade_letter"]]
    counts = sel.grade_letter.value_counts().to_dict()
    if expected is not None:
        assert len(sel) == sum(expected.values()) and counts == expected, \
            f"agreement counts {counts} (n={len(sel)}) != expected {expected}"
    print(f"[paper2] {len(sel)} agreement rows (delSc==0 & pair_ok): "
          + " / ".join(f"{counts.get(g, 0)} {g}" for g in ("C", "B", "A")))
    return sel.reset_index(drop=True)


def join_local_catalog(sel: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    """Reproduce fetch_vizier_paper2's name join to the local reproduction catalog for
    ra/dec (its RA/DEC columns; the VizieR _RA/_DE are the same 4-dp values)."""
    loc = local.rename(columns={"RA": "ra", "DEC": "dec"})[["name", "ra", "dec"]]
    j = sel.merge(loc, on="name", how="inner", validate="one_to_one")
    assert len(j) == len(sel), f"local-catalog join lost {len(sel) - len(j)} names"
    return j


def grade_d_rows() -> pd.DataFrame:
    """Consensus grade-D rejects from both catalogs (io.load_grade_d), score 1 / letter D,
    survey_key = their catalog key (storfer/inchausti -> ls-dr9/ls-dr10 via fetch)."""
    gd = io.load_grade_d("both").drop_duplicates("name").reset_index(drop=True)
    return pd.DataFrame({"name": gd.name.astype(str), "ra": gd.ra.astype(float),
                         "dec": gd.dec.astype(float), "survey_key": gd.catalog,
                         "score_1_4": 1, "grade_letter": "D", "label_source": LABEL_D})


# ----------------------------------------------------------------- firewall
def flag_bench_overlap(df: pd.DataFrame, bench: pd.DataFrame,
                       radius_arcsec: float = FIREWALL_ARCSEC) -> pd.Series:
    """True where a row shares a name with, or lies within `radius_arcsec` of, any bench row."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    by_name = df.name.astype(str).isin(set(bench.name.astype(str)))
    by_pos = pd.Series(False, index=df.index)
    bpos = bench.dropna(subset=["ra", "dec"])
    if len(bpos) and len(df):
        _, sep, _ = SkyCoord(df.ra.values * u.deg, df.dec.values * u.deg).match_to_catalog_sky(
            SkyCoord(bpos.ra.values * u.deg, bpos.dec.values * u.deg))
        by_pos = pd.Series(sep.arcsec < radius_arcsec, index=df.index)
    return by_name | by_pos


def carry_pool_split(df: pd.DataFrame, pool: pd.DataFrame) -> pd.Series:
    """parity_train_pool split for rows whose name is in the pool, '' otherwise."""
    p = pool.drop_duplicates("name")
    m = pd.Series(p.split.astype(str).values, index=p.name.astype(str))
    return df.name.astype(str).map(m).fillna("")


# ------------------------------------------------------------------ fewshot
def eligible(man: pd.DataFrame) -> pd.DataFrame:
    return man[~man.bench_overlap.astype(bool) & (man.pool_split.astype(str) != "gate")]


def pick_fewshot(man: pd.DataFrame, n_per: dict = N_FEWSHOT) -> pd.DataFrame:
    """Deterministic exemplar pick: per letter, the first n rows ordered by
    hash01(name, "fewshot") (every agreement row is unanimous, so the hash IS the tiebreak);
    independent of input row order. Excludes bench_overlap and pool_split == 'gate'."""
    el = eligible(man).copy()
    el["_h"] = [_util.hash01(str(n), "fewshot") for n in el.name]
    parts = []
    for letter, n in n_per.items():
        sub = el[el.grade_letter == letter].sort_values(["_h", "name"]).head(n)
        if len(sub) < n:
            print(f"[fewshot] only {len(sub)}/{n} eligible {letter} rows")
        parts.append(sub)
    pick = pd.concat(parts).drop(columns="_h").reset_index(drop=True)
    return pd.DataFrame({
        "name": pick.name, "ra": pick.ra, "dec": pick.dec, "survey_key": pick.survey_key,
        "label": pick.grade_letter.map(FEWSHOT_LABEL), "grade": pick.grade_letter, "note": "",
        "grade_source": np.where(pick.label_source == LABEL_AGREE, GRADE_SOURCE_AGREE,
                                 GRADE_SOURCE_D),
    })


# ---------------------------------------------------------------------- sft
def agreement_rationale(name: str, ra: float, dec: float, score: int, letter: str,
                        label_source: str) -> str:
    """Truthful per-example fact stem (3 hash-selected phrasings; no committee/consensus word)."""
    coord = f"({float(ra):.4f},{float(dec):.4f})"
    if label_source == LABEL_AGREE:
        stems = [f"{coord} DESI grz: Paper II expert score {score}/4 ({letter}); both graders gave this score.",
                 f"Paper II candidate at {coord}: two independent expert scores of {score}/4 ({letter}).",
                 f"{coord}: scored {score}/4 ({letter}) by both Paper II inspectors."]
    else:
        stems = [f"{coord} DESI grz: rejected on visual inspection (grade D).",
                 f"Human-rejected candidate at {coord}; no convincing lensing geometry on inspection.",
                 f"{coord}: grade-D reject from the VI campaign; arc-like feature not accepted."]
    out = stems[int(_util.hash01(name, "r") * len(stems))]
    assert not any(w.lower() in out.lower() for w in BANNED_WORDS), out
    return out


def agreement_target(r) -> str:
    """ImageGrade JSON through the one funnel (_target_json) with the golden p formula;
    confidence H for unanimous rows (CONF_AGREE) and CONF_D for grade-D rejects. contaminant
    follows the corpus_desi convention for nonlens rows (this file is a corpus_desi mix-in)."""
    name, g = str(r.name), str(r.grade_letter)
    lmh = CONF_AGREE if r.label_source == LABEL_AGREE else CONF_D
    return _target_json(g, p_lens=golden_p(g, lmh, name),
                        rationale=agreement_rationale(name, r.ra, r.dec, int(r.score_1_4), g,
                                                      str(r.label_source)),
                        crit=golden_criteria(g, name), confidence=golden_confidence(lmh, name),
                        contaminant=("contaminant" if _LABEL[g] == "nonlens" else None))


def image_paths(name: str, img_dir: Path) -> list[str] | None:
    """The corpus_desi view PNGs for a name (its naming: <safe>_{view}.png), or None if any
    is missing."""
    safe = str(name).replace("/", "_").replace(" ", "_")
    paths = [Path(img_dir) / f"{safe}_{v}.png" for v in desi.VIEWS]
    return [str(p) for p in paths] if all(p.exists() for p in paths) else None


def build_sft(man: pd.DataFrame, img_dir: Path, render: bool = False) -> tuple[list, dict]:
    """corpus_desi-shaped rows for eligible manifest rows (not bench, not gate/valsel in the
    DESI pool) with images. Returns (recs, stats)."""
    recs, n_img = [], {LABEL_AGREE: 0, LABEL_D: 0}
    el = eligible(man)
    el = el[~el.pool_split.astype(str).isin(SFT_EXCLUDE_POOL_SPLITS)]
    for r in el.itertuples():
        paths = image_paths(r.name, img_dir)
        if paths is None and render:
            _, paths = desi._render_row(r, Path(img_dir))      # cache-backed fetch + render (network)
        if paths is None:
            continue
        n_img[str(r.label_source)] += 1
        recs.append({"messages": [{"role": "system", "content": desi.DIRECT_SYS},
                                  {"role": "user", "content": desi.USER_MSG},
                                  {"role": "assistant", "content": agreement_target(r)}],
                     "images": paths, "label": _LABEL[str(r.grade_letter)], "name": str(r.name)})
    targets = [x["messages"][2]["content"] for x in recs]
    assert len(set(targets)) == len(targets), "target collision — per-example-unique rule violated"
    stats = {"eligible": int(len(el)), "with_images": n_img, "rows": len(recs),
             "pool_split": el[el.name.isin({r["name"] for r in recs})]
             .pool_split.astype(str).replace("", "none").value_counts().to_dict()}
    return recs, stats


# --------------------------------------------------------------------- main
def build_manifest(viz: pd.DataFrame, local: pd.DataFrame, gd: pd.DataFrame,
                   bench: pd.DataFrame, pool: pd.DataFrame,
                   expected: dict | None = EXPECTED_COUNTS) -> pd.DataFrame:
    """agreement rows (survey_key ls-dr9, paper2_agree) + grade-D rows, flagged vs the bench and
    joined to the pool split. `expected` = the published agreement counts to assert."""
    agree = join_local_catalog(select_agreement(viz, expected), local)
    agree["survey_key"] = SURVEY_KEY_PAPER2
    agree["label_source"] = LABEL_AGREE
    man = pd.concat([agree, gd], ignore_index=True)
    dup = man.name.duplicated()
    if dup.any():          # a Paper II name that also sits on a grade-D list keeps its Paper II row
        print(f"[manifest] dropping {int(dup.sum())} grade-D names duplicated in Paper II")
        man = man[~dup]
    man["bench_overlap"] = flag_bench_overlap(man, bench).values
    man["pool_split"] = carry_pool_split(man, pool).values
    cols = ["name", "ra", "dec", "survey_key", "score_1_4", "grade_letter", "label_source",
            "bench_overlap", "pool_split"]
    return man[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--vizier", default=str(VIZIER))
    ap.add_argument("--local-catalog", default=str(LOCAL_CATALOG))
    ap.add_argument("--img-dir", default=str(IMG_DIR), help="corpus_desi view PNGs")
    ap.add_argument("--sft-out", default=str(SFT_OUT),
                    help="mix-in jsonl (default under the gitignored finetune/corpus_golden/)")
    ap.add_argument("--render", action="store_true",
                    help="NETWORK: fetch+render missing views into --img-dir (default: off)")
    args = ap.parse_args()
    out = Path(args.out)

    viz = pd.read_csv(args.vizier)
    local = pd.read_csv(args.local_catalog)
    bench = pd.concat([pd.read_csv(config.OUT / f"parity_bench_arm{i}.csv")[["name", "ra", "dec"]]
                       for i in (1, 2)], ignore_index=True)
    pool = pd.read_csv(config.OUT / "parity_train_pool.csv", dtype={"grade": str})
    man = build_manifest(viz, local, grade_d_rows(), bench, pool)
    print(f"[manifest] {len(man)} rows: {man.label_source.value_counts().to_dict()} | "
          f"bench_overlap {int(man.bench_overlap.sum())} | pool_split "
          f"{man.pool_split.replace('', 'none').value_counts().to_dict()}")
    sha = _util.pin(man, out / "agreement_manifest.csv")
    print(f"[manifest] -> {out / 'agreement_manifest.csv'} (sha {sha})")

    fs = pick_fewshot(man)
    sha = _util.pin(fs, out / "fewshot_manifest_desi.csv")
    print(f"[fewshot] {len(fs)} exemplars {fs.grade.value_counts().to_dict()} -> "
          f"{out / 'fewshot_manifest_desi.csv'} (sha {sha}); use via LENSJUDGE_FEWSHOT_MANIFEST")

    recs, stats = build_sft(man, Path(args.img_dir), render=args.render)
    sft_out = Path(args.sft_out)
    sft_out.parent.mkdir(parents=True, exist_ok=True)
    with open(sft_out, "w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")
    print(f"[sft] {stats['eligible']} eligible rows (not bench_overlap, pool_split not in "
          f"{SFT_EXCLUDE_POOL_SPLITS}); with images: {stats['with_images']} -> {stats['rows']} "
          f"rows (DESI pool split {stats['pool_split']}) in {sft_out}")
    if stats["with_images"][LABEL_AGREE] == 0:
        print("[sft] WARNING: no Paper II agreement row has corpus_desi views — the mix-in is "
              "grade-D only until the 726 are rendered (--render, network)")


if __name__ == "__main__":
    main()
