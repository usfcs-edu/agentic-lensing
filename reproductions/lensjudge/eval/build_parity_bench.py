#!/usr/bin/env python3
"""Build the frozen, hash-pinned PARITY bench (Phase B of the human-parity program).

Two arms, one manifest each (see parity/FINDINGS.md for the design):

  arm1  paired human-vs-machine at DESI resolution: every graded candidate with a
        published follow-up outcome (parity/grade_purity_table.py ->
        outputs/parity_truth_master.csv). PRIMARY endpoint rows are the decided ones
        (confirmed_lens / non_lens); lens_z_only / inconclusive / pending rows are
        carried as secondary strata (truth_label empty). The human predictor is the
        published consensus grade; the machine predictor comes later (Phase C/D).

  arm2  machine-at-scale truth at DESI resolution: LS grz cutouts at the positions
        of the Euclid Q1 Strong Lensing Discovery Engine catalog (2,584 candidates,
        ~10 expert votes each; reproductions/euclid-q1/data/raw/). Truth tiers are
        attached when the per-object adjudications are staged
        (parity/data/external/euclid_q1_{nisp,modeling}.csv): 'spectroscopic'
        (NISP z-pairs) > 'modeling' (SLDE-A/B automated modeling) > ''.
        NOTE: no human-at-DESI-resolution comparator exists on this arm; it bounds
        the machine against truth and against the space-resolution expert panel.

  negative augmentation (arm1 FPR anchoring, non-truth): grade-D human rejects and
        random-galaxy negatives, flagged source='graded_D'/'random_neg' and NEVER
        counted in the primary endpoint.

Conventions follow eval/build_eval_set.py: seed 2026, sha256 pin written beside each
manifest, one row per unique candidate.

  python lensjudge/eval/build_parity_bench.py [--n-grade-d 250] [--n-random 500]
                                              [--check-cutouts 12]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import io  # noqa: E402

SEED = 2026
OUT = config.OUT if hasattr(config, "OUT") else Path(__file__).resolve().parents[1] / "outputs"
PARITY = Path(__file__).resolve().parents[1] / "parity"
EUCLID_CAT = config.REPRO / "euclid-q1" / "data" / "raw" / "q1_discovery_engine_lens_catalog.csv"

GMAP = {"A": 3, "B": 2, "C": 1, "D": 0}


def _pin(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    h = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]
    Path(str(path) + ".sha").write_text(h + "\n")
    return h


def build_arm1() -> pd.DataFrame:
    truth = pd.read_csv(OUT / "parity_truth_master.csv")
    master = pd.read_csv(OUT / "master_candidates.csv")
    m = truth.merge(master[["name", "ra", "dec", "source", "score"]],
                    left_on="cand_name", right_on="name", how="left", validate="1:1")
    missing = m.ra.isna().sum()
    if missing:
        print(f"  [arm1] WARNING {missing} truth rows failed the master join")
    lbl = m.outcome.map({"confirmed_lens": "lens", "non_lens": "nonlens"}).fillna("")
    rows = pd.DataFrame({
        "name": m.cand_name, "ra": m.ra, "dec": m.dec,
        "survey_key": m.source, "grade": m.grade,
        "human_ordinal": m.grade.map(GMAP),
        "truth_label": lbl, "outcome": m.outcome,
        "campaigns": m.campaigns, "arm": "arm1", "source": "truth_followup",
    })
    return rows.drop_duplicates("name").reset_index(drop=True)


def _load_euclid_truth() -> pd.DataFrame | None:
    """Per-object adjudications for arm2, best tier first.

    NOTE: parity/data/external/euclid_q1_nisp.csv is deliberately NOT consumed here.
    Its source (arXiv:2604.02726) is a single-author paper WITHDRAWN at v6, with
    385/440 'deflector redshifts' being photo-z and self-reported validation
    failures — not ground truth. The file is kept (rows carry the provenance
    warning) for reference only. Arm-2 truth therefore rests on the SLDE modeling
    tier until an Euclid-Collaboration spectroscopic adjudication is published.
    """
    frames = []
    for fname, tier in (("euclid_q1_modeling.csv", "modeling"),):
        p = PARITY / "data" / "external" / fname
        if p.exists():
            d = pd.read_csv(p)
            d["tier"] = tier
            frames.append(d[["name", "outcome", "tier", "detail"]])
            print(f"  [arm2] truth tier '{tier}': {len(d)} rows "
                  f"{d.outcome.value_counts().to_dict()}")
        else:
            print(f"  [arm2] truth tier '{tier}' not staged yet ({fname} missing)")
    if not frames:
        return None
    t = pd.concat(frames, ignore_index=True)
    # one adjudication per object: spectroscopic beats modeling; within a tier,
    # confirmed/non_lens beat softer outcomes
    orank = {"confirmed_lens": 0, "non_lens": 1, "lens_z_only": 2, "inconclusive": 3,
             "observed_pending": 4}
    trank = {"spectroscopic": 0, "modeling": 1}
    t["_o"], t["_t"] = t.outcome.map(orank), t.tier.map(trank)
    t = t.sort_values(["_t", "_o"]).groupby("name", as_index=False).first()
    return t.drop(columns=["_o", "_t"])


def build_arm2() -> pd.DataFrame:
    euc = pd.read_csv(EUCLID_CAT)
    rows = pd.DataFrame({
        "name": euc.id_str, "ra": euc.right_ascension, "dec": euc.declination,
        "survey_key": "ls-dr10",   # raw layer key for common.fetch.get_cube
        "grade": "", "human_ordinal": np.nan,
        "euclid_grade": euc.grade, "euclid_score": euc.expert_score,
        "euclid_votes": euc.expert_total_votes,
        "truth_label": "", "truth_tier": "", "outcome": "",
        "arm": "arm2", "source": "euclid_q1",
    })
    t = _load_euclid_truth()
    if t is not None:
        rows = rows.merge(t.rename(columns={"outcome": "adj_outcome"}), on="name", how="left")
        lbl = rows.adj_outcome.map({"confirmed_lens": "lens", "non_lens": "nonlens"}).fillna("")
        rows["truth_label"] = lbl
        rows["truth_tier"] = rows.pop("tier").fillna("")
        rows["outcome"] = rows.pop("adj_outcome").fillna("")
        rows = rows.drop(columns=["detail"], errors="ignore")
        # decided adjudications NOT in the discovery-engine catalog (e.g. SLDE-B's
        # pre-Q1 candidates): still truth-anchored positions inside the footprint —
        # append them with their own coordinates so arm2 keeps every decided object.
        extra_p = PARITY / "data" / "external" / "euclid_q1_modeling.csv"
        ext = pd.read_csv(extra_p)
        ext = ext[ext.outcome.isin(["confirmed_lens", "non_lens"])
                  & ~ext.name.isin(set(rows.name))]
        if len(ext):
            add = pd.DataFrame({
                "name": ext.name, "ra": ext.ra_deg, "dec": ext.dec_deg,
                "survey_key": "ls-dr10", "grade": "", "human_ordinal": np.nan,
                "euclid_grade": "", "euclid_score": np.nan, "euclid_votes": np.nan,
                "truth_label": ext.outcome.map({"confirmed_lens": "lens",
                                                "non_lens": "nonlens"}),
                "truth_tier": "modeling", "outcome": ext.outcome,
                "arm": "arm2", "source": "slde_b_extra",
            })
            print(f"  [arm2] +{len(add)} decided SLDE-B objects outside the Q1 catalog")
            rows = pd.concat([rows, add], ignore_index=True)
    return rows.drop_duplicates("name").reset_index(drop=True)


def build_negatives(n_grade_d: int, n_random: int) -> pd.DataFrame:
    rows = []
    gd = io.load_grade_d("both")
    gd = gd.sample(min(n_grade_d, len(gd)), random_state=SEED)
    for _, r in gd.iterrows():
        rows.append(dict(name=r["name"], ra=r["ra"], dec=r["dec"], survey_key=r["catalog"],
                         grade="D", human_ordinal=0, truth_label="", outcome="human_reject",
                         arm="arm1_aug", source="graded_D"))
    negp = config.INCH_DATA / "negatives_extra.parquet"
    if negp.exists() and n_random > 0:
        neg = pd.read_parquet(negp).sample(n_random, random_state=SEED)
        for _, r in neg.iterrows():
            rows.append(dict(name=str(r["row_id"]), ra=r.get("RA"), dec=r.get("DEC"),
                             survey_key="storfer", grade="", human_ordinal=np.nan,
                             truth_label="", outcome="random_field",
                             arm="arm1_aug", source="random_neg"))
    return pd.DataFrame(rows)


def check_cutouts(man: pd.DataFrame, n: int) -> None:
    from lensjudge.common import fetch  # late import: pulls requests
    sample = man[man.ra.notna()].sample(min(n, len(man)), random_state=SEED)
    ok = 0
    for _, r in sample.iterrows():
        cube = fetch.get_cube(name=str(r["name"]), ra=r["ra"], dec=r["dec"],
                              survey=str(r["survey_key"]))
        ok += cube is not None and cube.shape == (3, config.SIZE_PIX, config.SIZE_PIX)
    print(f"  [cutouts] {ok}/{len(sample)} sample rows resolve to a "
          f"(3,{config.SIZE_PIX},{config.SIZE_PIX}) grz cube")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grade-d", type=int, default=250)
    ap.add_argument("--n-random", type=int, default=500)
    ap.add_argument("--check-cutouts", type=int, default=0,
                    help="verify N random manifest rows fetch a grz cube")
    args = ap.parse_args()

    print("=== arm1: truth-anchored graded candidates (paired human-vs-machine) ===")
    a1 = build_arm1()
    dec = a1[a1.truth_label != ""]
    print(f"  {len(a1)} rows; decided {len(dec)} "
          f"(lens {(dec.truth_label == 'lens').sum()}, nonlens {(dec.truth_label == 'nonlens').sum()})")
    print(f"  by grade: {a1.grade.value_counts().to_dict()}")

    print("=== arm1 augmentation: non-truth negative strata ===")
    aug = build_negatives(args.n_grade_d, args.n_random)
    print(f"  {len(aug)} rows: {aug.source.value_counts().to_dict()}")

    print("=== arm2: Euclid Q1 positions at DESI resolution ===")
    a2 = build_arm2()
    t2 = a2[a2.truth_label != ""]
    print(f"  {len(a2)} rows; truth-labeled {len(t2)} "
          f"(lens {(t2.truth_label == 'lens').sum()}, nonlens {(t2.truth_label == 'nonlens').sum()}) "
          f"by tier {t2.truth_tier.value_counts().to_dict() if len(t2) else {}}")
    print(f"  euclid grades: {a2.euclid_grade.value_counts().to_dict()}")

    man1 = pd.concat([a1, aug], ignore_index=True)
    h1 = _pin(man1, OUT / "parity_bench_arm1.csv")
    h2 = _pin(a2, OUT / "parity_bench_arm2.csv")
    print(f"[manifest] arm1 {len(man1)} rows -> parity_bench_arm1.csv (sha {h1})")
    print(f"[manifest] arm2 {len(a2)} rows -> parity_bench_arm2.csv (sha {h2})")

    if args.check_cutouts:
        check_cutouts(man1, args.check_cutouts)
        check_cutouts(a2, args.check_cutouts)


if __name__ == "__main__":
    main()
