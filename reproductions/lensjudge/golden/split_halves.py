#!/usr/bin/env python3
"""golden/split_halves.py — the pre-registered align / validate halves of the golden frame.

The golden set serves two uses that must never touch: `align` (few-shot exemplars, SFT rows,
threshold calibration, rubric tuning) and `validate` (scored once per registered arm). The
halves are drawn AFTER pass-1 grading so they can be matched on what Xiaosheng actually said,
not on the pipeline's opinion of the items:

  unit of assignment   system_id (10" union-find in frame.csv): overlapping cutouts share
                       pixels, so they move together (ranks 16/17 at 8.78")
  forced               prior_exposure == 2 (ranks 1-15, seen with grade + literature in the
                       annotated docx) -> `align`, before anything random happens; this is
                       the one non-random assignment and is flagged `forced=True`
  stratified           (stratum x grade_letter) cells, 50/50, seed 2026: strict alternation
                       inside every cell in a seeded order, then the number of score >= 3
                       units is matched across halves to within 1 (his "matched lens counts
                       per half" ask) by moving single >= 3 units out of the heavy half, one
                       per most-heavy-leaning cell, so the compensation for the forced block
                       is spread thinly rather than hollowing out one cell. Deterministic.
  firewall             no unit within 2" of a unit in the other half (impossible by
                       construction at 10"; asserted anyway, build_train_splits.apply_firewall
                       pattern), and every 2" overlap with outputs/parity_train_pool.csv and
                       parity_bench_arm{1,2}.csv is REPORTED per half, not excluded: the
                       fixed strata (T/K/L) keep their DESI-side twins (365 of 371 overlapping
                       targets are `random_neg` in the DESI pool -- a DESI-trained student
                       was taught they are non-lenses), and registry.leak carries the flag.

  python lensjudge/golden/split_halves.py --labels golden/golden_labels.csv \
      --frame golden/frame.csv --seed 2026 --out golden/splits.csv

Output golden/splits.csv (+ .sha): unit_id, system_id, split, forced, stratum, grade_letter;
outputs/golden_splits_desi_overlap.csv: the reported overlaps. Deterministic under the seed.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge.golden import _util  # noqa: E402

GOLDEN = _util.HERE
OUT = _util.LENSJUDGE / "outputs"
HALVES = ("align", "validate")
LENS_MIN_SCORE = 3
EXCL_RADIUS_ARCSEC = 2.0
POOL_FILES = ("parity_train_pool.csv", "parity_bench_arm1.csv", "parity_bench_arm2.csv")
SPLIT_COLS = ["unit_id", "system_id", "split", "forced", "stratum", "grade_letter"]


def _units(labels: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """One row per graded unit with everything the assignment needs."""
    fcols = ["unit_id", "system_id", "prior_exposure"]
    for c in ("ra_deg", "dec_deg", "stratum", "candidate_id"):
        if c in frame.columns and c not in labels.columns:
            fcols.append(c)
    df = labels.merge(frame[fcols], on="unit_id", how="left", validate="one_to_one")
    missing = df["system_id"].isna()
    assert not missing.any(), f"units absent from frame: {df.loc[missing, 'unit_id'].tolist()}"
    df["system_id"] = df["system_id"].astype(int)
    df["prior_exposure"] = df["prior_exposure"].fillna(0).astype(int)
    df["score_1_4"] = df["score_1_4"].astype(int)
    if "grade_letter" not in df.columns:
        df["grade_letter"] = df["score_1_4"].map(_util.score_to_letter)
    df["ge3"] = (df["score_1_4"] >= LENS_MIN_SCORE).astype(int)
    df["forced"] = df["prior_exposure"] == 2
    return df.sort_values("unit_id").reset_index(drop=True)


def assign_halves(labels: pd.DataFrame, frame: pd.DataFrame, seed: int = _util.SEED) -> pd.DataFrame:
    """labels (pass-1 canonical) + frame -> splits DataFrame (SPLIT_COLS).

    Phase 0  forced systems (any unit prior_exposure == 2) -> align.
    Phase 1  per (stratum x letter) cell, seeded order, strict alternation; the first of each
             cell goes to the half behind on stratum count, then total, then a coin.
    Phase 2  while the score >= 3 counts differ by more than 1, move ONE single-unit,
             non-forced >= 3 system from the heavy half to the light one, picking the cell
             (then stratum) where the heavy half currently leads most, so the compensation
             for the forced block is spread across cells instead of landing in one."""
    df = _units(labels, frame)
    rng = np.random.default_rng(seed)

    groups = []
    for sid, g in df.groupby("system_id", sort=True):
        cells = list(zip(g["stratum"], g["grade_letter"]))
        groups.append({"system_id": int(sid), "units": g["unit_id"].tolist(), "cells": cells,
                       "cell": cells[0], "strata": g["stratum"].tolist(),
                       "n_ge3": int(g["ge3"].sum()), "forced": bool(g["forced"].any()),
                       "split": None, "moved": False})
    for k, i in enumerate(rng.permutation(len(groups))):   # one seeded rank per system
        groups[i]["rank"] = int(k)
    tally = {h: {"ge3": 0, "total": 0, "cell": defaultdict(int), "stratum": defaultdict(int)}
             for h in HALVES}

    def _add(grp, h, sign=+1):
        t = tally[h]
        t["ge3"] += sign * grp["n_ge3"]
        t["total"] += sign * len(grp["units"])
        for cell, st in zip(grp["cells"], grp["strata"]):
            t["cell"][cell] += sign
            t["stratum"][st] += sign
        grp["split"] = h if sign > 0 else None

    for grp in groups:                                      # phase 0
        if grp["forced"]:
            _add(grp, "align")

    free = [grp for grp in groups if not grp["forced"]]
    by_cell = defaultdict(list)
    for grp in free:
        by_cell[grp["cell"]].append(grp)
    for cell in sorted(by_cell):                            # phase 1
        for grp in sorted(by_cell[cell], key=lambda x: x["rank"]):
            def _behind(h):
                t = tally[h]
                return (t["cell"][cell], sum(t["stratum"][s] for s in grp["strata"]), t["total"])
            sc = {h: _behind(h) for h in HALVES}
            h = (HALVES[int(rng.integers(0, 2))] if sc["align"] == sc["validate"]
                 else min(HALVES, key=lambda x: sc[x]))
            _add(grp, h)

    def _diff():
        return tally["align"]["ge3"] - tally["validate"]["ge3"]
    while abs(_diff()) > 1:                                 # phase 2
        heavy, light = ("align", "validate") if _diff() > 0 else ("validate", "align")
        cands = [grp for grp in free if grp["split"] == heavy and grp["n_ge3"] == 1
                 and len(grp["units"]) == 1 and not grp["moved"]]
        assert cands, (f"score>=3 counts cannot be matched: align {tally['align']['ge3']} vs validate "
                       f"{tally['validate']['ge3']} and no movable >=3 unit left in {heavy} "
                       f"(forced >=3 in align = {sum(g['n_ge3'] for g in groups if g['forced'])})")
        th, tl = tally[heavy], tally[light]
        grp = max(cands, key=lambda g: (th["cell"][g["cell"]] - tl["cell"][g["cell"]],
                                        th["stratum"][g["strata"][0]] - tl["stratum"][g["strata"][0]],
                                        -g["rank"]))
        _add(grp, heavy, -1)
        _add(grp, light, +1)
        grp["moved"] = True

    rows = [{"unit_id": u, "system_id": grp["system_id"], "split": grp["split"], "forced": grp["forced"]}
            for grp in groups for u in grp["units"]]
    splits = (pd.DataFrame(rows).merge(df[["unit_id", "stratum", "grade_letter"]], on="unit_id")
              .sort_values("unit_id").reset_index(drop=True))[SPLIT_COLS]

    # invariants (the plan's "A/B counts matched, ranks 1-15 in align, dup pairs together")
    d = splits.merge(df[["unit_id", "ge3", "forced"]], on="unit_id", suffixes=("", "_u"))
    assert splits["split"].isin(HALVES).all()
    assert (d.loc[d["forced_u"], "split"] == "align").all(), "forced units must be in align"
    assert (d.groupby("system_id")["split"].nunique() == 1).all(), "a system straddles the halves"
    ge3 = d.groupby("split")["ge3"].sum().reindex(HALVES, fill_value=0)
    assert abs(int(ge3["align"]) - int(ge3["validate"])) <= 1, ge3.to_dict()
    return splits



# --------------------------------------------------------------------------- firewall
def _skycoord(ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    return SkyCoord(np.asarray(ra, float) * u.deg, np.asarray(dec, float) * u.deg)


def firewall(splits: pd.DataFrame, coords: pd.DataFrame, pool_dir: Path = OUT,
             radius_arcsec: float = EXCL_RADIUS_ARCSEC, halves: tuple = HALVES) -> pd.DataFrame:
    """(i) assert no unit of halves[0] within `radius` of a unit of halves[1] (n_bad == 0);
    (ii) report every golden unit within `radius` of a parity pool / bench row, per half.
    `coords`: unit_id, ra_deg, dec_deg (+ candidate_id). `halves` names the two split labels
    (default align/validate; the truth split passes ("design", "holdout") — with any other
    labels both selections would be empty and the assertion vacuous). Every `split` value
    must be one of the two. Returns the overlap table."""
    assert len(halves) == 2, f"halves must name two labels, got {halves!r}"
    s = splits.merge(coords[[c for c in ("unit_id", "ra_deg", "dec_deg", "candidate_id")
                             if c in coords.columns]], on="unit_id", how="left")
    ok = s["ra_deg"].notna() & s["dec_deg"].notna()
    assert ok.all(), f"units without coordinates: {s.loc[~ok, 'unit_id'].tolist()}"
    stray = sorted(set(s["split"].astype(str)) - set(halves))
    assert not stray, f"split labels {stray} are not in halves={halves!r}: the firewall would not test them"
    a = s[s["split"] == halves[0]]
    v = s[s["split"] == halves[1]]
    if len(a) and len(v):
        _, sep, _ = _skycoord(a["ra_deg"], a["dec_deg"]).match_to_catalog_sky(
            _skycoord(v["ra_deg"], v["dec_deg"]))
        n_bad = int((sep.arcsec < radius_arcsec).sum())
        assert n_bad == 0, f"leak audit FAILED: {n_bad} {halves[0]} units within {radius_arcsec}\" of {halves[1]}"
    print(f"leak audit: 0 position collisions between halves (<{radius_arcsec}\")")

    gc = _skycoord(s["ra_deg"], s["dec_deg"])
    names = set(s["candidate_id"].astype(str)) if "candidate_id" in s.columns else set()
    rows = []
    for fn in POOL_FILES:
        p = Path(pool_dir) / fn
        if not p.exists():
            print(f"firewall: {p} absent - overlap against it not checked")
            continue
        pool = pd.read_csv(p, low_memory=False).dropna(subset=["ra", "dec"]).reset_index(drop=True)
        if names:   # name collision is impossible across id systems, but cheap to assert
            assert not pool["name"].astype(str).isin(names).any(), f"golden candidate_id present in {fn}"
        idx, sep, _ = gc.match_to_catalog_sky(_skycoord(pool["ra"], pool["dec"]))
        hit = sep.arcsec < radius_arcsec
        for i in np.where(hit)[0]:
            pr = pool.iloc[idx[i]]
            rows.append({"unit_id": s["unit_id"].iloc[i], "split": s["split"].iloc[i],
                         "pool_file": fn, "pool_name": str(pr["name"]),
                         "pool_split": str(pr.get("split", "")) if "split" in pool.columns else "",
                         "pool_label_source": str(pr.get("label_source", "")) if "label_source" in pool.columns else "",
                         "sep_arcsec": round(float(sep.arcsec[i]), 3)})
    ov = pd.DataFrame(rows, columns=["unit_id", "split", "pool_file", "pool_name", "pool_split",
                                     "pool_label_source", "sep_arcsec"])
    for h in halves:
        sub = ov[ov["split"] == h]
        print(f"firewall: {h:<8s} {sub['unit_id'].nunique():3d} units within {radius_arcsec}\" of a DESI "
              f"pool/bench row (reported, not excluded)"
              + (f"  by file: {sub.groupby('pool_file')['unit_id'].nunique().to_dict()}" if len(sub) else ""))
    return ov


def summarize(splits: pd.DataFrame, labels: pd.DataFrame) -> str:
    d = splits.merge(labels[["unit_id", "score_1_4"]], on="unit_id")
    d["ge3"] = (d["score_1_4"].astype(int) >= LENS_MIN_SCORE).astype(int)
    lines = [f"units: {len(d)}  systems: {d['system_id'].nunique()}  forced: {int(d['forced'].sum())}",
             "per half: " + d.groupby("split").agg(n=("unit_id", "size"), n_ge3=("ge3", "sum"),
                                                     forced=("forced", "sum")).to_string().replace("\n", "\n          "),
             "stratum x half:\n" + pd.crosstab(d["stratum"], d["split"]).to_string(),
             "letter x half:\n" + pd.crosstab(d["grade_letter"], d["split"]).to_string()]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", default=str(GOLDEN / "golden_labels.csv"))
    ap.add_argument("--frame", default=str(GOLDEN / "frame.csv"))
    ap.add_argument("--seed", type=int, default=_util.SEED)
    ap.add_argument("--out", default=str(GOLDEN / "splits.csv"))
    ap.add_argument("--pool-dir", default=str(OUT), help="where parity_train_pool.csv / parity_bench_arm*.csv live")
    ap.add_argument("--overlap-out", default=str(OUT / "golden_splits_desi_overlap.csv"))
    a = ap.parse_args(argv)

    labels = _util.read_pinned(a.labels)
    frame = _util.read_pinned(a.frame)
    splits = assign_halves(labels, frame, seed=a.seed)
    coords = labels if {"ra_deg", "dec_deg"} <= set(labels.columns) else frame
    ov = firewall(splits, coords, pool_dir=Path(a.pool_dir))
    print(summarize(splits, labels))
    sha = _util.pin(splits, Path(a.out))
    print(f"[splits] {len(splits)} rows -> {a.out} (sha {sha}, seed {a.seed})")
    Path(a.overlap_out).parent.mkdir(parents=True, exist_ok=True)
    ov.to_csv(a.overlap_out, index=False)
    print(f"[overlaps] {len(ov)} rows -> {a.overlap_out}")


if __name__ == "__main__":
    main()
