#!/usr/bin/env python3
"""golden/sample_truth_negatives.py — N1, the base-rate negatives of the JWST truth set
(golden/truth_negatives.csv, 400 rows = 200 design + 200 holdout).

WHAT. A seeded draw from the run's inspected-and-NOT-flagged targets (inspections.csv
status == ok, flagged == False; 3,284 rows), after four exclusions, stratified to the
positives' (layout x field_class) mix:

  catalogue purge   any target with >= 1 catalogue position within PURGE_ARCSEC (5") is out.
                    The union is every J/data/lenscats/*.csv + J/data/simbad_lenses.csv +
                    J/data/lenscats/raw_misc/cowls_catalogue.csv (the 15_master_crossmatch.py
                    recipe with the COWLS file that script reads from a path that does not
                    exist here): 195,818 positions, asserted >= 190k. diag_truth §2 measured
                    the contamination of random unflagged targets at 1.9 % (2") / 4.5 % (5")
                    because the JWST footprint is dominated by targeted cluster fields; the
                    purge leaves only undiscovered lenses (<~ 0.5-1 %, quoted as a systematic,
                    never a relabel). The pre-purge match counts at 2 / 3 / 5" are printed.
  frame / positives every golden frame unit and every positive / anchor of the truth set
                    (build_truth_manifest.positives_table, the same rule the manifest uses),
                    AND anything within SYSTEM_ARCSEC (10") of one of them: the truth split
                    moves 10" union-find systems as a block, so a negative inside a positive's
                    system could never be assigned its own half. (The contract says "minus
                    frame units"; the 10" buffer is the superset that keeps 200/200 exact.)
  DESI pool         any target within 2" of outputs/parity_train_pool.csv or
                    parity_bench_arm{1,2}.csv (build_frame.load_desi_catalog) — 365 of the
                    371 overlapping targets are DESI `random_neg`; a DESI-trained student
                    must not meet its own training rows here.
  stratification    largest-remainder allocation of N_TOTAL over (layout x field_class)
                    cells proportional to the positives' mix, capped by the eligible pool
                    (build_frame.largest_remainder); a draw a capped cell cannot take passes
                    down the remainder order. Critique C7: a random draw would be ~30 %
                    6882 + 5594 and the FPR-vs-recall comparison becomes field-vs-field.

  half              design / holdout assigned HERE (the contract puts `half` in this file so
                    the 200 design-half negatives can ship as Nate's calibration ids before
                    the split exists): per cell, systems (10" union-find over the draw) in
                    a seeded order, each to the half behind on (cell count, field_class
                    count, total); then single-unit systems are moved from the heavy half,
                    cell where it leads most first, until the halves are exactly N_TOTAL/2
                    each. split_truth.py copies this column and asserts it.

Seed 2026 (`np.random.default_rng`), consumed in a fixed order: per-cell draws in sorted
cell order, then the half permutation. Positions are results.csv float64.

    python lensjudge/golden/sample_truth_negatives.py [--jwst-repo J] [--golden-dir golden]
        [--n 400] [--out golden/truth_negatives.csv]

Output: golden/truth_negatives.csv (+ .sha): candidate_id, ra_deg, dec_deg, proposal,
field_class, layout, mag_r, sw_obs, lw_obs, sw_filter, lw_filter, nearest_cat_sep_arcsec,
half.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge.golden import _util  # noqa: E402
from lensjudge.golden import build_frame as bf  # noqa: E402
from lensjudge.golden import build_truth_manifest as btm  # noqa: E402

SEED = _util.SEED
GOLDEN = _util.HERE
OUTPUTS = _util.LENSJUDGE / "outputs"
N_TOTAL = 400
HALVES = ("design", "holdout")
PURGE_ARCSEC = 5.0
REPORT_ARCSEC = (2.0, 3.0, 5.0)          # contamination estimate printed before the purge
DESI_ARCSEC = bf.MATCH_ARCSEC             # 2"
SYSTEM_ARCSEC = bf.SYSTEM_ARCSEC          # 10"
MIN_UNION = 190_000
NEG_COLS = ["candidate_id", "ra_deg", "dec_deg", "proposal", "field_class", "layout", "mag_r",
            "sw_obs", "lw_obs", "sw_filter", "lw_filter", "nearest_cat_sep_arcsec", "half"]
_RA_COLS = ("ra", "ra_deg", "RA", "RA_deg", "RAJ2000", "ra_j2000")
_DEC_COLS = ("dec", "dec_deg", "DEC", "DEC_deg", "DEJ2000", "dec_j2000")


# ============================================================================ catalogue union
def _radec(df: pd.DataFrame) -> pd.DataFrame | None:
    """(ra, dec) from whichever column pair a catalogue uses; None when it has none."""
    cols = {c.lower(): c for c in df.columns}
    ra = next((cols[c.lower()] for c in _RA_COLS if c.lower() in cols), None)
    dec = next((cols[c.lower()] for c in _DEC_COLS if c.lower() in cols), None)
    if ra is None or dec is None:
        return None
    r = pd.to_numeric(df[ra], errors="coerce"); d = pd.to_numeric(df[dec], errors="coerce")
    m = r.between(0, 360) & d.between(-90, 90)
    return pd.DataFrame({"ra": r[m].to_numpy(float), "dec": d[m].to_numpy(float)})


def load_catalogue_union(jwst_repo: Path = _util.JWST_REPO, min_rows: int = MIN_UNION) -> pd.DataFrame:
    """All lens-catalogue positions (ra, dec, catalog): J/data/lenscats/*.csv +
    simbad_lenses.csv + raw_misc/cowls_catalogue.csv. Asserts >= `min_rows`."""
    jwst_repo = Path(jwst_repo)
    files = sorted(glob.glob(str(jwst_repo / "data" / "lenscats" / "*.csv")))
    files += [str(jwst_repo / "data" / "simbad_lenses.csv"),
              str(jwst_repo / "data" / "lenscats" / "raw_misc" / "cowls_catalogue.csv")]
    parts = []
    for f in files:
        if not os.path.exists(f):
            print(f"WARN: catalogue {f} absent", flush=True)
            continue
        rd = _radec(pd.read_csv(f, low_memory=False))
        if rd is None or not len(rd):
            print(f"WARN: no ra/dec columns in {f}", flush=True)
            continue
        parts.append(rd.assign(catalog=os.path.basename(f)[:-4]))
    cat = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ra", "dec", "catalog"])
    assert len(cat) >= min_rows, f"catalogue union has {len(cat)} positions, expected >= {min_rows}"
    return cat


# ============================================================================ the draw
def eligible_pool(inspections: pd.DataFrame, targets: pd.DataFrame, catalogue: pd.DataFrame,
                  exclude_pos: pd.DataFrame, desi: pd.DataFrame, render: pd.DataFrame | None,
                  purge_arcsec: float = PURGE_ARCSEC, buffer_arcsec: float = SYSTEM_ARCSEC,
                  desi_arcsec: float = DESI_ARCSEC) -> tuple[pd.DataFrame, dict]:
    """Unflagged & ok targets minus the purge / frame-buffer / DESI exclusions, with the
    sampler's columns attached. `exclude_pos`: ra_deg, dec_deg (+ candidate_id) of every
    frame unit and truth positive / anchor. Returns (pool, stats)."""
    ins = inspections.copy()
    ins["id"] = ins["id"].astype(str)
    base = ins[(ins["status"].astype(str) == "ok") & ~ins["flagged"].fillna(False).astype(bool)].copy()
    S: dict = {"unflagged_ok": len(base)}
    # contamination estimate BEFORE the purge (diag_truth §2 table)
    _, sep_cat, _ = bf.sky_match(base["ra"], base["dec"], catalogue["ra"], catalogue["dec"], purge_arcsec)
    S["cat_matches"] = {f"{r:g}": int((sep_cat < r).sum()) for r in REPORT_ARCSEC}
    S["cat_rates"] = {k: v / max(len(base), 1) for k, v in S["cat_matches"].items()}
    base["nearest_cat_sep_arcsec"] = np.round(sep_cat, 3)
    purged = sep_cat < purge_arcsec
    S["purged"] = int(purged.sum())
    base = base[~purged]
    # frame units / positives and their 10" neighbourhood
    by_id = base["id"].isin(set(exclude_pos["candidate_id"].astype(str))) if "candidate_id" in exclude_pos else np.zeros(len(base), bool)
    _, sep_fr, near = bf.sky_match(base["ra"], base["dec"], exclude_pos["ra_deg"], exclude_pos["dec_deg"], buffer_arcsec)
    S["excluded_frame_or_positive"] = int((by_id | near).sum())
    base = base[~(by_id | near)]
    # DESI pool / benches
    _, _, hit = bf.sky_match(base["ra"], base["dec"], desi["ra"], desi["dec"], desi_arcsec)
    S["excluded_desi"] = int(hit.sum())
    base = base[~hit].reset_index(drop=True)
    # columns
    tg = targets.set_index(targets["id"].astype(str))
    t = tg.loc[base["id"]]
    pool = pd.DataFrame({"candidate_id": base["id"].to_numpy(),
                         "ra_deg": base["ra"].to_numpy(float), "dec_deg": base["dec"].to_numpy(float),
                         "proposal": t["proposal"].fillna("").astype(str).to_numpy(),
                         "mag_r": t["mag_r"].to_numpy(float),
                         "nearest_cat_sep_arcsec": base["nearest_cat_sep_arcsec"].to_numpy(float)})
    for c in ("sw_obs", "lw_obs", "sw_filter", "lw_filter"):
        pool[c] = t[c].fillna("").astype(str).to_numpy()
    pool["layout"] = bf.derive_layout(t["sw_obs"], t["lw_obs"], render, base["id"]) if len(pool) else np.array([], dtype=object)
    pool["field_class"] = [btm.field_class_of(p) for p in pool["proposal"]]
    S["eligible"] = len(pool)
    return pool, S


def allocate(pos_mix: dict, pool: pd.DataFrame, n: int = N_TOTAL) -> tuple[dict, int]:
    """(layout, field_class) -> n, largest remainder on the positives' mix, capped by the
    pool. Cells without positives get nothing."""
    cap = pool.groupby(["layout", "field_class"]).size().to_dict()
    return bf.largest_remainder(pos_mix, n, capacity=cap)


def draw_negatives(pool: pd.DataFrame, alloc: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Seeded per-cell draw (build_frame.draw: id-sorted pool, choice without replacement),
    cells in sorted order so the rng consumption is fixed."""
    picks = []
    for cell in sorted(alloc, key=str):
        sub = pool[(pool["layout"] == cell[0]) & (pool["field_class"] == cell[1])]
        picks.append(bf.draw(sub, alloc[cell], rng, "random", id_col="candidate_id"))
    out = pd.concat(picks, ignore_index=True) if picks else pool.iloc[0:0]
    assert out["candidate_id"].is_unique
    return out


def assign_halves(neg: pd.DataFrame, rng: np.random.Generator, halves=HALVES,
                  system_arcsec: float = SYSTEM_ARCSEC) -> np.ndarray:
    """design / holdout per row: 10" systems move together; per (layout, field_class) cell in
    a seeded order each system goes to the half behind on (cell, field_class, total); a
    final pass moves single systems from the heavy half until the totals are equal
    (possible whenever the total is even and a movable single exists)."""
    n = len(neg)
    if n == 0:
        return np.array([], dtype=object)
    sys_id = bf.union_find_systems(neg["ra_deg"], neg["dec_deg"], system_arcsec)
    cells = list(zip(neg["layout"].astype(str), neg["field_class"].astype(str)))
    groups: dict = {}
    for i, (s, c) in enumerate(zip(sys_id, cells)):
        g = groups.setdefault(int(s), {"rows": [], "cells": [], "fc": []})
        g["rows"].append(i); g["cells"].append(c); g["fc"].append(c[1])
    order = {int(s): k for k, s in enumerate(rng.permutation(sorted(groups)))}
    tally = {h: {"cell": defaultdict(int), "fc": defaultdict(int), "total": 0} for h in halves}
    half_of: dict = {}

    def _add(sid, h, sign=+1):
        t = tally[h]
        for c, fc in zip(groups[sid]["cells"], groups[sid]["fc"]):
            t["cell"][c] += sign; t["fc"][fc] += sign
        t["total"] += sign * len(groups[sid]["rows"])
        half_of[sid] = h if sign > 0 else None

    by_cell: dict = defaultdict(list)
    for sid, g in groups.items():
        by_cell[g["cells"][0]].append(sid)
    for cell in sorted(by_cell):
        for sid in sorted(by_cell[cell], key=lambda s: order[s]):
            fc = groups[sid]["fc"][0]
            sc = {h: (tally[h]["cell"][cell], tally[h]["fc"][fc], tally[h]["total"]) for h in halves}
            h = halves[int(rng.integers(0, 2))] if sc[halves[0]] == sc[halves[1]] else min(halves, key=lambda x: sc[x])
            _add(sid, h)
    # exact balance: move single-row systems from the heavy half, cell where it leads most
    while tally[halves[0]]["total"] != tally[halves[1]]["total"]:
        heavy, light = sorted(halves, key=lambda h: -tally[h]["total"])
        if tally[heavy]["total"] - tally[light]["total"] < 2:
            break                                   # an odd total: within 1 is the best possible
        cands = [s for s in groups if half_of[s] == heavy and len(groups[s]["rows"]) == 1]
        assert cands, "cannot balance the halves: no single-row system left in the heavy half"
        th, tl = tally[heavy], tally[light]
        sid = max(cands, key=lambda s: (th["cell"][groups[s]["cells"][0]] - tl["cell"][groups[s]["cells"][0]],
                                        th["fc"][groups[s]["fc"][0]] - tl["fc"][groups[s]["fc"][0]], -order[s]))
        _add(sid, heavy, -1); _add(sid, light, +1)
    out = np.empty(n, dtype=object)
    for sid, g in groups.items():
        for i in g["rows"]:
            out[i] = half_of[sid]
    return out


def sample(src: dict, catalogue: pd.DataFrame, n: int = N_TOTAL, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """positives' mix -> eligible pool -> allocation -> draw -> halves. Returns (neg, stats)."""
    pos, pnotes = btm.positives_table(src)
    anchors = btm.select_anchors(src)
    frame = src["frame"]
    res = src["results"].set_index("id")
    ex_ids = sorted(set(frame["candidate_id"].astype(str)) | set(pos["candidate_id"]) | set(anchors["candidate_id"]))
    exclude = pd.DataFrame({"candidate_id": ex_ids, "ra_deg": res.loc[ex_ids, "ra"].to_numpy(float),
                            "dec_deg": res.loc[ex_ids, "dec"].to_numpy(float)})
    pool, S = eligible_pool(src["inspections"], src["targets"], catalogue, exclude, src["desi"], src.get("render"))
    pos_mix = pos.groupby(["layout", "field_class"]).size().to_dict()
    S["positives"] = len(pos); S["pos_mix"] = {f"{k[0]}|{k[1]}": v for k, v in pos_mix.items()}
    alloc, left = allocate(pos_mix, pool, n)
    S["alloc"] = {f"{k[0]}|{k[1]}": v for k, v in alloc.items()}; S["unallocated"] = left
    rng = np.random.default_rng(seed)
    neg = draw_negatives(pool, alloc, rng)
    neg["half"] = assign_halves(neg, rng)
    neg = neg.sort_values("candidate_id", kind="stable").reset_index(drop=True)[NEG_COLS]
    S["n"] = len(neg)
    S["halves"] = neg["half"].value_counts().to_dict()
    return neg, S


def check(neg: pd.DataFrame, n: int = N_TOTAL) -> None:
    """The contract's invariants (also run by the tests)."""
    assert list(neg.columns) == NEG_COLS
    assert neg["candidate_id"].is_unique
    assert len(neg) == n, f"{len(neg)} negatives, expected {n}"
    vc = neg["half"].value_counts()
    assert set(vc.index) <= set(HALVES) and abs(int(vc.get(HALVES[0], 0)) - int(vc.get(HALVES[1], 0))) <= 1, vc.to_dict()
    assert (neg["nearest_cat_sep_arcsec"] >= PURGE_ARCSEC).all(), "a negative sits within the purge radius"
    assert neg["field_class"].isin(btm.FIELD_CLASSES).all()
    byfc = pd.crosstab(neg["field_class"], neg["half"]).reindex(columns=list(HALVES), fill_value=0)
    assert (abs(byfc[HALVES[0]] - byfc[HALVES[1]]) <= 1).all(), f"field_class halves unbalanced:\n{byfc}"


def report(neg: pd.DataFrame, S: dict, out: Path, sha: str) -> None:
    print(f"truth_negatives: {len(neg)} rows -> {out} (sha {sha}, seed {SEED})")
    print(f"pool: unflagged&ok {S['unflagged_ok']}; catalogue matches before purge "
          + ", ".join(f"{k}\" {v} ({100 * S['cat_rates'][k]:.2f}%)" for k, v in S["cat_matches"].items())
          + f"; purged at {PURGE_ARCSEC:g}\" {S['purged']}; excluded frame/positive(+{SYSTEM_ARCSEC:g}\") "
          f"{S['excluded_frame_or_positive']}; excluded DESI(2\") {S['excluded_desi']}; eligible {S['eligible']}")
    print(f"positives stratified to: {S['positives']} rows, mix {S['pos_mix']}")
    print(f"allocation (layout|field_class): {S['alloc']}" + (f"  UNALLOCATED {S['unallocated']}" if S["unallocated"] else ""))
    print("cells x half:")
    print(pd.crosstab([neg["layout"], neg["field_class"]], neg["half"]).to_string())
    print("field_class x half:")
    print(pd.crosstab(neg["field_class"], neg["half"]).to_string())
    print(f"residual contamination after the {PURGE_ARCSEC:g}\" purge (undiscovered lenses only, diag_truth "
          f"§2: ~0.5-1%) is quoted as a systematic: ~{len(neg) // 200}-{len(neg) // 100} of {len(neg)} rows, "
          f"~{len(neg) // 400}-{len(neg) // 200} per half; a positive-looking negative is reported, never relabelled")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jwst-repo", type=Path, default=_util.JWST_REPO)
    ap.add_argument("--outputs", type=Path, default=OUTPUTS)
    ap.add_argument("--golden-dir", type=Path, default=GOLDEN)
    ap.add_argument("--n", type=int, default=N_TOTAL)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=None, help="default <golden-dir>/truth_negatives.csv")
    a = ap.parse_args(argv)
    out = a.out or a.golden_dir / "truth_negatives.csv"
    src = btm.load_truth_sources(a.jwst_repo, a.outputs, a.golden_dir)
    cat = load_catalogue_union(a.jwst_repo)
    print(f"catalogue union: {len(cat)} positions from {cat['catalog'].nunique()} files", flush=True)
    neg, S = sample(src, cat, a.n, a.seed)
    check(neg, a.n)
    sha = _util.pin(neg, out)
    report(neg, S, out, sha)


if __name__ == "__main__":
    main()
