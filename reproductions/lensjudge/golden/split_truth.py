#!/usr/bin/env python3
"""golden/split_truth.py — the pre-registered design / holdout halves of the JWST truth set
(golden/truth_splits.csv), independent of the golden campaign's align / validate split.

`design` is where the advocate / critic / arbitrator prompts are iterated, τ0 and the letter
thresholds t_A / t_B are frozen, and the anchors are looked at; `holdout` is scored once per
registered tuple (run_truth_eval.py refuses anything else). The rules, in the order they
are applied (split_halves.py's machinery with a truth cell instead of a human letter):

  unit of assignment   system_id: 10" union-find over ALL manifest rows (overlapping stamps
                       share pixels — ranks 16/17 at 8.78"; rank 7 and its rank-14 twin at
                       1.17"), so a system moves as a block
  literature by        lit_galaxy / lit_cluster rows are assigned by WHOLE proposal (all of
  proposal             A2744 / A370 / MACS0416 / COSMOS-Web on one side): a rubric tuned on
                       one cluster's arcs cannot have met that cluster on the holdout — a
                       stronger firewall than 2". A proposal group drags every system it
                       touches along (connected components of proposal ∪ system links)
  forced -> design     every frame unit with prior_exposure == 2 (ranks 1-15, seen with grade
                       + literature in the annotated docx), every anchor (ranks 15, 7, 14,
                       16, 13 — PI-derived, design-only by construction; rank 16 = u0153
                       brings rank 17 through their shared system; the rank-14 alias brings
                       rank 7's system), and every row sharing a component with one of them
  negatives            design / holdout copied from truth_negatives.csv (assigned by the
                       sampler per layout x field_class cell, exactly 200/200); asserted
                       200/200 overall and within 1 per field_class
  cells                truth_class x scale: cowls -> cowls_band; literature -> galaxy vs
                       cluster (the class itself); stress_D / stress_U -> the frame substratum
                       (merger / ring_spiral / elliptical_nearmiss / other; rank band);
                       anomalymatch; anchor; negative -> field_class
  stratified           seed 2026: multi-row components (proposal groups, multi-unit systems)
                       first, largest first, each to the half behind on its cells, then on
                       positives, then on total; then single-row systems per cell in a seeded
                       order, each to the half behind on (cell, truth_class, total), coin on
                       a tie. Deterministic.
  firewall             split_halves.firewall(halves=("design", "holdout")): no row within 2"
                       of a row in the other half (impossible at 10"; asserted anyway), and
                       every 2" overlap with the DESI parity pool / benches REPORTED per half

Output golden/truth_splits.csv (+ .sha): candidate_id, system_id, half, forced,
forced_reason, truth_class, cell. The manifest's `half` column is then filled and the
manifest re-pinned (build_truth_manifest.attach_half), so one run leaves a consistent
state; `--no-manifest-update` skips that.

    python lensjudge/golden/split_truth.py [--manifest golden/truth_manifest.csv]
        [--negatives golden/truth_negatives.csv] [--frame golden/frame.csv] [--seed 2026]
        [--out golden/truth_splits.csv] [--pool-dir outputs]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge.golden import _util, split_halves  # noqa: E402
from lensjudge.golden import build_frame as bf  # noqa: E402
from lensjudge.golden import build_truth_manifest as btm  # noqa: E402

GOLDEN = _util.HERE
OUT = _util.LENSJUDGE / "outputs"
HALVES = ("design", "holdout")
SYSTEM_ARCSEC = bf.SYSTEM_ARCSEC           # 10"
EXCL_RADIUS_ARCSEC = split_halves.EXCL_RADIUS_ARCSEC   # 2"
LIT_CLASSES = ("lit_galaxy", "lit_cluster")
SPLIT_COLS = ["candidate_id", "system_id", "half", "forced", "forced_reason", "truth_class", "cell"]


# ============================================================================ helpers
def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes")
    try:
        return bool(v) and not pd.isna(v)
    except (TypeError, ValueError):
        return bool(v)


def cell_of(row: pd.Series, substratum: str = "") -> str:
    """The stratification cell of one manifest row (see the module docstring)."""
    tc = str(row["truth_class"])
    if tc == "cowls":
        return f"cowls:{row.get('cowls_band', '') or 'provenance'}"
    if tc in LIT_CLASSES:
        return tc
    if tc in ("stress_D", "stress_U"):
        return f"{tc}:{substratum}" if substratum else tc
    if tc == "negative":
        return f"negative:{row.get('field_class', '') or 'blank'}"
    return tc


def components(man: pd.DataFrame, system_id: np.ndarray) -> np.ndarray:
    """Assignment groups: union of (same system_id) and (literature rows of one proposal).
    Dense 0-based component ids."""
    n = len(man)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first_of_sys: dict = {}
    for i, s in enumerate(system_id):
        if s in first_of_sys:
            union(first_of_sys[s], i)
        else:
            first_of_sys[s] = i
    first_of_prop: dict = {}
    is_lit = man["truth_class"].isin(LIT_CLASSES).to_numpy()
    for i, (lit, p) in enumerate(zip(is_lit, man["proposal"].astype(str))):
        if not lit:
            continue
        if p in first_of_prop:
            union(first_of_prop[p], i)
        else:
            first_of_prop[p] = i
    roots = [find(i) for i in range(n)]
    ids: dict = {}
    return np.array([ids.setdefault(r, len(ids)) for r in roots], dtype=int)


# ============================================================================ assignment
def assign(man: pd.DataFrame, negatives: pd.DataFrame | None, frame: pd.DataFrame | None = None,
           seed: int = _util.SEED, halves=HALVES) -> pd.DataFrame:
    """manifest (+ the sampler's negatives, + the frame for stress substrata) -> SPLIT_COLS."""
    m = man.copy().reset_index(drop=True)
    m["candidate_id"] = m["name"].astype(str)
    m["truth_class"] = m["truth_class"].astype(str)
    m["is_anchor"] = m["is_anchor"].map(_truthy)
    m["in_frame"] = m["in_frame"].map(_truthy)
    m["is_positive"] = m["is_positive"].map(_truthy)
    m["prior_exposure"] = pd.to_numeric(m["prior_exposure"], errors="coerce").fillna(0).astype(int)
    m["unit_id"] = m["unit_id"].fillna("").astype(str)
    sub = pd.Series(dtype=str)
    if frame is not None and "substratum" in frame.columns:
        sub = frame.set_index(frame["unit_id"].astype(str))["substratum"].fillna("").astype(str)
    m["cell"] = [cell_of(r, sub.get(r["unit_id"], "") if r["unit_id"] else "") for _, r in m.iterrows()]
    m["system_id"] = bf.union_find_systems(m["ra"], m["dec"], SYSTEM_ARCSEC)
    comp = components(m, m["system_id"].to_numpy())
    m["_comp"] = comp

    # ---- forced reasons, then propagate to whole components ------------------------------
    reason = np.array([""] * len(m), dtype=object)
    pe2 = (m["prior_exposure"] == 2) & m["in_frame"]
    for i in np.where(pe2)[0]:
        reason[i] = "prior_exposure_2"
    for i in np.where(m["is_anchor"])[0]:
        reason[i] = (reason[i] + "|" if reason[i] else "") + "anchor"
    seeds = {c: [m.at[i, "candidate_id"] for i in np.where((comp == c) & (reason != ""))[0]] for c in set(comp[reason != ""])}
    for i in range(len(m)):
        if reason[i] == "" and comp[i] in seeds:
            reason[i] = "component_of:" + "+".join(seeds[comp[i]])
    m["forced"] = reason != ""
    m["forced_reason"] = reason

    # ---- negatives: the sampler's half is authoritative ------------------------------------
    pre = {}
    if negatives is not None and len(negatives):
        nh = negatives.set_index(negatives["candidate_id"].astype(str))["half"].astype(str)
        neg_ids = set(m.loc[m["truth_class"] == "negative", "candidate_id"])
        missing = sorted(neg_ids - set(nh.index))
        assert not missing, f"negatives without a sampler half: {missing[:5]}"
        pre = {cid: nh[cid] for cid in neg_ids}
    else:
        assert not (m["truth_class"] == "negative").any(), "manifest has negatives but no truth_negatives table"
    # a negative must not share a component with anything else (the sampler's 10" buffer)
    neg_comps = set(m.loc[m["truth_class"] == "negative", "_comp"])
    mixed = m[m["_comp"].isin(neg_comps) & (m["truth_class"] != "negative")]
    assert mixed.empty, f"negatives share a 10\" system with: {mixed['candidate_id'].tolist()[:5]}"

    # ---- tallies / greedy phases ----------------------------------------------------------
    rng = np.random.default_rng(seed)
    tally = {h: {"cell": defaultdict(int), "class": defaultdict(int), "pos": 0, "total": 0} for h in halves}
    half = np.array([None] * len(m), dtype=object)

    def _add(rows, h):
        t = tally[h]
        for i in rows:
            t["cell"][m.at[i, "cell"]] += 1
            t["class"][m.at[i, "truth_class"]] += 1
            t["pos"] += int(m.at[i, "is_positive"])
            t["total"] += 1
            half[i] = h

    groups: dict = defaultdict(list)
    for i, c in enumerate(comp):
        groups[c].append(i)
    order = {c: k for k, c in enumerate(rng.permutation(sorted(groups)))}     # one seeded rank per component

    for c, rows in groups.items():                                           # phase 0: forced
        if m.at[rows[0], "forced"]:
            _add(rows, halves[0])
    for c, rows in groups.items():                                           # phase 1: negatives
        if half[rows[0]] is None and m.at[rows[0], "truth_class"] == "negative":
            for i in rows:
                _add([i], pre[m.at[i, "candidate_id"]])
    free = [c for c, rows in groups.items() if half[rows[0]] is None]
    multi = sorted([c for c in free if len(groups[c]) > 1], key=lambda c: (-len(groups[c]), order[c]))
    for c in multi:                                                          # phase 2: blocks
        rows = groups[c]
        def _behind(h):
            t = tally[h]
            return (sum(t["cell"][m.at[i, "cell"]] for i in rows), t["pos"], t["total"])
        sc = {h: _behind(h) for h in halves}
        h = halves[int(rng.integers(0, 2))] if sc[halves[0]] == sc[halves[1]] else min(halves, key=lambda x: sc[x])
        _add(rows, h)
    singles = [c for c in free if len(groups[c]) == 1]
    by_cell: dict = defaultdict(list)
    for c in singles:
        by_cell[m.at[groups[c][0], "cell"]].append(c)
    for cell in sorted(by_cell):                                             # phase 3: singles
        for c in sorted(by_cell[cell], key=lambda x: order[x]):
            i = groups[c][0]
            tc = m.at[i, "truth_class"]
            sc = {h: (tally[h]["cell"][cell], tally[h]["class"][tc], tally[h]["total"]) for h in halves}
            h = halves[int(rng.integers(0, 2))] if sc[halves[0]] == sc[halves[1]] else min(halves, key=lambda x: sc[x])
            _add([i], h)
    assert all(h is not None for h in half)
    m["half"] = half
    out = m[SPLIT_COLS].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    check(out, m, halves)
    return out


def check(splits: pd.DataFrame, m: pd.DataFrame | None = None, halves=HALVES) -> None:
    """The contract's invariants (the tests call this on synthetic splits too)."""
    assert list(splits.columns) == SPLIT_COLS
    assert splits["candidate_id"].is_unique
    assert splits["half"].isin(halves).all(), "unknown half label"
    assert (splits.loc[splits["forced"].map(_truthy), "half"] == halves[0]).all(), "forced rows must be in design"
    assert (splits.groupby("system_id")["half"].nunique() == 1).all(), "a system straddles the halves"
    lit = splits[splits["truth_class"].isin(LIT_CLASSES)]
    if m is not None and len(lit):
        prop = m.set_index("candidate_id")["proposal"].astype(str)
        straddle = lit.assign(_p=lit["candidate_id"].map(prop)).groupby("_p")["half"].nunique()
        assert (straddle == 1).all(), f"literature proposals straddle the halves: {straddle[straddle > 1].index.tolist()}"
    neg = splits[splits["truth_class"] == "negative"]
    if len(neg):
        vc = neg["half"].value_counts()
        assert abs(int(vc.get(halves[0], 0)) - int(vc.get(halves[1], 0))) <= 1, f"negatives unbalanced: {vc.to_dict()}"
        byfc = pd.crosstab(neg["cell"], neg["half"]).reindex(columns=list(halves), fill_value=0)
        assert (abs(byfc[halves[0]] - byfc[halves[1]]) <= 1).all(), f"negatives by field_class unbalanced:\n{byfc}"
    if m is not None:
        anchors = m.loc[m["is_anchor"].map(_truthy), "candidate_id"]
        ah = splits.set_index("candidate_id").loc[anchors, "half"]
        assert (ah == halves[0]).all(), "an anchor is not in design"


def summarize(splits: pd.DataFrame, halves=HALVES) -> str:
    d = splits
    lines = [f"rows: {len(d)}  systems: {d['system_id'].nunique()}  forced: {int(d['forced'].map(_truthy).sum())}",
             "per half: " + d.groupby("half").size().reindex(list(halves), fill_value=0).to_dict().__repr__(),
             "truth_class x half:\n" + pd.crosstab(d["truth_class"], d["half"]).reindex(columns=list(halves), fill_value=0).to_string(),
             "cell x half:\n" + pd.crosstab(d["cell"], d["half"]).reindex(columns=list(halves), fill_value=0).to_string()]
    return "\n".join(lines)


# ============================================================================ CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", default=str(GOLDEN / "truth_manifest.csv"))
    ap.add_argument("--negatives", default=str(GOLDEN / "truth_negatives.csv"))
    ap.add_argument("--frame", default=str(GOLDEN / "frame.csv"))
    ap.add_argument("--seed", type=int, default=_util.SEED)
    ap.add_argument("--out", default=str(GOLDEN / "truth_splits.csv"))
    ap.add_argument("--pool-dir", default=str(OUT), help="where parity_train_pool.csv / parity_bench_arm*.csv live")
    ap.add_argument("--overlap-out", default=str(OUT / "truth_splits_desi_overlap.csv"))
    ap.add_argument("--no-manifest-update", action="store_true", help="do not fill/re-pin the manifest's `half`")
    a = ap.parse_args(argv)

    man = _util.read_pinned(a.manifest, dtype={"name": str, "unit_id": str}, float_precision="round_trip")
    neg = _util.read_pinned(a.negatives, dtype={"candidate_id": str}) if Path(a.negatives).exists() else None
    frame = _util.read_pinned(a.frame, dtype={"unit_id": str}) if Path(a.frame).exists() else None
    splits = assign(man, neg, frame, seed=a.seed)
    # the firewall works in unit_id terms; the truth unit is the candidate_id
    fw = splits.rename(columns={"candidate_id": "unit_id", "half": "split"})[["unit_id", "split"]]
    coords = pd.DataFrame({"unit_id": man["name"].astype(str), "ra_deg": man["ra"].astype(float),
                           "dec_deg": man["dec"].astype(float), "candidate_id": man["name"].astype(str)})
    ov = split_halves.firewall(fw, coords, pool_dir=Path(a.pool_dir), halves=HALVES)
    print(summarize(splits))
    sha = _util.pin(splits, Path(a.out))
    print(f"[truth_splits] {len(splits)} rows -> {a.out} (sha {sha}, seed {a.seed})")
    Path(a.overlap_out).parent.mkdir(parents=True, exist_ok=True)
    ov.to_csv(a.overlap_out, index=False)
    print(f"[overlaps] {len(ov)} rows -> {a.overlap_out}")
    if not a.no_manifest_update:
        man2 = btm.attach_half(man, splits)
        msha = _util.pin(man2, Path(a.manifest))
        print(f"[truth_manifest] `half` filled -> {a.manifest} (sha {msha}); half counts {man2['half'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
