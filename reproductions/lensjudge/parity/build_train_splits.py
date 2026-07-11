#!/usr/bin/env python3
"""Leak-aware training/selection splits for the Phase C machine graders.

THE RULE (approved plan): no system may train on, or be selected on, any row of the
parity bench. This script builds the one training pool everything in Phase C draws
from, with the firewall applied by NAME and by POSITION:

  excluded:  every parity_bench_arm1.csv name (truth rows AND the augmentation
             strata — the FPR anchor must stay unseen), and anything within
             EXCL_RADIUS_ARCSEC of any parity_bench_arm2.csv position (the Euclid
             fields are evaluation-only territory).

  pool:      graded A/B/C candidates (io.load_candidates) + grade-D rejects
             (io.load_grade_d) + random-galaxy negatives (negatives_extra.parquet)
             + ClaudeNet mimic bank when staged.

  targets:   human-SOFT p_lens targets following the v5 recipe's convention
             (A 0.95 / B 0.80 / C 0.45 / D 0.05 / random 0.02 / mimic 0.05).

  splits:    train / valsel (selection) / gate (frozen evidence gate, never used
             for selection — the Phase C analogue of v5's test2), stratified by
             (label_source x grade) with seed 2026.

  python lensjudge/parity/build_train_splits.py

Outputs (outputs/): parity_train_pool.csv (full pool + split column, SHA-pinned),
plus a printed leakage audit (0 name collisions, 0 position collisions required).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
import astropy.units as u  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import io  # noqa: E402

SEED = 2026
EXCL_RADIUS_ARCSEC = 2.0
OUT = Path(__file__).resolve().parents[1] / "outputs"
SOFT = {"A": 0.95, "B": 0.80, "C": 0.45, "D": 0.05, "random": 0.02, "mimic": 0.05}
GATE_PER_STRATUM = 60   # frozen evidence gate: 60 per (source x grade) stratum
VALSEL_FRAC = 0.15


# SCORE PROVENANCE (learned the hard way): graded rows' io p_meta is the
# REPRODUCTION ensemble (our_p_meta); grade-D rows only have the PUBLISHED
# NeuraLens probability. Mixing them across classes fabricates AUC (measured:
# 0.847 mismatched vs 0.646 like-for-like on the gate HARD contrast). We carry
# both explicitly: p_pub (published, comparable across all graded+D rows) and
# p_repro (reproduction ensemble, graded rows only).
_st = pd.read_csv(config.INCH_DATA / "candidate_scores_storfer.csv")[
    ["name", "probability", "our_p_meta"]].rename(columns={"probability": "p_pub"})
_inc = pd.read_csv(config.INCH_DATA / "candidate_scores_inchausti.csv")[
    ["name", "p_meta", "our_p_meta"]].rename(columns={"p_meta": "p_pub"})
PUB = pd.concat([_st, _inc]).drop_duplicates("name").set_index("name")


def build_pool() -> pd.DataFrame:
    rows = []
    cand = io.load_candidates("both")
    for _, r in cand.iterrows():
        pub = PUB.p_pub.get(r["name"])
        repro = PUB.our_p_meta.get(r["name"])
        rows.append(dict(name=r["name"], ra=r["ra"], dec=r["dec"], survey_key=r["catalog"],
                         grade=r["grade"], label_source="graded",
                         soft_target=SOFT[str(r["grade"]).strip().upper()],
                         p_pub=pub, p_repro=repro))
    gd = io.load_grade_d("both")
    for _, r in gd.iterrows():
        rows.append(dict(name=r["name"], ra=r["ra"], dec=r["dec"], survey_key=r["catalog"],
                         grade="D", label_source="graded_D", soft_target=SOFT["D"],
                         p_pub=r.get("p_meta"), p_repro=None))
    negp = config.INCH_DATA / "negatives_extra.parquet"
    if negp.exists():
        neg = pd.read_parquet(negp)
        for _, r in neg.iterrows():
            rows.append(dict(name=str(r["row_id"]), ra=r.get("RA"), dec=r.get("DEC"),
                             survey_key="storfer", grade="", label_source="random_neg",
                             soft_target=SOFT["random"], p_pub=None, p_repro=None))
    mimp = config.REPRO / "claudenet" / "data" / "v3" / "mimic_bank_seed.parquet"
    if mimp.exists():
        mm = pd.read_parquet(mimp)
        for _, r in mm.iterrows():
            rows.append(dict(name=str(r["row_id"]), ra=r.get("RA"), dec=r.get("DEC"),
                             survey_key="claudenet", grade="", label_source="mimic_neg",
                             soft_target=SOFT["mimic"], p_pub=r.get("p_final"), p_repro=None))
    df = pd.DataFrame(rows).drop_duplicates("name").reset_index(drop=True)
    return df


def apply_firewall(pool: pd.DataFrame) -> pd.DataFrame:
    a1 = pd.read_csv(OUT / "parity_bench_arm1.csv")
    a2 = pd.read_csv(OUT / "parity_bench_arm2.csv")
    bench_names = set(a1.name.astype(str)) | set(a2.name.astype(str))
    by_name = pool.name.astype(str).isin(bench_names)

    ok = pool.ra.notna() & pool.dec.notna()
    by_pos = pd.Series(False, index=pool.index)
    bench_pos = pd.concat([a1[["ra", "dec"]], a2[["ra", "dec"]]]).dropna()
    pc = SkyCoord(pool.loc[ok, "ra"].values * u.deg, pool.loc[ok, "dec"].values * u.deg)
    bc = SkyCoord(bench_pos.ra.values * u.deg, bench_pos.dec.values * u.deg)
    idx, sep, _ = pc.match_to_catalog_sky(bc)
    by_pos.loc[ok[ok].index] = sep.arcsec < EXCL_RADIUS_ARCSEC

    excl = by_name | by_pos
    print(f"firewall: {by_name.sum()} by name, {(by_pos & ~by_name).sum()} extra by "
          f"position (<{EXCL_RADIUS_ARCSEC}\"), {excl.sum()} rows excluded")
    kept = pool[~excl].reset_index(drop=True)

    # verify: nothing kept collides with the bench
    kc = SkyCoord(kept.dropna(subset=["ra", "dec"]).ra.values * u.deg,
                  kept.dropna(subset=["ra", "dec"]).dec.values * u.deg)
    _, sep2, _ = kc.match_to_catalog_sky(bc)
    n_bad = int((sep2.arcsec < EXCL_RADIUS_ARCSEC).sum())
    assert n_bad == 0, f"leak audit FAILED: {n_bad} kept rows within radius of bench"
    assert not kept.name.astype(str).isin(bench_names).any()
    print("leak audit: 0 name collisions, 0 position collisions — pool is clean")
    return kept


def assign_splits(pool: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    pool = pool.copy()
    pool["stratum"] = pool.label_source + "|" + pool.grade.fillna("")
    pool["split"] = "train"
    for _, grp in pool.groupby("stratum"):
        idx = rng.permutation(grp.index.to_numpy())
        n_gate = min(GATE_PER_STRATUM, max(1, len(idx) // 10))
        n_val = int(len(idx) * VALSEL_FRAC)
        pool.loc[idx[:n_gate], "split"] = "gate"
        pool.loc[idx[n_gate:n_gate + n_val], "split"] = "valsel"
    return pool


def main() -> None:
    pool = build_pool()
    print(f"pool: {len(pool)} rows  {pool.label_source.value_counts().to_dict()}")
    pool = apply_firewall(pool)
    pool = assign_splits(pool)
    print("splits x source:")
    print(pool.pivot_table(index="label_source", columns="split", values="name",
                           aggfunc="count", fill_value=0).to_string())
    out = OUT / "parity_train_pool.csv"
    pool.to_csv(out, index=False)
    h = hashlib.sha256(pool.to_csv(index=False).encode()).hexdigest()[:16]
    Path(str(out) + ".sha").write_text(h + "\n")
    print(f"[pool] {len(pool)} rows -> {out} (sha {h})")


if __name__ == "__main__":
    main()
