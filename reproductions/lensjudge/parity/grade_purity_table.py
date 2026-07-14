#!/usr/bin/env python3
"""First grade-stratified confirmation-rate ("purity") table for the Huang/Storfer program.

No paper in the series publishes P(real lens | grade). This script assembles every
recoverable per-target follow-up outcome and cross-matches it against the unified graded
candidate table (eval/crossmatch_external.py -> outputs/master_candidates.csv, 4,354 unique
candidates with consensus grades A/B/C from Huang 2020/2021 + Storfer 2024 + Inchausti 2025).

Truth sources (normalized outcome vocabulary: confirmed_lens | lens_z_only | non_lens |
inconclusive | observed_pending):
  - parity/data/external/*.csv        one CSV per campaign, extracted from the published
                                      papers by the truth-table-assembly workflow
                                      (foundry_i_hst, foundry_iii_nires, foundry_iv_muse,
                                      agel, hsc_desi_dr1)
  - foundry-ii master comparison      local reproduction CSV (DESI EDR spectroscopy;
                                      `section` = confirmed/known/non-lens/pending)
  - SuGOHI full catalog               rows with BOTH lens+source spec-z (sentinel -99)
                                      = spectroscopically confirmed (campaign: sugohi_specz)

SELECTION-BIAS CAVEATS (printed with the table, and load-bearing):
  1. Follow-up targets were CHOSEN, mostly best-first (Foundry I explicitly "the best");
     per-grade purity is an UPPER bound for A/B and close to meaningless for C unless the
     campaign was grade-agnostic.
  2. Confirmed lens != the grade was "right" in fine A-vs-B terms; this table addresses
     only the binary claim (grade => real lens).
  3. A candidate followed by several campaigns counts once, with outcome precedence
     confirmed_lens > non_lens > lens_z_only > inconclusive > observed_pending, and
     conflicts (confirmed in one campaign, refuted in another) are listed explicitly.

  python lensjudge/parity/grade_purity_table.py

Outputs: outputs/parity_grade_purity.csv (per grade x campaign and totals),
         outputs/parity_truth_master.csv (one row per truth-matched graded candidate).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from scipy.stats import beta
import astropy.units as u

HERE = Path(__file__).resolve().parent
# HERE = .../reproductions/lensjudge/parity -> parents[1] = reproductions/
REPRO = HERE.parents[1]
OUT = HERE.parent / "outputs"
EXTERNAL = HERE / "data" / "external"
MASTER = OUT / "master_candidates.csv"
FOUNDRY_II = REPRO / "foundry-ii" / "data" / "foundry_ii_master_comparison.csv"
SUGOHI_FULL = REPRO / "aion-1" / "data" / "raw" / "sugohi" / "sugohi_full.csv"

MATCH_RADIUS_ARCSEC = 2.0
PRECEDENCE = ["confirmed_lens", "non_lens", "lens_z_only", "inconclusive", "observed_pending"]


def load_truth() -> pd.DataFrame:
    frames = []
    # 1) workflow-extracted campaign tables
    if EXTERNAL.exists():
        for p in sorted(EXTERNAL.glob("*.csv")):
            df = pd.read_csv(p)
            need = {"campaign", "name", "ra_deg", "dec_deg", "outcome"}
            missing = need - set(df.columns)
            if missing:
                print(f"  [skip] {p.name}: missing columns {missing}")
                continue
            frames.append(df[["campaign", "name", "ra_deg", "dec_deg", "outcome",
                              *(c for c in ("detail", "candidate_provenance", "source_ref") if c in df)]])
            print(f"  [{p.stem:18s}] {len(df):4d} targets  {df.outcome.value_counts().to_dict()}")
    # 2) Foundry II (local): DESI EDR spectroscopy master comparison
    if FOUNDRY_II.exists():
        f2 = pd.read_csv(FOUNDRY_II)
        sec2out = {
            "confirmed": "confirmed_lens", "known": "confirmed_lens",
            "non-lens": "non_lens", "nonlens": "non_lens", "non_lens": "non_lens",
            "pending_source": "lens_z_only",  # lens z secured, source z pending
            "pending_zs": "lens_z_only",
            "pending": "observed_pending", "pending_both": "observed_pending",
        }
        sec = f2["section"].astype(str).str.strip().str.lower()
        out = sec.map(sec2out)
        unmapped = sorted(sec[out.isna()].unique())
        if unmapped:
            print(f"  [foundry_ii] WARNING unmapped sections -> observed_pending: {unmapped}")
        df = pd.DataFrame({
            "campaign": "foundry_ii_edr", "name": f2["name"],
            "ra_deg": f2["ra_deg"], "dec_deg": f2["dec_deg"],
            "outcome": out.fillna("observed_pending"),
            "detail": "section=" + f2["section"].astype(str),
            "source_ref": "arXiv:2509.18089",
        })
        frames.append(df)
        print(f"  [foundry_ii_edr    ] {len(df):4d} targets  {df.outcome.value_counts().to_dict()}")
    # 3) SuGOHI spectroscopic confirmations (both z's measured; sentinel -99)
    if SUGOHI_FULL.exists():
        su = pd.read_csv(SUGOHI_FULL)
        zl = pd.to_numeric(su["zl_spec"], errors="coerce")
        zs = pd.to_numeric(su["zs_spec"], errors="coerce")
        conf = su[(zl > 0) & (zs > 0)]
        df = pd.DataFrame({
            "campaign": "sugohi_specz", "name": conf["name"],
            "ra_deg": conf["ra"], "dec_deg": conf["dec"],
            "outcome": "confirmed_lens",
            "detail": ("zl=" + zl[conf.index].round(4).astype(str) + " zs=" + zs[conf.index].round(4).astype(str)
                       + " sugohi_grade=" + conf["grade"].astype(str)),
            "source_ref": "SuGOHI public list (Oguri)",
        })
        frames.append(df)
        print(f"  [sugohi_specz      ] {len(df):4d} targets  all confirmed_lens")
    if not frames:
        raise SystemExit("no truth sources found")
    truth = pd.concat(frames, ignore_index=True)
    truth = truth.dropna(subset=["ra_deg", "dec_deg"])
    bad = ~truth["outcome"].isin(PRECEDENCE)
    if bad.any():
        print(f"  WARNING: {bad.sum()} rows with unknown outcome values dropped: "
              f"{sorted(truth.loc[bad, 'outcome'].unique())}")
        truth = truth[~bad]
    return truth.reset_index(drop=True)


def jeffreys_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    return (float(beta.ppf(0.025, k + 0.5, n - k + 0.5)),
            float(beta.ppf(0.975, k + 0.5, n - k + 0.5)))


def main() -> None:
    master = pd.read_csv(MASTER)
    print(f"master graded candidates: {len(master)}  grades {master.grade.value_counts().to_dict()}\n")
    print("=== truth sources ===")
    truth = load_truth()

    # --- match truth targets to graded candidates ---
    mc = SkyCoord(master.ra.values * u.deg, master.dec.values * u.deg)
    tc = SkyCoord(truth.ra_deg.values * u.deg, truth.dec_deg.values * u.deg)
    idx, sep, _ = tc.match_to_catalog_sky(mc)
    truth["cand_idx"] = idx
    truth["sep_arcsec"] = sep.arcsec
    truth["matched"] = truth["sep_arcsec"] < MATCH_RADIUS_ARCSEC
    print(f"\ntruth targets matched to a graded candidate (<{MATCH_RADIUS_ARCSEC}\"): "
          f"{truth.matched.sum()}/{len(truth)}")
    for camp, grp in truth.groupby("campaign"):
        print(f"  {camp:18s} {grp.matched.sum():4d}/{len(grp):4d} matched")

    m = truth[truth.matched].copy()
    m["grade"] = master.grade.values[m.cand_idx]
    m["cand_name"] = master.name.values[m.cand_idx]
    m["cand_source"] = master.source.values[m.cand_idx]

    # --- one outcome per candidate, with precedence + explicit conflicts ---
    rank = {o: i for i, o in enumerate(PRECEDENCE)}
    m["_r"] = m.outcome.map(rank)
    per_cand = m.sort_values("_r").groupby("cand_idx", as_index=False).agg(
        cand_name=("cand_name", "first"), grade=("grade", "first"),
        outcome=("outcome", "first"), campaigns=("campaign", lambda s: "+".join(sorted(set(s)))),
        n_campaigns=("campaign", "nunique"), detail=("detail", "first"),
    )
    conflicts = m.groupby("cand_idx").outcome.apply(
        lambda s: ("confirmed_lens" in set(s)) and ("non_lens" in set(s)))
    conflict_ids = conflicts[conflicts].index
    if len(conflict_ids):
        print(f"\nCONFLICTS (confirmed in one campaign, refuted in another) — {len(conflict_ids)}:")
        print(m[m.cand_idx.isin(conflict_ids)][["cand_name", "campaign", "outcome", "detail"]]
              .to_string(index=False))

    per_cand.to_csv(OUT / "parity_truth_master.csv", index=False)

    # --- the purity table ---
    print("\n=== grade-stratified follow-up outcomes (one row per unique graded candidate) ===")
    xtab = pd.crosstab(per_cand.grade, per_cand.outcome).reindex(
        index=["A", "B", "C"], columns=PRECEDENCE).fillna(0).astype(int)
    print(xtab.to_string())

    rows = []
    for grade in ["A", "B", "C"]:
        sub = per_cand[per_cand.grade == grade]
        k = int((sub.outcome == "confirmed_lens").sum())
        r = int((sub.outcome == "non_lens").sum())
        n_dec = k + r
        lo, hi = jeffreys_ci(k, n_dec)
        rows.append({"grade": grade, "followed_up": len(sub), "confirmed": k,
                     "refuted": r, "lens_z_only": int((sub.outcome == "lens_z_only").sum()),
                     "pending_or_inconclusive": int(sub.outcome.isin(["inconclusive", "observed_pending"]).sum()),
                     "decided": n_dec,
                     "purity_decided": round(k / n_dec, 3) if n_dec else np.nan,
                     "purity_ci_lo": round(lo, 3) if n_dec else np.nan,
                     "purity_ci_hi": round(hi, 3) if n_dec else np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "parity_grade_purity.csv", index=False)
    print("\n=== purity per grade (decided = confirmed + refuted only) ===")
    print(res.to_string(index=False))

    unmatched = truth[~truth.matched]
    print(f"\nunmatched truth targets (not from the graded catalogs, e.g. DES/Jacobs-selected "
          f"AGEL or SuGOHI-only): {len(unmatched)}")
    print(f"saved {OUT / 'parity_grade_purity.csv'} and {OUT / 'parity_truth_master.csv'}")
    print("\nCAVEATS: follow-up targets were curated (mostly best-first); purity is an UPPER")
    print("bound, and per-grade comparisons inherit each campaign's selection function. The")
    print("only grade-agnostic campaign to date is the DESI Secondary Target Program (results")
    print("pending). This table tests the binary claim (grade => lens), not fine A-vs-B.")


if __name__ == "__main__":
    main()
