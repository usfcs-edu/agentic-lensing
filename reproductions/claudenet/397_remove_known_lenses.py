#!/usr/bin/env python3
"""397_remove_known_lenses.py — subtract the known-lens catalogs from the v4 DR11-south
candidate ranking, so what remains is the genuinely-unpublished set.

Input (default): the FULL v4 top-150k, `dr11s_v4_candidates_150k.csv`, produced by
`396_export_v4_new_candidates.py --full`. It carries an `is_new` flag = 1 for the
134,078 rows new relative to the previous union-95k sweep, 0 for the 15,922 that sweep
also surfaced. This script does NOT filter on is_new: it keeps the whole top-150k and
subtracts only *published* lenses. (Earlier the input was the 134,078 is_new==1 subset;
that dropped 14,520 real candidates including v4's #1-ranked one, so we now ship the full
ranking and carry is_new through as metadata.) The v4 sweep keeps known lenses in its parent
sample on purpose
(the 160/163 population choice), so catalog lenses rank high and they do: 86 of v4's
top-100 rows match a known lens. This script does the second subtraction, against the
literature: drop every candidate within --radius of a known lens.

Matching (astropy `match_to_catalog_sky`, the 114/163 pattern, 5" default):

  * REMOVAL uses the forward nearest neighbour (candidate -> nearest KLC entry).
    This is exact: if a candidate's *nearest* KLC entry is beyond R, then no KLC
    entry is within R. No radius-search needed.
  * RECOVERY ("how many known lenses did v4 re-find") uses the REVERSE nearest
    neighbour (KLC entry -> nearest candidate). Counting distinct KLC entries via
    the forward match under-counts, because two KLC entries can sit within R of
    each other (112 KLC rows do) and only the nearer one is ever reported.
  * Many-to-one is expected and handled: 256 KLC entries are matched by >1
    candidate row, because the Tractor catalog deblends one lens into several
    sources. Every matched row is removed, not just the nearest.

Outputs (CSV columns carried through from 396 incl. is_new, plus a fresh contiguous rank):
  <out>          survivors, known lenses removed, + rank_clean (1..N)  -> 145,297 rows
  <removed-out>  the dropped rows + sep_arcsec / klc_index / klc_ra / klc_dec /
                 klc_ref / klc_dup / klc_catalog  -> every removal is auditable
  <report>       JSON: counts, radius-sensitivity table, recovery, provenance

"Known" means the union of every known-lens catalog this repo ships: the KLC (--known,
12,210 entries) plus DEFAULT_EXTRA -- the Huang-group's four DESI searches
(huang2020/huang2021/storfer2024/inchausti2025) and external_lens_catalog.csv, auto-unioned.
This is NOT redundant with the KLC: Inchausti+2025 (DR10) is ENTIRELY absent from the KLC
(0 of 811 entries), so 559 of its published lenses survived until this catalog was staged;
Huang+2020 has 7 entries absent from the KLC too. Union = 16,573 rows (cross-catalog
duplicates NOT deduped: harmless for removal, inflates the recall denominator). See the
DEFAULT_EXTRA comment below for the per-catalog KLC-coverage audit.

  python 397_remove_known_lenses.py                      # 5", full union -> the shipped 145,297
  python 397_remove_known_lenses.py --radius 3
  python 397_remove_known_lenses.py --no-default-extra   # KLC only (huang2020 + 559 inchausti2025 leak)
  python 397_remove_known_lenses.py --flag-only          # keep all 150k rows, add is_known

Provenance note (inherited from 396): `mean5` is the RANKING score the top-150k was
selected by (arithmetic mean of the 5 v4 members). It is NOT a calibrated P(lens) at
the survey base rate. Removing known lenses makes this list *unpublished*, not *pure* —
graded p_lens still comes only from the LensJudge vetting stage, which has not been
run on this set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

import _clib as C

REPO = C.REPRO.parent                      # <repo>/reproductions/claudenet -> <repo>
DEFAULT_KNOWN = REPO / "data" / "klc_clean250629ymh.csv"
DEFAULT_CANDS = C.DATA / "v3" / "dr11s_v4_candidates_150k.csv"
RADII = (1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0)   # sensitivity table

# The KLC is NOT a superset of the Huang-group's own four DESI searches. Verified
# entry-by-entry against the shipped list (2026-07-13):
#   Huang+2020 (DR7)     342 entries, 335 in KLC (7 absent -> staged separately)
#   Huang+2021 (DR8)   1,312 entries, 1,312 in KLC (fully covered)
#   Storfer+2024 (DR9) 1,895 entries, 1,895 in KLC (fully covered)
#   Inchausti+2025(DR10) 811 entries,     0 in KLC (ENTIRELY ABSENT -- the KLC,
#                        compiled 2025-06-29, predates/omits the DR10 systems)
# So we union ALL FOUR group catalogs by default, not just huang2020: relying on the
# KLC to contain them silently left 559 published Inchausti-2025 lenses in the list.
# Huang2021/Storfer2024 add 0 removals (already in KLC) but staging them makes the
# guarantee self-documenting instead of dependent on the KLC's contents.
# --no-default-extra restores KLC-only.
DEFAULT_EXTRA = (
    C.HUANG20 / "data" / "huang2020_published_catalog.csv",    # DR7
    C.HUANG21 / "data" / "huang2021_published_catalog.csv",    # DR8
    C.INCH / "data" / "storfer2024_published_catalog.csv",     # DR9
    C.INCH / "data" / "inchausti2025_published_catalog.csv",   # DR10 -- not in the KLC
    C.ROOT / "v3" / "external_lens_catalog.csv",
)


def rel(p: Path) -> str:
    """Repo-relative path when possible: the JSON report is committed, so it must not
    hard-code an absolute homedir."""
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def find_col(df: pd.DataFrame, *names: str) -> str:
    """Case-insensitive column lookup (KLC uses Ra/Dec, 396 uses RA/DEC)."""
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    raise SystemExit(f"[397] FATAL: none of {names} in columns {list(df.columns)}")


def sky_of(df: pd.DataFrame, ra: str, dec: str) -> SkyCoord:
    return SkyCoord(ra=df[ra].to_numpy(dtype=np.float64) * u.deg,
                    dec=df[dec].to_numpy(dtype=np.float64) * u.deg)


def drop_bad_coords(df: pd.DataFrame, ra: str, dec: str, tag: str) -> pd.DataFrame:
    """Non-finite or out-of-range coords would silently corrupt the match."""
    r = df[ra].to_numpy(dtype=np.float64)
    d = df[dec].to_numpy(dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(d) & (r >= 0) & (r <= 360) & (d >= -90) & (d <= 90)
    if not ok.all():
        print(f"[397] WARNING: dropping {int((~ok).sum()):,} {tag} rows with "
              f"non-finite / out-of-range coords")
    return df[ok].reset_index(drop=True)


KMETA = ["klc_ra", "klc_dec", "klc_index", "klc_ref", "klc_dup", "klc_catalog"]


def load_known(paths: list[Path]) -> pd.DataFrame:
    """Normalise every known-lens CSV to the KMETA schema and concat. Accepts the
    KLC (index, Ra, Dec, ref, klc_dup) and the repo's published catalogs
    (name, RA, DEC[, grade]). Cross-catalog duplicates are NOT deduped: the
    forward-NN removal stays exact regardless, and the removed rows record which
    catalog they matched."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        ra, dec = find_col(df, "Ra", "RA"), find_col(df, "Dec", "DEC")
        df = drop_bad_coords(df, ra, dec, f"known-lens[{p.name}]")
        out = pd.DataFrame({"klc_ra": df[ra].astype(np.float64),
                            "klc_dec": df[dec].astype(np.float64)})
        out["klc_index"] = (df["index"] if "index" in df.columns
                            else df["name"] if "name" in df.columns
                            else np.arange(len(df)))
        ref = next((c for c in ("ref", "name", "survey") if c in df.columns), None)
        out["klc_ref"] = df[ref].astype(str) if ref else ""
        out["klc_dup"] = df["klc_dup"] if "klc_dup" in df.columns else 1
        out["klc_catalog"] = p.stem
        frames.append(out[KMETA])
    known = pd.concat(frames, ignore_index=True)
    assert len(known), "no usable known-lens rows"
    return known


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", default=str(DEFAULT_CANDS),
                    help="396 output: rank, rank_within_new, row_id, RA, DEC, mean5, ...")
    ap.add_argument("--known", default=str(DEFAULT_KNOWN),
                    help="known-lens catalog CSV (Ra, Dec[, ref, klc_dup])")
    ap.add_argument("--extra-known", action="append", default=[],
                    help="additional known-lens CSV (RA,DEC[,name]) unioned into --known; "
                         "repeatable, on top of DEFAULT_EXTRA (cf. 163's --extra-catalog)")
    ap.add_argument("--no-default-extra", action="store_true",
                    help="do NOT auto-union the repo's own known-lens catalogs (DEFAULT_EXTRA); "
                         "match against --known alone. Yields 131,337 survivors instead of "
                         "131,336 -- one published huang2020 lens then leaks through.")
    ap.add_argument("--radius", type=float, default=5.0,
                    help="match radius in arcsec (repo default 5, cf. 163)")
    ap.add_argument("--out", default=None, help="survivors CSV")
    ap.add_argument("--removed-out", default=None, help="dropped rows + match metadata CSV")
    ap.add_argument("--report", default=None, help="JSON summary")
    ap.add_argument("--flag-only", action="store_true",
                    help="keep all rows, add is_known / sep_arcsec instead of removing")
    a = ap.parse_args()

    cand_p = Path(a.candidates)
    auto = [] if a.no_default_extra else [p for p in DEFAULT_EXTRA if p.exists()]
    known_ps, seen = [], set()
    for p in [Path(a.known), *auto, *(Path(x) for x in a.extra_known)]:
        k = p.resolve()                      # dedupe: re-passing a default via --extra-known
        if k not in seen:                    # would double-count it in the denominators
            seen.add(k)
            known_ps.append(p)
    for p in [cand_p, *known_ps]:
        if not p.exists():
            raise SystemExit(f"[397] FATAL: {p} missing")
    if auto:
        print(f"[397] auto-unioned {len(auto)} repo catalog(s): "
              f"{', '.join(p.name for p in auto)}  (--no-default-extra to disable)")
    stem = cand_p.with_suffix("")
    tag = "known_flagged" if a.flag_only else "noknown"   # per-mode paths: a --flag-only
    out_p = Path(a.out) if a.out else Path(f"{stem}_{tag}.csv")      # run must not clobber
    rep_p = Path(a.report) if a.report else Path(f"{stem}_{tag}_report.json")  # the remove-run
    rem_p = Path(a.removed_out) if a.removed_out else Path(f"{stem}_known_removed.csv")

    # ---- load ------------------------------------------------------------------
    cand = pd.read_csv(cand_p)
    c_ra, c_dec = find_col(cand, "RA", "Ra"), find_col(cand, "DEC", "Dec")
    n_cand_raw = len(cand)
    cand = drop_bad_coords(cand, c_ra, c_dec, "candidate")
    if "row_id" in cand.columns:
        assert cand.row_id.is_unique, "candidates: duplicate row_id"
    print(f"[397] candidates : {len(cand):,} rows from {cand_p}")

    known, n_known_raw = load_known(known_ps), 0
    for p in known_ps:
        n_known_raw += sum(1 for _ in open(p)) - 1
    k_ra, k_dec = "klc_ra", "klc_dec"
    for tag, n in known.klc_catalog.value_counts().items():
        print(f"[397] known lenses: {n:6,} rows from [{tag}]")
    if len(known_ps) > 1:
        print(f"[397] known lenses: {len(known):,} total (union; cross-catalog duplicates "
              f"NOT deduped -- harmless for removal, inflates the recall denominator)")

    # ---- forward NN (exact for removal) ----------------------------------------
    sky_c, sky_k = sky_of(cand, c_ra, c_dec), sky_of(known, k_ra, k_dec)
    fwd_idx, fwd_sep, _ = sky_c.match_to_catalog_sky(sky_k)
    sep = fwd_sep.to(u.arcsec).value

    # ---- reverse NN (exact for distinct-known-lens recovery) --------------------
    _, rev_sep, _ = sky_k.match_to_catalog_sky(sky_c)
    rsep = rev_sep.to(u.arcsec).value

    print(f"\n[397] radius sensitivity (candidates removed / known lenses recovered):")
    sens = {}
    for r in sorted(set(RADII) | {a.radius}):
        n_rm = int((sep < r).sum())
        n_kl = int((rsep < r).sum())
        sens[f"{r:g}"] = {"removed": n_rm, "survivors": len(cand) - n_rm,
                          "known_recovered": n_kl,
                          "known_recall_raw": round(n_kl / len(known), 6)}
        mark = "  <-- selected" if r == a.radius else ""
        print(f"[397]   {r:5.1f}\"  removed {n_rm:6,}  survivors {len(cand)-n_rm:7,}"
              f"  known recovered {n_kl:5,}/{len(known):,} ({n_kl/len(known)*100:5.2f}%){mark}")

    is_known = sep < a.radius
    n_rm = int(is_known.sum())
    n_within = int((rsep < a.radius).sum())          # reverse: KLC rows with >=1 cand within R
    n_nearest = int(pd.unique(fwd_idx[is_known]).size)  # forward: KLC rows that are some cand's NN
    grp = pd.Series(fwd_idx[is_known]).value_counts()
    n_deblend_sys = int((grp > 1).sum())
    n_extra_rows = n_rm - n_nearest                  # rows beyond one per nearest-match group
    print(f"\n[397] @ {a.radius:g}\": removed {n_rm:,} candidate rows")
    print(f"[397]   -> {n_nearest:,} distinct KLC entries are some removed row's nearest match; "
          f"{n_extra_rows:,} rows are extra")
    print(f"[397]      deblended sources on {n_deblend_sys:,} already-matched systems "
          f"({n_nearest:,}+{n_extra_rows:,}={n_rm:,})")
    print(f"[397]   -> {n_within:,} KLC entries lie within {a.radius:g}\" of >=1 candidate "
          f"({n_within - n_nearest:,} more than the nearest-match count:")
    print(f"[397]      near-duplicate literature rows for a lens already matched via its twin)")

    # ---- write ------------------------------------------------------------------
    # kmeta is row-aligned to `cand`: kmeta.iloc[i] is the nearest KLC entry to cand.iloc[i]
    kmeta = known[KMETA].iloc[fwd_idx].reset_index(drop=True)

    if a.flag_only:
        out = cand.copy()
        out["is_known"] = is_known.astype(int)
        out["sep_arcsec"] = sep
        mask = pd.Series(is_known, index=kmeta.index)
        for c in kmeta.columns:
            col = kmeta[c]
            if pd.api.types.is_integer_dtype(col):
                col = col.astype("Int64")   # nullable int: masking must NOT promote an
            out[c] = col.where(mask)        # integer catalog ID to float ("8949.0")
        out.to_csv(out_p, index=False)      # pd.NA / NaN -> empty field on unmatched rows
        print(f"[397] wrote {out_p}: {len(out):,} rows (is_known={n_rm:,})")
    else:
        clean = cand[~is_known].copy().reset_index(drop=True)
        clean["rank_clean"] = np.arange(1, len(clean) + 1)
        lead = [c for c in ("rank", "rank_within_new") if c in clean.columns]
        clean = clean[lead + ["rank_clean"] + [c for c in clean.columns
                                               if c not in lead + ["rank_clean"]]]
        clean.to_csv(out_p, index=False)
        print(f"[397] wrote {out_p}: {len(clean):,} survivors  (cols={list(clean.columns)})")

        rem = cand[is_known].copy().reset_index(drop=True)
        rem["sep_arcsec"] = sep[is_known]
        for c in kmeta.columns:
            rem[c] = kmeta[c].to_numpy()[is_known]
        rem = rem.sort_values("sep_arcsec").reset_index(drop=True)
        rem.to_csv(rem_p, index=False)
        print(f"[397] wrote {rem_p}: {len(rem):,} removed rows (auditable)")

        # ---- self-check: no survivor may sit within radius of a known lens -------
        smin = float("inf")
        if len(clean):
            _, chk_sep, _ = sky_of(clean, c_ra, c_dec).match_to_catalog_sky(sky_k)
            smin = float(chk_sep.to(u.arcsec).value.min())
            assert smin >= a.radius, f"self-check FAILED: survivor at {smin:.4f}\" < {a.radius}\""
        assert len(clean) + len(rem) == len(cand), "self-check FAILED: rows lost"
        if "row_id" in cand.columns:
            assert set(clean.row_id) | set(rem.row_id) == set(cand.row_id), \
                "self-check FAILED: row_id set changed"
            assert clean.row_id.is_unique, "self-check FAILED: duplicate survivor row_id"
        print(f"[397] self-check OK: {len(clean):,}+{len(rem):,}={len(cand):,}; "
              f"nearest survivor->KLC sep = {smin:.3f}\" >= {a.radius:g}\"")

    rep = {
        "candidates_in": {"path": rel(cand_p), "rows_raw": n_cand_raw, "rows_used": len(cand)},
        "known_in": {"paths": [rel(p) for p in known_ps], "rows_raw": n_known_raw,
                     "rows_used": len(known),
                     "per_catalog": {str(k): int(v) for k, v
                                     in known.klc_catalog.value_counts().items()}},
        "radius_arcsec": a.radius,
        "matcher": "astropy match_to_catalog_sky (great-circle); forward NN exact for removal",
        "mode": "flag-only" if a.flag_only else "remove",
        "outputs": {"primary": rel(out_p), **({} if a.flag_only else {"removed": rel(rem_p)})},
        "matched_known_rows": n_rm,                       # rows within radius, both modes
        "removed_rows": 0 if a.flag_only else n_rm,       # actually dropped
        "rows_written": len(cand) if a.flag_only else len(cand) - n_rm,
        "survivors": len(cand) - n_rm,
        "klc_entries_nearest_matched": n_nearest,
        "klc_entries_within_radius": n_within,
        "extra_deblended_rows": n_extra_rows,
        "deblended_systems": n_deblend_sys,
        "known_recall_raw": round(n_within / len(known), 6),
        "radius_sensitivity": sens,
        "caveats": [
            "klc_entries_within_radius counts KLC *rows*, not distinct physical systems: "
            "the KLC still holds near-duplicate literature entries (112 rows have another "
            "KLC row within 5\"), so the physical-system count is slightly lower.",
            "known_recall_raw denominator is the FULL KLC (all sky, incl. north and "
            "non-DESI area). The true in-footprint recall needs the sweep parent "
            "manifest, which lives on Perlmutter -- this is a lower bound.",
            "The input is 396's NEW subset: the 15,922 top-150k rows retained from the "
            "old union-95k survivors are already excluded, and known lenses found by "
            "the previous sweep live disproportionately in that excluded bucket. So "
            "recall against this file understates v4's true recall of known lenses.",
            "mean5 is a RANKING score, not a calibrated P(lens). Removing the KLC makes "
            "the list unpublished, not pure; vetting (LensJudge) has not been run.",
        ],
    }
    rep_p.write_text(json.dumps(rep, indent=2))
    print(f"[397] wrote {rep_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
