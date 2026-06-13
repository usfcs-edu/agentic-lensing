#!/usr/bin/env python3
"""163_crossmatch_known.py — Phase 160 (DR9 full sweep): crossmatch the stage-2
survivors against every known-lens catalog -> NEW vs KNOWN status + the
recall-of-known-lenses sanity numbers (runs LOCALLY, CPU).

The sweep deliberately keeps known lenses in the parent sample (160's
population choice), so recovered catalog lenses are the recall sanity check
and everything unmatched is a genuinely-NEW candidate for 164/165.

  1. LOCAL crossmatch (always): nearest neighbour of each survivor in each of
     the four local catalogs (storfer2024/inchausti2025/huang2021 published
     CSVs via _clib.known_lens_catalogs + v1 positives_curated.parquet — the
     same files 110 used for its exclusion) at --radius (5" default), with
     astropy match_to_catalog_sky per catalog (the 114 pattern). known_local =
     any match; union columns nearest_sep_arcsec/nearest_catalog/nearest_name.
  2. Recall accounting: reverse match (catalog entry -> nearest survivor).
     Denominator = catalog entries within --radius of WHAT WAS ACTUALLY SWEPT:
     --coverage defaults to 'auto' = the union of the 160 sweep part manifests
     (falling back to data/parent_dr8.parquet only when no manifests exist —
     numerically identical today since the NOBS filter drops 0 rows, but the
     parent goes STALE if --min-nobs changes or a partial sweep ran with
     --allow-partial; the json records which denominator was used). Entries
     outside the coverage CANNOT be recovered by this sweep, so they are
     excluded from the recall denominator -> crossmatch_recall.json, which
     also exports each catalog's in-coverage entry indices (in_coverage_idx)
     so 164 can compute its selected-set recall on the SAME population. The
     numerator here is vs ALL stage-2 survivors; 164 recomputes it on the
     FINAL FDR-selected set.
  3. --remote (optional): astroquery NED + SIMBAD cone searches for the top
     --top-remote locally-UNMATCHED survivors by --score-col only.
     Rate-limited (--remote-delay s between queries); every reply is appended
     to --remote-cache (jsonl) immediately, so a killed run RESUMES where it
     stopped and cached rows are never re-queried (--remote-retry-errors
     retries only the error rows). A row is KNOWN_REMOTE when any returned
     object type is lens-like (SIMBAD otype in {gLe,gLS,LeI,LeG,LeQ,Le?,LI?,
     LS?} or any type string containing 'lens'); raw types are cached and
     written — the flag is ADVISORY (humans see 164's vetting pages anyway).

status precedence: KNOWN_LOCAL > KNOWN_REMOTE > NEW (= unmatched everywhere).

Inputs: the 162 stage-2 scores parquet (row_id, RA, DEC, footprint, ...,
p_final). Outputs: data/v2/sweep/crossmatch.parquet + crossmatch_recall.json.

    python 163_crossmatch_known.py                       # local catalogs only
    python 163_crossmatch_known.py --remote --top-remote 500
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

import _clib as C

V2 = C.DATA / "v2"
SWEEP = V2 / "sweep"
SIMBAD_LENS_OTYPES = {"gLe", "gLS", "LeI", "LeG", "LeQ", "Le?", "LI?", "LS?"}
_SIMBAD = None  # lazy astroquery.Simbad instance (votable fields configured once)


def sky_of(df) -> SkyCoord:
    return SkyCoord(ra=df.RA.to_numpy(dtype=np.float64) * u.deg,
                    dec=df.DEC.to_numpy(dtype=np.float64) * u.deg)


# ===== local catalogs ============================================================

def load_catalogs(extra_csvs=()) -> dict[str, pd.DataFrame]:
    """The four local known-lens catalogs (the 110-exclusion set) + any
    --extra-catalog CSVs. Returns {tag: df[RA, DEC, name]}, NaN coords dropped."""
    cats = {}

    def add(tag, df, src):
        df = df.dropna(subset=["RA", "DEC"]).reset_index(drop=True)
        if "name" not in df.columns:
            df["name"] = [f"{tag}_{i}" for i in range(len(df))]
        assert len(df), f"catalog {src} has no usable RA/DEC rows"
        cats[tag] = df[["RA", "DEC", "name"]].copy()

    for p in C.known_lens_catalogs():
        add(p.name.split("_")[0], pd.read_csv(p), p)
    cur = C.DATA / "positives_curated.parquet"
    if cur.exists():
        add("curated", pd.read_parquet(cur), cur)
    else:
        print(f"[163] WARNING: {cur} missing -> curated positives not crossmatched")
    for pth in extra_csvs:
        p = Path(pth)
        assert p.exists(), f"--extra-catalog {p} missing"
        add(p.stem, pd.read_csv(p), p)
    assert cats, "no known-lens catalogs found (run from the claudenet dir?)"
    return cats


def match_local(surv: pd.DataFrame, cats: dict, radius: float) -> pd.DataFrame:
    """Per-catalog nearest-neighbour columns (sep_/match_/name_<tag>) + the
    union (known_local, nearest_sep_arcsec/nearest_catalog/nearest_name)."""
    sky = sky_of(surv)
    out = surv.copy()
    near_sep = np.full(len(out), np.inf)
    near_cat = np.full(len(out), "", dtype=object)
    near_name = np.full(len(out), "", dtype=object)
    for tag, cat in cats.items():
        idx, sep, _ = sky.match_to_catalog_sky(sky_of(cat))
        s = sep.to(u.arcsec).value
        m = s < radius
        out[f"sep_{tag}"] = s
        out[f"match_{tag}"] = m
        out[f"name_{tag}"] = np.where(m, cat["name"].to_numpy(dtype=object)[idx], "")
        upd = s < near_sep
        near_sep[upd] = s[upd]
        near_cat[upd] = tag
        near_name[upd] = cat["name"].to_numpy(dtype=object)[idx[upd]]
        print(f"[local] {tag:14s} {len(cat):6,} entries -> "
              f"{int(m.sum()):6,}/{len(out):,} survivors within {radius:g}\"")
    out["known_local"] = np.column_stack(
        [out[f"match_{t}"].to_numpy() for t in cats]).any(axis=1)
    out["nearest_sep_arcsec"] = near_sep
    out["nearest_catalog"] = near_cat.astype(str)
    out["nearest_name"] = near_name.astype(str)
    print(f"[local] union: {int(out.known_local.sum()):,}/{len(out):,} survivors "
          f"match a known lens within {radius:g}\"")
    return out


def resolve_coverage(coverage: str) -> tuple[pd.DataFrame | None, str]:
    """--coverage 'auto' -> union of the 160 sweep part manifests (what was
    ACTUALLY swept), falling back to the raw parent; '' disables; an explicit
    path is used as-is. Returns (positions df or None, provenance note)."""
    if not coverage:
        return None, "disabled ('' passed) -> raw denominators only"
    if coverage != "auto":
        cp = Path(coverage)
        if not cp.exists():
            print(f"[recall] WARNING: --coverage {cp} missing -> recall denominators "
                  f"are the RAW catalog sizes (out-of-footprint entries included)")
            return None, f"{cp} MISSING -> raw denominators only"
        cov = pd.read_parquet(cp, columns=["RA", "DEC"])
        return cov, f"{cp} ({len(cov):,} positions; explicit --coverage)"
    parts = sorted(SWEEP.glob("sweep_manifest_part*of*.parquet"))
    if parts:
        cov = pd.concat([pd.read_parquet(p, columns=["RA", "DEC"]) for p in parts],
                        ignore_index=True)
        return cov, (f"union of {len(parts)} 160 sweep part manifests "
                     f"({len(cov):,} positions — exactly what was swept)")
    pp = C.DATA / "parent_dr8.parquet"
    if pp.exists():
        cov = pd.read_parquet(pp, columns=["RA", "DEC"])
        return cov, (f"{pp} ({len(cov):,} positions; FALLBACK — no 160 manifests "
                     f"found; stale if --min-nobs changed or a partial sweep ran)")
    return None, "auto: no 160 manifests and no parent -> raw denominators only"


def recall_accounting(cats: dict, surv: pd.DataFrame, cov: pd.DataFrame | None,
                      cov_note: str, radius: float) -> dict:
    """Catalog-entry-level recall into the survivor set (reverse match), with
    the in-coverage denominator when coverage positions are available. The
    per-catalog in-coverage entry indices are EXPORTED (in_coverage_idx) so
    164's selected-set recall can use the identical population. union_all =
    all catalogs concatenated (cross-catalog duplicates double-counted)."""
    surv_sky = sky_of(surv)
    cov_sky = None
    if cov is not None:
        cov_sky = sky_of(cov)
        print(f"[recall] coverage = {cov_note}")
    rep = {"radius_arcsec": radius, "coverage": cov_note,
           "n_survivors": int(len(surv)), "catalogs": {}}
    work = dict(cats)
    work["union_all"] = pd.concat(list(cats.values()), ignore_index=True)
    for tag, cat in work.items():
        cat_sky = sky_of(cat)
        _, sep_s, _ = cat_sky.match_to_catalog_sky(surv_sky)
        rec = sep_s.to(u.arcsec).value < radius
        ent = {"n_entries": int(len(cat)),
               "n_recovered_in_survivors": int(rec.sum()),
               "recall_raw": float(rec.mean())}
        if cov_sky is not None:
            _, sep_c, _ = cat_sky.match_to_catalog_sky(cov_sky)
            inc = sep_c.to(u.arcsec).value < radius
            ent["n_in_coverage"] = int(inc.sum())
            ent["n_recovered_in_coverage"] = int((rec & inc).sum())
            ent["recall_in_coverage"] = (float((rec & inc).sum() / inc.sum())
                                         if inc.any() else float("nan"))
            ent["in_coverage_idx"] = np.flatnonzero(inc).tolist()
        rep["catalogs"][tag] = ent
    rep["note"] = ("union_all concatenates the catalogs without deduplication; "
                   "numerator = stage-2 SURVIVORS (164 recomputes it on the "
                   "final FDR-selected set, restricted to in_coverage_idx). "
                   "Denominator caveat: entries in unswept/unscored parts stay "
                   "in the denominator as 'recoverable' — pass an explicit "
                   "--coverage if a partial sweep ran (162 --allow-partial) or "
                   "--min-nobs changed.")
    print(f"\n[recall] known-lens recovery into the stage-2 survivor set "
          f"({radius:g}\" match):")
    for tag, e in rep["catalogs"].items():
        line = (f"[recall]   {tag:14s} {e['n_recovered_in_survivors']:6,}"
                f"/{e['n_entries']:6,} raw ({e['recall_raw']:.3f})")
        if "recall_in_coverage" in e:
            line += (f"  in-coverage {e['n_recovered_in_coverage']:6,}"
                     f"/{e['n_in_coverage']:6,} ({e['recall_in_coverage']:.3f})")
        print(line)
    return rep


# ===== remote (NED/SIMBAD, cached + resumable) ===================================

def _is_lens_type(t: str) -> bool:
    return t in SIMBAD_LENS_OTYPES or "lens" in t.lower()


def _query_simbad(sky: SkyCoord, radius: float):
    global _SIMBAD
    from astroquery.simbad import Simbad
    if _SIMBAD is None:
        _SIMBAD = Simbad()
        try:
            _SIMBAD.add_votable_fields("otype")
        except Exception:
            pass
    t = _SIMBAD.query_region(sky, radius=radius * u.arcsec)
    if t is None or len(t) == 0:
        return 0, []
    cols = {c.lower(): c for c in t.colnames}
    oc = cols.get("otype") or cols.get("otypes") or cols.get("otype_s")
    types = [str(x) for x in t[oc]] if oc else []
    return len(t), types


def _query_ned(sky: SkyCoord, radius: float):
    from astroquery.ipac.ned import Ned
    t = Ned.query_region(sky, radius=radius * u.arcsec)
    if t is None or len(t) == 0:
        return 0, []
    oc = next((c for c in t.colnames if c.lower() == "type"), None)
    types = [x.decode() if isinstance(x, bytes) else str(x)
             for x in t[oc]] if oc else []
    return len(t), types


REMOTE_QUERY = {"simbad": _query_simbad, "ned": _query_ned}


def load_remote_cache(path: Path) -> dict:
    """jsonl -> {(row_id, service): record} (last record per key wins)."""
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[(str(r.get("row_id")), r.get("service"))] = r
    return cache


def remote_crossmatch(top: pd.DataFrame, services, radius: float, delay: float,
                      cache_path: Path, retry_errors: bool) -> dict:
    """Cone-search the top rows against NED/SIMBAD with an append-only jsonl
    cache: cached ok rows are skipped (resumable), each reply is flushed to
    disk immediately, queries are spaced by `delay` seconds."""
    try:
        import astroquery  # noqa: F401  (lazy: --remote only)
    except ImportError:
        raise SystemExit("[163] FATAL: --remote needs astroquery "
                         "(uv pip install astroquery into the claudenet venv)")
    cache = load_remote_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    n_query = n_cached = n_err = 0
    with open(cache_path, "a") as fh:
        for r in top.itertuples():
            sky = SkyCoord(ra=float(r.RA) * u.deg, dec=float(r.DEC) * u.deg)
            for svc in services:
                key = (str(r.row_id), svc)
                hit = cache.get(key)
                if hit is not None and (hit.get("status") == "ok" or not retry_errors):
                    n_cached += 1
                    continue
                if n_query:
                    time.sleep(delay)
                rec = {"row_id": str(r.row_id), "service": svc,
                       "ra": float(r.RA), "dec": float(r.DEC),
                       "radius_arcsec": radius, "ts": time.time()}
                try:
                    n_obj, types = REMOTE_QUERY[svc](sky, radius)
                    lens_types = sorted({t for t in types if _is_lens_type(t)})
                    rec.update(status="ok", n_obj=int(n_obj), types=types[:50],
                               lens_types=lens_types, lens=bool(lens_types))
                except Exception as ex:
                    rec.update(status="error", err=f"{type(ex).__name__}: {ex}")
                    n_err += 1
                n_query += 1
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                cache[key] = rec
                if n_query % 50 == 0:
                    print(f"[remote] {n_query} queries so far ({n_err} errors) ...")
    print(f"[remote] {n_query} live queries ({n_err} errors), {n_cached} served "
          f"from cache -> {cache_path}")
    return cache


def remote_columns(out: pd.DataFrame, cache: dict, services) -> pd.DataFrame:
    """Aggregate the cache into per-row columns (rows never queried keep
    remote_queried=False; errors leave a service un-ok -> queried stays False)."""
    queried, lens, n_obj, types = [], [], [], []
    for rid in out.row_id.astype(str):
        recs = [cache.get((rid, s)) for s in services]
        ok = [r for r in recs if r is not None and r.get("status") == "ok"]
        queried.append(len(ok) == len(services))
        lens.append(any(r.get("lens") for r in ok))
        n_obj.append(int(sum(r.get("n_obj", 0) for r in ok)))
        types.append(";".join(f"{r['service']}:" + "|".join(r.get("types", [])[:8])
                              for r in ok))
    out = out.copy()
    out["remote_queried"] = np.asarray(queried, bool)
    out["remote_lens"] = np.asarray(lens, bool)
    out["remote_n_obj"] = np.asarray(n_obj, np.int64)
    out["remote_types"] = types
    return out


# ===== main ======================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage2-scores", default=str(SWEEP / "stage2_scores.parquet"),
                    help="162 output: row_id, RA, DEC, footprint, ..., p_final")
    ap.add_argument("--score-col", default="p_final",
                    help="final score column (remote top-K ranking)")
    ap.add_argument("--radius", type=float, default=5.0, help="match radius (arcsec)")
    ap.add_argument("--coverage", default="auto",
                    help="recall-denominator positions: 'auto' = union of the 160 "
                         "sweep part manifests (what was actually swept; falls "
                         "back to data/parent_dr8.parquet), a parquet path, or "
                         "'' to disable")
    ap.add_argument("--extra-catalog", action="append", default=[],
                    help="extra known-lens CSV (RA,DEC[,name]); repeatable")
    ap.add_argument("--remote", action="store_true",
                    help="NED/SIMBAD cone searches for the top finalists")
    ap.add_argument("--top-remote", type=int, default=500,
                    help="how many locally-unmatched top rows to query remotely")
    ap.add_argument("--remote-services", default="ned,simbad")
    ap.add_argument("--remote-delay", type=float, default=1.0,
                    help="seconds between remote queries (rate limit)")
    ap.add_argument("--remote-cache", default=str(SWEEP / "remote_cache.jsonl"))
    ap.add_argument("--remote-retry-errors", action="store_true",
                    help="re-query cached rows whose last status was error")
    ap.add_argument("--out", default=str(SWEEP / "crossmatch.parquet"))
    ap.add_argument("--recall-out", default=str(SWEEP / "crossmatch_recall.json"))
    args = ap.parse_args()
    t0 = time.time()
    SWEEP.mkdir(parents=True, exist_ok=True)

    surv = pd.read_parquet(args.stage2_scores)
    for col in ("row_id", "RA", "DEC", args.score_col):
        assert col in surv.columns, f"{args.stage2_scores}: missing column {col!r}"
    surv = surv.copy()
    surv["row_id"] = surv["row_id"].astype(str)
    assert surv.row_id.is_unique, "stage-2 scores: duplicate row_ids"
    bad = ~(np.isfinite(surv.RA.to_numpy(np.float64))
            & np.isfinite(surv.DEC.to_numpy(np.float64)))
    if bad.any():
        print(f"[163] WARNING: dropping {int(bad.sum()):,} survivors with "
              f"non-finite RA/DEC")
        surv = surv[~bad].reset_index(drop=True)
    keep = [c for c in ("row_id", "RA", "DEC", "footprint", "brick",
                        "p_stage1", args.score_col) if c in surv.columns]
    surv = surv[keep]
    print(f"[163] {len(surv):,} stage-2 survivors from {args.stage2_scores}")

    # -- 1. local crossmatch + 2. recall ----------------------------------------
    cats = load_catalogs(args.extra_catalog)
    out = match_local(surv, cats, args.radius)
    cov, cov_note = resolve_coverage(args.coverage)
    recall = recall_accounting(cats, surv, cov, cov_note, args.radius)
    del cov

    # -- 3. remote (top locally-unmatched finalists only) ------------------------
    services = [s for s in args.remote_services.split(",") if s]
    assert set(services) <= set(REMOTE_QUERY), f"unknown service in {services}"
    if args.remote:
        top = (out[~out.known_local]
               .sort_values(args.score_col, ascending=False)
               .head(args.top_remote))
        print(f"[remote] querying {services} for the top {len(top):,} "
              f"locally-unmatched survivors (delay {args.remote_delay:g}s)")
        cache = remote_crossmatch(top, services, args.radius, args.remote_delay,
                                  Path(args.remote_cache), args.remote_retry_errors)
        out = remote_columns(out, cache, services)
    else:
        out["remote_queried"] = False
        out["remote_lens"] = False
        out["remote_n_obj"] = 0
        out["remote_types"] = ""

    out["status"] = np.where(out.known_local, "KNOWN_LOCAL",
                             np.where(out.remote_lens, "KNOWN_REMOTE", "NEW"))
    n = out.status.value_counts()
    print(f"\n[163] status: " + ", ".join(f"{k}={int(v):,}" for k, v in n.items())
          + f" (remote_queried={int(out.remote_queried.sum()):,})")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_p, index=False)
    recall["status_counts"] = {k: int(v) for k, v in n.items()}
    recall["n_remote_queried"] = int(out.remote_queried.sum())
    recall["stage2_scores"] = str(args.stage2_scores)
    Path(args.recall_out).write_text(json.dumps(recall, indent=2))
    print(f"[163] wrote {out_p} + {args.recall_out} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
