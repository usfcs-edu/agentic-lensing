#!/usr/bin/env python3
"""210_fetch_external_catalogs.py — pull external strong-lens catalogs from the
VizieR TAP service (the asu-tsv data endpoint and astroquery.Vizier.get_catalogs
both hang from this host; the TAP sync endpoint is responsive) into RA,DEC,name
CSVs for 163's --extra-catalog crossmatch.

These literature lens catalogs are NOT in the 3 local DECaLS CSVs 163 auto-loads
(storfer2024/inchausti2025/huang2021); matching the 737 against them catches
published lenses the local sets missed. SIMBAD cone search (163 --remote) is the
primary external check; NED is deferred (unreachable from this host).

Default VizieR catalogs (editable via --codes tag=Code):
  J/A+A/685/A34    SuGOHI (Schuldt+2024)  [HSC-heavy; limited DECaLS-south overlap]
  J/ApJS/243/17    DES strong-lens candidates (Jacobs+2019)
  J/MNRAS/484/3879 KiDS LinKS (Petrillo+2019)

Writes data/v2/campaign/ext_catalogs/<tag>.csv (RA,DEC,name). Resumable: an
existing non-empty CSV is skipped. Per-query hard timeout; a catalog that can't
be resolved is skipped and reported (documented, not silently dropped).

    /home2/benson/.venvs/claudenet/bin/python campaign/210_fetch_external_catalogs.py
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "campaign" / "ext_catalogs"
TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
TIMEOUT = 40

DEFAULT_CODES = {
    "sugohi_schuldt2024": "J/A+A/685/A34",
    "des_jacobs2019": "J/ApJS/243/17",
    "kids_links2019": "J/MNRAS/484/3879",
}
RA_PREF = ["RAJ2000", "RA_ICRS", "RAdeg", "_RAJ2000", "RAICRS"]
DEC_PREF = ["DEJ2000", "DE_ICRS", "DEdeg", "_DEJ2000", "DEICRS"]


def tap(query: str) -> pd.DataFrame | None:
    try:
        r = requests.get(TAP, params={"request": "doQuery", "lang": "ADQL",
                                       "format": "csv", "query": query}, timeout=TIMEOUT)
        if r.status_code != 200 or r.text.lstrip().startswith("<"):
            return None
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None


def _clean(s: str) -> str:
    # VizieR TAP_SCHEMA stores table/column names wrapped in literal single quotes
    return str(s).strip().strip("'").strip()


def discover(code: str):
    """Return (table_name, ra_col, dec_col, id_col) for the catalog's best
    coordinate table, via TAP_SCHEMA (names come back single-quote-wrapped)."""
    tabs = tap(f"SELECT table_name FROM TAP_SCHEMA.tables "
               f"WHERE table_name LIKE '%{code}/%'")
    if tabs is None or not len(tabs):
        return None
    for raw in tabs["table_name"]:
        t = _clean(raw)
        cols = tap(f"SELECT column_name FROM TAP_SCHEMA.columns "
                   f"WHERE table_name LIKE '%{t}%'")
        if cols is None or not len(cols):
            continue
        names = [_clean(c) for c in cols["column_name"]]
        ra = next((c for c in RA_PREF if c in names), None)
        dec = next((c for c in DEC_PREF if c in names), None)
        if ra and dec:
            idc = next((c for c in names if c.lower() in
                        ("name", "sugohi", "id", "lens", "desj", "objname", "simbad")), None)
            return t, ra, dec, idc
    return None


def fetch_one(tag: str, code: str) -> pd.DataFrame | None:
    info = discover(code)
    if info is None:
        print(f"[210] {tag} ({code}): no coord table resolved via TAP -> skip")
        return None
    t, ra, dec, idc = info
    sel = f'"{ra}","{dec}"' + (f',"{idc}"' if idc else "")
    df = tap(f'SELECT {sel} FROM "{t}"')
    if df is None or not len(df):
        print(f"[210] {tag}: TAP data query failed for {t}")
        return None
    out = pd.DataFrame({"RA": pd.to_numeric(df[ra], errors="coerce"),
                        "DEC": pd.to_numeric(df[dec], errors="coerce")})
    out["name"] = (df[idc].astype(str) if idc and idc in df else
                   [f"{tag}_{i}" for i in range(len(df))])
    out = out[out.RA.notna() & out.DEC.notna()].drop_duplicates(["RA", "DEC"])
    print(f"[210] {tag} ({code}): table {t} cols ({ra},{dec},{idc}) -> {len(out):,} rows")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="*", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    codes = (DEFAULT_CODES if not args.codes
             else dict(c.split("=", 1) for c in args.codes))

    summary = []
    for tag, code in codes.items():
        f = OUT / f"{tag}.csv"
        if f.exists() and f.stat().st_size > 50:
            n = len(pd.read_csv(f))
            print(f"[210] {tag}: cached ({n}) -> skip")
            summary.append((tag, n, "cached"))
            continue
        df = fetch_one(tag, code)
        if df is None or not len(df):
            summary.append((tag, 0, "FAILED"))
            continue
        df.to_csv(f, index=False)
        summary.append((tag, len(df), "ok"))

    print("\n[210] summary:")
    for tag, n, st in summary:
        print(f"  {tag:24s} {n:>8,}  {st}")
    ready = [OUT / f"{t}.csv" for t, n, s in summary if s in ("ok", "cached")]
    print(f"[210] {len(ready)} catalogs ready -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
