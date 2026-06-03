#!/usr/bin/env python3
"""
00_fetch_published_catalog.py

Download the published Dawes+2023 (ApJS 269:61) candidate catalog from VizieR
(J/ApJS/269/61, table2) via astroquery. The neuralens project website
(sites.google.com/usfca.edu/neuralens) does NOT host a downloadable catalog,
but the journal's machine-readable Table 2 is mirrored on VizieR.

table2 = 875 rows (1-2 image rows per system) for 436 unique candidate systems:
  Index, Name (DESI-{RA}{+-DEC}), Grade (A/B/C), zsp, zph, g/r/z mag,
  PXSig, PMSig, Type (Double/Quad), Sep (image separation, arcsec), _RA, _DE.

Output: data/dawes2023_vizier_table2.csv
"""
from __future__ import annotations

from pathlib import Path

from astroquery.vizier import Vizier


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
OUT = DATA / "dawes2023_vizier_table2.csv"


def main() -> None:
    Vizier.ROW_LIMIT = -1
    res = Vizier.get_catalogs("J/ApJS/269/61")
    t = res[0]
    df = t.to_pandas()
    print(f"[vizier] {len(df)} rows, {df['Name'].nunique()} unique systems")
    print(f"[vizier] Grade: {df['Grade'].value_counts().to_dict()}")
    print(f"[vizier] Type:  {df['Type'].value_counts().to_dict()}")
    df.to_csv(OUT, index=False)
    print(f"[save] wrote {OUT}")


if __name__ == "__main__":
    main()
