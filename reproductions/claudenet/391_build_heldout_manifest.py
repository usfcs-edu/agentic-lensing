#!/usr/bin/env python3
"""391_build_heldout_manifest.py — held-out recall-catalog manifest (Storfer2024 + Inchausti2025)
for the DR11 gate. These are the catalogs EXCLUDED from training; we extract their DR11 cutouts so
the fine-tuned members can be scored on the same held-out lenses the diagnosis used. Runs ON
PERLMUTTER (reads the catalog CSVs there). Output feeds 385 (brick-assign) -> 111 (extract).
"""
from __future__ import annotations
import re
import pandas as pd

CATDIR = "/global/homes/g/gdbenson/claudenet/data"


def slug(s):
    return re.sub(r"[^0-9A-Za-z]+", "_", str(s)).strip("_")[:36]


def main():
    frames = []
    for cat, f in [("sto", "storfer2024_published_catalog.csv"),
                   ("inc", "inchausti2025_published_catalog.csv")]:
        d = pd.read_csv(f"{CATDIR}/{f}")
        frames.append(pd.DataFrame({
            "row_id": [f"{cat}_{i}_{slug(n)}" for i, n in enumerate(d.name)],
            "RA": d.RA, "DEC": d.DEC, "grade": d.grade.astype(str), "cat": cat}))
    m = pd.concat(frames, ignore_index=True).dropna(subset=["RA", "DEC"])
    m = m[m.DEC <= 32.375].reset_index(drop=True)
    m.to_parquet("dr11s_heldout_manifest.parquet", index=False)
    print(f"[391] held-out south: {len(m)}  {m.groupby(['cat','grade']).size().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
