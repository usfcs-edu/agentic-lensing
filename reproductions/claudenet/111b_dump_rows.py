#!/usr/bin/env python3
"""111b_dump_rows.py — Phase 110 helper: pull selected rows from cutout shards
into one compressed npz for rsync back (runs ON Perlmutter, CPU-only).

Given an extraction --out-root (from 111_extract_cutouts_cfs.py) and a
--row-ids parquet/csv with a row_id column, gathers those cutouts from the
memmap shards into a single npz with arrays: cutouts (n, n_bands, size, size)
float32, row_ids (n,) str, ok (n,) bool — preserving the requested order.
Used for the 100K local audit subset and the top-200 purity audit.

    python 111b_dump_rows.py --out-root $SCRATCH/claudenet/cutouts/negeval \
        --row-ids audit_rowids.parquet --out negeval_audit.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def read_row_ids(path: str) -> list:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    return df["row_id"].astype(str).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--row-ids", required=True, help="parquet/csv with row_id column")
    ap.add_argument("--out", default=None, help="output npz (default <out-root>/dump_rows.npz)")
    args = ap.parse_args()
    t0 = time.time()
    out_root = Path(args.out_root)
    out = Path(args.out) if args.out else out_root / "dump_rows.npz"

    index = pd.read_parquet(out_root / "index.parquet")
    assert index.row_id.is_unique, "duplicate row_ids in index.parquet (stale shards?)"
    want = read_row_ids(args.row_ids)
    loc = index.set_index("row_id")[["shard", "idx_in_shard", "ok"]]
    have = [r for r in want if r in loc.index]
    if len(have) < len(want):
        print(f"[dump] WARNING: {len(want) - len(have):,} requested row_ids not in index "
              f"-> skipped")
    if not have:
        print("[dump] FATAL: none of the requested row_ids are in the index")
        return 1
    sub = loc.loc[have]
    assert len(sub) == len(have), "index lookup misaligned (duplicate row_ids?)"
    print(f"[dump] gathering {len(sub):,} rows from "
          f"{sub.shard.nunique()} shard(s) under {out_root}")

    cut = None
    order = np.arange(len(sub))
    for k, g in sub.assign(_pos=order).groupby("shard"):
        mm = np.load(out_root / f"cutouts_{int(k)}.npy", mmap_mode="r")
        if cut is None:
            cut = np.empty((len(sub),) + mm.shape[1:], dtype=np.float32)
        srt = np.argsort(g.idx_in_shard.values)  # sorted reads, original placement
        cut[g._pos.values[srt]] = mm[g.idx_in_shard.values[srt]]
        del mm
    np.savez_compressed(out, cutouts=cut,
                        row_ids=np.array(have, dtype=str),
                        ok=sub.ok.values.astype(bool))
    print(f"[done] {len(sub):,} rows, cutouts shape {cut.shape} -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
