#!/usr/bin/env python3
"""313_stage_mimic_fits.py — ClaudeNet v3 A2: stage mimic-bank cutouts as per-row FITS on
Perlmutter, gathered from the DR10 sweep .npy shards.

The C-now DR10 sweep extracted all 43.7M parent cutouts to $SCRATCH/claudenet/cutouts/
dr10/part{00..15}/ (cutouts_<k>.npy memmap shards + index.parquet keyed by row_id;
111_extract_cutouts_cfs.py). The mimic bank (312) is a subset of the NEW survivors, so its
cutouts already live in those shards. The v1 loaders / 121 retraining consume per-row FITS
(<fits_dir>/<row_id>.fits), so this gathers the requested rows out of the memmap shards
(111b_dump_rows.py logic) and writes them as FITS with the byte-exact v1 format
(120b_unpack_npz_to_fits.to_bytes). One mode per run:

  --mode train-pool : top-N stratified-hardest TRAIN mimics for the 121 blend. ALL rows of
                      the rare campaign-typed contaminants (lrg_companion/ring/merger/blend/
                      spiral/...), then the hardest BULK mimics by p_final (the v2-lean
                      ensemble score) with a per-type diversity cap, over-weighting
                      extended_lrg (the staf327 regime). -> data/v2/mimic_bank_fits/ (under
                      data/v2 so 121's basename-remap of mined fits_dir resolves it).
  --mode eval       : ALL held-out mimic-eval rows (source==dr10_sweep) — the metric's
                      negative denominator. -> data/v3/mimic_eval_fits/  (used only if you
                      want FITS eval; 315 scores eval straight from the shards, no FITS.)

Writes <out-dir>/<row_id>.fits + a manifest parquet [row_id,label=0,fits_dir,mimic_type].

    python 313_stage_mimic_fits.py --mode train-pool --n 8000
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C

M120b = C._load("cn_313_120b", C.ROOT / "120b_unpack_npz_to_fits.py")   # to_bytes

# campaign-typed rare contaminants (taken in full) vs the bulk CNN-high types
BULK = {"cnn_high_other", "cnn_high_unknown", "extended_lrg", "compact_rex", "exp_disk"}


def select_train_pool(bank: pd.DataFrame, n: int, per_type_cap_frac: float = 0.40) -> pd.DataFrame:
    """Rare types in full + hardest-by-p_final bulk, capped per bulk type for diversity."""
    b = bank[bank["source"] == "dr10_sweep"].copy()
    b["row_id"] = b["row_id"].astype(str)
    rare = b[~b["mimic_type"].isin(BULK)].sort_values("p_final", ascending=False)
    bulk = b[b["mimic_type"].isin(BULK)].sort_values("p_final", ascending=False)
    take = [rare]                                   # all rare campaign contaminants
    remaining = max(0, n - len(rare))
    cap = int(round(per_type_cap_frac * n))
    picked = []
    for t, g in bulk.groupby("mimic_type", sort=False):
        picked.append(g.head(cap))                  # hardest `cap` of each bulk type
    bulk_capped = pd.concat(picked).sort_values("p_final", ascending=False).head(remaining)
    take.append(bulk_capped)
    pool = pd.concat(take).drop_duplicates("row_id").head(n).reset_index(drop=True)
    print(f"[313] train-pool: {len(pool):,} rows "
          f"(rare {len(rare):,} all-in, bulk-capped {len(bulk_capped):,}); types: "
          f"{pool['mimic_type'].value_counts().to_dict()}")
    return pool[["row_id", "mimic_type", "p_final"]]


def gather_fits(rows: pd.DataFrame, parts_glob: str, out_dir: Path) -> pd.DataFrame:
    """Gather each row's cutout from whichever DR10 part holds it (per-part index.parquet)
    and write <out_dir>/<row_id>.fits. Returns the written manifest [row_id,label,fits_dir]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(rows["row_id"].astype(str))
    type_of = dict(zip(rows["row_id"].astype(str), rows["mimic_type"]))
    parts = sorted(glob.glob(parts_glob))
    assert parts, f"no parts under {parts_glob}"
    written, t0 = [], time.time()
    for p in parts:
        idx = pd.read_parquet(Path(p) / "index.parquet", columns=["row_id", "shard", "idx_in_shard", "ok"])
        idx["row_id"] = idx["row_id"].astype(str)
        sub = idx[idx["row_id"].isin(want) & idx["ok"]].copy()
        if sub.empty:
            continue
        for k, g in sub.groupby("shard"):
            mm = np.load(Path(p) / f"cutouts_{int(k)}.npy", mmap_mode="r")
            order = np.argsort(g["idx_in_shard"].values)          # sorted reads
            ids = g["row_id"].values[order]
            cubes = mm[g["idx_in_shard"].values[order]]
            for rid, cube in zip(ids, cubes):
                cube = np.asarray(cube, dtype=np.float32)
                if not np.isfinite(cube).all():
                    cube = np.nan_to_num(cube, nan=0.0)
                (out_dir / f"{rid}.fits").write_bytes(M120b.to_bytes(cube))
                written.append(rid)
            del mm
        print(f"[313]   {Path(p).name}: +{len(sub):,} (total {len(written):,})")
    miss = want - set(written)
    if miss:
        print(f"[313] WARNING: {len(miss):,}/{len(want):,} requested rows not found in any "
              f"part index (e.g. {sorted(miss)[:3]}) -> skipped")
    man = pd.DataFrame({"row_id": pd.array(written, dtype="string"), "label": 0,
                        "fits_dir": str(out_dir.resolve()),
                        "mimic_type": [type_of.get(r, "?") for r in written]})
    print(f"[313] wrote {len(written):,} FITS -> {out_dir} ({time.time() - t0:.1f}s)")
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=("train-pool", "eval"))
    ap.add_argument("--bank", default=str(C.DATA / "v3" / "mimic_bank.parquet"))
    ap.add_argument("--bank-eval", default=str(C.DATA / "v3" / "mimic_bank_eval.parquet"))
    ap.add_argument("--parts-glob", default=None,
                    help="default $SCRATCH/claudenet/cutouts/dr10/part*")
    ap.add_argument("--n", type=int, default=8000, help="train-pool size (train-pool mode)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--manifest-out", default=None)
    args = ap.parse_args()

    import os
    parts_glob = args.parts_glob or str(
        Path(os.environ.get("SCRATCH", ".")) / "claudenet" / "cutouts" / "dr10" / "part*")

    if args.mode == "train-pool":
        bank = pd.read_parquet(args.bank)
        rows = select_train_pool(bank, args.n)
        out_dir = Path(args.out_dir or (C.DATA / "v2" / "mimic_bank_fits"))
        man_out = Path(args.manifest_out or (C.DATA / "v2" / "mimic_bank_fits_manifest.parquet"))
    else:
        ev = pd.read_parquet(args.bank_eval)
        ev = ev[ev["source"] == "dr10_sweep"].copy()
        rows = ev[["row_id", "mimic_type", "p_final"]]
        print(f"[313] eval mode: {len(rows):,} dr10_sweep held-out rows")
        out_dir = Path(args.out_dir or (C.DATA / "v3" / "mimic_eval_fits"))
        man_out = Path(args.manifest_out or (C.DATA / "v3" / "mimic_eval_fits_manifest.parquet"))

    man = gather_fits(rows, parts_glob, out_dir)
    man_out.parent.mkdir(parents=True, exist_ok=True)
    man.to_parquet(man_out, index=False)
    print(f"[313] manifest -> {man_out} ({len(man):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
