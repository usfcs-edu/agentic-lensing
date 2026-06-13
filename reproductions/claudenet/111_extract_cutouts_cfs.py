#!/usr/bin/env python3
"""111_extract_cutouts_cfs.py — Phase 110: cutout extraction from local CFS
coadds (runs ON Perlmutter, CPU-only, multiprocess).

Extracts size x size multi-band cutouts for every manifest row (row_id, RA,
DEC, footprint, brick) directly from the Legacy Surveys coadd bricks on CFS
(/global/cfs/cdirs/cosmo/data/legacysurvey/<release>/<footprint>/coadd/...,
no HTTP) into float32 memmap shards plus an index:

    <out-root>/cutouts_<k>.npy     (n_chunk, n_bands, size, size) float32
    <out-root>/index_<k>.parquet   per-chunk index (doubles as resume marker)
    <out-root>/index.parquet       row_id, shard, idx_in_shard, ok, nan_frac
                                   [+ per-band <b>_ok flags for non-grz modes]

Per-brick logic mirrors inchausti-2025/20_build_negatives_brick_dr9.py (HDU 1
image as float32, WCS from the first available band, int(round()) centering,
reject cutouts crossing the brick edge -> ok=False). Brick groups are
partitioned into contiguous ~--shard-size chunks (a brick never splits across
shards); each worker fills one shard. A brick missing a REQUIRED (grz) band
-> all its rows get ok=False (footprint edges; never a crash). A missing
OPTIONAL band (e.g. i in --bands griz --release dr10, Phase 130) -> that
plane is zero-filled and <b>_ok=False, rows are kept. NaN pixels are kept and
reported per row as nan_frac — they propagate to NaN CNN scores in 112
(torch.clamp keeps NaN; matches the v1 FITS path), and 113 drops non-finite
score rows, so nan_frac>0 rows self-exclude from the eval. Resume-safe: a chunk whose
cutouts_<k>.npy AND index_<k>.parquet both exist is skipped. Exits nonzero
if the overall ok-fraction < --min-ok-frac (0.95).

    python 111_extract_cutouts_cfs.py --manifest data/v2/negeval_manifest.parquet \
        --out-root $SCRATCH/claudenet/cutouts/negeval --size 101 --bands grz \
        --release dr9 --workers 32 --shard-size 50000 [--row-ids subset.parquet]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

CFS_ROOT = "/global/cfs/cdirs/cosmo/data/legacysurvey"
REQUIRED = ("g", "r", "z")  # manifests filter NOBS_G/R/Z >= 1, so grz must exist;
                            # anything else (i) is optional -> zero-fill + flag


def band_path(root: str, release: str, footprint: str, brick: str, band: str) -> Path:
    return (Path(root) / release / footprint / "coadd" / brick[:3] / brick
            / f"legacysurvey-{brick}-image-{band}.fits.fz")


def load_brick(paths: dict, bands: tuple, required: tuple):
    """v1 load_brick repointed at CFS: HDU 1 float32 per band, WCS from the
    first AVAILABLE band. Returns (cube, wcs, band_ok); cube is None when a
    required band file is missing (footprint edge -> brick skipped)."""
    band_ok = {b: paths[b].exists() for b in bands}
    if not all(band_ok[b] for b in required):
        return None, None, band_ok
    arrs, wcs = [], None
    for b in bands:
        if not band_ok[b]:
            arrs.append(None)  # optional gap -> zero plane (sized below)
            continue
        with fits.open(paths[b]) as h:
            arrs.append(np.asarray(h[1].data, dtype=np.float32))
            if wcs is None:
                wcs = WCS(h[1].header)
    shape = next(a.shape for a in arrs if a is not None)
    arrs = [a if a is not None else np.zeros(shape, np.float32) for a in arrs]
    return np.stack(arrs, 0), wcs, band_ok


def extract(cube, wcs, ra: float, dec: float, size: int):
    """v1 extract: int(round()) centering; None if crossing the brick edge.
    y1 = y0 + size (not cy+half+1) so even sizes (Phase-130 --size 160) get
    exactly `size` pixels; identical to v1 for odd sizes."""
    px = wcs.world_to_pixel_values(ra, dec)
    cx, cy = int(round(float(px[0]))), int(round(float(px[1])))
    half = size // 2
    H, W = cube.shape[1], cube.shape[2]
    y0, x0 = cy - half, cx - half
    y1, x1 = y0 + size, x0 + size
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return None
    return cube[:, y0:y1, x0:x1]


def make_chunks(df: pd.DataFrame, shard_size: int) -> list:
    """Partition (footprint, brick) groups into contiguous ~shard_size-row
    chunks; a brick group never splits across shards."""
    chunks, cur, n = [], [], 0
    for _, g in df.groupby(["footprint", "brick"], sort=True):
        cur.append(g)
        n += len(g)
        if n >= shard_size:
            chunks.append(pd.concat(cur, ignore_index=True))
            cur, n = [], 0
    if cur:
        chunks.append(pd.concat(cur, ignore_index=True))
    return chunks


def process_chunk(task):
    """Worker: fill shard cutouts_<k>.npy for one chunk, write index_<k>.parquet
    LAST (so its presence marks completion), return (k, index_df, resumed)."""
    k, chunk, cfg = task
    out_root, bands, size = Path(cfg["out_root"]), cfg["bands"], cfg["size"]
    shard_f = out_root / f"cutouts_{k}.npy"
    index_f = out_root / f"index_{k}.parquet"
    if shard_f.exists() and index_f.exists():
        # resume guard: the stored index must match THIS run's chunking exactly,
        # else (changed --shard-size / --row-ids / manifest) reprocess from scratch
        try:
            idx = pd.read_parquet(index_f)
        except Exception:
            idx = None
        if idx is not None and list(idx.row_id) == list(chunk.row_id):
            return k, idx, True
        print(f"[extract:{k}] stale shard (chunking changed) -> reprocessing", flush=True)
    required = tuple(b for b in bands if b in REQUIRED)
    per_band_flags = set(bands) != set(REQUIRED)  # griz etc.: record <b>_ok per row
    mm = np.lib.format.open_memmap(shard_f, mode="w+", dtype=np.float32,
                                   shape=(len(chunk), len(bands), size, size))
    recs = []
    for (foot, brick), g in chunk.groupby(["footprint", "brick"], sort=False):
        paths = {b: band_path(cfg["cfs_root"], cfg["release"], foot, brick, b)
                 for b in bands}
        try:
            cube, wcs, band_ok = load_brick(paths, bands, required)
        except Exception as ex:  # unreadable file -> skip brick, never crash
            print(f"[extract:{k}] brick {foot}/{brick} unreadable: {ex}", flush=True)
            cube, wcs, band_ok = None, None, {b: False for b in bands}
        for pos, r in zip(g.index, g.itertuples()):
            rec = {"row_id": r.row_id, "shard": k, "idx_in_shard": int(pos),
                   "ok": False, "nan_frac": np.nan}
            if per_band_flags:
                rec.update({f"{b}_ok": bool(band_ok[b]) for b in bands})
            if cube is not None:
                ct = extract(cube, wcs, float(r.RA), float(r.DEC), size)
                if ct is not None:
                    mm[pos] = ct
                    rec["ok"] = True
                    present = [i for i, b in enumerate(bands) if band_ok.get(b)]
                    rec["nan_frac"] = float(np.isnan(ct[present]).mean()) if present else np.nan
            recs.append(rec)
        del cube
    mm.flush()
    del mm
    idx = pd.DataFrame(recs).sort_values("idx_in_shard", ignore_index=True)
    tmp = index_f.with_suffix(".tmp")          # atomic completion marker
    idx.to_parquet(tmp, index=False)
    tmp.rename(index_f)
    return k, idx, False


def read_row_ids(path: str) -> list:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    return df["row_id"].astype(str).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--size", type=int, default=101)
    ap.add_argument("--bands", default="grz", help="e.g. grz (dr9) or griz (dr10)")
    ap.add_argument("--release", default="dr9")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=50000)
    ap.add_argument("--row-ids", default=None,
                    help="optional parquet/csv with row_id column: extract only these")
    ap.add_argument("--cfs-root", default=CFS_ROOT,
                    help="coadd tree root (override for smoke tests off-CFS)")
    ap.add_argument("--min-ok-frac", type=float, default=0.95)
    args = ap.parse_args()
    t0 = time.time()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    bands = tuple(args.bands)

    df = pd.read_parquet(args.manifest)
    if args.row_ids:
        want = set(read_row_ids(args.row_ids))
        df = df[df.row_id.isin(want)]
        print(f"[extract] --row-ids: {len(df):,} of {len(want):,} requested rows in manifest")
    if df.empty:
        print("[extract] FATAL: no rows to extract")
        return 1
    print(f"[extract] {len(df):,} rows | bands={args.bands} size={args.size} "
          f"release={args.release} | {df.brick.nunique():,} bricks")

    chunks = make_chunks(df, args.shard_size)
    cfg = {"out_root": str(out_root), "bands": bands, "size": args.size,
           "release": args.release, "cfs_root": args.cfs_root}
    print(f"[extract] {len(chunks)} shards (~{args.shard_size:,} rows each), "
          f"{args.workers} workers")
    idxs, n_resumed = {}, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_chunk, (k, c, cfg)) for k, c in enumerate(chunks)]
        for fut in as_completed(futs):
            k, idx, resumed = fut.result()
            idxs[k] = idx
            n_resumed += resumed
            print(f"[extract] shard {k}: {int(idx.ok.sum()):,}/{len(idx):,} ok"
                  + (" (resumed)" if resumed else ""), flush=True)

    index = pd.concat([idxs[k] for k in sorted(idxs)], ignore_index=True)
    assert index.row_id.is_unique, "duplicate row_ids in final index (stale shards?)"
    assert len(index) == len(df) and set(index.row_id) == set(df.row_id), \
        "final index does not cover the manifest exactly"
    tmp = out_root / "index.parquet.tmp"
    index.to_parquet(tmp, index=False)
    tmp.rename(out_root / "index.parquet")
    n_ok, n = int(index.ok.sum()), len(index)
    nbytes = sum((out_root / f"cutouts_{k}.npy").stat().st_size for k in sorted(idxs))
    print(f"[done] {n_ok:,}/{n:,} rows ok ({n_ok / max(n, 1):.4f}); "
          f"{nbytes / 1e9:.2f} GB written; {n_resumed} shards resumed; "
          f"{(time.time() - t0) / 60:.1f} min -> {out_root / 'index.parquet'}")
    return 0 if n_ok / max(n, 1) >= args.min_ok_frac else 1


if __name__ == "__main__":
    raise SystemExit(main())
