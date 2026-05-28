#!/usr/bin/env python3
"""
11b_brick_inference_dr7.py — Phase 3b M2 (brick-driven).

Instead of fetching one cutout at a time from the rate-limited legacysurvey.org
viewer (which caps at ~0.86 rows/sec across 16 workers), we:

  1) group parent_dr7.parquet rows by BRICKNAME (~113K bricks, ~55 galaxies each)
  2) download the brick's grz image FITS once (3 files, ~15 MB each)
  3) load WCS, project each galaxy's RA/Dec to brick pixels
  4) slice 101x101 cutouts locally
  5) batch GPU forward-pass
  6) persist FITS iff sigmoid >= keep-thresh
  7) discard brick files

Expected ETA: ~1-3 days vs ~84 days for endpoint-based.

2-shard partition by BRICKNAME hash; run two copies, one per L4:

  CUDA_VISIBLE_DEVICES=0 ./11b_brick_inference_dr7.py --shard 0 --gpu 0 &
  CUDA_VISIBLE_DEVICES=1 ./11b_brick_inference_dr7.py --shard 1 --gpu 1 &

Resumable per-shard: skips bricks already in this shard's manifest.

Outputs (per shard N):
  data/brick_manifest_shard{N}.csv         (brick, status, n_galaxies, n_kept)
  data/inference_scores_shard{N}.parquet   (row_id, ra, dec, score)
  data/cutouts_fits_dr7/<row_id>.fits      (only if score >= keep-thresh)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402
_mod = import_module("01_lanusse_resnet")
CMUDeepLens = _mod.CMUDeepLens

DATA = HERE / "data"
FITS_OUT = DATA / "cutouts_fits_dr7"
BRICK_TMP = DATA / "brick_tmp"
FITS_OUT.mkdir(parents=True, exist_ok=True)
BRICK_TMP.mkdir(parents=True, exist_ok=True)

DR7_COADD = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr7/coadd"
BANDS = ("g", "r", "z")
SIZE_PIX = 101
HALF = SIZE_PIX // 2  # 50
TIMEOUT = 180  # brick downloads can be large
RETRIES = 4
RETRY_BACKOFF = 6.0
FLUSH_EVERY = 200  # bricks-per-shard between parquet flushes


def brick_url(brick: str, band: str) -> str:
    pre = brick[:3]
    return f"{DR7_COADD}/{pre}/{brick}/legacysurvey-{brick}-image-{band}.fits.fz"


def download_brick(brick: str, dest_dir: Path) -> tuple[dict[str, Path], str]:
    """Download all 3 bands for a brick into dest_dir. Returns (paths, err)."""
    paths = {}
    for band in BANDS:
        url = brick_url(brick, band)
        out = dest_dir / f"{brick}-{band}.fits.fz"
        if out.exists() and out.stat().st_size > 1024:
            paths[band] = out
            continue
        last_err = ""
        for attempt in range(1, RETRIES + 1):
            try:
                r = requests.get(url, timeout=TIMEOUT, stream=True)
                if r.status_code == 404:
                    return {}, f"404 brick={brick} band={band}"
                if r.status_code == 429:
                    time.sleep(20)
                    continue
                r.raise_for_status()
                tmp = out.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                tmp.rename(out)
                paths[band] = out
                last_err = ""
                break
            except Exception as e:
                last_err = str(e)
                if attempt < RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)
        if last_err:
            return {}, f"download band={band}: {last_err}"
    return paths, ""


def load_brick(paths: dict[str, Path]) -> tuple[np.ndarray, WCS] | None:
    """Returns (cube[3,H,W], wcs) or None on failure."""
    arrs = []
    wcs = None
    for band in BANDS:
        with fits.open(paths[band]) as hdul:
            hdu = hdul[1]  # CompImageHDU
            arrs.append(np.asarray(hdu.data, dtype=np.float32))
            if wcs is None:
                wcs = WCS(hdu.header)
    cube = np.stack(arrs, axis=0)
    return cube, wcs


def extract_cutout(cube: np.ndarray, wcs: WCS, ra: float, dec: float) -> np.ndarray | None:
    """Slice a 101x101 cutout around (RA, Dec). Returns None if out of bounds."""
    px = wcs.world_to_pixel_values(ra, dec)
    cx, cy = int(round(float(px[0]))), int(round(float(px[1])))
    H, W = cube.shape[1], cube.shape[2]
    y0, y1 = cy - HALF, cy + HALF + 1
    x0, x1 = cx - HALF, cx + HALF + 1
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return None
    return cube[:, y0:y1, x0:x1]


def cube_to_fits_bytes(cube: np.ndarray) -> bytes:
    """Pack a (3,H,W) float32 cube as a single-HDU FITS file (mirrors viewer output)."""
    hdu = fits.PrimaryHDU(data=cube.astype(np.float32))
    bio = io.BytesIO()
    hdu.writeto(bio)
    return bio.getvalue()


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mean = np.array(ckpt["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(ckpt["std"], dtype=np.float32).reshape(3, 1, 1)
    model = CMUDeepLens(in_channels=3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, mean, std, float(ckpt.get("val_auc", 0.0))


class ScoreWriter:
    def __init__(self, path: Path):
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.buffer: list[dict] = []

    def add(self, row_id: str, ra: float, dec: float, score: float) -> None:
        self.buffer.append({"row_id": row_id, "ra": ra, "dec": dec, "score": score})

    def maybe_flush(self, force: bool = False) -> None:
        if not self.buffer:
            return
        if not force and len(self.buffer) < 10_000:
            return
        df = pd.DataFrame(self.buffer)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self.writer.write_table(table)
        self.buffer.clear()

    def close(self) -> None:
        self.maybe_flush(force=True)
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def shard_of(brick: str, n_shards: int) -> int:
    """Stable hash-based shard assignment by BRICKNAME."""
    h = hashlib.md5(brick.encode()).digest()
    return int.from_bytes(h[:4], "big") % n_shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default=str(DATA / "parent_dr7.parquet"))
    ap.add_argument("--ckpt", default=str(DATA / "checkpoint_best.pt"))
    ap.add_argument("--shard", type=int, required=True, choices=[0, 1])
    ap.add_argument("--n-shards", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--brick-workers", type=int, default=2,
                    help="bricks being downloaded/processed in parallel")
    ap.add_argument("--keep-thresh", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit-bricks", type=int, default=None)
    ap.add_argument("--keep-bricks", action="store_true",
                    help="leave brick FITS files on disk after processing (default: delete)")
    args = ap.parse_args()

    print(f"[init] shard={args.shard}/{args.n_shards}  gpu={args.gpu}  "
          f"brick-workers={args.brick_workers}  thresh={args.keep_thresh}")

    manifest_path = DATA / f"brick_manifest_shard{args.shard}.csv"
    scores_path = DATA / f"inference_scores_shard{args.shard}.parquet"

    print(f"[init] loading parent sample")
    parent = pd.read_parquet(args.parent)
    parent["row_id"] = (parent["BRICKID"].astype(str) + "_"
                        + parent["OBJID"].astype(str))
    # Build per-brick groups
    parent["_shard"] = parent["BRICKNAME"].apply(lambda b: shard_of(b, args.n_shards))
    sub = parent[parent["_shard"] == args.shard]
    print(f"[init] shard {args.shard}: {len(sub):,} rows across {sub['BRICKNAME'].nunique():,} bricks")

    # Resumability
    done_bricks: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            for line in csv.reader(f):
                if line and line[0]:
                    done_bricks.add(line[0])
        print(f"[init] manifest has {len(done_bricks):,} prior bricks")

    bricks = sorted(sub["BRICKNAME"].unique())
    bricks = [b for b in bricks if b not in done_bricks]
    if args.limit_bricks:
        bricks = bricks[: args.limit_bricks]
    print(f"[init] {len(bricks):,} bricks to process in this run")
    if not bricks:
        print("[done] nothing to do")
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, mean, std, val_auc = load_model(Path(args.ckpt), device)
    print(f"[init] model on {device}  val_auc={val_auc:.4f}")

    score_writer = ScoreWriter(scores_path)
    mfp = open(manifest_path, "a", newline="")
    mwriter = csv.writer(mfp)

    n_kept_total = 0
    n_processed_galaxies = 0

    def graceful_shutdown(*_):
        print("\n[signal] flushing & exit")
        score_writer.close()
        mfp.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # Map BRICKNAME -> sub-rows (cache once)
    by_brick = {name: grp for name, grp in sub.groupby("BRICKNAME")}

    def process_one_brick(brick: str) -> dict:
        """Download brick, extract cutouts, score, return summary."""
        per_shard_tmp = BRICK_TMP / f"shard{args.shard}_{brick}"
        per_shard_tmp.mkdir(parents=True, exist_ok=True)
        paths, err = download_brick(brick, per_shard_tmp)
        if err:
            for p in per_shard_tmp.iterdir():
                p.unlink(missing_ok=True)
            per_shard_tmp.rmdir()
            return {"brick": brick, "status": "fail", "err": err,
                    "cubes": [], "rows": [], "scored": []}

        cube_wcs = load_brick(paths)
        cube, wcs = cube_wcs
        rows = by_brick[brick]
        cubes = []
        meta = []
        edge_skipped = 0
        for _, r in rows.iterrows():
            ct = extract_cutout(cube, wcs, float(r["RA"]), float(r["DEC"]))
            if ct is None:
                edge_skipped += 1
                continue
            cubes.append(ct)
            meta.append(r)

        # Cleanup brick files immediately (saves disk)
        if not args.keep_bricks:
            for p in paths.values():
                p.unlink(missing_ok=True)
            per_shard_tmp.rmdir()

        return {"brick": brick, "status": "ok", "err": "",
                "cubes": cubes, "rows": meta, "edge_skipped": edge_skipped}

    last_flush_brick = 0
    pbar = tqdm(total=len(bricks), desc=f"shard{args.shard}")
    try:
        # Sequential per-shard brick processing (brick-workers controls how many
        # bricks we PREfetch in parallel; GPU work happens after each finishes).
        with ThreadPoolExecutor(max_workers=args.brick_workers) as ex:
            it = iter(bricks)
            in_flight: dict = {}
            for _ in range(min(args.brick_workers, len(bricks))):
                try:
                    b = next(it)
                    in_flight[ex.submit(process_one_brick, b)] = b
                except StopIteration:
                    break

            done_count = 0
            while in_flight:
                fut = next(as_completed(list(in_flight)))
                in_flight.pop(fut)
                result = fut.result()
                done_count += 1
                pbar.update(1)

                # Refill
                try:
                    b = next(it)
                    in_flight[ex.submit(process_one_brick, b)] = b
                except StopIteration:
                    pass

                if result["status"] != "ok":
                    mwriter.writerow([result["brick"], "fail", "0", "0",
                                      result["err"][:80]])
                    continue

                cubes = result["cubes"]
                if not cubes:
                    mwriter.writerow([result["brick"], "ok", "0", "0", ""])
                    continue

                # GPU inference in batches
                xs = np.stack(cubes)  # (N, 3, 101, 101)
                xs = (xs - mean) / std
                xs = np.clip(xs, -250.0, 250.0)
                xt = torch.from_numpy(xs)
                probs = np.empty(len(xt), dtype=np.float32)
                for s in range(0, len(xt), args.batch_size):
                    e = min(s + args.batch_size, len(xt))
                    with torch.no_grad():
                        lo = model(xt[s:e].to(device)).cpu().numpy()
                    probs[s:e] = 1.0 / (1.0 + np.exp(-lo))

                # Persist
                n_kept = 0
                for r, p, cube in zip(result["rows"], probs, cubes):
                    rid = str(r["row_id"])
                    score_writer.add(rid, float(r["RA"]), float(r["DEC"]), float(p))
                    if p >= args.keep_thresh:
                        (FITS_OUT / f"{rid}.fits").write_bytes(cube_to_fits_bytes(cube))
                        n_kept += 1
                n_kept_total += n_kept
                n_processed_galaxies += len(cubes)
                mwriter.writerow([result["brick"], "ok",
                                  str(len(cubes)), str(n_kept), ""])
                pbar.set_postfix(gal=n_processed_galaxies,
                                 kept=n_kept_total)

                if done_count - last_flush_brick >= FLUSH_EVERY:
                    score_writer.maybe_flush(force=True)
                    mfp.flush()
                    last_flush_brick = done_count
        pbar.close()
    finally:
        score_writer.close()
        mfp.close()

    print(f"\n[done] shard={args.shard}  bricks={len(bricks):,}  "
          f"galaxies={n_processed_galaxies:,}  kept={n_kept_total:,} "
          f"({n_kept_total/max(n_processed_galaxies,1):.2%})")


if __name__ == "__main__":
    main()
