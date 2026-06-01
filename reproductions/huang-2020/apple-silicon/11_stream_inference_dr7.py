#!/usr/bin/env python3
"""
11_stream_inference_dr7.py — Phase 3b M2.

For each row in data/parent_dr7.parquet matching this shard:
  1) Download 101x101 grz FITS cutout from DECaLS DR7 layer.
  2) Normalize with Phase 3a checkpoint mean/std + clamp.
  3) Forward-pass through CMUDeepLens on the assigned GPU.
  4) Persist FITS+JPEG iff score >= --keep-thresh (default 0.5).
  5) Append (row_id, RA, Dec, score) to per-shard parquet.

Designed as a 2-shard launcher; run two copies in parallel:

  CUDA_VISIBLE_DEVICES=0 ./11_stream_inference_dr7.py --shard 0 --gpu 0 &
  CUDA_VISIBLE_DEVICES=1 ./11_stream_inference_dr7.py --shard 1 --gpu 1 &

Resumable: skips rows whose row_id is already in this shard's manifest.

Outputs (per shard N=0 or 1):
  data/inference_manifest_shard{N}.csv  (row_id, status, score)
  data/inference_scores_shard{N}.parquet (incremental, append-on-flush)
  data/cutouts_fits_dr7/<row_id>.fits   (only if score >= keep-thresh)
  data/cutouts_jpeg_dr7/<row_id>.jpg
"""
from __future__ import annotations

import argparse
import csv
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
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402
_mod = import_module("01_lanusse_resnet")
CMUDeepLens = _mod.CMUDeepLens
from device import pick_device  # noqa: E402  (MPS/CUDA/CPU device selection)

DATA = HERE / "data"
FITS_DIR = DATA / "cutouts_fits_dr7"
JPEG_DIR = DATA / "cutouts_jpeg_dr7"
FITS_DIR.mkdir(parents=True, exist_ok=True)
JPEG_DIR.mkdir(parents=True, exist_ok=True)

LAYER = "decals-dr7"
PIXSCALE = 0.262
SIZE_PIX = 101
TIMEOUT = 60
RETRIES = 5
RETRY_BACKOFF = 4.0
RATELIMIT_BACKOFF = 30.0
FLUSH_EVERY = 10_000  # rows-per-shard between parquet flushes


def make_url(ra: float, dec: float, fmt: str) -> str:
    base = "https://www.legacysurvey.org/viewer"
    endpoint = "jpeg-cutout" if fmt == "jpeg" else "fits-cutout"
    bands = "" if fmt == "jpeg" else "&bands=grz"
    return (f"{base}/{endpoint}"
            f"?ra={ra:.6f}&dec={dec:.6f}&size={SIZE_PIX}&layer={LAYER}"
            f"&pixscale={PIXSCALE}{bands}")


def http_get(url: str) -> bytes:
    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT, stream=True)
            if r.status_code == 429:
                last_err = "HTTP 429"
                time.sleep(RATELIMIT_BACKOFF)
                continue
            r.raise_for_status()
            data = r.content
            if len(data) < 256:
                last_err = f"too-small ({len(data)} B)"
                raise RuntimeError(last_err)
            return data
        except Exception as e:
            last_err = str(e)
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    raise RuntimeError(last_err)


def fits_bytes_to_cube(b: bytes) -> np.ndarray | None:
    try:
        with fits.open(io.BytesIO(b), memmap=False) as hdul:
            data = hdul[0].data
        if data is None or data.ndim != 3 or data.shape[0] != 3 \
                or data.shape[1] != SIZE_PIX or data.shape[2] != SIZE_PIX:
            return None
        return data.astype(np.float32)
    except Exception:
        return None


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mean = np.array(ckpt["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(ckpt["std"], dtype=np.float32).reshape(3, 1, 1)
    model = CMUDeepLens(in_channels=3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, mean, std, float(ckpt.get("val_auc", 0.0))


def fetch_one(rid: str, ra: float, dec: float) -> dict:
    """Worker: download FITS only. JPEGs come later in a post-step for kept rows."""
    result = {"row_id": rid, "ra": ra, "dec": dec,
              "fits_bytes": None, "status": "ok", "err": ""}
    try:
        result["fits_bytes"] = http_get(make_url(ra, dec, "fits"))
    except Exception as e:
        result["status"] = "fail"
        result["err"] = str(e)
    return result


class ScoreWriter:
    """Append batches of (row_id, RA, DEC, score) to a per-shard parquet."""
    def __init__(self, path: Path):
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.buffer: list[dict] = []

    def add(self, row_id: str, ra: float, dec: float, score: float) -> None:
        self.buffer.append({"row_id": row_id, "ra": ra, "dec": dec, "score": score})

    def maybe_flush(self, force: bool = False) -> None:
        if not self.buffer:
            return
        if not force and len(self.buffer) < FLUSH_EVERY:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default=str(DATA / "parent_dr7.parquet"))
    ap.add_argument("--ckpt", default=str(DATA / "checkpoint_best.pt"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0,
                    help="LOGICAL gpu id (after CUDA_VISIBLE_DEVICES). Usually 0.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="GPU forward-pass batch size")
    ap.add_argument("--keep-thresh", type=float, default=0.5,
                    help="persist cutouts with sigmoid >= this; otherwise discard")
    ap.add_argument("--limit", type=int, default=None, help="for testing")
    args = ap.parse_args()

    print(f"[init] shard={args.shard}/{args.n_shards}  gpu={args.gpu}  "
          f"workers={args.workers}  batch={args.batch_size}  thresh={args.keep_thresh}")

    # Per-shard output paths
    manifest_path = DATA / f"inference_manifest_shard{args.shard}.csv"
    scores_path = DATA / f"inference_scores_shard{args.shard}.parquet"

    # Load parent sample, filter to this shard's row_ids (deterministic)
    print(f"[init] loading parent sample from {args.parent}")
    parent = pd.read_parquet(args.parent)
    parent["row_id"] = parent.apply(
        lambda r: f"{int(r['BRICKID'])}_{int(r['OBJID'])}", axis=1)
    # Cheap shard assignment: hash(brickid+objid) % n_shards.
    # We use BRICKID directly because it's already a uniformly-spread int.
    parent["_shard"] = parent["BRICKID"].astype(np.int64) % args.n_shards
    sub = parent[parent["_shard"] == args.shard].reset_index(drop=True)
    print(f"[init] shard {args.shard}: {len(sub):,} rows of {len(parent):,} total")

    # Resumability: load existing manifest, build set of completed row_ids
    done: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            for line in csv.reader(f):
                if line and line[0]:
                    done.add(line[0])
        print(f"[init] manifest has {len(done):,} prior rows; will skip these")
    todo = sub[~sub["row_id"].isin(done)].reset_index(drop=True)
    if args.limit:
        todo = todo.iloc[: args.limit].reset_index(drop=True)
    print(f"[init] {len(todo):,} rows to process in this run")

    if len(todo) == 0:
        print("[done] nothing to do")
        return

    device = pick_device(args.gpu)
    model, mean, std, val_auc = load_model(Path(args.ckpt), device)
    print(f"[init] model on {device}; checkpoint val_auc={val_auc:.4f}")

    score_writer = ScoreWriter(scores_path)
    manifest = open(manifest_path, "a", newline="")
    mwriter = csv.writer(manifest)

    n_kept = 0
    n_total = 0
    n_fail = 0
    last_flush = 0

    def graceful_shutdown(*_):
        print("\n[signal] flushing buffers and exiting")
        score_writer.close()
        manifest.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            pbar = tqdm(total=len(todo), desc=f"shard{args.shard}")
            # Submit in chunks of batch-size so the GPU sees a steady stream
            for start in range(0, len(todo), args.batch_size):
                end = min(start + args.batch_size, len(todo))
                batch = todo.iloc[start:end]
                futs = []
                for _, row in batch.iterrows():
                    futs.append(ex.submit(fetch_one,
                                          row["row_id"], float(row["RA"]),
                                          float(row["DEC"])))
                # Collect downloaded raw bytes
                items: list[dict] = []
                for fut in as_completed(futs):
                    items.append(fut.result())

                # Build a GPU batch from successful FITS cubes
                cubes = []
                meta = []
                fails = []
                for it in items:
                    if it["status"] != "ok" or it["fits_bytes"] is None:
                        fails.append(it)
                        continue
                    cube = fits_bytes_to_cube(it["fits_bytes"])
                    if cube is None:
                        it["status"] = "fail"
                        it["err"] = "bad-fits"
                        fails.append(it)
                        continue
                    x = (cube - mean) / std
                    x = np.clip(x, -250.0, 250.0)
                    cubes.append(x)
                    meta.append(it)

                if cubes:
                    xb = torch.from_numpy(np.stack(cubes)).to(device)
                    with torch.no_grad():
                        logits = model(xb).cpu().numpy()
                    probs = 1.0 / (1.0 + np.exp(-logits))
                    for it, p in zip(meta, probs):
                        rid = it["row_id"]
                        score_writer.add(rid, it["ra"], it["dec"], float(p))
                        if p >= args.keep_thresh:
                            n_kept += 1
                            (FITS_DIR / f"{rid}.fits").write_bytes(it["fits_bytes"])
                        mwriter.writerow([rid, "ok", f"{p:.6f}"])
                for it in fails:
                    n_fail += 1
                    mwriter.writerow([it["row_id"], "fail", it.get("err", "")[:80]])

                n_total += len(items)
                pbar.update(len(items))
                pbar.set_postfix(kept=n_kept, fail=n_fail)

                # Periodic flushes
                if n_total - last_flush >= FLUSH_EVERY:
                    score_writer.maybe_flush(force=True)
                    manifest.flush()
                    last_flush = n_total

            pbar.close()
    finally:
        score_writer.close()
        manifest.close()

    print(f"\n[done] shard={args.shard}  processed={n_total:,}  "
          f"kept={n_kept:,} ({n_kept/max(n_total,1):.2%})  fail={n_fail:,}")


if __name__ == "__main__":
    main()
