#!/usr/bin/env python3
"""
11b_brick_inference_dr8.py — Phase 4b step 3 (brick-driven, two-model).

DR8 generalisation of huang-2020/11b_brick_inference_dr7.py with three changes:

  (i)   FOOTPRINT-AWARE brick routing. DR8 splits the sky into south (DECaLS,
        coadd at dr8/south/coadd/) and north (BASS+MzLS, dr8/north/coadd/). Each
        parent-sample row carries a `footprint` column (from 10_..._dr8.py); we
        group by (footprint, BRICKNAME) and download each band image from the
        matching region's coadd directory.

  (ii)  TWO-MODEL scoring in a SINGLE pass. Huang+2021 deploys the original L18
        ResNet AND the shielded ResNet together. Network (brick download) is the
        bottleneck, so we download each brick ONCE and score it with BOTH models,
        writing one score parquet per model. Each model is normalised with ITS
        OWN mean/std from its checkpoint.

  (iii) More shards. DR8 has ~350-400K bricks (vs DR7's ~113K); run more shards
        across the two L4s (2 processes per L4).

Launch (4 shards, 2 per L4 — set CUDA_DEVICE_ORDER so 8,9 are the L4s):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 ./11b_brick_inference_dr8.py --shard 0 --n-shards 4 --gpu 0 &
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 ./11b_brick_inference_dr8.py --shard 1 --n-shards 4 --gpu 0 &
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=9 ./11b_brick_inference_dr8.py --shard 2 --n-shards 4 --gpu 0 &
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=9 ./11b_brick_inference_dr8.py --shard 3 --n-shards 4 --gpu 0 &
  wait

Resumable per-shard. Outputs (per shard N):
  data/brick_manifest_shard{N}.csv
  data/inference_scores_l18_shard{N}.parquet        (row_id, ra, dec, score)
  data/inference_scores_shielded_shard{N}.parquet
  data/cutouts_fits_dr8/<row_id>.fits  (kept if EITHER model >= keep-thresh)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
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
CMUDeepLens = import_module("01_lanusse_resnet").CMUDeepLens
ShieldedDeepLens = import_module("01b_shielded_resnet").ShieldedDeepLens

DATA = HERE / "data"
FITS_OUT = DATA / "cutouts_fits_dr8"
BRICK_TMP = DATA / "brick_tmp"
FITS_OUT.mkdir(parents=True, exist_ok=True)
BRICK_TMP.mkdir(parents=True, exist_ok=True)

BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr8"
COADD = {"south": f"{BASE}/south/coadd", "north": f"{BASE}/north/coadd"}
BANDS = ("g", "r", "z")
SIZE_PIX = 101
HALF = SIZE_PIX // 2
TIMEOUT = 180
RETRIES = 4
RETRY_BACKOFF = 6.0
FLUSH_EVERY = 200


def brick_url(footprint: str, brick: str, band: str) -> str:
    pre = brick[:3]
    return f"{COADD[footprint]}/{pre}/{brick}/legacysurvey-{brick}-image-{band}.fits.fz"


def download_brick(footprint: str, brick: str, dest_dir: Path) -> tuple[dict[str, Path], str]:
    paths = {}
    for band in BANDS:
        url = brick_url(footprint, brick, band)
        out = dest_dir / f"{brick}-{band}.fits.fz"
        if out.exists() and out.stat().st_size > 1024:
            paths[band] = out
            continue
        last_err = ""
        for attempt in range(1, RETRIES + 1):
            try:
                r = requests.get(url, timeout=TIMEOUT, stream=True)
                if r.status_code == 404:
                    return {}, f"404 {footprint}/{brick} band={band}"
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


def load_brick(paths: dict[str, Path]) -> tuple[np.ndarray, WCS]:
    arrs = []
    wcs = None
    for band in BANDS:
        with fits.open(paths[band]) as hdul:
            hdu = hdul[1]  # CompImageHDU
            arrs.append(np.asarray(hdu.data, dtype=np.float32))
            if wcs is None:
                wcs = WCS(hdu.header)
    return np.stack(arrs, axis=0), wcs


def extract_cutout(cube: np.ndarray, wcs: WCS, ra: float, dec: float) -> np.ndarray | None:
    px = wcs.world_to_pixel_values(ra, dec)
    cx, cy = int(round(float(px[0]))), int(round(float(px[1])))
    H, W = cube.shape[1], cube.shape[2]
    y0, y1 = cy - HALF, cy + HALF + 1
    x0, x1 = cx - HALF, cx + HALF + 1
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return None
    return cube[:, y0:y1, x0:x1]


def cube_to_fits_bytes(cube: np.ndarray) -> bytes:
    hdu = fits.PrimaryHDU(data=cube.astype(np.float32))
    bio = io.BytesIO()
    hdu.writeto(bio)
    return bio.getvalue()


def load_model(kind: str, ckpt_path: Path, device: torch.device):
    """kind in {'l18','shielded'}. Returns (model, mean(3,1,1), std(3,1,1), val_auc)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mean = np.array(ckpt["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(ckpt["std"], dtype=np.float32).reshape(3, 1, 1)
    if kind == "shielded":
        model = ShieldedDeepLens(in_channels=3, final_out=int(ckpt.get("final_out", 32)))
    else:
        model = CMUDeepLens(in_channels=3)
    model.to(device)
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
        if not self.buffer or (not force and len(self.buffer) < 10_000):
            return
        table = pa.Table.from_pandas(pd.DataFrame(self.buffer), preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self.writer.write_table(table)
        self.buffer.clear()

    def close(self) -> None:
        self.maybe_flush(force=True)
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def shard_of(key: str, n_shards: int) -> int:
    h = hashlib.md5(key.encode()).digest()
    return int.from_bytes(h[:4], "big") % n_shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default=str(DATA / "parent_dr8.parquet"))
    ap.add_argument("--ckpt-l18", default=str(DATA / "checkpoint_best.pt"))
    ap.add_argument("--ckpt-shielded", default=str(DATA / "checkpoint_best_shielded_dr9.pt"))
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--brick-workers", type=int, default=3)
    ap.add_argument("--keep-thresh", type=float, default=0.1)  # paper operating point
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit-bricks", type=int, default=None)
    ap.add_argument("--keep-bricks", action="store_true")
    args = ap.parse_args()

    print(f"[init] shard={args.shard}/{args.n_shards}  gpu={args.gpu}  "
          f"brick-workers={args.brick_workers}  keep-thresh={args.keep_thresh}")

    manifest_path = DATA / f"brick_manifest_shard{args.shard}.csv"
    score_writers = {
        "l18": ScoreWriter(DATA / f"inference_scores_l18_shard{args.shard}.parquet"),
        "shielded": ScoreWriter(DATA / f"inference_scores_shielded_shard{args.shard}.parquet"),
    }

    print("[init] loading parent sample")
    parent = pd.read_parquet(args.parent)
    parent["row_id"] = (parent["BRICKID"].astype(str) + "_" + parent["OBJID"].astype(str))
    parent["_key"] = parent["footprint"].astype(str) + "/" + parent["BRICKNAME"].astype(str)
    parent["_shard"] = parent["_key"].apply(lambda k: shard_of(k, args.n_shards))
    sub = parent[parent["_shard"] == args.shard]
    print(f"[init] shard {args.shard}: {len(sub):,} rows across "
          f"{sub['_key'].nunique():,} (footprint,brick) units")

    done_keys: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            for line in csv.reader(f):
                if line and line[0]:
                    done_keys.add(line[0])
        print(f"[init] manifest has {len(done_keys):,} prior units")

    by_key = {name: grp for name, grp in sub.groupby("_key")}
    keys = sorted(k for k in by_key if k not in done_keys)
    if args.limit_bricks:
        keys = keys[: args.limit_bricks]
    print(f"[init] {len(keys):,} units to process in this run")
    if not keys:
        print("[done] nothing to do")
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    models = {
        "l18": load_model("l18", Path(args.ckpt_l18), device),
        "shielded": load_model("shielded", Path(args.ckpt_shielded), device),
    }
    for kind, (_, _, _, va) in models.items():
        print(f"[init] {kind} model on {device}  val_auc={va:.4f}")

    mfp = open(manifest_path, "a", newline="")
    mwriter = csv.writer(mfp)
    n_kept_total = 0
    n_processed = 0

    def graceful_shutdown(*_):
        print("\n[signal] flushing & exit")
        for sw in score_writers.values():
            sw.close()
        mfp.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    def process_one(key: str) -> dict:
        footprint, brick = key.split("/", 1)
        tmp = BRICK_TMP / f"shard{args.shard}_{footprint}_{brick}"
        tmp.mkdir(parents=True, exist_ok=True)
        paths, err = download_brick(footprint, brick, tmp)
        if err:
            for p in tmp.iterdir():
                p.unlink(missing_ok=True)
            tmp.rmdir()
            return {"key": key, "status": "fail", "err": err, "cubes": [], "rows": []}
        cube, wcs = load_brick(paths)
        rows, cubes = [], []
        edge = 0
        for _, r in by_key[key].iterrows():
            ct = extract_cutout(cube, wcs, float(r["RA"]), float(r["DEC"]))
            if ct is None:
                edge += 1
                continue
            cubes.append(ct)
            rows.append(r)
        if not args.keep_bricks:
            for p in paths.values():
                p.unlink(missing_ok=True)
            tmp.rmdir()
        return {"key": key, "status": "ok", "err": "", "cubes": cubes,
                "rows": rows, "edge_skipped": edge}

    def score(model_tuple, raw: np.ndarray) -> np.ndarray:
        model, mean, std, _ = model_tuple
        xs = np.clip((raw - mean) / std, -250.0, 250.0)
        xt = torch.from_numpy(xs)
        probs = np.empty(len(xt), dtype=np.float32)
        for s in range(0, len(xt), args.batch_size):
            e = min(s + args.batch_size, len(xt))
            with torch.no_grad():
                lo = model(xt[s:e].to(device)).cpu().numpy()
            probs[s:e] = 1.0 / (1.0 + np.exp(-lo))
        return probs

    last_flush = 0
    pbar = tqdm(total=len(keys), desc=f"shard{args.shard}")
    try:
        with ThreadPoolExecutor(max_workers=args.brick_workers) as ex:
            it = iter(keys)
            in_flight: dict = {}
            for _ in range(min(args.brick_workers, len(keys))):
                try:
                    k = next(it)
                    in_flight[ex.submit(process_one, k)] = k
                except StopIteration:
                    break
            done_count = 0
            while in_flight:
                fut = next(as_completed(list(in_flight)))
                in_flight.pop(fut)
                result = fut.result()
                done_count += 1
                pbar.update(1)
                try:
                    k = next(it)
                    in_flight[ex.submit(process_one, k)] = k
                except StopIteration:
                    pass

                if result["status"] != "ok":
                    mwriter.writerow([result["key"], "fail", "0", "0", result["err"][:80]])
                    continue
                cubes = result["cubes"]
                if not cubes:
                    mwriter.writerow([result["key"], "ok", "0", "0", ""])
                    continue

                raw = np.stack(cubes)  # (N,3,101,101)
                probs = {kind: score(models[kind], raw) for kind in models}
                max_p = np.maximum(probs["l18"], probs["shielded"])

                n_kept = 0
                for i, r in enumerate(result["rows"]):
                    rid = str(r["row_id"])
                    ra, dec = float(r["RA"]), float(r["DEC"])
                    for kind in models:
                        score_writers[kind].add(rid, ra, dec, float(probs[kind][i]))
                    if max_p[i] >= args.keep_thresh:
                        (FITS_OUT / f"{rid}.fits").write_bytes(cube_to_fits_bytes(cubes[i]))
                        n_kept += 1
                n_kept_total += n_kept
                n_processed += len(cubes)
                mwriter.writerow([result["key"], "ok", str(len(cubes)), str(n_kept), ""])
                pbar.set_postfix(gal=n_processed, kept=n_kept_total)

                if done_count - last_flush >= FLUSH_EVERY:
                    for sw in score_writers.values():
                        sw.maybe_flush(force=True)
                    mfp.flush()
                    last_flush = done_count
        pbar.close()
    finally:
        for sw in score_writers.values():
            sw.close()
        mfp.close()

    print(f"\n[done] shard={args.shard}  units={len(keys):,}  galaxies={n_processed:,}  "
          f"kept={n_kept_total:,} ({n_kept_total/max(n_processed,1):.2%})")


if __name__ == "__main__":
    main()
