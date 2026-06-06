#!/usr/bin/env python3
"""01_sync_data_from_phoenix.py — pull the FITS cutouts, checkpoints, and tabular
baseline artifacts from the phoenix data node into claudenet/data/.

The cutouts do not exist on this box; they live on
phoenix:/raid/benson/git/agentic-lensing/reproductions/.../data/. This is a
read-only rsync pull. Verifies file counts and aborts on a gross mismatch.

    /home2/benson/.venvs/claudenet/bin/python 01_sync_data_from_phoenix.py
    # options: --host phoenix  --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _clib as C

# (remote subpath under PHOENIX_RAID, expected file count or None) — cutout dirs.
DIRS = [
    ("inchausti-2025/data/cutouts_fits_curated_dr9", 2398),
    ("inchausti-2025/data/cutouts_fits_neg_dr9", 65010),
    ("inchausti-2025/data/cutouts_fits_litpos_dr9", 5573),
    ("inchausti-2025/data/cutouts_fits_candidates_storfer", 1895),
    ("inchausti-2025/data/cutouts_fits_candidates_inchausti", 811),
]
# individual files (checkpoints + tabular)
FILES = [
    "inchausti-2025/data/checkpoint_best_shielded194k_staged.pt",
    "inchausti-2025/data/checkpoint_best_efficientnet_staged.pt",
    "inchausti-2025/data/checkpoint_best_meta_staged.pt",
    "inchausti-2025/data/checkpoint_best_shielded194k_stagec.pt",
    "inchausti-2025/data/checkpoint_best_efficientnet_stagec.pt",
    "inchausti-2025/data/checkpoint_best_meta_stagec.pt",
    "inchausti-2025/data/positives_curated.parquet",
    "inchausti-2025/data/negatives_extra.parquet",
    "inchausti-2025/data/storfer2024_published_catalog.csv",
    "inchausti-2025/data/inchausti2025_published_catalog.csv",
    "inchausti-2025/data/huang2021_published_catalog.csv",
    "inchausti-2025/data/operating_point.csv",
    "inchausti-2025/data/meta_metrics_staged.json",
    "inchausti-2025/data/meta_metrics_stagec.json",
    "inchausti-2025/data/training_split_staged.parquet",
    "inchausti-2025/data/training_split_stagec.parquet",
]


def rsync(host, remote, local, is_dir, dry):
    local = Path(local)
    src = f"{host}:{C.PHOENIX_RAID}/{remote}"
    cmd = ["rsync", "-aL", "--partial", "--info=stats1"]
    if dry:
        cmd.append("--dry-run")
    if is_dir:
        local.mkdir(parents=True, exist_ok=True)
        cmd += [src + "/", str(local) + "/"]   # trailing slash: copy CONTENTS into local dir
    else:
        local.parent.mkdir(parents=True, exist_ok=True)
        cmd += [src, str(local)]
    print(f"[rsync] {src} -> {local}")
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="phoenix")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # preflight
    if subprocess.run(["ssh", args.host, "true"]).returncode != 0:
        print(f"[abort] cannot ssh {args.host}", file=sys.stderr)
        return 1

    for remote, _ in DIRS:
        name = remote.split("/")[-1]
        rsync(args.host, remote, C.DATA / name, True, args.dry_run)
    for remote in FILES:
        rsync(args.host, remote, C.DATA / Path(remote).name, False, args.dry_run)

    if args.dry_run:
        print("[dry-run] done"); return 0

    # verify counts
    ok = True
    for remote, n_exp in DIRS:
        name = remote.split("/")[-1]
        got = len(list((C.DATA / name).glob("*.fits")))
        flag = "OK" if (n_exp is None or got >= int(0.98 * n_exp)) else "LOW"
        if flag != "OK":
            ok = False
        print(f"[verify] {name:42s} {got:>6d}/{n_exp} {flag}")
    missing = [Path(f).name for f in FILES if not (C.DATA / Path(f).name).exists()]
    if missing:
        ok = False
        print(f"[verify] MISSING files: {missing}")
    print("[sync] OK" if ok else "[sync] INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
