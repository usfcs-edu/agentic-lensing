#!/usr/bin/env python3
"""
04_scaffold_symlinks.py

Phase-5 scaffolding. Build the data/ symlinks into the sibling huang-2020 /
huang-2021 reproductions (training cutouts, positives/negatives, the DR8 parent-
sample cutouts + two-model scores, baseline checkpoints, published catalogs),
pre-fetch the EfficientNetV2-S pretrained weights so training is offline-safe,
and print an environment / disk / GPU report.

Idempotent: re-running only (re)creates missing or wrong symlinks; existing real
files (trained checkpoints, score parquets, catalogs produced by later scripts)
are never touched.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
H2020 = (HERE.parent / "huang-2020" / "data").resolve()
H2021 = (HERE.parent / "huang-2021" / "data").resolve()

# symlink-name -> real target. Training inputs reuse huang-2020; DR8 parent
# sample + scores reuse huang-2021; baseline checkpoints from both.
LINKS = {
    # --- training inputs (reused, never re-downloaded) ---
    "cutouts_fits_dr9":             H2020 / "cutouts_fits_dr9",
    "cutouts_fits_dr7_train":       H2020 / "cutouts_fits_dr7_train",
    "positives_huang2020.parquet":  H2020 / "positives_huang2020.parquet",
    "positives_all.parquet":        H2020 / "positives_all.parquet",
    "negatives.parquet":            H2020 / "negatives.parquet",
    "neuralens_catalog.csv":        H2020 / "neuralens_catalog.csv",
    # --- baseline checkpoints (L18 + Huang-2021 60K shielded) for comparison ---
    "checkpoint_best_l18_dr9.pt":          H2020 / "checkpoint_best.pt",
    "checkpoint_best_shielded60k_dr9.pt":  H2021 / "checkpoint_best_shielded_dr9.pt",
    # --- DR8 parent sample + existing two-model scores (for ensemble re-score) ---
    "cutouts_fits_dr8":                       H2021 / "cutouts_fits_dr8",
    "parent_dr8.parquet":                     H2021 / "parent_dr8.parquet",
    "brick_manifest_dr8.csv":                 H2021 / "brick_manifest_dr8.csv",
    "inference_scores_l18_dr8.parquet":       H2021 / "inference_scores_l18_dr8.parquet",
    "inference_scores_shielded_dr8.parquet":  H2021 / "inference_scores_shielded_dr8.parquet",
    # --- published Huang-2021 catalog (leak / provenance overlap) ---
    "huang2021_published_catalog.csv":  H2021 / "huang2021_published_catalog.csv",
}


def build_links() -> None:
    print(f"[scaffold] DATA = {DATA}")
    DATA.mkdir(exist_ok=True)
    n_ok = n_miss = n_made = 0
    for name, target in LINKS.items():
        link = DATA / name
        if not target.exists():
            print(f"  [MISS] {name:42s} -> target absent: {target}")
            n_miss += 1
            continue
        # Refresh only if not already a symlink pointing at the right target.
        if link.is_symlink() and Path(os.readlink(link)) == target:
            n_ok += 1
            continue
        if link.is_symlink() or link.exists():
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                print(f"  [SKIP] {name}: real directory present, not overwriting")
                n_ok += 1
                continue
        link.symlink_to(target)
        print(f"  [LINK] {name:42s} -> {target}")
        n_made += 1
    print(f"[scaffold] symlinks: {n_ok} ok, {n_made} created, {n_miss} missing targets")


def prefetch_effnet_weights() -> None:
    print("\n[scaffold] pre-fetching EfficientNetV2-S pretrained weights ...")
    try:
        import timm
        m = timm.create_model("tf_efficientnetv2_s", pretrained=True,
                              num_classes=0, in_chans=3)
        n = sum(p.numel() for p in m.parameters())
        print(f"  [OK] tf_efficientnetv2_s pretrained backbone loaded "
              f"({n:,} params); weights now cached for offline training.")
    except Exception as e:
        print(f"  [WARN] could not fetch pretrained weights: {e}")
        print("         training will fall back to random init (hurts EffNet AUC); "
              "see report.")


def env_report() -> None:
    print("\n[scaffold] environment:")
    try:
        import torch
        print(f"  torch   {torch.__version__}  cuda_available={torch.cuda.is_available()}  "
              f"n_gpu={torch.cuda.device_count()}")
    except Exception as e:
        print(f"  torch   import failed: {e}")
    try:
        import timm
        print(f"  timm    {timm.__version__}")
    except Exception as e:
        print(f"  timm    import failed: {e}")
    # GPU snapshot (PCI_BUS_ID ordering: A16s=0-7, L4s=8,9).
    try:
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.free,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, env=env, timeout=30)
        print("  GPUs (PCI_BUS_ID order — use 0-7 for A16s, leave 8,9 for L4s):")
        for line in out.stdout.strip().splitlines():
            print(f"    {line}")
    except Exception as e:
        print(f"  nvidia-smi failed: {e}")
    # Disk.
    try:
        usage = shutil.disk_usage("/raid")
        print(f"  /raid free: {usage.free / 1e12:.1f} TB / {usage.total / 1e12:.1f} TB")
    except Exception as e:
        print(f"  disk check failed: {e}")


if __name__ == "__main__":
    build_links()
    prefetch_effnet_weights()
    env_report()
    print("\n[scaffold] done.")
