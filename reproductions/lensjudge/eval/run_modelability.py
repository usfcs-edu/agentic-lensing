#!/usr/bin/env python3
"""Standalone modelability ablation: run the Foundry-I lens-model fit over an eval
manifest (GPU-sharded) and score its lens_score against the consensus labels.

This is the direct answer to "does the modelability criterion lift the imaging AUC off
chance": we compute the GIGA-Lens lens_score per candidate and report its ROC-AUC for
(a) lens A/B/C vs Grade-D human-rejects (the hard pool the vision graders fail on) and
(b) A vs random-galaxy negatives (the easier pool). Cubes are fetched in this (JAX-free)
process, then a quicklens_proto subprocess per GPU fits its shard (JIT amortized).

  python lensjudge/eval/run_modelability.py --manifest lensjudge/outputs/lensbench_slice48.csv \
         --gpus 0,1,2,3,4,5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.io import fits as _fits  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import fetch  # noqa: E402

FITS_DIR = config.CACHE / "quicklens" / "fits"   # for fetched (off-disk) cubes


def _materialize(man: pd.DataFrame) -> list[dict]:
    """Resolve each candidate to a (3,101,101) FITS path (the proto's batch loader
    reads FITS primary-HDU). On-disk cutouts are passed directly; fetched cubes
    (Grade-D / gold) are written to a FITS. Returns [{name,fits,grade,source}]."""
    FITS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for _, r in man.iterrows():
        name = str(r["name"])
        survey = r.get("survey_key", "storfer")
        on_disk = fetch.on_disk_path(name, survey)
        if on_disk is not None:
            path = on_disk
        else:
            path = FITS_DIR / f"{name}.fits"
            if not path.exists():
                cube = fetch.get_cube(name=name, ra=r.get("ra"), dec=r.get("dec"),
                                      survey=survey)
                if cube is None:
                    continue
                _fits.PrimaryHDU(data=np.asarray(cube, dtype=np.float32)).writeto(
                    path, overwrite=True)
        items.append({"name": name, "fits": str(path),
                      "grade": str(r.get("grade_truth", r.get("grade"))),
                      "source": r.get("source", "?")})
    return items


def _run_shard(gpu: str, paths: list[str]) -> str:
    env = {"CUDA_VISIBLE_DEVICES": gpu, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
           "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.6", "TF_GPU_ALLOCATOR": "cuda_malloc_async",
           "PATH": "/usr/bin:/bin"}
    return subprocess.Popen(
        [str(config.GIGALENS_PY), str(config.QUICKLENS_SCRIPT), *paths],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)


def _auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    y = [1] * len(pos) + [0] * len(neg)
    s = list(pos) + list(neg)
    if len(set(y)) < 2 or len(s) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(config.OUT / "lensbench_slice48.csv"))
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--out", default=str(config.OUT / "modelability_scores.parquet"))
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    items = _materialize(man)
    print(f"[modelability] {len(items)} cubes materialized; sharding across GPUs {args.gpus}")
    gpus = args.gpus.split(",")
    shards = {g: [] for g in gpus}
    for i, it in enumerate(items):
        shards[gpus[i % len(gpus)]].append(it["fits"])

    procs = {g: _run_shard(g, p) for g, p in shards.items() if p}
    by_path = {}
    for g, proc in procs.items():
        out, _ = proc.communicate()
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("{") and '"path"' in line:
                try:
                    rec = json.loads(line)
                    by_path[rec["path"]] = rec
                except json.JSONDecodeError:
                    pass

    rows = []
    for it in items:
        rec = by_path.get(it["fits"], {})
        rows.append({**it, "lens_score": rec.get("lens_score"),
                     "theta_E": rec.get("theta_E"), "dchi2_frac": rec.get("dchi2_frac"),
                     "plausible": rec.get("plausible"), "n_images": rec.get("n_images")})
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    ok = df[df.lens_score.notna()]
    print(f"[modelability] fits OK {len(ok)}/{len(df)}")

    lens = ok[ok.grade.isin(["A", "B", "C"])]["lens_score"]
    A = ok[ok.grade == "A"]["lens_score"]
    D = ok[ok.grade == "D"]["lens_score"]
    rand = ok[ok.source == "random_neg"]["lens_score"]
    gradeD = ok[ok.source == "graded_D"]["lens_score"]
    print(f"\n  mean lens_score by grade: "
          f"{ok.groupby('grade').lens_score.mean().round(3).to_dict()}")
    print(f"  AUC  A/B/C-vs-D(all)     = {_auc(lens, D):.3f}  (the hard-pool question)")
    print(f"  AUC  A-vs-random_neg     = {_auc(A, rand):.3f}")
    print(f"  AUC  A-vs-graded_D reject= {_auc(A, gradeD):.3f}")
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
