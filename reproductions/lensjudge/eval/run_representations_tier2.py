#!/usr/bin/env python3
"""Tier-2 representation-feature AUC (photutils isophote / skimage frangi / SEP).

Resolves each manifest row to a FITS cube, runs tools/representations_proto.py in
.venvs/lens across CPU-parallel chunks, and reports Tier-2 feature ROC-AUC (with
bootstrap CIs) on the same HARD / EASY / GOLD contrasts as the Tier-1 gate — to test
whether cleaner algorithms lift the modest easy-regime / gold numbers.

  python lensjudge/eval/run_representations_tier2.py --manifest lensjudge/outputs/lensbench_large.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.io import fits as _fits  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import fetch  # noqa: E402
from lensjudge.eval.run_representations import _auc  # noqa: E402

LENS_PY = "/home/benson/.venvs/lens/bin/python"
PROTO = config.HERE / "tools" / "representations_proto.py"
FITS_DIR = config.CACHE / "quicklens" / "fits"


def _fits_path(r):
    name = str(r["name"]); survey = r.get("survey_key", "storfer")
    p = fetch.on_disk_path(name, survey)
    if p is not None:
        return name, str(p), str(r.get("grade_truth", "")), r.get("source", "?")
    fp = FITS_DIR / f"{name}.fits"
    if not fp.exists():
        cube = fetch.get_cube(name=name, ra=r.get("ra"), dec=r.get("dec"), survey=survey)
        if cube is None:
            return None
        FITS_DIR.mkdir(parents=True, exist_ok=True)
        _fits.PrimaryHDU(np.asarray(cube, np.float32)).writeto(fp, overwrite=True)
    return name, str(fp), str(r.get("grade_truth", "")), r.get("source", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(config.OUT / "lensbench_large.csv"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(config.OUT / "representation_tier2.parquet"))
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    items = [x for x in (_fits_path(r) for _, r in man.iterrows()) if x]
    print(f"[tier2] {len(items)} cubes; running proto in .venvs/lens across {args.workers} chunks")
    by_path = {it[1]: it for it in items}
    chunks = [items[i::args.workers] for i in range(args.workers)]

    def run_chunk(chunk):
        if not chunk:
            return ""
        paths = [c[1] for c in chunk]
        return subprocess.run([LENS_PY, str(PROTO), *paths], capture_output=True,
                              text=True, timeout=1800).stdout

    recs = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for out in ex.map(run_chunk, chunks):
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("{") and '"path"' in line:
                    try:
                        d = json.loads(line); recs[d["path"]] = d
                    except json.JSONDecodeError:
                        pass

    rows = []
    for name, path, grade, source in items:
        d = recs.get(path, {})
        rows.append({"name": name, "grade": grade, "source": source,
                     **{k: d.get(k) for k in ("iso_residual_flux_fraction", "frangi_arcness",
                                              "sep_n_sources", "sep_n_tangential", "sep_max_ellip")}})
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    ok = df["frangi_arcness"].notna().sum()
    print(f"[tier2] features OK {ok}/{len(df)} -> {args.out}")

    feats = ["iso_residual_flux_fraction", "frangi_arcness", "sep_n_sources",
             "sep_n_tangential", "sep_max_ellip"]
    g = df.source == "graded"
    contrasts = {
        "HARD (A/B/C vs graded_D)": (g & df.grade.isin(["A", "B", "C"]), df.source == "graded_D"),
        "EASY (A vs random_neg)": (g & (df.grade == "A"), df.source == "random_neg"),
        "GOLD (confirmed vs non-lens)": ((df.source == "gold") & (df.grade == "A"),
                                         (df.source == "gold") & (df.grade == "D")),
    }
    for cname, (pm, nm) in contrasts.items():
        print(f"\n=== {cname}: {int(pm.sum())} vs {int(nm.sum())} ===")
        for c in feats:
            a, (lo, hi) = _auc(df[pm][c], df[nm][c])
            star = " *" if (lo > 0.5 or hi < 0.5) else ""
            print(f"   {c:30s} AUC={a:.3f}  [{lo:.2f},{hi:.2f}]{star}")


if __name__ == "__main__":
    main()
