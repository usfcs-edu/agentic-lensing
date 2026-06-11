#!/usr/bin/env python3
"""make_tier1_submanifest.py — build a small, byte-bounded sub-manifest for the
Tier-1 from-scratch training fidelity check (layer b).

Selects the smallest sv3-bright pixels from the full manifest until ~target spectra
are gathered, and rewrites the coadd/redrock paths to a Mac-local data root (no /raid
symlink needed for Tier-1). Prints the chosen pixel rel-dirs (consumed by the sync).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-manifest", default=str(
        HERE / "_raid/benson/data/desi_dr1_medium/manifest.jsonl"))
    ap.add_argument("--out", default=str(HERE / "data/tier1_submanifest.jsonl"))
    ap.add_argument("--pixels-out", default=str(HERE / "data/.tier1_pixels.txt"))
    ap.add_argument("--out-data-root", default=str(
        HERE / "_raid_tier1/benson/data/desi_dr1_medium"))
    ap.add_argument("--target-spectra", type=int, default=2400)
    ap.add_argument("--min-rows", type=int, default=120)
    ap.add_argument("--max-rows", type=int, default=350)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.full_manifest)]
    cand = [r for r in recs if a.min_rows <= r["n_rows"] <= a.max_rows]
    cand.sort(key=lambda r: r["n_rows"])  # smallest first → least I/O

    chosen, total, pixels = [], 0, []
    for r in cand:
        rel = os.path.dirname(r["coadd"]).split("desi_dr1_medium/", 1)[-1]
        coadd = os.path.join(a.out_data_root, rel, os.path.basename(r["coadd"]))
        redrock = os.path.join(a.out_data_root, rel, os.path.basename(r["redrock"]))
        chosen.append({"coadd": coadd, "redrock": redrock, "n_rows": r["n_rows"],
                       "survey": r["survey"], "program": r["program"], "healpix": r["healpix"]})
        pixels.append(rel)
        total += r["n_rows"]
        if total >= a.target_spectra:
            break

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in chosen:
            f.write(json.dumps(r) + "\n")
    Path(a.pixels_out).write_text("\n".join(pixels) + "\n")
    print(f"[tier1] {len(chosen)} pixels, {total} spectra -> {a.out}")
    print(f"[tier1] pixel rel-dirs -> {a.pixels_out}")


if __name__ == "__main__":
    main()
