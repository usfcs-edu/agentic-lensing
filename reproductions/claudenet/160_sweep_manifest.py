#!/usr/bin/env python3
"""160_sweep_manifest.py — Phase 160: FULL-DR9-parent sweep manifests for the
two-stage lens scan (runs LOCALLY, CPU only).

POPULATION (deliberate, fixed): the sweep covers data/parent_dr8.parquet —
the 17,290,814 non-PSF DEV/COMP galaxies of the Huang parent sample, i.e. the
SAME parent population the Huang/Storfer/Inchausti lineage scans (their
deployment selection). It is NOT an all-type ~45M selection: staying on the
lineage population keeps the sweep comparable to the published searches and
keeps the NegEval-1M thresholds (drawn from this same population) honest.
Two further deliberate choices, both opposite to 110_build_negpool_manifest:
  * known lenses are NOT excluded — they are the recall sanity check
    (163_crossmatch_known.py flags the recovered ones);
  * v1-negative bricks are NOT excluded — the sweep is a deployment scan,
    not an eval set, so leak guards do not apply.
The only filter is grz coverage (NOBS_G/R/Z >= --min-nobs, default 1), the
same coverage requirement 111_extract_cutouts_cfs.py needs to extract at all.

Output: --chunk K manifest part files under data/v2/sweep/, each in 111's
manifest schema [row_id, RA, DEC, footprint, brick] with row_id =
"<f>_<BRICKID>_<OBJID>" where <f> is the footprint initial (n/s). NOTE this
DIFFERS from 110's "<BRICKID>_<OBJID>" scheme on purpose: 110 sampled each
sky brick into exactly one footprint so it never collided, but the full
parent has 2,652 (BRICKID, OBJID) pairs duplicated across the two footprints
(overlap strips) — the footprint qualifier keeps row_id globally unique,
which 111/161–165 all rely on (they join on row_id). Both collision counts
(cross-footprint row_id pairs, shared bricknames) are RECOMPUTED at build
time and written into sweep_manifest_summary.json, so the numbers quoted in
this prose are regenerated, not hard-coded.
Parts are FOOTPRINT-PURE (each part is entirely north or entirely south,
allocated ~proportionally to row counts — at least one part per footprint)
and BRICK-DISJOINT keyed on (footprint, brick) — 4,647 bricknames occur in
BOTH footprints, so brickname alone is NOT a valid disjointness key; the
same name in different footprints means different coadd files. Rows are
sorted by footprint/brick/OBJID and split only at brick boundaries, so K
independent `111_extract_cutouts_cfs.py` slurm jobs (one --out-root per
part) never read the same coadd brick. Deterministic: no sampling, no seed.

SCRATCH COST of the full extraction (documented, deliberate): 17,290,814
rows x 3 bands x 101 x 101 x 4 bytes = ~2.12 TB of cutout shards across the
K part roots (plus small per-shard index parquets). $SCRATCH has 20 TB and
the NegEval-1M rate was 8.8 min/1M rows on 60 CPU workers, so the full sweep
extraction is ~2.5 h of one fat CPU node, or K parallel shared-CPU jobs.

    /home2/benson/.venvs/claudenet/bin/python 160_sweep_manifest.py --chunk 8
    # logic check on a small synthetic parent, no writes:
    python 160_sweep_manifest.py --parent /tmp/parent_tiny.parquet \
        --chunk 4 --dry-run

    # downstream (orchestrator, one extraction job per part):
    sbatch --export=ALL,CMD='python 111_extract_cutouts_cfs.py \
        --manifest data/v2/sweep/sweep_manifest_part00of08.parquet \
        --out-root $SCRATCH/claudenet/cutouts/sweep/part00 \
        --size 101 --bands grz --release dr9 --workers 60' nersc/shared_cpu.slurm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C

SWEEP = C.DATA / "v2" / "sweep"
COLS = ["BRICKID", "BRICKNAME", "OBJID", "RA", "DEC",
        "NOBS_G", "NOBS_R", "NOBS_Z", "footprint"]


def allocate_parts(counts: pd.Series, k: int) -> dict:
    """Rows-proportional part counts per footprint, >=1 each, summing to k
    when possible (k < n_footprints forces one part per footprint anyway)."""
    foots = sorted(counts.index)
    total = int(counts.sum())
    alloc = {f: max(1, int(round(k * counts[f] / total))) for f in foots}
    # fix rounding drift toward exactly k (never below 1 per footprint)
    while sum(alloc.values()) > k:
        cands = [f for f in foots if alloc[f] > 1]
        if not cands:
            break
        alloc[max(cands, key=lambda f: alloc[f])] -= 1
    while sum(alloc.values()) < k:
        alloc[max(foots, key=lambda f: counts[f] / alloc[f])] += 1
    return alloc


def split_at_bricks(sub: pd.DataFrame, n_parts: int) -> list:
    """Split one footprint's brick-sorted rows into <= n_parts contiguous
    slices, cutting ONLY at brick boundaries (a brick never straddles parts)."""
    bricks = sub["brick"].to_numpy()
    # row index just past the END of each brick group (cumulative counts)
    ends = np.r_[np.flatnonzero(bricks[1:] != bricks[:-1]) + 1, len(sub)]
    cuts = [0]
    for j in range(1, n_parts):
        target = int(round(j * len(sub) / n_parts))
        e = int(ends[np.searchsorted(ends, target, side="left")])
        if e > cuts[-1] and e < len(sub):
            cuts.append(e)
    cuts.append(len(sub))
    return [sub.iloc[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent", default=str(C.DATA / "parent_dr8.parquet"),
                    help="parent parquet (override with a small sample for dry runs)")
    ap.add_argument("--chunk", type=int, default=8,
                    help="number of manifest part files (= parallel 111 extraction jobs)")
    ap.add_argument("--min-nobs", type=int, default=1,
                    help="require NOBS_G/R/Z >= this (the 111 extractability filter)")
    ap.add_argument("--out-dir", default=str(SWEEP))
    ap.add_argument("--dry-run", action="store_true", help="run everything but skip writes")
    args = ap.parse_args()
    t0 = time.time()
    assert args.chunk >= 1, "--chunk must be >= 1"
    out_dir = Path(args.out_dir)

    # 1. parent + the ONLY filter (grz coverage); population is deliberate, see docstring
    p = pd.read_parquet(args.parent, columns=COLS)
    n0 = len(p)
    p = p[(p.NOBS_G >= args.min_nobs) & (p.NOBS_R >= args.min_nobs)
          & (p.NOBS_Z >= args.min_nobs)]
    print(f"[parent] {n0:,} rows -> {len(p):,} after NOBS_GRZ>={args.min_nobs} "
          f"(known lenses + v1 bricks deliberately KEPT — deployment sweep)")
    if p.empty:
        print("[160] FATAL: no rows after the coverage filter")
        return 1

    # 1b. regenerate the two documented collision counts (cheap groupbys) so the
    #     docstring's 2,652 / 4,647 figures live in the summary json, not prose
    pairs = p[["BRICKID", "OBJID", "footprint"]].drop_duplicates()
    n_xfoot = int(pairs.duplicated(["BRICKID", "OBJID"]).sum())
    bn = p[["BRICKNAME", "footprint"]].drop_duplicates()
    n_shared_bn = int(bn.duplicated("BRICKNAME").sum())
    del pairs, bn
    print(f"[parent] {n_xfoot:,} cross-footprint (BRICKID, OBJID) collisions; "
          f"{n_shared_bn:,} bricknames present in BOTH footprints")

    # 2. 111-schema manifest, brick-sorted (footprint -> brick -> OBJID).
    #    row_id is FOOTPRINT-QUALIFIED (differs from 110, see docstring): the
    #    real parent has 2,652 cross-footprint (BRICKID, OBJID) collisions.
    df = pd.DataFrame({
        "row_id": (p.footprint.astype(str).str[0] + "_"
                   + p.BRICKID.astype(int).astype(str) + "_"
                   + p.OBJID.astype(int).astype(str)),
        "RA": p.RA.astype(float), "DEC": p.DEC.astype(float),
        "footprint": p.footprint.astype(str), "brick": p.BRICKNAME.astype(str),
        "_objid": p.OBJID.astype(int),
    })
    del p
    df = df.sort_values(["footprint", "brick", "_objid"], ignore_index=True)
    df = df.drop(columns="_objid")
    assert df.row_id.is_unique, \
        "duplicate <footprint>_<BRICKID>_<OBJID> row_ids in parent"

    # 3. footprint-pure, brick-disjoint parts
    counts = df.footprint.value_counts()
    foots = sorted(counts.index)
    alloc = allocate_parts(counts, args.chunk)
    print("[alloc] " + ", ".join(f"{f}: {counts[f]:,} rows -> {alloc[f]} part(s)"
                                 for f in foots))
    parts = []
    for f in foots:
        got = split_at_bricks(df[df.footprint == f], alloc[f])
        if len(got) < alloc[f]:
            print(f"[warn] {f}: only {len(got)}/{alloc[f]} parts "
                  f"(fewer bricks than requested parts)")
        parts.extend(got)
    k = len(parts)

    # 4. assert exact coverage + brick disjointness across parts. Keyed on
    #    (footprint, brick): 4,647 bricknames exist in BOTH footprints, so a
    #    name-only key would false-fire across the north/south part boundary.
    assert sum(len(q) for q in parts) == len(df), "parts do not cover the manifest"
    seen: set = set()
    for q in parts:
        b = set(zip(q.footprint, q.brick))
        assert not (b & seen), "a (footprint, brick) group straddles two parts"
        seen |= b
    print(f"[assert] {k} parts cover {len(df):,} rows / {len(seen):,} "
          f"(footprint, brick) groups; footprint-pure + brick-disjoint: OK")

    # 5. write parts + summary
    summary = {"parent": args.parent, "min_nobs": args.min_nobs,
               "row_id_scheme": "<footprint-initial>_<BRICKID>_<OBJID> "
                                "(footprint-qualified; differs from 110)",
               "n_parent": n0, "n_manifest": len(df),
               "n_crossfootprint_rowid_collisions": n_xfoot,
               "n_shared_bricknames": n_shared_bn,
               "n_bricks": len(seen),   # distinct (footprint, brick) groups
               "per_footprint": {f: int(counts[f]) for f in foots},
               "n_parts": k, "parts": []}
    for j, q in enumerate(parts):
        name = f"sweep_manifest_part{j:02d}of{k:02d}.parquet"
        info = {"file": name, "footprint": q.footprint.iloc[0],
                "n_rows": len(q), "n_bricks": int(q.brick.nunique())}
        summary["parts"].append(info)
        print(f"[part] {name}: {info['n_rows']:,} rows, {info['n_bricks']:,} bricks "
              f"({info['footprint']})")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / (name + ".tmp")
            q.reset_index(drop=True).to_parquet(tmp, index=False)
            tmp.rename(out_dir / name)
    if args.dry_run:
        print(f"[dry-run] skipping writes ({time.time() - t0:.1f}s)")
        return 0
    sf = out_dir / "sweep_manifest_summary.json"
    sf.write_text(json.dumps(summary, indent=2))
    print(f"[done] {k} parts + {sf} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
