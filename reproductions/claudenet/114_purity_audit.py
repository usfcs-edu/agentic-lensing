#!/usr/bin/env python3
"""114_purity_audit.py — Phase 110 (NegEval-1M): PU-contamination audit of the
pool tail (runs LOCALLY, CPU).

NegEval-1M is PU-labelled: every row is *presumed* negative, so undiscovered
real lenses in the extreme tail would corrupt the matched-FPR thresholds. This
script audits the top-K pool rows by the v1 "average" combiner (the flagship):

  1. Cross-match the top-K RA/DEC (from the negeval manifest) against the SAME
     known-lens catalogs 110 used for its 10-arcsec exclusion (published
     Storfer/Inchausti/Huang CSVs + v1 curated positives, via
     110_build_negpool_manifest.load_lens_sky) at 10" AND 5". Any 10" match is
     reported as a BUG (110 already excluded those). Writes
     data/v2/audit/topk.csv (row_id, RA, DEC, score, rank, ... ) — this file
     doubles as the --row-ids input for 111b_dump_rows.py on Perlmutter.
  2. If --audit-npz exists (the top-K grz cutouts gathered remotely by
     111b_dump_rows.py and rsynced back): render Lupton-RGB contact sheets
     (R=z, G=r, B=g; Q=8.0, stretch=0.5 — the repo's
     16_build_inspection_viewer.py stretch) as rank-labelled 10x10 grids ->
     data/v2/audit/topk_page<i>.png, for human/model visual grading.
  3. Threshold sensitivity: recompute the 1e-3/1e-4 thresholds of the average
     combiner after deleting the top-j pool rows, j in {0,5,10,20,50,100,200}
     -> data/v2/audit/threshold_sensitivity.csv. This bounds how far
     undiscovered-lens contamination could move the operating point.

Workflow: run 113 first (it writes the pool combiner scores), then this script
WITHOUT --audit-npz to get topk.csv -> run 111b on Perlmutter with it -> rsync
the npz back -> rerun this script to render the contact sheets.

    python 114_purity_audit.py                       # steps 1+3 (and 2 if the
                                                     # default npz exists)
    python 114_purity_audit.py \\
        --pool-scores data/v2/scores_negeval_pool_combined.parquet \\
        --audit-npz data/v2/negeval_audit_topk.npz \\
        --manifest data/v2/negeval_manifest.parquet --topk 200
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C
import _ensemble as E

# Lupton stretch copied from inchausti-2025/16_build_inspection_viewer.py
LUPTON_Q = 8.0
LUPTON_STRETCH = 0.5
SENS_J = (0, 5, 10, 20, 50, 100, 200)
SENS_FPRS = (1e-3, 1e-4)
V2 = C.DATA / "v2"


def crossmatch_topk(top: pd.DataFrame) -> pd.DataFrame:
    """Nearest known-lens separation for each top-K row, using the SAME catalog
    construction as 110 (reused via importlib, not re-implemented)."""
    from astropy import units as u
    NP = C._load("cn_114_negpool", C.ROOT / "110_build_negpool_manifest.py")
    lens_sky, n_lens = NP.load_lens_sky()
    print(f"[audit] cross-matching top-{len(top)} vs {n_lens:,} known-lens positions")
    from astropy.coordinates import SkyCoord
    sky = SkyCoord(ra=top.RA.values * u.deg, dec=top.DEC.values * u.deg)
    _, sep, _ = sky.match_to_catalog_sky(lens_sky)
    top = top.copy()
    top["sep_arcsec"] = sep.to(u.arcsec).value
    top["match10"] = top.sep_arcsec < 10.0
    top["match5"] = top.sep_arcsec < 5.0
    return top


def render_contact_sheets(top: pd.DataFrame, npz_path: Path, out_dir: Path, grid: int,
                          title: str | None = None, prefix: str = "topk"):
    """Rank-ordered grid x grid Lupton-RGB contact sheets from the 111b npz
    (cutouts (n, 3, 101, 101) float32 in grz band order, row_ids, ok).
    `top` needs columns row_id/rank/score/match5; `title`/`prefix` default to
    the 114 purity-audit captions (164 reuses this for the sweep vet pages)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import make_lupton_rgb

    z = np.load(npz_path)
    cut, ok = z["cutouts"], z["ok"].astype(bool)
    loc = {str(r): i for i, r in enumerate(z["row_ids"])}
    n_missing = sum(str(r) not in loc for r in top.row_id)
    if n_missing:
        print(f"[sheet] WARNING: {n_missing} top-K row_ids absent from {npz_path.name}")
    per_page = grid * grid
    n_pages = (len(top) + per_page - 1) // per_page
    for page in range(1, n_pages + 1):
        rows = top.iloc[(page - 1) * per_page: page * per_page]
        fig, axes = plt.subplots(grid, grid, figsize=(grid * 1.7, grid * 1.85))
        for ax in np.ravel(axes):
            ax.set_axis_off()
        for ax, (_, r) in zip(np.ravel(axes), rows.iterrows()):
            i = loc.get(str(r.row_id))
            if i is not None and ok[i]:
                cube = np.nan_to_num(cut[i].astype(np.float32), nan=0.0)
                rgb = make_lupton_rgb(cube[2], cube[1], cube[0],
                                      Q=LUPTON_Q, stretch=LUPTON_STRETCH)
                ax.imshow(rgb[::-1, :, :], interpolation="nearest")
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=6, color="0.5")
            star = " *LENS5\"" if bool(r.match5) else ""
            ax.set_title(f"#{int(r['rank'])} p={r.score:.3f}{star}", fontsize=6,
                         color="#d7191c" if r.match5 else "black", pad=2)
        out = out_dir / f"{prefix}_page{page}.png"
        base = title or f"NegEval-1M top-{len(top)} by average combiner"
        fig.suptitle(f"{base} — page {page}/{n_pages} (rank order)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[sheet] wrote {out} ({len(rows)} tiles)")


def threshold_sensitivity(scores: np.ndarray, out_dir: Path) -> pd.DataFrame:
    """Thresholds of the average combiner after deleting the top-j pool rows."""
    desc = np.sort(scores[np.isfinite(scores)])[::-1]
    rows = []
    for j in SENS_J:
        rem = desc[j:]
        row = {"removed_top_j": j, "n_remaining": int(rem.size)}
        for f in SENS_FPRS:
            row[f"thr_fpr_{f:g}"] = E.fpr_threshold(rem, f)
        rows.append(row)
    df = pd.DataFrame(rows)
    for f in SENS_FPRS:
        df[f"dthr_fpr_{f:g}"] = df[f"thr_fpr_{f:g}"] - df[f"thr_fpr_{f:g}"].iloc[0]
    out = out_dir / "threshold_sensitivity.csv"
    df.to_csv(out, index=False)
    print(f"\n[sens] average-combiner threshold vs top-j removal -> {out}")
    print(f"[sens] {'j':>5} {'n_remaining':>12} "
          + " ".join(f"{'thr@' + format(f, 'g'):>12} {'delta':>10}" for f in SENS_FPRS))
    for _, r in df.iterrows():
        line = f"[sens] {int(r.removed_top_j):>5} {int(r.n_remaining):>12,}"
        for f in SENS_FPRS:
            line += f" {r[f'thr_fpr_{f:g}']:>12.6f} {r[f'dthr_fpr_{f:g}']:>+10.6f}"
        print(line)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool-scores", default=str(V2 / "scores_negeval_pool_combined.parquet"),
                    help="pool combiner scores written by 113 (needs --score-col)")
    ap.add_argument("--score-col", default="average",
                    help="combiner column to rank/threshold on")
    ap.add_argument("--audit-npz", default=str(V2 / "negeval_audit_topk.npz"),
                    help="top-K cutouts npz from 111b_dump_rows (skipped if absent)")
    ap.add_argument("--manifest", default=str(V2 / "negeval_manifest.parquet"))
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--grid", type=int, default=10, help="contact-sheet grid edge")
    ap.add_argument("--out-dir", default=str(V2 / "audit"))
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = pd.read_parquet(args.pool_scores)
    if args.score_col not in pool.columns:
        print(f"[audit] FATAL: column {args.score_col!r} not in {args.pool_scores} "
              f"(have {list(pool.columns)}). Run 113_thresholds_ci_evt.py first — it "
              f"writes scores_negeval_pool_combined.parquet with the combiner columns.")
        return 1
    pool = pool[np.isfinite(pool[args.score_col])]
    print(f"[audit] {len(pool):,} pool rows; ranking by {args.score_col!r}")

    # -- 1. top-K + cross-match vs the 110 known-lens catalogs -------------------
    top = pool.sort_values(args.score_col, ascending=False).head(args.topk).copy()
    top["rank"] = np.arange(1, len(top) + 1)
    top["score"] = top[args.score_col]
    man = pd.read_parquet(args.manifest)[["row_id", "RA", "DEC", "footprint", "brick"]]
    n0 = len(top)
    top = top.merge(man, on="row_id", how="inner")
    assert len(top) == n0, f"{n0 - len(top)} top-K row_ids missing from the manifest"
    top = crossmatch_topk(top)
    n10, n5 = int(top.match10.sum()), int(top.match5.sum())
    if n10:
        print(f"[audit] BUG: {n10} top-K rows within 10\" of a known lens — 110 "
              f"excluded these at build time; manifest/scores mismatch?")
        print(top[top.match10][["rank", "row_id", "RA", "DEC", "score", "sep_arcsec"]]
              .to_string(index=False))
    else:
        print(f"[audit] 10\" known-lens matches in top-{len(top)}: 0 (as built) ; "
              f"5\" matches: {n5}")
    cols = ["row_id", "RA", "DEC", "score", "rank", "footprint", "brick",
            "sep_arcsec", "match10", "match5"]
    topk_csv = out_dir / "topk.csv"
    top[cols].to_csv(topk_csv, index=False)
    print(f"[audit] wrote {topk_csv} (feed it to 111b_dump_rows.py --row-ids on "
          f"Perlmutter to gather the cutout npz)")

    # -- 2. contact sheets (only when the 111b npz has been rsynced back) --------
    npz_path = Path(args.audit_npz)
    if npz_path.exists():
        render_contact_sheets(top.sort_values("rank"), npz_path, out_dir, args.grid)
    else:
        print(f"[sheet] {npz_path} not found -> contact sheets skipped "
              f"(run 111b remotely with --row-ids {topk_csv.name}, rsync, rerun)")

    # -- 3. threshold sensitivity to top-j removal -------------------------------
    threshold_sensitivity(pool[args.score_col].to_numpy(dtype=np.float64), out_dir)

    print(f"\n[114] done ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
