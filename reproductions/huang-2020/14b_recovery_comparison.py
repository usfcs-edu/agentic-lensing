#!/usr/bin/env python3
"""
14b_recovery_comparison.py — side-by-side DR9-trained vs DR7-trained recovery.

Cross-matches both Phase 3b score parquets (built by separate runs of
11b_brick_inference_dr7.py with different checkpoints) against the published
Huang+2020 catalog and produces a comparison table + figure.

Inputs:
  data/inference_scores_dr9trained.parquet   (Phase 3b with checkpoint_best.pt)
  data/inference_scores_dr7trained.parquet   (Phase 3b with checkpoint_best_dr7.pt)
  data/huang2020_published_catalog.csv

Outputs:
  data/recovery_compare.csv
  papers/figures/recovery_compare.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGDIR = HERE / "papers" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

MATCH_RADIUS_ARCSEC = 5.0
THRESHOLDS = (0.5, 0.7, 0.9)
RUNS = (
    ("DR9-trained", DATA / "inference_scores_dr9trained.parquet"),
    ("DR7-trained", DATA / "inference_scores_dr7trained.parquet"),
)


def crossmatch_one(pub: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    pub_sky = SkyCoord(ra=pub["RA"].values * u.deg, dec=pub["DEC"].values * u.deg)
    scr_sky = SkyCoord(ra=scores["ra"].values * u.deg, dec=scores["dec"].values * u.deg)
    idx, sep2d, _ = pub_sky.match_to_catalog_sky(scr_sky)
    sep = sep2d.to(u.arcsec).value
    out = pub.copy()
    out["sep_arcsec"] = sep
    out["score"] = scores["score"].values[idx]
    out["in_parent"] = sep < MATCH_RADIUS_ARCSEC
    return out


def main() -> None:
    pub = pd.read_csv(DATA / "huang2020_published_catalog.csv")
    print(f"[init] published candidates: {len(pub)}")

    matched_by_run: dict[str, pd.DataFrame] = {}
    for label, path in RUNS:
        if not path.exists():
            raise SystemExit(f"missing {path}")
        scr = pd.read_parquet(path)
        m = crossmatch_one(pub, scr)
        matched_by_run[label] = m
        print(f"[{label}] scored={len(scr):,}  in_parent={int(m['in_parent'].sum())}/{len(m)}  "
              f"p>=0.9={int((m['in_parent'] & (m['score'] >= 0.9)).sum())}")

    # Build comparison: rows = grade × threshold, cols = run-fraction
    rows = []
    grades = ("A", "B", "C", "ALL")
    for grade in grades:
        for t in THRESHOLDS:
            row = {"grade": grade, "threshold": t}
            for label, m in matched_by_run.items():
                gsel = (m["grade"] == grade) if grade != "ALL" else pd.Series([True] * len(m))
                ng = int(gsel.sum())
                npass = int((gsel & m["in_parent"] & (m["score"] >= t)).sum())
                row[f"n_pass_{label}"] = npass
                row[f"frac_{label}"] = npass / max(ng, 1)
            # Delta in fractions
            row["delta_DR7_minus_DR9"] = (row["frac_DR7-trained"] - row["frac_DR9-trained"])
            rows.append(row)
    cmp = pd.DataFrame(rows)
    cmp.to_csv(DATA / "recovery_compare.csv", index=False)
    print("\n[summary] recovery comparison:")
    with pd.option_context("display.float_format", "{:.3f}".format,
                            "display.width", 140):
        print(cmp.to_string(index=False))

    # Figure: grouped bars, grade × threshold × {DR9, DR7}
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    width = 0.35
    grade_xs = np.arange(3)  # A, B, C
    for ax, t in zip(axs, THRESHOLDS):
        for i, (label, m) in enumerate(matched_by_run.items()):
            fracs = []
            for g in ("A", "B", "C"):
                gsel = m["grade"] == g
                ng = int(gsel.sum())
                npass = int((gsel & m["in_parent"] & (m["score"] >= t)).sum())
                fracs.append(npass / max(ng, 1))
            ax.bar(grade_xs + (i - 0.5) * width, fracs, width,
                   label=label, color=("#2c7bb6", "#d7191c")[i])
        ax.set_xticks(grade_xs)
        ax.set_xticklabels(("A", "B", "C"))
        ax.set_xlabel("Huang+2020 grade")
        ax.set_title(f"recovery at p ≥ {t:.1f}")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    axs[0].set_ylabel("recovered fraction")
    axs[-1].legend(loc="lower left", framealpha=0.9)
    fig.suptitle("Phase 3b recovery: DR9-trained vs DR7-trained (paper-exact)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "recovery_compare.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {FIGDIR / 'recovery_compare.png'}")


if __name__ == "__main__":
    main()
