#!/usr/bin/env python3
"""
14_crossmatch_recovery.py — Phase 3b M4.

Cross-match our inference_scores.parquet against the published Huang+2020
catalog (huang2020_published_catalog.csv). For each published candidate,
find the nearest match in our scored set within 5″, then report the
fraction recovered at thresholds {0.5, 0.7, 0.9} broken out by grade.

Also produces score-histogram figure for the tech-report.

Outputs:
  data/recovery_summary.csv     (per-grade x threshold counts)
  data/recovery_matched.csv     (per-candidate: published name + our score)
  papers/figures/recovery_by_grade.png
  papers/figures/score_histogram.png
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


def main() -> None:
    pub_path = DATA / "huang2020_published_catalog.csv"
    scr_path = DATA / "inference_scores.parquet"
    if not pub_path.exists():
        raise SystemExit(f"missing {pub_path} — run 13_extract_huang2020_catalog.py first")
    if not scr_path.exists():
        raise SystemExit(f"missing {scr_path} — run 11_stream_inference_dr7.py + 12_merge_shards.py first")

    pub = pd.read_csv(pub_path)
    scr = pd.read_parquet(scr_path)
    print(f"[init] published: {len(pub)} candidates  scored: {len(scr):,} galaxies")

    pub_sky = SkyCoord(ra=pub["RA"].values * u.deg, dec=pub["DEC"].values * u.deg)
    scr_sky = SkyCoord(ra=scr["ra"].values * u.deg, dec=scr["dec"].values * u.deg)

    # For each published candidate, find nearest neighbour in our scored set.
    idx, sep2d, _ = pub_sky.match_to_catalog_sky(scr_sky)
    sep_arcsec = sep2d.to(u.arcsec).value
    matched_score = scr["score"].values[idx]

    matched_df = pub.copy()
    matched_df["nearest_row_id"] = scr["row_id"].values[idx]
    matched_df["nearest_sep_arcsec"] = sep_arcsec
    matched_df["nearest_score"] = matched_score
    matched_df["in_scored_set"] = sep_arcsec < MATCH_RADIUS_ARCSEC
    matched_df.to_csv(DATA / "recovery_matched.csv", index=False)
    print(f"[done] wrote data/recovery_matched.csv  "
          f"({int(matched_df['in_scored_set'].sum())}/{len(matched_df)} within {MATCH_RADIUS_ARCSEC:.0f}″)")

    # Build summary table: rows=grade, cols=thresholds
    rows = []
    for grade in ("A", "B", "C"):
        gsel = matched_df["grade"] == grade
        n_g = int(gsel.sum())
        n_in = int((gsel & matched_df["in_scored_set"]).sum())
        row = {"grade": grade, "n_published": n_g, "n_in_parent_sample": n_in}
        for t in THRESHOLDS:
            n_pass = int((gsel & matched_df["in_scored_set"]
                          & (matched_df["nearest_score"] >= t)).sum())
            row[f"n_score_ge_{t:.1f}"] = n_pass
            row[f"frac_score_ge_{t:.1f}"] = n_pass / max(n_g, 1)
        rows.append(row)
    summary = pd.DataFrame(rows)
    # Aggregate "ALL"
    n_all = len(matched_df)
    n_in_all = int(matched_df["in_scored_set"].sum())
    row = {"grade": "ALL", "n_published": n_all, "n_in_parent_sample": n_in_all}
    for t in THRESHOLDS:
        n_pass = int((matched_df["in_scored_set"]
                      & (matched_df["nearest_score"] >= t)).sum())
        row[f"n_score_ge_{t:.1f}"] = n_pass
        row[f"frac_score_ge_{t:.1f}"] = n_pass / max(n_all, 1)
    summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
    summary.to_csv(DATA / "recovery_summary.csv", index=False)
    print("\n[summary] recovery by grade × threshold:")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))

    # Figure 1: stacked bar of recovery fraction by grade
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    grades = ["A", "B", "C", "ALL"]
    width = 0.25
    xs = np.arange(len(grades))
    colors = ["#2c7bb6", "#abd9e9", "#fdae61"]
    for i, t in enumerate(THRESHOLDS):
        fracs = [float(summary[summary.grade == g][f"frac_score_ge_{t:.1f}"].iloc[0])
                 for g in grades]
        ax.bar(xs + (i - 1) * width, fracs, width=width,
               label=f"p ≥ {t:.1f}", color=colors[i])
    ax.set_xticks(xs)
    ax.set_xticklabels(grades)
    ax.set_xlabel("Huang+2020 grade")
    ax.set_ylabel("recovered fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 3b recovery of Huang+2020 candidates")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "recovery_by_grade.png", dpi=140)
    plt.close(fig)
    print(f"[done] wrote {FIGDIR / 'recovery_by_grade.png'}")

    # Figure 2: score histogram (log-y) for the full parent sample, with
    # the published-candidates' scores overlaid
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(scr["score"].values, bins=bins, alpha=0.55,
            color="#666666", label=f"all parent ({len(scr):,})")
    pub_in = matched_df[matched_df["in_scored_set"]]
    for grade, color in zip(("A", "B", "C"), ("#2c7bb6", "#5e95c4", "#a4c8e1")):
        sub = pub_in[pub_in["grade"] == grade]["nearest_score"].values
        if len(sub) > 0:
            ax.hist(sub, bins=bins, alpha=0.85, label=f"grade {grade} ({len(sub)})",
                    color=color, edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("ResNet sigmoid score")
    ax.set_ylabel("count")
    ax.set_title("Score distribution: parent sample vs. published candidates")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGDIR / "score_histogram.png", dpi=140)
    plt.close(fig)
    print(f"[done] wrote {FIGDIR / 'score_histogram.png'}")


if __name__ == "__main__":
    main()
