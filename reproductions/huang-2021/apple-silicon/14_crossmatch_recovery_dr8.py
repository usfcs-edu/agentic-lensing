#!/usr/bin/env python3
"""
14_crossmatch_recovery_dr8.py — Phase 4c step 2.

Cross-match our DR8 inference scores (both models) against the 1,312 published
Huang+2021 candidates, and report recovery by grade × model × threshold —
separating the honest (leak-free) result from the leaked one.

Leakage structure (key to interpreting Huang 2021 the way Huang 2020 §11 did):
  Our two nets were trained on the 949 L18-model NeuraLens rows
  (positives_huang2020.parquet). The 363 shielded-model rows were NOT in
  training. So of the 1,312 published candidates:
    - the ~949 that match a training positive within 5″  -> LEAKED (inflated)
    - the ~363 shielded-only discoveries                  -> HONEST test
  We tag each candidate with in_training and report recovery for both buckets.
  The shielded model's recovery of the 363 leak-free candidates is the
  cleanest "did the architecture generalise" signal.

Recovery = published candidate has a scored galaxy within 5″ scoring >= t.
Paper operating point is p >= 0.1; we also report 0.5 and 0.9.

Inputs:
  data/huang2021_published_catalog.csv               (13_)
  data/inference_scores_l18_dr8.parquet              (12_)
  data/inference_scores_shielded_dr8.parquet         (12_)
  data/positives_huang2020.parquet                   (training positives)

Outputs:
  data/recovery_dr8_matched.csv      per-candidate nearest score per model
  data/recovery_dr8_summary.csv      grade × model × threshold × leak-bucket
  papers/figures/recovery_dr8_by_grade.png
  papers/figures/recovery_dr8_two_model.png
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

MATCH_RADIUS = 5.0  # arcsec
THRESHOLDS = (0.1, 0.5, 0.9)
MODELS = ("l18", "shielded")


def nearest_scores(pub_sky, scr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scr_sky = SkyCoord(ra=scr["ra"].values * u.deg, dec=scr["dec"].values * u.deg)
    idx, sep2d, _ = pub_sky.match_to_catalog_sky(scr_sky)
    return scr["score"].values[idx], sep2d.to(u.arcsec).value


def main() -> None:
    pub = pd.read_csv(DATA / "huang2021_published_catalog.csv")
    pub_sky = SkyCoord(ra=pub["RA"].values * u.deg, dec=pub["DEC"].values * u.deg)
    print(f"[init] {len(pub)} published candidates")

    # Per-model nearest score + separation.
    for kind in MODELS:
        path = DATA / f"inference_scores_{kind}_dr8.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path} — run 11b + 12 first")
        scr = pd.read_parquet(path)
        sc, sep = nearest_scores(pub_sky, scr)
        pub[f"score_{kind}"] = sc
        pub[f"sep_{kind}"] = sep
        pub[f"in_parent_{kind}"] = sep < MATCH_RADIUS
        print(f"[{kind}] scored {len(scr):,}; "
              f"{int(pub[f'in_parent_{kind}'].sum())}/{len(pub)} published within {MATCH_RADIUS:.0f}″")

    # Leakage tag: within 5" of a training positive.
    train = pd.read_parquet(DATA / "positives_huang2020.parquet")
    train_sky = SkyCoord(ra=train["RA"].values * u.deg, dec=train["DEC"].values * u.deg)
    _, sep_tr, _ = pub_sky.match_to_catalog_sky(train_sky)
    pub["in_training"] = sep_tr.to(u.arcsec).value < MATCH_RADIUS
    n_leak = int(pub["in_training"].sum())
    print(f"[leak] {n_leak}/{len(pub)} published candidates are training positives "
          f"(leaked); {len(pub)-n_leak} are leak-free")

    # Combined: best score across the two models.
    pub["score_combined"] = pub[[f"score_{k}" for k in MODELS]].max(axis=1)
    pub["in_parent_combined"] = pub[[f"in_parent_{k}" for k in MODELS]].any(axis=1)

    pub.to_csv(DATA / "recovery_dr8_matched.csv", index=False)

    # Summary: for each leak-bucket × grade × model, fraction recovered at each t.
    rows = []
    buckets = [("all", pub.index),
               ("honest", pub.index[~pub["in_training"]]),
               ("leaked", pub.index[pub["in_training"]])]
    for bname, bidx in buckets:
        b = pub.loc[bidx]
        for grade in ("A", "B", "C", "ALL"):
            g = b if grade == "ALL" else b[b["grade"] == grade]
            n = len(g)
            if n == 0:
                continue
            for kind in list(MODELS) + ["combined"]:
                inpar = g[f"in_parent_{kind}"]
                row = {"bucket": bname, "grade": grade, "model": kind, "n_published": n,
                       "n_in_parent": int(inpar.sum())}
                for t in THRESHOLDS:
                    npass = int((inpar & (g[f"score_{kind}"] >= t)).sum())
                    row[f"n_ge_{t}"] = npass
                    row[f"frac_ge_{t}"] = npass / n
                rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(DATA / "recovery_dr8_summary.csv", index=False)

    print("\n[honest recovery — leak-free shielded discoveries, by grade]")
    hs = summary[(summary.bucket == "honest")]
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(hs[["grade", "model", "n_published", "frac_ge_0.1", "frac_ge_0.5",
                  "frac_ge_0.9"]].to_string(index=False))

    # Figure 1: honest recovery by grade, the two models side by side at p>=0.1.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    grades = ["A", "B", "C", "ALL"]
    width = 0.25
    xs = np.arange(len(grades))
    for i, kind in enumerate(["l18", "shielded", "combined"]):
        fr = []
        for g in grades:
            sub = summary[(summary.bucket == "honest") & (summary.grade == g)
                          & (summary.model == kind)]
            fr.append(float(sub["frac_ge_0.1"].iloc[0]) if len(sub) else 0.0)
        ax.bar(xs + (i - 1) * width, fr, width, label=kind)
    ax.set_xticks(xs); ax.set_xticklabels(grades)
    ax.set_xlabel("Huang+2021 grade"); ax.set_ylabel("recovered fraction (p ≥ 0.1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("DR8 recovery of leak-free Huang+2021 candidates")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGDIR / "recovery_dr8_by_grade.png", dpi=140)
    plt.close(fig)

    # Figure 2: two-model complementarity — score_l18 vs score_shielded scatter.
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    hon = pub[~pub["in_training"]]
    cmap = {"A": "#2c7bb6", "B": "#fdae61", "C": "#999999"}
    for grade in ("C", "B", "A"):
        s = hon[hon["grade"] == grade]
        ax.scatter(s["score_l18"], s["score_shielded"], s=14, alpha=0.6,
                   color=cmap[grade], label=f"grade {grade}")
    ax.axhline(0.1, ls="--", c="k", lw=0.6); ax.axvline(0.1, ls="--", c="k", lw=0.6)
    ax.set_xlabel("L18 score"); ax.set_ylabel("shielded score")
    ax.set_title("Two-model scores (leak-free candidates)")
    ax.legend(); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout(); fig.savefig(FIGDIR / "recovery_dr8_two_model.png", dpi=140)
    plt.close(fig)

    print(f"\n[done] wrote recovery_dr8_{{matched,summary}}.csv + 2 figures")


if __name__ == "__main__":
    main()
