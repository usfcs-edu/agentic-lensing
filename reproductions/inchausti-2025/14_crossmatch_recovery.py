#!/usr/bin/env python3
"""
14_crossmatch_recovery.py — Phase-5 recovery + leakage analysis.

From the direct candidate scores (13_), report how well our reproduced models
recover the published Storfer-2024 (1,895) and Inchausti-2025 (811) candidates,
broken out by grade x model x threshold, separating leaked from honest
candidates (a candidate within 5" of one of our training positives is LEAKED —
the same caveat carried from Phase 4).

For Inchausti we additionally compare our reproduced per-model probabilities to
the published ones (the catalog ships ResNet/EfficientNet/meta scores).

Inputs:
  data/candidate_scores_storfer.csv, data/candidate_scores_inchausti.csv  (13_)
  data/positives_huang2020.parquet                                        (leak tag)
Outputs:
  data/recovery_<catalog>_summary.csv
  papers/figures/recovery_<catalog>_by_grade.png
  papers/figures/inchausti_pub_vs_ours.png
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
MODELS = {"resnet": "our_p_resnet", "effnet": "our_p_effnet",
          "meta": "our_p_meta", "avg": "our_p_avg"}


def tag_leak(cat: pd.DataFrame) -> pd.DataFrame:
    train = pd.read_parquet(DATA / "positives_huang2020.parquet")
    tsky = SkyCoord(ra=train["RA"].values * u.deg, dec=train["DEC"].values * u.deg)
    csky = SkyCoord(ra=cat["RA"].values * u.deg, dec=cat["DEC"].values * u.deg)
    _, sep, _ = csky.match_to_catalog_sky(tsky)
    cat = cat.copy()
    cat["in_training"] = sep.to(u.arcsec).value < MATCH_RADIUS
    return cat


def recovery_table(cat: pd.DataFrame) -> pd.DataFrame:
    cat = cat[cat["cutout_ok"]].copy()
    rows = []
    buckets = [("all", cat.index),
               ("honest", cat.index[~cat["in_training"]]),
               ("leaked", cat.index[cat["in_training"]])]
    for bname, bidx in buckets:
        b = cat.loc[bidx]
        for grade in ("A", "B", "C", "ALL"):
            g = b if grade == "ALL" else b[b["grade"] == grade]
            n = len(g)
            if n == 0:
                continue
            for mkey, col in MODELS.items():
                row = {"bucket": bname, "grade": grade, "model": mkey, "n": n}
                for t in THRESHOLDS:
                    row[f"frac_ge_{t}"] = float((g[col] >= t).mean())
                rows.append(row)
    return pd.DataFrame(rows)


def run(key: str) -> None:
    f = DATA / f"candidate_scores_{key}.csv"
    if not f.exists():
        print(f"[{key}] missing {f.name} — run 13 first; skipping")
        return
    cat = tag_leak(pd.read_csv(f))
    n_leak = int(cat["in_training"].sum())
    print(f"\n########## {key}: {len(cat)} published; {n_leak} leaked, "
          f"{len(cat)-n_leak} leak-free ##########")
    summ = recovery_table(cat)
    summ.to_csv(DATA / f"recovery_{key}_summary.csv", index=False)

    print(f"[{key}] honest (leak-free) recovery by grade × model:")
    hs = summ[summ.bucket == "honest"]
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(hs[["grade", "model", "n", "frac_ge_0.1", "frac_ge_0.5", "frac_ge_0.9"]]
              .to_string(index=False))

    # Figure: honest recovery by grade @p>=0.5, the four scorers.
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    grades = ["A", "B", "C", "ALL"]
    xs = np.arange(len(grades)); w = 0.2
    for i, mkey in enumerate(MODELS):
        fr = [float(hs[(hs.grade == g) & (hs.model == mkey)]["frac_ge_0.5"].iloc[0])
              if len(hs[(hs.grade == g) & (hs.model == mkey)]) else 0.0 for g in grades]
        ax.bar(xs + (i - 1.5) * w, fr, w, label=mkey)
    ax.set_xticks(xs); ax.set_xticklabels(grades)
    ax.set_xlabel(f"{key} grade"); ax.set_ylabel("recovered fraction (p ≥ 0.5)")
    ax.set_ylim(0, 1.05); ax.set_title(f"Leak-free recovery of {key} candidates")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGDIR / f"recovery_{key}_by_grade.png", dpi=140)
    plt.close(fig)

    # Inchausti: our reproduced meta vs published meta.
    if key == "inchausti" and {"p_resnet", "p_effnet", "p_meta"} <= set(cat.columns):
        m = cat["cutout_ok"] & cat["our_p_meta"].notna() & cat["p_meta"].notna()
        d = cat[m]
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (pub, our, name) in zip(axs, [
                ("p_resnet", "our_p_resnet", "ResNet"),
                ("p_effnet", "our_p_effnet", "EfficientNet"),
                ("p_meta", "our_p_meta", "Meta-learner")]):
            ax.scatter(d[pub], d[our], s=10, alpha=0.4, color="#2c7bb6")
            ax.plot([0, 1], [0, 1], "k--", lw=0.7)
            corr = np.corrcoef(d[pub], d[our])[0, 1]
            ax.set_xlabel(f"published {name} prob"); ax.set_ylabel(f"our {name} prob")
            ax.set_title(f"{name}  (r={corr:.2f})"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
            print(f"[inchausti] corr(published {name}, ours) = {corr:.3f}")
        fig.tight_layout(); fig.savefig(FIGDIR / "inchausti_pub_vs_ours.png", dpi=140)
        plt.close(fig)

    # Headline numbers.
    for bucket in ("all", "honest"):
        a = summ[(summ.bucket == bucket) & (summ.grade == "ALL") & (summ.model == "meta")]
        if len(a):
            print(f"[{key}] {bucket} ALL-grade meta recovery: "
                  f"p≥0.1 {a['frac_ge_0.1'].iloc[0]:.3f}  p≥0.5 {a['frac_ge_0.5'].iloc[0]:.3f}  "
                  f"p≥0.9 {a['frac_ge_0.9'].iloc[0]:.3f}")


def main() -> None:
    for key in ("storfer", "inchausti"):
        run(key)
    print("\n[done] recovery summaries + figures written")


if __name__ == "__main__":
    main()
