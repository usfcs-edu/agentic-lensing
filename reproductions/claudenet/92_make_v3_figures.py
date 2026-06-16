#!/usr/bin/env python3
"""92_make_v3_figures.py — v3 figures for the ClaudeNet report (companion to 91).
Robust to missing inputs (skips a figure if its source artifact is absent). All
numbers come from the tracked v3 artifacts under data/v3/ and lensjudge/outputs/.

  F1  model_progression.png  — lens-vs-mimic separation across model generations
        (orig/v2-lean/v3blend8/v3-head) on the mimic-FPR metric AND the INDEPENDENT
        Euclid grade-A-vs-C AUC (the honest D6 picture: clear gains on the broad
        mimic population, no significant edge at the hard frontier).
  F2  euclid_flip.png         — the DESI->Euclid p_lens flip on the real v3∩Euclid
        cross-matches (resolution lever).
  F3  euclid_recall.png       — Euclid grade-resolved recall: DESI detectability
        ceiling (~40% of Euclid A/B not in the parent) + v3 grade-selectivity
        (grade-A survivor recall 8.6x grade-C).

    /home2/benson/.venvs/claudenet/bin/python 92_make_v3_figures.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _clib as C

FIG = C.ROOT / "papers" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
V3 = C.DATA / "v3"
LJ = C.ROOT.parent / "lensjudge" / "outputs"

MODELS = ["orig (effnet_B, random-neg)", "v2lean (hard-mined, 5)", "v3blend8 (8)", "v3head (learned)"]
SHORT = ["orig\n(random-neg)", "v2-lean", "v3blend8", "v3-head"]


def jload(p):
    return json.load(open(p)) if p.exists() else None


def fig_progression():
    mp = jload(V3 / "model_progression.json")
    if not mp:
        return
    dr10 = [mp[m]["dr10_storfer"] for m in MODELS]
    seed = [mp[m]["seed_storfer"] for m in MODELS]
    auc = [mp[m]["euclid_AvsC_auc"] for m in MODELS]
    x = np.arange(len(MODELS))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: recovery@mimic-FPR(0.05), DR10-eval (broad pop) vs seed (hardest, selection-biased)
    w = 0.38
    axL.bar(x - w / 2, dr10, w, label="DR10-eval mimics (broad)", color="#2c7fb8")
    axL.bar(x + w / 2, seed, w, label="seed mimics (hardest; selection-biased)", color="#d95f0e", alpha=0.85)
    axL.set_xticks(x); axL.set_xticklabels(SHORT, fontsize=8)
    axL.set_ylabel("recovery @ matched mimic-FPR (0.05), Storfer"); axL.set_ylim(0, 1)
    axL.set_title("Lens-vs-mimic separation across generations")
    axL.legend(fontsize=7, loc="upper left"); axL.grid(axis="y", alpha=0.3)
    axL.annotate("seed scrambled by\nselection bias (D6)", xy=(0, seed[0]), xytext=(0.2, 0.80),
                 fontsize=7, ha="left", arrowprops=dict(arrowstyle="->", lw=0.7))
    # right: independent Euclid grade-A-vs-C AUC (all within ~1 SE -> hard frontier resolution-bounded)
    se = np.sqrt(np.mean(auc) * (1 - np.mean(auc)) / 66)  # ~SE, n_A=66
    axR.bar(x, auc, 0.6, yerr=se, color="#31a354", capsize=4)
    axR.axhspan(np.mean(auc) - se, np.mean(auc) + se, color="grey", alpha=0.15)
    axR.set_xticks(x); axR.set_xticklabels(SHORT, fontsize=8)
    axR.set_ylabel("Euclid grade-A vs C AUC (independent)"); axR.set_ylim(0.5, 0.75)
    axR.set_title(f"Independent hard-frontier test\n(all within ±1 SE ≈ {se:.02f}: resolution-bounded)")
    axR.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "model_progression.png", dpi=130); plt.close(fig)


def fig_flip():
    frames = []
    for f in ("d1_xmatch_vet.parquet", "d4_edf_vet.parquet", "d5_dr11edffs_vet.parquet"):
        p = LJ / f
        if p.exists():
            d = pd.read_parquet(p)
            if {"p_lens_tier1", "p_lens_tier2"} <= set(d.columns):
                frames.append(d[["p_lens_tier1", "p_lens_tier2", "grade_pred"]])
    if not frames:
        return
    d = pd.concat(frames, ignore_index=True).dropna(subset=["p_lens_tier1", "p_lens_tier2"])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cmap = {"A": "#1a9850", "B": "#91cf60", "C": "#fee08b", "D": "#d73027"}
    for g, c in cmap.items():
        s = d[d.grade_pred == g]
        if len(s):
            ax.scatter(s.p_lens_tier1, s.p_lens_tier2, s=26, c=c, edgecolor="k", lw=0.3,
                       label=f"{g} (n={len(s)})", alpha=0.85)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=0.8)
    ax.set_xlabel("DESI grz $p_{\\rm lens}$ (tier-1)"); ax.set_ylabel("Euclid 0.1$''$ $p_{\\rm lens}$ (tier-2)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"The resolution flip: DESI-ambiguous → Euclid-confirmed\n"
                 f"(n={len(d)} v3∩Euclid cross-matches; median {d.p_lens_tier1.median():.2f}→{d.p_lens_tier2.median():.2f})")
    ax.legend(fontsize=8, title="LensJudge grade @0.1$''$"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "euclid_flip.png", dpi=130); plt.close(fig)


def fig_euclid_recall():
    eo = jload(V3 / "euclid_overlap_summary.json")
    if not eo:
        return
    g = ["A", "B", "C"]
    par = [eo["by_grade"][k]["in_parent_frac"] for k in g]
    sur = [eo["by_grade"][k]["in_survivors_frac"] for k in g]
    x = np.arange(3)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.0))
    axL.bar(x, par, 0.6, color=["#1a9850", "#91cf60", "#fee08b"])
    axL.axhline(1.0, color="k", lw=0.6, ls=":")
    axL.set_xticks(x); axL.set_xticklabels([f"Euclid {k}" for k in g]); axL.set_ylim(0, 1)
    axL.set_ylabel("fraction in the DESI parent")
    axL.set_title("DESI detectability ceiling\n(~40% of Euclid A/B are not DESI-detectable)")
    for i, v in enumerate(par):
        axL.text(i, v + 0.02, f"{v*100:.0f}%", ha="center", fontsize=8)
    axL.grid(axis="y", alpha=0.3)
    axR.bar(x, sur, 0.6, color=["#1a9850", "#91cf60", "#fee08b"])
    axR.set_xticks(x); axR.set_xticklabels([f"Euclid {k}" for k in g])
    axR.set_ylabel("fraction recovered into v3 survivors")
    enr = eo["AB"]["A_over_C_enrichment"]
    axR.set_title(f"v3 grade-selectivity\n(grade-A recall {enr:.1f}× grade-C)")
    for i, v in enumerate(sur):
        axR.text(i, v + 0.001, f"{v*100:.1f}%", ha="center", fontsize=8)
    axR.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "euclid_recall.png", dpi=130); plt.close(fig)


def main():
    for f in (fig_progression, fig_flip, fig_euclid_recall):
        try:
            f()
        except Exception as e:
            print(f"[92] {f.__name__} skipped: {e}")
    print(f"[92] v3 figures in {FIG}:",
          [p.name for p in sorted(FIG.glob('*.png')) if p.name in
           ("model_progression.png", "euclid_flip.png", "euclid_recall.png")])


if __name__ == "__main__":
    raise SystemExit(main())
