#!/usr/bin/env python3
"""91_make_figures.py — key figures for the ClaudeNet tech report. Robust to
missing inputs (skips a panel if its result file is absent).

    /home2/benson/.venvs/claudenet/bin/python 91_make_figures.py
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
D = C.DATA


def jload(n):
    p = D / n
    return json.load(open(p)) if p.exists() else None


def fig_diversity():
    div = jload("diversity.json")
    if not div:
        return
    names = div["members"]; M = np.array(div["spearman"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] < 0.6 else "black")
    ax.set_title(f"Member score correlation (Spearman)\nmean off-diag {div['spearman_offdiag_mean']:.2f} "
                 f"(repo meta bases ~1.0)", fontsize=9)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(FIG / "diversity_heatmap.png", dpi=130); plt.close(fig)


def fig_flagship():
    p = D / "flagship_operating_point.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    metrics = [("storfer_1", "Storfer@1%"), ("storfer_01", "Storfer@0.1%"),
               ("inchausti_1", "Inch@1%"), ("inchausti_01", "Inch@0.1%")]
    pick = {"baseline:meta": "published meta", "combiner:average": "ClaudeNet avg",
            "combiner:rf": "ClaudeNet rf"}
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(metrics)); w = 0.25
    for i, (scorer, lab) in enumerate(pick.items()):
        row = df[df.scorer == scorer]
        if not len(row):
            continue
        vals = [row[m].iloc[0] for m, _ in metrics]
        ax.bar(x + (i - 1) * w, vals, w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in metrics])
    ax.set_ylim(0.7, 1.0); ax.set_ylabel("recovery @ matched FPR")
    ax.set_title("Phase 1: engineered-diversity ensemble vs published meta-learner")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "flagship_operating_point.png", dpi=130); plt.close(fig)


def fig_conformal():
    conf = jload("conformal_selection.json")
    if not conf:
        return
    rows = conf["average"]
    a = [r["alpha"] for r in rows]; emp = [r["empirical_fdr"] for r in rows]
    comp = [r["completeness"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 0.5], [0, 0.5], "k--", alpha=0.5, label="nominal = empirical")
    ax.plot(a, emp, "o-", label="empirical FDR")
    ax.plot(a, comp, "s-", color="green", label="completeness")
    ax.set_xlabel("target FDR (alpha)"); ax.set_ylabel("rate")
    ax.set_title("Phase 4: conformal selection — certified FDR control")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "conformal_fdr.png", dpi=130); plt.close(fig)


def fig_selective():
    uq = jload("uncertainty.json")
    if not uq or "selective" not in uq:
        return
    sel = uq["selective"]
    cov = sorted(float(k) for k in sel)
    err = [sel[str(c)] if str(c) in sel else sel[c] for c in cov]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(cov, err, "o-")
    ax.set_xlabel("coverage (fraction retained, conf-first)"); ax.set_ylabel("error rate")
    ax.set_title("Phase 6: selective prediction\n(abstain on high ensemble disagreement)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "selective_prediction.png", dpi=130); plt.close(fig)


def fig_label_efficiency():
    p = D / "label_efficiency.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    s = df[df.cat == "storfer"]
    fig, ax = plt.subplots(figsize=(5, 4))
    for method, mk in (("aion_probe", "o-"), ("shielded", "s-")):
        m = s[s.method == method].sort_values("n_pos")
        if len(m):
            ax.plot(m["n_pos"], m["rec_1"], mk, label=method)
    ax.set_xlabel("# labeled lens positives"); ax.set_ylabel("Storfer recovery @1%FPR")
    ax.set_xscale("log"); ax.set_title("Phase 3: label efficiency")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "label_efficiency.png", dpi=130); plt.close(fig)


def main():
    for f in (fig_diversity, fig_flagship, fig_conformal, fig_selective, fig_label_efficiency):
        try:
            f()
        except Exception as e:
            print(f"[91] {f.__name__} skipped: {e}")
    print(f"[91] figures in {FIG}:", [p.name for p in sorted(FIG.glob('*.png'))])


if __name__ == "__main__":
    raise SystemExit(main())
