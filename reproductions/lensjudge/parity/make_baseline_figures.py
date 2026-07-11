#!/usr/bin/env python3
"""Figures for the human-baseline report (papers/human_baseline.tex).

  fig 1  baseline_pair_heatmap.{pdf,png}  joint distribution of the two graders'
                                          integer scores (unordered pairs, Paper II)
  fig 2  baseline_purity.{pdf,png}        per-grade confirmation purity with
                                          95% Jeffreys intervals

  python lensjudge/parity/make_baseline_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
FIGS = HERE.parent / "papers" / "figures"

INK = "#1a1a2e"
MUTED = "#5a5a6e"
BLUE = "#2f6fb0"


def fig_pair_heatmap() -> None:
    df = pd.read_csv(HERE / "data" / "vizier_huang2021_cand.csv")
    df = df[df["pair_ok"]]
    lo = (df["Score"] - df["delSc"] / 2).round().astype(int)
    hi = (df["Score"] + df["delSc"] / 2).round().astype(int)
    counts = np.zeros((4, 4))
    for a, b in zip(lo, hi):
        counts[a - 1, b - 1] += 1  # a <= b always (unordered pair, low first)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    # mask the lower triangle (identity lost => only unordered pairs exist) and the
    # cells excluded by the acceptance cut (mean score < 2 never enters the catalog)
    truncated = [(0, 0), (0, 1)]  # (1,1) and (1,2): mean < 2, structurally absent
    show = np.where(np.triu(np.ones_like(counts)) > 0, counts, np.nan)
    for i, j in truncated:
        show[i, j] = np.nan
    ax.imshow(show, cmap="Blues", origin="lower", vmin=0, vmax=counts.max())
    for i, j in truncated:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f2f2f5",
                                   edgecolor="none", zorder=1))
        ax.text(j, i, "cut", ha="center", va="center", fontsize=8.5, color=MUTED,
                style="italic", zorder=2)
    for i in range(4):
        for j in range(4):
            if j < i or (i, j) in truncated:
                continue
            v = int(counts[i, j])
            frac = counts[i, j] / counts.max()
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=11,
                    color="white" if frac > 0.55 else INK)
    labels = ["1\n(reject)", "2\n(C)", "3\n(B)", "4\n(A)"]
    ax.set_xticks(range(4), labels, fontsize=9)
    ax.set_yticks(range(4), labels, fontsize=9)
    ax.set_xlabel("higher of the two scores", fontsize=10, color=INK)
    ax.set_ylabel("lower of the two scores", fontsize=10, color=INK)
    ax.set_title("Two-grader score pairs (Huang+21, n=1,310)",
                 fontsize=10.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=MUTED)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"baseline_pair_heatmap.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {FIGS}/baseline_pair_heatmap.[pdf,png]")


def fig_purity() -> None:
    d = pd.read_csv(OUT / "parity_grade_purity.csv")
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.plot([r.purity_ci_lo, r.purity_ci_hi], [yi, yi], color=MUTED, lw=2,
                solid_capstyle="round", zorder=2)
        ax.plot(r.purity_decided, yi, "o", color=BLUE, ms=9, zorder=3)
        ax.text(1.045, yi, f"{r.confirmed}/{r.decided}", va="center", fontsize=9.5,
                color=MUTED)
    ax.set_yticks(y, [f"grade {g}" for g in d.grade], fontsize=10.5)
    ax.set_xlim(0.5, 1.1)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("confirmation purity among decided follow-ups (95% Jeffreys)",
                  fontsize=9.5, color=INK)
    ax.axvline(1.0, color="#d0d0d8", lw=0.8, zorder=1)
    ax.grid(axis="x", color="#ececf2", lw=0.7, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title("Confirmation rate by consensus grade", fontsize=10.5, color=INK)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"baseline_purity.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {FIGS}/baseline_purity.[pdf,png]")


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    fig_pair_heatmap()
    fig_purity()
