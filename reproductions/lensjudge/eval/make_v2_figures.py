#!/usr/bin/env python3
"""make_v2_figures.py -- the three LensJudge v2 figures for papers/ (Part II).

All numbers come from tracked v2 artifacts under outputs/; the script reads them
directly and is robust to a missing input (it skips that figure with a warning).

  F1  v2_benchmark.png        -- v1-lean -> v2 on the two benchmarks: detection
        (lens-vs-random, Bench-A) barely moves, lens-vs-MIMIC (Bench-B) jumps
        0.52 -> 0.71 (+0.196). The v2 headline.
  F2  v2_agency_ablation.png  -- the 2x2 rubric x loop (+thinking) ablation: the
        agentic loop adds nothing to detection but is the ONLY arm that reaches
        the mimic-rejection gain (which therefore lives in the loop, not the
        rubric text or a thinking scratchpad).
  F3  v2_escalation_flip.png  -- the DESI(tier-1) -> Euclid 0.1" (tier-2) p_lens
        flip on the 140 escalated cross-matches (D1 + D4 + D5 + south-val): nearly
        every object is DESI-ambiguous yet Euclid-confirmed.

Run:  python3 eval/make_v2_figures.py        (from the lensjudge dir)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent          # reproductions/lensjudge
OUT = HERE / "outputs"
FIG = HERE / "papers" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# colour palette (kept consistent across the three figures)
C_V1 = "#9aa0a6"     # v1 / baseline grey
C_V2 = "#0F4D92"     # v2 / loop -- the report's link-blue
C_DET = "#5b8c5a"    # detection green
C_MIM = "#b5651d"    # mimic-rejection orange
CHANCE = "#c0392b"   # chance line red

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})


def jload(p: Path):
    return json.load(open(p)) if p.exists() else None


# agency_ablation.json carries every arm (incl. the lean/v2 rows) -> single source
# for F1 and F2. Arm keys as written by run_agency_ablation.py:
ARMS = jload(OUT / "agency_ablation.json")
A_LEAN = "lean  (v1 rubric, LOOP)"
A_DIRECT_V1 = "direct(v1 rubric, no loop)"
A_V2 = "v2    (v2 rubric, LOOP)"
A_DIRECT_V2T = "direct(v2 rubric, no loop)"
A_DIRECT_V2I = "direct(v2-inline,  no loop)"
A_DIRECT_V2THINK = "direct(v2-inline + THINKING, no loop)"


# --------------------------------------------------------------------------- F1
def fig_benchmark():
    if not ARMS or "arms" not in ARMS:
        print("[F1] skip: agency_ablation.json missing"); return
    a = ARMS["arms"]
    v1, v2 = a[A_LEAN], a[A_V2]
    groups = ["Detection\n(lens-vs-random)", "Mimic rejection\n(lens-vs-mimic)"]
    v1v = [v1["A_auc"], v1["B_auc"]]
    v2v = [v2["A_auc"], v2["B_auc"]]
    x = np.arange(2); w = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    b1 = ax.bar(x - w / 2, v1v, w, label="v1-lean (v1 rubric)", color=C_V1)
    b2 = ax.bar(x + w / 2, v2v, w, label="v2 (rubric$_{v2}$ + escalate)", color=C_V2)
    ax.axhline(0.5, ls="--", lw=1, color=CHANCE, zorder=0)
    ax.text(1.46, 0.508, "chance", color=CHANCE, fontsize=8, ha="right")
    for rects, vals in ((b1, v1v), (b2, v2v)):
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8.5)
    # delta annotations (placed well above the value labels to avoid collision)
    for i, (lo, hi) in enumerate(zip(v1v, v2v)):
        d = hi - lo
        ax.annotate(f"+{d:.3f}", xy=(x[i] + w / 2, hi + 0.093),
                    ha="center", fontsize=9.5, fontweight="bold",
                    color=C_V2 if i == 1 else "#666")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("binary ROC-AUC"); ax.set_ylim(0, 0.85)
    ax.set_title("LensJudge v2: the rubric$_{v2}$ + loop lift lens-vs-mimic\n"
                 "(detection barely moves; mimic rejection jumps +0.196)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "v2_benchmark.png"); plt.close(fig)
    print(f"[F1] wrote {FIG/'v2_benchmark.png'}  A {v1v[0]:.3f}->{v2v[0]:.3f}  "
          f"B {v1v[1]:.3f}->{v2v[1]:.3f}")


# --------------------------------------------------------------------------- F2
def fig_ablation():
    if not ARMS or "arms" not in ARMS:
        print("[F2] skip: agency_ablation.json missing"); return
    a = ARMS["arms"]
    order = [A_LEAN, A_V2, A_DIRECT_V1, A_DIRECT_V2T, A_DIRECT_V2I, A_DIRECT_V2THINK]
    labels = ["lean\n(v1, loop)", "v2\n(v2, loop)", "direct\n(v1, no-loop)",
              "direct\n(v2-tool)", "direct\n(v2-inline)", "direct\n(v2i+think)"]
    loop = [a[k]["loop"] for k in order]
    Aauc = [a[k]["A_auc"] for k in order]
    Bauc = [a[k]["B_auc"] for k in order]
    cost = [a[k]["mean_cost"] for k in order]
    turns = [a[k]["mean_turns"] for k in order]
    x = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharey=True)
    for ax, vals, title, col in (
            (axes[0], Aauc, "Detection (lens-vs-random)", C_DET),
            (axes[1], Bauc, "Mimic rejection (lens-vs-mimic)", C_MIM)):
        bars = ax.bar(x, vals, 0.66,
                      color=[col if lp else "#d9d9d9" for lp in loop],
                      edgecolor=["#222" if lp else "#999" for lp in loop],
                      hatch=["" if lp else "//" for lp in loop], linewidth=1)
        ax.axhline(0.5, ls="--", lw=1, color=CHANCE, zorder=0)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title); ax.set_ylim(0, 0.82)
    axes[0].set_ylabel("binary ROC-AUC")
    # cost / turns annotation under the detection panel x-axis
    for xi, (c, t) in enumerate(zip(cost, turns)):
        axes[0].annotate(f"${c:.3f}\n{t:.0f}t" if t == int(t) else f"${c:.3f}\n{t:.1f}t",
                         xy=(xi, -0.235), xycoords=("data", "axes fraction"),
                         ha="center", va="top", fontsize=6.8, color="#444")
    # legend for loop vs no-loop
    from matplotlib.patches import Patch
    leg = [Patch(facecolor="#888", edgecolor="#222", label="agentic loop"),
           Patch(facecolor="#d9d9d9", edgecolor="#999", hatch="//", label="no loop (direct)")]
    axes[1].legend(handles=leg, loc="upper right", frameon=False, fontsize=8.5)
    fig.suptitle("Agency ablation: the loop adds nothing to detection, "
                 "but only the loop reaches the 0.71 mimic-rejection gain",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93)); fig.savefig(FIG / "v2_agency_ablation.png")
    plt.close(fig)
    print(f"[F2] wrote {FIG/'v2_agency_ablation.png'}  arms={len(order)}")


# --------------------------------------------------------------------------- F3
ESC_RUNS = [
    ("d4_edf_vet.parquet", "DR10 EDF-F/S (D4)", "#0F4D92", "o"),
    ("d5_dr11edffs_vet.parquet", "DR11 EDF-F/S (D5)", "#b5651d", "s"),
    ("d1_xmatch_vet.parquet", "DR10 Q1 overlap (D1)", "#5b8c5a", "^"),
    ("euclid_south_val.parquet", "south live-val (B1)", "#7b2cbf", "D"),
]


def fig_escalation():
    rows = []
    for fn, lbl, col, mk in ESC_RUNS:
        p = OUT / fn
        if not p.exists():
            print(f"[F3] note: {fn} absent"); continue
        d = pd.read_parquet(p)
        if "escalated" in d:
            d = d[d["escalated"] == True]                     # noqa: E712
        d = d.dropna(subset=["p_lens_tier1", "p_lens_tier2"])
        if len(d):
            rows.append((d, lbl, col, mk))
    if not rows:
        print("[F3] skip: no escalation parquets"); return

    allt1 = np.concatenate([d["p_lens_tier1"].to_numpy() for d, *_ in rows])
    allt2 = np.concatenate([d["p_lens_tier2"].to_numpy() for d, *_ in rows])
    n = len(allt1); m1, m2 = np.median(allt1), np.median(allt2)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    # quadrant shading: DESI-ambiguous (x<0.5) AND Euclid-confirmed (y>0.5)
    ax.axhspan(0.5, 1.02, xmin=0, xmax=0.5 / 1.02, color="#eaf3ea", zorder=0)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888", zorder=1)
    ax.axhline(0.5, ls=":", lw=0.8, color="#bbb"); ax.axvline(0.5, ls=":", lw=0.8, color="#bbb")
    for d, lbl, col, mk in rows:
        ax.scatter(d["p_lens_tier1"], d["p_lens_tier2"], s=34, marker=mk,
                   facecolor=col, edgecolor="white", linewidth=0.5, alpha=0.85,
                   label=f"{lbl}  (n={len(d)})", zorder=3)
    # median crosshair + flip annotation
    ax.scatter([m1], [m2], s=240, marker="*", color="#d4af37",
               edgecolor="#222", linewidth=0.8, zorder=5)
    ax.annotate(f"median flip\n{m1:.2f} $\\rightarrow$ {m2:.2f}",
                xy=(m1, m2), xytext=(0.30, 0.40), fontsize=9.5, fontweight="bold",
                ha="center", arrowprops=dict(arrowstyle="->", color="#222", lw=1.1))
    ax.text(0.25, 0.965, "DESI-ambiguous, Euclid-confirmed", fontsize=8.5,
            color="#2e7d32", ha="center", style="italic")
    ax.set_xlabel(r"tier-1  $p_{\mathrm{lens}}$  (DESI $grz$, $1.3''$)")
    ax.set_ylabel(r"tier-2  $p_{\mathrm{lens}}$  (Euclid VIS, $0.1''$)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_title(f"Two-tier escalation flip on {n} escalated cross-matches\n"
                 r"(DESI tier-1 $1.3''$ $\rightarrow$ Euclid tier-2 $0.1''$)")
    ax.legend(loc="lower right", frameon=False, fontsize=8.2)
    fig.tight_layout(); fig.savefig(FIG / "v2_escalation_flip.png"); plt.close(fig)
    print(f"[F3] wrote {FIG/'v2_escalation_flip.png'}  n={n}  median {m1:.3f}->{m2:.3f}")


if __name__ == "__main__":
    fig_benchmark()
    fig_ablation()
    fig_escalation()
