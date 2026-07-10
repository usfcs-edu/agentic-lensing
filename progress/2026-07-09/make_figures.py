#!/usr/bin/env python3
"""
Generate the seven new figures for the ClaudeNet v4 / DR11-south progress deck.

Every datum below is a literal transcribed from a tracked source; the per-figure
docstring names it. The v4 result artifacts (resweep_v4_summary.json,
dr11_finetune_gate.json, survivors_dr11s_v4.parquet) live on Perlmutter scratch
and are NOT in this repo, so v4 numbers are transcribed from the LaTeX reports:
    reproductions/claudenet/papers/v4_section.tex
    reproductions/dr11-campaign-v4/papers/main.tex

DENOMINATOR DISCIPLINE. DR11 grade-A "recall" appears with three different
denominators across the sources. The authoritative three-config comparison is
dr11-campaign-v4/papers/main.tex Table 2 (position-crossmatch, training-excluded).
The 54%->75% "same-budget" result from v4_section.tex uses the in-parent
denominator and is never plotted on the same axis. Axis labels name the
denominator.

Run:
    python3 progress/2026-07-09/make_figures.py
Output:
    progress/2026-07-09/figures/*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parent / "figures"

# Deck palette (tools/spectrumfm/build_deck.py) plus a CVD-checked categorical set.
INK = "#222222"
NAVY = "#182C5B"
MUTED = "#666666"
GRID = "#E6E6E1"
SPINE = "#C3C2B7"
GOOD = "#1E7A3C"   # reserved: gains, DONE
CRIT = "#C63A3A"   # reserved: limits, PENDING, collapse

S1, S2, S3, S4 = "#2A78D6", "#EB6834", "#1BAF7A", "#4A3AA7"
RAMP = ["#9EC5F4", "#2A78D6", "#184F95"]   # ordinal: worse -> better


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": SPINE,
        "axes.labelcolor": "#333333",
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "grid.linestyle": "-",
    })


def _footer(fig, lines) -> None:
    """Reserve space under the axes, then write footnote lines into it.

    constrained_layout does not account for fig.text added afterwards, so the
    layout rect must be shrunk explicitly or the footer lands on the tick labels.
    """
    if isinstance(lines, str):
        lines = [lines]
    pad = 0.030 + 0.030 * len(lines)
    fig.get_layout_engine().set(rect=(0.0, pad, 1.0, 1.0))
    for i, line in enumerate(reversed(lines)):
        fig.text(0.012, 0.011 + i * 0.029, line, fontsize=10.5, color=MUTED)


def _frame(fig, title, footer=None, title_lines: int = 1) -> None:
    """Draw a figure title and footer into explicitly reserved bands.

    fig.suptitle is not used: once the layout rect is shrunk for the footer, the
    engine stops reserving room for the suptitle and it lands on the axes titles.
    Reserving both bands by hand is deterministic.
    """
    footer = footer or []
    if isinstance(footer, str):
        footer = [footer]
    bottom = 0.030 + 0.030 * len(footer) if footer else 0.020
    # The engine's rect bounds the axes box, but per-axes titles are drawn ABOVE
    # that bound. So the reserve must cover the figure title *and* a per-axes
    # title, or the two collide. The 0.055 term is that per-axes title.
    top = 1.0 - (0.085 * title_lines + 0.055)
    fig.get_layout_engine().set(rect=(0.0, bottom, 1.0, top))
    fig.text(0.5, 0.985, title, ha="center", va="top",
             fontsize=15, color=NAVY, fontweight="bold")
    for i, line in enumerate(reversed(footer)):
        fig.text(0.012, 0.011 + i * 0.029, line, fontsize=10.5, color=MUTED)


def _save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    # No bbox_inches="tight": it changes the physical inches and breaks the
    # >=11pt-at-1:1 legibility guarantee when embedded at figsize width.
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path.relative_to(Path(__file__).resolve().parents[2])}")
    return path


# --------------------------------------------------------------------------
# FIG 1 -- the lineage and its sweeps
# Sources: huang-2020/, huang-2021/, inchausti-2025/ READMEs; claudenet/README.md;
#          dr11-campaign/papers/main.tex; dr11-campaign-v4/papers/main.tex
# --------------------------------------------------------------------------
def fig1_lineage_sweeps() -> None:
    labels = [
        "Huang+20\nDR7",
        "Huang+21\nDR8",
        "Storfer+24\nDR9",
        "Inchausti+25\nDR10",
        "ClaudeNet v2\nDR9",
        "ClaudeNet v3\nDR11-S",
        "ClaudeNet v4\nDR11-S",
    ]
    scanned = [6_242_507, 17_290_814, 45_260_000, 43_000_000,
               17_290_814, 53_809_040, 53_809_040]
    # Yield is *not* one quantity -- the unit changes with the vetting regime.
    yields = [342, 1312, 1895, 811, 1449, 24, None]
    regime = ["by-eye", "by-eye", "by-eye", "by-eye",
              "conformal", "agentic+HSC", "pending"]
    regime_color = {"by-eye": MUTED, "conformal": S1,
                    "agentic+HSC": S4, "pending": CRIT}

    x = np.arange(len(labels))
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(12.0, 6.4), sharex=True,
        height_ratios=[1.15, 1.0], constrained_layout=True,
    )

    axA.bar(x, scanned, width=0.55, color=NAVY)
    axA.set_yscale("log")
    axA.set_ylabel("galaxies scanned")
    axA.set_ylim(1e6, 4e8)
    axA.grid(axis="y", zorder=0)
    axA.set_axisbelow(True)
    axA.annotate(
        "", xy=(6, 1.2e8), xytext=(0, 1.2e8),
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4),
    )
    axA.text(3.0, 1.45e8, "8.6x more galaxies scanned  (6.2M  ->  53.8M)",
             ha="center", va="bottom", fontsize=11.5, color=MUTED)

    for xi, yi, r in zip(x, yields, regime):
        if yi is None:
            axB.scatter([xi], [40], s=140, facecolors="none",
                        edgecolors=CRIT, linewidths=2.0, zorder=3)
            axB.text(xi, 62, "PENDING", ha="center", fontsize=10.5,
                     color=CRIT, fontweight="bold")
            continue
        axB.bar([xi], [yi], width=0.55, color=regime_color[r], zorder=2)
        axB.text(xi, yi * 1.18, f"{yi:,}", ha="center",
                 fontsize=10.5, color=INK)

    axB.set_yscale("log")
    axB.set_ylabel("candidates reported")
    axB.set_ylim(10, 1.2e5)
    axB.grid(axis="y", zorder=0)
    axB.set_axisbelow(True)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=11)

    # Vetting-regime key, as a legend rather than chips painted over the bars.
    handles = [
        Rectangle((0, 0), 1, 1, color=MUTED, label="by-eye grading"),
        Rectangle((0, 0), 1, 1, color=S1, label="conformal FDR $\\leq$ 0.05"),
        Rectangle((0, 0), 1, 1, color=S4, label="agentic LensJudge + HSC tier-2"),
    ]
    axB.legend(handles=handles, loc="upper left", ncols=1, fontsize=10,
               bbox_to_anchor=(0.0, 1.02))

    axB.text(6.32, 7.5e4,
             "Yield does not track sweep size --\nand the unit changes with the vetting bar.\n"
             "v3's 24 HSC tier-2 A/B contain\n3 genuinely new systems.",
             ha="right", va="top", fontsize=10.5, color=INK,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=SPINE, lw=0.8))

    _frame(fig,
           "Eight years of sweeps: scale grew, yield did not -- and the vetting bar rose",
           footer=[
               "Model size is not the lever: a 194k-param shielded ResNet matches a 20.5M-param "
               "EfficientNetV2-S within 0.003 AUC (Inchausti 2025).",
               "Counts are not commensurable across regimes -- a by-eye 'candidate' and an "
               "HSC tier-2-vetted grade-A are different objects.",
           ])
    _save(fig, "fig1_lineage_sweeps.png")


# --------------------------------------------------------------------------
# FIG 2 -- recovery @ matched FPR, NegEval-1M
# Source: claudenet/README.md "The v2-lean headline"; papers/main.tex tab:v2lean
# --------------------------------------------------------------------------
def fig2_recovery_matched_fpr() -> None:
    fpr = np.array([1e-2, 1e-3, 1e-4])
    storfer = {"meta": [0.854, 0.679, 0.513],
               "v1": [0.903, 0.754, 0.394],
               "v2lean": [0.963, 0.895, 0.734]}
    inchausti = {"meta": [0.932, 0.769, 0.607],
                 "v1": [0.968, 0.891, 0.614],
                 "v2lean": [0.996, 0.961, 0.871]}
    # Paired-bootstrap 95% CI on the v2lean - v1 difference (10,000 reps).
    d_str = (0.340, 0.306, 0.381)
    d_inc = (0.256, 0.217, 0.301)

    series = [("published meta", "meta", S2),
              ("ClaudeNet v1 flagship", "v1", S1),
              ("ClaudeNet v2-lean", "v2lean", NAVY)]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.0),
                                   sharey=True, constrained_layout=True)

    for ax, data, title, delta in [
        (axL, storfer, "Storfer catalog", d_str),
        (axR, inchausti, "Inchausti catalog", d_inc),
    ]:
        ax.axvspan(6e-5, 1.7e-4, color=GRID, alpha=0.7, zorder=0)
        for label, key, color in series:
            ax.plot(fpr, data[key], "-o", color=color, lw=2.0, ms=7,
                    mec="white", mew=1.6, label=label, zorder=3)
        ax.set_xscale("log")
        ax.invert_xaxis()               # threshold tightens left -> right
        ax.set_xticks([1e-2, 1e-3, 1e-4])
        ax.set_xticklabels(["1%", "0.1%", "0.01%"])
        ax.set_xlabel("matched false-positive rate  (stricter $\\rightarrow$)")
        ax.set_ylim(0.27, 1.04)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title, color=NAVY)

        # Bracket the paired v2lean - v1 delta at the operating point.
        d, lo, hi = delta
        y1, y2 = data["v1"][2], data["v2lean"][2]
        ax.annotate("", xy=(8.0e-5, y2), xytext=(8.0e-5, y1),
                    arrowprops=dict(arrowstyle="<->", color=GOOD, lw=1.6))
        ax.text(7.0e-5, (y1 + y2) / 2,
                f"v2-lean - v1\n+{d:.3f}\n({lo:+.3f}, {hi:+.3f})",
                fontsize=10, color=GOOD, va="center", ha="left")

    axL.set_ylabel("recovery (held-out TPR)")
    # Legend lives in the right panel: the left panel's empty quadrant is needed
    # for the collapse annotation.
    axR.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02))

    # v1's collapse on Storfer at the operating point.
    axL.scatter([1e-4], [0.394], s=190, facecolors="none",
                edgecolors=CRIT, linewidths=2.2, zorder=4)
    axL.annotate(
        "v1's naive average collapses below\nthe published meta (0.394 < 0.513):\n"
        "one degraded member's negative\ntail poisons the average",
        xy=(1.12e-4, 0.394), xytext=(6.8e-3, 0.52),
        arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.3,
                        connectionstyle="arc3,rad=-0.15"),
        fontsize=10.5, color=CRIT, va="top",
    )
    axL.text(1.02e-4, 1.03, "real-sweep\noperating point", fontsize=9.5,
             color=MUTED, ha="center", va="top")

    _frame(fig,
           "The gain grows as the threshold tightens   "
           "(NegEval-1M: 1,000,000 held-out negatives)",
           footer="Brackets show the paired-bootstrap 95% CI on the v2-lean minus v1 "
                  "difference (10,000 reps). Per-point CIs are not available.")
    _save(fig, "fig2_recovery_matched_fpr.png")


# --------------------------------------------------------------------------
# FIG 3 -- the v4 fine-tune gate
# Source: claudenet/papers/v4_section.tex Table tab:v4gate
#         (== dr11-campaign-v4/papers/main.tex Table 1)
# --------------------------------------------------------------------------
def fig3_v4_finetune_gate() -> None:
    rows = [
        ("Inchausti grade-A", 0.724, 0.816, False),
        ("Inchausti grade-B", 0.515, 0.812, False),
        ("Storfer grade-A\n(lrg+companion)", 0.544, 0.796, True),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)

    for i, (name, v3, v4, hard) in enumerate(rows):
        y = len(rows) - 1 - i
        color = CRIT if hard else NAVY
        ax.plot([v3, v4], [y, y], lw=3.0, color=color, alpha=0.55, zorder=2,
                solid_capstyle="round")
        ax.scatter([v3], [y], s=130, color=RAMP[0], ec="white", lw=1.4, zorder=3)
        ax.scatter([v4], [y], s=150, color=color, ec="white", lw=1.4, zorder=3)
        ax.text(v4 + 0.012, y, f"+{v4 - v3:.2f}", va="center",
                fontsize=12, color=CRIT if hard else GOOD, fontweight="bold")
        ax.text(v3 - 0.012, y, f"{v3:.3f}", va="center", ha="right",
                fontsize=10.5, color=MUTED)

    ax.scatter([], [], s=130, color=RAMP[0], ec="white", label="v3 baseline (five-lean)")
    ax.scatter([], [], s=150, color=NAVY, ec="white", label="v4 fine-tuned")
    # Inside the axes: a legend anchored above them is clipped once the title
    # band is reserved.
    ax.legend(loc="upper left", ncols=2, fontsize=10.5)

    ax.annotate(
        "the hard residual v3 could not crack",
        xy=(0.796, 0.0), xytext=(0.60, -0.62),
        arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.3),
        fontsize=11, color=CRIT,
    )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=11.5)
    ax.set_xlim(0.44, 0.92)
    ax.set_ylim(-0.78, 2.80)
    ax.set_xlabel("held-out recall @ matched-FPR 95k-survivor operating point")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    _frame(fig,
           "v4 fine-tune gate: every held-out set improves",
           footer=[
               "Storfer/Inchausti held out of training as positives and negatives.",
               "Ranking AUC does not regress: Inchausti grade-A 0.996 -> 0.998.",
           ])
    _save(fig, "fig3_v4_finetune_gate.png")


# --------------------------------------------------------------------------
# FIG 4 -- DR11-south: the selector was the bottleneck
# Source: dr11-campaign-v4/papers/main.tex Table 2 (tab:recall) + Section 3.
# Complements recall_recoverability.png (a continuous no-retrain budget curve).
# --------------------------------------------------------------------------
def fig4_selector_bottleneck() -> None:
    configs = ["union-95k", "mean-150k\n(v3-lean)", "v4 re-sweep\n(fine-tuned)"]
    inch_A = [0.54, 0.80, 0.87]
    stor_A = [0.32, 0.61, 0.825]
    inch_B_v4 = 0.87
    frac = {"Inchausti-A": "60/69", "Storfer-A": "85/103", "Inchausti-B": "76/87"}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 5.2),
                                   width_ratios=[2.4, 1.0], constrained_layout=True)

    groups = np.arange(3)          # Inchausti-A, Storfer-A, Inchausti-B
    w = 0.26
    for j, cfg in enumerate(configs):
        vals = [inch_A[j], stor_A[j], inch_B_v4 if j == 2 else np.nan]
        axL.bar(groups + (j - 1) * w, vals, width=w, color=RAMP[j],
                label=cfg.replace("\n", " "), zorder=2)
        for g, v in zip(groups, vals):
            if not np.isnan(v):
                axL.text(g + (j - 1) * w, v + 0.016, f"{v:.2f}",
                         ha="center", fontsize=10, color=INK)

    # Count fractions ride above the v4 bar, not under the tick labels.
    for g, key in zip(groups, ["Inchausti-A", "Storfer-A", "Inchausti-B"]):
        v = [inch_A[2], stor_A[2], inch_B_v4][g]
        axL.text(g + w, v + 0.062, frac[key], ha="center", fontsize=9.5, color=MUTED)
    axL.text(2 + w, inch_B_v4 / 2, "v4 only", ha="center", va="center",
             rotation=90, fontsize=9.5, color="white")

    axL.set_xticks(groups)
    axL.set_xticklabels(["Inchausti grade-A", "Storfer grade-A\n(hard)",
                         "Inchausti grade-B"], fontsize=11)
    # Headroom above 1.0 so the legend never sits on the bar labels.
    axL.set_ylim(0, 1.32)
    axL.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axL.set_ylabel("held-out grade recall\n(position-crossmatch, training-excluded)")
    axL.grid(axis="y", zorder=0)
    axL.set_axisbelow(True)
    axL.legend(loc="upper left", ncols=3, fontsize=10.5, columnspacing=1.1,
               handlelength=1.2, handletextpad=0.5)
    axL.set_title("Recall at the deployed operating point", color=NAVY)

    # Right: what the top-150k is actually made of.
    retained, new = 15_922, 134_078
    axR.barh([0], [retained], height=0.5, color=RAMP[0], zorder=2)
    axR.barh([0], [new], left=[retained], height=0.5, color=RAMP[1], zorder=2)
    axR.text(retained + new / 2, 0.0, f"{new:,} new\n(89.4%)",
             ha="center", va="center", fontsize=11.5, color="white", fontweight="bold")
    axR.annotate(f"{retained:,} retained\nfrom union-95k  (10.6%)",
                 xy=(retained / 2, 0.26), xytext=(30_000, 0.72),
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2),
                 fontsize=10.5, color=INK)
    axR.set_yticks([])
    axR.set_xlim(0, 152_000)
    axR.set_xticks([0, 50_000, 100_000, 150_000])
    axR.set_xticklabels(["0", "50k", "100k", "150k"])
    axR.set_xlabel("v4 top-150k composition")
    axR.set_ylim(-0.62, 1.05)
    axR.grid(axis="x", zorder=0)
    axR.set_axisbelow(True)
    axR.set_title("A substantially new pool", color=NAVY)

    _frame(fig,
           "DR11-south: the selector, not the model, was the bottleneck",
           footer=[
               "Same scores, better selection: union -> calibrated mean buys +0.26-0.29 recall "
               "before the fine-tune adds the rest.",
               "Only 16.7% of the 95,104 union survivors reached v4's top-150k. "
               "Source: dr11-campaign-v4/papers/main.tex Table 2.",
           ])
    _save(fig, "fig4_selector_bottleneck.png")


# --------------------------------------------------------------------------
# FIG 5 -- the confidence chart
# Source: claudenet/data/v3/model_progression.json; papers/v3_section.tex
#         tab:progression. Euclid A-vs-C AUC is the one independent test.
# --------------------------------------------------------------------------
def fig5_confidence_frontier() -> None:
    gens = ["orig\n(random-neg)", "v2-lean", "v3blend8", "v3-head"]
    euclid = np.array([0.613, 0.647, 0.660, 0.615])
    se, n_a = 0.06, 66
    broad = np.array([0.598, 0.713, 0.790, 0.885])
    x = np.arange(4)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.8),
                                   sharey=True, constrained_layout=True)

    axL.axhspan(euclid.min() - se, euclid.max() + se, color=MUTED, alpha=0.11, zorder=0)
    axL.errorbar(x, euclid, yerr=se, fmt="o", color=MUTED, ms=9,
                 capsize=5, lw=1.6, mec="white", mew=1.4, zorder=3)
    axL.text(1.5, euclid.max() + se + 0.028,
             f"all four within $\\pm$1 SE (SE$\\approx${se}, $n_A$={n_a})\n"
             "-- statistically indistinguishable",
             ha="center", fontsize=11, color=CRIT, fontweight="bold")
    axL.set_title("Euclid grade-A vs C AUC\n(independent, unbiased)", color=NAVY)
    axL.set_ylabel("score")

    axR.plot(x, broad, "-o", color=NAVY, lw=2.2, ms=9, mec="white", mew=1.4, zorder=3)
    for xi, v in zip(x, broad):
        axR.text(xi, v + 0.030, f"{v:.3f}", ha="center", fontsize=10.5, color=INK)
    axR.annotate("", xy=(3, 0.885), xytext=(0, 0.598),
                 arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.4, ls=":"))
    axR.text(1.35, 0.66, "+0.29", fontsize=13, color=GOOD, fontweight="bold")
    axR.set_title("DR10-eval broad-mimic recovery\n(the population we can measure)", color=NAVY)

    for ax in (axL, axR):
        ax.set_xticks(x)
        ax.set_xticklabels(gens, fontsize=10.5)
        ax.set_ylim(0.50, 1.00)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

    _frame(fig,
           "We improve steadily on the broad mimic population --\n"
           "but not at the resolution-bounded frontier",
           title_lines=2,
           footer=[
               "Shared y-axis, deliberately unzoomed: zooming the left panel would manufacture "
               "a difference the standard errors deny.",
               "The seed mimic-FPR metric is selection-biased, so any \"Nx over the published "
               "models\" claim is retracted.",
           ])
    _save(fig, "fig5_confidence_frontier.png")


# --------------------------------------------------------------------------
# FIG 6 -- the v4 pipeline
# Source: v4_section.tex; dr11-campaign-v4/main.tex; claudenet scripts 380-396
# --------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, *, pending=False, fill="white",
         fontsize=10.5, weight="normal"):
    edge = CRIT if pending else NAVY
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06",
        linewidth=1.8, edgecolor=edge, facecolor=fill,
        linestyle="--" if pending else "-", zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, zorder=3, fontweight=weight)


def _arrow(ax, x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
        color=MUTED, lw=1.5, shrinkA=2, shrinkB=2, zorder=1,
    ))
    if label:
        ax.text((x1 + x2) / 2 + 0.12, (y1 + y2) / 2, label,
                fontsize=9.5, color=MUTED, va="center", ha="left")


def fig6_v4_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 6.8), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.8)
    ax.set_axis_off()

    cx, w, h = 3.3, 5.0, 0.62
    ys = [5.85, 4.95, 4.05, 3.15, 2.25, 1.20, 0.30]

    _box(ax, cx, ys[0], w, h,
         "DR11-south parent  -  53,809,040 galaxies\n"
         "TYPE $\\in$ {SER,EXP,DEV,REX},  NOBS$_{grz}\\geq$3,  $m_z<20$")
    _box(ax, cx, ys[1], w, h, "101 px  $grz$  cutouts")
    _box(ax, cx, ys[2], w, h,
         "STAGE 1  -  5-member calibrated MEAN\n"
         "AUC 0.9955 vs 2M random DR11 negatives", weight="bold")
    _box(ax, cx, ys[3], w, h, "top 150,000 by mean")
    _box(ax, cx, ys[4], w, h,
         "15,922 retained (union-95k)  +  134,078 NEW")
    _box(ax, cx, ys[5], w, h + 0.10,
         "STAGE 2  -  LensJudge v3 cascade\n"
         "DESI tier-1 triage  ->  HSC PDR3 tier-2 (0.168\"/px)", pending=True)
    _box(ax, cx, ys[6], w, h, "HSC tier-2 grade A/B", pending=True)

    mid = cx + w / 2
    for a, b in zip(ys[:-1], ys[1:]):
        _arrow(ax, mid, a, mid, b + (h + 0.10 if b == ys[5] else h))

    # Ensemble roster.
    _box(ax, 9.0, 3.55, 2.75, 1.60,
         "effnet_B        (frozen)\nzoobot_N       (frozen)\n"
         "effnet_S2_b50_dr11\neffnet_B3_b50_dr11\nresnet46_C_b50_dr11",
         fill="#F4F8FE", fontsize=9.8)
    ax.text(10.38, 5.28, "stage-1 roster", ha="center", fontsize=10.5,
            color=NAVY, fontweight="bold")
    _arrow(ax, 9.0, 4.35, cx + w, ys[2] + h / 2)

    # Fine-tune sidecar.
    _box(ax, 0.15, 3.30, 2.85, 1.85,
         "FINE-TUNE SIDECAR\n\n"
         "3,171 net-new positives (13x)\n"
         "-> 5,806 tier-subsampled\n"
         "+ 30k random + 20k hard negs\n\n"
         "12 epochs @ lr 3e-4\nwarm-start from _b50",
         fill="#F4F8FE", fontsize=9.8)
    _arrow(ax, 3.0, 4.22, cx, ys[2] + h / 2)
    ax.text(1.57, 3.05, "swaps in the 3 retrained members",
            ha="center", fontsize=9.5, color=MUTED)

    # Status key + banner.
    ax.add_patch(FancyBboxPatch((0.15, 1.35), 2.85, 0.62,
                                boxstyle="round,pad=0.06", linewidth=1.4,
                                edgecolor=CRIT, facecolor="#FDF2F2", zorder=2))
    ax.text(1.57, 1.66,
            "v4: Stage 1 COMPLETE\nStage 2 vetting PENDING",
            ha="center", va="center", fontsize=10.5, color=CRIT, fontweight="bold")
    ax.text(1.57, 0.72, "solid navy = done\ndashed red = pending",
            ha="center", va="center", fontsize=10, color=MUTED)

    ax.text(6.0, 6.68, "The ClaudeNet v4 DR11-south pipeline",
            ha="center", fontsize=15, color=NAVY, fontweight="bold")
    _save(fig, "fig6_v4_pipeline.png")


# --------------------------------------------------------------------------
# FIG 7 -- what ClaudeNet fixed, and what it cannot
# Source: claudenet/README.md phase table; v3_section.tex; v4_section.tex
# --------------------------------------------------------------------------
def fig7_bottlenecks_fixed() -> None:
    bottlenecks = ["ensemble diversity",
                   "negative quality /\noperating point",
                   "contaminant\nseparation",
                   "domain + release\nshift"]
    gens = ["v1", "v2", "v3", "v4"]
    # "-" open, "o" partial, "+" fixed
    status = [
        ["+", "+", "+", "+"],
        ["o", "+", "+", "+"],
        ["-", "-", "+", "+"],
        ["-", "-", "-", "+"],
    ]
    note = [
        ["decorrelated\nroster", "", "", ""],
        ["hard-neg\nmining", "NegEval-1M", "", ""],
        ["", "", "typed mimic\nbank", "hard DR11\nnegs"],
        ["MMD hurt", "", "", "native\nfine-tune"],
    ]
    face = {"+": "#E4F2E8", "o": "#FDEDE4", "-": "#EFEFEF"}
    edge = {"+": GOOD, "o": "#EC835A", "-": "#CFCFCF"}
    glyph = {"+": "fixed", "o": "partial", "-": "open"}

    fig, ax = plt.subplots(figsize=(10.0, 5.5), constrained_layout=True)
    ax.set_xlim(-0.05, 4.05)
    ax.set_ylim(-1.35, 4.35)
    ax.set_axis_off()

    for j, g in enumerate(gens):
        ax.text(j + 0.5, 4.12, g, ha="center", fontsize=13,
                color=NAVY, fontweight="bold")
    for i, b in enumerate(bottlenecks):
        y = 3 - i
        ax.text(-0.12, y + 0.5, b, ha="right", va="center",
                fontsize=11, color=INK)
        for j in range(4):
            s = status[i][j]
            ax.add_patch(Rectangle((j + 0.02, y + 0.02), 0.96, 0.96,
                                   facecolor=face[s], edgecolor=edge[s], lw=1.5))
            ax.text(j + 0.5, y + 0.68, glyph[s], ha="center", va="center",
                    fontsize=10.5, color=edge[s] if s != "-" else MUTED,
                    fontweight="bold" if s == "+" else "normal")
            if note[i][j]:
                ax.text(j + 0.5, y + 0.30, note[i][j], ha="center", va="center",
                        fontsize=8.8, color=MUTED)

    ax.add_patch(FancyBboxPatch((0.02, -1.22), 3.96, 0.98,
                                boxstyle="round,pad=0.03", linewidth=1.6,
                                edgecolor=CRIT, facecolor="#FDF2F2"))
    ax.text(2.0, -0.44, "Still unfixed -- and not fixable by a better model:",
            ha="center", fontsize=11.5, color=CRIT, fontweight="bold")
    ax.text(2.0, -0.90,
            "resolution ceiling 0.262\"/px (DECam native)          "
            "footprint: DECam-only (no BASS/MzLS north)",
            ha="center", fontsize=11, color=CRIT)

    ax.text(2.0, 4.62, "Four diagnosed bottlenecks, closed across v1 $\\rightarrow$ v4",
            ha="center", fontsize=15, color=NAVY, fontweight="bold")
    _save(fig, "fig7_bottlenecks_fixed.png")


def main() -> None:
    _style()
    print("generating deck figures ->", OUT_DIR)
    fig1_lineage_sweeps()
    fig2_recovery_matched_fpr()
    fig3_v4_finetune_gate()
    fig4_selector_bottleneck()
    fig5_confidence_frontier()
    fig6_v4_pipeline()
    fig7_bottlenecks_fixed()
    print("done.")


if __name__ == "__main__":
    main()
