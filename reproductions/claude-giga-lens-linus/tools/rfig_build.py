#!/usr/bin/env python3
"""P5 report figures (rfig_*.png) for campaign claude-giga-lens-linus.

Run under the cgl2 venv (source /raid/benson/.venvs/cgl2/bin/activate); CPU only.
Every number is taken from the artifact of record (paths in comments); where the
spine (research/synthesis_tables.md) flags a discrepancy, the ARTIFACT value is used.
Existing figs/*.png are untouched; all outputs are prefixed rfig_.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = "/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens-linus"
OLD = "/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens"
FIGS = os.path.join(ROOT, "figs")

# ---- palette (dataviz skill reference instance, light mode; PNGs for a print report) ----
BLUE = "#2a78d6"     # series 1: this campaign / scene API / S1
ORANGE = "#eb6834"   # series 2: old stack / frozen reference
AQUA = "#1baf7a"     # series 3: MCLMC
GOOD = "#0ca30c"     # status: converged / eliminated-clean
WARN = "#fab219"     # status: partial / mixed / ambiguous
SERIOUS = "#ec835a"  # status: blocked (budget / cost / reference)
CRIT = "#d03b3b"     # status: no-verdict / fail / demoted / standing indictment
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURF = "#fcfcfb"


def tint(hex_color, f=0.85):
    """Blend a hex color toward white by fraction f (f=1 -> white)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": False,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
    "savefig.dpi": 200,
})


def load(path):
    with open(path) as f:
        return json.load(f)


# =====================================================================================
# 1. rfig_decision_matrix.png  — targets x vehicles capability grid
#    Source: research/synthesis_tables.md §1 (spine), statuses per gate_ledger_final.md
# =====================================================================================
def fig_decision_matrix():
    cols = [
        "S1 prior-seeded MAMS-SMC\n(the campaign bet)",
        "Warm MAMS\n(MAP-warm, no tempering)",
        "MCLMC-mutation SMC\n(DEMOTED at B0)",
        "Classic MAP→SVI→HMC\n(frozen B3 settings)",
    ]
    # (row label, cost line, [(status, headline, detail) x4])
    NR = ("NOTRUN", "not run", "")
    rows = [
        ("T0 analytic\nmix2 / funnel10 / illcond46\n(B0)", "0 A100-h (local GPU)",
         [("PARTIAL", "PASS 2/3",
           "mix2 PASS (logZ err 0.073);\nillcond46 z-PASS, logZ err 2.40 nats\n= 6.6$\\sigma$_boot; funnel10 FAIL −0.66 nat"),
          ("NOTRUN", "not run", "(HMC-mutation SMC\narm ran instead)"),
          ("FAIL", "DEMOTED",
           "illcond46 ΔlogZ +10.79±0.54\n= 20.1$\\sigma$ vs MAMS\n⇒ never an evidence kernel"),
          NR]),
        ("hs2 easy scene\n22-d unimodal ×4 systems\n(B3)", "12.4 L4-h free tier; 0 A100-h",
         [("BLOCKED", "BLOCKED-BY-BUDGET",
           "0/4 arms inside the 5 h/arm\nL4 fence at N=512;\nthe COST row is the result"),
          NR, NR,
          ("BLOCKED", "BLOCKED-BY-REFERENCE",
           "4/4 complete, 0/4 accepted\n($\\hat{R}$ 4.36–7.87, ESS 64–84\nvs $\\hat{R}$<1.05 ∧ ESS≥200)")]),
        ("DSPL thin-ridge\n21-d, orig vs ratio coords\n(B2/B2′)", "2.9 A100-h",
         [("PARTIAL", "MIXED",
           "$\\lambda$=1 @64 stages, sanity CLEAN;\nfalsifier NOT-DECIDABLE-AS-REGISTERED;\nB2′ PASS-UNDERPOWERED (p=0.062)"),
          NR, NR, NR]),
        ("T2 46-d cond≈10$^{14}$\nv2d marg46 / diagonal scene\n(B4 + arbitration)", "B4 3.5 h; arb 3.8 h + 0.9 h",
         [("BLOCKED", "BLOCKED-BY-COST",
           "healthy sampler, 3× wall-capped\n($\\lambda$=0.587 @3.5 h A100)\n⇒ vehicle budget-infeasible"),
          ("NOTRUN", "not run", "(classic arm was\nwarm-started instead)"),
          NR,
          ("FAIL", "NO-VERDICT",
           "unconverged: $\\hat{R}$ 44.3, chains railed\n$\\gamma\\approx$1.99 at prior edge — reproduces the\nknown T2 single-stage pathology")]),
        ("T3 74-d bimodal\nfoundry_v3b74, diagonal\n(B5)", "1.87 A100-h (+1.66 infra-fail)",
         [("CONV", "CONVERGED",
           "G1 PASS both basins @$\\lambda$=1,\nretention 1.000; G2 FAIL-as-written 11$\\sigma$\n— indicts frozen reference coverage"),
          NR,
          ("PARTIAL", "AMBIGUOUS (G3)",
           "×56 minor-mode inflation\n[22–146]; within-basin\n$\\gamma$ intact (1.0935 vs 1.0919)"),
          NR]),
        ("Carousel 33-d multi-plane\nREAL MUSE cutouts\n(B1r, team-provided data)", "11.3 + 4.5 A100-h (+11.0 wave-1)",
         [("BLOCKED", "BLOCKED-BY-COST",
           "$\\lambda$=0.151 @36 stages,\n0 posterior samples — healthy\nsampler, killed by cost"),
          ("BLOCKED", "BLOCKED-BY-COST",
           "wall fence at burn 11/30:\n0 draws, ESS=0\n(41.5% of B* grads)"),
          NR,
          ("NOTRUN", "not run", "(candidate vehicle;\nest 17–20 h)")]),
        ("v3b correlated REAL\n46-d whitened likelihood\n(L0-G2)", "0.53 A100-h",
         [("CONV", "CONVERGED",
           "PASS: $\\gamma$ Δ 0.0027, logZ Δ 0.11 nats\ncross-stack; best evidence\nrow of the campaign"),
          NR, NR, NR]),
    ]
    scol = {"CONV": GOOD, "PARTIAL": WARN, "BLOCKED": SERIOUS, "FAIL": CRIT, "NOTRUN": GRID}

    ncol, nrow = len(cols), len(rows)
    lw_label = 2.6           # row-label column width
    cw, rh = 3.05, 1.28      # cell size
    W = lw_label + ncol * cw
    H = nrow * rh + 1.3
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    top = H - 0.75

    fig.text(0.012, 1 - 0.28 / H, "The sampler decision matrix — target class × vehicle (P2 deliverable)",
             fontsize=13, fontweight="bold", color=INK, va="top")
    # column headers
    for j, c in enumerate(cols):
        ax.text(lw_label + (j + 0.5) * cw, top + 0.30, c, ha="center", va="center",
                fontsize=8.6, fontweight="bold", color=INK)
    for i, (rlab, cost, cells) in enumerate(rows):
        y0 = top - (i + 1) * rh
        ax.text(0.06, y0 + rh * 0.62, rlab, ha="left", va="center", fontsize=7.8,
                fontweight="bold", color=INK)
        ax.text(0.06, y0 + rh * 0.16, cost, ha="left", va="center", fontsize=7.0,
                color=MUTED, style="italic")
        for j, (st, head, det) in enumerate(cells):
            x0 = lw_label + j * cw
            face = tint(scol[st], 0.82) if st != "NOTRUN" else "#f4f3f0"
            edge = scol[st] if st != "NOTRUN" else GRID
            ax.add_patch(FancyBboxPatch((x0 + 0.06, y0 + 0.05), cw - 0.12, rh - 0.10,
                                        boxstyle="round,pad=0.02,rounding_size=0.06",
                                        fc=face, ec=edge, lw=1.1))
            if st == "NOTRUN":
                ax.text(x0 + cw / 2, y0 + rh * 0.62, head, ha="center", va="center",
                        fontsize=7.6, color=MUTED, style="italic")
                if det:
                    ax.text(x0 + cw / 2, y0 + rh * 0.30, det, ha="center", va="center",
                            fontsize=6.4, color=MUTED)
            else:
                hcol = scol[st] if st != "PARTIAL" else "#8a6400"  # darker label for yellow fill
                ax.text(x0 + cw / 2, y0 + rh - 0.24, head, ha="center", va="center",
                        fontsize=7.9, fontweight="bold", color=hcol)
                ax.text(x0 + cw / 2, y0 + rh * 0.40, det, ha="center", va="center",
                        fontsize=6.5, color=INK2)
    # legend + footer
    ly = top - nrow * rh - 0.30
    items = [("CONVERGED (evidence delivered)", GOOD), ("PARTIAL / MIXED / AMBIGUOUS", WARN),
             ("BLOCKED (budget / cost / reference)", SERIOUS), ("NO-VERDICT / FAIL / DEMOTED", CRIT),
             ("not run", GRID)]
    x = 0.06
    for lab, c in items:
        ax.add_patch(FancyBboxPatch((x, ly - 0.075), 0.22, 0.15,
                                    boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc=tint(c, 0.75) if c != GRID else "#f4f3f0", ec=c, lw=1.1))
        ax.text(x + 0.30, ly, lab, ha="left", va="center", fontsize=7.4, color=INK2)
        x += 0.30 + 0.085 * len(lab) + 0.42
    ax.text(0.06, ly - 0.40,
            "S1 is the only vehicle that produced logZ anywhere; every S1 non-completion is a wall-cap on a "
            "healthy sampler (cost, never health).  Classic at frozen single-stage settings produced no accepted "
            "posterior in the campaign.  Sources: research/synthesis_tables.md §1; per-cell artifacts in data/.",
            ha="left", va="center", fontsize=7.2, color=MUTED)
    fig.savefig(os.path.join(FIGS, "rfig_decision_matrix.png"), bbox_inches="tight")
    plt.close(fig)


# =====================================================================================
# 2. rfig_bracket_scoreboard.png — mechanism/suspects program ending at the standing
#    nonstationarity mechanism + certified gamma. Source: synthesis_tables.md §2.
# =====================================================================================
def fig_bracket_scoreboard():
    cards = [  # (tag, suspect, verdict, vcol, measurement)
        ("X1-G0", "Profile curvature (radial mass structure)", "DEAD", GOOD,
         "no monotone r_eff ordering in 24/24 variants; fine & binned constrain\n"
         "the slope at the SAME radius ($\\Delta$r_eff 0.008″) yet $\\Delta\\gamma$=0.71\n"
         "⇒ would need |d$\\gamma$/d ln r| ≈ 226 vs O(1) — P4 retired at 0 GPU-h (D7)"),
        ("T0.3", "Companion galaxy (LL2/LL3 misfit)", "EXONERATED", GOOD,
         "whiten-then-drop (keep_w 9273→8247): $\\gamma$ shift −0.0021\n"
         "(0.26$\\sigma$_stat; confirm band was ≥ +0.024)"),
        ("T0.4-2", "Spectrum flooring (information discard at zeros)", "FALSIFIED", GOOD,
         "data reject floored kernels by ~3.2k ($\\lambda$=0.1) to ~33k ($\\lambda$=1) nats;\n"
         "pathology localized to down-weighted high-S large-scale modes"),
        ("T0.4-1", "Stationarity of the noise-model class", "REJECTED — STANDING", CRIT,
         "v3b max|z| = 3.67, calibrated p = 0.010; arc-excluded p = 0.010\n"
         "(replicated pattern); cross-block $\\rho$(0,1) spread = 16.4× the\n"
         "±0.008 drizzle-registration envelope"),
        ("T0.4-3", "Real-space head-to-head (context, report-only)", "CONTEXT", MUTED,
         "same v3b pixels, real space: $\\gamma\\approx$1.29 fits best ($\\chi^2$_pp 1.58) vs\n"
         "anchor 1.433 (7.44) vs corr-low 1.103 (8.32) — and the whitened\n"
         "metric INVERTS the ordering (+501 nats for corr-low)"),
        ("T1.1", "Injection methodology (7.80 A100-h)", "CONFOUNDED", WARN,
         "$\\gamma$_rec biases +0.082/+0.139/+0.075 — positive-signed, outside BOTH\n"
         "zones; control FAILS HIGH AND SICK (1/128 unique) ⇒ scene-subtraction\n"
         "residue implicated; differential corr−diag −0.0526 (≈16% of gap)"),
        ("T1.1b", "Residue-masked refit (entry gate)", "STOPPED AT GATE", WARN,
         "whiten-then-drop dof loss 85.8% vs ≤40% budget (arc-band px 6373→108);\n"
         "fit-side masking structurally incompatible ⇒ data-side injections queued"),
        ("—", "Source model / PSF representation", "ALIVE (untested)", WARN,
         "inherits the bracket question from X1-G0's fixed-radius finding;\n"
         "PSF-marg MVP never run (pool reallocated by D8)"),
    ]
    fig = plt.figure(figsize=(13.6, 10.1))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13.6); ax.set_ylim(0, 10.1); ax.axis("off")

    ax.text(0.25, 9.75, "The bracket: why $\\gamma$_binned,corr = 1.103 ≠ 1.433 native anchor "
                        "($\\Delta\\gamma$ = 0.33, 17$\\sigma$) — suspects scoreboard",
            fontsize=13, fontweight="bold", color=INK, va="center")

    ch, gap = 0.98, 0.065
    x0, cwid = 0.25, 8.6
    top = 9.35
    for i, (tag, suspect, verdict, vcol, meas) in enumerate(cards):
        y0 = top - (i + 1) * (ch + gap)
        ax.add_patch(FancyBboxPatch((x0, y0), cwid, ch,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=tint(vcol, 0.90), ec=vcol, lw=1.3))
        ax.text(x0 + 0.16, y0 + ch - 0.23, tag, fontsize=9.5, fontweight="bold",
                color=INK, va="center")
        ax.text(x0 + 0.95, y0 + ch - 0.23, suspect, fontsize=9.0, color=INK, va="center")
        # verdict pill
        px = x0 + cwid - 0.18 - 0.115 * len(verdict)
        ax.add_patch(FancyBboxPatch((px, y0 + ch - 0.36), 0.105 * len(verdict) + 0.12, 0.27,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    fc=vcol if vcol != WARN else "#8a6400", ec="none"))
        ax.text(px + 0.055 * len(verdict) + 0.06, y0 + ch - 0.225, verdict, fontsize=7.6,
                fontweight="bold", color="white", ha="center", va="center")
        ax.text(x0 + 0.16, y0 + 0.33, meas, fontsize=7.3, color=INK2, va="center")

    # arrow to the standing-mechanism card
    ax.add_patch(FancyArrowPatch((x0 + cwid + 0.12, 5.25), (x0 + cwid + 0.62, 5.25),
                                 arrowstyle="-|>", mutation_scale=26, color=INK2, lw=2.2))
    # final card
    fx, fw = x0 + cwid + 0.70, 3.85
    ax.add_patch(FancyBboxPatch((fx, 2.05), fw, 6.35,
                                boxstyle="round,pad=0.03,rounding_size=0.10",
                                fc=tint(CRIT, 0.93), ec=CRIT, lw=1.8))
    ax.text(fx + fw / 2, 8.02, "STANDING MECHANISM", fontsize=10.5, fontweight="bold",
            color=CRIT, ha="center", va="center")
    ax.text(fx + 0.18, 6.85,
            "A NONSTATIONARY correlated background\ncomponent priced as stationary lets the\n"
            "whitened metric discount large-scale\nreal-space misfit ⇒ biases $\\gamma$ low.\n\n"
            "Noise-model-CLASS misspecification —\nnot source/PSF — is the prime suspect.\n"
            "Unrefuted by T1.1 (in-class injection\nscene lacks the misfit lever arm).",
            fontsize=8.2, color=INK, va="center")
    ax.plot([fx + 0.18, fx + fw - 0.18], [5.52, 5.52], color=BASE, lw=1)
    ax.text(fx + fw / 2, 5.18, "CERTIFIED NUMBER (T0.2)", fontsize=8.6, fontweight="bold",
            color=INK, ha="center", va="center")
    ax.text(fx + fw / 2, 4.52, "$\\gamma$_binned(corr, low) = 1.1032 ± 0.0086",
            fontsize=11.5, fontweight="bold", color=BLUE, ha="center", va="center")
    ax.text(fx + 0.18, 3.32,
            "$\\sigma$_tot = ($\\sigma$_stat$^2$+$\\sigma$_seed$^2$)$^{1/2}$, seeds {2,3,4};\n"
            "low-basin preference seed-stable\n(ΔlogZ −28.9/−32.3/−31.2, ≥16$\\sigma$_seed);\n"
            "reproduced cross-stack on the scene API:\nΔ$\\gamma$ 0.0027, ΔlogZ 0.11 nats (L0-G2).",
            fontsize=7.8, color=INK2, va="center")
    ax.text(0.25, 0.42,
            "Verdict colors: green = suspect eliminated; red = rejected model class (the standing mechanism); "
            "yellow = confounded / stopped / untested; gray = context.  All numbers artifact-traced:\n"
            "data/x1_g0_effective_radii.json, data/t02_t03_gate_eval.json, data/t04_*.json, data/t11_gate_eval.json, "
            "data/t11b_residue_mask_report.json, data/results-perlmutter/l0g2_v3b_scene_seed2.json.",
            fontsize=7.2, color=MUTED, va="center")
    fig.savefig(os.path.join(FIGS, "rfig_bracket_scoreboard.png"), bbox_inches="tight")
    plt.close(fig)


# =====================================================================================
# 3. rfig_crossstack_l0g2.png — L0-G2 cross-stack reproduction
# =====================================================================================
def fig_crossstack():
    l0 = load(os.path.join(ROOT, "data/results-perlmutter/l0g2_v3b_scene_seed2.json"))
    old = load(os.path.join(OLD, "data/results/e2_v3b_low_smc_canary_fix.json"))
    t02 = load(os.path.join(ROOT, "data/t02_t03_gate_eval.json"))
    # 128 final SMC particles forwarded through the OLD stack's own 46-d bijector
    # (cgl.e2.build_target('v3b').model.bij, CPU bijector-only — 25a/01a precedent;
    #  median reproduces the ledgered 1.103198 exactly, q16 to 1e-6)
    g_old = np.load(os.path.join(ROOT, "data/rfig_oldstack_gamma_low_constrained.npy"))

    lo = l0["legs"]["low"]
    ob = old["basins"]["low"]
    sig_tot = 0.00864  # T0.2 certification (synthesis §2.3)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    fig.suptitle("L0-G2 cross-stack reproduction: old validated stack (phoenix) vs scene-API port "
                 "(Perlmutter A100) on the v3b correlated real-data target", fontsize=11,
                 fontweight="bold", y=1.02)

    # (a) gamma_low posterior
    ax = axes[0]
    ax.hist(g_old, bins=24, density=True, color=tint(ORANGE, 0.55), edgecolor=ORANGE,
            lw=0.8, label="old stack (128 SMC particles)")
    ax.axvspan(1.1032 - sig_tot, 1.1032 + sig_tot, color=GRID, alpha=0.55, zorder=0,
               label="certified 1.1032 ± 0.0086 ($\\sigma$_tot)")
    ymax = ax.get_ylim()[1]
    yq = -0.085 * ymax
    ax.errorbar([lo["gamma_median"]], [yq],
                xerr=[[lo["gamma_median"] - lo["gamma_q16"]], [lo["gamma_q84"] - lo["gamma_median"]]],
                fmt="o", color=BLUE, ms=7, capsize=4, lw=2.2, clip_on=False,
                label="scene API: median, q16–q84")
    ax.errorbar([ob["gamma_median"]], [yq * 1.9],
                xerr=[[ob["gamma_median"] - ob["gamma_q16"]], [ob["gamma_q84"] - ob["gamma_median"]]],
                fmt="s", color=ORANGE, ms=6, capsize=4, lw=2.2, clip_on=False)
    ax.annotate("$\\Delta\\gamma$ = 0.0027\n(gate ≤ 0.017)", xy=(1.082, ymax * 0.55),
                fontsize=8, color=INK2, ha="center")
    ax.set_ylim(yq * 2.6, ymax * 1.32)
    ax.set_xlim(1.065, 1.135)
    ax.set_xlabel("$\\gamma$ (low basin)"); ax.set_ylabel("density")
    ax.set_title("(a) $\\gamma$_low posterior, both stacks", fontsize=9.5)
    ax.legend(fontsize=6.8, loc="upper left", frameon=False)

    # (b) logZ_low
    ax = axes[1]
    vals = [ob["logZ"], lo["logZ"]]
    ax.scatter([0, 1], vals, s=90, color=[ORANGE, BLUE], zorder=3)
    ax.plot([0, 1], vals, color=BASE, lw=1.2, zorder=2)
    ax.annotate("ΔlogZ = 0.11 nats\nacross stacks & machines", xy=(0.5, np.mean(vals) + 0.35),
                ha="center", fontsize=8.6, color=INK)
    # seed-spread scale bar (P1: sigma_seed_logZ low = 1.22 nats)
    ssz = t02["t02"]["basins"]["low"]["sigma_seed_logZ"]
    ax.errorbar([0.5], [np.mean(vals) - 1.15], yerr=[[ssz / 2], [ssz / 2]], fmt="none",
                ecolor=MUTED, capsize=4, lw=1.4)
    ax.annotate(f"seed spread $\\sigma$_seed = {ssz:.2f} nats\n(scale bar, P1 T0.2)",
                xy=(0.56, np.mean(vals) - 1.15), fontsize=7.2, color=MUTED, va="center")
    ax.set_xticks([0, 1], ["old stack\n−4771.08", "scene API\n−4770.97"])
    ax.set_ylabel("logZ (low basin)")
    ax.set_ylim(min(vals) - 2.2, max(vals) + 1.4)
    ax.set_title("(b) logZ agreement", fontsize=9.5)

    # (c) dlogZ(steep-low)
    ax = axes[2]
    fam = t02["t02"]["dlogZ_steep_minus_low_by_seed"]
    seeds = sorted(fam)
    y = [fam[s] for s in seeds]
    m, ss = float(np.mean(y)), t02["t02"]["sigma_seed_dlogZ"]
    ax.axhspan(m - ss, m + ss, color=GRID, alpha=0.55, label="seed family mean ± $\\sigma$_seed (1.79)")
    ax.scatter(range(3), y, s=70, color=ORANGE, zorder=3, label="old stack seeds {2,3,4}")
    ax.scatter([3], [l0["H"]["dlogZ_steep_minus_low"]], s=95, color=BLUE, marker="D",
               zorder=3, label="scene API (−29.36)")
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xticks(range(4), ["seed 2", "seed 3", "seed 4", "scene\nAPI"])
    ax.set_ylabel("$\\Delta$logZ (steep − low), nats")
    ax.set_title("(c) basin preference: sign & magnitude in family", fontsize=9.5)
    ax.set_ylim(-36.5, -25)
    ax.legend(fontsize=6.8, loc="lower left", frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.005, -0.06,
             "Artifacts: old stack e2_v3b_low_smc_canary_fix.{json,npz} (prev. campaign; final particles forwarded through the "
             "old stack's own bijector,\ndata/rfig_oldstack_gamma_low_constrained.npy — median reproduces 1.103198 exactly); "
             "scene API data/results-perlmutter/l0g2_v3b_scene_seed2.json\n(0.53 A100-h; draws npz on $PSCRATCH, summary "
             "quantiles plotted); seed family data/t02_t03_gate_eval.json.", fontsize=6.8, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "rfig_crossstack_l0g2.png"), bbox_inches="tight")
    plt.close(fig)


# =====================================================================================
# 4. rfig_b5_evidence.png — B5 per-basin evidence picture (data/b5_gate_eval.json)
# =====================================================================================
def fig_b5_evidence():
    b5 = load(os.path.join(ROOT, "data/b5_gate_eval.json"))
    ref = b5["p2c_reference"]
    g2, g3 = b5["G2"], b5["G3"]
    logZ_low_mams, s_low = g3["logZ_low_mams"], g3["sigma_boot_mams"]
    logZ_steep_mams = b5["G1"]["steep_mams"]["logZ"]
    s_steep = b5["G1"]["steep_mams"]["logZ_boot_sigma"]

    off_low = logZ_low_mams - ref["logZ_low"]
    off_steep = logZ_steep_mams - ref["logZ_steep"]
    e_low = np.hypot(s_low, ref["logZ_low_sigma"])
    e_steep = np.hypot(s_steep, ref["logZ_steep_sigma"])

    fig = plt.figure(figsize=(12.6, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.52, wspace=0.5)
    axA = fig.add_subplot(gs[0, :2])
    axB = fig.add_subplot(gs[0, 2:])
    axC1 = fig.add_subplot(gs[1, 0])
    axC2 = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[1, 2:])
    fig.suptitle("B5 (74-d bimodal foundry_v3b74): the per-basin evidence picture", fontsize=12,
                 fontweight="bold", y=0.985)

    # (a) per-basin absolute offsets vs P2c
    bars = axA.bar([0, 1], [off_low, off_steep], width=0.5,
                   color=[tint(BLUE, 0.25), tint(BLUE, 0.55)], edgecolor=BLUE,
                   yerr=[e_low, e_steep], capsize=5, ecolor=INK2)
    axA.axhline(0, color=BASE, lw=1.2)
    axA.set_xticks([0, 1], ["low basin\n+25.16 ± 0.74", "steep basin\n+4.86 ± 1.74"])
    axA.set_ylabel("logZ(S1 MAMS, $\\lambda$=1) − logZ(P2c ref), nats")
    axA.set_title("(a) MORE evidence than the frozen reference\nin BOTH basins (same sign)", fontsize=9.5)
    axA.spines[["top", "right"]].set_visible(False)

    # (b) the G2 gate — direct labels, no legend (relief rule)
    axB.axhspan(ref["dlogZ_steep_minus_low"] - g2["band_3sigma"],
                ref["dlogZ_steep_minus_low"] + g2["band_3sigma"], color=tint(ORANGE, 0.6))
    axB.axhline(ref["dlogZ_steep_minus_low"], color=ORANGE, lw=1.6)
    axB.text(0.03, ref["dlogZ_steep_minus_low"] + 0.7, "P2c frozen ref 162.2 ± 3$\\sigma$ band (±5.54)",
             fontsize=7.6, color="#a03d10", va="bottom")
    axB.errorbar([0.42], [g2["dlogZ_measured"]], yerr=[3 * g2["sigma_boot_combined"]], fmt="D",
                 color=BLUE, ms=9, capsize=5, lw=2)
    axB.text(0.47, g2["dlogZ_measured"], "measured (S1 MAMS, $\\lambda$=1)\n141.90 ± 0.45_boot (3$\\sigma$ bar)",
             fontsize=7.6, color=BLUE, va="center")
    axB.annotate("−20.30 nats = 11.0$\\sigma$\nG2 FAIL-as-written", xy=(0.12, 150.5), fontsize=9,
                 color=CRIT, fontweight="bold")
    axB.set_xlim(0, 1); axB.set_xticks([])
    axB.set_ylim(138, 170)
    axB.set_ylabel("$\\Delta$logZ (steep − low), nats")
    axB.set_title("(b) the gate: $\\Delta$logZ vs frozen P2c reference\n(n=1 seed; attribution needs seed-repeat)", fontsize=9.5)
    axB.spines[["top", "right"]].set_visible(False)

    # (c) gamma-disjoint ensembles vs reference-chain support
    cov = g2["reference_comparability"]["within_basin_coverage_evidence"]
    for ax, ours, refr, ttl, note in [
        (axC1, cov["low_ours_gamma_range"], cov["low_refchains_gamma_range"],
         "(c) low basin: $\\gamma$ support", "ref chains\n(full range)"),
        (axC2, cov["steep_ours_gamma_range"], cov["steep_refchains_gamma_med_q95"],
         "(d) steep basin: $\\gamma$ support", "ref chains\n(med–q95)"),
    ]:
        ax.plot(ours, [1, 1], lw=8, color=BLUE, solid_capstyle="butt")
        ax.plot(refr, [0, 0], lw=8, color=ORANGE, solid_capstyle="butt")
        ax.set_yticks([0, 1], [note, "$\\lambda$=1\nensemble"], fontsize=7.5)
        ax.set_ylim(-0.7, 1.7)
        ax.set_title(ttl + "\nDISJOINT", fontsize=9)
        ax.set_xlabel("$\\gamma$")
        ax.spines[["top", "right"]].set_visible(False)
    axC1.annotate("MAMS 1.0919 / MCLMC 1.0935\n(two kernels agree)", xy=(1.10, 1.38), fontsize=6.8,
                  color=INK2, ha="left")

    # (e) MCLMC x56 inflation
    axD.axvspan(1.5, 3.0, color=tint(AQUA, 0.6), label="pre-registered ceiling 1.5–3×")
    ci = g3["distortion_ci95_boot"]
    axD.errorbar([g3["distortion_factor"]], [0.5],
                 xerr=[[g3["distortion_factor"] - ci[0]], [ci[1] - g3["distortion_factor"]]],
                 fmt="o", ms=10, color=AQUA, capsize=5, lw=2.2,
                 label="MCLMC/MAMS minor-mode weight ×56 [22–146]")
    axD.axvline(1, color=BASE, lw=1.2)
    axD.set_xscale("log")
    axD.set_xlim(0.7, 300); axD.set_ylim(0, 1); axD.set_yticks([])
    axD.set_xlabel("minor-mode (low) weight inflation factor, MCLMC vs MAMS (log scale)")
    axD.set_title("(e) G3: +4.03 ± 0.49 nats = 8.3$\\sigma$_boot ≠ 0, 6.0$\\sigma$ above the ceiling — BUT inside the\n"
                  "imported 5.56-nat $\\sigma$_seed band ⇒ frozen clauses conflict: AMBIGUOUS (within-basin $\\gamma$ intact)",
                  fontsize=8.6)
    axD.legend(fontsize=7.4, loc="upper right", frameon=False)
    axD.spines[["top", "right"]].set_visible(False)

    fig.text(0.005, -0.015,
             "Artifact: data/b5_gate_eval.json (jobs 56006049 low-MAMS, 56251555 steep-MAMS attempt #4, 56006052 low-MCLMC); "
             "occupancy robust across every estimator\n(w_low ≤ 1.3e−60); the G2 FAIL indicts the frozen reference's "
             "within-basin coverage at least as much as the vehicle (finding B).", fontsize=6.8, color=MUTED, va="top")
    fig.savefig(os.path.join(FIGS, "rfig_b5_evidence.png"), bbox_inches="tight")
    plt.close(fig)


# =====================================================================================
# 5. rfig_budget_burn.png — A100-h burn by phase vs caps + cumulative (CAMPAIGN.md ledger)
# =====================================================================================
def fig_budget():
    phases = [  # (label, actual, cap, cap kind)
        ("P1 T0.2/T0.3", 2.21, 9.0, "est committed 9.0"),
        ("T1.1 (D7 pool)", 7.80, 10.0, "freed-P4 pool 10.0"),
        ("P2 benchmark", 22.45, 24.0, "cap 24"),
        ("P2b B1r (D8)", 15.85, 18.0, "cap 18"),
        ("P3 anchor/port", 5.21, 17.0, "cap 17"),
    ]
    total, cap = 53.51, 100.0

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.4), width_ratios=[1.05, 1])
    fig.suptitle("A100-hour burn: phase caps respected everywhere; campaign closed at 53.51 of the 100-h hard stop",
                 fontsize=11.5, fontweight="bold", y=1.0)

    y = np.arange(len(phases))[::-1]
    for yi, (lab, act, capv, capnote) in zip(y, phases):
        axA.barh(yi, capv, height=0.62, color="#f0efec", edgecolor=BASE, lw=1)
        axA.barh(yi, act, height=0.62, color=BLUE, edgecolor="none")
        axA.text(capv + 0.35, yi, f"{act:.2f} / {capnote}", va="center", fontsize=7.6, color=INK2)
    axA.set_yticks(y, [p[0] for p in phases], fontsize=8.5)
    axA.set_xlabel("A100-h")
    axA.set_xlim(0, 33)
    axA.set_title("(a) by phase: actual (blue) vs cap (track)", fontsize=9.5)
    axA.spines[["top", "right"]].set_visible(False)

    # cumulative ledger milestones (CAMPAIGN.md harvest rows; sub-day x offsets for display only)
    xs = [15.0, 16.0, 19.0, 20.0, 20.5, 21.0, 21.35]
    ys = [10.01, 29.27, 32.76, 36.37, 47.70, 52.61, 53.51]
    lbl = ["P1+T1.1", "wave-1", "recovery", "fix-wave", "P2b S1r", "final wave", "l0arbc"]
    offs = [(5, -10), (5, -10), (5, -10), (5, -10), (7, -14), (-62, 4), (7, -14)]
    axB.axhline(cap, color=CRIT, lw=1.4, ls="--")
    axB.text(15.05, cap - 4.5, "100 A100-h HARD STOP (D5)", fontsize=7.6, color=CRIT)
    axB.step(xs, ys, where="post", color=BLUE, lw=2)
    axB.scatter(xs, ys, s=34, color=BLUE, zorder=3)
    for xi, yi, li, off in zip(xs, ys, lbl, offs):
        axB.annotate(li, (xi, yi), textcoords="offset points", xytext=off, fontsize=6.6, color=MUTED)
    axB.annotate(f"close: {total:.2f} h (53.5%)", (xs[-1], ys[-1]), textcoords="offset points",
                 xytext=(-30, 12), fontsize=8.4, color=INK, fontweight="bold")
    axB.set_xticks([15, 16, 17, 18, 19, 20, 21],
                   ["Jul 15", "16", "17", "18", "19", "20", "21"])
    axB.set_ylabel("cumulative A100-h"); axB.set_ylim(0, 110); axB.set_xlim(14.8, 22.0)
    axB.set_title("(b) cumulative burn at ledger harvest points", fontsize=9.5)
    axB.spines[["top", "right"]].set_visible(False)

    fig.text(0.005, -0.045,
             "Ledger: CAMPAIGN.md A100-hour table (sacct -X actuals; component sum 53.52 vs quoted total 53.51 is per-row "
             "rounding, FLAG-7).\nFree-tier GPU-h not counted here: B3 12.38 phoenix-L4 h; X2 SBC 29.3 A16-h. "
             "Zero-cost closures: X1-G0, T1.1b-G0, B2′, Fermat teaser.", fontsize=6.8, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "rfig_budget_burn.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_decision_matrix()
    fig_bracket_scoreboard()
    fig_crossstack()
    fig_b5_evidence()
    fig_budget()
    print("wrote:", ", ".join(sorted(f for f in os.listdir(FIGS) if f.startswith("rfig_"))))
