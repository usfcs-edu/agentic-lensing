"""61_harvest_e2.py — E2 harvest: figures FIRST, then the verdict metrics.

Inputs (already on disk, produced by the pre-registered E2 runs):
  data/mclmc_diag_odell.{npz,json}   O2 apples-to-apples fit on Evan's product
                                     (16 chains x 4 seed groups, E1-frozen gates)
  data/o4_{v3b,v2d}_stage2.{npz,json} O4 Evan-style stage-2 rescue on OUR stored
                                     E1 chains (checkpoint research/checkpoint_e2_o4.md)
  data/mclmc_diag_{v3b,v2d}.{npz,json} E1 stage-1 chains (stage-1 references)
  data/mclmc_warm_v2d_x46.npz        warm cloud (odell init gammas via init_idx)

Outputs:
  figs/e2_odell_migration.png  Evan's-product gamma landscape (companion to
                               figs/e1_v3b_migration.png)
  figs/e2_o4_rescue.png        stage-1 vs stage-2 R-hat/gamma, both arms
  data/e2_harvest.json         verdicts + full number set (metrics stage only;
                               every number recomputed from the npz and asserted
                               vs the run jsons)

House rules: plots generated and INSPECTED before the metrics stage writes any
verdict; no gamma from a failed fit is quotable (demonstration telemetry only);
no team-private content. Run (cgl2 venv, CPU fine):
  /raid/benson/.venvs/cgl2/bin/python 61_harvest_e2.py figs
  /raid/benson/.venvs/cgl2/bin/python 61_harvest_e2.py metrics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens-linus")
DATA, FIGS = ROOT / "data", ROOT / "figs"

# dataviz reference palette, categorical slots 1-4 (same instance the E1
# figures validated; re-validated this session: PASS, contrast WARN on
# green/yellow handled by direct labels + legends).
GC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUT, GRID, SPINE, GREY = "#3a3a38", "#8a8a86", "#e8e7e3", "#d5d4d0", "#8a8a86"
BAND_TINT = "#edf1ed"
BOX = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2)

ANCHOR = dict(med=1.4330, ci68=(1.3995, 1.4685))   # foundry-i hmc_v13_v2d
E1_V2D = dict(med=1.4683, ci68=(1.4343, 1.5048))   # E1 quotable (v2d MCLMC)
CORR_LOW = 1.1032                                  # certified correlated money number
BAND = (1.15, 2.0)                                 # E1-G2 == E2-G2 containment
RHAT_GATE, ESS_GATE = 1.01, 1000.0                 # frozen E1-G1 == E2-G1


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_odell():
    j = json.loads((DATA / "mclmc_diag_odell.json").read_text())
    z = np.load(DATA / "mclmc_diag_odell.npz", allow_pickle=True)
    # odell warm start = the E1 v2d healthy-anchor cloud transported (KEYED);
    # gamma is a mass param, preserved exactly by the transport (roundtrip
    # dgamma 2.2e-16 in the warm meta), so init gammas index gamma46 directly.
    w = np.load(DATA / "mclmc_warm_v2d_x46.npz", allow_pickle=True)
    ng = int(j["config"]["n_groups"])
    gi = list(z["mass_names"]).index("gamma")
    gam = np.stack([z[f"mass_g{g}"][:, :, gi] for g in range(ng)])   # (G,C,D)
    g46 = np.asarray(w["gamma46"], dtype=np.float64)
    init = np.stack([g46[z[f"init_idx_g{g}"]] for g in range(ng)])   # (G,C)
    return j, z, gam, init


def load_o4(arm):
    j = json.loads((DATA / f"o4_{arm}_stage2.json").read_text())
    z = np.load(DATA / f"o4_{arm}_stage2.npz", allow_pickle=True)
    gi = list(z["mass_names"]).index("gamma")
    gam = np.asarray(z["mass"][:, :, gi], dtype=np.float64)          # (16, 5000)
    return j, z, gam


def style(ax):
    ax.tick_params(colors=MUT, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(SPINE)


# =========================================================================== #
# Figure 1: odell migration landscape (companion to figs/e1_v3b_migration.png)
# =========================================================================== #
def fig_odell_migration():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    j, z, gam, init = load_odell()
    ng, nc, nd = gam.shape
    med = np.median(gam, axis=2)
    q16, q84 = np.quantile(gam, 0.16, axis=2), np.quantile(gam, 0.84, axis=2)
    viol = j["gates"]["E1_G2"]["violations"]
    fo = [v["frac_outside"] for v in viol]

    fig, (ax, axk) = plt.subplots(
        1, 2, figsize=(11.8, 6.4), dpi=200, sharey=True,
        gridspec_kw=dict(width_ratios=[3.1, 1.0], wspace=0.04))
    ylim = (1.02, 1.58)
    for a in (ax, axk):
        style(a)
        a.set_ylim(*ylim)
        a.axhspan(BAND[0], ylim[1], color=BAND_TINT, lw=0, zorder=0)
        a.axhline(BAND[0], color=MUT, lw=1.2)
        a.axhspan(*ANCHOR["ci68"], color=GRID, alpha=0.55, lw=0)
        a.axhline(ANCHOR["med"], color=MUT, lw=1.0, ls=(0, (4, 3)))
        a.axhline(CORR_LOW, color=MUT, lw=1.0, ls=(0, (1, 2)))

    for g in range(ng):
        x = np.arange(nc) + g * nc
        ax.vlines(x, med[g], init[g], color=GREY, lw=0.6, alpha=0.55, zorder=2)
        ax.plot(x, init[g], "o", ms=4.2, mfc="white", mec=GREY, mew=1.0,
                ls="none", zorder=3)
        ax.vlines(x, q16[g], q84[g], color=GC[g], lw=2.6, alpha=0.9, zorder=4)
        ax.plot(x, med[g], "o", ms=4.6, color=GC[g], mec="white", mew=0.7,
                ls="none", zorder=5)
        if g:
            ax.axvline(g * nc - 0.5, color=SPINE, lw=0.8)
        ax.text(g * nc + nc / 2 - 0.5, 1.572,
                f"group {g}\nseed {j['config']['rng_key_seeds'][g]}",
                color=INK, fontsize=8, ha="center", va="top")

    ax.text(63.4, ANCHOR["med"] + 0.006, "warm start = E1 v2d healthy-anchor "
            "cloud transported (med 1.4334); anchor 1.4330 [1.3995, 1.4685] ",
            color=INK, fontsize=8, ha="right", bbox=BOX)
    ax.text(63.4, BAND[0] + 0.006, "E2-G2 containment band [1.15, 2.0] ",
            color=INK, fontsize=8, ha="right", bbox=BOX)
    ax.text(63.4, 1.118, "corr-low certified 1.1032 (dotted);  E1 v3b "
            "migration shelf 1.1047 ", color=INK, fontsize=8, ha="right",
            va="bottom", bbox=BOX)
    ax.text(0.2, 1.035, f"{len(viol)}/64 chains end OUT of band "
            f"(frac. of kept draws outside: min {min(fo):.2f}, "
            f"median {float(np.median(fo)):.2f})\n"
            "open circle = warm-start $\\gamma$ of each chain;  "
            "bar = kept-draw 16–84%;  dot = chain median",
            color=INK, fontsize=8.6, bbox=BOX)
    ax.set_xlabel("chain (grouped by seed group)", color=INK, fontsize=9.5)
    ax.set_ylabel(r"$\gamma$", color=INK, fontsize=11)
    ax.set_xlim(-1.2, 64.2)

    ys = np.linspace(*ylim, 400)
    ki = gaussian_kde(init.reshape(-1))(ys)
    kk = gaussian_kde(gam.reshape(-1)[::40])(ys)
    axk.fill_betweenx(ys, 0, ki / ki.max(), color=GREY, alpha=0.35, lw=0,
                      label="init cloud (transported anchor)")
    axk.plot(kk / kk.max(), ys, color=INK, lw=1.7,
             label="kept draws, 64 chains pooled")
    for g in range(ng):
        axk.plot([1.02], [np.median(med[g])], marker="<", color=GC[g], ms=6,
                 ls="none")
    axk.set_xlim(0, 1.14)
    axk.set_xticks([])
    axk.legend(frameon=False, fontsize=7.6, labelcolor=INK, loc="center right")
    axk.set_title("density (scaled)", color=MUT, fontsize=8)

    fig.suptitle("E2 — Evan Odell's own 0.064125\" cutout (his prep, his PSF): "
                 "MCLMC chains migrate out of the containment band.  Warm "
                 "start at the 1.433 anchor cloud; EVERY chain (64/64, all 4 "
                 "seed groups) ends at $\\gamma \\approx$ 1.06–1.14   "
                 "[NO-QUOTE: E2-G1 convergence FAILED]",
                 color=INK, fontsize=10.3, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = FIGS / "e2_odell_migration.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {out}")


# =========================================================================== #
# Figure 2: O4 rescue — stage-1 vs stage-2 R-hat / gamma, both arms
# =========================================================================== #
def fig_o4_rescue():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j3, z3, gam3 = load_o4("v3b")
    j2, z2, gam2 = load_o4("v2d")
    e13 = json.loads((DATA / "mclmc_diag_v3b.json").read_text())
    e12 = json.loads((DATA / "mclmc_diag_v2d.json").read_text())

    # arm identity colors (fixed): v3b = slot-2 orange, v2d = slot-1 blue
    C3, C2 = GC[1], GC[0]
    thin = 5
    nd = gam3.shape[1]
    xs = np.arange(0, nd, thin)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.6), dpi=200)

    # ---- (0,0) v3b stage-2 gamma traces ----------------------------------- #
    ax = axes[0, 0]
    style(ax)
    for c in range(gam3.shape[0]):
        ax.plot(xs, gam3[c, ::thin], color=C3, lw=0.5, alpha=0.30)
    tail3 = j3["provenance"]["stage1_tail"]["tail_gamma"]
    ax.set_ylim(1.05, 1.17)
    ax.axhspan(BAND[0], 1.17, color=BAND_TINT, lw=0, zorder=0)
    ax.axhline(BAND[0], color=MUT, lw=1.1)
    ax.text(nd * 0.99, BAND[0] + 0.001, " containment band edge 1.15",
            color=INK, fontsize=7.5, ha="right", bbox=BOX)
    ax.axhline(tail3["med"], color=MUT, lw=1.0, ls=(0, (4, 3)))
    ax.text(nd * 0.99, tail3["med"] + 0.001,
            f" stage-1 tail median {tail3['med']:.4f}", color=INK,
            fontsize=7.5, ha="right", bbox=BOX)
    ax.axhline(CORR_LOW, color=MUT, lw=1.0, ls=(0, (1, 2)))
    ax.text(nd * 0.01, CORR_LOW + 0.001, " corr-low 1.1032", color=INK,
            fontsize=7.5, ha="left", bbox=BOX)
    p3 = j3["diagnostics"]["pooled_16"]
    ax.set_title("v3b arm — stage-2 $\\gamma$ trace (16 chains): stays AT the "
                 "shelf, but $\\hat{R}$ does NOT certify\n"
                 f"$\\hat{{R}}_\\gamma$ {p3['rhat']['mass:gamma']:.3f}, "
                 f"$\\hat{{R}}$_worst {p3['rhat_worst']:.4f}, min-ESS "
                 f"{p3['ess_min']:.0f}  →  falsifier F1 FIRED",
                 color=INK, fontsize=9.5)
    ax.set_ylabel(r"$\gamma$", color=INK, fontsize=10)
    ax.set_xlabel("kept draw", color=INK, fontsize=8.5)

    # ---- (0,1) v2d stage-2 gamma traces ----------------------------------- #
    ax = axes[0, 1]
    style(ax)
    for c in range(gam2.shape[0]):
        ax.plot(xs, gam2[c, ::thin], color=C2, lw=0.5, alpha=0.22)
    ax.axhspan(*E1_V2D["ci68"], color=GRID, alpha=0.55, lw=0)
    ax.axhline(E1_V2D["med"], color=MUT, lw=1.0, ls=(0, (4, 3)))
    ax.text(nd * 0.99, E1_V2D["ci68"][0] - 0.005, "E1 v2d quotable 1.4683 "
            "[1.4343, 1.5048]", color=INK, fontsize=7.5, ha="right",
            va="top", bbox=BOX)
    p2 = j2["diagnostics"]["pooled_16"]
    ax.set_title("v2d control — stage-2 converges cleanly at the anchor "
                 "family\n"
                 f"$\\hat{{R}}$_worst {p2['rhat_worst']:.4f}, min-ESS "
                 f"{p2['ess_min']:.0f}, $\\gamma$_med "
                 f"{j2['gamma']['pooled_quantiles']['0.5']:.4f}  →  "
                 "prediction HOLDS (0.002$\\sigma$ from stage 1)",
                 color=INK, fontsize=9.5)
    ax.set_ylabel(r"$\gamma$", color=INK, fontsize=10)
    ax.set_xlabel("kept draw", color=INK, fontsize=8.5)

    # ---- (1,0) R-hat dumbbells: stage 1 -> stage 2 ------------------------ #
    ax = axes[1, 0]
    style(ax)
    items = [
        ("v3b\nworst", e13["gates"]["E1_G1"]["rhat_worst"],
         p3["rhat_worst"], C3),
        ("v3b\n$\\gamma$", e13["diagnostics"]["pooled"]["rhat"]["mass:gamma"],
         p3["rhat"]["mass:gamma"], C3),
        ("v2d\nworst", e12["gates"]["E1_G1"]["rhat_worst"],
         p2["rhat_worst"], C2),
        ("v2d\n$\\gamma$", e12["diagnostics"]["pooled"]["rhat"]["mass:gamma"],
         p2["rhat"]["mass:gamma"], C2),
    ]
    for i, (lab, r1, r2, col) in enumerate(items):
        ax.plot([i, i], [r1, r2], color=col, lw=1.6, alpha=0.8, zorder=3)
        ax.plot([i], [r1], "o", ms=7, mfc="white", mec=col, mew=1.6,
                ls="none", zorder=4)
        ax.plot([i], [r2], "o", ms=7, color=col, mec="white", mew=0.8,
                ls="none", zorder=5)
        if r1 - r2 > 0.01:
            ax.annotate("", xy=(i, r2), xytext=(i, r1 - (r1 - r2) * 0.25),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
            ax.text(i + 0.13, r2, f"{r2:.4f}", color=INK, fontsize=7.8,
                    va="center")
            ax.text(i + 0.13, r1, f"{r1:.4f}", color=MUT, fontsize=7.8,
                    va="center")
        else:                       # near-identical pair: split the labels
            ax.text(i + 0.13, r1 + 0.006, f"{r1:.4f}", color=MUT,
                    fontsize=7.8, va="bottom")
            ax.text(i + 0.13, r2 - 0.004, f"{r2:.4f}", color=INK,
                    fontsize=7.8, va="top")
    ax.axhline(RHAT_GATE, color=MUT, lw=1.2)
    ax.text(-0.42, RHAT_GATE + 0.004, f" gate {RHAT_GATE}", color=INK,
            fontsize=8, va="bottom", ha="left", bbox=BOX)
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_yticks([1.0, 1.01, 1.05, 1.1, 1.2, 1.4])
    ax.set_yticklabels(["1.00", "1.01", "1.05", "1.10", "1.20", "1.40"])
    ax.set_ylim(0.996, 1.47)
    ax.set_xticks(range(4))
    ax.set_xticklabels([t[0] for t in items], fontsize=8.5)
    ax.set_xlim(-0.5, 3.8)
    ax.set_title("split-$\\hat{R}$, stage 1 (open, 64 pooled chains) → "
                 "stage 2 (filled, 16 chains):\nthe rescue IMPROVES the v3b "
                 "diagnostics but leaves them over the gate",
                 color=INK, fontsize=9.5)
    ax.set_ylabel("rank-normalized split-$\\hat{R}$ (log)", color=INK,
                  fontsize=8.5)

    # ---- (1,1) gamma location: tail vs stage-2 ---------------------------- #
    ax = axes[1, 1]
    style(ax)
    tail2 = j2["provenance"]["stage1_tail"]["tail_gamma"]
    q3 = j3["gamma"]["pooled_quantiles"]
    q2 = j2["gamma"]["pooled_quantiles"]
    rows = [
        ("v3b stage-1 tail", tail3["med"], tail3["q16"], tail3["q84"], C3, "o", "white"),
        ("v3b stage-2", q3["0.5"], q3["0.16"], q3["0.84"], C3, "o", C3),
        ("v2d stage-1 tail", tail2["med"], tail2["q16"], tail2["q84"], C2, "o", "white"),
        ("v2d stage-2", q2["0.5"], q2["0.16"], q2["0.84"], C2, "o", C2),
    ]
    ax.axvspan(BAND[0], 1.62, color=BAND_TINT, lw=0, zorder=0)
    ax.axvline(BAND[0], color=MUT, lw=1.1)
    ax.axvline(ANCHOR["med"], color=MUT, lw=1.0, ls=(0, (4, 3)))
    ax.axvline(CORR_LOW, color=MUT, lw=1.0, ls=(0, (1, 2)))
    for i, (lab, m, lo, hi, col, mk, mfc) in enumerate(rows):
        y = 3 - i
        ax.plot([lo, hi], [y, y], color=col, lw=2.6, alpha=0.9)
        ax.plot([m], [y], mk, ms=7.5, mfc=mfc, mec=col, mew=1.6, ls="none")
        ax.text(m, y + 0.18, lab, color=INK, fontsize=8.2, ha="center")
    ax.text(1.087, 3.62, "$\\Delta\\gamma$ stage2$-$tail $= -0.019$",
            color=INK, fontsize=8, ha="center", bbox=BOX)
    ax.text(1.468, 1.62, "$\\Delta\\gamma = -0.0001$", color=INK,
            fontsize=8, ha="center", bbox=BOX)
    ax.text(BAND[0], -0.55, " band edge 1.15", color=INK, fontsize=7.5,
            ha="left", bbox=BOX)
    ax.text(CORR_LOW, -0.55, "corr-low 1.1032 ", color=INK, fontsize=7.5,
            ha="right", bbox=BOX)
    ax.text(ANCHOR["med"], -0.55, " anchor 1.4330", color=INK, fontsize=7.5,
            ha="left", bbox=BOX)
    ax.set_xlim(1.03, 1.62)
    ax.set_ylim(-0.8, 3.95)
    ax.set_yticks([])
    ax.set_xlabel(r"$\gamma$ (demonstration telemetry — NOT a science quote)",
                  color=INK, fontsize=9)
    ax.set_title("both arms equilibrate AT their stage-1 location:\n"
                 "preconditioning does not move the basin, it only "
                 "(partially) launders the diagnostics", color=INK,
                 fontsize=9.5)

    fig.suptitle("E2-O4 — Evan-style stage-2 rescue on OUR stored E1 chains "
                 "(16 × 5000+5000, his stated scale):  v3b improves "
                 "($\\hat{R}$_worst 1.379 → 1.188) but does NOT certify "
                 "(F1 fired: '$\\hat{R}$-blindness' NOT supported at these "
                 "settings);  v2d control clean", color=INK, fontsize=10.2,
                 y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = FIGS / "e2_o4_rescue.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {out}")


# =========================================================================== #
# metrics — data/e2_harvest.json (AFTER the plots have been LOOKED AT)
# =========================================================================== #
def metrics():
    import arviz as az

    rep = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               script="61_harvest_e2.py",
               checkpoints=["CAMPAIGN.md 2026-07-23 E2 design checkpoint "
                            "(O1/O2, frozen pre-run)",
                            "research/checkpoint_e2_o4.md (frozen pre-run)"],
               figures=["figs/e2_odell_migration.png",
                        "figs/e2_o4_rescue.png"],
               asserts={})
    A = rep["asserts"]

    # ---------------- O2: the odell fit ----------------------------------- #
    j, z, gam, init = load_odell()
    flat = gam.reshape(-1)
    for k in (0.025, 0.16, 0.5, 0.84, 0.975):
        ref = j["gamma"]["pooled_quantiles"][str(k)]
        assert abs(float(np.quantile(flat, k)) - ref) < 1e-9, (k, ref)
    A["odell_gamma_quantiles_recomputed_vs_json"] = "PASS (<1e-9, all 5)"
    r = float(az.rhat(az.convert_to_dataset(gam.reshape(-1, gam.shape[2])))
              ["x"].values)
    assert abs(r - j["diagnostics"]["pooled"]["rhat"]["mass:gamma"]) < 1e-6, r
    A["odell_gamma_rhat_arviz_rerun_vs_json"] = f"PASS ({r:.6f}, <1e-6)"

    g1 = j["gates"]["E1_G1"]
    viol = j["gates"]["E1_G2"]["violations"]
    fo = [v["frac_outside"] for v in viol]
    qs = {k: float(v) for k, v in j["gamma"]["pooled_quantiles"].items()}
    per_group = j["gamma"]["per_group"]
    pg_diag = [dict(rhat_worst=g["rhat_worst"],
                    rhat_worst_param=g["rhat_worst_param"],
                    ess_min=g["ess_min"])
               for g in j["diagnostics"]["per_group"]]
    init_med = float(np.median(init))
    dmig = init - np.median(gam, axis=2)          # (G,C) per-chain migration
    rep["O2_odell"] = dict(
        verdict=("FAIL E2-G1 + FAIL E2-G2 — NO-QUOTE + FINDING; "
                 "pre-registered interpretation FORK 2 FIRES"),
        product=("Evan Odell's own DESI-165 cutout, registered 0.064125\"/px "
                 "(fliplr parity), his PSF, O1-G-passed package md5 "
                 + j["provenance"]["odell_product_md5"]),
        gates=dict(
            E2_G1=dict(rhat_worst=g1["rhat_worst"],
                       rhat_worst_param=g1["rhat_worst_param"],
                       ess_min=g1["ess_min"],
                       ess_min_param=g1["ess_min_param"],
                       gamma_rhat=j["diagnostics"]["pooled"]["rhat"]["mass:gamma"],
                       gamma_ess=j["diagnostics"]["pooled"]["ess"]["mass:gamma"],
                       gates=dict(rhat=RHAT_GATE, ess=ESS_GATE),
                       passed=False),
            E2_G2=dict(band=BAND, n_out=len(viol), n_chains=64,
                       frac_outside_min=float(min(fo)),
                       frac_outside_median=float(np.median(fo)),
                       passed=False)),
        per_group_diagnostics=pg_diag,
        per_group_gamma_endpoints=per_group,
        gamma_pooled_quantiles_NOQUOTE=qs,
        init_cloud=dict(med=init_med,
                        lo=float(init.min()), hi=float(init.max()),
                        source="E1 v2d healthy-anchor x46 cloud transported "
                               "(KEYED); run-json warm gamma46 med 1.4334"),
        migration=dict(
            delta_gamma_pooled=float(qs["0.5"] - init_med),
            per_chain_init_minus_endpoint=dict(
                min=float(dmig.min()), med=float(np.median(dmig)),
                max=float(dmig.max())),
            endpoint_vs_corr_low=float(qs["0.5"] - CORR_LOW),
            endpoint_vs_e1_v3b_shelf=float(qs["0.5"] - 1.1047),
            statement=("all 64 chains left the transported 1.433 anchor "
                       "cloud during burn-in and equilibrated at the same "
                       "low-gamma shelf as the E1 v3b migration and the "
                       "corr-low posterior")),
        fork_readout=dict(
            fork1_native_anchor_robust="NO (chains left the anchor family)",
            fork2_low_drift_extends=(
                "FIRES: the migration extends to a SECOND resampled product "
                "(fourth pixelization, 0.064125\") AND a SECOND independent "
                "preparation pipeline (Evan's own drizzle/prep, his PSF) — "
                "matches Evan's independent 'very migratory chains' report"),
            fork3_other="not reached"),
        wall_h=j["wall_h"], wall_h_total=j["wall_h_total"],
        peak_mb=j["peak_mb"],
        health=dict(nonan_frac=j["nonan_frac"],
                    kernel_nonan_frac=j["kernel_nonan_frac"]))

    # ---------------- O4: both arms vs the checkpoint ---------------------- #
    o4 = {}
    for arm in ("v3b", "v2d"):
        ja, za, gama = load_o4(arm)
        flat = gama.reshape(-1)
        for k in (0.025, 0.16, 0.5, 0.84, 0.975):
            ref = ja["gamma"]["pooled_quantiles"][str(k)]
            assert abs(float(np.quantile(flat, k)) - ref) < 1e-9, (arm, k)
        r = float(az.rhat(az.convert_to_dataset(gama))["x"].values)
        assert abs(r - ja["diagnostics"]["pooled_16"]["rhat"]["mass:gamma"]) \
            < 1e-6, (arm, r)
        A[f"o4_{arm}_gamma_quantiles_recomputed_vs_json"] = "PASS (<1e-9)"
        A[f"o4_{arm}_gamma_rhat_arviz_rerun_vs_json"] = f"PASS ({r:.6f})"
        # stage-1 tail medians recomputed from the stored E1 npz
        e1z = np.load(DATA / f"mclmc_diag_{arm}.npz", allow_pickle=True)
        e1j = json.loads((DATA / f"mclmc_diag_{arm}.json").read_text())
        gi = list(e1z["mass_names"]).index("gamma")
        ngg = int(e1j["config"]["n_groups"])
        g_all = np.concatenate([e1z[f"mass_g{g}"][:, :, gi] for g in range(ngg)])
        tail = g_all[:, g_all.shape[1] // 2:].reshape(-1)
        tmed = float(np.median(tail))
        ref = ja["provenance"]["stage1_tail"]["tail_gamma"]["med"]
        assert abs(tmed - ref) < 1e-9, (arm, tmed, ref)
        A[f"o4_{arm}_stage1_tail_median_recomputed"] = f"PASS ({tmed:.6f})"
        o4[arm] = (ja, e1j)

    j3, e13 = o4["v3b"]
    j2, e12 = o4["v2d"]
    p3, p2 = (j3["diagnostics"]["pooled_16"], j2["diagnostics"]["pooled_16"])
    t3 = j3["provenance"]["stage1_tail"]["tail_gamma"]
    t2 = j2["provenance"]["stage1_tail"]["tail_gamma"]
    q3 = j3["gamma"]["pooled_quantiles"]
    q2 = j2["gamma"]["pooled_quantiles"]

    # O4-D2 sigma_comb per the checkpoint (two CI68 half-widths)
    hw_e1 = 0.5 * (E1_V2D["ci68"][1] - E1_V2D["ci68"][0])
    hw_s2 = 0.5 * (q2["0.84"] - q2["0.16"])
    sig_comb = float(np.hypot(hw_e1, hw_s2))
    d_ctrl = float(q2["0.5"] - E1_V2D["med"])

    rep["O4_rescue"] = dict(
        checkpoint="research/checkpoint_e2_o4.md",
        verdict=dict(
            O4_D1_blindness_v3b=(
                "NOT MET — FALSIFIER F1 FIRED: stage-2 R-hat_worst 1.1875 "
                ">= 1.01 and min-ESS 81 < 1000; no pristine-R-hat-at-the-"
                "shelf at 16 x 5000+5000. The rescue IMPROVES the "
                "diagnostics (R-hat_worst 1.379 -> 1.188, gamma R-hat "
                "1.261 -> 1.128) but does NOT certify. The strong "
                "'R-hat-blindness' claim is NOT supported at our settings "
                "— reported as loudly as a pass per the checkpoint."),
            F2_return_toward_band=(
                "did NOT fire: stage-2 gamma_med 1.0849 < 1.15 (moved "
                "0.019 FURTHER from the band edge) — the E1 migration "
                "interpretation stands"),
            O4_D2_control_v2d=(
                "MET: R-hat_worst 1.0029 < 1.01, min-ESS 8305 >= 1000, "
                f"gamma_med within {abs(d_ctrl) / sig_comb:.4f} sigma_comb "
                "of the E1 v2d posterior — control prediction HELD"),
            F3_control_fails="did not fire"),
        v3b=dict(
            stage1=dict(rhat_worst=e13["gates"]["E1_G1"]["rhat_worst"],
                        ess_min=e13["gates"]["E1_G1"]["ess_min"],
                        gamma_rhat=e13["diagnostics"]["pooled"]["rhat"]["mass:gamma"],
                        chains="64 pooled (4x16), 4000 kept"),
            stage1_tail=t3,
            stage2=dict(rhat_worst=p3["rhat_worst"],
                        rhat_worst_param=p3["rhat_worst_param"],
                        ess_min=p3["ess_min"],
                        ess_min_param=p3["ess_min_param"],
                        gamma_rhat=p3["rhat"]["mass:gamma"],
                        gamma_ess=p3["ess"]["mass:gamma"],
                        gamma_quantiles_NOQUOTE=q3,
                        split_half_rhat_worst=[
                            h["rhat_worst"]
                            for h in j3["diagnostics"]["split_halves_8x2"]],
                        band_frac_outside_all=1.0,
                        chains="16, 5000 kept (one seed group)"),
            improvement=dict(
                d_rhat_worst=float(p3["rhat_worst"]
                                   - e13["gates"]["E1_G1"]["rhat_worst"]),
                d_gamma_rhat=float(
                    p3["rhat"]["mass:gamma"]
                    - e13["diagnostics"]["pooled"]["rhat"]["mass:gamma"]),
                note=("ESS comparison stage1 vs stage2 is across different "
                      "chain counts (64 vs 16) — R-hat is the like-for-like "
                      "metric here")),
            delta_gamma_stage2_minus_tail=float(q3["0.5"] - t3["med"]),
            wall_h=j3["wall_h"]),
        v2d=dict(
            stage1=dict(rhat_worst=e12["gates"]["E1_G1"]["rhat_worst"],
                        ess_min=e12["gates"]["E1_G1"]["ess_min"],
                        gamma_rhat=e12["diagnostics"]["pooled"]["rhat"]["mass:gamma"],
                        chains="64 pooled (2x32), 4000 kept"),
            stage1_tail=t2,
            stage2=dict(rhat_worst=p2["rhat_worst"],
                        ess_min=p2["ess_min"],
                        gamma_rhat=p2["rhat"]["mass:gamma"],
                        gamma_ess=p2["ess"]["mass:gamma"],
                        gamma_quantiles_NOQUOTE=q2,
                        split_half_rhat_worst=[
                            h["rhat_worst"]
                            for h in j2["diagnostics"]["split_halves_8x2"]],
                        band_frac_outside_all=0.0,
                        chains="16, 5000 kept (one seed group)"),
            control_check=dict(delta_vs_e1=d_ctrl, sigma_comb=sig_comb,
                               n_sigma=float(abs(d_ctrl) / sig_comb),
                               e1_ref=E1_V2D),
            delta_gamma_stage2_minus_tail=float(q2["0.5"] - t2["med"]),
            wall_h=j2["wall_h"]),
        what_survives=(
            "supported, weaker statement: tail preconditioning moves the "
            "diagnostics most of the way toward the pass region while the "
            "chains stay AT the migrated shelf — a looser convergence "
            "threshold (e.g. R-hat < 1.2) WOULD have certified the "
            "unphysical solution; at the frozen gates (1.01/1000) the "
            "diagnostics still flag it. Evan's reported R-hat << 1.01 on "
            "his second stage is not reproduced at his stated scale on our "
            "stored chains."))

    rep["provenance"] = dict(
        odell_json_md5=md5(DATA / "mclmc_diag_odell.json"),
        odell_npz_md5=md5(DATA / "mclmc_diag_odell.npz"),
        o4_v3b_npz_md5=md5(DATA / "o4_v3b_stage2.npz"),
        o4_v2d_npz_md5=md5(DATA / "o4_v2d_stage2.npz"),
        e1_v3b_npz_md5=md5(DATA / "mclmc_diag_v3b.npz"),
        e1_v2d_npz_md5=md5(DATA / "mclmc_diag_v2d.npz"),
        warm_x46_md5=md5(DATA / "mclmc_warm_v2d_x46.npz"))
    rep["gpu_h_e2_fits"] = dict(
        odell=j["wall_h_total"], o4_v3b=j3["wall_h"], o4_v2d=j2["wall_h"],
        total=float(j["wall_h_total"] + j3["wall_h"] + j2["wall_h"]),
        tier="phoenix L4 free tier (no A100-h rows per checkpoint)")

    out = DATA / "e2_harvest.json"
    out.write_text(json.dumps(rep, indent=1))
    print(f"[metrics] wrote {out}")
    print(json.dumps(rep["O2_odell"]["gates"], indent=1))
    print(json.dumps(rep["O4_rescue"]["verdict"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["figs", "metrics"])
    a = ap.parse_args()
    if a.stage == "figs":
        fig_odell_migration()
        fig_o4_rescue()
    else:
        metrics()
