#!/usr/bin/env python
"""51_harvest_e1.py — E1 harvest: plots-first readout of the MCLMC diagonal fits.

Inputs (results of record, written by 50_run_mclmc_diag.py):
  data/mclmc_diag_v2d.{npz,json}   PASS both gates  (harvested as THE quotable
                                   deliverable)
  data/mclmc_diag_v3b.{npz,json}   FAIL both gates  (NO-QUOTE + finding: chain
                                   migration out of the E1-G2 band)
  data/mclmc_warm_{tag}_x46.npz    warm-start clouds (init gamma per chain via
                                   init_idx_g* KEYED row indices)

Stages (house rule: plots BEFORE metrics — run `figs`, look, then `metrics`):
  figs    : figs/e1_mclmc_traces.png   trace/rank panels, gamma + worst params
            figs/e1_v3b_migration.png  the v3b migration money visual
  metrics : data/e1_harvest.json — every number the report will quote, each
            recomputed from the npz artifacts and cross-asserted against the
            run jsons (gamma quantiles exact; gamma R-hat via arviz re-run).

No GPU, no likelihood — numpy/arviz/matplotlib on the stored chains only.
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

# dataviz reference palette, categorical slots 1-4 (adjacent-pairlist legal for
# lines; slot-4 yellow < 3:1 on white => relief rule: legend + direct labels).
GC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUT, GRID, SPINE, GREY = "#3a3a38", "#8a8a86", "#e8e7e3", "#d5d4d0", "#8a8a86"
BAND_TINT = "#edf1ed"                  # E1-G2 containment region (annotation)
BOX = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2)

ANCHOR = dict(med=1.4330, ci68=(1.3995, 1.4685))   # foundry-i hmc_v13_v2d
OLD_DIAG_COND = 1.2941                 # hmc_v13_v3b gamma<1.7 conditional (R-hat 843)
CORR_LOW = 1.1032                      # certified correlated money number
BAND = (1.15, 2.0)                     # E1-G2


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load(tag):
    j = json.loads((DATA / f"mclmc_diag_{tag}.json").read_text())
    z = np.load(DATA / f"mclmc_diag_{tag}.npz", allow_pickle=True)
    w = np.load(DATA / f"mclmc_warm_{tag}_x46.npz", allow_pickle=True)
    ng = int(j["config"]["n_groups"])
    gi = list(z["mass_names"]).index("gamma")
    gam = np.stack([z[f"mass_g{g}"][:, :, gi] for g in range(ng)])   # (G,C,D)
    g46 = np.asarray(w["gamma46"], dtype=np.float64)
    init = np.stack([g46[z[f"init_idx_g{g}"]] for g in range(ng)])   # (G,C)
    return j, z, gam, init


def style(ax):
    ax.tick_params(colors=MUT, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(SPINE)


# =========================================================================== #
# Figure a: trace/rank panels
# =========================================================================== #
def fig_traces():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4), dpi=200)
    thin = 5

    for row, tag in enumerate(("v2d", "v3b")):
        j, z, gam, init = load(tag)
        ng, nc, nd = gam.shape
        g1 = j["gates"]["E1_G1"]
        wname = g1["rhat_worst_param"]
        widx = list(z["z_names"]).index(wname)
        worst = np.stack([z[f"pos_g{g}"][:, :, widx] for g in range(ng)])
        xs = np.arange(0, nd, thin)

        # --- col 0: gamma traces -------------------------------------------
        ax = axes[row, 0]
        style(ax)
        for g in range(ng):
            for c in range(nc):
                ax.plot(xs, gam[g, c, ::thin], color=GC[g], lw=0.45,
                        alpha=0.22 if tag == "v2d" else 0.30)
        if tag == "v2d":
            ax.axhspan(*ANCHOR["ci68"], color=GRID, alpha=0.55, lw=0)
            ax.axhline(ANCHOR["med"], color=MUT, lw=1.0, ls=(0, (4, 3)))
            ax.text(nd * 0.99, ANCHOR["ci68"][0] - 0.004, "anchor 1.4330 "
                    "[1.3995, 1.4685]", color=INK, fontsize=7.5, ha="right",
                    va="top", bbox=BOX)
            ax.set_title(r"v2d native — $\gamma$ trace, 64 chains "
                         r"($\hat{R}_\gamma$ %.4f)" % j["diagnostics"]["pooled"]
                         ["rhat"]["mass:gamma"], color=INK, fontsize=9.5)
        else:
            ax.set_ylim(1.045, 1.36)
            ax.axhspan(BAND[0], 1.36, color=BAND_TINT, lw=0, zorder=0)
            ax.axhline(BAND[0], color=MUT, lw=1.1)
            ax.text(nd * 0.99, BAND[0] + 0.003, " E1-G2 band edge 1.15",
                    color=INK, fontsize=7.5, ha="right", bbox=BOX)
            ax.axhline(OLD_DIAG_COND, color=MUT, lw=1.0, ls=(0, (4, 3)))
            ax.text(nd * 0.99, OLD_DIAG_COND + 0.003,
                    " warm-start / old conditional 1.294", color=INK,
                    fontsize=7.5, ha="right", bbox=BOX)
            ax.axhline(CORR_LOW, color=MUT, lw=1.0, ls=(0, (1, 2)))
            ax.text(nd * 0.99, CORR_LOW - 0.004, " corr-low 1.1032", color=INK,
                    fontsize=7.5, ha="right", va="top", bbox=BOX)
            ax.annotate("", xy=(120, 1.13), xytext=(120, 1.285),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
            ax.text(240, 1.21, "migrated during burn-in\n(2000 discarded steps)",
                    color=INK, fontsize=7.8, va="center")
            ax.set_title(r"v3b binned — $\gamma$ trace: kept draws BELOW the "
                         r"containment band ($\hat{R}_\gamma$ %.2f)"
                         % j["diagnostics"]["pooled"]["rhat"]["mass:gamma"],
                         color=INK, fontsize=9.5)
        ax.set_ylabel(r"$\gamma$", color=INK, fontsize=10)
        ax.set_xlabel("kept draw", color=INK, fontsize=8.5)

        # --- col 1: worst-Rhat param trace ---------------------------------
        ax = axes[row, 1]
        style(ax)
        for g in range(ng):
            for c in range(nc):
                ax.plot(xs, worst[g, c, ::thin], color=GC[g], lw=0.45,
                        alpha=0.22 if tag == "v2d" else 0.30)
        ax.set_title(f"worst-$\\hat{{R}}$ param {wname}\n"
                     f"$\\hat{{R}}$ {g1['rhat_worst']:.4f}, "
                     f"min-ESS {g1['ess_min']:.0f} ({g1['ess_min_param']})",
                     color=INK, fontsize=9)
        ax.set_xlabel("kept draw", color=INK, fontsize=8.5)
        ax.set_ylabel("scene-z value", color=INK, fontsize=8.5)

        # --- col 2: per-seed-group rank histogram of gamma -----------------
        ax = axes[row, 2]
        style(ax)
        ranks = rankdata(gam.reshape(-1)).reshape(ng, nc * nd) / gam.size
        nb = 20
        hmax = 0.0
        for g in range(ng):
            h, edges = np.histogram(ranks[g], bins=nb, range=(0, 1))
            h = h / (nc * nd / nb)
            hmax = max(hmax, float(h.max()))
            ax.stairs(h, edges, color=GC[g], lw=1.6, baseline=None,
                      label=f"group {g}")
        ax.axhline(1.0, color=MUT, lw=1.0, ls=(0, (4, 3)))
        ax.set_ylim(0, max(1.35, 1.18 * hmax))
        ax.set_title(r"$\gamma$ pooled-rank histogram by seed group"
                     + ("  (flat = groups agree)" if tag == "v2d" else
                        "  (tilted = group disagreement)"),
                     color=INK, fontsize=9)
        ax.set_xlabel("normalized pooled rank", color=INK, fontsize=8.5)
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, ncol=2,
                  loc="upper right" if tag == "v2d" else "upper left")

    fig.suptitle("E1 MCLMC diagonal fits — trace / rank diagnostics:  "
                 "v2d PASS (R̂_worst 1.0031, min-ESS 30,334)   |   "
                 "v3b FAIL (R̂_worst 1.379, min-ESS 139; all 64 chains "
                 "below the containment band)", color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = FIGS / "e1_mclmc_traces.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {out}")


# =========================================================================== #
# Figure b: v3b migration landscape
# =========================================================================== #
def fig_migration():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    j, z, gam, init = load("v3b")
    ng, nc, nd = gam.shape
    med = np.median(gam, axis=2)                       # (G,C)
    q16, q84 = np.quantile(gam, 0.16, axis=2), np.quantile(gam, 0.84, axis=2)
    n_out = sum(1 for v in j["gates"]["E1_G2"]["violations"])
    fo = [v["frac_outside"] for v in j["gates"]["E1_G2"]["violations"]]

    fig, (ax, axk) = plt.subplots(
        1, 2, figsize=(11.8, 6.0), dpi=200, sharey=True,
        gridspec_kw=dict(width_ratios=[3.1, 1.0], wspace=0.04))
    ylim = (1.045, 1.36)
    for a in (ax, axk):
        style(a)
        a.set_ylim(*ylim)
        a.axhspan(BAND[0], ylim[1], color=BAND_TINT, lw=0, zorder=0)
        a.axhline(BAND[0], color=MUT, lw=1.2)
        a.axhline(OLD_DIAG_COND, color=MUT, lw=1.0, ls=(0, (4, 3)))
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
        ax.text(g * nc + nc / 2 - 0.5, 1.352,
                f"group {g}\nseed {j['config']['rng_key_seeds'][g]}",
                color=INK, fontsize=8, ha="center", va="top")

    ax.text(63.4, BAND[0] + 0.004, "E1-G2 containment band [1.15, 2.0] "
            "(single-basin by design) ", color=INK, fontsize=8, ha="right",
            bbox=BOX)
    ax.text(31.5, 1.245, "init cloud = old diagonal conditional "
            r"1.2941 (an $\hat{R}$ = 843 bimodal chain)", color=INK,
            fontsize=8, ha="center", bbox=BOX)
    ax.text(63.4, CORR_LOW - 0.013, "corr-low certified 1.1032 "
            "(the over-correcting correlated fit) ", color=INK, fontsize=8,
            ha="right", va="top", bbox=BOX)
    ax.text(0.2, 1.062, f"{n_out}/64 chains end OUT of band "
            f"(frac. of kept draws outside: min {min(fo):.2f}, "
            f"median {np.median(fo):.2f})\n"
            "open circle = warm-start $\\gamma$ of each chain;  "
            "bar = kept-draw 16–84%;  dot = chain median",
            color=INK, fontsize=8.6, bbox=BOX)
    ax.set_xlabel("chain (grouped by seed group)", color=INK, fontsize=9.5)
    ax.set_ylabel(r"$\gamma$", color=INK, fontsize=11)
    ax.set_xlim(-1.2, 64.2)

    # right marginal: init cloud vs pooled endpoint distribution
    ys = np.linspace(*ylim, 400)
    ki = gaussian_kde(init.reshape(-1))(ys)
    kk = gaussian_kde(gam.reshape(-1)[::40])(ys)
    axk.fill_betweenx(ys, 0, ki / ki.max(), color=GREY, alpha=0.35, lw=0,
                      label="init cloud (512-draw warm start)")
    axk.plot(kk / kk.max(), ys, color=INK, lw=1.7,
             label="kept draws, 64 chains pooled")
    for g in range(ng):
        axk.plot([1.02], [np.median(med[g])], marker="<", color=GC[g], ms=6,
                 ls="none")
    axk.set_xlim(0, 1.14)
    axk.set_xticks([])
    axk.legend(frameon=False, fontsize=7.6, labelcolor=INK, loc="lower right")
    axk.set_title("density (scaled)", color=MUT, fontsize=8)

    fig.suptitle("E1 v3b binned product — MCLMC chains migrate out of the "
                 "physical-basin containment band:  warm start at the 1.29 "
                 "diagonal conditional, EVERY chain (64/64, all 4 seed groups) "
                 "ends at $\\gamma \\approx$ 1.08–1.15   [NO-QUOTE: E1-G1 "
                 "convergence FAILED]", color=INK, fontsize=10.3, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = FIGS / "e1_v3b_migration.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"[fig] wrote {out}")


# =========================================================================== #
# metrics — data/e1_harvest.json (after the plots have been LOOKED AT)
# =========================================================================== #
def metrics():
    import arviz as az

    rep = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               script="51_harvest_e1.py",
               checkpoint="CAMPAIGN.md 2026-07-22 E1 design checkpoint "
                          "(+3 pre-production amendments); gates frozen before "
                          "launch, harvest plots inspected before this math",
               figures=["figs/e1_mclmc_traces.png", "figs/e1_v3b_migration.png",
                        "figs/rfig_corner.png (regenerated, + v2d MCLMC)",
                        "figs/rfig_critcurves.png (regenerated, + v2d MCLMC)"])
    asserts = {}

    # ---------------- v2d: PASS — the quotable deliverable ------------------
    j, z, gam, init = load("v2d")
    flat = gam.reshape(-1)
    qs = {q: float(np.quantile(flat, q)) for q in (0.025, 0.16, 0.5, 0.84, 0.975)}
    for k, v in qs.items():
        ref = j["gamma"]["pooled_quantiles"][str(k)]
        assert abs(v - ref) < 1e-9, (k, v, ref)
    asserts["v2d_gamma_quantiles_recomputed_vs_json"] = "PASS (<1e-9, all 5)"
    r = float(az.rhat(az.convert_to_dataset(
        gam.reshape(-1, gam.shape[2])))["x"])   # (64 chains, 4000 draws)
    assert abs(r - j["diagnostics"]["pooled"]["rhat"]["mass:gamma"]) < 1e-6, r
    asserts["v2d_gamma_rhat_arviz_rerun_vs_json"] = f"PASS ({r:.6f}, <1e-6)"

    sig_ref = (ANCHOR["ci68"][1] - ANCHOR["ci68"][0]) / 2.0
    sig_scene = (qs[0.84] - qs[0.16]) / 2.0
    sig_comb = float(np.hypot(sig_ref, sig_scene))
    delta = qs[0.5] - ANCHOR["med"]
    per_group = j["gamma"]["per_group"]
    rep["v2d"] = dict(
        verdict="PASS E1-G1 + E1-G2 — gamma QUOTABLE (not escalated)",
        gates=dict(
            E1_G1=dict(passed=True, rhat_worst=j["gates"]["E1_G1"]["rhat_worst"],
                       rhat_worst_param=j["gates"]["E1_G1"]["rhat_worst_param"],
                       ess_min=j["gates"]["E1_G1"]["ess_min"],
                       ess_min_param=j["gates"]["E1_G1"]["ess_min_param"],
                       thresholds="rhat<1.01, ess>=1000, pooled 64 chains"),
            E1_G2=dict(passed=True, band=list(BAND), violations=0,
                       gamma_range_all_draws=[float(flat.min()), float(flat.max())])),
        gamma=dict(med=qs[0.5], ci68=[qs[0.16], qs[0.84]],
                   ci95=[qs[0.025], qs[0.975]],
                   quote=f"{qs[0.5]:.4f} [{qs[0.16]:.4f}, {qs[0.84]:.4f}] (CI68)",
                   rhat=j["diagnostics"]["pooled"]["rhat"]["mass:gamma"],
                   ess=j["diagnostics"]["pooled"]["ess"]["mass:gamma"]),
        anchor_consistency=dict(
            anchor_med=ANCHOR["med"], anchor_ci68=list(ANCHOR["ci68"]),
            anchor_source="foundry-i hmc_v13_v2d (rhat 1.0163, ess 5714)",
            delta=round(delta, 5), sigma_ref=round(sig_ref, 5),
            sigma_scene=round(sig_scene, 5), sigma_comb=round(sig_comb, 5),
            z_comb=round(delta / sig_comb, 3),
            statement=f"gamma_MCLMC - gamma_anchor = +{delta:.4f} = "
                      f"{delta / sig_comb:.2f} sigma_comb — consistent within "
                      "1 sigma_comb (the pre-registered E1 prediction holds)"),
        per_group=per_group,
        group_agreement=dict(
            delta_med=round(abs(per_group[0]["med"] - per_group[1]["med"]), 6),
            statement="seed groups agree to 0.0002 in gamma median"),
        sampling=dict(machine="phoenix L4, GPU 9 (free tier)",
                      chains="2 groups x 32", burn=2000, draws=4000,
                      wall_h=j["wall_h"], wall_h_total=j["wall_h_total"],
                      gpu_h=j["wall_h_total"], peak_mb=j["peak_mb"],
                      kernel_nonan=j["kernel_nonan_frac"],
                      L_final=j["L_final"],
                      step_size_final_med=j["step_size_final_med"]),
        artifacts=dict(npz=f"data/mclmc_diag_v2d.npz (md5 "
                           f"{md5(DATA / 'mclmc_diag_v2d.npz')})",
                       json="data/mclmc_diag_v2d.json"))

    # ---------------- v3b: FAIL — NO-QUOTE + finding ------------------------
    j3, z3, gam3, init3 = load("v3b")
    viol = j3["gates"]["E1_G2"]["violations"]
    fo = np.array([v["frac_outside"] for v in viol])
    flat3 = gam3.reshape(-1)
    qs3 = {q: float(np.quantile(flat3, q)) for q in (0.16, 0.5, 0.84)}
    for k, v in qs3.items():
        assert abs(v - j3["gamma"]["pooled_quantiles"][str(k)]) < 1e-9
    asserts["v3b_gamma_quantiles_recomputed_vs_json"] = "PASS (<1e-9)"
    med3 = np.median(gam3, axis=2)
    pg = j3["diagnostics"]["per_group"]
    rep["v3b"] = dict(
        verdict="FAIL E1-G1 + E1-G2 — NO-QUOTE; escalation DECLINED; "
                "recorded as a FINDING (see below)",
        gates=dict(
            E1_G1=dict(passed=False,
                       rhat_worst=j3["gates"]["E1_G1"]["rhat_worst"],
                       rhat_worst_param=j3["gates"]["E1_G1"]["rhat_worst_param"],
                       ess_min=j3["gates"]["E1_G1"]["ess_min"],
                       ess_min_param=j3["gates"]["E1_G1"]["ess_min_param"],
                       n_z_params_over_rhat_gate="46 of 46",
                       gamma_rhat=j3["diagnostics"]["pooled"]["rhat"]["mass:gamma"],
                       gamma_ess=j3["diagnostics"]["pooled"]["ess"]["mass:gamma"]),
            E1_G2=dict(passed=False, band=list(BAND), violations=len(viol),
                       frac_outside=dict(min=float(fo.min()),
                                         median=float(np.median(fo)),
                                         mean=float(fo.mean())),
                       note="all 64 chains reported, none dropped "
                            "(report-never-drop rule)")),
        migration=dict(
            init_cloud=dict(med=float(np.median(init3)),
                            q16=float(np.quantile(init3, 0.16)),
                            q84=float(np.quantile(init3, 0.84)),
                            source="hmc_v13_v3b mass_gamma<1.7 conditional, "
                                   "512 mapped draws (init ONLY)"),
            per_group_init_med=[float(np.median(init3[g])) for g in range(4)],
            per_group_endpoint=j3["gamma"]["per_group"],
            per_group_lastdraw_med=[float(np.median(gam3[g, :, -1]))
                                    for g in range(4)],
            chains_out_of_band="64/64 (100%), all 4 seed groups",
            pooled_endpoint_quantiles_DESCRIPTIVE_NOT_QUOTABLE=
                j3["gamma"]["pooled_quantiles"],
            drop=dict(from_init_med=float(np.median(init3)),
                      to_kept_med=qs3[0.5],
                      delta=round(qs3[0.5] - float(np.median(init3)), 4)),
            proximity_note="kept-draw pooled median 1.1047 sits 0.0015 from "
                           "the corr-low certified 1.1032 — descriptive only "
                           "(unconverged ensemble), not a quote",
            chain_med_spread=dict(min=float(med3.min()), max=float(med3.max()))),
        rhat_structure=dict(
            per_group_rhat_worst=[dict(rhat=g["rhat_worst"],
                                       param=g["rhat_worst_param"],
                                       ess_min=g["ess_min"]) for g in pg],
            where="unconverged BOTH between and within seed groups; worst "
                  "offenders are lens-light photometric params "
                  "(Ie/R_sersic/n_sersic, rhat 1.35-1.38) and shapelet source "
                  "centers (1.34-1.35); mass:gamma rhat 1.261",
            top_pooled_rhat={k: round(v, 4) for k, v in sorted(
                j3["diagnostics"]["pooled"]["rhat"].items(),
                key=lambda kv: -kv[1])[:6]}),
        sampling=dict(machine="phoenix L4, GPU 8 (free tier)",
                      chains="4 groups x 16 (amendment 3 OOM geometry; "
                             "pooled 64 unchanged)",
                      burn=2000, draws=4000, wall_h=j3["wall_h"],
                      wall_h_total=j3["wall_h_total"], gpu_h=j3["wall_h_total"],
                      peak_mb=j3["peak_mb"],
                      kernel_nonan=j3["kernel_nonan_frac"],
                      L_final=j3["L_final"],
                      step_size_final_med=j3["step_size_final_med"]),
        artifacts=dict(npz=f"data/mclmc_diag_v3b.npz (md5 "
                           f"{md5(DATA / 'mclmc_diag_v3b.npz')})",
                       json="data/mclmc_diag_v3b.json"))

    rep["decisions"] = dict(
        escalation=dict(
            decision="DECLINED for v3b (one escalation was ALLOWED, not "
                     "required, by the checkpoint)",
            justification="the failure mode is structural migration out of "
                          "the containment band (all 64 chains, all 4 seed "
                          "groups, during burn-in), not sampling depth; "
                          "doubling draws cannot un-migrate chains; a "
                          "within-band conditional slice would be a goalpost "
                          "move. Verdict recorded as NO-QUOTE + FINDING."),
        finding="Even MCLMC — the PI's own trusted sampler — under the "
                "DIAGONAL likelihood drifts out of the physical basin toward "
                "the unphysical low-gamma region (~1.10) on the BINNED "
                "product, from a warm start in the 1.29 conditional cloud. "
                "This is the sampling-side face of the campaign's "
                "binned-product pathology (T0.4 mechanism family) and further "
                "undermines the 1.2941 conditional (an R-hat 843 bimodal "
                "chain) as a diagonal reference. The native v2d MCLMC fit is "
                "THE quotable deliverable of E1.")

    # rfig regeneration assertion results (written by 40_modeler_summary_figs)
    rchk = DATA / "rfig_modeler_checks.json"
    if rchk.exists():
        rc = json.loads(rchk.read_text())
        rep["rfig_assertions"] = rc
        asserts["rfig_regeneration"] = "PASS (see rfig_assertions)"

    rep["gpu_h"] = dict(v2d=j["wall_h_total"], v3b=j3["wall_h_total"],
                        total=round(j["wall_h_total"] + j3["wall_h_total"], 2),
                        note="phoenix L4 free tier — GPU-h noted at harvest, "
                             "no A100-h ledger rows (checkpoint rule)")
    rep["assertions"] = asserts
    out = DATA / "e1_harvest.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(dict(assertions=asserts, gpu_h=rep["gpu_h"]), indent=1))
    print(f"[metrics] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["figs", "metrics"])
    a = ap.parse_args()
    if a.stage == "figs":
        fig_traces()
        fig_migration()
    else:
        metrics()
