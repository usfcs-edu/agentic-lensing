"""P1 harvest: evaluate the pre-registered T0.2 (seed-repeat) and T0.3
(companion-mask) gates from the five completed Perlmutter jobs 55951082-86.

Runs under the OLD validated venv (/raid/benson/.venvs/cgl/bin/python) because
the z->physical gamma transform is the OLD stack's marg-model bijector
(cgl.e2.build_target('v3b').model.to_physical_mass) -- the exact extraction
convention of the P1c money numbers (weighted quantiles via
cgl.e2._weighted_quantile inside run_correlated_smc; the per-run JSON summaries
ARE that extraction's output and are treated as authoritative). CPU-only: only
the bijector is evaluated, never the lensing likelihood.

Inputs:
  NEW  data/results-perlmutter/e2_v3b_{low,steep}_smc_seed{3,4}.{npz,json}
       data/results-perlmutter/e2_v3b_low_smc_compmask.{npz,json}
  OLD  ../claude-giga-lens/data/results/e2_v3b_low_smc_canary_fix.{npz,json}   (seed 2 production)
       ../claude-giga-lens/data/results/e2_v3b_steep_smc_canary_p96.{npz,json} (seed 2 production)

Outputs (plots BEFORE metrics, per house rule):
  figs/t02_seed_overlay.png, figs/t03_compmask_overlay.png,
  data/t02_t03_gate_eval.json
"""
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path("/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens-linus")
OLD = Path("/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens")
NEW = ROOT / "data" / "results-perlmutter"
OLDR = OLD / "data" / "results"
FIGS = ROOT / "figs"

RUNS = {
    ("low", 2): (OLDR / "e2_v3b_low_smc_canary_fix", "low"),
    ("low", 3): (NEW / "e2_v3b_low_smc_seed3", "low"),
    ("low", 4): (NEW / "e2_v3b_low_smc_seed4", "low"),
    ("steep", 2): (OLDR / "e2_v3b_steep_smc_canary_p96", "steep"),
    ("steep", 3): (NEW / "e2_v3b_steep_smc_seed3", "steep"),
    ("steep", 4): (NEW / "e2_v3b_steep_smc_seed4", "steep"),
    ("compmask", 2): (NEW / "e2_v3b_low_smc_compmask", "low"),
}

# sacct 2026-07-15 (harvest session): elapsed, single A100 shared-QOS each
SACCT = {
    "55951082": "00:30:40", "55951083": "00:30:30", "55951084": "00:17:39",
    "55951085": "00:18:23", "55951086": "00:35:40",
}


def load_run(stem, basin):
    js = json.load(open(stem.with_suffix(".json")))
    d = np.load(stem.with_suffix(".npz"), allow_pickle=True)
    parts = np.asarray(d[f"{basin}_particles"], dtype=np.float64)
    return js["basins"][basin], parts, js


def main():
    from cgl import e2

    target = e2.build_target("v3b")   # bijector only; no likelihood evals

    summ, gammas, sanity = {}, {}, {}
    for (name, seed), (stem, basin) in RUNS.items():
        b, parts, js = load_run(stem, basin)
        g = np.asarray(target.model.to_physical_mass(parts)["gamma"], dtype=np.float64)
        gammas[(name, seed)] = g
        # cross-check: equal-weight particle median vs the run's own weighted median
        med_check = float(np.median(g)) - b["gamma_median"]
        summ[(name, seed)] = b
        sanity[(name, seed)] = dict(
            n_lambda_steps=b["n_lambda_steps"], ess_weight=b["ess_weight"],
            n_unique=b["n_unique"], n_floored_q=b["n_floored_q"],
            frac_gamma_gt_split=b["frac_gamma_gt_split"], wall_s=b["wall_s"],
            n_particles=b["n_particles"],
            n_unique_gamma_saved=int(len(np.unique(g))),
            equalweight_median_minus_weighted=med_check,
            q16_equals_median=bool(b["gamma_q16"] == b["gamma_median"]),
        )
        print(f"[{name} seed{seed}] gamma_med={b['gamma_median']:.6f} "
              f"logZ={b['logZ']:.3f} w_ess={b['ess_weight']:.1f}/{b['n_particles']} "
              f"n_uniq={b['n_unique']} lam_steps={b['n_lambda_steps']} "
              f"|eqw_med-w_med|={abs(med_check):.5f}")

    # ---------------- plots BEFORE metrics ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    C = {2: "#2a78d6", 3: "#008300", 4: "#e87ba4"}   # dataviz categorical slots 1-3
    INK, MUT = "#3a3a38", "#8a8a86"

    def kde_line(ax, x, color, label):
        kde = gaussian_kde(x)
        lo, hi = x.min(), x.max()
        pad = 0.15 * (hi - lo + 1e-9)
        xs = np.linspace(lo - pad, hi + pad, 400)
        ax.plot(xs, kde(xs), color=color, lw=2, label=label)
        ax.plot([np.median(x)], [0], marker="|", ms=14, mew=2, color=color,
                clip_on=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    for ax, basin, ttl in zip(axes, ["low", "steep"],
                              ["v3b-low (p128)", "v3b-steep (p96)"]):
        for seed in (2, 3, 4):
            lab = f"seed {seed}" + (" (production)" if seed == 2 else "")
            kde_line(ax, gammas[(basin, seed)], C[seed], lab)
        ax.set_title(f"T0.2 {ttl}: equal-weight SMC particles", color=INK, fontsize=10)
        ax.set_xlabel(r"$\gamma$ (physical)", color=INK)
        ax.tick_params(colors=MUT, labelsize=8)
        ax.grid(True, color="#e8e7e3", lw=0.7)
        for s in ax.spines.values():
            s.set_color("#d5d4d0")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    axes[0].set_ylabel("density", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "t02_seed_overlay.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    kde_line(ax, gammas[("low", 2)], "#2a78d6", "production (full keep_w, seed 2)")
    kde_line(ax, gammas[("compmask", 2)], "#008300", "companion-eroded keep_w (seed 2)")
    m2 = summ[("low", 2)]["gamma_median"]
    mc = summ[("compmask", 2)]["gamma_median"]
    ax.axvspan(m2 - 0.024, m2 + 0.024, color="#f0efec", zorder=0,
               label=r"pre-registered $\pm 3\sigma_{stat}$ band")
    ax.axvline(m2, color="#2a78d6", lw=1, ls="--")
    ax.axvline(mc, color="#008300", lw=1, ls="--")
    ax.set_title(f"T0.3 companion-mask discriminator: "
                 f"$\\Delta\\gamma_{{med}}$ = {mc - m2:+.4f} (band ±0.024)",
                 color=INK, fontsize=10)
    ax.set_xlabel(r"$\gamma$ (physical)", color=INK)
    ax.set_ylabel("density", color=INK)
    ax.tick_params(colors=MUT, labelsize=8)
    ax.grid(True, color="#e8e7e3", lw=0.7)
    for s in ax.spines.values():
        s.set_color("#d5d4d0")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "t03_compmask_overlay.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {FIGS}/t02_seed_overlay.png + t03_compmask_overlay.png")

    # ---------------- T0.2 gates ----------------
    seeds = (2, 3, 4)
    gate = {}
    for basin in ("low", "steep"):
        gm = np.array([summ[(basin, s)]["gamma_median"] for s in seeds])
        lz = np.array([summ[(basin, s)]["logZ"] for s in seeds])
        gate[basin] = dict(
            gamma_median_by_seed={s: float(v) for s, v in zip(seeds, gm)},
            logZ_by_seed={s: float(v) for s, v in zip(seeds, lz)},
            sigma_seed_gamma=float(np.std(gm, ddof=1)),
            sigma_seed_logZ=float(np.std(lz, ddof=1)),
            gamma_seed_mean=float(gm.mean()),
        )
    dlz = {s: summ[("steep", s)]["logZ"] - summ[("low", s)]["logZ"] for s in seeds}
    sig_dlz = float(np.sqrt(gate["low"]["sigma_seed_logZ"] ** 2
                            + gate["steep"]["sigma_seed_logZ"] ** 2))
    # n=3 chi-dist sampling error on a sample sigma: SD(s)/sigma = sqrt(1-c4^2), c4(3)=sqrt(pi)/2
    chi_frac = float(np.sqrt(1.0 - np.pi / 4.0))

    t02 = dict(
        basins=gate,
        dlogZ_steep_minus_low_by_seed={s: float(v) for s, v in dlz.items()},
        dlogZ_sign_seed_stable=bool(all(v < 0 for v in dlz.values())),
        sigma_seed_dlogZ=sig_dlz,
        n_seeds=3, chi_dist_frac_error_on_sigma=chi_frac,
        gates=dict(
            gamma_low=dict(value=gate["low"]["sigma_seed_gamma"], thresh=0.008,
                           passed=bool(gate["low"]["sigma_seed_gamma"] <= 0.008)),
            gamma_steep=dict(value=gate["steep"]["sigma_seed_gamma"], thresh=0.008,
                             passed=bool(gate["steep"]["sigma_seed_gamma"] <= 0.008)),
            dlogZ=dict(value=sig_dlz, thresh=5.0, passed=bool(sig_dlz < 5.0)),
            kill_sigma_gamma_gt_0p024=bool(
                max(gate["low"]["sigma_seed_gamma"],
                    gate["steep"]["sigma_seed_gamma"]) > 0.024),
        ),
    )

    # ---------------- T0.3 ----------------
    p, c = summ[("low", 2)], summ[("compmask", 2)]
    shift = c["gamma_median"] - p["gamma_median"]
    t03 = dict(
        gamma_median_production=p["gamma_median"],
        gamma_median_compmask=c["gamma_median"],
        shift=float(shift),
        band_pre_registered=0.024,
        moves_upward_ge_band=bool(shift >= 0.024),
        verdict=("COMPANION-MISFIT TRANSMISSION REAL" if shift >= 0.024
                 else "COMPANION EXONERATED (static within 0.024)"),
        sigma_stat_production=p["gamma_sigma"], sigma_stat_compmask=c["gamma_sigma"],
        logZ_production=p["logZ"], logZ_compmask=c["logZ"],
        logZ_shift_NOT_COMPARABLE="keep_w 9273 -> 8247 (-1026 whitened dof): "
                                  "different data => logZ not comparable across runs",
        n_keep_w=dict(production=9273, compmask=8247),
        ess=dict(production=p["ess_weight"], compmask=c["ess_weight"]),
        n_lambda_steps=dict(production=p["n_lambda_steps"], compmask=c["n_lambda_steps"]),
    )

    # ---------------- finalized downstream thresholds ----------------
    sg_low = gate["low"]["sigma_seed_gamma"]
    sigma_tot = float(np.sqrt(0.008 ** 2 + sg_low ** 2))
    finalized = dict(
        sigma_seed_gamma_low=sg_low, sigma_seed_gamma_steep=gate["steep"]["sigma_seed_gamma"],
        sigma_seed_dlogZ=sig_dlz,
        sigma_tot_gamma_low=sigma_tot,
        B5_G2_dlogZ_thresh="3*sqrt(sigma_boot^2 + %.3f^2) nats (floor %.2f at sigma_boot=0)"
                           % (sig_dlz, 3 * sig_dlz),
        T11_sigma_floor=sigma_tot,
        T11_exonerate_band=float(3 * sigma_tot),
        T11_confirm_thresh=float(-3 * 3 * sigma_tot),
        X1_G1="RETIRED with P4 (D7); ~15-nat placeholder never finalized",
    )

    out = dict(
        inputs={f"{k[0]}_seed{k[1]}": str(v[0]) + ".{npz,json}" for k, v in RUNS.items()},
        sacct_elapsed=SACCT,
        t02=t02, t03=t03, finalized_downstream=finalized,
        sanity={f"{k[0]}_seed{k[1]}": v for k, v in sanity.items()},
    )
    outp = ROOT / "data" / "t02_t03_gate_eval.json"
    json.dump(out, open(outp, "w"), indent=1, default=float)
    print(f"\n[gate-eval] wrote {outp}")

    print("\n===== T0.2 =====")
    for b in ("low", "steep"):
        g = gate[b]
        print(f"  {b}: gamma medians {g['gamma_median_by_seed']} -> "
              f"sigma_seed(gamma)={g['sigma_seed_gamma']:.5f} "
              f"(gate<=0.008 {'PASS' if g['sigma_seed_gamma'] <= 0.008 else 'FAIL'}); "
              f"sigma_seed(logZ)={g['sigma_seed_logZ']:.3f}")
    print(f"  dlogZ by seed {t02['dlogZ_steep_minus_low_by_seed']} "
          f"sign-stable={t02['dlogZ_sign_seed_stable']}")
    print(f"  sigma_seed(dlogZ)={sig_dlz:.3f} (gate<5 "
          f"{'PASS' if sig_dlz < 5 else 'FAIL'}); KILL(>0.024)="
          f"{t02['gates']['kill_sigma_gamma_gt_0p024']}")
    print("===== T0.3 =====")
    print(f"  shift={shift:+.5f} vs band 0.024 -> {t03['verdict']}")
    print("===== finalized =====")
    print(f"  sigma_tot(gamma,low)={sigma_tot:.5f}; T1.1 bands exonerate<{3*sigma_tot:.4f}, "
          f"confirm<-{9*sigma_tot:.4f}; B5-G2 {finalized['B5_G2_dlogZ_thresh']}")


if __name__ == "__main__":
    main()
