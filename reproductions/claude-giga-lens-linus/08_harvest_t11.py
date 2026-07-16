"""T1.1 harvest: pre-registered injection-recovery readout (jobs 55952480/81/83
+ 55958518 resubmit of failed 55952482).

Extends 07_harvest_t02_t03.py's conventions exactly: runs under the OLD
validated venv (/raid/benson/.venvs/cgl/bin/python), CPU-only (bijector only,
never the likelihood); the per-run JSON summaries are the authoritative
extraction (weighted quantiles via cgl.e2._weighted_quantile inside
run_correlated_smc); saved equal-weight particles are transformed via
build_target('v3b').model.to_physical_mass for plots + nuisance readout and
cross-checked against the weighted medians.

PRE-REGISTERED GATES (T1.1 design checkpoint + P1-harvest finalization row,
CAMPAIGN.md; NOT moved):
  CONFIRM   median(gamma_rec - 1.43298) < -0.078
  EXONERATE |median(gamma_rec - 1.43298)| < 0.026
  between   = partial, quantified with its sign
  control   gamma_rec(diag) in [1.29, 1.43] predicted
  per-injection z = (gamma_rec_i - 1.43298)/sigma_i (sigma_i = the injection's
  own posterior sigma); NO coverage claims at n=3.

Outputs (plots BEFORE metrics, per house rule):
  figs/t11_recovery_overlay.png, data/t11_gate_eval.json
"""
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path("/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens-linus")
NEW = ROOT / "data" / "results-perlmutter"
FIGS = ROOT / "figs"

GAMMA_TRUTH = 1.4329787059806258          # asserted vs all three truth jsons
REAL_CORR_LOW = 1.1032                    # production real-data money number
CONFIRM_THRESH = -0.078                   # finalized (sigma_tot=0.0086 row)
EXONERATE_BAND = 0.026
CONTROL_WINDOW = (1.29, 1.43)

RUNS = {
    "inj1": (NEW / "t11_inj1_smc", "low"),
    "inj2": (NEW / "t11_inj2_smc", "low"),
    "inj3": (NEW / "t11_inj3_smc", "low"),
    "diag": (NEW / "t11_inj1_diag_smc", "low"),
}

# sacct -X 2026-07-15 (this readout session), single A100 shared-QOS each
SACCT = {
    "55952480": ("COMPLETED", "01:53:29", 1.89),
    "55952481": ("COMPLETED", "02:00:02", 2.00),
    "55952482": ("FAILED",    "01:29:15", 1.49),   # hbm40g OOM; prep survived
    "55952483": ("COMPLETED", "01:59:32", 1.99),
    "55958518": ("COMPLETED", "00:25:47", 0.43),   # SKIP_PREP=1 SMC-only rerun
}

# healthy-production baseline for saved-particle diversity (computed this
# session from the T0.2/T0.3/production npz: unique rows / 128)
BASELINE_UNIQUE_ROWS = {"seed2_production": 37, "seed3": 14, "seed4": 27,
                        "compmask": 17}


def load_run(stem, basin):
    js = json.load(open(stem.with_suffix(".json")))
    d = np.load(stem.with_suffix(".npz"), allow_pickle=True)
    parts = np.asarray(d[f"{basin}_particles"], dtype=np.float64)
    return js["basins"][basin], parts, js


def main():
    from cgl import e2, likelihood

    target = e2.build_target("v3b")   # bijector only; no likelihood evals

    truth = {}
    for i in (1, 2, 3):
        t = json.load(open(ROOT / "data" / f"t11_inj{i}_truth.json"))
        assert abs(t["truth_gamma"] - GAMMA_TRUTH) < 1e-12, t["truth_gamma"]
        truth[f"inj{i}"] = t
    truth["diag"] = truth["inj1"]     # control fits injection 1

    summ, parts_all, gammas, sanity = {}, {}, {}, {}
    for name, (stem, basin) in RUNS.items():
        b, parts, js = load_run(stem, basin)
        g = np.asarray(target.model.to_physical_mass(parts)["gamma"], dtype=np.float64)
        summ[name], parts_all[name], gammas[name] = b, parts, g
        med_check = float(np.median(g)) - b["gamma_median"]
        # config provenance asserts (production path, correct data + whitener)
        cfg = js["config"]
        assert cfg["seed"] == 2 and cfg["smc_particles"] == 128
        assert f"t11_inj{name[-1] if name != 'diag' else '1'}_v3b.npz" in cfg["data_file"]
        wname = js["whiten"]["whitener"]
        assert ("delta_diag" in wname) == (name == "diag"), (name, wname)
        sanity[name] = dict(
            reached_lambda1=True,   # run_adaptive_tempered_smc raises otherwise;
                                    # artifact written => terminated at lambda=1
            n_lambda_steps=b["n_lambda_steps"], ess_weight=b["ess_weight"],
            n_unique_resample_idx=b["n_unique"], n_floored_q=b["n_floored_q"],
            frac_gamma_gt_split=b["frac_gamma_gt_split"],
            n_unique_particle_rows=int(len(np.unique(parts, axis=0))),
            n_particles=b["n_particles"], wall_s=b["wall_s"],
            equalweight_median_minus_weighted=med_check,
            q16_equals_median=bool(b["gamma_q16"] == b["gamma_median"]),
            median_equals_q84=bool(b["gamma_q84"] == b["gamma_median"]),
            whitener=wname, n_keep_w=js["whiten"]["n_keep_w"],
            data_file=cfg["data_file"], logZ=b["logZ"],
        )
        print(f"[{name}] gamma_med={b['gamma_median']:.6f} "
              f"[{b['gamma_q16']:.4f},{b['gamma_q84']:.4f}] sig={b['gamma_sigma']:.4f} "
              f"logZ={b['logZ']:.2f} w_ess={b['ess_weight']:.1f}/128 "
              f"n_uniq_rows={sanity[name]['n_unique_particle_rows']} "
              f"lam_steps={b['n_lambda_steps']} frac_steep={b['frac_gamma_gt_split']:.3f} "
              f"|eqw_med-w_med|={abs(med_check):.5f}")

    # sick-run assessment (vs production-family baseline 14-37 unique rows/128)
    sick = {}
    for name in RUNS:
        s = sanity[name]
        flags = []
        if s["n_unique_particle_rows"] <= 2:
            flags.append("POINT-MASS FINAL POPULATION (total resample collapse)")
        if summ[name]["gamma_sigma"] < 1e-6:
            flags.append("gamma_sigma ~ 0 (degenerate posterior; z-score undefined)")
        if s["q16_equals_median"] or s["median_equals_q84"]:
            flags.append("dominant duplicate cluster at a quantile (same class as "
                         "T0.2 seed3-low; reported, not sick)")
        sick[name] = dict(flags=flags,
                          is_sick=bool(s["n_unique_particle_rows"] <= 2
                                       or summ[name]["gamma_sigma"] < 1e-6))

    # ---------------- plots BEFORE metrics ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    C = {"inj1": "#2a78d6", "inj2": "#008300", "inj3": "#e87ba4", "diag": "#8a5cd6"}
    INK, MUT = "#3a3a38", "#8a8a86"

    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=150)
    ax.axvspan(GAMMA_TRUTH - EXONERATE_BAND, GAMMA_TRUTH + EXONERATE_BAND,
               color="#f0efec", zorder=0, label="exonerate band (truth ± 0.026)")
    ax.axvspan(*CONTROL_WINDOW, ymin=0.0, ymax=0.06, color="#c9b8ea", zorder=1,
               label="control prediction 1.29–1.43")
    for name in ("inj1", "inj2", "inj3"):
        x = gammas[name]
        kde = gaussian_kde(x)
        lo, hi = x.min(), x.max()
        pad = 0.15 * (hi - lo + 1e-9)
        xs = np.linspace(lo - pad, hi + pad, 400)
        lab = (f"{name} shift{tuple(truth[name]['source_shift_arcsec'])}\" "
               f"med={summ[name]['gamma_median']:.3f}")
        ax.plot(xs, kde(xs), color=C[name], lw=2, label=lab)
        ax.plot([summ[name]["gamma_median"]], [0], marker="|", ms=14, mew=2,
                color=C[name], clip_on=False)
    dv = summ["diag"]["gamma_median"]
    ax.axvline(dv, color=C["diag"], lw=2, ls=":",
               label=f"diag control (COLLAPSED point mass) {dv:.3f}")
    ax.axvline(GAMMA_TRUTH, color=INK, lw=1.6,
               label=r"$\gamma_{truth}$ = 1.43298")
    ax.axvline(REAL_CORR_LOW, color="#c23b22", lw=1.6, ls="--",
               label="real-data corr-low 1.1032")
    ax.set_title("T1.1 injection recovery: production corr-SMC posteriors vs truth "
                 "(equal-weight particles)", color=INK, fontsize=10)
    ax.set_xlabel(r"$\gamma$ (physical)", color=INK)
    ax.set_ylabel("density", color=INK)
    ax.tick_params(colors=MUT, labelsize=8)
    ax.grid(True, color="#e8e7e3", lw=0.7)
    for s in ax.spines.values():
        s.set_color("#d5d4d0")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "t11_recovery_overlay.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {FIGS}/t11_recovery_overlay.png")

    # ---------------- gates (exact, pre-registered) ----------------
    biases = {n: summ[n]["gamma_median"] - GAMMA_TRUTH for n in ("inj1", "inj2", "inj3")}
    zs = {n: biases[n] / summ[n]["gamma_sigma"] for n in ("inj1", "inj2", "inj3")}
    med_bias = float(np.median(list(biases.values())))
    med_bias_no_inj2 = float(np.median([biases["inj1"], biases["inj3"]]))

    def verdict(mb):
        if mb < CONFIRM_THRESH:
            return "CONFIRM (low bias)"
        if abs(mb) < EXONERATE_BAND:
            return "EXONERATE"
        if CONFIRM_THRESH <= mb <= -EXONERATE_BAND:
            return f"PARTIAL (negative, {mb:+.4f})"
        return (f"OUTSIDE PRE-REGISTERED ZONES: bias POSITIVE {mb:+.4f} "
                "(opposite the predicted direction)")

    ctrl_bias = summ["diag"]["gamma_median"] - GAMMA_TRUTH
    ctrl_in_window = bool(CONTROL_WINDOW[0] <= summ["diag"]["gamma_median"]
                          <= CONTROL_WINDOW[1])
    diff_corr_minus_diag_inj1 = summ["inj1"]["gamma_median"] - summ["diag"]["gamma_median"]

    gates = dict(
        gamma_truth=GAMMA_TRUTH,
        per_injection={n: dict(
            gamma_median=summ[n]["gamma_median"], gamma_q16=summ[n]["gamma_q16"],
            gamma_q84=summ[n]["gamma_q84"], gamma_sigma=summ[n]["gamma_sigma"],
            bias=float(biases[n]), z=float(zs[n]), logZ=summ[n]["logZ"],
            sigma_def="the injection's own SMC posterior sigma (weighted, "
                      "run_correlated_smc extraction)") for n in ("inj1", "inj2", "inj3")},
        median_bias_n3=med_bias,
        median_bias_without_inj2=med_bias_no_inj2,
        thresholds=dict(confirm_lt=CONFIRM_THRESH, exonerate_abs_lt=EXONERATE_BAND),
        verdict_n3=verdict(med_bias),
        verdict_without_inj2=verdict(med_bias_no_inj2),
        control=dict(
            gamma_median=summ["diag"]["gamma_median"], bias=float(ctrl_bias),
            window=list(CONTROL_WINDOW), in_window=ctrl_in_window,
            logZ=summ["diag"]["logZ"],
            logZ_not_comparable="different likelihood (diag, n_keep_w 16653 "
                                "unwhitened dof vs 9273 whitened): not comparable",
            sick=sick["diag"],
            z_undefined="gamma_sigma=4.4e-16 => own-sigma z-score undefined"),
        differential_corr_minus_diag_inj1_same_data=float(diff_corr_minus_diag_inj1),
        no_coverage_claims="n=3; per-injection z reported; no coverage claims",
    )

    # ---------------- nuisance recovery vs truth ----------------
    # equal-weight particle physical quantiles per named param (P1 cross-check
    # showed eqw vs weighted medians agree <= 0.003 on gamma; weighted particle
    # population is not persisted in the npz)
    nuis = {}
    for name in ("inj1", "inj2", "inj3", "diag"):
        t = dict((k, v) for k, v in truth[name]["physical_named"])
        parts = parts_all[name]
        flat = [likelihood._flat_named46(
            target.model.bij.forward(list(p[:, None]))) for p in parts]
        labels = [k for k, _ in flat[0]]
        vals = np.array([[v for _, v in row] for row in flat])   # (128, 46)
        zmed = np.median(parts, axis=0)
        rows = {}
        for j, lab in enumerate(labels):
            v = vals[:, j]
            med, q16, q84 = np.percentile(v, [50, 16, 84])
            spread = max((q84 - q16) / 2.0, 1e-12)
            dev = (med - t[lab]) / spread
            rows[lab] = dict(median=float(med), q16=float(q16), q84=float(q84),
                             truth=float(t[lab]), dev_sigma=float(dev))
        railed = [lab for j, lab in enumerate(labels)
                  if abs(zmed[j]) > 3.5]           # unconstrained-space rail proxy
        big_dev = {lab: r["dev_sigma"] for lab, r in rows.items()
                   if abs(r["dev_sigma"]) > 3 and rows[lab]["q84"] > rows[lab]["q16"]}
        nuis[name] = dict(params=rows, railed_z_gt_3p5=railed,
                          dev_gt_3sigma=big_dev,
                          max_abs_z_unconstrained=float(np.max(np.abs(zmed))))
        src = {k: rows[k] for k in rows if k.startswith(("srcS.", "srcShp."))}
        print(f"[{name}] nuisances: max|z_unconstr|={nuis[name]['max_abs_z_unconstrained']:.2f} "
              f"railed={railed} |dev|>3sig={list(big_dev)}")
        for k, r in src.items():
            print(f"    {k:18s} med={r['median']:+.4f} [{r['q16']:+.4f},{r['q84']:+.4f}] "
                  f"truth={r['truth']:+.4f} dev={r['dev_sigma']:+.2f}sig")

    out = dict(
        inputs={n: str(s[0]) + ".{npz,json}" for n, s in RUNS.items()},
        truth_files={f"inj{i}": str(ROOT / "data" / f"t11_inj{i}_truth.json")
                     for i in (1, 2, 3)},
        sacct={j: dict(state=s, elapsed=e, a100_h=h) for j, (s, e, h) in SACCT.items()},
        a100_h_t11_total=float(sum(h for _, _, h in SACCT.values())),
        sanity=sanity, sick=sick,
        baseline_unique_rows_production_family=BASELINE_UNIQUE_ROWS,
        gates=gates, nuisances=nuis,
    )
    outp = ROOT / "data" / "t11_gate_eval.json"
    json.dump(out, open(outp, "w"), indent=1, default=float)
    print(f"\n[gate-eval] wrote {outp}")

    print("\n===== T1.1 PRE-REGISTERED GATES =====")
    for n in ("inj1", "inj2", "inj3"):
        g = gates["per_injection"][n]
        print(f"  {n}: gamma_rec={g['gamma_median']:.4f} [{g['gamma_q16']:.4f},"
              f"{g['gamma_q84']:.4f}] bias={g['bias']:+.4f} z={g['z']:+.2f} "
              f"logZ={g['logZ']:.2f}")
    print(f"  median bias (n=3) = {med_bias:+.4f} -> {gates['verdict_n3']}")
    print(f"  median bias (no inj2) = {med_bias_no_inj2:+.4f} -> "
          f"{gates['verdict_without_inj2']}")
    print(f"  control: {summ['diag']['gamma_median']:.4f} vs window "
          f"{CONTROL_WINDOW} -> {'IN' if ctrl_in_window else 'OUT (HIGH)'}; "
          f"SICK={sick['diag']['is_sick']}")
    print(f"  corr - diag on same data (inj1): {diff_corr_minus_diag_inj1:+.4f}")


if __name__ == "__main__":
    main()
