"""P2 B2 harvest (ORIG ARM ONLY): evaluate the pre-registered dspl20_orig
arm-mass gate from job 55985447's completed first arm.

CONTEXT (harvest session 2026-07-16): job 55985447 TIMED OUT at its 2:30 wall.
The dspl20_orig arm COMPLETED and wrote b2_dspl20_orig_s1_seed2.{npz,json}
(logZ printed by the runner; artifacts cp'd to CFS by the in-job cp). The
in-job dspl20_ratio CONTROL arm was cut off mid-SMC (built at +2 min after the
orig arm finished, killed ~80 min later, wrote nothing -- 23_run_p2_scene.py
has no progress checkpointing). Gate row therefore reads:
ORIG ARM EVALUATED / CONTROL ARM PENDING-RERUN.

Gate (PLAN §6 B2 verbatim, CAMPAIGN.md design checkpoint 2026-07-16):
  m_hat = weighted mass fraction of Om0 < 0.146 on dspl20_orig;
  PASS iff |m_hat - 0.103| <= 0.045.
  (0.103 = their pre-registered Run-A arm mass; 0.146 = their arm split point.)
  Weights: the saved particle set is post-(systematic-resample)+mutation at
  lambda=1 (common.run_tempered_smc processes the final increment, resamples,
  and mutates AT lambda=1), i.e. EQUAL WEIGHT -- m_hat is the indicator mean.
  SMC sanity gates: lambda=1 reached (driver raises otherwise; artifact
  written => reached), unique particles >= N/4, finite logliks (fail-loud).

Runs under the NEW campaign venv (/raid/benson/.venvs/cgl2/bin/python),
CPU-only, BIJECTOR ONLY (never the lensing likelihood): the authoritative
Om0 extraction recomputes constrained params from the saved z particles via
the deterministic zoo target builder (the runner's constrained_readout is
"never load-bearing" by its own docstring; it is cross-checked here).

Stages (plots BEFORE metrics, per house rule):
  python 26_harvest_b2.py plot    -> figs/b2_om0_posterior.png (no gate math)
  python 26_harvest_b2.py gates   -> data/b2_gate_eval.json + verdict print

RATIO-CONTROL EXTENSION (recovery-lane harvest 2026-07-19, job 56006065):
the standalone dspl20_ratio control COMPLETED (lambda=1 at 64 stages).
Control gate (design checkpoint 2026-07-16 verbatim): "control dspl20_ratio
must reproduce their Run A r2 ~ N(1.32417, 6.7e-4) (weighted r2 =
u_fn(Om0,w0) evaluated at harvest via the copied ratio-coords module)".
The checkpoint froze the REFERENCE, not a numeric tolerance; the
operationalization below is DECLARED AT HARVEST (flagged in the output
json, never silently): (i) shape: sigma(r2)/6.7e-4 in [1/1.5, 1.5] (the
campaign's standing width-ratio convention); (ii) location: |mean(r2) -
1.32417| reported in units of 6.7e-4 with the fresh-realization caveat
(data_seed=0 fresh lenstronomy realization; their baseline realization is
unreproducible per their own docstring, so the DATA u* shifts by O(1
posterior sigma) between realizations and location is NOT a hard gate);
(iii) the control's own minor-arm mass m_ctrl(Om0<0.146) vs their Run-A
0.103 +/- 0.045 band decides the pre-registered falsifier branch
(postmortem section 4, written BEFORE this control ran):
  - control shows arm mass in-band AND r2 shape reproduces  => falsifier
    FIRES (orig outside WHILE control passes): MC-SMC prior-seeded in
    ORIGINAL coords genuinely fails to traverse the thin-ridge truncation.
  - control also shows ~zero minor-arm mass => the 0.103 band is
    uncalibrated for this realization; gate re-registration against the
    control's own arm mass required (ledgered amendment).
Extra stages:
  python 26_harvest_b2.py plot-ratio   -> figs/b2_ratio_control.png (no gate math)
  python 26_harvest_b2.py gates-ratio  -> data/b2_gate_eval.json (combined
                                          orig + control + falsifier verdict)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CGL2_ALLOW_CPU", "1")   # bijector-only harvest, no likelihood

ROOT = Path("/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens-linus")
NEW = ROOT / "data" / "results-perlmutter"
STEM = NEW / "b2_dspl20_orig_s1_seed2"
FIG = ROOT / "figs" / "b2_om0_posterior.png"
OUT = ROOT / "data" / "b2_gate_eval.json"

SPLIT = 0.146          # their arm split point (Om0 < SPLIT = the minor arm)
REF_MASS = 0.103       # their pre-registered Run-A arm mass
TOL = 0.045            # PLAN §6 B2 band
TRUTH_OM0 = 0.3        # generation truth (meta.truth.cosmo.Om0)

# ratio-control references (design checkpoint 2026-07-16; _ratio_coords_copy)
RSTEM = NEW / "b2_dspl20_ratio_s1_seed2"
RFIG = ROOT / "figs" / "b2_ratio_control.png"
RUNA_R2_MEAN = 1.32417     # their Run A r2 posterior (frozen reference)
RUNA_R2_SIGMA = 6.7e-4
USTAR = 1.32392            # THEIR realization's data u* (module docstring)
WIDTH_BAND = (1.0 / 1.5, 1.5)  # harvest-declared shape operationalization

INK, MUT = "#3a3a38", "#8a8a86"
BLUE, GREEN, BAND = "#2a78d6", "#008300", "#f0efec"


def load_om0():
    js = json.load(open(STEM.with_suffix(".json")))
    d = np.load(STEM.with_suffix(".npz"), allow_pickle=True)
    parts = np.asarray(d["particles"], dtype=np.float64)
    om0_saved = np.asarray(d["cosmo__Om0"], dtype=np.float64)

    # authoritative recompute: z -> constrained via the deterministic builder
    import jax.numpy as jnp
    from cgl2 import zoo
    target = zoo.build("dspl20_orig")     # bijector only; no likelihood evals
    u = target.prob_model.model.bijector.forward(jnp.asarray(parts))
    om0 = np.asarray(u["cosmo/Om0"], dtype=np.float64).reshape(-1)
    xchk = float(np.max(np.abs(om0 - om0_saved)))
    return js, parts, om0, om0_saved, xchk


def do_plot(om0, m_ref_only=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.hist(om0, bins=48, range=(0.0, 0.7), density=True, color=BLUE,
            alpha=0.85, edgecolor="white", linewidth=0.4,
            label="equal-weight SMC particles (N=512)")
    ax.axvline(SPLIT, color=INK, lw=1.2, ls="--")
    ax.axvline(TRUTH_OM0, color=GREEN, lw=1, ls=":")
    ax.text(SPLIT, ax.get_ylim()[1] * 0.97, "  arm split 0.146",
            color=INK, fontsize=8, va="top")
    ax.text(TRUTH_OM0, ax.get_ylim()[1] * 0.97, "  truth 0.3",
            color=GREEN, fontsize=8, va="top")
    ax.set_title("B2 dspl20_orig S1 seed 2: $\\Omega_{m0}$ posterior "
                 "(ORIGINAL coords; job 55985447 arm 1)",
                 color=INK, fontsize=10)
    ax.set_xlabel(r"$\Omega_{m0}$", color=INK)
    ax.set_ylabel("density", color=INK)
    ax.tick_params(colors=MUT, labelsize=8)
    ax.grid(True, color="#e8e7e3", lw=0.7)
    for s in ax.spines.values():
        s.set_color("#d5d4d0")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    plt.close(fig)
    print(f"[b2] wrote {FIG} (NO gate math this stage)")


def do_gates(js, om0, om0_saved, xchk):
    n = om0.shape[0]
    m_hat = float(np.mean(om0 < SPLIT))
    dev = m_hat - REF_MASS
    gate_pass = abs(dev) <= TOL
    binom_sigma = float(np.sqrt(max(m_hat * (1 - m_hat), 1e-12) / n))

    uniq = js["unique_particle_trace"]
    ess = js["ess_trace"]
    sanity = dict(
        lambda1_reached=bool(js["lambda_schedule"][-1] == 1.0),
        n_stages=int(js["n_stages"]),
        final_unique=int(uniq[-1]), min_unique=int(min(uniq)),
        unique_ge_quarterN=bool(min(uniq) >= n // 4),
        final_incr_ess=float(ess[-1]), min_incr_ess=float(min(ess)),
        wall_s=js["wall_s"], peak_mb=js["peak_mb"],
        grad_evals=js["grad_evals"], n_logp=js["n_logp"],
    )
    out = dict(
        script="26_harvest_b2.py",
        job="55985447 (TIMEOUT 02:30:33; orig arm COMPLETED, ratio control arm "
            "cut off mid-SMC -- PENDING-RERUN)",
        target="dspl20_orig", arm="s1", n_particles=n, seed=js["seed"],
        gate="|m_hat(Om0<0.146) - 0.103| <= 0.045 (PLAN §6 B2, pre-registered)",
        m_hat=m_hat, reference_mass=REF_MASS, split=SPLIT, tol=TOL,
        deviation=dev, m_hat_binomial_sigma=binom_sigma,
        verdict="PASS" if gate_pass else "FAIL",
        control_arm="dspl20_ratio PENDING-RERUN (their Run A r2 ~ "
                    "N(1.32417, 6.7e-4) reproduction NOT yet evaluated; the "
                    "orig-arm PASS/FAIL is conditional wrt the falsifier "
                    "clause until the control lands)",
        weights_note="equal-weight particle set (post-resample+mutation at "
                     "lambda=1); m_hat = indicator mean",
        om0_median=float(np.median(om0)),
        om0_q16=float(np.quantile(om0, 0.16)),
        om0_q84=float(np.quantile(om0, 0.84)),
        logZ=js["logZ"], logZ_boot_sigma=js["logZ_boot_sigma"],
        readout_crosscheck_max_abs_diff=xchk,
        readout_crosscheck_pass=bool(xchk <= 1e-9),
        sanity=sanity,
        fig=str(FIG.relative_to(ROOT)),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[b2] m_hat = {m_hat:.4f} (ref {REF_MASS} +/- {TOL}; dev {dev:+.4f}; "
          f"binom sigma {binom_sigma:.4f}) -> {out['verdict']}")
    print(f"[b2] Om0 posterior: median {out['om0_median']:.4f} "
          f"[{out['om0_q16']:.4f}, {out['om0_q84']:.4f}] eq-weight; "
          f"logZ = {js['logZ']:.2f} +/- {js['logZ_boot_sigma']:.2f}")
    print(f"[b2] sanity: stages {sanity['n_stages']}, lambda1 "
          f"{sanity['lambda1_reached']}, min unique {sanity['min_unique']} "
          f"(>=N/4 {sanity['unique_ge_quarterN']}), xchk {xchk:.2e}")
    print(f"[b2] wrote {OUT}")


def load_ratio():
    """Load the control npz; recompute (Om0, w0) from z via the deterministic
    builder's bijector, then r2 = u_fn(Om0, w0) via the copied ratio-coords
    module (the design checkpoint's harvest recipe, verbatim). BIJECTOR +
    u_fn ONLY -- no likelihood evals."""
    js = json.load(open(RSTEM.with_suffix(".json")))
    d = np.load(RSTEM.with_suffix(".npz"), allow_pickle=True)
    parts = np.asarray(d["particles"], dtype=np.float64)

    import jax
    import jax.numpy as jnp
    from cgl2 import zoo
    target = zoo.build("dspl20_ratio")    # bijector only; no likelihood evals
    u = target.prob_model.model.bijector.forward(jnp.asarray(parts))
    # grouped tuple-key prior emits a joint column block (Om0, w0) in tuple
    # order; both bounds boxes are disjoint so a range assert proves ordering
    cw = np.asarray(u["cosmo/Om0|cosmo/w0"], dtype=np.float64)
    om0, w0 = cw[:, 0].copy(), cw[:, 1].copy()
    assert (om0 >= 0).all() and (om0 <= 1).all(), "column 0 is not Om0"
    assert (w0 >= -2).all() and (w0 <= -1 / 3).all(), "column 1 is not w0"

    # readout cross-check on a saved constrained column (npz carries no cosmo
    # readout for the grouped prior; theta_E is the checkable proxy)
    te_saved = np.asarray(d["planes__0__mass__0__theta_E"], dtype=np.float64)
    te = np.asarray(u["planes/0/mass/0/theta_E"], dtype=np.float64).reshape(-1)
    xchk = float(np.max(np.abs(te - te_saved)))

    # r2 = u_fn(Om0, w0), the zoo builder's own construction (zoo.py ratio branch)
    from cgl2._ratio_coords_copy import deflection_ratio_u_fn
    from gigalens.jax.cosmo import w0waCDM_Cosmo
    u_fn = deflection_ratio_u_fn(
        w0waCDM_Cosmo(z_lens=zoo.DSPL_Z_LENS, z_source_ref=zoo.DSPL_Z_SOURCE1),
        (zoo.DSPL_Z_SOURCE2,), (1.0,), fixed=dict(H0=70.0, k=0.0, wa=0.0))
    r2 = np.asarray(jax.vmap(u_fn)(jnp.asarray(om0), jnp.asarray(w0)),
                    dtype=np.float64).reshape(-1)
    return js, parts, om0, w0, r2, xchk


def do_plot_ratio(r2, om0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.6), dpi=150)

    ax1.hist(r2, bins=48, density=True, color=BLUE, alpha=0.85,
             edgecolor="white", linewidth=0.4,
             label="control r2 = u_fn(Om0,w0)\n(equal-weight, N=512)")
    xs = np.linspace(RUNA_R2_MEAN - 6 * RUNA_R2_SIGMA,
                     RUNA_R2_MEAN + 6 * RUNA_R2_SIGMA, 400)
    lo, hi = ax1.get_xlim()
    xs = np.linspace(min(lo, xs[0]), max(hi, xs[-1]), 600)
    ax1.plot(xs, np.exp(-0.5 * ((xs - RUNA_R2_MEAN) / RUNA_R2_SIGMA) ** 2)
             / (RUNA_R2_SIGMA * np.sqrt(2 * np.pi)),
             color=INK, lw=1.2, ls="--",
             label="their Run A N(1.32417, 6.7e-4)")
    ax1.axvline(USTAR, color=GREEN, lw=1, ls=":",
                label="THEIR data u* 1.32392")
    ax1.set_title("B2 control: r2 posterior vs their Run A (job 56006065)",
                  color=INK, fontsize=9.5)
    ax1.set_xlabel("r2 (deflection ratio coordinate)", color=INK)
    ax1.set_ylabel("density", color=INK)
    ax1.legend(frameon=False, fontsize=7.5, labelcolor=INK)

    ax2.hist(om0, bins=48, range=(0.0, 0.7), density=True, color=BLUE,
             alpha=0.85, edgecolor="white", linewidth=0.4,
             label="control Om0 (equal-weight)")
    ax2.axvline(SPLIT, color=INK, lw=1.2, ls="--")
    ax2.axvline(TRUTH_OM0, color=GREEN, lw=1, ls=":")
    ax2.text(SPLIT, ax2.get_ylim()[1] * 0.97, "  arm split 0.146",
             color=INK, fontsize=8, va="top")
    ax2.text(TRUTH_OM0, ax2.get_ylim()[1] * 0.97, "  truth 0.3",
             color=GREEN, fontsize=8, va="top")
    ax2.set_title("B2 control: implied $\\Omega_{m0}$ (ratio coords, mapped back)",
                  color=INK, fontsize=9.5)
    ax2.set_xlabel(r"$\Omega_{m0}$", color=INK)
    ax2.set_ylabel("density", color=INK)
    ax2.legend(frameon=False, fontsize=7.5, labelcolor=INK)

    for ax in (ax1, ax2):
        ax.tick_params(colors=MUT, labelsize=8)
        ax.grid(True, color="#e8e7e3", lw=0.7)
        for s in ax.spines.values():
            s.set_color("#d5d4d0")
    fig.tight_layout()
    RFIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(RFIG)
    plt.close(fig)
    print(f"[b2r] wrote {RFIG} (NO gate math this stage)")


def do_gates_ratio(js, om0, w0, r2, xchk):
    n = r2.shape[0]
    r2_mean, r2_std = float(np.mean(r2)), float(np.std(r2, ddof=1))
    r2_med = float(np.median(r2))
    width_ratio = r2_std / RUNA_R2_SIGMA
    shape_pass = bool(WIDTH_BAND[0] <= width_ratio <= WIDTH_BAND[1])
    mean_off = (r2_mean - RUNA_R2_MEAN) / RUNA_R2_SIGMA
    mean_off_vs_ustar = (r2_mean - USTAR) / RUNA_R2_SIGMA

    m_ctrl = float(np.mean(om0 < SPLIT))
    m_ctrl_binom = float(np.sqrt(max(m_ctrl * (1 - m_ctrl), 1e-12) / n))
    # rule-of-three 95% upper bound when the count is zero
    m_ctrl_ro3 = float(3.0 / n) if m_ctrl == 0.0 else None
    m_ctrl_in_band = bool(abs(m_ctrl - REF_MASS) <= TOL)

    uniq = js["unique_particle_trace"]
    ess = js["ess_trace"]
    sanity = dict(
        lambda1_reached=bool(js["lambda_schedule"][-1] == 1.0),
        n_stages=int(js["n_stages"]),
        final_unique=int(uniq[-1]), min_unique=int(min(uniq)),
        unique_ge_quarterN=bool(min(uniq) >= n // 4),
        final_incr_ess=float(ess[-1]), min_incr_ess=float(min(ess)),
        accept_range=[float(min(js["accept_trace"])),
                      float(max(js["accept_trace"]))],
        wall_s=js["wall_s"], peak_mb=js["peak_mb"],
        grad_evals=js["grad_evals"], n_logp=js["n_logp"],
    )

    # ---- falsifier decision (pre-registered branches, postmortem section 4) --
    orig = json.load(open(OUT))            # the orig-arm eval written by `gates`
    if "orig_arm" in orig:                 # rerun on an already-combined file
        orig = orig["orig_arm"]
    assert orig["target"] == "dspl20_orig", "run `gates` before `gates-ratio`"
    orig_m = float(orig["m_hat"])
    orig_outside = bool(abs(orig_m - REF_MASS) > TOL)

    control_passes = bool(shape_pass and m_ctrl_in_band)
    if control_passes and orig_outside:
        falsifier = "FIRES"
        reading = ("orig-coords arm mass outside the band WHILE the control "
                   "passes on the same data => prior-seeded MC-SMC in the "
                   "ORIGINAL coordinates does NOT fix their coordinate "
                   "pathology (honest negative for the central bet; their "
                   "exact reparameterization remains the right tool on DSPL).")
    elif (m_ctrl < REF_MASS - TOL) and orig_outside:
        falsifier = "NOT-DECIDABLE-AS-REGISTERED (band uncalibrated)"
        reading = ("the control ALSO shows minor-arm mass below the Run-A "
                   "band on this fresh realization => the 0.103 +/- 0.045 "
                   "band is uncalibrated for realization data_seed=0; the "
                   "gate must be re-registered against the control's own arm "
                   "mass (ledgered amendment, not a silent move). The "
                   "orig-vs-control AGREEMENT is then the readable result.")
    else:
        falsifier = "DOES-NOT-FIRE"
        reading = ("control does not pass as operationalized; B2 cell is "
                   "data/adapter-suspect rather than sampler-negative.")

    # orig-vs-control consistency (same data realization, same seed recipe)
    agree_z = None
    if m_ctrl > 0 or orig_m > 0:
        pool = (m_ctrl * n + orig_m * 512) / (n + 512)
        se = np.sqrt(max(pool * (1 - pool), 1e-12) * (1 / n + 1 / 512))
        agree_z = float((m_ctrl - orig_m) / se)

    combined = dict(
        script="26_harvest_b2.py (gates-ratio, combined)",
        generated="2026-07-19 recovery-lane harvest",
        jobs=dict(orig="55985447 arm 1 (COMPLETED in-wall)",
                  ratio_control="56006065 standalone (COMPLETED 01:49:02)"),
        orig_arm=orig,
        ratio_control=dict(
            target="dspl20_ratio", arm="s1", n_particles=n,
            seed=js["seed"], logZ=js["logZ"],
            logZ_boot_sigma=js["logZ_boot_sigma"],
            gate=("reproduce Run A r2 ~ N(1.32417, 6.7e-4); "
                  "operationalization DECLARED AT HARVEST (see notes)"),
            r2_mean=r2_mean, r2_std=r2_std, r2_median=r2_med,
            r2_q16=float(np.quantile(r2, 0.16)),
            r2_q84=float(np.quantile(r2, 0.84)),
            runa_mean=RUNA_R2_MEAN, runa_sigma=RUNA_R2_SIGMA,
            their_data_ustar=USTAR,
            width_ratio=width_ratio, width_band=list(WIDTH_BAND),
            shape_pass=shape_pass,
            mean_offset_in_runa_sigma=float(mean_off),
            mean_offset_vs_their_ustar_sigma=float(mean_off_vs_ustar),
            location_note=(
                "location is NOT a hard gate: data_seed=0 is a FRESH "
                "lenstronomy realization (their baseline unreproducible per "
                "their docstring); the data's u* shifts O(1 posterior sigma) "
                "between realizations"),
            m_ctrl=m_ctrl, m_ctrl_binomial_sigma=m_ctrl_binom,
            m_ctrl_rule_of_three_95=m_ctrl_ro3,
            m_ctrl_in_runa_band=m_ctrl_in_band,
            om0_median=float(np.median(om0)),
            om0_q16=float(np.quantile(om0, 0.16)),
            om0_q84=float(np.quantile(om0, 0.84)),
            w0_median=float(np.median(w0)),
            readout_crosscheck_max_abs_diff=xchk,
            readout_crosscheck_pass=bool(xchk <= 1e-9),
            readout_crosscheck_note="theta_E proxy (npz carries no cosmo "
                                    "readout for the grouped ratio prior)",
            sanity=sanity,
        ),
        falsifier_verdict=falsifier,
        falsifier_reading=reading,
        orig_vs_control_arm_mass_z=agree_z,
        delta_logZ_orig_minus_ratio=float(orig["logZ"] - js["logZ"]),
        fig=str(RFIG.relative_to(ROOT)),
    )
    json.dump(combined, open(OUT, "w"), indent=1)
    print(f"[b2r] r2: mean {r2_mean:.5f} std {r2_std:.2e} "
          f"(Run A {RUNA_R2_MEAN} +/- {RUNA_R2_SIGMA:.1e}); "
          f"width ratio {width_ratio:.2f} (band {WIDTH_BAND[0]:.2f}-"
          f"{WIDTH_BAND[1]:.1f}) -> shape {'PASS' if shape_pass else 'FAIL'}")
    print(f"[b2r] mean offset {mean_off:+.2f} sigma_RunA "
          f"({mean_off_vs_ustar:+.2f} vs their data u*; fresh-realization "
          f"caveat applies)")
    print(f"[b2r] m_ctrl(Om0<{SPLIT}) = {m_ctrl:.4f} "
          f"(binom sigma {m_ctrl_binom:.4f}"
          + (f", rule-of-three 95% < {m_ctrl_ro3:.4f}" if m_ctrl_ro3 else "")
          + f") vs Run-A band {REF_MASS} +/- {TOL} -> "
          f"{'IN-BAND' if m_ctrl_in_band else 'OUTSIDE'}")
    print(f"[b2r] FALSIFIER: {falsifier}")
    print(f"[b2r] wrote {OUT}")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if stage in ("plot", "gates"):
        js, parts, om0, om0_saved, xchk = load_om0()
        print(f"[b2] loaded {STEM.name}: particles {parts.shape}, "
              f"readout-vs-recompute max|d| = {xchk:.3e}")
        if stage == "plot":
            do_plot(om0)
        else:
            do_gates(js, om0, om0_saved, xchk)
    elif stage in ("plot-ratio", "gates-ratio"):
        js, parts, om0, w0, r2, xchk = load_ratio()
        print(f"[b2r] loaded {RSTEM.name}: particles {parts.shape}, "
              f"theta_E readout-vs-recompute max|d| = {xchk:.3e}")
        if stage == "plot-ratio":
            do_plot_ratio(r2, om0)
        else:
            do_gates_ratio(js, om0, w0, r2, xchk)
    else:
        raise SystemExit(
            f"unknown stage {stage!r} (plot|gates|plot-ratio|gates-ratio)")


if __name__ == "__main__":
    main()
