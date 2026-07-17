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


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "plot"
    js, parts, om0, om0_saved, xchk = load_om0()
    print(f"[b2] loaded {STEM.name}: particles {parts.shape}, "
          f"readout-vs-recompute max|d| = {xchk:.3e}")
    if stage == "plot":
        do_plot(om0)
    elif stage == "gates":
        do_gates(js, om0, om0_saved, xchk)
    else:
        raise SystemExit(f"unknown stage {stage!r} (plot|gates)")


if __name__ == "__main__":
    main()
