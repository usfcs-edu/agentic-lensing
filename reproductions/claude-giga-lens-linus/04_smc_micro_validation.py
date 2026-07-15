#!/usr/bin/env python
"""04_smc_micro_validation.py -- gate B0: MC-SMC correctness + zoo adapters.

Pre-registered gates (README.md §verification, FROZEN at P0; plans/PLAN.md §5):

  B0-parity : adapter logp (log_prior + log_like) vs native ProbModel.log_prob
              at 64 matched prior draws, max |diff| <= 1e-8 -- for every scene
              target that BUILDS (a blocked build is recorded, not faked).
  B0-mix2   : |logZ - analytic| <= 3 sigma_boot AND minor-mode weight
              |w_hat - 0.2| <= 0.053 (3x binomial at N=512).
  B0-funnel : t0_funnel10 |logZ - 0| <= 3 sigma_boot.
  B0-illcond: t0_illcond46 worst-param |z| < 3.
  B0-bias   : MCLMC-vs-MAMS delta-logZ screen: |logZ_mclmc - logZ_mams| /
              sqrt(sig_mclmc^2 + sig_mams^2) > 3 on any T0 target =>
              'MCLMC demoted' (cost-frontier-only).

FROZEN estimator definitions (this file is their single source):
  * sigma_boot: per-stage particle bootstrap of the logZ increments (B=200),
    genealogy ignored (documented UNDER-estimate => the 3-sigma gates are
    conservative). cgl2/samplers/common.py::run_tempered_smc.
  * minor-mode weight (mix2): fraction of final particles with z[...,0] < 0
    (the analytic minor mode is centered at -2 e0; modes are 8 comp-sigma apart
    so the sign split is exact to ~1e-4).
  * illcond worst-param z: z_j = (mean_hat_j - mean_true_j) /
    sqrt(var_true_j / ESS_eff), ESS_eff = unique-particle count of the final
    stage (duplicate-degenerate ensembles get a WIDER se, i.e. harder to pass
    by degeneracy, not easier).

Run (L4): CUDA_VISIBLE_DEVICES=8 GIGALENS_X64=1 python 04_smc_micro_validation.py
CPU fallback for the T0 analytics (GPU contention): CGL2_ALLOW_CPU=1 JAX_PLATFORMS=cpu ...
Writes data/smc_b0_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np

# aarch64 f64 XLA livelock workaround (campaign-documented; P0 incident
# 2026-07-15: the MAMS mutate jit spun >8 min at 500% CPU on the L4 until
# priority-fusion was disabled). Must be set BEFORE jax initializes.
if "--xla_disable_hlo_passes" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_disable_hlo_passes=priority-fusion"
                               ).strip()
# House pattern (01/01a/03): pin CUDA enumeration to PCI order so
# CUDA_VISIBLE_DEVICES=8/9 select the L4s as documented. Verifier fix
# 2026-07-15: this line was missing, so the B0 gate run's CUDA device 8 was
# actually an A16 under CUDA's default fastest-first ordering (numerics
# unaffected -- f64 gates are hardware-independent; wall-times are A16's).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import cgl2  # noqa: F401,E402  (x64 bootstrap BEFORE jax import)
from cgl2 import guards, paths

PARITY_TOL = 1e-8
N_PARITY_DRAWS = 64
N_PARTICLES = 512
MIX2_W_TOL = 0.053          # 3x binomial sigma at N=512, w=0.2
ILLCOND_Z_MAX = 3.0
BIAS_SCREEN_NSIGMA = 3.0

T0_GATES = ("t0_mix2", "t0_funnel10", "t0_illcond46")
SCENE_TARGETS = ("dspl20_orig", "dspl20_ratio", "carousel33",
                 *[f"hs2_sys{i}" for i in range(8)])


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# --------------------------------------------------------------------------- #
# part 1: scene-adapter logp parity
# --------------------------------------------------------------------------- #
def run_parity(scene_names, quick=False):
    import jax
    from cgl2 import zoo

    rows = {}
    for name in scene_names:
        t0 = time.time()
        row = dict(status="?", tol=PARITY_TOL, n_draws=N_PARITY_DRAWS)
        try:
            tgt = zoo.build(name)
            key = jax.random.PRNGKey(64064)
            Z = np.asarray(tgt.prior_sample_fn(key, N_PARITY_DRAWS))
            lp = np.asarray(tgt.logprior_batch(Z), dtype=np.float64)
            ll = np.asarray(tgt.loglik_batch(Z), dtype=np.float64)
            native = np.asarray(tgt.native_logprob_batch(Z), dtype=np.float64)
            adapter = lp + ll
            finite = bool(np.all(np.isfinite(adapter))
                          and np.all(np.isfinite(native)))
            max_abs = float(np.max(np.abs(adapter - native)))
            row.update(
                status="PASS" if (finite and max_abs <= PARITY_TOL) else "FAIL",
                z_dim=int(tgt.z_dim),
                max_abs_diff=max_abs,
                all_finite=finite,
                data_note=tgt.meta.get("data", "native"),
                blocked_note=tgt.meta.get("blocked"),
                provenance=tgt.provenance,
            )
            del tgt
        except Exception as e:  # noqa: BLE001 -- recorded, never faked
            row.update(status="BLOCKED", error=f"{type(e).__name__}: {e}",
                       trace=traceback.format_exc(limit=8))
        row["wall_s"] = round(time.time() - t0, 2)
        rows[name] = row
        print(f"[parity] {name:14s} {row['status']:8s} "
              + (f"max|d|={row.get('max_abs_diff'):.3e} dim={row.get('z_dim')}"
                 if row["status"] in ("PASS", "FAIL")
                 else str(row.get('error', ''))[:120]),
              flush=True)
    return rows


# --------------------------------------------------------------------------- #
# part 2: T0 SMC gates per kernel
# --------------------------------------------------------------------------- #
def run_t0(kernels, n_particles, seed, quick=False):
    import jax
    from cgl2 import zoo
    from cgl2.samplers import smc_micro

    n_boot = 50 if quick else 200
    results = {}
    for kernel in kernels:
        results[kernel] = {}
        for name in T0_GATES:
            t0 = time.time()
            tgt = zoo.build(name)
            z0 = np.asarray(
                tgt.prior_sample_fn(jax.random.PRNGKey(9000 + seed),
                                    n_particles))
            row = dict(kernel=kernel, target=name, n_particles=n_particles,
                       seed=seed)
            try:
                out = smc_micro.run_smc(
                    tgt.logprior_fn, tgt.loglik_fn, z0, kernel=kernel,
                    seed=seed, n_boot=n_boot)
                ref = tgt.reference
                logz_err = abs(out["logZ"] - ref["logZ"])
                sig = out["logZ_boot_sigma"]
                # FROZEN gate menu (README §verification): logZ is gated on
                # mix2 and funnel10; illcond46 is gated on worst-param z ONLY
                # (its logZ is recorded as informational).
                gates = {}
                if name in ("t0_mix2", "t0_funnel10"):
                    gates["logZ"] = bool(logz_err <= 3 * sig)
                extras = {}
                if name == "t0_mix2":
                    w_hat = float(np.mean(out["particles"][:, 0] < 0.0))
                    gates["minor_mode_weight"] = bool(
                        abs(w_hat - ref["minor_mode_weight"]) <= MIX2_W_TOL)
                    extras["minor_mode_weight_hat"] = w_hat
                if name == "t0_illcond46":
                    ess_eff = float(out["unique_particle_trace"][-1])
                    mean_hat = out["particles"].mean(0)
                    se = np.sqrt(np.asarray(ref["var"]) / ess_eff)
                    zscores = (mean_hat - np.asarray(ref["mean"])) / se
                    worst = float(np.max(np.abs(zscores)))
                    gates["worst_param_z"] = bool(worst < ILLCOND_Z_MAX)
                    extras.update(worst_param_z=worst, ess_eff=ess_eff)
                row.update(
                    status="PASS" if all(gates.values()) else "FAIL",
                    gates=gates,
                    logZ=out["logZ"], logZ_ref=ref["logZ"],
                    logZ_abs_err=logz_err, logZ_boot_sigma=sig,
                    n_stages=out["n_stages"],
                    lambda_schedule=out["lambda_schedule"],
                    ess_trace=[round(v, 1) for v in out["ess_trace"]],
                    unique_particle_trace=out["unique_particle_trace"],
                    accept_trace=[round(v, 3) for v in out["accept_trace"]],
                    step_size_trace=[round(v, 4) for v in
                                     out["step_size_trace"]],
                    n_int_trace=out["n_int_trace"],
                    grad_evals=out["grad_evals"], n_logp=out["n_logp"],
                    **extras)
            except Exception as e:  # noqa: BLE001
                row.update(status="ERROR", error=f"{type(e).__name__}: {e}",
                           trace=traceback.format_exc(limit=8))
            row["wall_s"] = round(time.time() - t0, 2)
            results[kernel][name] = row
            msg = (f"logZ={row.get('logZ', float('nan')):.3f} "
                   f"(ref {row.get('logZ_ref', float('nan')):.3f} "
                   f"+-{row.get('logZ_boot_sigma', float('nan')):.3f}) "
                   f"stages={row.get('n_stages')} "
                   f"grads={row.get('grad_evals', {}).get('total')}"
                   if row["status"] != "ERROR" else str(row.get("error"))[:120])
            print(f"[t0] {kernel:5s} {name:13s} {row['status']:5s} {msg}",
                  flush=True)
    return results


def bias_screen(t0_results):
    """MCLMC-vs-MAMS delta-logZ screen over the T0 targets (frozen B0 gate)."""
    rows = {}
    demoted = False
    m1, m2 = t0_results.get("mclmc", {}), t0_results.get("mams", {})
    for name in T0_GATES:
        a, b = m1.get(name), m2.get(name)
        if not (a and b and a.get("status") != "ERROR"
                and b.get("status") != "ERROR"):
            rows[name] = dict(status="SKIPPED (missing run)")
            continue
        d = a["logZ"] - b["logZ"]
        sig = float(np.hypot(a["logZ_boot_sigma"], b["logZ_boot_sigma"]))
        nsig = abs(d) / sig if sig > 0 else float("inf")
        flag = bool(nsig > BIAS_SCREEN_NSIGMA)
        demoted = demoted or flag
        rows[name] = dict(delta_logZ=float(d), sigma=sig, n_sigma=float(nsig),
                          biased=flag)
        print(f"[bias] {name:13s} dlogZ={d:+.3f} ({nsig:.2f} sigma) "
              f"{'BIASED' if flag else 'ok'}", flush=True)
    return dict(per_target=rows, mclmc_demoted=demoted,
                rule=f"> {BIAS_SCREEN_NSIGMA} sigma on any T0 target => "
                     "MCLMC demoted to cost-frontier-only")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernels", default="hmc,mams,mclmc")
    ap.add_argument("--scene-targets", default=",".join(SCENE_TARGETS))
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--skip-t0", action="store_true")
    ap.add_argument("--n-particles", type=int, default=N_PARTICLES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="smoke mode: fewer particles/bootstraps (NOT a gate run)")
    ap.add_argument("--out", default=str(paths.DATA_DIR / "smc_b0_report.json"))
    args = ap.parse_args()

    guards.require_vendor_ref()
    guards.require_jax_pin()
    guards.require_x64()
    guards.require_gpu()
    guards.require_single_device()

    import jax
    n_particles = 128 if args.quick else args.n_particles
    kernels = [k for k in args.kernels.split(",") if k]
    scene_names = [s for s in args.scene_targets.split(",") if s]

    report = dict(
        script="04_smc_micro_validation.py",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
        jax_version=jax.__version__,
        x64=bool(jax.config.jax_enable_x64),
        quick=bool(args.quick),
        n_particles=n_particles,
        seed=args.seed,
        frozen_gates=dict(
            parity_tol=PARITY_TOL, n_parity_draws=N_PARITY_DRAWS,
            mix2=f"|logZ-analytic|<=3sig_boot AND |w-0.2|<={MIX2_W_TOL}",
            funnel10="|logZ|<=3sig_boot",
            illcond46=f"worst-param |z|<{ILLCOND_Z_MAX} "
                      "(se from unique-particle ESS)",
            bias=f"|logZ_mclmc-logZ_mams|>{BIAS_SCREEN_NSIGMA}sig => demoted",
        ),
        notes=dict(
            funnel_bias_context=(
                "P0 diagnosis 2026-07-15: t0_funnel10 logZ carries a "
                "systematic -0.4..-0.9-nat deficit that is N-INDEPENDENT "
                "(N=512: -0.55..-0.92; N=2048: -0.65/-0.72) and "
                "kernel-independent (hmc/mams/mclmc), i.e. a mutation-"
                "coverage bias of the affine-preconditioned tempered-SMC "
                "class on the funnel neck, NOT a bug in this port: the OLD "
                "VALIDATED claude-giga-lens bj_smc shows the identical "
                "deficit on the same target (logZ = -0.633/-0.761/-0.544 at "
                "N=1000; ../claude-giga-lens/data/results/bj_smc/t0_funnel10/"
                "s{0,1,2}_A.json). The naive per-stage particle bootstrap "
                "cannot see this bias, so |logZ|<=3sig_boot is structurally "
                "unpassable for this sampler class at these budgets."),
        ),
    )

    if not args.skip_parity:
        report["parity"] = run_parity(scene_names, quick=args.quick)
    if not args.skip_t0:
        report["t0"] = run_t0(kernels, n_particles, args.seed,
                              quick=args.quick)
        if "mclmc" in kernels and "mams" in kernels:
            report["bias_screen"] = bias_screen(report["t0"])

    # ---- verdict table --------------------------------------------------------
    verdict = {}
    for name, row in report.get("parity", {}).items():
        verdict[f"parity/{name}"] = row["status"]
    for kernel, targets in report.get("t0", {}).items():
        for name, row in targets.items():
            verdict[f"t0/{kernel}/{name}"] = row["status"]
    if "bias_screen" in report:
        verdict["bias/mclmc_demoted"] = report["bias_screen"]["mclmc_demoted"]
    report["verdict"] = verdict
    report["b0_pass"] = all(
        v == "PASS" for k, v in verdict.items()
        if isinstance(v, str) and not k.startswith("parity/")) and all(
        v in ("PASS", "BLOCKED") for k, v in verdict.items()
        if isinstance(v, str) and k.startswith("parity/"))

    paths.DATA_DIR.mkdir(exist_ok=True)
    out = args.out
    with open(out, "w") as fh:
        json.dump(_jsonable(report), fh, indent=1)
    print(f"\n[b0] verdict: {json.dumps(verdict, indent=1)}")
    print(f"[b0] b0_pass={report['b0_pass']} (BLOCKED parity rows are "
          "documented failures-to-build, not gate failures)")
    print(f"[b0] wrote {out}")


if __name__ == "__main__":
    main()
