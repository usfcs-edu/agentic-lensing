"""Compile diagnostic: was the original gigalens HMC merely SLOW, or genuinely
PATHOLOGICAL to JIT-compile?

We AOT-lower-and-compile the ONE-STEP kernel (bootstrap_results + one_step inside
a single @jax.jit) for three samplers and time .lower() and .compile() separately,
recording HLO text length, op counts, and cost_analysis.

  (A) gigalens stack: PreconditionedHamiltonianMonteCarlo(num_leapfrog_steps=3)
      -> GradientBasedTrajectoryLengthAdaptation(max_leapfrog_steps=30)
      -> DualAveragingStepSizeAdaptation.  (identity/diag momentum to isolate
      the compile from any NaN.)  RUN UNDER `timeout 900` IN A SEPARATE PROCESS:
        timeout 900 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
          python 26_compile_diagnostic.py --kernel A
      exit 124 => did not finish in 15 min => genuinely pathological.

  (B) fixed-leapfrog: PreconditionedHamiltonianMonteCarlo(num_leapfrog_steps=16,
      momentum_distribution=momentum_distribution('inv'))
      -> DualAveragingStepSizeAdaptation.

  (C) reference: PreconditionedNoUTurnSampler(max_tree_depth=6)
      -> DualAveragingStepSizeAdaptation   (the v11f NUTS kernel).

NOTE: the original hung run used pmap over 10 devices PLUS the heavier lstsq
model (HLO shows f32[10,31,256,256] grouped convolutions). This single-device
diagnostic is therefore a strict LOWER BOUND on the original pathology.
"""
import argparse
import re
import time

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

import _hmc_lib as L

tfd = tfp.distributions
tfe = tfp.experimental

NUM_ADAPT = 80  # small but nonzero; adapter init is part of the compile


def _count_ops(text):
    return {
        "convolution": len(re.findall(r"convolution", text)),
        "dot": len(re.findall(r"\bdot\b|dot_general|\bdot\(", text)),
        "while": len(re.findall(r"\bwhile\b", text)),
        "len": len(text),
    }


def measure(name, kernel, state, key):
    """AOT-lower-and-compile one_step(bootstrap_results(state)); time each phase."""
    print(f"\n===== kernel ({name}) =====", flush=True)
    try:
        print(f"[{name}] state shape = {state.shape}", flush=True)
    except Exception:  # noqa: BLE001
        pass

    def fn(state, key):
        pkr = kernel.bootstrap_results(state)
        new_state, _ = kernel.one_step(state, pkr, seed=key)
        return new_state

    jfn = jax.jit(fn)

    t0 = time.time()
    lowered = jfn.lower(state, key)
    t_lower = time.time() - t0
    print(f"[{name}] .lower() time = {t_lower:.2f} s", flush=True)

    lo_text = lowered.as_text()
    lo_ops = _count_ops(lo_text)
    print(f"[{name}] lowered (StableHLO) text len = {lo_ops['len']}  "
          f"convolution={lo_ops['convolution']} dot={lo_ops['dot']} "
          f"while={lo_ops['while']}", flush=True)

    t0 = time.time()
    compiled = lowered.compile()
    t_compile = time.time() - t0
    print(f"[{name}] .compile() time = {t_compile:.2f} s", flush=True)

    try:
        co_text = compiled.as_text()
        co_ops = _count_ops(co_text)
        print(f"[{name}] compiled (optimized HLO) text len = {co_ops['len']}  "
              f"convolution={co_ops['convolution']} dot={co_ops['dot']} "
              f"while={co_ops['while']}", flush=True)
    except Exception as e:  # noqa: BLE001
        co_ops = None
        print(f"[{name}] compiled.as_text() unavailable: {e!r}", flush=True)

    try:
        ca = compiled.cost_analysis()
        if isinstance(ca, dict):
            flops = ca.get("flops")
        elif isinstance(ca, (list, tuple)) and ca and isinstance(ca[0], dict):
            flops = ca[0].get("flops")
        else:
            flops = None
        print(f"[{name}] cost_analysis flops = {flops}", flush=True)
    except Exception as e:  # noqa: BLE001
        flops = None
        print(f"[{name}] cost_analysis unavailable: {e!r}", flush=True)

    print(f"[{name}] SUMMARY lower={t_lower:.2f}s compile={t_compile:.2f}s "
          f"lowered_len={lo_ops['len']} conv={lo_ops['convolution']} "
          f"dot={lo_ops['dot']} while={lo_ops['while']} flops={flops}", flush=True)
    return dict(name=name, t_lower=t_lower, t_compile=t_compile,
               lowered=lo_ops, compiled=co_ops, flops=flops)


def _dassa(inner, getter=None, setter=None, logacc=None):
    kw = dict(inner_kernel=inner, num_adaptation_steps=NUM_ADAPT)
    if setter is not None:
        kw["step_size_setter_fn"] = setter
        kw["step_size_getter_fn"] = getter
        kw["log_accept_prob_getter_fn"] = logacc
    return tfp.mcmc.DualAveragingStepSizeAdaptation(**kw)


def build_kernel_A(tlp):
    """gigalens stack: PHMC(l=3) -> GBTLA(max_leapfrog=30) -> DASSA.
    Identity/diag momentum to isolate compile from NaN. Multi-chain: the
    target_log_prob_fn must accept a (n_chains, 74) batch, so vmap the v11f
    single-vector log-prob over the leading chain axis."""
    batched_tlp = jax.vmap(tlp)
    mom = L.momentum_distribution('diag')
    phmc = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=batched_tlp,
        momentum_distribution=mom,
        step_size=0.3,
        num_leapfrog_steps=3,
    )
    gbtla = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
        phmc,
        num_adaptation_steps=NUM_ADAPT,
        max_leapfrog_steps=30,
    )
    return _dassa(gbtla)


def build_kernel_B(tlp):
    """fixed-leapfrog: PHMC(l=16, momentum='inv') -> DASSA."""
    mom = L.momentum_distribution('inv')
    phmc = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=tlp,
        momentum_distribution=mom,
        step_size=0.3,
        num_leapfrog_steps=16,
    )
    return _dassa(phmc)


def build_kernel_C(tlp, qz_start):
    """reference v11f NUTS: PreconditionedNoUTurnSampler(max_tree_depth=6) -> DASSA."""
    # v11f used MultivariateNormalTriL(scale_tril=chol(jittered v10 SVI cov)).
    # Use momentum_distribution('fwd') which is the same MVN-with-scale_tril form
    # over the v11f empirical Sigma_hat (apples-to-apples covariance object).
    mom = L.momentum_distribution('fwd')
    nuts = tfe.mcmc.PreconditionedNoUTurnSampler(
        target_log_prob_fn=tlp,
        momentum_distribution=mom,
        step_size=0.5,
        max_tree_depth=6,
    )
    return _dassa(
        nuts,
        getter=lambda pkr: pkr.step_size,
        setter=lambda pkr, ss: pkr._replace(step_size=ss),
        logacc=lambda pkr: pkr.log_accept_ratio,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=["A", "B", "C", "BC"], default="BC")
    args = ap.parse_args()

    print(f"devices: {jax.devices()}", flush=True)
    m = L.build_model()
    tlp = m.target_log_prob_fn
    state = m.qz_start
    key = jax.random.PRNGKey(0)
    lp = tlp(state)
    lp.block_until_ready()
    print(f"ndim={m.ndim}  log_p(qz_start)={float(lp):.1f}", flush=True)

    if args.kernel == "A":
        # GradientBasedTrajectoryLengthAdaptation uses the ChEES criterion, which
        # requires >=2 chains, so it cannot even be lowered single-chain. The
        # original gigalens HMC ran a batch of chains per device (q_z.sample(
        # n_hmc//dev_cnt)). Replicate that with a small chain batch (2 = minimum)
        # to obtain a faithful single-device compile of the (A) stack.
        n_chains = 2
        state_A = jnp.broadcast_to(state, (n_chains, m.ndim))
        print(f"[A] using {n_chains} chains (ChEES needs >=2); "
              f"state_A shape={state_A.shape}", flush=True)
        measure("A", build_kernel_A(tlp), state_A, key)
    elif args.kernel == "B":
        measure("B", build_kernel_B(tlp), state, key)
    elif args.kernel == "C":
        measure("C", build_kernel_C(tlp, state), state, key)
    else:  # BC
        measure("B", build_kernel_B(tlp), state, key)
        measure("C", build_kernel_C(tlp, state), state, key)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
