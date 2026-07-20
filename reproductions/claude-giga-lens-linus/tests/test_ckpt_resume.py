"""CPU unit gates for the P2 wave-1 post-mortem retrofit (cgl2.samplers.ckpt).

Pre-registered coverage (research/p2_wave1_postmortem_redesign.md §5a +
the 2026-07-16 recovery-lane tasking):
  (1) RC3 regression: the runner summary assembly can never again crash on a
      driver result containing 'kernel' (jobs 55985449/50 crash class);
  (2) delegation-only proof: stock frozen driver vs ckpt-instrumented run are
      BIT-IDENTICAL (wrapper + ll recorder observe, never touch numerics);
  (3) resume bit-identity gate: fresh run vs interrupted+resumed run give
      identical particles/logZ (and bootstrap sigma, traces, ledgers) on a
      CPU toy — B1's <=2-nat logZ repeatability gate needs full-logZ resume;
  (4) resume-after-completion rebuilds the identical result with zero new
      forward evaluations (the exact B5 lost-at-the-write recovery path).
All CPU toys (conftest pins JAX_PLATFORMS=cpu, x64 ON).
"""
import numpy as np
import pytest

from cgl2 import zoo
from cgl2.samplers import ckpt, common, smc_micro

TRACE_KEYS = ("lambda_schedule", "ess_trace", "unique_particle_trace",
              "accept_trace", "step_size_trace", "n_int_trace",
              "logZ_increments")
SEED = 3
N = 128


def _kwargs():
    # target_ess 0.95 (vs the frozen production 0.7) purely to force a longer
    # lambda schedule on the cheap toy (gauss2 anneals in 2 stages at 0.7 —
    # too short for an INTERIOR interrupt); machinery under test is identical.
    return dict(target_ess=0.95, max_stages=400, n_boot=50,
                pilot_size=16, eps0=0.5, precondition=True,
                boot_seed=20260715 + SEED)


def _setup():
    import jax

    t = zoo.build_gauss2()
    z0 = np.asarray(t.prior_sample_fn(jax.random.PRNGKey(7), N))
    return t, z0


def _kern():
    return smc_micro.make_kernel("mams", num_mcmc_steps=smc_micro.NUM_MCMC_STEPS)


def _key():
    import jax

    return jax.random.PRNGKey(SEED)


def _assert_bit_identical(a, b):
    """Exact equality of everything the harvest quotes (no tolerances)."""
    assert np.array_equal(a["particles"], b["particles"])
    assert a["logZ"] == b["logZ"]
    assert a["logZ_boot_sigma"] == b["logZ_boot_sigma"]
    for k in TRACE_KEYS:
        assert a[k] == b[k], k
    assert a["grad_evals"] == b["grad_evals"]
    assert a["n_logp"] == b["n_logp"]
    assert a["n_stages"] == b["n_stages"]


# --------------------------------------------------------------------------- #
# (1) RC3 regression: summary assembly with 'kernel' still in res
# --------------------------------------------------------------------------- #
def test_rc3_summary_kernel_collision_unreproducible():
    fake_res = dict(particles=np.zeros((4, 2)), kernel="mams+ckpt",
                    logZ=-1.25, logZ_boot_sigma=0.1, n_stages=3,
                    grad_evals=dict(tune=1, mutate=2, total=3),
                    lambda_schedule=[0.1, 0.5, 1.0])
    # the pre-fix pattern is EXACTLY the 55985449/50 crash (documented class)
    with pytest.raises(TypeError):
        dict(script="24_run_p2_oldstack.py", kernel="from-kern-name",
             **fake_res)
    out = ckpt.summarize_res(fake_res, script="24_run_p2_oldstack.py",
                             cell="b5", wall_s=1.0)
    assert out["kernel"] == "mams+ckpt"          # res is authoritative
    assert "particles" not in out                # never json-dumped
    assert out["logZ"] == -1.25 and out["n_stages"] == 3
    assert out["grad_evals"] == dict(tune=1, mutate=2, total=3)
    # copy semantics: the caller's res is untouched
    assert "kernel" in fake_res and "particles" in fake_res


# --------------------------------------------------------------------------- #
# (2) stock frozen driver vs ckpt-instrumented run: exact-zero difference
# --------------------------------------------------------------------------- #
def test_ckpt_instrumentation_bit_identity_vs_stock(tmp_path):
    t, z0 = _setup()
    res_stock = common.run_tempered_smc(
        t.logprior_fn, t.loglik_fn, z0, _key(), kernel=_kern(), **_kwargs())
    d = tmp_path / "ck"
    res_ck = ckpt.run_tempered_smc_ckpt(
        t.logprior_fn, t.loglik_fn, z0, _key(), kernel=_kern(), ckpt_dir=d,
        **_kwargs())
    _assert_bit_identical(res_stock, res_ck)
    assert res_ck["kernel"] == res_stock["kernel"] + "+ckpt"
    # RC1 observability: one full-state checkpoint + one recorded ll per stage
    n = res_ck["n_stages"]
    assert len(sorted(d.glob("stage_*.npz"))) == n
    assert len(sorted(d.glob("ll_*.npy"))) == n
    # recorded lls reproduce the quoted evidence increments exactly
    ck0 = np.load(d / "stage_000.npz")
    assert set(ck0.files) >= {"z", "lam", "eps", "n_int", "accept",
                              "grad_tune", "grad_mutate"}
    # a second fresh run into a non-empty dir must refuse (no run mixing)
    with pytest.raises(RuntimeError, match="stage checkpoints"):
        ckpt.run_tempered_smc_ckpt(
            t.logprior_fn, t.loglik_fn, z0, _key(), kernel=_kern(),
            ckpt_dir=d, **_kwargs())


# --------------------------------------------------------------------------- #
# (3) THE GATE: fresh vs interrupted+resumed — identical particles/logZ
# --------------------------------------------------------------------------- #
def test_resume_bit_identity_after_interrupt(tmp_path):
    t, z0 = _setup()
    res_full = ckpt.run_tempered_smc_ckpt(
        t.logprior_fn, t.loglik_fn, z0, _key(), kernel=_kern(),
        ckpt_dir=tmp_path / "full", **_kwargs())
    assert res_full["n_stages"] >= 3, "toy too easy for an interior interrupt"

    d = tmp_path / "intr"
    with pytest.raises(ckpt.WallCapReached):  # deterministic wall-cap path
        ckpt.run_tempered_smc_ckpt(
            t.logprior_fn, t.loglik_fn, z0, _key(), kernel=_kern(),
            ckpt_dir=d, stop_after_stages=2, **_kwargs())
    assert len(sorted(d.glob("stage_*.npz"))) == 2  # progress preserved

    res_res = ckpt.run_tempered_smc_ckpt(
        t.logprior_fn, t.loglik_fn, z0, _key(), kernel=_kern(),
        ckpt_dir=d, resume=True, **_kwargs())
    assert res_res["resumed_from_stage"] == 1
    _assert_bit_identical(res_full, res_res)  # incl. FULL logZ + boot sigma

    # config-desync fence: resuming with a different boot_seed must refuse
    kw = _kwargs()
    kw["boot_seed"] += 1
    with pytest.raises(RuntimeError, match="desync"):
        ckpt.run_tempered_smc_ckpt(
            t.logprior_fn, t.loglik_fn, z0, _key(), kernel=_kern(),
            ckpt_dir=d, resume=True, **kw)


# --------------------------------------------------------------------------- #
# (4) resume after completion: the B5 lost-at-the-write recovery path
# --------------------------------------------------------------------------- #
def test_resume_after_completion_rebuilds_identical(tmp_path):
    t, z0 = _setup()
    d = tmp_path / "done"
    res1 = ckpt.run_tempered_smc_ckpt(
        t.logprior_fn, t.loglik_fn, z0, _key(), kernel=_kern(), ckpt_dir=d,
        **_kwargs())
    n_ll = len(sorted(d.glob("ll_*.npy")))
    res2 = ckpt.run_tempered_smc_ckpt(
        t.logprior_fn, t.loglik_fn, z0, _key(), kernel=_kern(), ckpt_dir=d,
        resume=True, **_kwargs())
    assert res2["resumed_from_stage"] == res1["n_stages"] - 1
    _assert_bit_identical(res1, res2)
    # zero new weight-step evaluations were spent
    assert len(sorted(d.glob("ll_*.npy"))) == n_ll


# --------------------------------------------------------------------------- #
# (5) P2b B1-REDUCED round-boundary fences (research/checkpoint_b1_reduced.md):
#     pure-logic gate for the S6br Track-A budget-matched stop rule — both
#     fences default OFF (the frozen 30+120 design is untouched), the grad
#     fence stops AT/AFTER budget only, the wall fence strictly after the cap.
# --------------------------------------------------------------------------- #
def test_round_fence_defaults_off_and_semantics():
    # defaults (unset env => budget None/0, cap None/0): never stops
    assert ckpt.round_fence_reason(10**12, 10**9, None, None) is None
    assert ckpt.round_fence_reason(10**12, 10**9, 0, 0) is None
    # grad fence: strictly below budget continues; at/over budget stops
    assert ckpt.round_fence_reason(999, 0.0, 1000, None) is None
    assert ckpt.round_fence_reason(1000, 0.0, 1000, None) is not None
    assert "grad_budget" in ckpt.round_fence_reason(2000, 0.0, 1000, None)
    # wall fence: at the cap continues (stop is strict >); past it stops
    assert ckpt.round_fence_reason(0, 3600.0, None, 3600.0) is None
    assert "wall_cap" in ckpt.round_fence_reason(0, 3600.1, None, 3600.0)
    # grad fence is checked first (budget-matched stop reported as such even
    # when both fences are tripped)
    assert "grad_budget" in ckpt.round_fence_reason(1000, 9999.0, 1000, 1.0)
