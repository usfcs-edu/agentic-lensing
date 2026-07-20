"""Stage-checkpoint / resume / wall-cap machinery for the SMC runners.

P2 wave-1 post-mortem fix (research/p2_wave1_postmortem_redesign.md §5a, RC1 +
RC3): 18.69 GPU-h of the wave produced 1.12 h of readable science because the
runners had no per-stage observability, no checkpoints, and a summary-assembly
defect. This module is the approved retrofit, executed 2026-07-16.

Transplanted WITH ATTRIBUTION from 10_anchor_arbitration.py's accepted pattern
(LoggingKernel + WallCapReached wall-cap protocol + stage_resume), upgraded per
the approved plan:

  * `CheckpointKernel` — delegating wrapper around a validated MutationKernel
    (build/tune/mutate delegated verbatim; the wrapper only OBSERVES). Per
    stage: `stage_NNN.npz` full-state checkpoint + one progress print + a
    pre-registered wall-cap raise AFTER the checkpoint is written (exit-3
    PARTIAL protocol is the caller's job; no gate math on a non-lambda=1
    ensemble — the l0 protocol verbatim).
  * `LoglikRecorder` — weight-step log-likelihood recording through
    common.run_tempered_smc's EXISTING keyword-only `loglik_batch_fn` seam.
    Zero numerics impact (it returns the inner evaluator's output object
    untouched and stores a float64 copy — exactly the cast the driver applies).
    This is what lets a RESUMED run keep the FULL logZ + per-stage bootstrap
    (solves the l0 resume's `logZ_partial_from_resume` limitation — B1's
    <=2-nat logZ repeatability gate needs full logZ).
  * `run_tempered_smc_ckpt` — single entry point for the runners. Fresh runs
    go through the FROZEN driver `common.run_tempered_smc` (untouched) with
    the wrapper + recorder attached. `resume=True` continues from the newest
    stage checkpoint via a loop copied WITH ATTRIBUTION from the frozen
    driver, with the driver's jax key chain and numpy Generator FAST-FORWARDED
    BY REPLAY of the recorded weight-step log-likelihoods. A resumed run is
    therefore BIT-IDENTICAL to an uninterrupted run — particles, logZ,
    bootstrap sigma, every trace (gate: tests/test_ckpt_resume.py). This is
    STRONGER than the l0 resume (which took a fresh key segment and lost
    pre-resume logZ increments).

DTYPE CAVEAT (documented, ledgered with the retrofit): the checkpointed
ensemble is reconstructed from the (mu, chol, U_new) the kernel sees. Under
GIGALENS_X64=1 (all scene cells, B4, the CPU toys) that reconstruction is
bit-exact = the driver's own float64 update. Under GIGALENS_X64=0 (B5 only,
strict-f32 target) mu/chol reach the kernel as float32 casts, so a resumed
run is protocol-identical but carries an f32-rounding-level ensemble
perturbation at the resume point (same class as the ledgered l0 RNG caveat;
the bit-identity gate is stated for x64 runs).

RC3: `summarize_res` is the shared runner summary assembly — it pops
'particles'/'kernel' from a COPY of the driver result BEFORE the **-merge, so
`dict()` can never see a duplicate keyword again (the 55985449/50 crash class;
regression-locked in tests/test_ckpt_resume.py).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from cgl2.samplers import common

STAGE_FMT = "stage_{:03d}.npz"
LL_FMT = "ll_{:03d}.npy"


class WallCapReached(RuntimeError):
    """Pre-registered wall cap hit AFTER a stage checkpoint was written."""


def round_fence_reason(grads_billed: int, wall_s: float,
                       grad_budget: int | None,
                       wall_cap_s: float | None) -> str | None:
    """Round-boundary fence for ROUND-LOOP arms (B1r S6br Track-A
    budget-matched descope, research/checkpoint_b1_reduced.md).

    Called at the TOP of each round (never mid-round). Returns None to
    continue, else a stop-reason string the runner records in the summary and
    then writes its artifacts NORMALLY (graceful stop — the round loop has no
    stage checkpoints, so a slurm TIMEOUT there would repeat the RC1
    zero-artifact loss; this fence is the mitigation).

    * grad fence: stop once total BILLED grads (MAP warm-start + tune +
      mutate — the pre-registered S6b billing) have reached the budget; the
      realized budget overshoots the target by at most one round (billed at
      actuals by the harvest).
    * wall fence: pre-registered in-job cap strictly inside the slurm wall.
    Both fences default OFF (None/0) — the frozen 30+120-round design is
    untouched unless the descoped cell arms them explicitly.
    """
    if grad_budget and grads_billed >= int(grad_budget):
        return (f"grad_budget: billed {int(grads_billed)} >= "
                f"budget {int(grad_budget)}")
    if wall_cap_s and wall_s > float(wall_cap_s):
        return (f"wall_cap: {wall_s:.0f}s > {float(wall_cap_s):.0f}s "
                "(pre-registered in-job fence)")
    return None


def _jsonable(x):
    """Copied with attribution from 23_run_p2_scene.py (identical in 24)."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def summarize_res(res: dict, **base) -> dict:
    """RC3-safe runner summary assembly (shared by runners 23 and 24).

    Pops 'particles' and 'kernel' from a COPY of the driver result before the
    **-merge: `dict(kernel=..., **res)` with 'kernel' still inside res is
    exactly the `TypeError: dict() got multiple values for keyword argument
    'kernel'` that destroyed both B5 runs (jobs 55985449/50) AFTER their
    science completed. res['kernel'] is authoritative (mirrors runner 23's
    original `kernel=res.pop("kernel")`).
    """
    r = dict(res)
    r.pop("particles", None)
    out = dict(base)
    out["kernel"] = r.pop("kernel")
    out.update(_jsonable(r))
    return out


def _atomic_savez(path: Path, **payload) -> None:
    """np.savez via tmp+rename so a mid-write kill never leaves a truncated
    checkpoint (a poisoned newest-checkpoint would block resume). The tmp name
    (leading dot, .tmp suffix) never matches the stage_*/ll_* globs; writing
    through a file handle stops numpy appending its own extension."""
    tmp = path.parent / ("." + path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload)
    os.replace(tmp, path)


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.parent / ("." + path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def _stock_batch(loglik_fn):
    """The frozen driver's own default weight-step evaluator, built identically
    (common.run_tempered_smc line: jax.jit(jax.vmap(loglik_fn)))."""
    import jax

    return jax.jit(jax.vmap(loglik_fn))


class LoglikRecorder:
    """Weight-step ll recording through the loglik_batch_fn seam (delegation
    only: returns the inner evaluator's output object untouched)."""

    def __init__(self, ckpt_dir: Path, inner, start_stage: int = 0):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.inner = inner
        self.i = int(start_stage)

    def __call__(self, Z):
        out = self.inner(Z)
        # float64 copy == exactly the cast the driver applies to `out`
        _atomic_save(self.ckpt_dir / LL_FMT.format(self.i),
                     np.asarray(out, dtype=np.float64))
        self.i += 1
        return out


class CheckpointKernel(common.MutationKernel):
    """Delegating logging/checkpoint wrapper (10_anchor_arbitration.py
    LoggingKernel transplant; build/tune/mutate delegated verbatim).

    Per mutate call (== one lambda stage): write the full-state stage
    checkpoint, print one progress row, then (checkpoint already on disk)
    raise WallCapReached if the pre-registered wall cap is exceeded.
    `stop_after_stages` is the deterministic test hook for the bit-identity
    gate — it exercises the SAME raise-after-checkpoint code path.
    """

    def __init__(self, inner, ckpt_dir: Path, wall_cap_h: float | None = None,
                 label: str = "smc", start_stage: int = 0,
                 stop_after_stages: int | None = None):
        self.inner = inner
        self.name = inner.name + "+ckpt"
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.wall_cap_s = None if wall_cap_h is None else float(wall_cap_h) * 3600.0
        self.label = str(label)
        self.stage = int(start_stage)
        self.stop_after_stages = stop_after_stages
        self.trace = []
        self.t0 = time.time()
        self._t_tune = 0.0

    def build(self, logprior_fn, loglik_fn, dim):
        self.t0 = time.time()
        return self.inner.build(logprior_fn, loglik_fn, dim)

    def tune(self, key, lam, mu, chol, U_pilot, eps0):
        t = time.time()
        tuning = self.inner.tune(key, lam, mu, chol, U_pilot, eps0)
        self._t_tune = time.time() - t
        return tuning

    def mutate(self, key, lam, mu, chol, U, tuning):
        t = time.time()
        U_new, stats = self.inner.mutate(key, lam, mu, chol, U, tuning)
        t_mut = time.time() - t
        mu_np = np.asarray(mu, dtype=np.float64)
        chol_np = np.asarray(chol, dtype=np.float64)
        z = mu_np[None, :] + np.asarray(U_new, dtype=np.float64) @ chol_np.T
        _atomic_savez(
            self.ckpt_dir / STAGE_FMT.format(self.stage),
            z=z, lam=np.float64(lam),
            eps=np.float64(tuning.step_size), n_int=np.int64(tuning.n_int),
            accept=np.float64(stats["accept_mean"]),
            grad_tune=np.int64(tuning.n_grad),
            grad_mutate=np.int64(stats["n_grad"]),
            n_logp_tune=np.int64(tuning.n_logp),
            n_logp_mutate=np.int64(stats.get("n_logp", 0)))
        row = dict(stage=self.stage, lam=float(lam),
                   eps=float(tuning.step_size), n_int=int(tuning.n_int),
                   accept=float(stats["accept_mean"]),
                   t_tune_s=round(self._t_tune, 2), t_mutate_s=round(t_mut, 2),
                   wall_s=round(time.time() - self.t0, 1))
        self.trace.append(row)
        print(f"[{self.label}] {row}", flush=True)
        self.stage += 1
        if (self.stop_after_stages is not None
                and self.stage >= int(self.stop_after_stages)):
            raise WallCapReached(
                f"test hook: stopped after {self.stage} checkpointed stages "
                f"(lambda={lam:.6f}) — checkpoints kept")
        if (self.wall_cap_s is not None
                and time.time() - self.t0 > self.wall_cap_s):
            raise WallCapReached(
                f"pre-registered wall cap {self.wall_cap_s / 3600:.2f} h "
                f"reached at stage {self.stage}, lambda={lam:.6f} — "
                "checkpoints kept")
        return U_new, stats


def _run_meta(z0, kernel, target_ess, max_stages, n_boot, pilot_size, eps0,
              precondition, boot_seed, loglik_batch_fn):
    z = np.asarray(z0)
    return dict(n=int(z.shape[0]), dim=int(z.shape[1]),
                target_ess=float(target_ess), max_stages=int(max_stages),
                n_boot=int(n_boot), pilot_size=int(pilot_size),
                eps0=float(eps0), precondition=bool(precondition),
                boot_seed=int(boot_seed), kernel=str(kernel.name),
                chunked_loglik=bool(loglik_batch_fn is not None))


def run_tempered_smc_ckpt(logprior_fn, loglik_fn, z0_particles, key, *,
                          kernel, ckpt_dir, wall_cap_h: float | None = None,
                          resume: bool = False,
                          target_ess: float = 0.7, max_stages: int = 400,
                          n_boot: int = 200, pilot_size: int = 64,
                          eps0: float = 0.5, precondition: bool = True,
                          boot_seed: int = 20260715, loglik_batch_fn=None,
                          label: str = "smc", kern_holder: list | None = None,
                          stop_after_stages: int | None = None) -> dict:
    """Checkpointed/resumable front-end to the FROZEN common.run_tempered_smc.

    Fresh path: the frozen driver runs untouched with the delegating
    CheckpointKernel + LoglikRecorder attached (bit-identical to a stock run;
    gate in tests/test_ckpt_resume.py). Raises WallCapReached per the l0
    exit-3 PARTIAL protocol (checkpoints already on disk).

    resume=True with stage checkpoints present: continue from the newest
    checkpoint, bit-identical to the uninterrupted run (see module docstring);
    with an empty checkpoint dir it falls back to a fresh run (first-leg
    semantics). `key` must be the run's ORIGINAL PRNGKey in both cases.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cks = sorted(ckpt_dir.glob("stage_*.npz"))
    if cks and not resume:
        raise RuntimeError(
            f"{ckpt_dir} already holds {len(cks)} stage checkpoints — pass "
            "--resume to continue that run, or point --ckpt-dir at a fresh "
            "directory (refusing to mix runs)")
    if resume and cks:
        return _resume_tempered_smc(
            logprior_fn, loglik_fn, z0_particles, key, kernel=kernel,
            ckpt_dir=ckpt_dir, wall_cap_h=wall_cap_h, target_ess=target_ess,
            max_stages=max_stages, n_boot=n_boot, pilot_size=pilot_size,
            eps0=eps0, precondition=precondition, boot_seed=boot_seed,
            loglik_batch_fn=loglik_batch_fn, label=label,
            kern_holder=kern_holder, stop_after_stages=stop_after_stages)
    if resume:
        print(f"[{label}] --resume with no stage checkpoints in {ckpt_dir} "
              "— running fresh", flush=True)

    meta = _run_meta(z0_particles, kernel, target_ess, max_stages, n_boot,
                     pilot_size, eps0, precondition, boot_seed,
                     loglik_batch_fn)
    (ckpt_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    wrapped = CheckpointKernel(kernel, ckpt_dir, wall_cap_h, label=label,
                               stop_after_stages=stop_after_stages)
    if kern_holder is not None:
        kern_holder.append(wrapped)
    rec = LoglikRecorder(
        ckpt_dir, loglik_batch_fn if loglik_batch_fn is not None
        else _stock_batch(loglik_fn))
    return common.run_tempered_smc(
        logprior_fn, loglik_fn, z0_particles, key, kernel=wrapped,
        target_ess=target_ess, max_stages=max_stages, n_boot=n_boot,
        pilot_size=pilot_size, eps0=eps0, precondition=precondition,
        boot_seed=boot_seed, loglik_batch_fn=rec)


def _resume_tempered_smc(logprior_fn, loglik_fn, z0_particles, key, *,
                         kernel, ckpt_dir, wall_cap_h, target_ess, max_stages,
                         n_boot, pilot_size, eps0, precondition, boot_seed,
                         loglik_batch_fn, label, kern_holder,
                         stop_after_stages) -> dict:
    """Resume from the newest stage checkpoint, BIT-IDENTICAL to an
    uninterrupted run.

    The stage loop and bootstrap below are copied WITH ATTRIBUTION from
    cgl2.samplers.common.run_tempered_smc (the frozen driver, untouched) —
    identical per-stage semantics, identical constants. The pre-resume replay
    reconstructs every driver-visible quantity (delta/lambda/logZ increments/
    ESS/resample/unique counts) from the RECORDED weight-step log-likelihoods
    and fast-forwards the driver's jax key chain and numpy Generator by
    performing the SAME split/choice calls in the SAME order — no forward
    model evaluations are spent on completed stages.
    """
    import jax
    import jax.numpy as jnp
    from scipy.linalg import solve_triangular

    ckpt_dir = Path(ckpt_dir)
    cks = sorted(ckpt_dir.glob("stage_*.npz"))
    n_done = len(cks)
    for m, f in enumerate(cks):  # contiguity fence
        if f.name != STAGE_FMT.format(m):
            raise RuntimeError(f"non-contiguous checkpoints in {ckpt_dir}: "
                               f"expected {STAGE_FMT.format(m)}, found {f.name}")
    meta_path = ckpt_dir / "run_meta.json"
    if meta_path.exists():  # desync fence: resume kwargs must match the run
        meta = json.loads(meta_path.read_text())
        want = _run_meta(z0_particles, kernel, target_ess, max_stages, n_boot,
                         pilot_size, eps0, precondition, boot_seed,
                         loglik_batch_fn)
        for k in ("n", "dim", "target_ess", "n_boot", "pilot_size", "eps0",
                  "precondition", "boot_seed"):
            if meta.get(k) != want[k]:
                raise RuntimeError(
                    f"resume config desync on '{k}': checkpointed run has "
                    f"{meta.get(k)!r}, this invocation {want[k]!r}")

    last = np.load(cks[-1])
    z = np.asarray(last["z"], dtype=np.float64)
    n, dim = z.shape
    if z0_particles is not None:
        z0 = np.asarray(z0_particles)
        if z0.shape != (n, dim):
            raise RuntimeError(f"checkpoint ensemble {(n, dim)} != z0 shape "
                               f"{z0.shape} — wrong run directory?")

    lls = []
    for m in range(n_done):
        f = ckpt_dir / LL_FMT.format(m)
        if not f.exists():
            raise RuntimeError(f"missing recorded weight-step loglik {f} — "
                               "cannot rebuild full logZ (resume refused)")
        ll = np.load(f)
        if ll.shape != (n,):
            raise RuntimeError(f"{f}: shape {ll.shape} != ({n},)")
        lls.append(ll)

    # ---- pre-resume replay (cheap numpy/PRNG ops only) -----------------------
    rng = np.random.default_rng(boot_seed)
    lam = 0.0
    logz = 0.0
    stage_deltas, stage_logliks, increments = [], [], []
    lambdas, ess_trace, uniq_trace = [], [], []
    acc_trace, eps_trace, nint_trace = [], [], []
    grad_evals = dict(tune=0, mutate=0)
    n_logp = 0
    for m in range(n_done):
        ck = np.load(cks[m])
        ll = lls[m]
        n_logp += n
        delta = common.solve_delta_lambda(ll, lam, target_ess)
        lam_new = min(1.0, lam + delta)
        inc = common.logmeanexp(delta * ll)
        logz += inc
        increments.append(inc)
        stage_deltas.append(delta)
        stage_logliks.append(ll)
        logw = delta * ll
        w = np.exp(logw - logw.max())
        w = w / w.sum()
        ess_trace.append(common.weight_ess(w))
        key, k_u = jax.random.split(key)
        u_draw = float(jax.random.uniform(k_u))
        idx = common.systematic_resample(w, u_draw)
        uniq_trace.append(int(len(np.unique(idx))))
        key, k_pilot, k_mut = jax.random.split(key, 3)
        rng.choice(n, size=min(pilot_size, n), replace=False)  # fast-forward
        if float(ck["lam"]) != lam_new:  # replay/checkpoint identity fence
            raise RuntimeError(
                f"replay desync at stage {m}: rebuilt lambda {lam_new!r} != "
                f"checkpointed {float(ck['lam'])!r} — ll files and stage "
                "checkpoints are not from the same run")
        acc_trace.append(float(ck["accept"]))
        eps_trace.append(float(ck["eps"]))
        nint_trace.append(int(ck["n_int"]))
        grad_evals["tune"] += int(ck["grad_tune"])
        grad_evals["mutate"] += int(ck["grad_mutate"])
        n_logp += int(ck["n_logp_tune"]) + int(ck["n_logp_mutate"])
        lam = lam_new
        lambdas.append(lam)
    eps_warm = float(last["eps"])
    n_stages = n_done
    print(f"[{label}] resumed from stage {n_done - 1}: lam={lam:.6e}, "
          f"eps={eps_warm:.4f}, N={n} (logZ so far {logz:.3f}, full-history "
          "bootstrap retained)", flush=True)

    wrapped = CheckpointKernel(kernel, ckpt_dir, wall_cap_h, label=label,
                               start_stage=n_done,
                               stop_after_stages=stop_after_stages)
    if kern_holder is not None:
        kern_holder.append(wrapped)
    wrapped.build(logprior_fn, loglik_fn, dim)
    loglik_batch = LoglikRecorder(
        ckpt_dir, loglik_batch_fn if loglik_batch_fn is not None
        else _stock_batch(loglik_fn), start_stage=n_done)

    # ---- stage loop copied WITH ATTRIBUTION from common.run_tempered_smc ----
    while lam < 1.0 and n_stages < max_stages:
        ll = np.asarray(loglik_batch(jnp.asarray(z)), dtype=np.float64)
        if not np.all(np.isfinite(ll)):
            bad = int(np.sum(~np.isfinite(ll)))
            raise RuntimeError(
                f"stage {n_stages}: {bad}/{n} non-finite log-likelihoods at the "
                "current particle set -- prior draws outside the likelihood domain?")
        n_logp += n

        delta = common.solve_delta_lambda(ll, lam, target_ess)
        lam_new = min(1.0, lam + delta)
        inc = common.logmeanexp(delta * ll)
        logz += inc
        increments.append(inc)
        stage_deltas.append(delta)
        stage_logliks.append(ll.copy())

        logw = delta * ll
        w = np.exp(logw - logw.max())
        w = w / w.sum()
        ess_trace.append(common.weight_ess(w))

        if precondition:
            mu, cov = common.weighted_mean_cov(z, w)
            cov_reg = common.regularize_cov(cov, common.weight_ess(w))
            chol = np.linalg.cholesky(cov_reg)
        else:
            mu = np.zeros(dim)
            chol = np.eye(dim)

        key, k_u = jax.random.split(key)
        u_draw = float(jax.random.uniform(k_u))
        idx = common.systematic_resample(w, u_draw)
        uniq_trace.append(int(len(np.unique(idx))))
        z = z[idx]

        mu_j = jnp.asarray(mu)
        chol_j = jnp.asarray(chol)
        U = jnp.asarray(solve_triangular(chol, (z - mu[None, :]).T,
                                         lower=True).T)

        key, k_pilot, k_mut = jax.random.split(key, 3)
        pilot_idx = rng.choice(n, size=min(pilot_size, n), replace=False)
        tuning = wrapped.tune(k_pilot, lam_new, mu_j, chol_j, U[pilot_idx],
                              eps_warm)
        grad_evals["tune"] += tuning.n_grad
        n_logp += tuning.n_logp
        eps_warm = tuning.step_size

        U_new, stats = wrapped.mutate(k_mut, lam_new, mu_j, chol_j, U, tuning)
        grad_evals["mutate"] += int(stats["n_grad"])
        n_logp += int(stats.get("n_logp", 0))
        z = np.asarray(mu[None, :]
                       + np.asarray(U_new, dtype=np.float64) @ chol.T)

        acc_trace.append(float(stats["accept_mean"]))
        eps_trace.append(float(tuning.step_size))
        nint_trace.append(int(tuning.n_int))
        lam = lam_new
        lambdas.append(lam)
        n_stages += 1

    if lam < 1.0:
        raise RuntimeError(
            f"tempered SMC did not reach lambda=1 in {max_stages} stages "
            f"(lambda={lam:.6f}); raise max_stages or target_ess")

    # ---- particle bootstrap over the FULL stage history (driver verbatim) ----
    boots = np.empty(n_boot)
    for b in range(n_boot):
        tot = 0.0
        for delta, ll in zip(stage_deltas, stage_logliks):
            bidx = rng.integers(0, n, n)
            tot += common.logmeanexp(delta * ll[bidx])
        boots[b] = tot
    logz_boot_sigma = float(np.std(boots))

    grad_evals["total"] = grad_evals["tune"] + grad_evals["mutate"]
    return dict(
        particles=z,
        logZ=float(logz),
        logZ_boot_sigma=logz_boot_sigma,
        logZ_increments=increments,
        lambda_schedule=lambdas,
        ess_trace=ess_trace,
        unique_particle_trace=uniq_trace,
        accept_trace=acc_trace,
        step_size_trace=eps_trace,
        n_int_trace=nint_trace,
        grad_evals=grad_evals,
        n_logp=n_logp,
        n_stages=n_stages,
        kernel=wrapped.name,
        resumed_from_stage=n_done - 1,
    )
