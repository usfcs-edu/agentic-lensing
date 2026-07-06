"""S6 adapter: nautilus (neural-network-aided nested/importance sampling).

Gradient-free contender. Uses the zoo's unit-cube face: prior_transform
([0,1]^dim -> PHYSICAL x in bijector-leaf order) + log_like_x (data
likelihood at physical x), both present on T0/T1 targets. vectorized=True
with a fixed n_batch (the gigalens targets build one LensSimulator per batch
size; nautilus's filler keeps batches at n_batch).

Budget protocol: gradient-free methods get 2x the track's gradient budget in
LIKELIHOOD evals, enforced via run(n_like_max=...). For the ESS/grad tables
the ledger books n_grad = ceil(n_like / 2) as GRADIENT-EQUIVALENTS
(phase "grad_equiv"; the raw n_like is booked as n_logp) -- the
pre-registered 2-like-evals = 1-grad exchange rate.

Outputs: log_z + equal-weight posterior draws, mapped back to unconstrained
z (identity for T0; bijector inverse via the leaf-order nested rebuild for
T1, self-checked against log_like_batch at 8 points before use).

ESS semantics: equal-weight draws are exchangeable -> efficiency uses
sampler.n_eff (nautilus's importance ESS), capped by the number of unique
equal-weight points; rank diagnostics on shuffled pseudo-chains stored for
reference only.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cgl import guards, metrics
from cgl.io import CellResult
from cgl.samplers import common
from cgl.zoo.api import LensPosterior, np_dtype

SCALE_KEYS = ("n_like_max", "n_eff")

DEFAULT_CONFIG = dict(
    n_live=1500,
    n_networks=4,
    n_batch=512,
    f_live=0.01,
    discard_exploration=False,
    n_pseudo_chains=8,
    xz_check_tol=2e-3,             # rel tol, f32 sim round-trip
)

DEFAULT_BUDGET = dict(n_eff=3000, n_like_max=None)   # None -> 2x grad budget


def _probe_template_perm(target: LensPosterior):
    """Map each nested-template leaf (dict-iteration order) to its z index.

    x columns are in LABELS order (probed z-leaf order; prior_transform
    builds column i from leaf_dists[labels[i]]), while the tfp bijector's
    nested dicts iterate in the REORDERED key order (the documented
    e2-before-e1 trap) -- positional filling is wrong. Probe: perturb one
    nested leaf at a time and see which z coordinate bij.inverse moves.
    """
    import jax.numpy as jnp

    fdtype = np_dtype(target.dtype)
    bij = target.bijector
    dim = target.dim
    base_z = np.full(dim, 0.3, dtype=fdtype)          # interior point
    nested0 = bij.forward([jnp.asarray(base_z[i][None]) for i in range(dim)])
    z0 = np.asarray(jnp.stack(bij.inverse(nested0)),
                    dtype=np.float64).ravel()

    # flatten template values (dict-iteration order) with paths
    paths = []
    for b, block in enumerate(nested0):
        for c, comp in enumerate(block):
            for key_name in comp:
                paths.append((b, c, key_name))

    perm = []
    for (b, c, key_name) in paths:
        nested_p = [[{k: (jnp.asarray(v) if not (bb == b and cc == c
                                                 and k == key_name)
                          else jnp.asarray(v)
                          * (1.0 + np.asarray(0.03, dtype=fdtype))
                          + np.asarray(0.01, dtype=fdtype))
                      for k, v in comp2.items()}
                     for cc, comp2 in enumerate(block2)]
                    for bb, block2 in enumerate(nested0)]
        z_p = np.asarray(jnp.stack(bij.inverse(nested_p)),
                         dtype=np.float64).ravel()
        moved = np.where(np.abs(z_p - z0) > 1e-5)[0]
        if len(moved) != 1:
            raise RuntimeError(
                f"template-leaf probe ambiguous for {(b, c, key_name)}: "
                f"moved z indices {moved.tolist()}")
        perm.append(int(moved[0]))
    assert sorted(perm) == list(range(dim))
    return paths, perm


def _x_to_z(target: LensPosterior, X: np.ndarray) -> np.ndarray:
    """Physical leaf-order x (labels/z order columns) -> unconstrained z."""
    if target.bijector is None:
        return np.asarray(X, dtype=np.float64)
    import jax.numpy as jnp

    fdtype = np_dtype(target.dtype)
    bij = target.bijector
    X = np.atleast_2d(np.asarray(X, dtype=fdtype))
    paths, perm = _probe_template_perm(target)
    filled = {}
    for (b, c, key_name), j in zip(paths, perm):
        filled.setdefault(b, {}).setdefault(c, {})[key_name] = \
            jnp.asarray(X[:, j], dtype=fdtype)
    nested = [[filled[b][c] for c in sorted(filled[b])]
              for b in sorted(filled)]
    z = jnp.stack(bij.inverse(nested)).T
    return np.asarray(z, dtype=np.float64)


def run_cell(target: LensPosterior, seed: int, budget: Optional[dict] = None,
             config: Optional[dict] = None,
             freeze_points: Optional[list] = None) -> CellResult:
    from nautilus import Sampler

    guards.require_single_device()
    budget = {**DEFAULT_BUDGET, **(budget or {})}
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    fdtype = np_dtype(target.dtype)
    if target.dtype == "float64":
        guards.require_x64()
    if target.prior_transform is None or target.log_like_x is None:
        raise RuntimeError(
            f"target {target.name} lacks the unit-cube face "
            "(prior_transform/log_like_x); nautilus cannot run it")

    freeze_check = common.assert_freeze_fidelity(target, freeze_points, fdtype)
    ledger = metrics.BudgetLedger()
    timing = {}
    extras = {}
    dim = target.dim

    n_like_max = budget.get("n_like_max")
    if not n_like_max:
        gb = budget.get("n_grad_budget")
        n_like_max = int(2 * gb) if gb else 10**9
    n_like_max = int(n_like_max)

    # ---- fixed-batch likelihood face --------------------------------------------------
    # ZOO CONTRACT TRAP (P2b finding): the T1 log_like_x face is fully
    # jitted, so its FIRST call at a new batch size constructs a
    # LensSimulator inside the trace -> TracerArrayConversionError. The
    # eager log_prob_batch path shares the same per-batch-size sim cache,
    # so warming n_batch through it once and PADDING every likelihood call
    # to exactly n_batch keeps the jitted face on the cached simulator
    # (nautilus also sends short remainder batches, hence the pad).
    n_batch = int(cfg["n_batch"])
    common.warmup_batch(target, 8, n_batch)   # 8 = the xz self-check batch

    def likelihood(x):
        x = np.atleast_2d(np.asarray(x))
        n = x.shape[0]
        if n > n_batch:                      # defensive chunking
            return np.concatenate([likelihood(x[i:i + n_batch])
                                   for i in range(0, n, n_batch)])
        if n < n_batch:
            xp = np.concatenate(
                [x, np.repeat(x[-1:], n_batch - n, axis=0)], axis=0)
            return np.asarray(target.log_like_x(xp),
                              dtype=np.float64)[:n]
        return np.asarray(target.log_like_x(x), dtype=np.float64)

    # ---- x<->z consistency self-check (8 prior-cube points) -------------------------
    rng = np.random.default_rng(int(seed) + 6_20260706)
    u_chk = rng.uniform(0.05, 0.95, size=(8, dim))
    x_chk = np.atleast_2d(np.asarray(target.prior_transform(u_chk)))
    z_chk = _x_to_z(target, x_chk)
    ll_x = likelihood(x_chk).ravel()
    import jax.numpy as jnp
    ll_z = np.asarray(target.log_like_batch(
        jnp.asarray(z_chk, dtype=fdtype)), dtype=np.float64).ravel()
    rel = np.abs(ll_x - ll_z) / np.maximum(np.abs(ll_x), 1.0)
    extras["xz_roundtrip_max_rel"] = float(rel.max())
    if rel.max() > float(cfg["xz_check_tol"]):
        raise RuntimeError(
            f"x<->z face inconsistency on {target.name}: max rel "
            f"{rel.max():.3e} > {cfg['xz_check_tol']}")

    # ---- nautilus run ------------------------------------------------------------------

    t0 = time.time()
    sampler = Sampler(
        target.prior_transform, likelihood, n_dim=dim,
        n_live=int(cfg["n_live"]), n_networks=int(cfg["n_networks"]),
        n_batch=int(cfg["n_batch"]), vectorized=True, pass_dict=False,
        seed=int(seed))
    success = sampler.run(
        f_live=float(cfg["f_live"]), n_eff=float(budget["n_eff"]),
        n_like_max=n_like_max,
        discard_exploration=bool(cfg["discard_exploration"]), verbose=False)
    timing["nautilus_s"] = time.time() - t0

    n_like = int(sampler.n_like)
    ledger.add("grad_equiv", n_grad=int(np.ceil(n_like / 2)),
               note="GRADIENT-EQUIVALENTS: nautilus is gradient-free; "
                    "protocol books 2 likelihood evals = 1 grad")
    ledger.add("likelihood", n_logp=n_like,
               note=f"raw likelihood evals (cap {n_like_max})")
    extras.update(run_success=bool(success), n_like=n_like,
                  n_like_max=n_like_max, n_eff=float(sampler.n_eff),
                  log_z=float(sampler.log_z), pseudo_chains=True)
    if not success:
        extras["run_incomplete"] = ("n_like_max or timeout hit before "
                                    "convergence criteria")
    if target.reference is not None and target.reference.logZ is not None:
        extras["logz_compare"] = metrics.compare_logZ(
            float(sampler.log_z), target.reference.logZ)

    # ---- posterior draws -> z-space pseudo-chains -----------------------------------------
    points, log_w, log_l = sampler.posterior(equal_weight=True)
    n_pts = points.shape[0]
    z = _x_to_z(target, points)
    n_unique = int(np.unique(points[:, 0]).size)
    ess_est = float(min(float(sampler.n_eff), n_unique))
    samples = common.particles_to_chains(z, int(cfg["n_pseudo_chains"]), rng)
    extras["n_posterior_points"] = int(n_pts)
    extras["n_unique_points"] = n_unique

    diagnostics = dict(log_l_posterior=np.asarray(log_l))
    return common.finalize_cell(
        sampler="nautilus_runner", target=target, seed=seed, budget=budget,
        config=cfg, samples=samples, diagnostics=diagnostics, ledger=ledger,
        timing=timing, freeze_check=freeze_check, extras=extras,
        ess_override=dict(source="nautilus_n_eff_min_unique", ess=ess_est))
