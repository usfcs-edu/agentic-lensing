"""Lens-posterior zoo API: the target contract every sampler adapter codes to.

A zoo target is a `LensPosterior`: a posterior over an UNCONSTRAINED z-space
with a batched log-density as the PRIMARY surface, an explicit prior/likelihood
split (log_prior includes the bijector forward log-det-Jacobian, so
log_like = log_prob - log_prior is the pure data term -- this is what makes
likelihood-only tempering (SMC/PT) and z-space evidence well defined), an
InitBundle (MAP / floored SVI / prior-sampling starts), and an optional
Reference (stored chains / analytics, ALWAYS with provenance).

Conventions (frozen for P2; P2b adapters build against exactly this):
  * z is a flat float vector, length `dim`; `labels[k]` names the PHYSICAL
    leaf that z index k maps to through `bijector` (bijector-leaf order --
    the documented gu-2022 reordering trap; never assume declaration order).
  * `log_prob_batch(Z: (N, dim)) -> (N,)` is primary; `log_prob(z)` derives
    from it. Both are jax-jittable closures returning the target's dtype.
  * `log_prob = log_prior + log_like` exactly (unnormalized posterior;
    the evidence is NOT subtracted). `reference.logZ`, where present, is
    log integral exp(log_like) * exp(log_prior) dz.
  * dtype is per-target ("float32"|"float64") and one process cannot mix:
    float64 targets require GIGALENS_X64=1 before jax import; float32
    tfp-based targets (T1/T3) must run with x64 OFF. Driver scripts are
    single-target-per-process.
  * BATCH-SIZE WARMUP CONTRACT: gigalens-backed targets (T1/T3) build a
    LensSimulator per batch size on first call. Constructing one inside a
    jit/scan/sample_chain trace is a TracerArrayConversionError, so every
    adapter must call log_prob_batch ONCE EAGERLY with the final batch shape
    before closing it into a jitted kernel (cgl.fitting and the S0 HMC phase
    already do this).
  * noise_model is "diag" for every P2a target; "correlated" is the P1 hook.

This module is import-light: numpy/stdlib only at import time; jax is only
touched inside builders/methods.
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Callable, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# dtype / env helpers
# --------------------------------------------------------------------------- #
def assert_dtype_env(dtype: str, *, strict_f32: bool = False) -> None:
    """Fail fast when the process x64 state cannot represent the target.

    float64 targets need jax_enable_x64 (GIGALENS_X64=1 before jax import).
    tfp-prior-based float32 targets (T1/T3) additionally pass strict_f32=True:
    under x64 their weak-typed python-float prior constants silently promote
    to float64 and the model is no longer the one the stored chains sampled.
    """
    import jax

    x64 = bool(jax.config.jax_enable_x64)
    if dtype == "float64" and not x64:
        raise RuntimeError(
            f"target dtype float64 but jax_enable_x64 is OFF; run this target "
            f"in a GIGALENS_X64=1 process (single-target-per-process policy).")
    if dtype == "float32" and x64 and strict_f32:
        raise RuntimeError(
            "target dtype float32 (tfp-based) but jax_enable_x64 is ON; the "
            "prior constants would promote to float64 and the posterior would "
            "not match the stored float32 chains. Run in a GIGALENS_X64=0 "
            "process (single-target-per-process policy).")


def np_dtype(dtype: str):
    return np.float64 if dtype == "float64" else np.float32


def sha256_array(a: np.ndarray) -> str:
    """Canonical hash: float64 little-endian bytes of the flattened array."""
    return hashlib.sha256(
        np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


def sha256_labels(labels: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(labels).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class InitBundle:
    """Initialization info for sampler adapters.

    prior_sample_fn(key, n) -> (n, dim) unconstrained draws (the gigalens MAP
    multistart convention: prior.sample then bij.inverse; synthetic targets
    sample their explicit Gaussian prior).

    svi_scale_tril, where present, is ALWAYS the eigenvalue-floored Cholesky
    from guards.floor_svi_covariance (Bug-2 guard); svi_n_floored records how
    many eigenvalues the floor raised.
    """
    prior_sample_fn: Callable[[Any, int], Any]
    map_z: Optional[np.ndarray] = None            # (dim,) best-known mode
    svi_loc: Optional[np.ndarray] = None          # (dim,)
    svi_scale_tril: Optional[np.ndarray] = None   # (dim, dim) floored chol
    svi_n_floored: int = 0
    hess_diag_raw: Optional[np.ndarray] = None    # |diag H_raw| (T2 diagraw mass)
    multi_starts: Optional[np.ndarray] = None     # (k, dim) multi-basin starts
    notes: str = ""


@dataclasses.dataclass
class Reference:
    """Reference posterior information. `provenance` is REQUIRED, always.

    mode_assigner describes how to assign samples to modes:
      {"method": "mahalanobis"}                  -> use mode_centers/mode_covs
      {"method": "param_threshold",
       "param": "gamma", "thresholds": [1.8]}    -> threshold a physical param
    mode_weights_trusted=False means the stored weights are descriptive only
    (e.g. T3 chain-count fractions with ZERO inter-basin migrations).
    """
    provenance: str
    samples: Optional[np.ndarray] = None          # (T, C, dim) eager
    samples_path: Optional[Sequence[str]] = None  # lazy: npz path(s)
    samples_key: str = "samples"
    mode_labels: Optional[Sequence[str]] = None
    mode_weights: Optional[np.ndarray] = None
    mode_weights_trusted: bool = False
    mode_centers: Optional[np.ndarray] = None     # (K, dim) z-space
    mode_covs: Optional[np.ndarray] = None        # (K, dim, dim) z-space
    mode_assigner: Optional[dict] = None
    logZ: Optional[float] = None
    truth: Optional[dict] = None                  # analytic moments / mock truth
    exact_sample_fn: Optional[Callable[[int, int], np.ndarray]] = None
    caveats: str = ""

    def load_samples(self) -> Optional[np.ndarray]:
        """Return (T, C, dim) reference samples (lazy npz load; None if none)."""
        if self.samples is not None:
            return self.samples
        if not self.samples_path:
            return None
        chains = []
        for p in self.samples_path:
            arr = np.load(p)[self.samples_key]
            if arr.ndim == 2:              # single chain (T, dim)
                arr = arr[:, None, :]
            chains.append(arr)
        if len(chains) == 1:
            return chains[0]
        t_min = min(c.shape[0] for c in chains)
        return np.concatenate([c[:t_min] for c in chains], axis=1)


@dataclasses.dataclass
class LensPosterior:
    """One zoo target. See module docstring for the full contract."""
    name: str
    tier: str                       # "T0".."T4"
    dim: int
    dtype: str                      # "float32" | "float64"
    noise_model: str                # "diag" (P1 hook: "correlated")
    log_prob_batch: Callable        # (N, dim) -> (N,)   PRIMARY
    log_prior_batch: Callable       # (N, dim) -> (N,)   incl. fwd log-det-J
    labels: Sequence[str]           # len dim, bijector-leaf order
    mass_labels: Sequence[str]      # science-critical subset of labels
    to_physical: Callable           # (N, dim) -> dict[str, np.ndarray]
    init: InitBundle
    bijector: Any = None            # tfp bijector; None = identity (T0)
    log_like_batch_fn: Optional[Callable] = None   # independent impl if any
    prior_transform: Optional[Callable] = None     # [0,1]^dim -> physical x
    log_like_x: Optional[Callable] = None          # data loglik at physical x
    chi2_fn: Optional[Callable] = None             # masked reduced chi2 (T3)
    reference: Optional[Reference] = None
    meta: dict = dataclasses.field(default_factory=dict)

    # ---- derived single-point / likelihood surfaces -------------------------
    def log_prob(self, z):
        import jax.numpy as jnp
        return jnp.squeeze(self.log_prob_batch(jnp.asarray(z)[None, :]))

    def log_prior(self, z):
        import jax.numpy as jnp
        return jnp.squeeze(self.log_prior_batch(jnp.asarray(z)[None, :]))

    def log_like_batch(self, Z):
        """Independent implementation where available, else prob - prior."""
        if self.log_like_batch_fn is not None:
            return self.log_like_batch_fn(Z)
        return self.log_prob_batch(Z) - self.log_prior_batch(Z)

    def log_like(self, z):
        import jax.numpy as jnp
        return jnp.squeeze(self.log_like_batch(jnp.asarray(z)[None, :]))

    @property
    def has_independent_loglike(self) -> bool:
        return self.log_like_batch_fn is not None

    def mass_indices(self) -> list:
        return [list(self.labels).index(m) for m in self.mass_labels]

    def labels_hash(self) -> str:
        return sha256_labels(self.labels)


# --------------------------------------------------------------------------- #
# generic gigalens-structure helpers (shared by T1/T3; T2 uses index_labels)
# --------------------------------------------------------------------------- #
def flatten_gigalens_nested(x, prefixes) -> list:
    """Flatten a gigalens nested-physical structure into [(label, value)].

    x is the bij.forward output: a list of blocks, each a list of dicts.
    prefixes mirrors that structure with a string per dict. Values are
    consumed in dict-iteration order (== the z-leaf order; the documented
    gu-2022 fix). Batch dim, if any, must be 1.
    """
    out = []
    for block, block_prefixes in zip(x, prefixes):
        for comp, prefix in zip(block, block_prefixes):
            for key, val in comp.items():
                out.append((prefix + key, float(np.asarray(val).squeeze())))
    return out


def probe_labels(bij, ndim, prefixes, dtype: str, probe: float = 3.0,
                 atol: float = 1e-5) -> list:
    """z-index -> label via single-coordinate perturbation probing.

    Generic port of foundry-i `_hmc_lib_marg._label_index_map`: perturb one z
    coordinate at a time and see which physical leaf moves. Robust against
    any dict reordering inside the tfp bijector stack.
    """
    fdtype = np_dtype(dtype)
    base = np.zeros(ndim, dtype=fdtype)
    b = flatten_gigalens_nested(bij.forward(list(base[:, None])), prefixes)
    labels0 = [lab for lab, _ in b]
    bvals = np.array([v for _, v in b])
    labels = []
    for j in range(ndim):
        z = base.copy()
        z[j] = probe
        pj = flatten_gigalens_nested(bij.forward(list(z[:, None])), prefixes)
        pv = np.array([v for _, v in pj])
        changed = np.where(np.abs(pv - bvals) > atol)[0]
        if len(changed) == 1:
            labels.append(labels0[changed[0]])
        elif len(changed) == 0:
            labels.append(f"z{j}_unmapped")
        else:
            labels.append("|".join(labels0[c] for c in changed))
    return labels


def make_prior_z_sampler(prior, bij, dim: int, dtype: str):
    """(key, n) -> (n, dim) z draws: prior.sample then bij.inverse then stack.

    This is exactly the gigalens ModellingSequence.MAP start convention.
    """
    def prior_sample_fn(key, n: int):
        import jax.numpy as jnp
        start = prior.sample(n, seed=key)
        z = jnp.stack(bij.inverse(start)).T          # (n, dim)
        return jnp.asarray(z, dtype=np_dtype(dtype))

    return prior_sample_fn
