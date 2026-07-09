"""T3 zoo target: the foundry-i paper-scale 74-dim BIMODAL posterior on the
binned 0.08" product cutout_v3b (float32).

Built through foundry-i's `_data_lib.build_all` (imported by path, unpatched)
so log_prob reproduces the EXACT model the stored hmc_v13_v3b.npz chains
sampled: EPL(50)+Shear, 4 Sersic lens light, Sersic+Shapelets(n_max=6)
source (all amps sampled, use_lstsq=False), masked ForwardProbModel with the
precomputed err map and the err->masked convention (Independent Normal over
UNMASKED pixels; masked reduced chi^2 as the second output).

Vendor pinning: cgl.paths.bootstrap_vendor() runs FIRST and every gigalens
module _data_lib touches is imported here BEFORE _data_lib is imported, so
_data_lib.bootstrap_vendor()'s sys.path insertion of foundry-i's own vendor
copy (bit-identical ref anyway) can never shadow the campaign vendor.

Stored-chain facts (measured 2026-07-06, hmc_v13_v3b.npz):
  samples (3500, 48, 74) f32 + mass_* (3500, 48); NO stored logp values, so a
  bit-check of the density is impossible; instead 21_validate_zoo checks (a)
  to_physical(samples) reproduces the stored mass_* arrays (bijector/label
  parity) and (b) chi^2/logp consistency at stored draws. Basin structure:
  45/48 chains at gamma~1.294 ("low"), 3/48 chains (4,6,8) at gamma~2.42
  ("steep"); ZERO chains cross gamma=1.8 => mode weights are chain-count
  fractions, NOT trusted posterior mass (that is exactly why P2c builds a PT
  reference).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from cgl import guards
from cgl.paths import (FOUNDRY_I, HMC_V13_V3B, MAP_V11_V3B_COLD,
                       MAP_V11_V3B_COLD2D, MAP_V11_V3B_WARM, SVI_V12_V3BR,
                       bootstrap_vendor)
from cgl.zoo.api import (InitBundle, LensPosterior, Reference, assert_dtype_env,
                         make_prior_z_sampler, probe_labels)

MASS_LABELS = ["mass.theta_E", "mass.gamma", "mass.e1", "mass.e2",
               "mass.center_x", "mass.center_y", "shear.gamma1",
               "shear.gamma2"]
GAMMA_THRESHOLD = 1.8  # mid-gap: low basin < 1.31, steep basin > 2.32


def _import_data_lib():
    """Import foundry-i _data_lib with the campaign vendor pinned."""
    bootstrap_vendor()
    # Pin every gigalens module _data_lib (and we) will touch BEFORE its own
    # bootstrap_vendor can prepend the foundry-i vendor copy to sys.path.
    import gigalens.jax.model      # noqa: F401
    import gigalens.jax.profiles.light.sersic    # noqa: F401
    import gigalens.jax.profiles.light.shapelets  # noqa: F401
    import gigalens.jax.profiles.mass.epl        # noqa: F401
    import gigalens.jax.profiles.mass.shear      # noqa: F401
    import gigalens.jax.simulator  # noqa: F401
    import gigalens.model          # noqa: F401
    import gigalens.simulator      # noqa: F401
    p = str(FOUNDRY_I)
    if p not in sys.path:
        sys.path.insert(1, p)
    import _data_lib
    return _data_lib


def build() -> LensPosterior:
    assert_dtype_env("float32", strict_f32=True)
    guards.require_gpu()
    D = _import_data_lib()

    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    from gigalens.jax.simulator import LensSimulator

    tfd = tfp.distributions

    d, prior, phys, prob, sim_config = D.build_all(n_max=6,
                                                   data_file="cutout_v3b.npz")
    bij = prob.bij
    ndim = int(sum(np.size(v) for v in
                   jax.tree_util.tree_leaves(prior.sample(
                       seed=jax.random.PRNGKey(0)))))
    assert ndim == 74, f"expected the 74-dim v3b model, got {ndim}"

    _sims: dict = {}

    def _sim(n: int) -> LensSimulator:
        if n not in _sims:
            _sims[n] = LensSimulator(phys, sim_config, bs=n)
        return _sims[n]

    def log_prob_batch(Z):
        Z = jnp.atleast_2d(jnp.asarray(Z, dtype=jnp.float32))
        return prob.log_prob(_sim(int(Z.shape[0])), Z)[0]

    def chi2_batch(Z):
        """Masked reduced chi^2 (the foundry-i Stage-B gate quantity)."""
        Z = jnp.atleast_2d(jnp.asarray(Z, dtype=jnp.float32))
        return prob.log_prob(_sim(int(Z.shape[0])), Z)[1]

    def _log_prior_impl(Z):
        z = list(Z.T)
        x = bij.forward(z)
        return prior.log_prob(x) + bij.forward_log_det_jacobian(z)

    _log_prior_jit = jax.jit(_log_prior_impl)

    def log_prior_batch(Z):
        return _log_prior_jit(jnp.atleast_2d(jnp.asarray(Z, dtype=jnp.float32)))

    # independent data term: the carousel masked-Normal convention (_data_lib)
    mask = np.asarray(d["keep_mask"], dtype=bool)
    obs = jnp.asarray(d["img"], dtype=jnp.float32)
    err = jnp.asarray(d["err_map"], dtype=jnp.float32)
    mask_j = jnp.asarray(mask)
    observed_dist = tfd.Independent(
        tfd.Normal(obs[mask_j], err[mask_j]), reinterpreted_batch_ndims=1)

    def _log_like_impl(Z):
        x = bij.forward(list(Z.T))
        im_sim = _sim(int(Z.shape[0])).simulate(x).reshape((-1, *obs.shape))
        return observed_dist.log_prob(im_sim[:, mask_j])

    _log_like_jit = jax.jit(_log_like_impl)

    def log_like_batch(Z):
        return _log_like_jit(jnp.atleast_2d(jnp.asarray(Z, dtype=jnp.float32)))

    # ---- labels (probed; includes the 28 shapelet amp leaves) ----------------
    prefixes = [["mass.", "shear."], ["LL0.", "LL1.", "LL2.", "LL3."],
                ["srcS.", "srcShp."]]
    labels = probe_labels(bij, ndim, prefixes, "float32")
    for m in MASS_LABELS:
        assert m in labels, f"mass label {m} not found in probed labels"

    def to_physical(Z):
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float32))
        return D.mass_params_from_z(prob, Z.reshape(-1, ndim))

    # ---- nautilus unit-cube face (P2c: enables the T3 nautilus track) --------
    # The v3b prior is a nested JointDistributionSequential of
    # JointDistributionNamed leaves, ALL independent 1-D marginals (LogNormal /
    # TruncatedNormal / Normal / Uniform incl. the 28 shapelet-amp Normals), so
    # a leaf-wise quantile transform is exact. Marginals are read GENERICALLY
    # from the built prior tree (no hyperparameters duplicated), keyed by the
    # SAME probed labels, then ordered into the z-leaf order. log_like_x uses
    # prob.pack_bij (physical -> nested, NO unconstraining bijector). The
    # nautilus adapter's own 8-point x<->z self-check (log_like_x vs
    # log_like_batch, plus its probe-based permutation) is the safety net: a
    # mis-ordered face fails LOUDLY before the run.
    def _leaf_marginals() -> dict:
        marg = {}
        for block, block_prefixes in zip(prior.model, prefixes):
            for jdn, prefix in zip(block.model, block_prefixes):
                for key, dist in jdn.model.items():
                    marg[prefix + key] = dist
        return marg

    _marg = _leaf_marginals()
    _missing = [lab for lab in labels if lab not in _marg]
    assert not _missing, f"nautilus face: no prior marginal for {_missing[:6]}"
    dists_in_leaf_order = [_marg[lab] for lab in labels]
    pack_bij = prob.pack_bij

    def prior_transform(u):
        u = np.clip(np.atleast_2d(np.asarray(u, dtype=np.float32)),
                    1e-7, 1.0 - 1e-7)
        cols = [np.asarray(dists_in_leaf_order[i].quantile(u[:, i]))
                for i in range(ndim)]
        return np.stack(cols, axis=1).squeeze()

    def _log_like_x_impl(X):
        """Data log-likelihood at PHYSICAL x (leaf order); no bijector."""
        nested = pack_bij.forward(list(X.T))
        im_sim = _sim(int(X.shape[0])).simulate(nested).reshape((-1, *obs.shape))
        return observed_dist.log_prob(im_sim[:, mask_j])

    _log_like_x_jit = jax.jit(_log_like_x_impl)

    def log_like_x(x):
        x = jnp.atleast_2d(jnp.asarray(x, dtype=jnp.float32))
        return np.asarray(_log_like_x_jit(x))

    # ---- init: multi-basin MAP starts + the SVI the stored chains used -------
    def _best_z(path):
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        return np.asarray(z["best_z"], dtype=np.float64), meta

    starts, start_notes = [], []
    for path, tag in [(MAP_V11_V3B_COLD2D, "low (gamma~1.369, chi2 1.594)"),
                      (MAP_V11_V3B_COLD, "steep-ish (gamma~1.465, chi2 1.598)"),
                      (MAP_V11_V3B_WARM, "high (gamma~2.672, chi2 1.516)")]:
        if path.exists():
            bz, meta = _best_z(path)
            starts.append(bz)
            start_notes.append(f"{path.name}: {tag}")
    multi_starts = np.stack(starts) if starts else None

    sv = np.load(SVI_V12_V3BR)
    svi_loc = np.asarray(sv["loc"], dtype=np.float64)
    cov_reg, chol, n_floored = guards.floor_svi_covariance(
        np.asarray(sv["cov"], dtype=np.float64))
    init = InitBundle(
        prior_sample_fn=make_prior_z_sampler(prior, bij, ndim, "float32"),
        map_z=starts[0] if starts else None,
        svi_loc=svi_loc,
        svi_scale_tril=chol,
        svi_n_floored=n_floored,
        multi_starts=multi_starts,
        notes="multi_starts = foundry-i map_v11_v3b* best_z (74-dim runs): "
              + "; ".join(start_notes)
              + ". svi = svi_v12_v3br.npz (15000-step, n_vi=1000 SVI the "
              "stored v3b chains started from), guard-floored "
              f"({n_floored} eigs).",
    )

    # ---- reference: the stored bimodal chains (weights NOT trusted) ----------
    reference = None
    if HMC_V13_V3B.exists():
        g = np.load(HMC_V13_V3B, allow_pickle=True)
        chain_gamma = np.asarray(g["mass_gamma"], dtype=np.float64)  # (T, C)
        chain_mean = chain_gamma.mean(axis=0)
        occ_low = float((chain_mean < GAMMA_THRESHOLD).mean())
        meta = json.loads(str(g["meta"]))
        reference = Reference(
            provenance=("foundry-i 43_hmc_paper_scale.py tag=v3b: 48-chain "
                        "PHMC+ChEES(GBTLA,max_leapfrog=30)+DualAveraging, "
                        "burn 750 / 3500 kept, float32, started from "
                        "svi_v12_v3br draws; stored hmc_v13_v3b.npz; meta="
                        + json.dumps(meta)),
            samples_path=[str(HMC_V13_V3B)],
            samples_key="samples",
            mode_labels=("gamma_low", "gamma_steep"),
            mode_weights=np.array([occ_low, 1.0 - occ_low]),
            mode_weights_trusted=False,
            mode_assigner={"method": "param_threshold", "param": "gamma",
                           "thresholds": [GAMMA_THRESHOLD]},
            truth={"chain_mean_gamma_low": float(
                       chain_mean[chain_mean < GAMMA_THRESHOLD].mean()),
                   "chain_mean_gamma_steep": float(
                       chain_mean[chain_mean >= GAMMA_THRESHOLD].mean()),
                   "n_chains_low": int((chain_mean < GAMMA_THRESHOLD).sum()),
                   "n_chains_steep": int((chain_mean >= GAMMA_THRESHOLD).sum())},
            caveats=("DIAGONAL-likelihood chains; ZERO inter-basin migrations "
                     "(no chain crosses gamma=1.8), so mode weights are "
                     "chain-count fractions of the SVI-start distribution, "
                     "NOT posterior mass -- P2c builds the PT reference. "
                     "npz stores samples+mass params only (no logp values); "
                     "density parity is checked via mass_* reproduction + "
                     "chi^2 consistency (21_validate_zoo)."),
        )

    return LensPosterior(
        name="foundry_v3b74", tier="T3", dim=ndim, dtype="float32",
        noise_model="diag",
        log_prob_batch=log_prob_batch, log_prior_batch=log_prior_batch,
        log_like_batch_fn=log_like_batch,
        labels=labels, mass_labels=MASS_LABELS,
        to_physical=to_physical,
        init=init, bijector=bij,
        prior_transform=prior_transform,  # P2c: T3 nautilus track (self-checked)
        log_like_x=log_like_x,
        chi2_fn=chi2_batch,
        reference=reference,
        meta={"data_file": "cutout_v3b.npz", "n_max": 6,
              "gamma_threshold": GAMMA_THRESHOLD,
              "model": "EPL(50)+Shear / 4 Sersic LL / Sersic+Shapelets(6) src, "
                       "masked ForwardProbModel (_data_lib conventions)",
              "err_inflation": "mask applied via Independent Normal over "
                               "unmasked pixels (keep_mask), err_map "
                               "precomputed (drizzle-corrected, sky chi2=1)"},
    )
