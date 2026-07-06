"""T1 zoo targets: the 12 gu-2022 mock systems (22-dim, float32, GPU).

Wraps the EXACT prior/bijector/likelihood of
    /raid/benson/git/agentic-lensing/reproductions/gu-2022/02_fit_system.py
(`build_prior` is COPIED WITH ATTRIBUTION below, hyperparameters verbatim)
against the VENDORED gigalens-sean via cgl.paths.bootstrap_vendor. The
posterior is gigalens ForwardProbModel.log_prob on the stored mock image --
the same closure the stored reference fits sampled.

Bijector-leaf-order trap (the documented 02_fit_system fix): the tfp bijector
REORDERS dict keys within each block (e.g. e2 before e1). z index k, the
physical leaf k, and any per-param diagnostic k are aligned to the SAME leaf
flattening; labels here are derived by single-coordinate PROBING
(cgl.zoo.api.probe_labels) and cross-checked against the stored fit npz
phys_labels in 21_validate_zoo.

dtype: float32 (the mode the reference fits ran); building under x64 is
refused (assert_dtype_env strict_f32).
"""
from __future__ import annotations

import json

import numpy as np

from cgl import guards
from cgl.paths import GU2022_FITS, GU2022_MOCKS, bootstrap_vendor
from cgl.zoo.api import (InitBundle, LensPosterior, Reference, assert_dtype_env,
                         make_prior_z_sampler, probe_labels)

# gu-2022 human/spec parameter order (NOT the bijector leaf order).
PARAM_LABELS = [
    "theta_E", "gamma", "e1", "e2", "center_x", "center_y",
    "gamma1", "gamma2",
    "ll_R_sersic", "ll_n_sersic", "ll_e1", "ll_e2",
    "ll_center_x", "ll_center_y", "ll_Ie",
    "src_R_sersic", "src_n_sersic", "src_e1", "src_e2",
    "src_center_x", "src_center_y", "src_Ie",
]
MASS_LABELS = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y",
               "gamma1", "gamma2"]
_PREFIXES = [["", ""], ["ll_"], ["src_"]]


def build_prior(tfd, jnp):
    """COPIED WITH ATTRIBUTION from gu-2022/02_fit_system.py::build_prior
    (hyperparameters verbatim; any edit here voids T1 reference parity)."""
    lens_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
            gamma=tfd.TruncatedNormal(2.0, 0.5, 1.0, 3.0),
            e1=tfd.Normal(0.0, 0.2),
            e2=tfd.Normal(0.0, 0.2),
            center_x=tfd.Normal(0.0, 0.1),
            center_y=tfd.Normal(0.0, 0.1),
        )),
        tfd.JointDistributionNamed(dict(
            gamma1=tfd.Normal(0.0, 0.06),
            gamma2=tfd.Normal(0.0, 0.06),
        )),
    ])
    lens_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
            n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.1, -0.15, 0.15),
            e2=tfd.TruncatedNormal(0.0, 0.1, -0.15, 0.15),
            center_x=tfd.Normal(0.0, 0.02),
            center_y=tfd.Normal(0.0, 0.02),
            Ie=tfd.LogNormal(jnp.log(300.0), 0.5),
        )),
    ])
    source_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
            n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
            center_x=tfd.Normal(0.0, 0.5),
            center_y=tfd.Normal(0.0, 0.5),
            Ie=tfd.LogNormal(jnp.log(150.0), 0.9),
        )),
    ])
    return tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior])


def _leaf_prior_dists(tfd, jnp):
    """label -> the 1-D prior marginal (all T1 leaves are independent).
    Used for the nautilus unit-cube face (leaf-wise quantile transform)."""
    return {
        "theta_E": tfd.LogNormal(jnp.log(1.25), 0.4),
        "gamma": tfd.TruncatedNormal(2.0, 0.5, 1.0, 3.0),
        "e1": tfd.Normal(0.0, 0.2), "e2": tfd.Normal(0.0, 0.2),
        "center_x": tfd.Normal(0.0, 0.1), "center_y": tfd.Normal(0.0, 0.1),
        "gamma1": tfd.Normal(0.0, 0.06), "gamma2": tfd.Normal(0.0, 0.06),
        "ll_R_sersic": tfd.LogNormal(jnp.log(1.6), 0.25),
        "ll_n_sersic": tfd.Uniform(0.5, 8.0),
        "ll_e1": tfd.TruncatedNormal(0.0, 0.1, -0.15, 0.15),
        "ll_e2": tfd.TruncatedNormal(0.0, 0.1, -0.15, 0.15),
        "ll_center_x": tfd.Normal(0.0, 0.02),
        "ll_center_y": tfd.Normal(0.0, 0.02),
        "ll_Ie": tfd.LogNormal(jnp.log(300.0), 0.5),
        "src_R_sersic": tfd.LogNormal(jnp.log(0.25), 0.25),
        "src_n_sersic": tfd.Uniform(0.5, 8.0),
        "src_e1": tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
        "src_e2": tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
        "src_center_x": tfd.Normal(0.0, 0.5),
        "src_center_y": tfd.Normal(0.0, 0.5),
        "src_Ie": tfd.LogNormal(jnp.log(150.0), 0.9),
    }


def build(idx: int) -> LensPosterior:
    """Build gu2022_sysXXX. float32, GPU (guards.require_gpu)."""
    assert_dtype_env("float32", strict_f32=True)
    guards.require_gpu()
    bootstrap_vendor()

    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    from gigalens.jax.model import ForwardProbModel
    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.simulator import LensSimulator
    from gigalens.model import PhysicalModel
    from gigalens.simulator import SimulatorConfig

    tfd = tfp.distributions

    mock_path = GU2022_MOCKS / f"system_{idx:03d}.npz"
    fit_path = GU2022_FITS / f"system_{idx:03d}_fit.npz"
    d = np.load(mock_path, allow_pickle=True)
    image = np.array(d["image"], dtype=np.float32)
    psf = np.array(d["psf"], dtype=np.float32)
    sigma_bkg = float(d["sigma_bkg"])
    exp_time = float(d["exp_time"])
    num_pix = int(d["num_pix"])
    delta_pix = float(d["delta_pix"])
    supersample = int(d["supersample"])
    truth_nested = json.loads(str(d["truth_json"]))

    # ---- model, exactly as 02_fit_system.py builds it -----------------------
    prior = build_prior(tfd, jnp)
    sim_config = SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix,
                                 supersample=supersample, kernel=psf)
    phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False)],
        [sersic.SersicEllipse(use_lstsq=False)],
    )
    prob_model = ForwardProbModel(prior, image, background_rms=sigma_bkg,
                                  exp_time=exp_time)
    bij, pack_bij = prob_model.bij, prob_model.pack_bij
    ndim = int(sum(np.size(v) for v in
                   jax.tree_util.tree_leaves(prior.sample(
                       seed=jax.random.PRNGKey(0)))))
    assert ndim == 22

    # LensSimulator batch size is STATIC; cache one per batch size.
    _sims: dict = {}

    def _sim(n: int) -> LensSimulator:
        if n not in _sims:
            _sims[n] = LensSimulator(phys_model, sim_config, bs=n)
        return _sims[n]

    obs = jnp.asarray(image, dtype=jnp.float32)
    rms = jnp.float32(sigma_bkg)
    et = jnp.float32(exp_time)

    def log_prob_batch(Z):
        Z = jnp.atleast_2d(jnp.asarray(Z, dtype=jnp.float32))
        return prob_model.log_prob(_sim(int(Z.shape[0])), Z)[0]

    def chi2_batch(Z):
        Z = jnp.atleast_2d(jnp.asarray(Z, dtype=jnp.float32))
        return prob_model.log_prob(_sim(int(Z.shape[0])), Z)[1]

    def _log_prior_impl(Z):
        z = list(Z.T)
        x = bij.forward(z)
        return prior.log_prob(x) + bij.forward_log_det_jacobian(z)

    _log_prior_jit = jax.jit(_log_prior_impl)

    def log_prior_batch(Z):
        return _log_prior_jit(jnp.atleast_2d(jnp.asarray(Z, dtype=jnp.float32)))

    def _log_like_impl(Z):
        """Independent copy of the ForwardProbModel data term (same math)."""
        x = bij.forward(list(Z.T))
        n = Z.shape[0]
        im_sim = _sim(n).simulate(x)
        im_sim = im_sim.reshape((-1, *obs.shape))
        err_map = jnp.sqrt(rms ** 2 + im_sim / et)
        return tfd.Independent(
            tfd.Normal(im_sim, err_map), reinterpreted_batch_ndims=2
        ).log_prob(obs)

    _log_like_jit = jax.jit(_log_like_impl)

    def log_like_batch(Z):
        return _log_like_jit(jnp.atleast_2d(jnp.asarray(Z, dtype=jnp.float32)))

    # ---- labels (probed), truth, physical map --------------------------------
    labels = probe_labels(bij, ndim, _PREFIXES, "float32")
    assert sorted(labels) == sorted(PARAM_LABELS), f"label mismatch: {labels}"
    mass_idx = [labels.index(m) for m in MASS_LABELS]

    def to_physical(Z):
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float32))
        phys = bij.forward(list(jnp.asarray(Z.reshape(-1, ndim)).T))
        out = {}
        for k in ("theta_E", "gamma", "e1", "e2", "center_x", "center_y"):
            out[k] = np.asarray(phys[0][0][k])
        out["gamma1"] = np.asarray(phys[0][1]["gamma1"])
        out["gamma2"] = np.asarray(phys[0][1]["gamma2"])
        return out

    # truth in bijector-leaf order + its z-vector (mode neighborhood)
    block_prefix = {(0, 0): "", (0, 1): "", (1, 0): "ll_", (2, 0): "src_"}
    truth_by_label = {}
    for (i, j), pre in block_prefix.items():
        for key, val in truth_nested[i][j].items():
            truth_by_label[pre + key] = float(val)
    truth_vec = np.array([truth_by_label[lab] for lab in labels])
    truth_f32 = jax.tree_util.tree_map(np.float32, truth_nested)
    z_truth = np.asarray(jnp.stack(bij.inverse(truth_f32)).squeeze(),
                         dtype=np.float64)

    # ---- nautilus unit-cube face ---------------------------------------------
    leaf_dists = _leaf_prior_dists(tfd, jnp)
    dists_in_leaf_order = [leaf_dists[lab] for lab in labels]

    def prior_transform(u):
        u = np.clip(np.atleast_2d(np.asarray(u, dtype=np.float32)),
                    1e-7, 1.0 - 1e-7)
        cols = [np.asarray(dists_in_leaf_order[i].quantile(u[:, i]))
                for i in range(ndim)]
        return np.stack(cols, axis=1).squeeze()

    def _log_like_x_impl(X):
        """Data log-likelihood at PHYSICAL x (leaf order); no bijector."""
        nested = pack_bij.forward(list(X.T))
        n = X.shape[0]
        im_sim = _sim(n).simulate(nested).reshape((-1, *obs.shape))
        err_map = jnp.sqrt(rms ** 2 + im_sim / et)
        return tfd.Independent(
            tfd.Normal(im_sim, err_map), reinterpreted_batch_ndims=2
        ).log_prob(obs)

    _log_like_x_jit = jax.jit(_log_like_x_impl)

    def log_like_x(x):
        x = jnp.atleast_2d(jnp.asarray(x, dtype=jnp.float32))
        return np.asarray(_log_like_x_jit(x))

    # ---- init / reference ----------------------------------------------------
    init = InitBundle(
        prior_sample_fn=make_prior_z_sampler(prior, bij, ndim, "float32"),
        map_z=None,
        notes="baseline recipe computes MAP/SVI itself (the gigalens protocol); "
              "prior_sample_fn = prior.sample + bij.inverse multistart. "
              f"truth z-vector available in reference.truth['z_truth'].",
    )
    reference = None
    if fit_path.exists():
        f = np.load(fit_path, allow_pickle=True)
        reference = Reference(
            provenance=(f"gu-2022/02_fit_system.py fit of {mock_path.name} "
                        "(MAP 128x250 adabelief -> SVI 200x500 -> 50-chain "
                        "PHMC+DualAveraging, 250 burn / 750 kept, float32, "
                        f"A16 GPU; stored {fit_path.name}). Truth from the "
                        "mock generator truth_json."),
            samples_path=[str(fit_path)],
            samples_key="samples_unconstrained",     # (750, 50, 22) f32
            mode_labels=("mode0",),
            mode_weights=np.array([1.0]),
            mode_weights_trusted=True,
            mode_centers=z_truth[None, :],
            mode_assigner={"method": "mahalanobis"},
            truth={"by_label": {lab: truth_by_label[lab] for lab in labels},
                   "vec_leaf_order": truth_vec.tolist(),
                   "z_truth": z_truth.tolist(),
                   "stored_min_ess": float(np.min(f["ess"])),
                   "stored_max_rhat": float(np.max(f["rhat"]))},
            caveats=("stored fit is the diagonal-noise reference run "
                     "(fixed-L=5 PHMC unless the batch passed --gbtla; flag "
                     "not recorded in the npz)."),
        )

    return LensPosterior(
        name=f"gu2022_sys{idx:03d}", tier="T1", dim=ndim, dtype="float32",
        noise_model="diag",
        log_prob_batch=log_prob_batch, log_prior_batch=log_prior_batch,
        log_like_batch_fn=log_like_batch,
        labels=labels, mass_labels=MASS_LABELS,
        to_physical=to_physical,
        init=init, bijector=bij,
        prior_transform=prior_transform, log_like_x=log_like_x,
        chi2_fn=chi2_batch,
        reference=reference,
        meta={"mock": str(mock_path), "fit": str(fit_path),
              "num_pix": num_pix, "delta_pix": delta_pix,
              "supersample": supersample, "sigma_bkg": sigma_bkg,
              "exp_time": exp_time,
              "model": "EPL(50)+Shear / Sersic lens light / Sersic source, "
                       "ForwardProbModel (Poisson+sky err from model image)"},
    )
