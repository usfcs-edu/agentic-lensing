"""Shared model builders for the multi-device / multi-node sharding suite.

Kept free of pytest and of any device/env setup so BOTH the pytest suite (`conftest.py`,
faked CPU devices) and the standalone multi-node driver (`multinode_validate.py`, real
`jax.distributed` across nodes) construct the exact same models. Importing this module does
NOT initialise the XLA backend (only `import jax`), so callers stay in control of device
setup / `jax.distributed.initialize()` ordering.

Two systems, both forward-mode (amplitudes sampled) on the 60x60 demo image:

  * ``build_single_lensplane`` — one lens plane (EPL + shear + Sersic lens light) and one
    source plane (Sersic). The original ported system.
  * ``build_multiplane`` — TWO mass planes (EPL at z1, EPL + Sersic light at z2) plus a
    source plane (Sersic at z3): genuine recursive multi-lens-plane ray-shooting, for the
    "does multiplane inference shard across nodes?" check.

The fit quality is irrelevant (these validate sharding/shapes, not recovery); the truncated
/ positive-support priors only need to keep ``log_prob`` finite over their support.
"""
import os

import numpy as np
import jax
from jax import numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from gigalens.simulator import SimulatorConfig
from gigalens.jax.cosmo import wCDM_Cosmo
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_prob_model import ImageData, ProbModel
from gigalens.jax.inference import ModellingSequence

tfd = tfp.distributions

_ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "src", "gigalens", "assets")
# single-plane redshifts / multiplane redshifts (z1 < z2 are both LENS planes)
_Z_LENS, _Z_SOURCE = 0.5, 2.0
_Z1, _Z2, _Z3 = 0.5, 1.0, 2.0


# --- priors -------------------------------------------------------------------------
def _epl_priors():
    return dict(
        theta_E=tfd.LogNormal(jnp.log(1.25), 0.25),
        gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 3.0),
        e1=tfd.Normal(0.0, 0.1),
        e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.05),
        center_y=tfd.Normal(0.0, 0.05),
    )


def _shear_priors():
    return dict(gamma1=tfd.Normal(0.0, 0.05), gamma2=tfd.Normal(0.0, 0.05))


def _lens_light_priors():
    return dict(
        R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
        n_sersic=tfd.Uniform(2.0, 6.0),
        e1=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3),
        center_x=tfd.Normal(0.0, 0.05),
        center_y=tfd.Normal(0.0, 0.05),
        Ie=tfd.LogNormal(jnp.log(500.0), 0.3),
    )


def _source_priors():
    return dict(
        R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15),
        n_sersic=tfd.Uniform(0.5, 4.0),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.25),
        center_y=tfd.Normal(0.0, 0.25),
        Ie=tfd.LogNormal(jnp.log(150.0), 0.5),
    )


def _assets():
    kernel = np.load(os.path.join(_ASSETS, "psf.npy")).astype(np.float32)
    observed = np.load(os.path.join(_ASSETS, "demo.npy"))
    return kernel, observed


def _finish(model, cfg, observed):
    # float64 to match the new-API demos under jax_enable_x64 (keeps the MAP->SVI->HMC/MCLMC
    # chain dtype-consistent). Sharding is dtype-independent.
    ds = ImageData(observed, cfg, background_rms=0.2, exp_time=100, sees="all")
    prob = ProbModel(model, ds, mode="forward")
    seq = ModellingSequence.from_scene(prob)
    return model, cfg, prob, seq


def build_single_lensplane():
    """One lens plane (EPL+shear+lens-light) + one source plane. -> (model, cfg, prob, seq)"""
    kernel, observed = _assets()
    cosmo = Component(wCDM_Cosmo(z_lens=_Z_LENS, z_source_ref=10.0),
                      dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
    model = LensModel(
        [
            Plane(redshift=_Z_LENS,
                  mass=[Component(EPL(), _epl_priors()), Component(Shear(), _shear_priors())],
                  light=[Component(SersicEllipse(), _lens_light_priors())]),
            Plane(redshift=_Z_SOURCE, light=[Component(SersicEllipse(), _source_priors())]),
        ],
        cosmo=cosmo,
    )
    cfg = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=2, kernel=kernel,
                          likelihood_precision="float64")
    return _finish(model, cfg, observed)


def build_multiplane():
    """TWO mass planes (EPL@z1, EPL+light@z2) + source@z3: recursive multi-lens-plane.

    -> (model, cfg, prob, seq). supersample=1 keeps the two-plane ray-shoot compile cheap;
    the sharding path is identical regardless of supersampling.
    """
    kernel, observed = _assets()
    cosmo = Component(wCDM_Cosmo(z_lens=_Z1, z_source_ref=10.0),
                      dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
    model = LensModel(
        [
            Plane(redshift=_Z1, mass=[Component(EPL(), _epl_priors())]),
            Plane(redshift=_Z2, mass=[Component(EPL(), _epl_priors())],
                  light=[Component(SersicEllipse(), _lens_light_priors())]),
            Plane(redshift=_Z3, light=[Component(SersicEllipse(), _source_priors())]),
        ],
        cosmo=cosmo,
    )
    cfg = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=1, kernel=kernel,
                          likelihood_precision="float64")
    return _finish(model, cfg, observed)


def start_vector(prob, seed=0):
    """A flat unconstrained start vector, shape (num_free_params,), from one prior draw."""
    start = prob.bij.inverse(prob.prior.sample(1, seed=jax.random.PRNGKey(seed)))
    return np.asarray(start).reshape(-1)


def make_qz(start, scale=1e-2):
    """A tight diagonal MVN surrogate around ``start`` for HMC / MCLMC."""
    start = jnp.asarray(start)
    return tfd.MultivariateNormalDiag(start, jnp.ones_like(start) * scale)
