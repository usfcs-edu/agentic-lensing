"""Guard against the recurring "C-8" column->name mislabel.

The sampler works on a flat ``z`` vector whose column order is defined by the
bijector's ``pack_sequence_as`` = JAX tree-flatten (sorted-key) order — NOT the
``_unique`` insertion order or a profile's ``_params`` order. ``LensModel.
z_param_names`` is the one canonical source. This test verifies it by the only
convention-agnostic method: perturb each z-column and confirm the physical
parameter that moves is exactly the one ``z_param_names`` claims.

If someone reverts to insertion-order naming, this test fails.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates.jax import distributions as tfd

from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.cosmo import wCDM_Cosmo


def _build():
    # Params are declared in a deliberately NON-sorted order per component, so
    # insertion order != sorted order and a mislabel would be caught.
    lens = Component(EPL(20), dict(
        theta_E=tfd.Normal(1.5, 0.1), gamma=tfd.Uniform(1.5, 2.5),
        e1=tfd.Normal(0., 0.1), e2=tfd.Normal(0., 0.1),
        center_x=tfd.Normal(0., 0.1), center_y=tfd.Normal(0., 0.1)))
    sh = Component(Shear(), dict(gamma1=tfd.Normal(0., 0.05), gamma2=tfd.Normal(0., 0.05)))
    src = Component(SersicEllipse(use_lstsq=True), dict(
        R_sersic=tfd.Uniform(0.1, 1.), n_sersic=tfd.Uniform(0.5, 4.),
        e1=tfd.Normal(0., 0.1), e2=tfd.Normal(0., 0.1),
        center_x=tfd.Normal(0., 0.1), center_y=tfd.Normal(0., 0.1)))
    # fixed cosmology (all numbers -> no extra free params)
    cosmo = Component(wCDM_Cosmo(z_lens=0.5, z_source_ref=2.0),
                      dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
    return LensModel([Plane(redshift=0.5, mass=[lens, sh]),
                      Plane(redshift=2.0, light=[src])], cosmo=cosmo)


def _flat_params(model, z):
    x = model.bijector.forward(list(jnp.atleast_2d(z).T))
    p = model.to_params(x)
    out = {}

    def rec(d, pre):
        if isinstance(d, dict):
            for k, v in d.items():
                rec(v, pre + "/" + str(k))
        elif isinstance(d, (list, tuple)):
            for i, v in enumerate(d):
                rec(v, pre + "/" + str(i))
        else:
            out[pre] = float(np.asarray(d).ravel()[0])

    rec(p, "")
    return out


def test_z_param_names_matches_perturbation():
    model = _build()
    names = model.z_param_names
    n = model.num_free_params
    assert names is not None and len(names) == n

    z0 = jnp.zeros(n)
    base = _flat_params(model, z0)
    for i in range(n):
        fp = _flat_params(model, z0.at[i].add(0.7))
        moved = max(base, key=lambda k: abs(fp[k] - base[k]))
        assert moved.lstrip("/") == names[i], (
            f"column {i}: z_param_names says {names[i]!r} but perturbing that "
            f"column moved {moved.lstrip('/')!r} — column->name map is wrong (C-8).")


def test_z_param_names_not_insertion_order():
    # Regression guard: within a component the correct (sorted-key) order differs
    # from EPL's declared _params insertion order, which is the classic mislabel.
    model = _build()
    names = [n.split("/")[-1] for n in model.z_param_names]
    # EPL block occupies the first 6 columns (mass[0]); sorted-key order starts
    # with the centers, NOT theta_E (the insertion-order first element).
    assert names[0] == "center_x" and names[5] == "theta_E", names[:6]
