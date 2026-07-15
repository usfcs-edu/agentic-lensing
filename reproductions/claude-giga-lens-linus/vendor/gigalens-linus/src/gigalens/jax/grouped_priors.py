"""Building blocks for grouped (tuple-key) priors — see ``scene.py``.

The first concrete member is a **disk-bounded ellipticity** prior. Ellipticity is
parameterized by ``(e1, e2)`` with modulus ``c = sqrt(e1^2 + e2^2) = (1-q)/(1+q)``
(EPL/SIE convention), so the axis ratio is ``q = (1-c)/(1+c)``. Bounding each of
``e1``/``e2`` in a *box* is anisotropic in position angle and lets the corner reach
``c ~ 0.71`` (``q ~ 0.17``); bounding the *modulus* via a polar ``(c, phi)``
parameterization reintroduces a coordinate singularity at ``c -> 0`` (the round
regime real lens galaxies actually occupy).

``DiskBijector`` avoids both: it maps all of R^2 onto the open disk ``|e| < R`` in
Cartesian coordinates,

    e = R * u / sqrt(1 + |u|^2),      u in R^2,

so the hard cap ``|e| < R`` (i.e. ``q > (1-R)/(1+R)``) holds for *every* u, the map
is regular at ``u = 0`` (``e ~ R*u``, no funnel), and the boundary sits at
``|u| -> inf`` — there is no finite reflecting wall for HMC/MCLMC to fight. A plain
Gaussian on ``u`` induces a prior peaked toward round with a controllable spread and
a genuine support cap. ``DiskEllipticity`` bundles that Gaussian-on-u + the disk map
into a distribution whose ``experimental_default_event_space_bijector`` is the disk
map, so it drops straight into a scene tuple key: ``("e1", "e2"): DiskEllipticity(...)``.

The same class serves any disk-bounded pair (e.g. external shear ``(g1, g2)`` with
``e_max = g_max``); the modulus/axis-ratio language is just the ellipticity framing.
"""
from __future__ import annotations

import math
from typing import Sequence

from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd, bijectors as tfb


class DiskBijector(tfb.Bijector):
    """Diffeomorphism ``R^2 -> {|e| < radius}``:  ``e = radius * u / sqrt(1 + |u|^2)``.

    Event ndims 1 (acts on the last axis, a 2-vector). Analytic inverse and
    log-det-Jacobian; the latter is ``2*ln(radius) - 2*ln(1 + |u|^2)`` (the n=2 case
    of ``det J = radius^2 / (1 + |u|^2)^2``).
    """

    def __init__(self, radius: float = 0.5, validate_args: bool = False,
                 name: str = "disk"):
        parameters = dict(locals())
        if not (float(radius) > 0.0):
            raise ValueError(f"DiskBijector radius must be > 0; got {radius}.")
        self._radius = float(radius)
        super().__init__(
            forward_min_event_ndims=1,
            is_constant_jacobian=False,
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    @property
    def radius(self) -> float:
        return self._radius

    def _forward(self, u):
        r2 = jnp.sum(jnp.square(u), axis=-1, keepdims=True)
        return self._radius * u / jnp.sqrt(1.0 + r2)

    def _inverse(self, e):
        r2 = self._radius ** 2
        e2 = jnp.sum(jnp.square(e), axis=-1, keepdims=True)
        # e2 < r2 on the open disk; floor guards the boundary/round-off so inverse is
        # finite. jnp.maximum (not jnp.clip's a_min=, dropped in newer JAX).
        denom = jnp.sqrt(jnp.maximum(r2 - e2, jnp.finfo(jnp.result_type(e)).tiny))
        return e / denom

    def _forward_log_det_jacobian(self, u):
        r2 = jnp.sum(jnp.square(u), axis=-1)  # reduce the event axis -> batch shape
        return 2.0 * math.log(self._radius) - 2.0 * jnp.log1p(r2)


def DiskEllipticity(e_max: float = 0.5, scale: float = 0.25,
                    mode: Sequence[float] = (0.0, 0.0), dtype=None,
                    validate_args: bool = False, name: str = "DiskEllipticity"):
    """Disk-bounded ellipticity prior over ``(e1, e2)`` (event_shape ``[2]``).

    Returns a ``tfd.TransformedDistribution``: ``u ~ Normal(u0, scale)`` in R^2 pushed
    through :class:`DiskBijector`, so samples satisfy ``|e| < e_max`` (``q >
    (1-e_max)/(1+e_max)``). Its ``experimental_default_event_space_bijector`` IS the disk
    map, so it drops into a scene tuple key: ``("e1","e2"): DiskEllipticity(...)``.

    (A plain ``TransformedDistribution`` rather than a subclass — a redefined-``__init__``
    subclass trips TFP's ``_parameter_properties`` warning and CompositeTensor issues.)

    Args:
      e_max: hard cap on the ellipticity modulus ``|e|`` (disk radius). ``0.5`` gives
        ``q >= 1/3``; the induced prior keeps essentially all mass well inside.
      scale: std of the unconstrained Gaussian on each of ``u1, u2``. Larger = more
        ellipticity spread. ~0.25 centers on ``q ~ 0.75`` at ``e_max = 0.5``.
      mode: the ellipticity ``(e01, e02)`` the prior is centered on (default round).
        Must satisfy ``|mode| < e_max``.
      dtype: float dtype of the prior. Defaults to the ambient JAX default float
        (float64 when x64 is enabled, else float32). It MUST match the model's other
        priors: a scene packs one flat ``z`` of a single dtype (LensModel enforces
        dtype-uniformity), so for a float64 model pass ``dtype=jnp.float64`` — or rely on
        the x64 default, which already yields float64.
    """
    e_max = float(e_max)
    m0, m1 = float(mode[0]), float(mode[1])
    mm = m0 * m0 + m1 * m1
    if not (mm < e_max * e_max):
        raise ValueError(
            f"DiskEllipticity mode {mode} has |mode|={math.sqrt(mm):.3f} >= e_max="
            f"{e_max}; the centre must lie strictly inside the disk.")
    # u0 = mode / sqrt(e_max^2 - |mode|^2)  (the disk-map inverse of `mode`).
    denom = math.sqrt(e_max * e_max - mm)
    u0 = [m0 / denom, m1 / denom]
    if dtype is None:
        dtype = jnp.zeros(()).dtype  # ambient default float (float64 iff x64 enabled)
    base = tfd.MultivariateNormalDiag(
        loc=jnp.asarray(u0, dtype),
        scale_diag=jnp.full((2,), float(scale), dtype))
    return tfd.TransformedDistribution(
        base, DiskBijector(radius=e_max, validate_args=validate_args), name=name)
