import functools

import jax.numpy as jnp
from jax import jit, lax

from gigalens.jax.profile import MassProfile
from gigalens.jax.functions.hyp2f1 import hyp2f1
from gigalens.physicality import (Domain, axis_ratio_constraint,
                                   ellipticity_constraint)

_EPL_DOMAINS = {
    "theta_E": Domain(lo=0.0, lo_open=True, rationale=(
        "theta_E=0 silently renders identically-zero deflection (no lens); "
        "theta_E<0 makes b<0, so (b/R)**(gamma-2) is a fractional power of a "
        "negative number (NaN) except at gamma=2, where it silently renders the "
        "sign-flipped repulsive model (verified 2026-07-10).")),
    "gamma": Domain(lo=1.0, hi=3.0, lo_open=True, hi_open=True, soft=(1.1, 3.0),
                    rationale=(
        "the Tessore & Metcalf (2015) angular series is valid for slope "
        "t=gamma-1 in (0,2); the recurrence prefactor -f*(2n-(2-t))/(2n+(2-t)) "
        "has poles at gamma=2n+3 (gamma=5 verified NaN 2026-07-10), and outside "
        "(1,3) the fixed-niter series is not a convergent representation. "
        "gamma<=1 is a non-lensing (non-convergent total mass) slope; gamma>=3 "
        "has a non-integrable center. Soft band gamma>=1.1: slopes just above 1 "
        "are unlikely, and in some cases inferencer can be biased toward them — a plausibility "
        "prompt, not a hard bound.")),
    "e1": Domain(rationale="bounded jointly with e2 (see _joint_constraints)."),
    "e2": Domain(rationale="bounded jointly with e1 (see _joint_constraints)."),
    "center_x": Domain(rationale="any finite position is valid."),
    "center_y": Domain(rationale="any finite position is valid."),
}
_EPL_JOINT = (
    ellipticity_constraint(1.0, rationale=(
        "deriv clips c=sqrt(e1^2+e2^2) to [0,1] (epl.py); at c=1, q=0 makes "
        "b=0 and the kernel silently renders identically-zero deflection "
        "(verified 2026-07-10: |e|=1.3 -> all-zero).")),
    axis_ratio_constraint(0.2, rationale=(
        "axis ratio q<0.2 (ellipticity magnitude c>2/3) is an extremely "
        "flattened mass distribution — physically valid but rare for a galaxy- "
        "or group-scale deflector, so a plausibility prompt.")),
)


class EPL(MassProfile):
    _name = "EPL"
    _params = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y"]
    _domains = _EPL_DOMAINS
    _joint_constraints = _EPL_JOINT

    def __init__(self, niter=18):
        super().__init__()
        self.niter = niter

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, theta_E, gamma, e1, e2, center_x, center_y):
        phi = jnp.arctan2(e2, e1) / 2
        c = jnp.clip(jnp.sqrt(e1 ** 2 + e2 ** 2), 0, 1)
        q = (1 - c) / (1 + c)
        theta_E_conv = theta_E / (jnp.sqrt((1.0 + q ** 2) / (2.0 * q)))
        b = theta_E_conv * jnp.sqrt((1 + q ** 2) / 2)
        t = gamma - 1

        x, y = x - center_x, y - center_y
        x, y = self.rotate(x, y, phi)

        R = jnp.clip(jnp.sqrt((q * x) ** 2 + y ** 2), 1e-10, 1e10)
        angle = jnp.arctan2(y, q * x)
        f = (1 - q) / (1 + q)
        Cs, Ss = jnp.cos(angle), jnp.sin(angle)
        Cs2, Ss2 = jnp.cos(2 * angle), jnp.sin(2 * angle)

        def update(n, val):
            prefac_ = -f * (2 * n - (2 - t)) / (2 * n + (2 - t))
            last_x, last_y, fx_, fy_ = val
            last_x, last_y = prefac_ * (Cs2 * last_x - Ss2 * last_y), prefac_ * (
                    Ss2 * last_x + Cs2 * last_y
            )
            fx_ += last_x
            fy_ += last_y
            return last_x, last_y, fx_, fy_

        _, _, fx, fy = lax.fori_loop(1, self.niter, update, (Cs, Ss, Cs, Ss))
        prefac = (2 * b) / (1 + q) * ((b / R) ** (t - 1))
        fx, fy = fx * prefac, fy * prefac
        return self.rotate(fx, fy, -phi)

    @functools.partial(jit, static_argnums=(0,))
    def rotate(self, x, y, phi):
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi

    def potential(self, x, y, theta_E, gamma, e1, e2, center_x, center_y):
        r"""Lensing potential :math:`\psi` of the elliptical power law.

        For a power-law profile :math:`\psi` is homogeneous of degree :math:`2 - t` with
        :math:`t = \gamma - 1` (:math:`\kappa \sim r^{-t} \Rightarrow \psi \sim r^{2-t}`),
        so by Euler's theorem :math:`(x-c_x)\alpha_x + (y-c_y)\alpha_y = (2 - t)\,\psi`
        (exact for the elliptical case too, since :math:`\psi` is homogeneous in the
        shifted coordinates). Hence the closed form below. Validated against lenstronomy
        to ~1e-15 (point-source Link-A test).

        Deliberately NOT ``@jit``-decorated (unlike ``deriv``): it is a thin wrapper always
        evaluated inside the already-jitted likelihood, so an extra jit boundary buys no
        speed and only changes XLA float reassociation — mathematically identical, but
        enough to move a razor-stiff fit. Keeping it un-jitted makes this a numerically
        exact code-motion of the former inline computation.
        """
        alpha_x, alpha_y = self.deriv(x, y, theta_E, gamma, e1, e2, center_x, center_y)
        t = gamma - 1
        return ((x - center_x) * alpha_x + (y - center_y) * alpha_y) / (2 - t)


class EPL_sean(MassProfile):
    _name = "EPL"
    _params = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y"]
    # Same parameterization and same clip/series structure as EPL (c clipped to
    # [0,1]; hyp2f1(1, t/2, 2-t/2, ...) shares the t in (0,2) validity domain).
    _domains = _EPL_DOMAINS
    _joint_constraints = _EPL_JOINT

    def __init__(self, niter=18):
        super().__init__()
        self.niter = niter

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, theta_E, gamma, e1, e2, center_x, center_y):
        phi = jnp.arctan2(e2, e1) / 2
        c = jnp.clip(jnp.sqrt(e1 ** 2 + e2 ** 2), 0, 1)
        q = (1 - c) / (1 + c)
        theta_E_conv = theta_E / (jnp.sqrt((1.0 + q ** 2) / (2.0 * q)))
        b = theta_E_conv * jnp.sqrt((1 + q ** 2) / 2)
        t = gamma - 1

        x, y = x - center_x, y - center_y
        x, y = self.rotate(x, y, phi)

        R = jnp.clip(jnp.sqrt((q * x) ** 2 + y ** 2), 1e-10, 1e10)
        angle = jnp.arctan2(y, q * x)
        f = (1 - q) / (1 + q)

        alpha = jnp.exp(1j*angle)*hyp2f1(1, t/2, 2-t/2, -f*jnp.exp(1j*2*angle), niter=self.niter)
        prefac = (2 * b) / (1 + q) * ((b / R) ** (t - 1))
        fx, fy = alpha.real * prefac, alpha.imag * prefac
        return self.rotate(fx, fy, -phi)

    @functools.partial(jit, static_argnums=(0,))
    def rotate(self, x, y, phi):
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi