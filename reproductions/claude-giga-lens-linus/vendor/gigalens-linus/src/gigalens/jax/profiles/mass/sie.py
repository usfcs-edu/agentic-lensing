import functools

import jax.numpy as jnp
from jax import jit

from gigalens.jax.profile import MassProfile
from gigalens.physicality import (Domain, axis_ratio_constraint,
                                   ellipticity_constraint)


class SIE(MassProfile):
    _name = "SIE"
    s_scale = 1e-4
    _params = ["theta_E", "e1", "e2", "center_x", "center_y"]
    _domains = {
        "theta_E": Domain(lo=0.0, rationale=(
            "DEFINITION-BASED bound (not clip/NaN-derived; flagged for human "
            "review): the kernel faithfully computes the sign-flipped repulsive "
            "model for theta_E<0 (verified 2026-07-10) — raising here encodes "
            "the judgment that negative surface density is not a lens, and "
            "blocks deliberate negative-mass components.")),
        "e1": Domain(rationale="bounded jointly with e2 (see _joint_constraints)."),
        "e2": Domain(rationale="bounded jointly with e1 (see _joint_constraints)."),
        "center_x": Domain(rationale="any finite position is valid."),
        "center_y": Domain(rationale="any finite position is valid."),
    }
    _joint_constraints = (
        ellipticity_constraint(0.9999, exclude_circular=True, rationale=(
            "_param_conv clips c=sqrt(e1^2+e2^2) at 0.9999 (sie.py), so |e| >= "
            "0.9999 silently computes a different (extremely flattened, near-zero "
            "deflection) lens; at exactly e1=e2=0, q=1 makes deriv divide by "
            "sqrt(1-q^2)=0 -> all-NaN deflection (both verified 2026-07-10). For "
            "a circular lens use SIS.")),
        axis_ratio_constraint(0.2, rationale=(
            "axis ratio q<0.2 (ellipticity magnitude c>2/3) is an extremely "
            "flattened deflector — physically valid but rare, so a plausibility "
            "prompt.")),
    )

    @functools.partial(jit, static_argnums=(0,))
    def _param_conv(self, theta_E, e1, e2):
        s_scale = 0
        phi = jnp.arctan2(e2, e1) / 2
        c = jnp.minimum(jnp.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        q = (1 - c) / (1 + c)
        theta_E_conv = theta_E / (jnp.sqrt((1.0 + q ** 2) / (2.0 * q)))
        b = theta_E_conv * jnp.sqrt((1 + q ** 2) / 2)
        s = s_scale * jnp.sqrt((1 + q ** 2) / (2 * q ** 2))
        return b, s, q, phi

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, theta_E, e1, e2, center_x, center_y):
        b, s, q, phi = self._param_conv(theta_E, e1, e2)

        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        psi = jnp.sqrt(q ** 2 * (s ** 2 + x ** 2) + y ** 2)
        fx = (
                b
                / jnp.sqrt(1.0 - q ** 2)
                * jnp.arctan(jnp.sqrt(1.0 - q ** 2) * x / (psi + s))
        )
        fy = (
                b
                / jnp.sqrt(1.0 - q ** 2)
                * jnp.arctanh(jnp.sqrt(1.0 - q ** 2) * y / (psi + q ** 2 * s))
        )
        fx, fy = self._rotate(fx, fy, -phi)
        return fx, fy

    @functools.partial(jit, static_argnums=(0,))
    def _rotate(self, x, y, phi):
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi
