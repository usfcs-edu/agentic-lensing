import functools

import jax.numpy as jnp
from jax import jit

from gigalens.jax.profile import MassProfile
from gigalens.physicality import Domain


class SIS(MassProfile):
    _name = "SIS"
    _params = ["theta_E", "center_x", "center_y"]
    _domains = {
        "theta_E": Domain(lo=0.0, rationale=(
            "DEFINITION-BASED bound (not clip/NaN-derived; flagged for human "
            "review): the kernel faithfully computes the sign-flipped repulsive "
            "deflection for theta_E<0 (verified 2026-07-10); raising blocks "
            "deliberate negative-mass components.")),
        "center_x": Domain(rationale="any finite position is valid."),
        "center_y": Domain(rationale="any finite position is valid."),
    }

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, theta_E, center_x, center_y):
        dx, dy = x - center_x, y - center_y
        R = jnp.sqrt(dx ** 2 + dy ** 2)
        a = jnp.where(R == 0, 0.0, theta_E / R)
        return a * dx, a * dy

    @functools.partial(jit, static_argnums=(0,))
    def hessian(self, x, y, theta_E, center_x, center_y):

        x, y = x - center_x, y - center_y
        R = (x**2 + y**2)**(3./2)
        a = jnp.where(R == 0, 0.0, theta_E / R)

        f_xx = y**2 * a
        f_yy = x**2 * a
        f_xy = -x * y * a
        return f_xx, f_xy, f_xy, f_yy
