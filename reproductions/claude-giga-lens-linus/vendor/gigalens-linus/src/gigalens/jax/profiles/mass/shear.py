import functools

from jax import jit

from gigalens.jax.profile import MassProfile
from gigalens.physicality import Domain, shear_magnitude_constraint


class Shear(MassProfile):
    _name = "SHEAR"
    _params = ["gamma1", "gamma2"]
    _domains = {
        # The kernel is exactly linear (deriv = (g1*x + g2*y, g2*x - g1*y)): it
        # computes precisely the stated external shear for ANY finite values, so
        # no hard bound is code-derivable. Plausibility limits on |gamma| are a
        # soft-bound question — see _joint_constraints below.
        "gamma1": Domain(rationale="linear kernel is exact for any finite shear."),
        "gamma2": Domain(rationale="linear kernel is exact for any finite shear."),
    }
    _joint_constraints = (
        shear_magnitude_constraint(0.2, rationale=(
            "external-shear magnitude sqrt(gamma1^2+gamma2^2)>0.2 is much larger "
            "than realistic line-of-sight shear on galaxy/group scales — "
            "physically valid (the kernel is exact) but implausible, so a "
            "plausibility prompt, not a hard bound.")),
    )

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, gamma1, gamma2):
        return gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y

    def potential(self, x, y, gamma1, gamma2):
        r"""Lensing potential of external shear:
        :math:`\psi = \tfrac12\,[\gamma_1 x^2 + 2\gamma_2\, x y - \gamma_1 y^2]` (about the
        origin), whose gradient is exactly ``deriv``. Not ``@jit``-decorated — see
        :meth:`gigalens.jax.profiles.mass.epl.EPL.potential` for the rationale."""
        return 0.5 * (gamma1 * x * x + 2 * gamma2 * x * y - gamma1 * y * y)

    @functools.partial(jit, static_argnums=(0,))
    def hessian(self, x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        gamma1 = gamma1
        gamma2 = gamma2
        kappa = 0.
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy
