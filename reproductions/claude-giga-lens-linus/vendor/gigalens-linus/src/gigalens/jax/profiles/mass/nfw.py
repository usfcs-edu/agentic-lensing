import functools

import jax.numpy as jnp
from jax import jit

from gigalens.jax.profile import MassProfile
from gigalens.physicality import (Domain, axis_ratio_constraint,
                                   ellipticity_constraint)

_NFW_RS_DOMAIN = Domain(lo=0.0, lo_open=True, rationale=(
    "rho0 = alpha_Rs / (4 Rs^2 (1-ln2)) divides by Rs (nfw.py deriv; Rs=0 "
    "verified inf/NaN 2026-07-10), and nfwAlpha silently clamps Rs below at "
    "_r_min=1e-7, so Rs<0 silently renders a ~zero-deflection model "
    "(verified: |alpha| ~ 3e-20)."))
_NFW_CENTER = Domain(rationale="any finite position is valid.")


class NFW(MassProfile):

    _name = 'NFW'
    _params = ['Rs', 'alpha_Rs', 'center_x', 'center_y']
    _r_min = 0.0000001
    _c = 0.000001
    _domains = {
        "Rs": _NFW_RS_DOMAIN,
        "alpha_Rs": Domain(lo=0.0, rationale=(
            "DEFINITION-BASED bound (not clip/NaN-derived; flagged for human "
            "review): alpha_Rs sets the sign of rho0 and the kernel faithfully "
            "renders the repulsive mirror model for alpha_Rs<0 (verified "
            "2026-07-10); raising blocks deliberate negative-mass components.")),
        "center_x": _NFW_CENTER,
        "center_y": _NFW_CENTER,
    }

    def __init__(self):
        super(NFW, self).__init__()

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, Rs, alpha_Rs, center_x, center_y):
        rho0 = alpha_Rs / (4. * Rs ** 2 * (1. - jnp.log(2.)))
        x, y = x - center_x, y - center_y
        R = jnp.sqrt(x ** 2 + y ** 2)
        f_x, f_y = self.nfwAlpha(R, Rs, rho0, x, y)
        return f_x, f_y

    @functools.partial(jit, static_argnums=(0,))
    def nfwAlpha(self, R, Rs, rho0, ax_x, ax_y):
        R = jnp.maximum(self._r_min, R)
        Rs = jnp.maximum(self._r_min, Rs)
        x = R / Rs
        gx = self.g_(x)
        a = 4 * rho0 * Rs * gx / x ** 2
        return a * ax_x, a * ax_y

    @functools.partial(jit, static_argnums=(0,))
    def g_(self, x):
        x = jnp.maximum(self._c, x)
        # FIX: Clamp inputs to the valid domain of each branch *before* evaluating
        # them. jnp.where (3-arg form) evaluates BOTH branches for every element,
        # so the false branch would compute sqrt(x²-1) for x<1 (→ NaN) and the
        # true branch would compute acosh(1/x) for x>1 (→ NaN) in the original
        # code.  Those NaNs are masked out in the forward pass, but in older JAX
        # versions they can leak into gradient computation.
        x_lt1 = jnp.clip(x, self._c, 1.0 - 1e-6)   # safe for acosh(1/x), sqrt(1-x²)
        x_gt1 = jnp.clip(x, 1.0 + 1e-6, jnp.inf)   # safe for acos(1/x),  sqrt(x²-1)
        g_lt1 = jnp.log(x_lt1 / 2.) + 1. / jnp.sqrt(1. - x_lt1 ** 2) * jnp.acosh(1. / x_lt1)
        g_gt1 = jnp.log(x_gt1 / 2.) + 1. / jnp.sqrt(x_gt1 ** 2 - 1.) * jnp.acos(1. / x_gt1)
        # At x==1 the limit of both branches is 1 + log(0.5) = 1 - log(2) ≈ 0.307;
        # the x_gt1 clamp makes x_gt1 = 1+1e-6 there, which evaluates to ≈ 0.307. ✓
        return jnp.where(x < 1., g_lt1, g_gt1)

    @functools.partial(jit, static_argnums=(0,))
    def F_(self, x):
        # FIX (crash cause): the original code used jnp.where(x < 1) with a
        # single argument, which returns *indices* (like np.nonzero).  Under JAX
        # JIT the number of True elements is data-dependent, so the resulting
        # index arrays have dynamic shapes.  JAX cannot represent dynamic shapes
        # at trace time and raises ConcretizationTypeError, which in many JAX
        # versions kills the kernel rather than raising a clean Python exception.
        #
        # Fix: use the 3-arg ternary jnp.where and clamp each branch's input to
        # its valid domain (same NaN-safety pattern as g_ above).
        x = jnp.maximum(self._c, x)
        x_lt1 = jnp.clip(x, self._c, 1.0 - 1e-6)
        x_gt1 = jnp.clip(x, 1.0 + 1e-6, jnp.inf)
        f_lt1 = 1. / (x_lt1 ** 2 - 1.) * (
            1. - 2. / jnp.sqrt(1. - x_lt1 ** 2) * jnp.arctanh(jnp.sqrt((1. - x_lt1) / (1. + x_lt1)))
        )
        f_gt1 = 1. / (x_gt1 ** 2 - 1.) * (
            1. - 2. / jnp.sqrt(x_gt1 ** 2 - 1.) * jnp.arctan(jnp.sqrt((x_gt1 - 1.) / (1. + x_gt1)))
        )
        # F_(1) = 1/3 is the analytic limit from both sides (matches Lenstronomy)
        return jnp.where(x < 1., f_lt1, jnp.where(x > 1., f_gt1, 1. / 3.))

    @functools.partial(jit, static_argnums=(0,))
    def hessian(self, x, y, Rs, alpha_Rs, center_x, center_y):
        rho0 = alpha_Rs / (4. * Rs ** 2 * (1. - jnp.log(2.)))
        Rs = jnp.maximum(self._r_min, Rs)
        x, y = x - center_x, y - center_y
        R = jnp.sqrt(x ** 2 + y ** 2)
        R = jnp.maximum(self._c, R)
        X = R / Rs
        gx = self.g_(X)
        Fx = self.F_(X)
        kappa = 2 * rho0 * Rs * Fx
        a = 2 * rho0 * Rs * (2 * gx / X ** 2 - Fx)
        gamma1, gamma2 = a * (y ** 2 - x ** 2) / R ** 2, -a * 2 * (x * y) / R ** 2
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy


_NFW_ELL_JOINT = (
    ellipticity_constraint(0.9999, rationale=(
        "_param_conv clips c=sqrt(e1^2+e2^2) at 0.9999 (nfw.py): |e| >= 0.9999 "
        "silently computes a different (maximally flattened) halo.")),
    axis_ratio_constraint(0.2, rationale=(
        "axis ratio q<0.2 (ellipticity magnitude c>2/3) is an extremely "
        "flattened halo — physically valid but rare, so a plausibility prompt.")),
)


class NFW_ELLIPSE(MassProfile):

    _name = 'NFW_ELLIPSE'
    _params = ['Rs', 'alpha_Rs', 'e1', 'e2', 'center_x', 'center_y']
    _domains = {
        "Rs": _NFW_RS_DOMAIN,
        "alpha_Rs": NFW._domains["alpha_Rs"],
        "e1": Domain(rationale="bounded jointly with e2 (see _joint_constraints)."),
        "e2": Domain(rationale="bounded jointly with e1 (see _joint_constraints)."),
        "center_x": _NFW_CENTER,
        "center_y": _NFW_CENTER,
    }
    _joint_constraints = _NFW_ELL_JOINT

    def __init__(self):
        self.nfw = NFW()
        super(NFW_ELLIPSE, self).__init__()

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, Rs, alpha_Rs, e1, e2, center_x, center_y):
        rho0 = alpha_Rs / (4. * Rs ** 2 * (1. - jnp.log(2.)))
        e, phi = self._param_conv(e1, e2)

        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        x, y = x * jnp.sqrt(1 - e), y * jnp.sqrt(1 + e)
        R = jnp.sqrt(x ** 2 + y ** 2)

        fx, fy = self.nfw.nfwAlpha(R, Rs, rho0, x, y)
        fx *= jnp.sqrt(1 - e)
        fy *= jnp.sqrt(1 + e)
        fx, fy = self._rotate(fx, fy, -phi)
        return fx, fy

    @functools.partial(jit, static_argnums=(0,))
    def _rotate(self, x, y, phi):
        cos_phi = jnp.cos(phi)
        sin_phi = jnp.sin(phi)
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi

    @functools.partial(jit, static_argnums=(0,))
    def _param_conv(self, e1, e2):
        phi = jnp.arctan2(e2, e1) / 2
        c = jnp.minimum(jnp.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        q = (1 - c) / (1 + c)
        e = jnp.abs(1 - q ** 2) / (1 + q ** 2)
        return e, phi


class NFW_ELLIPSE_EINSTEIN(MassProfile):
    #* Exact same profile, but reparamterized in terms of Einstein radius instead of alpha_Rs
    _name = 'NFW_ELLIPSE_EINSTEIN'
    _params = ['theta_E', 'Rs', 'e1', 'e2', 'center_x', 'center_y']
    _domains = {
        "theta_E": Domain(lo=0.0, lo_open=True, rationale=(
            "rho0 = theta_E^2 / (4 Rs^3 g_(theta_E/Rs)) (nfw.py deriv): theta_E<0 "
            "hits g_'s input clamp at 1e-6 and silently renders garbage deflection "
            "(verified 2026-07-10: |alpha| ~ 9e10); theta_E=0 divides 0/0.")),
        "Rs": Domain(lo=0.0, lo_open=True, rationale=(
            "Rs^3 flips the sign of rho0 for Rs<0 and the nfwAlpha clamp then "
            "silently renders a ~zero model (verified: |alpha| ~ 5e-9); Rs=0 "
            "divides by zero.")),
        "e1": Domain(rationale="bounded jointly with e2 (see _joint_constraints)."),
        "e2": Domain(rationale="bounded jointly with e1 (see _joint_constraints)."),
        "center_x": _NFW_CENTER,
        "center_y": _NFW_CENTER,
    }
    _joint_constraints = _NFW_ELL_JOINT

    def __init__(self):
        self.nfw = NFW()
        super(NFW_ELLIPSE_EINSTEIN, self).__init__()

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, theta_E, Rs, e1, e2, center_x, center_y):
        rho0 = theta_E**2 / (4 * Rs**3 * self.nfw.g_(theta_E/Rs))
        e, phi = self._param_conv(e1, e2)

        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        x, y = x * jnp.sqrt(1 - e), y * jnp.sqrt(1 + e)
        R = jnp.sqrt(x ** 2 + y ** 2)

        fx, fy = self.nfw.nfwAlpha(R, Rs, rho0, x, y)
        fx *= jnp.sqrt(1 - e)
        fy *= jnp.sqrt(1 + e)
        fx, fy = self._rotate(fx, fy, -phi)
        return fx, fy

    @functools.partial(jit, static_argnums=(0,))
    def _rotate(self, x, y, phi):
        cos_phi = jnp.cos(phi)
        sin_phi = jnp.sin(phi)
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi

    @functools.partial(jit, static_argnums=(0,))
    def _param_conv(self, e1, e2):
        phi = jnp.arctan2(e2, e1) / 2
        c = jnp.minimum(jnp.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        q = (1 - c) / (1 + c)
        e = jnp.abs(1 - q ** 2) / (1 + q ** 2)
        return e, phi