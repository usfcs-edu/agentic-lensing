import functools

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.special import beta, gammaln

from gigalens.jax.profile import MassProfile
from gigalens.jax.functions.hyp2f1_claude import hyp2f1_series, hyp2f1_goursat, hyp2f1_one_minus_z, hyp2f1_half_one_threehalf, hyp2f1_a_1_ap1
from gigalens.jax.functions.spence import spence
from gigalens.physicality import (Domain, JointConstraint, axis_ratio_constraint,
                                   ellipticity_constraint)

class BPL(MassProfile):
    _name = "BPL"
    _params = ["b", "alpha", "alpha_c", "r_c", "e1", "e2", "center_x", "center_y"]
    _domains = {
        "b": Domain(lo=0.0, lo_open=True, rationale=(
            "(b/R_el)**(alpha-1) is a fractional power of a negative number for "
            "b<0 (NaN for non-integer exponent; at alpha=2 it silently renders "
            "the sign-flipped repulsive model, verified 2026-07-10).")),
        "alpha": Domain(lo=1.0, hi=3.0, lo_open=True, hi_open=True, rationale=(
            "alpha_2 divides by (3-alpha) (bpl.py; alpha=3 verified NaN "
            "2026-07-10) and inv_B carries an (alpha-1) factor that kills the "
            "core term at alpha=1; outer slope outside (1,3) is non-integrable "
            "or non-lensing (definition).")),
        "alpha_c": Domain(lo=0.0, hi=3.0, hi_open=True, rationale=(
            "2/(3-alpha_c) has a pole at alpha_c=3 (bpl.py; verified NaN "
            "2026-07-10); a negative inner slope inverts the core (definition).")),
        "r_c": Domain(lo=0.0, lo_open=True, rationale=(
            "deriv silently clamps r_c below at 1e-12 (bpl.py), so r_c<=0 "
            "computes a different (coreless) model. Bounded above jointly with "
            "b (see _joint_constraints).")),
        "e1": Domain(rationale="bounded jointly with e2 (see _joint_constraints)."),
        "e2": Domain(rationale="bounded jointly with e1 (see _joint_constraints)."),
        "center_x": Domain(rationale="any finite position is valid."),
        "center_y": Domain(rationale="any finite position is valid."),
    }
    _joint_constraints = (
        ellipticity_constraint(1.0, rationale=(
            "deriv clips c=sqrt(e1^2+e2^2) to [1e-12, 1] (bpl.py): |e|>=1 "
            "silently computes the maximally-flattened q->0 model.")),
        axis_ratio_constraint(0.2, rationale=(
            "axis ratio q<0.2 (ellipticity magnitude c>2/3) is an extremely "
            "flattened deflector — physically valid but rare, so a plausibility "
            "prompt.")),
        JointConstraint(
            name="r_c <= b", params=("r_c", "b"),
            ok=lambda r_c, b: r_c <= b,
            rationale=(
                "deriv silently clamps r_c to min(r_c, b) (bpl.py): any r_c > b "
                "computes the r_c = b model bitwise-identically (verified "
                "2026-07-10), so the sampled/fixed r_c is not the model "
                "rendered.")),
    )

    def __init__(self, niter=18):
        super().__init__()
        self.niter = niter

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, b, alpha, alpha_c, r_c, e1, e2, center_x, center_y):
        r_c = jnp.minimum(r_c, b)
        r_c = jnp.maximum(1e-12, r_c)
        
        phi = jnp.arctan2(e2, e1) / 2
        c = jnp.clip(jnp.sqrt(e1 ** 2 + e2 ** 2), 1e-12, 1)
        q = (1 - c) / (1 + c)
        
        x, y = x - center_x, y - center_y
        x, y = self.rotate(x, y, phi)
        
        R_el = jnp.sqrt(q*x**2+y**2/q) + 0j
        # In deriv:
        z = x + 1j * y
        safe_z = jnp.where(jnp.abs(z) < 1e-12, 1e-12 + 0j, z)
        zeta_squared = (1/q - q) / safe_z**2

        C = r_c**2*zeta_squared
        
        arg_zel = 1.0 - R_el**2 / r_c**2
        # clamp magnitude away from 0 so sqrt's gradient (0.5/sqrt) stays finite
        safe_arg_zel = jnp.where(jnp.abs(arg_zel) < 1e-9, 1e-9 + 0j, arg_zel)
        z_el = jnp.sqrt(safe_arg_zel)

        alpha = self.alpha_1(safe_z, R_el, b, alpha, zeta_squared) + self.alpha_2(r_c, alpha, b, alpha_c, C, safe_z, R_el, z_el)
        alpha_x = alpha.real
        alpha_y = -alpha.imag
        return self.rotate(alpha_x, alpha_y, -phi)

    def fancy_F(self, a, z):
        safe_a = jnp.where(a == 1/2, 0.0, a)
        safe_z = jnp.where(z == 0, 1e-12 + 0j, z)
        
        one = 1/(1-2*safe_a)*(hyp2f1_a_1_ap1(safe_a,safe_z)-2*safe_a*hyp2f1_half_one_threehalf(safe_z))        
        two = (spence(jnp.sqrt(safe_z))-spence(-jnp.sqrt(safe_z)))/(2*jnp.sqrt(safe_z))
        
        return jnp.where(a==1/2, two, one)
    
    def S0(self, alpha_c, alpha, C, z_el, R_el, r_c, niter=50):
        """
        Evaluate the series:
    
            1/sqrt(1 - C) * sum_{n=0}^inf
                [ (alpha_c/2)^(n) - (alpha/2)^(n) ] / (3/2)^(n)
                * 2 z_el^(2n+3) / (2n+3)
                * F(1/2, (2n+3)/2, (2n+5)/2, C z_el^2 / (C - 1))
    
        The Pochhammer ratios r_n = (x)^(n) / (3/2)^(n) are carried by the
        recurrence r_{n+1} = r_n * (x + n) / (3/2 + n), avoiding the
        large-cancellation and digamma poles of the gammaln difference (which
        misbehaves under AD as alpha_c -> 0). The running power
        zpow = z_el**(2n+3) is likewise carried by a single multiply, replacing
        the log/exp reconstruction whose gradient blew up for small z_el.
    
        rc/ra are scalar (or param-batch-shaped under vmap); acc/zpow are
        spatial-grid-shaped. Vectorization over the grid is preserved.
        """
        in_region = jnp.abs(R_el) < r_c
        z_el_safe = jnp.where(in_region, z_el, jnp.zeros_like(z_el))
        z2 = z_el_safe ** 2

        arg = C * z2 / (C - 1.0)
    
        def body(n, carry):
            acc, rc, ra, zpow = carry      # rc=(ac/2)^(n)/(3/2)^(n), ra=(a/2)^(n)/(3/2)^(n), zpow=z_el**(2n+3)
    
            coeff = rc - ra
            b = (2.0 * n + 3.0) / 2.0
            hyp = hyp2f1_goursat(0.5, b, arg)
            acc = acc + coeff * (2.0 * zpow / (2.0 * n + 3.0)) * hyp
            
            denom = 1.5 + n
            rc_next = rc * (alpha_c / 2.0 + n) / denom
            ra_next = ra * (alpha / 2.0 + n) / denom
            zpow_next = zpow * z2           # z_el**(2(n+1)+3) = z_el**(2n+3) * z_el**2
    
            return (acc, rc_next, ra_next, zpow_next)
    
        init = (
            jnp.zeros_like(z2),                  # acc   — spatial grid shape
            jnp.ones_like(alpha_c + alpha),      # rc_0  — scalar, or param-batch shape
            jnp.ones_like(alpha_c + alpha),      # ra_0  — scalar, or param-batch shape
            z_el_safe ** 3,                      # zpow_0 = z_el**(2*0+3), spatial grid shape, AD-safe at 0
        )
        acc, _, _, _ = lax.fori_loop(0, niter, body, init)
    
        total = acc / jnp.sqrt(1.0 - C)
        return jnp.where(in_region, total, jnp.zeros_like(total))

    def alpha_1(self, z, R_el, b, alpha, zeta_squared):
        safe_R_el = jnp.where(jnp.abs(R_el) < 1e-12, 1e-12 + 0j, R_el)
        return safe_R_el**2 / z * (b / safe_R_el)**(alpha-1) * hyp2f1_goursat(1/2, (3-alpha)/2, zeta_squared*safe_R_el**2)

    def alpha_2(self, r_c, alpha, b, alpha_c, C, z, R_el, z_el):
        # B = lambda x: beta(1/2, (x-1)/2)
        inv_B = jnp.exp(gammaln(alpha/2)-gammaln((alpha+1)/2))*(alpha-1)/(2*jnp.sqrt(jnp.pi))
        first = r_c**2/z*(3-alpha)*inv_B*(b/r_c)**(alpha-1)
        second = 2/(3-alpha_c)*self.fancy_F((3-alpha_c)/2, C)-2/(3-alpha)*self.fancy_F((3-alpha)/2,C)-self.S0(alpha_c, alpha, C, z_el, R_el, r_c, niter=self.niter)
        return first*second
        
    @functools.partial(jit, static_argnums=(0,))
    def rotate(self, x, y, phi):
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi
