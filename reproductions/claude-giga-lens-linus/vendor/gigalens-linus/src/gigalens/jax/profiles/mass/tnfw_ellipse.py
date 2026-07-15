import functools

import jax.numpy as jnp
from jax import jit
import jax

from gigalens.jax.profile import MassProfile

# add sersic ellipse light, mass to light ratio

class TNFW_Ellipse(MassProfile):
    _name = "TNFW_Ellipse"
    _params = ["Rs", "alpha_Rs", "r_trunc", "e1", "e2", "center_x", "center_y"]

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, Rs, alpha_Rs, r_trunc, e1, e2, center_x, center_y):

        phi_G = jnp.arctan2(e2, e1) / 2
        c = jnp.sqrt(e1**2 + e2**2)
        c = jnp.minimum(c, 0.9999)
        q = (1 - c) / (1 + c)
        cos_phi = jnp.cos(phi_G)
        sin_phi = jnp.sin(phi_G)
        e = abs(1 - q**2) / (1 + q**2)
        x_shift = x - center_x
        y_shift = y - center_y
        dx = (cos_phi * x_shift + sin_phi * y_shift) * jnp.sqrt(1 - e)
        dy = (-sin_phi * x_shift + cos_phi * y_shift) * jnp.sqrt(1 + e)



        
        rho0 = alpha_Rs / (4.0 * Rs ** 2 * (1.0 + jnp.log(0.5)))
        R = jnp.sqrt(dx ** 2 + dy ** 2)
        R = jnp.maximum(R, 1e-6)
        x = R / Rs
        tau = r_trunc / Rs

        L = jnp.log(x / (tau + jnp.sqrt(tau ** 2 + x ** 2)))
        F = self.F(x)
        gx = (
                (tau ** 2)
                / (tau ** 2 + 1) ** 2
                * (
                        (tau ** 2 + 1 + 2 * (x ** 2 - 1)) * F
                        + tau * jnp.pi
                        + (tau ** 2 - 1) * jnp.log(tau)
                        + jnp.sqrt(tau ** 2 + x ** 2) * (-jnp.pi + L * (tau ** 2 - 1) / tau)
                )
        )
        a = 4 * rho0 * Rs * gx / x ** 2
        f_x_prim, f_y_prim = a * dx, a * dy

       
        f_x_prim *= jnp.sqrt(1 - e)
        f_y_prim *= jnp.sqrt(1 + e)
        f_x = cos_phi * f_x_prim - sin_phi * f_y_prim
        f_y = sin_phi * f_x_prim + cos_phi * f_y_prim
        return f_x, f_y


    @functools.partial(jit, static_argnums=(0,))
    def F(self, x):
        orig_shape = jnp.shape(x)
        # Flatten x for vmap
        x_flat = jnp.reshape(x, (-1,))

        def compute_F(x_val):
            # When x < 1:
            def f_inside(x_val):
                return 1.0 / jnp.sqrt(1.0 - x_val**2) * jnp.arctanh(jnp.sqrt(1.0 - x_val**2))
            # When x > 1:
            def f_outside(x_val):
                return 1.0 / jnp.sqrt(x_val**2 - 1.0) * jnp.arctan(jnp.sqrt(x_val**2 - 1.0))
            # When x == 1: return 1.0
            def f_equal(x_val):
                # FIX: Multiply by 0 and add 1 to preserve the JAX tracer metadata
                return x_val * 0.0 + 1.0

            # Use a nested conditional:
            return jax.lax.cond(
                x_val < 1.0,
                f_inside,
                lambda x_val: jax.lax.cond(x_val > 1.0, f_outside, f_equal, x_val),
                operand=x_val
            )

        # Vectorize the scalar function so that it is applied element-wise.
        F_vmap = jax.vmap(compute_F)
        result_flat = F_vmap(x_flat)
        return jnp.reshape(result_flat, orig_shape)
    
    # @functools.partial(jit, static_argnums=(0,))
    # def F(self, x):
    #     x = jnp.asarray(x)                      # scalar or array
    #     f_lt1 = 1.0 / jnp.sqrt(1 - x**2) * jnp.arctanh(jnp.sqrt(1 - x**2))
    #     f_gt1 = 1.0 / jnp.sqrt(x**2 - 1) * jnp.arctan(jnp.sqrt(x**2 - 1))
    #     f_eq1 = jnp.ones_like(x)
    #     return jnp.where(x < 1, f_lt1,
    #                      jnp.where(x > 1, f_gt1, f_eq1))


