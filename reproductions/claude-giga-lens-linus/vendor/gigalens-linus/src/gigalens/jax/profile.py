import functools
import jax
from jax import jit, vjp, vmap
from jax.tree_util import Partial
from jax import numpy as jnp
import gigalens.profile
from abc import ABC


class MassProfile(gigalens.profile.MassProfile, ABC):
    """Tensorflow interface for a mass profile."""

    @functools.partial(jit, static_argnums=(0,))
    def hessian(self, x, y, **kwargs):
        """Calculates hessian with autograd in reverse mode.

                Args:
                    x: :math:`x` coordinate at which to evaluate the deflection
                    y: :math:`y` coordinate at which to evaluate the deflection
                    **kwargs: Mass profile parameters. Each parameter must be shaped in a way that is broadcastable with x and y

                Returns:
                    A tuple :math:`(\\f_xx, \\f_xy, \\f_yx, \\f_yy)` containing the hessian matrix in the :math:`x` and :math:`y` directions
        """
        # x = jax.lax.pcast(x, 'device', to='varying')
        # y = jax.lax.pcast(y, 'device', to='varying')
        x = jax.lax.pvary(x, 'device')
        y = jax.lax.pvary(y, 'device')
        
        partial_deriv = Partial(self.deriv, **kwargs)
        _, vjp_deriv = vjp(partial_deriv, x, y)
        std_basis = (
            jnp.stack([jnp.ones_like(x), jnp.zeros_like(x)]),
            jnp.stack([jnp.zeros_like(x), jnp.ones_like(x)])
        )
        # std_basis = jax.lax.pcast(std_basis, 'device', to='varying')
        # std_basis = jax.lax.pvary(std_basis, 'device')
        
        (f_xx, f_yx), (f_xy, f_yy) = vmap(vjp_deriv, in_axes=0, out_axes=0)(std_basis)
        return f_xx, f_xy, f_yx, f_yy

    def potential(self, x, y, **kwargs):
        r"""Lensing (deflection) potential :math:`\psi(x, y)`, where :math:`\alpha = \nabla\psi`.

        Unlike ``hessian``/``convergence``/``shear``, the potential is NOT recoverable
        generically from ``deriv`` (it requires integrating the deflection), so each
        profile that participates in potential-dependent observables — the Fermat
        potential and hence time delays — must implement it. The base raises rather than
        returning a silent wrong value: a profile without a potential fails loudly and BY
        NAME, instead of being mis-dispatched by parameter-name sniffing.

        Args:
            x, y: coordinates at which to evaluate the potential.
            **kwargs: this profile's mass parameters (broadcastable with ``x``/``y``).

        Returns:
            The lensing potential at ``(x, y)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement potential() (the lensing potential "
            "psi, with alpha = grad psi). It is required for Fermat-potential / time-delay "
            "observables such as the point-source likelihood. Implement potential() on this "
            "profile, or restrict the model to profiles that define it (currently EPL and "
            "external Shear).")

    @functools.partial(jit, static_argnums=(0,))
    def convergence(self, x, y, **kwargs):
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, **kwargs)
        kappa = (f_xx + f_yy) / 2
        return kappa

    @functools.partial(jit, static_argnums=(0,))
    def shear(self, x, y, **kwargs):
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, **kwargs)
        gamma1 = (f_xx - f_yy) / 2
        gamma2 = f_xy
        return gamma1, gamma2
