import jax.numpy as jnp
from gigalens.cosmo import CosmoBase

class wCDM_Cosmo(CosmoBase):
    _name = "wCDM"
    _params = ['H0', 'Om0', 'k', 'w0']

    def __init__(self, z_lens, z_source_ref):  # z_source_ref required (no silent default)
        super(wCDM_Cosmo, self).__init__(z_lens, z_source_ref)
        self.z_lens = jnp.array([z_lens])
        self.z_source_ref = jnp.array([z_source_ref])

    def efunc(self, z, H0, Om0, k, w0):
        """
        dimensionless Friedmann equation
        """
        matter = Om0 * (1 + z) ** 3
        Or0 = self.omega_rad0(H0)
        relativistic = Or0 * (1 + z)**4
        Ok0 = - k / H0 ** 2
        curvature = Ok0 * (1 + z)**2
        Ode0 = (1.0 - Om0 - Or0 - Ok0)
        dark_energy = Ode0 * (1 + z) ** (3 * (1 + w0))

        E = jnp.sqrt(matter + relativistic + dark_energy + curvature)
        return E

    @staticmethod
    def _integrate(func, z_min, z_max, n_grid=1000):
        z = jnp.linspace(z_min, z_max, n_grid)
        f = func(z)
        integrated = jnp.trapezoid(f, z, axis=0)
        return integrated

class w0waCDM_Cosmo(CosmoBase):
    _name = "w0waCDM"
    _params = ['H0', 'Om0', 'k', 'w0', 'wa']

    def __init__(self, z_lens, z_source_ref):  # z_source_ref required (no silent default)
        super(w0waCDM_Cosmo, self).__init__(z_lens, z_source_ref)
        self.z_lens = jnp.array([z_lens])
        self.z_source_ref = jnp.array([z_source_ref])

    def efunc(self, z, H0, Om0, k, w0, wa):
        """
        dimensionless Friedmann equation
        """
        matter = Om0 * (1 + z) ** 3
        Or0 = self.omega_rad0(H0)
        relativistic = Or0 * (1 + z)**4
        Ok0 = - k / H0 ** 2
        curvature = Ok0 * (1 + z)**2
        Ode0 = (1.0 - Om0 - Or0 - Ok0)
        # w_de = self.dark_energy_eos(z, w0, wa)
        # dark_energy = Ode0 * (1 + z) ** (3 * (1 + w_de))
        # this is the same as above but with the astropy cosmology implementation
        dark_energy = Ode0 * (1 + z) ** (3 * (1 + w0 + wa)) * jnp.exp(-3 * wa * z / (1 + z))

        E = jnp.sqrt(matter + relativistic + dark_energy + curvature)
        return E

    @staticmethod
    def _integrate(func, z_min, z_max, n_grid=1000):
        z = jnp.linspace(z_min, z_max, n_grid)
        f = func(z)
        integrated = jnp.trapezoid(f, z, axis=0)
        return integrated
