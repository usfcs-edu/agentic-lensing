from abc import ABC, abstractmethod
from gigalens.profile import Parameterized

class CosmoBase(Parameterized, ABC):

    Neff = 3.04  # number of relativistic species
    c = 299792.458  # km/s #speed of light
    
    def __init__(self, z_lens, z_source_ref):
        # z_source_ref is REQUIRED (no silent default): it is the source redshift at
        # which the mass deflections are normalized, i.e. where deflection_ratio == 1.
        # Mass parameters (theta_E, alpha_Rs, shear, ...) are only meaningful relative
        # to this reference, so it must be stated explicitly to match the convention the
        # mass parameters were derived under.
        super(CosmoBase, self).__init__()
        # Domain guard (misuse_register C1, construction-time): both redshifts are
        # concrete at construction, so an invalid geometry raises here rather than
        # surfacing later as a negative/NaN deflection_ratio.
        import math
        zl, zr = float(z_lens), float(z_source_ref)
        if not (math.isfinite(zl) and math.isfinite(zr)):
            raise ValueError(
                f"z_lens={z_lens} and z_source_ref={z_source_ref} must be finite.")
        if zl <= 0:
            raise ValueError(
                f"z_lens={z_lens} must be > 0: a lens at or behind the observer has no "
                "lensing geometry (misuse_register C1).")
        if zr <= zl:
            raise ValueError(
                f"z_source_ref={z_source_ref} must be > z_lens={z_lens}: the reference "
                "source plane defines deflection_ratio == 1 and must lie behind the "
                "lens (misuse_register C1).")
        self.z_lens = z_lens
        self.z_source_ref = z_source_ref

    def _validate_source_redshift(self, z_source):
        """Domain guard (misuse_register C1, first-use): a source in front of the lens
        or at z <= 0 cannot be lensed — raise instead of returning a finite negative
        ``deflection_ratio``. ``z_source == z_lens`` is allowed (ratio is exactly 0).
        Traced values are skipped, and under jit EVERY argument is a tracer — including
        a plain-number redshift the user fixed — so this guard only covers eager calls.
        The jitted scene path is guarded at construction instead
        (``LensModel._validate_concrete_redshifts``); genuinely sampled redshifts are
        the prior's responsibility.
        """
        import numpy as np
        try:
            z = np.asarray(z_source, dtype=float)
        except Exception:
            return
        z_lens = float(np.asarray(self.z_lens, dtype=float).reshape(-1)[0])
        if np.any(~np.isfinite(z)) or np.any(z <= 0):
            raise ValueError(
                f"z_source={z_source} must be finite and > 0 (misuse_register C1).")
        if np.any(z < z_lens):
            raise ValueError(
                f"z_source={z_source} is in front of the lens (z_lens={z_lens}): such a "
                "source is not lensed, so its deflection_ratio is undefined "
                "(misuse_register C1).")

    def _flag_unphysical_densities(self, H0, **kwargs):
        """Physicality flag (misuse_register C3): a finite-but-unphysical density
        composition (Om0 < 0, or Om0/k large enough that Ode0 = 1 - Om0 - Or0 - Ok0 < 0)
        can still yield plausible finite distances with no signal to the user. Per the
        resolved policy this FLAGS (``warnings.warn``), never raises, because the
        computation itself may stay valid. Traced values are skipped, and under jit
        every param is a tracer, so this only fires on eager calls; the jitted scene
        path checks fixed cosmo constants at ``LensModel`` construction instead, and
        sampled params are the prior's responsibility. Note Om0=1.0 flat (EdS) warns:
        with the radiation term, Ode0 = -Or0 ≈ -8.5e-5 < 0 — consistent with efunc's
        own parameterization, where that configuration IS slightly over-closed.
        """
        import numpy as np
        import warnings
        try:
            H0a = np.asarray(H0, dtype=float)
            Om0 = np.asarray(kwargs.get("Om0", 0.0), dtype=float)
            k = np.asarray(kwargs.get("k", 0.0), dtype=float)
        except Exception:
            return
        Or0 = self.omega_rad0(H0a)
        Ok0 = -k / H0a ** 2
        Ode0 = 1.0 - Om0 - Or0 - Ok0
        if np.any(Om0 < 0):
            warnings.warn(
                f"unphysical cosmology: Om0={Om0} < 0. Distances remain finite but are "
                "physically meaningless (misuse_register C3).", UserWarning, stacklevel=3)
        elif np.any(Ode0 < 0):
            warnings.warn(
                f"unphysical cosmology: Ode0={Ode0} < 0 (Om0={Om0}, k={k}). Distances "
                "remain finite but are physically meaningless (misuse_register C3).",
                UserWarning, stacklevel=3)

    @abstractmethod
    def efunc(self, z, H0, Om0, k, w0):
        pass

    def omega_rad0(self, H0):
        h = H0 / 100
        return 2.469e-5 * h ** -2.0 * (1.0 + 0.2271 * self.Neff)

    def comoving_distance_z1z2(self, z1, z2, H0, **kwargs):
        import jax.numpy as jnp
        self._flag_unphysical_densities(H0, **kwargs)
        # Batching convention: the integration endpoints (redshifts) and the cosmological
        # parameters SHARE ONE batch axis -- the sample axis. During MAP/HMC each cosmo
        # leaf has shape (batch,), and a SAMPLED source/plane redshift has that same
        # (batch,); a fixed redshift or fixed param is scalar and broadcasts into it.
        # ``_integrate`` lays the 1000-point quadrature grid on axis 0, so broadcast the
        # endpoints to the common batch first -> the grid is (n_grid, *batch) and ``efunc``
        # combines z and params element-wise over *batch*. This replaces the old
        # "give params their own trailing axes" reshape, which assumed SCALAR endpoints and
        # produced an (n_grid, batch, batch) OUTER PRODUCT once the redshift was also
        # sampled (both a redshift and cosmo params batched at once). Scalar endpoints +
        # (batch,) params reduce to the same (n_grid, batch) grid as before, so the
        # scalar-redshift result is unchanged. A redshift and params whose batch shapes are
        # not broadcast-compatible now raise here (a genuine mis-shape), which is the
        # intended contract.
        batch = jnp.broadcast_shapes(jnp.shape(z1), jnp.shape(z2), jnp.shape(H0),
                                     *[jnp.shape(v) for v in kwargs.values()])
        z1b = jnp.broadcast_to(z1, batch)
        z2b = jnp.broadcast_to(z2, batch)

        def integrand(z):
            return 1.0 / self.efunc(z, H0, **kwargs)
        return (self.c / H0) * self._integrate(integrand, z1b, z2b)

    def comoving_distance(self, z, **kwargs):
        return self.comoving_distance_z1z2(0, z, **kwargs)

    def _omega_k0(self, **kwargs):
        """Present-day curvature density Omega_k0, matching efunc's curvature term.

        efunc adds ``Ok0 * (1+z)**2`` with ``Ok0 = -k / H0**2`` (`jax/cosmo.py`), so
        the curvature density is ``Omega_k0 = -k / H0**2``; it is 0 for a flat model.
        """
        return -kwargs.get("k", 0.0) / (kwargs["H0"] ** 2)

    def _transverse_comoving_distance(self, z, **kwargs):
        """Transverse (proper-motion) comoving distance D_M(0, z), curvature-aware.

        D_M = D_H * S_k(D_C / D_H), where S_k is sinh / identity / sin for an open /
        flat / closed universe (Hogg 2000, eq. 16). Reduces *exactly* to the
        line-of-sight comoving distance D_C when Omega_k0 = 0, so flat results are
        unchanged. ``sinh(u)/u`` and ``sin(u)/u`` are evaluated with their u->0 limit
        (= 1) via a Taylor branch so there is no 0/0 at z=0 or in the flat limit.
        """
        import jax.numpy as jnp

        Dc = self.comoving_distance(z, **kwargs)
        Dh = self.c / kwargs["H0"]
        Ok = self._omega_k0(**kwargs)
        # Gradient-safe curvature term. sqrt(|Ok|) has an INFINITE derivative at Ok=0, so
        # for a flat (or near-flat) model, differentiating Dc w.r.t. a cosmo param (e.g.
        # sampling H0 in MAP/HMC) produces inf*0 -> NaN. The S_k branch (sinh/sin) is also
        # only selected for a curved model, but ``where`` still backprops through the
        # unused branch, so a NaN/overflow there contaminates the (selected) flat result.
        # Guard BOTH: clamp |Ok| away from 0 under the sqrt, and zero ``u`` outside the
        # curved branch. The flat branch returns factor=1 exactly and curved values are
        # untouched, so the forward result is unchanged; only the masked branch's value
        # and gradient are made finite.
        absOk = jnp.abs(Ok)
        curved = absOk > 1e-12
        u = jnp.where(curved, jnp.sqrt(jnp.where(curved, absOk, 1.0)) * (Dc / Dh), 0.0)
        small = jnp.abs(u) < 1e-6
        u_nz = jnp.where(small, 1.0, u)
        sinhc = jnp.where(small, 1.0 + u**2 / 6.0, jnp.sinh(u) / u_nz)
        sinc = jnp.where(small, 1.0 - u**2 / 6.0, jnp.sin(u) / u_nz)
        factor = jnp.where(Ok > 1e-12, sinhc, jnp.where(Ok < -1e-12, sinc, 1.0))
        return Dc * factor

    def _transverse_comoving_term_z1z2(self, z1, z2, **kwargs):
        """Transverse comoving distance D_M(z1, z2) (a.k.a. lenstronomy ``T_xy``).

        Hogg 2000, eq. 19 numerator: the comoving distance between two redshifts in a
        universe of arbitrary curvature. Reduces to ``D_C(z2) - D_C(z1)`` when flat.
        ``D_A(z1,z2) = this / (1 + z2)``.
        """
        import jax.numpy as jnp

        Dh = self.c / kwargs["H0"]
        Ok = self._omega_k0(**kwargs)
        Dm1 = self._transverse_comoving_distance(z1, **kwargs)
        Dm2 = self._transverse_comoving_distance(z2, **kwargs)
        return (Dm2 * jnp.sqrt(1.0 + Ok * (Dm1 / Dh) ** 2)
                - Dm1 * jnp.sqrt(1.0 + Ok * (Dm2 / Dh) ** 2))

    def transverse_comoving_distance_z1z2(self, z1, z2, **kwargs):
        """Public alias for the transverse comoving distance D_M(z1, z2)."""
        return self._transverse_comoving_term_z1z2(z1, z2, **kwargs)

    def angular_distance(self, z, **kwargs):
        return self._transverse_comoving_distance(z, **kwargs) / (1 + z)

    def angular_distance_z1z2(self, z1, z2, **kwargs):
        return self._transverse_comoving_term_z1z2(z1, z2, **kwargs) / (1 + z2)

    def luminosity_distance(self, z, **kwargs):
        # D_L = (1+z) D_M (curvature-aware); = (1+z) D_C when flat.
        return (1 + z) * self._transverse_comoving_distance(z, **kwargs)

    def distance_matrix(self, redshifts, **kwargs):
        """Lower-triangular transverse-comoving-distance matrix for multi-plane tracing.

        Returns ``D`` of shape ``(N+1, N+1)`` where row/col 0 is the observer (z=0) and
        ``D[j, i] = D_M(z_i, z_j)`` for ``j > i`` (0 on/above the diagonal) — the same
        layout as the experimental ``_build_D_matrix`` / lenstronomy ``Background.T_xy``,
        but built from gigalens' own (differentiable) cosmology so plane redshifts and
        cosmological parameters can be sampled. ``redshifts`` is the list of plane
        redshifts (observer's z=0 is prepended here).
        """
        import jax.numpy as jnp

        zs = [0.0] + list(redshifts)
        n = len(zs)
        # With a batch of cosmo params the non-zero entries are (*batch,), so the
        # upper-triangular zero fill must match that shape (mixing () with (*batch,)
        # would make jnp.stack raise). The batch axes end up TRAILING on the stacked
        # result, e.g. (N+1, N+1, *batch); the multiplane trace consumes it via
        # ``D[k+1, k] -> (*batch,)``. Scalar params give batch_shape () -> unchanged.
        batch_shape = (jnp.broadcast_shapes(*[jnp.shape(v) for v in kwargs.values()])
                       if kwargs else ())
        zero = jnp.zeros(batch_shape)
        rows = []
        for j in range(n):
            row = []
            for i in range(n):
                if j > i:
                    row.append(self.transverse_comoving_distance_z1z2(zs[i], zs[j], **kwargs))
                else:
                    row.append(zero)
            rows.append(jnp.stack(row))
        return jnp.stack(rows)

    def lensing_distance(self, z_source, **kwargs):
        self._validate_source_redshift(z_source)
        d_ls = self.angular_distance_z1z2(self.z_lens, z_source, **kwargs)
        d_s = self.angular_distance(z_source, **kwargs)
        return d_ls / d_s

    def deflection_ratio(self, z_source, **kwargs):
        d_lensing = self.lensing_distance(z_source, **kwargs)
        d_ref = self.lensing_distance(self.z_source_ref, **kwargs)
        return d_lensing / d_ref

    @staticmethod
    @abstractmethod
    def _integrate(func, z_min, z_max, n_grid=1000):
        pass