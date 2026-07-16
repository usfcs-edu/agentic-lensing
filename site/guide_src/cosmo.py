"""Cosmology helpers — deliberately small, because the repo's use of it is small.

An audit of every campaign in this repo found that the lens-modeling likelihood
contains NO cosmology at all: it is entirely angular (arcsec in, arcsec out).
Cosmology enters in exactly three places, and always as FlatLambdaCDM(70, 0.3):

  1. reproductions/hsu-2025/07_classify_einstein_dimple.py  — theta_E from sigma_v
     (and reproductions/lensjudge/tools/spectrum.py, the same check as an agent tool)
  2. reproductions/sheu-2024b/04_setup_multiplane.py        — Sigma_crit, M(<theta_E)
  3. reproductions/sheu-2023/05_lightcurve_salt3.py         — SN Ia distance modulus

So this module is 60 lines, and the guide's cosmology part is three chapters
rather than ten. That proportion is a finding, not an omission: it tells you
where to spend your attention.

The one thing that bites everyone: angular diameter distances DO NOT ADD.
D_ds != D_s - D_d. Use ``d_ds`` (astropy's angular_diameter_distance_z1z2),
never a subtraction. For z_l=0.5, z_s=2.0 the difference is 1097 vs 468 Mpc —
a factor of 2.3, silent, and it lands straight in Sigma_crit.
"""
from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM

# The repo-wide choice. Every campaign uses exactly this.
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


def d_a(z):
    """Angular diameter distance to z, in Mpc."""
    return COSMO.angular_diameter_distance(z).to_value(u.Mpc)


def d_ds(z_l, z_s):
    """Angular diameter distance BETWEEN two redshifts, in Mpc.

    NOT d_a(z_s) - d_a(z_l). See the module docstring.
    """
    return COSMO.angular_diameter_distance_z1z2(z_l, z_s).to_value(u.Mpc)


def sigma_crit(z_l, z_s):
    """Critical surface density, in Msun/Mpc^2.

        Sigma_cr = c^2 / (4 pi G) * D_s / (D_d D_ds)

    The conversion factor between "surface mass density" and the dimensionless
    convergence kappa: kappa = Sigma / Sigma_cr. kappa = 1 is the threshold for
    strong lensing, which is what makes Sigma_cr the natural unit.

    Reproduces reproductions/sheu-2024b/04_setup_multiplane.py:128.
    """
    Dd, Ds, Dds = d_a(z_l) * u.Mpc, d_a(z_s) * u.Mpc, d_ds(z_l, z_s) * u.Mpc
    val = (const.c**2 / (4 * np.pi * const.G) * Ds / (Dd * Dds))
    return val.to_value(u.Msun / u.Mpc**2)


def arcsec_to_mpc(z_l):
    """Physical transverse Mpc subtended by 1 arcsec at redshift z_l."""
    return (d_a(z_l) * u.Mpc * (1 * u.arcsec).to_value(u.rad)).to_value(u.Mpc)


def mass_within_theta_e(theta_e_arcsec, z_l, z_s):
    """M(<theta_E) = Sigma_cr * pi * (D_d theta_E)^2, in Msun.

    The cleanest measurement in strong lensing: the mass inside the Einstein
    radius follows from the geometry alone, with no dependence on the density
    profile. It is why theta_E is quoted in every lensing paper.

    Reproduces sheu-2024b's 4.62e13 Msun for the Carousel cluster.
    """
    R_E = arcsec_to_mpc(z_l) * theta_e_arcsec
    return sigma_crit(z_l, z_s) * np.pi * R_E**2


def theta_e_from_sigma_v(sigma_v_kms, z_l, z_s):
    """theta_E in arcsec for an SIS lens (Hsu+2025 Eq. 1).

    Reproduces reproductions/hsu-2025/07_classify_einstein_dimple.py:107.
    """
    from lensing import theta_e_sis

    return theta_e_sis(sigma_v_kms, d_ds(z_l, z_s) / d_a(z_s))
