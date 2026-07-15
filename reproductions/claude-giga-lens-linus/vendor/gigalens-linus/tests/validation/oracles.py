"""Adapters to the reference oracles (lenstronomy, astropy).

Keeping the oracle calls in one place means a test reads as "gigalens vs oracle" and
the mapping (parameter names, conventions, multiplane setup) is reviewed once, here,
rather than re-derived in every test. If a convention mismatch is found, it is fixed
in this file.
"""
from __future__ import annotations

import numpy as np


# --- single-plane mass deflection -------------------------------------------------
def lenstronomy_epl_deriv(x, y, **kw):
    from lenstronomy.LensModel.Profiles.epl import EPL
    return EPL().derivatives(x=x, y=y, **kw)


# --- multiplane ray-shooting ------------------------------------------------------
def lenstronomy_multiplane_ray_shooting(
    theta_x, theta_y, lens_model_list, kwargs_list, redshift_list, z_source, astropy_cosmo
):
    """Source-plane (β_x, β_y) for a recursive multi-plane lens, via lenstronomy.

    This is the oracle for gigalens' ``MultiPlaneTrace`` final-plane output. Uses
    lenstronomy's own multi-plane machinery and astropy cosmology so the *only* thing
    under test is gigalens' recursion + distance matrix.

    Parameters mirror lenstronomy's ``LensModel`` multi-plane API:
      lens_model_list : list[str]   e.g. ["EPL", "SIS"]
      kwargs_list     : list[dict]  per-deflector kwargs
      redshift_list   : list[float] per-deflector redshift
      z_source        : float
    """
    from lenstronomy.LensModel.lens_model import LensModel

    lm = LensModel(
        lens_model_list=lens_model_list,
        multi_plane=True,
        lens_redshift_list=list(redshift_list),
        z_source=z_source,
        cosmo=astropy_cosmo,
    )
    bx, by = lm.ray_shooting(theta_x, theta_y, kwargs_list)
    return np.asarray(bx), np.asarray(by)


# --- cosmological distances -------------------------------------------------------
def astropy_angular_diameter_distance(astropy_cosmo, z):
    """D_A(0, z) in Mpc."""
    return float(astropy_cosmo.angular_diameter_distance(z).value)


def astropy_angular_diameter_distance_z1z2(astropy_cosmo, z1, z2):
    """D_A(z1, z2) in Mpc (handles curvature; flat for FlatwCDM)."""
    return float(astropy_cosmo.angular_diameter_distance_z1z2(z1, z2).value)


def astropy_comoving_distance(astropy_cosmo, z):
    """Line-of-sight comoving distance D_C(z) in Mpc."""
    return float(astropy_cosmo.comoving_distance(z).value)
