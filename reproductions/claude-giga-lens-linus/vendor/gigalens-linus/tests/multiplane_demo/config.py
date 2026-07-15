"""Single source of truth for the 3-lens-plane demo system.

A deliberately physical multi-plane strong lens used to validate the gigalens
forward model + (later) inference recovery against lenstronomy. Edit numbers HERE
only; the oracle build, the gigalens scene, and any recovery test all read this.

Stack (increasing redshift):
  z1 : EPL mass only            -- foreground deflector, no light
  z2 : EPL mass + Sersic lens light + Sersic source light (lensed by z1 only)
  z3 : Sersic source light      -- lensed by z1 + z2

Conventions (load-bearing -- see tests/multiplane_demo/README.md):
  * EPL theta_E is referenced to the FINAL source plane z3 (one value per deflector
    for the whole system; matches gigalens' multi-plane convention, R1-validated).
  * Sersic 'Ie' == lenstronomy 'amp' == surface brightness at R_sersic.
  * Cosmology: flat wCDM, radiation-matched FlatwCDM for the astropy/lenstronomy side
    so cosmology is out of the error budget.
"""

# ---- cosmology (fixed for both simulation and modelling) ----------------------
COSMO = dict(H0=70.0, Om0=0.3, w0=-1.0)          # flat wCDM
TCMB0, NEFF = 2.7255, 3.046                      # radiation-matched for astropy side

# ---- redshifts ----------------------------------------------------------------
Z1, Z2, Z3 = 0.3, 0.6, 2.0

# ---- mass: EPL deflectors (theta_E referenced to z3) --------------------------
# z1 is offset from the z2 lens-galaxy core so its inner (z1-lensed lens-light) ring is
# NOT concentric with the outer (z3) ring -- a more realistic non-aligned foreground.
EPL_Z1 = dict(theta_E=1.0, gamma=2.00, e1=0.04, e2=-0.03, center_x=0.30, center_y=0.25)
EPL_Z2 = dict(theta_E=0.2, gamma=2.05, e1=0.05, e2=0.04,  center_x=0.05, center_y=-0.05)

# ---- light: Sersic profiles ---------------------------------------------------
# z2 lens light: bright de Vaucouleurs co-centred with the z2 mass.
# (A z2 *source* was removed to simplify the system: plane 2 now carries mass + lens
#  light only. Consequence: the z1 plane has no dedicated lensed feature; it is
#  constrained only via its foreground deflection of the z3 ray path + mild lensing of
#  this lens light, so z1's mass is more weakly identified -- revisit at recovery.)
LENS_LIGHT_Z2 = dict(Ie=4.0,  R_sersic=1.00, n_sersic=4.0, e1=0.05,  e2=0.04,
                     center_x=0.05, center_y=-0.05)
# z3 source: compact, strongly lensed by z2 (theta_E=1.2") into a clear Einstein ring.
SRC_LIGHT_Z3  = dict(Ie=6.0,  R_sersic=0.15, n_sersic=1.0, e1=0.10,  e2=-0.08,
                     center_x=0.05, center_y=0.08)

# ---- observation --------------------------------------------------------------
DELTA_PIX = 0.05      # arcsec / pixel  (HST/ACS-like)
NUM_PIX = 80          # 4" x 4" field of view
SUPERSAMPLE = 3       # sub-pixel rendering before pooling
PSF_FWHM = 0.10       # arcsec, Gaussian PSF
PSF_HALF_PIX = 10     # native kernel half-width (-> 21x21)
