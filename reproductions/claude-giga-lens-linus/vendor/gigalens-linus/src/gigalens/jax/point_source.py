r"""Lensed point-source modelling for the **new** GIGA-Lens scene API.

This module slots the lensed point-source pipeline (Baltazar, Ratier-Werbin, Huang et al.
2026 — "A Novel Lensed Point Source Modeling Pipeline using GIGA-Lens") into the new
scene stack as a first-class observation type, via the dataset seam of
:mod:`gigalens.jax.scene_prob_model`:

* :class:`PointSourceData` subclasses :class:`~gigalens.jax.scene_prob_model.Dataset` — it
  holds the observed image positions / fluxes / time delays and the hand-tuned loss
  weights, and ``sees`` the :class:`~gigalens.jax.profiles.light.point_source.PointSource`
  light Component that carries the intrinsic flux ``amp`` (the source POSITION is not a
  parameter — it is derived as the mean of the delensed image positions, see below).
* :class:`PointSourceLikelihoodTerm` subclasses
  :class:`~gigalens.jax.scene_prob_model.LikelihoodTerm` — its ``log_like(params)``
  computes the three weighted terms ``L_D + L_F + L_TD`` (source-plane compactness,
  inverse-magnification flux, relative time delay) reading the structured scene params
  (``params["planes"][i]["mass"][j]``, the source ``PointSource`` params, and
  ``params["cosmo"]``).

Everything downstream (``ModellingSequence`` MAP/SVI/HMC/MCLMC) runs unchanged through the
single ``prob_model.log_prob(z)`` seam.

**Cosmology — two paths, chosen automatically.** The time-delay distance
``D_dt = (1+z_l)·D_d·D_s/D_ds`` is the same formula either way; only the comoving-distance
integrator differs:

* **Fixed flat LambdaCDM** (Om0 fixed, k==0, w0==-1, no free ``wa``): the comoving
  distances are precomputed with astropy (``FlatLambdaCDM``) — the SAME integrator as the
  reference notebook — so the time delay is **bit-faithful** to it. Only ``H0`` remains
  live, through the ``c/H0`` prefactor (still differentiable in H0).
* **Sampled cosmology** (``Om0``/``w0``/``k`` freed) or a non-flat/non-LambdaCDM fixed
  cosmology: the fully differentiable :class:`gigalens.jax.cosmo.wCDM_Cosmo` (trapezoid)
  is used, so the freed cosmo params flow through with exact gradients (differs from
  astropy at ~1e-5). This is the genuine cosmology-inference upgrade over the old API.

The path is selected at construction from the scene's cosmo priors and reported on the
term as ``cosmology_mode``.

**Design constraints (Phase 1).**

* Deflection is computed directly from the scene's **mass Components** (their ``.deriv``),
  NOT via a ``SceneSimulator`` — so the point-source term needs no imaging grid /
  ``sim_config``. Magnification ``det A`` is the forward-mode-AD Hessian of that total
  deflection (paper §2.2.2), carried over verbatim from the notebook.
* Exactly one mass plane, and the source plane's cosmological reference must be the source
  itself (``cosmo.z_source_ref == z_source``) so ``deflection_ratio == 1`` and the raw
  deflection / standard single-source time-delay formula apply unchanged. Both are checked
  at construction. Multi-plane point sources and ``deflection_ratio != 1`` are deferred.
* The hand-tuned weights are kept (documented as uncalibrated): the returned ``chi2`` is
  the weighted loss as a MONOTONE STAND-IN, so posterior widths are NOT physical
  measurement uncertainties. Recasting the weights as inverse-variances (a real reduced
  chi²) is the future step that would also make this term commensurable with a Gaussian
  imaging term and unlock joint fitting.
"""

import warnings

import jax
import numpy as np
from jax import numpy as jnp

from gigalens.jax.scene_prob_model import Dataset, LikelihoodTerm

# Physical constants (identical to the notebook / old-API port).
_C_M_S = 2.9979246e8               # speed of light, m/s
_MPC_M = 3.0856775800000002e22     # Mpc -> m
_ARCSEC_RAD = 4.84813681109536e-06  # arcsec -> rad
_SEC_PER_DAY = 86400.0


# ---------------------------------------------------------------------------
# Physics (ported from the notebook; reads the scene's mass Components).
# ``mass_profiles`` is a Python list of profile objects (static); ``mass_params`` is the
# dict ``params["planes"][lens_i]["mass"]`` with integer keys 0..len-1 aligned to it.
# ---------------------------------------------------------------------------
def _total_deflection(mass_profiles, mass_params, x, y):
    """Total deflection ``(alpha_x, alpha_y) = sum_j profile_j.deriv(...)`` at ``(x, y)``.

    Assumes ``deflection_ratio == 1`` (single source plane at the cosmo reference), so the
    raw sum is the deflection to the source plane — checked at term construction.
    """
    ax = 0.0
    ay = 0.0
    for j, prof in enumerate(mass_profiles):
        dfx, dfy = prof.deriv(x, y, **mass_params[j])
        ax = ax + dfx
        ay = ay + dfy
    return ax, ay


def _magnification(mass_profiles, mass_params, x_img, y_img):
    """Return ``det A`` (the *inverse* magnification) at each image position.

    Forward-mode AD Hessian of the total deflection (paper §2.2.2 prefers it near the
    critical curve). ``x_img``/``y_img`` are 1-D ``(n_images,)``; the cosmological/mass
    parameter batch ``B`` lives inside ``mass_params`` and rides through untouched, so the
    result is ``(n_images,) + B``.
    """
    def defl(xx, yy):
        return _total_deflection(mass_profiles, mass_params, xx, yy)

    def per_image(xi, yi):
        (f_xx, f_xy), (f_yx, f_yy) = jax.jacfwd(defl, argnums=(0, 1))(xi, yi)
        return (1 - f_xx) * (1 - f_yy) - f_xy * f_yx

    return jax.vmap(per_image)(x_img, y_img)


def _lensing_potential(mass_profiles, mass_params, x, y):
    r"""Lensing potential ``psi`` = sum of each mass profile's ``potential`` at ``(x, y)``.

    Delegates to each profile's own ``potential`` method (:meth:`gigalens.jax.profile.\
    MassProfile.potential`) rather than dispatching on parameter names. A profile that
    does not implement ``potential`` raises there, loudly and BY NAME — so a profile that
    merely shares a parameter (e.g. a non-EPL profile that happens to have ``theta_E``, or
    a non-shear profile with ``gamma1``) is never silently mis-handled by key-sniffing.
    """
    psi = 0.0
    for j, prof in enumerate(mass_profiles):
        psi = psi + prof.potential(x, y, **mass_params[j])
    return psi


def _fermat_potential(mass_profiles, mass_params, x, y, src_x, src_y):
    r"""Fermat potential ``Phi = 0.5|theta - beta|^2 - psi`` (no line-of-sight terms)."""
    geometry = ((x - src_x) ** 2 + (y - src_y) ** 2) / 2.0
    return geometry - _lensing_potential(mass_profiles, mass_params, x, y)


def _time_delay_days(fermat_pot, cosmo_profile, cosmo_params, z_lens, z_source):
    r"""Time delay (days) from the Fermat potential and the differentiable cosmology.

    ``D_dt = (1+z_l) D_d D_s / D_ds`` with ``D_d = D_A(z_l)``, ``D_s = D_A(z_s)``,
    ``D_ds = D_A(z_l, z_s)`` from :class:`gigalens.jax.cosmo` (Mpc). Then
    ``dt = D_dt/c * Phi`` with the arcsec^2 -> rad^2 and s -> day conversions. H0 (and any
    freed Om0/w0) enter with exact gradients.
    """
    d_d = cosmo_profile.angular_distance(z_lens, **cosmo_params)
    d_s = cosmo_profile.angular_distance(z_source, **cosmo_params)
    d_ds = cosmo_profile.angular_distance_z1z2(z_lens, z_source, **cosmo_params)
    d_dt_mpc = (1.0 + z_lens) * d_d * d_s / d_ds
    return (d_dt_mpc * _MPC_M) / _C_M_S * fermat_pot * _ARCSEC_RAD ** 2 / _SEC_PER_DAY


def precompute_comoving_distances(z_lens, z_source, Om0):
    """Fixed-Omega_m dimensionless comoving-distance integrals int_0^z dz'/E(z').

    Uses astropy ``FlatLambdaCDM._integral_comoving_distance_z1z2_scalar`` (quad) — the
    SAME integrator as the reference notebook, so a fixed-Omega_m time delay is bit-faithful
    to it (the differentiable trapezoid cosmology of :func:`_time_delay_days` differs at
    ~1e-5). H0-free (these integrals do not depend on H0); ``Ob0=0`` matches the notebook.
    """
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=Om0, Ob0=0.0)
    cl = float(cosmo._integral_comoving_distance_z1z2_scalar(0, z_lens))
    cs = float(cosmo._integral_comoving_distance_z1z2_scalar(0, z_source))
    return cl, cs


def _time_delay_days_fixed(fermat_pot, H0, comoving_lens, comoving_source, z_lens, z_source):
    r"""Time delay (days) at FIXED (flat LambdaCDM) cosmology — the notebook formula verbatim
    (``JAX-H0-SN_demo.ipynb`` cell 12).

    ``comoving_lens``/``comoving_source`` are the precomputed fixed-Omega_m integrals from
    :func:`precompute_comoving_distances`. ``H0`` is live (``params["cosmo"]["H0"]``) and
    enters only through the ``c/H0`` prefactor, so the time delay stays differentiable in
    H0 while the comoving integral is fixed. Byte-identical to the reference for fixed Om0.
    """
    dd = (_C_M_S / H0) * (1 / 1000) * comoving_lens / (1 + z_lens)
    ds = (_C_M_S / H0) * (1 / 1000) * comoving_source / (1 + z_source)
    dds = ((_C_M_S / H0) * (1 / 1000) * comoving_source
           - (_C_M_S / H0) * (1 / 1000) * comoving_lens) / (1 + z_source)
    ddt = (1 + z_lens) * dd * ds / dds * _MPC_M  # assumes kappa_ext = 0
    return ddt / _C_M_S * fermat_pot / _SEC_PER_DAY * _ARCSEC_RAD ** 2


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------
def _as_fixed_float(value, what):
    """Return ``float(value)`` or raise if it is not a concrete scalar (Phase 1 fixes
    redshifts; a sampled redshift is a deferred feature, not a silent no-op).

    Accepts a size-1 array (``wCDM_Cosmo`` stores ``z_lens`` / ``z_source_ref`` as
    ``jnp.array([...])``); anything with more than one element (a genuinely
    batched/sampled redshift) raises.
    """
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        arr = None
    if arr is None or arr.size != 1 or not np.isfinite(arr[0]):
        raise ValueError(
            f"{what} must be a fixed, finite number for the point-source term "
            f"(got {value!r}). Sampling redshifts with the point-source likelihood is not "
            "supported yet — fix the plane redshifts / cosmo reference in the scene.")
    return float(arr[0])


def _fixed_cosmo_value(v):
    """Return ``float(v)`` if a scene-prior entry is a FIXED constant, else ``None``.

    A ``tfd.Distribution`` (free) or a ``shared(...)`` handle (linked/free) is not fixed;
    a plain number is. Used to decide whether the cosmology is fully fixed (→ bit-faithful
    astropy time delay) or has a sampled parameter (→ differentiable trapezoid)."""
    import tensorflow_probability.substrates.jax as tfp
    from gigalens.jax.scene import SharedParam
    if isinstance(v, (tfp.distributions.Distribution, SharedParam)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PointSourceData(Dataset):
    r"""One lensed point-source observation: image positions, fluxes, and time delays.

    Args:
        source_component: the :class:`~gigalens.jax.profiles.light.point_source.PointSource`
            light :class:`~gigalens.jax.scene.Component` on the source plane whose ``amp``
            is the sampled intrinsic flux (the source position is derived, not a parameter).
            Matched by object identity against the model's light Components.
        x_img, y_img: observed image positions (arcsec), **sorted by arrival time**
            (earliest first). 1-D, length = number of images.
        flux_obs: observed inverse-magnification targets ``(1/mu_obs)^2`` per image
            (squared: image parity is unmeasurable), matching ``(det A / amp)^2``.
        td_obs: observed relative time delays (days) per image, with ``td_obs[0] == 0``.
        weight_dist, weight_flux, weight_time_delay: the hand-tuned loss weights
            (paper §2.2.4). Required — no defaults, no silent scientific choices.

    All arguments are required and validated finite at construction (the M3 "raise, never
    default" pattern); passing ``None`` raises.
    """

    # Marker read by ``ProbModel``'s mixing guard (avoids a scene_prob_model -> point_source
    # import cycle). A point-source term's weighted loss is not commensurable with a
    # Gaussian imaging likelihood, so the two may not currently share a ProbModel.
    is_pointsource = True

    def __init__(self, source_component, x_img, y_img, flux_obs, td_obs,
                 weight_dist, weight_flux, weight_time_delay):
        _required = dict(source_component=source_component, x_img=x_img, y_img=y_img,
                         flux_obs=flux_obs, td_obs=td_obs, weight_dist=weight_dist,
                         weight_flux=weight_flux, weight_time_delay=weight_time_delay)
        missing = [k for k, v in _required.items() if v is None]
        if missing:
            raise ValueError(
                "PointSourceData requires all of " + ", ".join(sorted(_required))
                + " (no silent scientific defaults). Missing: " + ", ".join(missing))

        x = np.asarray(x_img, dtype=float).reshape(-1)
        y = np.asarray(y_img, dtype=float).reshape(-1)
        f = np.asarray(flux_obs, dtype=float).reshape(-1)
        td = np.asarray(td_obs, dtype=float).reshape(-1)
        n = x.shape[0]
        if not (y.shape[0] == f.shape[0] == td.shape[0] == n):
            raise ValueError(
                "PointSourceData x_img/y_img/flux_obs/td_obs must have equal length "
                f"(got {x.shape[0]}, {y.shape[0]}, {f.shape[0]}, {td.shape[0]}).")
        if n < 2:
            raise ValueError(
                f"PointSourceData needs at least 2 images (got {n}); relative time delays "
                "and source-plane compactness are undefined for a single image.")
        for name, arr in (("x_img", x), ("y_img", y), ("flux_obs", f), ("td_obs", td)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"PointSourceData {name} has non-finite (NaN/inf) values (M3): the "
                    "likelihood must not be silently poisoned.")
        for name, w in (("weight_dist", weight_dist), ("weight_flux", weight_flux),
                        ("weight_time_delay", weight_time_delay)):
            wv = float(w)
            if not (np.isfinite(wv) and wv >= 0):
                raise ValueError(
                    f"PointSourceData {name} must be finite and non-negative (got {w!r}).")

        self.source_component = source_component
        self.x_img = x
        self.y_img = y
        self.flux_obs = f
        self.td_obs = td
        self.weight_dist = float(weight_dist)
        self.weight_flux = float(weight_flux)
        self.weight_time_delay = float(weight_time_delay)
        self.n_images = n
        # Three residual channels (compactness / flux / time delay), n_images each, as the
        # reduced-χ² normalizer SCALE. This is a stand-in denominator (the weighted loss is
        # not a real χ²), documented on the term.
        self.event_size = 3 * n
        # No imaging grid: the point-source term reads mass Components' .deriv directly.
        # ModellingSequence tolerates a sim_config-less dataset (see its __init__).
        self.sim_config = None

        warnings.warn(
            "PointSourceData uses HAND-TUNED loss weights (weight_dist/flux/time_delay): "
            "the resulting log_prob is an unnormalized penalized loss, NOT a calibrated "
            "likelihood, so posterior widths are NOT physical measurement uncertainties. "
            "Recast the weights as per-observable inverse-variances to make them so.",
            stacklevel=2)

    def resolve_sees(self, model):
        """The point source ``sees`` its own source-plane emitter Component (identity).

        The emitter must be a NON-RENDERING light Component (``renders=False``, i.e. a
        point source): a point-source dataset observes positions/fluxes/time delays, not
        pixels, so pointing it at a renderable (extended) light Component is a wiring bug."""
        all_light = model.light_components
        if id(self.source_component) not in {id(c) for c in all_light}:
            raise ValueError(
                "PointSourceData.source_component is not a light Component of this model "
                "(matched by object identity); pass the same PointSource instance used to "
                "build the LensModel.")
        if getattr(self.source_component.profile, "renders", True):
            raise ValueError(
                "PointSourceData.source_component is a RENDERING light Component "
                f"({self.source_component.profile}); a point-source dataset observes a "
                "non-rendering emitter (renders=False, e.g. PointSource), not extended "
                "light. Use a PointSource component, or model extended light with ImageData.")
        return [self.source_component]

    def build_term(self, model, seen, *, default_mode="lstsq"):
        return PointSourceLikelihoodTerm(self, model, seen)


# ---------------------------------------------------------------------------
# Likelihood term
# ---------------------------------------------------------------------------
class PointSourceLikelihoodTerm(LikelihoodTerm):
    r"""Three-term point-source loss as a scene :class:`LikelihoodTerm`.

    ``log_like(params)`` returns ``(-(w_D L_D + w_F L_F + w_TD L_TD), weighted)`` where:

    * ``L_D``  — source-plane compactness: mean Euclidean distance between consecutive
      delensed image positions (drives them to a common source), reference notebook cell 28.
    * ``L_F``  — flux: fit ``(det A / amp)^2`` to the observed ``(1/mu_obs)^2``.
    * ``L_TD`` — relative time delay vs the earliest-arriving image, using the source
      position derived as the mean of the delensed positions.

    ``reports_chi2 = True`` with ``chi2 = weighted`` (a MONOTONE STAND-IN, not a real χ²)
    so the ProbModel's ≥1-Gaussian-term requirement is satisfied; the reduced-χ² channel
    is therefore only a convergence monitor, not a goodness-of-fit.
    """

    reports_chi2 = True

    def __init__(self, dataset: PointSourceData, model, seen):
        self.dataset = dataset
        self.event_size = dataset.event_size

        # --- locate the single mass plane and its profiles ---------------------------
        mass_planes = [i for i, p in enumerate(model.planes) if p.has_mass]
        if len(mass_planes) != 1:
            raise ValueError(
                "PointSourceLikelihoodTerm supports exactly one mass plane "
                f"(found {len(mass_planes)}); multi-plane point sources are deferred.")
        self.lens_i = mass_planes[0]
        self.mass_profiles = [c.profile for c in model.planes[self.lens_i].mass]

        # --- locate the source-plane PointSource component (identity) ----------------
        src_component = dataset.source_component
        found = None
        for i, p in enumerate(model.planes):
            for j, c in enumerate(p.light):
                if c is src_component:
                    found = (i, j)
        if found is None:
            raise ValueError(
                "PointSourceLikelihoodTerm could not find the source PointSource component "
                "in the model's planes (identity match failed).")
        self.src_i, self.src_j = found

        # --- fixed redshifts + differentiable cosmology ------------------------------
        self.z_lens = _as_fixed_float(model.planes[self.lens_i].redshift, "lens-plane redshift")
        self.z_source = _as_fixed_float(model.planes[self.src_i].redshift, "source-plane redshift")
        if model.cosmo is None:
            raise ValueError(
                "PointSourceLikelihoodTerm requires a cosmology on the LensModel "
                "(LensModel(..., cosmo=Component(wCDM_Cosmo(z_lens, z_source_ref), {...}))) "
                "so the time-delay distance is defined.")
        self.cosmo_profile = model.cosmo.profile
        z_ref = _as_fixed_float(self.cosmo_profile.z_source_ref, "cosmo z_source_ref")
        # float32-appropriate tolerance: z_source_ref may be stored as float32 (~1e-7
        # relative error), while a source genuinely OFF the reference plane differs by
        # O(0.01+), so this cleanly separates the two.
        if not np.isclose(z_ref, self.z_source, rtol=1e-5, atol=1e-6):
            raise ValueError(
                "PointSourceLikelihoodTerm requires cosmo.z_source_ref == source-plane "
                f"redshift (got z_source_ref={z_ref}, z_source={self.z_source}); this makes "
                "deflection_ratio == 1 so the raw deflection and the standard single-source "
                "time-delay formula apply. A source off the cosmo reference plane is deferred.")

        # --- time-delay cosmology path ------------------------------------------------
        # When the cosmology is a FIXED flat LambdaCDM (Om0 fixed, k==0, w0==-1, no free
        # wa), use astropy-precomputed comoving distances — bit-faithful to the reference
        # notebook (same quad integrator). Only H0 remains live, via the c/H0 prefactor.
        # If any cosmo parameter is sampled (Om0/w/k inference) — or the fixed cosmology is
        # non-flat / non-LambdaCDM — fall back to the differentiable trapezoid cosmology.
        cp = model.cosmo.priors
        om0 = _fixed_cosmo_value(cp.get("Om0"))
        kk = _fixed_cosmo_value(cp.get("k"))
        w0 = _fixed_cosmo_value(cp.get("w0"))
        has_wa = "wa" in cp
        wa = _fixed_cosmo_value(cp.get("wa")) if has_wa else 0.0
        is_fixed_flat_lcdm = (
            om0 is not None and kk is not None and w0 is not None
            and kk == 0.0 and w0 == -1.0 and (not has_wa or wa == 0.0))
        if is_fixed_flat_lcdm:
            self._comoving = precompute_comoving_distances(self.z_lens, self.z_source, om0)
            self.cosmology_mode = "fixed-flat-lcdm-astropy"
        else:
            self._comoving = None
            self.cosmology_mode = "differentiable-trapezoid"

        # Static JAX arrays for the (fixed) observed data.
        self.x = jnp.asarray(dataset.x_img, dtype=jnp.float32)
        self.y = jnp.asarray(dataset.y_img, dtype=jnp.float32)
        self.flux_obs = jnp.asarray(dataset.flux_obs, dtype=jnp.float32)
        self.td_obs = jnp.asarray(dataset.td_obs, dtype=jnp.float32)
        self.w_dist = dataset.weight_dist
        self.w_flux = dataset.weight_flux
        self.w_td = dataset.weight_time_delay

    def _time_delay(self, fermat, cosmo_params):
        """Relative-arrival time delay via the construction-selected cosmology path."""
        if self._comoving is not None:
            cl, cs = self._comoving
            return _time_delay_days_fixed(
                fermat, cosmo_params["H0"], cl, cs, self.z_lens, self.z_source)
        return _time_delay_days(
            fermat, self.cosmo_profile, cosmo_params, self.z_lens, self.z_source)

    def log_like(self, params):
        mass_params = params["planes"][self.lens_i]["mass"]
        amp = params["planes"][self.src_i]["light"][self.src_j]["amp"]
        cosmo_params = params["cosmo"]

        # Batch ndim rides in the params (scalar for one MCMC position, (n,) for batched
        # MAP). Reshape the fixed image positions so the image axis leads and the param
        # batch broadcasts on the trailing axes -> every loss reduces over the image axis
        # back to the param batch shape.
        n = self.dataset.n_images
        bd = jnp.ndim(amp)
        xr = self.x.reshape((n,) + (1,) * bd)
        yr = self.y.reshape((n,) + (1,) * bd)
        flux_obs = self.flux_obs.reshape((n,) + (1,) * bd)
        td_obs = self.td_obs.reshape((n,) + (1,) * bd)

        # Delens each observed image to the source plane.
        ax, ay = _total_deflection(self.mass_profiles, mass_params, xr, yr)
        beta = jnp.stack([xr - ax, yr - ay], axis=0)   # (2, n_images, *batch)

        # L_D: source-plane compactness = mean over images of the Euclidean distance
        # between CONSECUTIVE (cyclically-shifted) delensed positions. Verbatim from the
        # reference notebook cell 28: it drives all images to a common source point without
        # referencing an explicit source parameter.
        shifted = jnp.concatenate([beta[:, -1:], beta[:, :-1]], axis=1)
        dist_loss = jnp.mean(jnp.sum((beta - shifted) ** 2, axis=0) ** 0.5, axis=0)

        # Source position: DERIVED as the mean of the delensed positions (NOT a sampled
        # parameter — faithful to the reference), used only for the time-delay Fermat term.
        source = jnp.mean(beta, axis=1)                 # (2, *batch)
        src_x, src_y = source[0], source[1]

        # L_F: fit (det A / amp)^2 to the observed inverse magnification.
        mag = _magnification(self.mass_profiles, mass_params, self.x, self.y)
        flux = mag ** 2 / amp ** 2
        flux_loss = jnp.mean((flux - flux_obs) ** 2, axis=0)

        # L_TD: relative time delay vs the earliest-arriving image (index 0).
        fermat = _fermat_potential(self.mass_profiles, mass_params, xr, yr, src_x, src_y)
        td = self._time_delay(fermat, cosmo_params)
        td = td - td[0:1]
        td_loss = jnp.mean((td - td_obs) ** 2, axis=0)

        weighted = (self.w_dist * dist_loss + self.w_flux * flux_loss
                    + self.w_td * td_loss)
        return -weighted, weighted
