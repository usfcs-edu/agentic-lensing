#!/usr/bin/env python
"""Every worked number in the guide, computed here and nowhere else.

    ~/.venvs/lensjudge/bin/python site/guide_src/worked_examples.py --emit
    ~/.venvs/lensjudge/bin/python site/guide_src/worked_examples.py --check

The rule the guide lives by: **a number may appear in the prose only if it is
returned by a function here**, tagged in the markdown as

    <!-- check: ch15.sigma_crit = 2.376e15 ± 1e12 -->

and verified by ``lint_guide.py`` + the workflow's verification pass. Prose that
quotes an untagged number is deleted, not trusted.

This is not ceremony. This repo's own final report carries a "~17 sigma" claim
in its abstract, README and commit message that reconciles with none of the
uncertainties it quotes — its own footnote says ~9.5 sigma. One person doing
one division would have caught it. So: every number, checked, every build.

Entries marked REPRO carry the repo's published value in ``expect``; --check
asserts we reproduce it. Those are the guide's credibility: if these drift, the
guide is wrong, or the repo changed and the guide must follow.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import astropy.units as u
import cosmo
import lensing as L
import numpy as np

HERE = Path(__file__).resolve().parent

# The registry lives in its own module so a driver can import exactly one
# guide's examples per process; see examples_registry.py and guides.py. This
# file is CONTENT — it self-registers on import and owns no CLI.
from examples_registry import example  # noqa: E402


# --------------------------------------------------------------------------- #
# Part I — the Abel projection (rho ~ r^-gamma projects to Sigma ~ R^(1-gamma))
# --------------------------------------------------------------------------- #
@example(
    "ch03",
    expect={
        # gamma=2 (isothermal): Sigma ~ 1/R exactly, so doubling R halves it.
        "sigma_ratio_isothermal": (0.5, 1e-9),
        # gamma=1.103 (the money number, cgl's gamma_binned(corr,low)): Sigma is
        # nearly FLAT -- doubling R only drops it ~7%.
        "sigma_ratio_money": (0.9310948198302291, 1e-6),
        "sigma_ratio_mid": (0.7071067811865476, 1e-6),
        # Each ratio must equal the pure exponent prediction (R1/R2)^(gamma-1),
        # with the profile's amplitude and (3-gamma)/2 prefactor cancelling out.
        "exponent_match_isothermal": (0.0, 1e-9),
        "exponent_match_money": (0.0, 1e-9),
        "exponent_match_mid": (0.0, 1e-9),
    },
    note="The Abel-projection exponent 1-gamma, checked against site/guide_src/"
    "lensing.py's EPL convergence law (the same function ch18/ch20 use), by a "
    "ratio at two radii that cancels every multiplicative constant.",
)
def ch03_abel_projection():
    R1, R2 = 0.5, 1.0  # arcsec; q=1 (circular) isolates the radial exponent

    def sigma_ratio(gamma):
        k1 = L.epl_kappa(R1, 0.0, theta_E=1.0, slope=gamma, q=1.0)
        k2 = L.epl_kappa(R2, 0.0, theta_E=1.0, slope=gamma, q=1.0)
        return float(k2 / k1)

    def predicted_ratio(gamma):
        return (R1 / R2) ** (gamma - 1.0)

    gamma_isothermal = 2.0     # rho ~ r^-2, the isothermal sphere
    gamma_money = 1.103        # gamma_binned(corr, low); CAMPAIGN.md:134
    gamma_mid = 1.5            # an interpolating case, for the exercises

    v = dict(R1=R1, R2=R2, gamma_isothermal=gamma_isothermal,
              gamma_money=gamma_money, gamma_mid=gamma_mid)
    for tag, gamma in [("isothermal", gamma_isothermal),
                        ("money", gamma_money), ("mid", gamma_mid)]:
        ratio = sigma_ratio(gamma)
        pred = predicted_ratio(gamma)
        v[f"sigma_ratio_{tag}"] = ratio
        v[f"predicted_ratio_{tag}"] = pred
        v[f"exponent_match_{tag}"] = abs(ratio - pred)
    return v


# --------------------------------------------------------------------------- #
# Part I — the potential trio (psi, alpha = grad psi, kappa = (1/2) laplacian psi)
# --------------------------------------------------------------------------- #
@example(
    "ch06",
    expect={
        # Green's function of the 2-D Laplacian: away from a point mass its log
        # potential is harmonic -- all the convergence sits in the delta at r=0.
        "point_mass_laplacian_offcenter": (0.0, 1e-6),
        # Poisson for lensing: div(alpha) = laplacian(psi) should equal 2*kappa,
        # checked by finite-differencing L.sis_deflection itself (no closed-form
        # shortcut) against the SIS's analytic kappa.
        "sis_poisson_residual": (0.0, 1e-6),
    },
    note="The potential trio, checked two ways: the point-mass log potential is "
    "harmonic off-centre (Green's function of the Laplacian), and div(alpha) = "
    "2 kappa for the SIS, both against site/guide_src/lensing.py.",
)
def ch06_potential_trio():
    theta_E = 1.0
    h = 1e-4
    x0, y0 = np.array([1.2]), np.array([0.9])  # r0 = 1.5 exactly
    r0 = float(np.hypot(x0, y0)[0])

    def laplacian_fd(f, x, y, h=h):
        return (f(x + h, y) + f(x - h, y) + f(x, y + h) + f(x, y - h) - 4 * f(x, y)) / h**2

    # (1) Point mass: psi = theta_E^2 * ln(r). The Green's function of the 2-D
    # Laplacian is (up to normalisation) ln(r); a point mass just carries the
    # mass in the prefactor. Off-centre it must be harmonic -- there is no
    # surface density anywhere except at the point itself.
    def psi_point(x, y):
        r = np.hypot(x, y)
        r = np.where(r == 0, 1e-12, r)
        return theta_E**2 * np.log(r)

    lap_point = float(laplacian_fd(psi_point, x0, y0)[0])

    # (2) SIS: alpha = L.sis_deflection is the SAME function ch18's Jacobian
    # differentiates. Here we ask a different question of it: does its
    # divergence equal 2 kappa? kappa_SIS(theta) = theta_E / (2 theta).
    def div_alpha_numeric(defl_fn, x, y, h=h):
        axp, _ = defl_fn(x + h, y)
        axm, _ = defl_fn(x - h, y)
        _, ayp = defl_fn(x, y + h)
        _, aym = defl_fn(x, y - h)
        return (axp - axm) / (2 * h) + (ayp - aym) / (2 * h)

    div_num = float(
        div_alpha_numeric(lambda x, y: L.sis_deflection(x, y, theta_E), x0, y0)[0]
    )
    kappa_analytic = theta_E / (2.0 * r0)

    return dict(
        theta_E=theta_E,
        r0=r0,
        point_mass_laplacian_offcenter=lap_point,
        sis_div_alpha_numeric=div_num,
        sis_kappa_analytic=kappa_analytic,
        sis_two_kappa=2.0 * kappa_analytic,
        sis_poisson_residual=div_num - 2.0 * kappa_analytic,
    )


# --------------------------------------------------------------------------- #
# Part I — The mathematical spine
# --------------------------------------------------------------------------- #
@example(
    "ch08",
    expect={
        # foundry-i/README.md:32 and cgl/guards.py:74-91 (assert_psf_sampling):
        # the SAME noise, before and after fixing a PSF sampling-convention bug.
        "chi2_nu_psf_broadened": (3.4, 0.05),
        "chi2_nu_psf_fixed": (1.05, 0.05),
        # foundry-i/README.md:27 and cgl/guards.py:129-142
        # (assert_model_subtracted_sky): the retracted "celebrated" chi2 and
        # its honest recalibration, in the OTHER direction (too good, not too bad).
        "chi2_nu_sky_artifact": (0.451, 0.001),
        "chi2_nu_sky_honest": (0.92, 0.01),
        # papers/main.tex Table tab:parity, Gate E: the audited Occam term at
        # the stored parity point map_marg_pd.npz.
        "occam_logdetA_parity": (323.229, 0.001),
        "occam_condA_parity": (1.4e4, 1.0),
        # From-scratch check (plain numpy, no jax): marg.py's closed-form
        # logL, with its dropped normalization restored, equals a brute-force
        # numerical integral of the SAME Gaussian evidence to machine precision.
        # Laplace is not an approximation here; it is the exact answer.
        "toy_laplace_exact_diff": (0.0, 1e-9),
        # Exercise 8.3: the toy logL WITHOUT the Occam term (gigalens's plain
        # lstsq path) is always larger (less negative) than marg.py's -- by
        # exactly half the log-determinant.
        "toy_occam_correction_nats": (2.0127, 0.001),
    },
    note="Chi-squared retractions in both directions, and the Occam term as an exact Gaussian integral",
)
def ch08_bayes_occam_and_chi2():
    v = dict(
        chi2_nu_psf_broadened=3.4,
        chi2_nu_psf_fixed=1.05,
        chi2_nu_sky_artifact=0.451,
        chi2_nu_sky_honest=0.92,
    )
    v["psf_fix_ratio"] = v["chi2_nu_psf_broadened"] / v["chi2_nu_psf_fixed"]
    v["sky_inflation_ratio"] = v["chi2_nu_sky_honest"] / v["chi2_nu_sky_artifact"]

    # main.tex's parity table, Gate E: -1/2 logdet(A) vs numpy slogdet, agreeing
    # to 1e-10 at the stored validation point -- the Occam term is a real,
    # audited number, not a plot device.
    v["occam_logdetA_parity"] = 323.229
    v["occam_condA_parity"] = 1.4e4

    # Toy 1-parameter linear-Gaussian model, same algebra as cgl/marg.py's
    # marg_loglik (y = a*x + noise, ridge prior a ~ N(0, 1/lam)), reimplemented
    # here in plain numpy -- marg.py itself is jax, but the arithmetic is
    # identical: b = x.R, A = x.x + lam, a* = b/A, logdetA = log(A).
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.1, 3.9, 6.2, 7.8, 10.3])
    lam = 1.0
    b = float(np.dot(x, y))
    A = float(np.dot(x, x) + lam)
    a_star = b / A
    logdetA = float(np.log(A))
    # marg.py's logL: "data part, consts dropped" (cgl/marg.py:1, 19-20).
    logL_marg_style = -0.5 * np.dot(y, y) + 0.5 * b * a_star - 0.5 * logdetA
    # Restore the dropped normalization (+ (k/2) log 2*pi, k=1 amplitude here)
    # to get the FULL evidence, matching main.tex eq:logl's "+ const".
    logZ_closed = logL_marg_style + 0.5 * np.log(2.0 * np.pi)

    # Brute-force check: integrate the un-normalized posterior over a on a fine
    # grid and compare. If Laplace is exact for a linear-Gaussian posterior,
    # these must agree to machine precision, not merely "be close."
    sigma_a = 1.0 / np.sqrt(A)
    a_grid = np.linspace(a_star - 14 * sigma_a, a_star + 14 * sigma_a, 4_000_001)
    resid2 = np.sum((y[None, :] - a_grid[:, None] * x[None, :]) ** 2, axis=1)
    log_integrand = -0.5 * resid2 - 0.5 * lam * a_grid ** 2
    m = log_integrand.max()
    logZ_numeric = m + np.log(np.trapezoid(np.exp(log_integrand - m), a_grid))

    v["toy_a_star"] = a_star
    v["toy_logdetA"] = logdetA
    v["toy_logL_marg_style"] = logL_marg_style
    v["toy_logZ_closed"] = logZ_closed
    v["toy_logZ_numeric"] = logZ_numeric
    v["toy_laplace_exact_diff"] = abs(logZ_closed - logZ_numeric)

    # Exercise 8.3: drop the Occam term (gigalens's plain lstsq path evaluates
    # the likelihood at the mode with no -1/2 logdetA correction) and see how
    # much better the (uncorrected) number looks.
    v["toy_logL_no_occam"] = logL_marg_style + 0.5 * logdetA
    v["toy_occam_correction_nats"] = 0.5 * logdetA
    return v


# --------------------------------------------------------------------------- #
# Part III — Cosmology
# --------------------------------------------------------------------------- #
@example(
    "ch13",
    expect={
        "hubble_time_gyr": (13.968, 0.01),
        "hubble_distance_mpc": (4282.75, 0.5),
        "h0_per_gyr": (0.07159, 0.0005),
        "scale_factor_cikota_source": (0.52715, 0.001),
        "naive_cz_carousel_kms": (429302.8, 50.0),
    },
    note="Scale factor, Hubble's law from a Taylor expansion, and what H0=70 means "
    "-- H0 alone, with no Om0/Ode0 (that mixture is Ch14's job).",
)
def ch13_expansion():
    C = cosmo.COSMO  # the repo-wide FlatLambdaCDM(H0=70, Om0=0.3); only H0 used here
    v = dict(
        H0_km_s_mpc=C.H0.value,
        hubble_time_gyr=C.hubble_time.to_value(u.Gyr),      # 1/H0
        hubble_distance_mpc=C.hubble_distance.to_value(u.Mpc),  # c/H0
        h0_per_gyr=C.H0.to_value(1 / u.Gyr),
        # The repo's own redshifts, read as scale factors a = 1/(1+z) at the
        # moment each photon was emitted (a(t0) = 1 today, by convention).
        z_cikota_lens=0.271,                                 # cikota-2023 lens
        z_cikota_source=0.897,                                # cikota-2023 source
        z_carousel_lens=0.49,                                 # sheu-2024b lens
        z_carousel_ref=1.432,                                 # sheu-2024b reference
        scale_factor_cikota_lens=C.scale_factor(0.271),      # cikota-2023 lens
        scale_factor_cikota_source=C.scale_factor(0.897),    # cikota-2023 source
        scale_factor_carousel_lens=C.scale_factor(0.49),     # sheu-2024b lens
        scale_factor_carousel_ref=C.scale_factor(1.432),     # sheu-2024b reference
        # The same z_l=0.5, z_s=2.0 pair Ch15 uses for the D_ds != D_s - D_d gotcha.
        scale_factor_z05=C.scale_factor(0.5),
        scale_factor_z20=C.scale_factor(2.0),
        # Why "v = cz" stops meaning a velocity once z is not << 1: the
        # Carousel's own reference redshift, misread as a Newtonian recession
        # speed, exceeds c outright.
        naive_cz_carousel_kms=1.432 * L.C_KM_S,
        # The linear Hubble law, both directions, at a round 100 Mpc: exercise
        # fodder for "derive v = H0 d, then invert it to get z back".
        hubble_law_v_at_100mpc_kms=C.H0.value * 100.0,
        redshift_approx_at_100mpc=(C.H0.value * 100.0) / L.C_KM_S,
        # cgl/euclid_io.py:54 -- the money-number pipeline's own inert defaults,
        # used only so "Einstein radius" has an angular meaning; nothing in that
        # pipeline's gradient depends on either number.
        euclid_default_z_lens=0.5,
        euclid_default_z_src=1.0,
    )
    return v


@example(
    "ch15",
    expect={
        # reproductions/sheu-2024b/04_setup_multiplane.py prints 2.38e15 / 4.62e13
        "sigma_crit_carousel": (2.376e15, 5e12),
        "mass_within_theta_e_carousel": (4.621e13, 1e11),
        # Published Table 2 (Sheu+2024, reproductions/sheu-2024b/README.md):
        # M(<theta_E) = 4.78e13 Msun for the circular-equivalent identity. Our
        # geometry-only estimate (Sigma_crit * pi R_E^2, which assumes a
        # circular lens) lands 3.3% low -- the published value comes from the
        # full elliptical (q=0.87) mass model, not the circular approximation.
        "mass_repro_vs_paper_ratio": (0.9667, 0.001),
        # Same method, applied to the Carousel's OTHER deflector (Ld, theta_E =
        # 0.99", published M = 2.77e11 Msun, q=0.69 -- more elliptical than La's
        # q=0.87, so a slightly larger deviation from the circular estimate is
        # expected, not a red flag). Exercise 15.3.
        "mass_repro_vs_paper_ratio_ld": (0.963, 0.001),
        # Comoving distance is additive along the line of sight; dividing that
        # additive difference by (1+z_s) exactly once must reproduce astropy's
        # own D_ds -- this is the derivation of the non-additivity fix, checked.
        "d_ds_dc_diff": (0.0, 1e-6),
        # D_L = (1+z)^2 D_A, checked against astropy's own luminosity_distance.
        "d_l_da_diff": (0.0, 1e-6),
        # The naive-subtraction gap has a closed form entirely in terms of D_d:
        # D_ds - (D_s - D_d) = D_d * (z_s - z_l)/(1+z_s). Checked against the
        # actual gap between the two already-computed numbers above.
        "additivity_gap_diff": (0.0, 1e-6),
    },
    note="Carousel cluster lens (Sheu 2024b): z_l=0.49, z_ref=1.432, theta_E=13.03 arcsec",
)
def ch15_distances_and_sigma_crit():
    z_l, z_ref, theta_E = 0.49, 1.432, 13.03
    v = dict(
        sigma_crit_carousel=cosmo.sigma_crit(z_l, z_ref),
        sigma_crit_carousel_msun_pc2=cosmo.sigma_crit(z_l, z_ref) / 1e12,
        mass_within_theta_e_carousel=cosmo.mass_within_theta_e(theta_E, z_l, z_ref),
        arcsec_to_mpc_carousel=cosmo.arcsec_to_mpc(z_l),
    )
    v["mass_repro_vs_paper_ratio"] = v["mass_within_theta_e_carousel"] / 4.78e13
    v["r_e_carousel_mpc"] = v["arcsec_to_mpc_carousel"] * theta_E
    # Exercise 15.3: the same method, applied to the Carousel's secondary
    # deflector Ld (theta_E = 0.99", published M = 2.77e11 Msun, q=0.69).
    theta_E_ld = 0.99
    v["mass_within_theta_e_carousel_ld"] = cosmo.mass_within_theta_e(theta_E_ld, z_l, z_ref)
    v["mass_repro_vs_paper_ratio_ld"] = v["mass_within_theta_e_carousel_ld"] / 2.77e11
    # The non-additivity gotcha, quantified: D_s - D_d is NOT D_ds.
    z_l2, z_s2 = 0.5, 2.0
    v["d_d_05"] = cosmo.d_a(z_l2)
    v["d_s_20"] = cosmo.d_a(z_s2)
    v["d_ds_05_20"] = cosmo.d_ds(z_l2, z_s2)
    v["naive_subtraction"] = v["d_s_20"] - v["d_d_05"]
    v["subtraction_error_ratio"] = v["d_ds_05_20"] / v["naive_subtraction"]
    # The fix, derived: comoving distance IS additive (it's a line-of-sight
    # integral of c dz'/H(z')), so subtract those, then divide once by (1+z_s)
    # -- never subtract two D_A's, each already divided by its OWN (1+z).
    v["d_c_05"] = cosmo.COSMO.comoving_distance(z_l2).to_value(u.Mpc)
    v["d_c_20"] = cosmo.COSMO.comoving_distance(z_s2).to_value(u.Mpc)
    v["d_ds_from_dc"] = (v["d_c_20"] - v["d_c_05"]) / (1.0 + z_s2)
    v["d_ds_dc_diff"] = abs(v["d_ds_from_dc"] - v["d_ds_05_20"])
    # The third distance: luminosity distance, D_L = (1+z)^2 D_A.
    v["d_l_20"] = cosmo.COSMO.luminosity_distance(z_s2).to_value(u.Mpc)
    v["d_l_from_da"] = (1.0 + z_s2) ** 2 * v["d_s_20"]
    v["d_l_da_diff"] = abs(v["d_l_from_da"] - v["d_l_20"])
    # The naive-subtraction gap, in closed form (both sides checked against
    # each other, not asserted): D_ds - (D_s - D_d) = D_d * (z_s-z_l)/(1+z_s).
    v["additivity_gap"] = v["d_ds_05_20"] - v["naive_subtraction"]
    v["additivity_gap_closed_form"] = v["d_d_05"] * (z_s2 - z_l2) / (1.0 + z_s2)
    v["additivity_gap_diff"] = abs(v["additivity_gap"] - v["additivity_gap_closed_form"])
    return v


# --------------------------------------------------------------------------- #
# Part IV — Lensing
# --------------------------------------------------------------------------- #
@example(
    "ch19",
    expect={
        # The defining identity. Not a coincidence — it IS the definition.
        "sis_mean_kappa_at_theta_e": (1.0, 1e-4),
        # reproductions/cikota-2023: sigma_SIE 347 km/s; our repro theta_E 2.10",
        # paper 2.52". The SIS estimate lands between them.
        "theta_e_cikota": (2.233, 0.01),
    },
    note="Einstein radius: the mean-convergence identity, and Cikota's Einstein cross",
)
def ch19_einstein_radius():
    v = dict(
        sis_mean_kappa_at_theta_e=L.mean_kappa_within(lambda t: 0.5 * 1.0 / t, 1.0),
        # A typical massive elliptical: sigma_v=250 km/s, z_l=0.5, z_s=2.0
        theta_e_typical=cosmo.theta_e_from_sigma_v(250.0, 0.5, 2.0),
        # Cikota 2023's Einstein cross DESI-253.2534+26.8843
        theta_e_cikota=cosmo.theta_e_from_sigma_v(347.0, 0.271, 0.897),
    )
    return v


@example(
    "ch18",
    expect={
        "sis_mu_analytic_match": (0.0, 1e-9),
        "sis_mu_inside_analytic_match": (0.0, 1e-8),
        "sie_crit_x_extent_match": (0.0, 1e-3),
        "sie_cut_radius_max_match": (0.0, 1e-9),
    },
    note="Magnification as a Jacobian determinant (SIS), a parity flip inside "
    "theta_E, and the SIE tangential critical curve + radial-branch caustic "
    "(the cut) that Figure 18.1 plots -- reproduced here by independent "
    "bisection/angular-sampling numerics, not by importing figures.py.",
)
def ch18_magnification():
    theta_E = 1.0
    defl = lambda px, py: L.sis_deflection(px, py, theta_E)  # noqa: E731
    th = np.array([2.0])
    a = L.lens_jacobian(defl, th, np.array([0.0]))
    mu_num = float(L.magnification(*a)[0])
    mu_ana = float(th[0] / (th[0] - theta_E))  # SIS radial-image magnification

    # A second image position, INSIDE theta_E: the tangential eigenvalue
    # 1 - theta_E/theta crosses zero at theta = theta_E, so det A changes
    # sign there. |mu| = 1 here -- same total flux as the source, not
    # magnified at all, but det A < 0: a parity-flipped (mirror) image.
    th_in = np.array([0.5])
    a_in = L.lens_jacobian(defl, th_in, np.array([0.0]))
    mu_in_num = float(L.magnification(*a_in)[0])
    mu_in_ana = float(th_in[0] / (th_in[0] - theta_E))

    # --- SIE critical curve / caustic, the numbers behind Figure 18.1 -------
    # site/guide_src/figures.py's sie_caustics() finds the tangential branch
    # by contouring det A = 0 on a grid (matplotlib) and the radial branch's
    # caustic (the cut) by walking a small circle in angle. This reproduces
    # both with the SAME theta_E, q and the SAME lensing.py primitives, but
    # by bisection along each axis instead of a contour, so it has no
    # matplotlib dependency and is an independent check, not a re-import.
    theta_E_sie, q_sie = 1.0, 0.7
    defl_sie = lambda px, py: L.sie_deflection(px, py, theta_E_sie, q_sie)  # noqa: E731

    def _bisect_root(f, lo, hi, n=60):
        flo = f(lo)
        for _ in range(n):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if np.sign(fm) == np.sign(flo):
                lo, flo = mid, fm
            else:
                hi = mid
        return 0.5 * (lo + hi)

    crit_x_extent = _bisect_root(
        lambda x: L.det_a(defl_sie, np.array([x]), np.array([0.0]))[0], 0.5, 1.5)
    crit_y_extent = _bisect_root(
        lambda y: L.det_a(defl_sie, np.array([0.0]), np.array([y]))[0], 0.3, 1.2)

    # Radial branch: a singular SIE's radial critical curve collapses onto
    # theta=0, but its caustic image does not, because alpha's DIRECTION
    # still varies with angle even as its magnitude stays finite at r -> 0.
    # Walk a small circle in angle and push it through beta = theta - alpha.
    eps = 1e-3
    phi = np.linspace(0.0, 2 * np.pi, 720)
    cx, cy = eps * np.cos(phi), eps * np.sin(phi)
    acx, acy = defl_sie(cx, cy)
    cut_radius_max = float(np.hypot(cx - acx, cy - acy).max())

    # figures.json's ch18-sie-caustics manifest values (grid contour), for
    # the direct side-by-side check.
    manifest_crit_x_extent = 0.9999266980051081
    manifest_cut_radius_max = 0.8102368876963119

    return dict(
        sis_mu_numerical=mu_num,
        sis_mu_analytic=mu_ana,
        sis_mu_analytic_match=abs(mu_num - mu_ana),
        sis_mu_inside_numerical=mu_in_num,
        sis_mu_inside_analytic=mu_in_ana,
        sis_mu_inside_analytic_match=abs(mu_in_num - mu_in_ana),
        sie_crit_x_extent=crit_x_extent,
        sie_crit_y_extent=crit_y_extent,
        sie_crit_x_extent_manifest=manifest_crit_x_extent,
        sie_crit_x_extent_match=abs(crit_x_extent - manifest_crit_x_extent),
        sie_cut_radius_max=cut_radius_max,
        sie_cut_radius_max_manifest=manifest_cut_radius_max,
        sie_cut_radius_max_match=abs(cut_radius_max - manifest_cut_radius_max),
    )


# --------------------------------------------------------------------------- #
# Part II — units
# --------------------------------------------------------------------------- #
@example(
    "ch09",
    expect={
        # 180*3600/pi, to the precision lensing.ARCSEC_PER_RAD is defined at.
        "arcsec_per_rad": (206264.806, 0.001),
        # sin(theta) - theta relative to theta, at theta = 1 arcsec in radians:
        # the small-angle approximation's own error, not an order-of-magnitude
        # claim about it.
        "small_angle_rel_error_1as": (3.917e-12, 0.01e-12),
        # Pogson: m = zeropoint - 2.5 log10(f); DESI Legacy zeropoint is 22.5
        # (reproductions/aion-1/03_fetch_provabgs.py:65-66). The DR11-south
        # sweep's own faint-end cut is m_z < 20 (dr11-campaign/papers/main.tex:94).
        "flux_nmgy_at_mz20": (10.0, 1e-9),
        # DESI Legacy DR10 g-band coadd seeing FWHM (cikota-2023/papers/main.tex:270)
        # over the survey's own native pixel scale (aion-1/_ls_cutout.py:36).
        "fwhm_in_native_px": (5.153, 0.01),
        # How much finer the campaign's own "fine" drizzle product (0.04"/px,
        # cgl/e2.py:55) is than the DESI Legacy Survey pixel it was resampled
        # from.
        "oversample_fine_vs_survey": (6.55, 0.01),
        # 1 AU / theta(1 arcsec), theta small enough that tan(theta) = theta:
        # derived from nothing but the small-angle approximation above, and
        # checked here against astropy's own parsec.
        "pc_from_small_angle_km": (3.0856775814913664e13, 1e6),
        "pc_definition_match_km": (0.0, 1e-6),
        # The zero point ITSELF, read backwards: flux = 1 nanomaggie by
        # definition when m = zp, and f = 100 nmgy sits exactly 5 mag brighter.
        "mag_at_f100nmgy": (17.5, 1e-9),
        # det(delta_pix * I) for the fine product: cgl/e2.py:455's own
        # conversion_factor, a Jacobian determinant in one line.
        "pixel_area_fine_arcsec2": (0.0016, 1e-9),
    },
    note="arcsec/radian, the Pogson magnitude law, and the repo's own pixel scales",
)
def ch09_units():
    import astropy.units as u

    theta = 1.0 / L.ARCSEC_PER_RAD  # one arcsec, in radians
    zp, m_cut = 22.5, 20.0           # Legacy zeropoint; DR11-south's own m_z cut
    px_survey = 0.262                # DESI Legacy native pixel scale
    px_fine, px_binned, px_native = 0.04, 0.08, 0.13  # cgl/e2.py:55-59
    fwhm_legacy = 1.35                # cikota-2023/papers/main.tex:270
    au_km = (1 * u.au).to_value(u.km)
    pc_km = au_km / theta            # the parsec, built from the AU + small angle
    return dict(
        arcsec_per_rad=L.ARCSEC_PER_RAD,
        one_arcsec_in_rad=theta,
        small_angle_rel_error_1as=abs(math.sin(theta) - theta) / theta,
        flux_nmgy_at_mz20=10.0 ** ((zp - m_cut) / 2.5),
        mag_at_f100nmgy=zp - 2.5 * math.log10(100.0),
        fwhm_in_native_px=fwhm_legacy / px_survey,
        oversample_fine_vs_survey=px_survey / px_fine,
        oversample_binned_vs_survey=px_survey / px_binned,
        oversample_native_vs_survey=px_survey / px_native,
        pixel_area_fine_arcsec2=px_fine**2,
        au_km=au_km,
        pc_from_small_angle_km=pc_km,
        pc_in_ly=(pc_km * u.km).to_value(u.lyr),
        pc_definition_match_km=abs(pc_km - (1 * u.pc).to_value(u.km)),
    )


# --------------------------------------------------------------------------- #
# Part II — the physical universe
# --------------------------------------------------------------------------- #
@example(
    "ch10",
    expect={
        "bn_n1": (1.6721, 1e-4),
        "bn_n4": (7.6697, 1e-4),
        "bn_n1_exact": (1.67835, 1e-4),
        "bn_n4_exact": (7.66925, 1e-4),
        "bn_ratio_n4_over_n1": (4.5869, 1e-3),
        "theta_e_ratio_2x_sigma": (4.0, 1e-6),
        "anchor_prior_sigmas": (2.268, 1e-3),
    },
    note="Sersic b_n (fit vs. exact incomplete-gamma), the SIS theta_E ~ sigma_v^2 "
         "scaling, and the isothermal prior this repo's own EPL model builds in",
)
def ch10_galaxies():
    from scipy.special import gammaincinv

    v = dict(
        bn_n1=L.sersic_bn(1),
        bn_n4=L.sersic_bn(4),
        bn_ratio_n4_over_n1=L.sersic_bn(4) / L.sersic_bn(1),
        # The defining condition is P(2n, b_n) = 1/2 (half the light within R_e);
        # gammaincinv(a, p) solves exactly that. The Capaccioli/Ciotti-Bertin fit
        # sersic_bn(n) = 1.9992n - 0.3271 is what gigalens and lensing.py use in
        # its place; comparing the two shows how good the fit is, and where.
        bn_n1_exact=float(gammaincinv(2 * 1, 0.5)),
        bn_n4_exact=float(gammaincinv(2 * 4, 0.5)),
    )
    # A fiducial massive elliptical lens: sigma_v=250 km/s at z_l=0.5, z_s=2.0
    # (the same fiducial system Ch 19 uses for its Einstein-radius example).
    v["theta_e_typical_elliptical"] = cosmo.theta_e_from_sigma_v(250.0, 0.5, 2.0)
    # theta_E ~ sigma_v^2 for an SIS (Hsu Eq. 1 / Ch 19): doubling sigma_v must
    # quadruple theta_E, with nothing else changed.
    v["theta_e_150"] = cosmo.theta_e_from_sigma_v(150.0, 0.5, 2.0)
    v["theta_e_300"] = cosmo.theta_e_from_sigma_v(300.0, 0.5, 2.0)
    v["theta_e_ratio_2x_sigma"] = v["theta_e_300"] / v["theta_e_150"]
    # This repo's own EPL mass prior (cgl/e2.py:110):
    # TruncatedNormal(2.0, 0.25, low=1.0, high=2.7). Centering it on 2.0 IS the
    # isothermal conspiracy, written into code.
    v["gamma_prior_mean"] = 2.0
    v["gamma_prior_sigma"] = 0.25
    # How many prior-sigma is the ANCHOR (Ch 25's trusted diagonal-native value,
    # gamma=1.433) from that isothermal center? Contrast with the money number's
    # own 3.588 (Ch 25) once you get there.
    v["anchor_prior_sigmas"] = (v["gamma_prior_mean"] - 1.433) / v["gamma_prior_sigma"]
    return v


# --------------------------------------------------------------------------- #
# Part II — the observation
# --------------------------------------------------------------------------- #
@example(
    "ch11",
    expect={
        # data/noise_kernel_report.json: closed form 0.76805, enumerated 0.76799
        "drizzle_t1_fine": (0.76805, 1e-5),
        # Out-of-domain sanity check (02_fit_noise_kernels.py:225-227): the
        # closed form is only valid for r>=2; at r=0.987 it returns a negative,
        # non-physical "correlation" -- the tell that you left its domain.
        "drizzle_t1_native": (-0.020, 0.001),
        # 02_fit_noise_kernels.py:79,202: NATIVE_PIX=0.1283 (WFC3/IR header fact)
        # / 0.04 (fine output scale) = the r the closed form is anchored at.
        "r_fine": (3.2075, 1e-3),
        # main.tex Table tab:products: the native product's headline plate
        # scale (0.13 in the kernel-fitting script is the same number rounded).
        "px_native_table": (0.128, 1e-3),
        "rho1_binned_measured": (0.615, 1e-3),
        "rho1_native_measured": (0.305, 1e-3),
        "neff_over_n_fine": (0.017, 1e-3),
        # cgl/mocks.py:65-66,227: err = sqrt(sigma_bkg^2 + model/t_exp), the
        # repo's own noise model, evaluated at one concrete pixel.
        "err_example": (0.223606797749979, 1e-9),
        # foundry-i/README.md + PERLMUTTER_CAMPAIGN.md: the PSF sampling-convention
        # defect (guards.assert_psf_sampling) and its fix, same noise, same data.
        "psf_chi2_broadened": (3.4, 0.15),
        "psf_chi2_fixed": (1.051, 0.01),
        # foundry-i/README.md:27 (final report, reproductions/foundry-i/index.md
        # #sec:defects, item 4): the retracted chi2=0.451, recalibrated on
        # model-subtracted residuals.
        "sky_chi2_artifact": (0.451, 0.001),
        "sky_chi2_honest": (0.92, 0.01),
    },
    note="Drizzle: the one line that explains why the diagonal likelihood is wrong, "
    "plus the PSF and sky-calibration incidents that motivate the noise model",
)
def ch11_drizzle():
    # 02_fit_noise_kernels.py:79-83,202-203: r = native detector pixel / output
    # pixel scale. NATIVE_PIX is a WFC3/IR header fact, not a fitted quantity.
    native_pix_wfc3ir = 0.1283
    px_fine, px_binned, px_native_product = 0.04, 0.08, 0.13
    r_fine = native_pix_wfc3ir / px_fine

    v = dict(
        native_pix_wfc3ir=native_pix_wfc3ir,
        px_fine=px_fine,
        px_binned=px_binned,
        px_native_product=px_native_product,
        px_native_table=0.128,
        r_fine=r_fine,
        drizzle_t1_fine=L.drizzle_lag1(r_fine),   # v3 fine skycell
        # v2d native product, r=0.987: BELOW the closed form's valid range r>=2
        # (02_fit_noise_kernels.py:225-227) -- quoted only as a "do not trust
        # this regime" example, never as a claim about the native correlation.
        drizzle_t1_native=L.drizzle_lag1(native_pix_wfc3ir / px_native_product),
    )
    # main.tex Table tab:products: along-axis lag-1 correlation of the
    # model-subtracted residual, measured (not just the drizzle-anchor piece),
    # on each of the three products. N_eff/N ~ 0.017 on the fine product; with
    # rho(1)=0.815 along each axis, a 2x2 fine block is ~78% internally correlated.
    rho1 = 0.815
    v["rho1_fine_measured"] = rho1
    v["rho1_binned_measured"] = 0.615
    v["rho1_native_measured"] = 0.305
    v["neff_over_n_fine"] = 0.017
    v["frac_independent_naive_2x2"] = float((1 - rho1) ** 2)
    # Overconfidence of a diagonal likelihood: it believes it has N independent
    # pixels when it has N_eff. Errors shrink as sqrt(N), so it is too tight by:
    v["diagonal_overconfidence_factor"] = float(math.sqrt(1.0 / 0.017))

    # cgl/mocks.py:65-66,227: the repo's own noise model, background + Poisson,
    # instantiated with its own constants (gu-2022 mock convention). A single
    # concrete pixel, model=1.0 (flux units), makes err = sqrt(sigma_bkg^2 +
    # model/t_exp) something you can check on a calculator.
    sigma_bkg_mock, exp_time_mock, model_example = 0.2, 100.0, 1.0
    v["sigma_bkg_mock"] = sigma_bkg_mock
    v["exp_time_mock"] = exp_time_mock
    v["model_example"] = model_example
    v["err_example"] = float(
        math.sqrt(sigma_bkg_mock**2 + model_example / exp_time_mock)
    )

    # foundry-i R0c (guards.assert_psf_sampling): an empirical PSF sampled at its
    # own finer pixel scale, fed uncorrected into a subgrid_kernel call that
    # assumes delta_pix sampling, double-applies the refinement and broadens the
    # effective PSF 2x. Fixing ONLY this (same noise model, same everything else)
    # moved the native-scale chi2_nu floor:
    v["psf_chi2_broadened"] = 3.4
    v["psf_chi2_fixed"] = 1.051
    # foundry-i Phase R (guards.assert_model_subtracted_sky): the sky sigma was
    # calibrated on RAW image fluctuations that were ~70% diffuse lens-wing
    # flux, not noise -- inflating sigma_bkg and making chi2 look too good.
    v["sky_chi2_artifact"] = 0.451
    v["sky_chi2_honest"] = 0.92
    return v


@example(
    "ch12",
    expect={
        # reproductions/hsu-2025/data/xmatch_table2.json, system DESI-004.5374+01.0382
        "z_ratio_desi004": (2.2954, 0.001),
        # reproductions/hsu-2025/data/dr1_stats.json (05_run_full_fof.py)
        "n_groups_after_ratio_cut": (13530, 0),
        # reproductions/hsu-2025/data/classified_stats.json (07_classify_einstein_dimple.py)
        "sigma_v_median_kms": (217.09, 0.01),
        "frac_with_reliable_sigma_v": (0.3132, 0.0005),
        "delta_lambda_caK_at_median_sigma_v": (2.848, 0.001),
        "lambda_obs_src_halpha_if_used": (11486.15, 0.01),
        # reproductions/lensjudge/parity/FINDINGS.md, the withdrawn NISP paper
        "nisp_photz_fraction": (0.875, 0.001),
    },
    note="Redshift as a template match, sigma_v as a line WIDTH: reproductions/hsu-2025",
)
def ch12_spectroscopy():
    C_KM_S = 299792.458
    # A real DESI FoF pair (reproductions/hsu-2025/data/xmatch_table2.json,
    # matched to Hsu+2025 Table 2 system DESI-004.5374+01.0382).
    z_lens, z_src = 0.3266195325989532, 0.7497079335303389
    # Two lines chosen for what actually shows up in each spectrum: the lens is
    # an old-stellar-population absorber (Ca II K), the source a star-forming
    # emitter ([OII]) -- the same 1+z stretch, two different physical features.
    lambda_rest_caK = 3933.66    # Angstrom, absorption (old stars -> the lens)
    lambda_rest_oii = 3727.42    # Angstrom, emission, doublet mean (the source)
    lambda_rest_halpha = 6564.61  # Angstrom, vacuum, Balmer-alpha
    v = dict(
        z_lens=z_lens,
        z_src=z_src,
        z_ratio_desi004=z_src / z_lens,
        lambda_obs_lens_caK=lambda_rest_caK * (1 + z_lens),
        lambda_obs_src_oii=lambda_rest_oii * (1 + z_src),
        # The same Balmer line, redshifted to the SOURCE's distance: it runs
        # off the red end of an optical spectrograph, which is why a template
        # match (Ch. 7's cross-correlation) rather than a hardcoded line list.
        lambda_rest_halpha=lambda_rest_halpha,
        lambda_obs_src_halpha_if_used=lambda_rest_halpha * (1 + z_src),
    )
    # The FoF discovery cut itself: 05_run_full_fof.py:40,113-116 keeps any
    # group with z_max/z_min >= 1.3 -- "two different distances," algorithmically,
    # before anyone looks at an image.
    v["z_ratio_threshold"] = 1.3
    v["n_raw_fibers"] = 28_425_963
    v["n_after_prefilter"] = 15_786_243
    v["n_groups_after_ratio_cut"] = 13_530
    v["n_spectra_after_ratio_cut"] = 27_334

    # sigma_v from line WIDTH, not line shift (classified_stats.json).
    v["n_pairs_total"] = 13_530
    v["n_with_reliable_sigma_v"] = 4_238
    v["n_without_reliable_sigma_v"] = 13_530 - 4_238
    v["frac_with_reliable_sigma_v"] = 4_238 / 13_530
    v["sigma_v_median_kms"] = 217.08814239501953
    v["sigma_v_p16_kms"] = 152.1157489013672
    v["sigma_v_p84_kms"] = 292.2651416015625
    v["theta_e_median_arcsec"] = 0.6814775728707314

    # Doppler-broadening scale at the median sigma_v, on the same Ca II K line:
    # a shift (Section 2) moves a line; a width (this section) smears it.
    v["sigma_v_over_c_median"] = v["sigma_v_median_kms"] / C_KM_S
    v["delta_lambda_caK_at_median_sigma_v"] = (
        lambda_rest_caK * v["sigma_v_median_kms"] / C_KM_S
    )
    # The fitter's own trap: a FAILED FastSpecFit fit returns a CAP value, not a
    # null (07_classify_einstein_dimple.py:145-153) -- caught only by requiring
    # VDISP_IVAR > 0, the same discipline this guide applies to its own numbers.
    v["failed_fit_cap_kms"] = 250.0

    # A cautionary real failure: a withdrawn paper that called photometric
    # redshifts "deflector redshifts" (parity/FINDINGS.md, Phase B).
    v["nisp_photz_fraction"] = 385 / 440
    v["nisp_blind_recovery_frac"] = 0.35
    return v


# --------------------------------------------------------------------------- #
# Part V — the money number
# --------------------------------------------------------------------------- #
@example(
    "ch25",
    expect={
        "evidence_swing_nats": (191.1, 0.05),
        "sigma_vs_anchor_own_error": (9.7, 0.1),
        "sigma_vs_anchor_combined": (9.5, 0.15),
        # Cross-scale bracket (main.tex Table tab:crossscale): every gamma
        # this campaign measured on the same real system, diagonal vs
        # correlated, at three drizzle scales.
        "gamma_diag_low": (1.293, 0.001),
        "gamma_diag_steep": (2.423, 0.001),
        "gamma_fine_steep": (1.816, 0.001),
        "gamma_native_corr": (2.353, 0.001),
        "gamma_fine_diag_artifact": (2.585, 0.001),
        "theta_e_money": (2.624, 0.001),
        # The report's OWN sigma convention (combined-in-quadrature), applied
        # to the fine-steep row, gives 3.1 -- the same convention gives 9.4
        # for the money row above, never 17.
        "sigma_fine_steep_vs_anchor": (3.1, 0.1),
        # The naive stored-chain occupancy (CAMPAIGN.md, P2c partial #2a) vs.
        # what the evidence actually says under the SAME diagonal likelihood.
        "naive_w_low_occupancy": (0.9375, 0.001),
        # The saddle that forced tempered SMC instead of HMC (main.tex
        # sec:samplersaga) -- the full derivation is Ch. 26's; this chapter
        # only needs the two numbers that explain why HMC was abandoned here.
        "saddle_min_eigenvalue": (-14.85, 0.01),
        "gamma_map_saddle": (1.27, 0.001),
        "gamma_best_density_point": (1.10, 0.005),
        "rhat_saddle_metric": (22.3, 0.01),
    },
    note="The 191-nat flip and the sigma arithmetic the report itself gets wrong",
)
def ch25_money_number():
    # CAMPAIGN.md / main.tex Table tab:basinflip
    dlogz_diag = +162.2        # diagonal likelihood: favours STEEP
    dlogz_corr = -28.9         # correlated likelihood: favours LOW
    gamma_money, sig_money = 1.103, 0.008      # gamma_binned(corr, low)
    gamma_anchor, sig_anchor = 1.433, 0.034    # diagonal-native anchor

    swing = abs(dlogz_diag - dlogz_corr)
    diff = gamma_anchor - gamma_money
    v = dict(
        dlogz_diagonal=dlogz_diag,
        dlogz_correlated=dlogz_corr,
        evidence_swing_nats=swing,
        # e^191 -- "decisive" on Jeffreys' scale ends at ~5 nats. This is 38x that.
        bayes_factor_log10=swing / math.log(10),
        gamma_money=gamma_money,
        gamma_anchor=gamma_anchor,
        gamma_difference=diff,
        # The three defensible sigma numbers. NONE of them is 17.
        sigma_vs_anchor_own_error=diff / sig_anchor,
        sigma_vs_anchor_combined=diff / math.hypot(sig_money, sig_anchor),
        sigma_vs_money_error_only=diff / sig_money,
        # gamma's prior: TruncatedNormal(2.0, 0.25, low=1.0, high=2.7) at cgl/e2.py:110.
        # The money number sits this far above the prior's HARD lower wall...
        gamma_prior_low_wall=1.0,
        distance_above_wall=gamma_money - 1.0,
        # ...which is 12.9 of its OWN sigmas. So the wall is not clipping the
        # posterior -- but it is 3.6 prior-sigmas below the prior mean of 2.0,
        # i.e. the modeller never expected to be here.
        distance_above_wall_in_own_sigma=(gamma_money - 1.0) / sig_money,
        prior_sigmas_below_prior_mean=(2.0 - gamma_money) / 0.25,
        gamma_prior_mean=2.0,
        gamma_prior_sigma=0.25,
        gamma_prior_high_wall=2.7,
    )

    # The full cross-scale bracket: main.tex Table tab:crossscale / Fig. money.
    # Every basin, every product, both likelihoods, one real system.
    gamma_diag_low, sig_diag_low = 1.293, 0.012
    gamma_diag_steep, sig_diag_steep = 2.423, 0.027
    gamma_fine_steep, sig_fine_steep = 1.816, 0.117
    gamma_native_corr, sig_native_corr = 2.353, 0.096
    v.update(
        gamma_diag_low=gamma_diag_low, sig_diag_low=sig_diag_low,
        gamma_diag_steep=gamma_diag_steep, sig_diag_steep=sig_diag_steep,
        gamma_fine_steep=gamma_fine_steep, sig_fine_steep=sig_fine_steep,
        gamma_native_corr=gamma_native_corr, sig_native_corr=sig_native_corr,
        gamma_fine_diag_artifact=2.585,   # MAP only, v3 diagonal -- the upsampling artifact
        theta_e_money=2.624, sigma_theta_e_money=0.005,
        # The SAME combined-quadrature convention the report uses for the
        # fine-steep row ("3.1 sigma above"), applied consistently: it gives
        # 9.4 for the money row (matches sigma_vs_anchor_combined above), not 17.
        sigma_fine_steep_vs_anchor=(
            (gamma_fine_steep - gamma_anchor) / math.hypot(sig_fine_steep, sig_anchor)
        ),
    )

    # The basin-evidence flip itself, and the naive occupancy it overturns.
    # CAMPAIGN.md "P2c partial #2a" (diagonal SMC) and "P1c MONEY NUMBER"
    # (correlated SMC), both on the SAME binned product.
    v.update(
        logz_low_diag=38351.17, logz_steep_diag=38513.37,
        logz_low_corr=-4771.08, logz_steep_corr=-4799.96,
        n_chains_low=45, n_chains_steep=3, n_chains_total=48,
        naive_w_low_occupancy=45 / 48,
        n_inter_basin_migrations=0,
        n_smc_particles=128, n_smc_lambda_steps=28,
        ess_smc_low=77, ess_smc_high=118,
    )

    # The saddle (main.tex sec:samplersaga / CAMPAIGN.md "P1c metric-fix
    # attempts"): why HMC could not extract this number and SMC could.
    # Ch. 26 derives the geometry; this chapter only needs these two facts.
    v.update(
        saddle_min_eigenvalue=-14.85, n_negative_eigenvalues=5,
        gamma_map_saddle=1.27, logp_map_saddle=-4757,
        gamma_best_density_point=1.10, logp_best_density_point=-4683,
        rhat_saddle_metric=22.3,
    )
    return v


@example(
    "ch28",
    expect={
        "human_grade_auc": (0.577, 0.001),
        # Phase C gate scoreboard (FINDINGS.md "PHASE C FINAL GATE SCOREBOARD"):
        # three unrelated architectures -- an engineered-feature probe, a
        # published CNN, a fine-tuned 27B vision-language student -- all land
        # within the same band on the HARD (A/B-vs-D) wall.
        "rep_probe_wall_auc": (0.425, 0.001),
        "cnn_wall_auc": (0.646, 0.001),
        "student_wall_auc": (0.644, 0.001),
        # E1 (agreement-with-the-label): neither the fine-tuned 27B student
        # nor Claude Sonnet 5 (the C3 matched grader) comes anywhere near the
        # human intergrader QWK of 0.42, let alone the 0.776 team-member bar.
        "student_qwk_vs_consensus": (0.044, 0.001),
        "claude_matched_qwk_gate": (0.025, 0.001),
        # E2 (agreement-with-truth): the same student that cannot reproduce
        # the letter grade discriminates confirmed-vs-refuted truth AT LEAST
        # as well as the human grade does -- the grade and the truth are not
        # the same target.
        "student_truth_auc": (0.685, 0.001),
        "student_truth_auc_lo": (0.538, 0.001),
        "student_truth_auc_hi": (0.819, 0.001),
        "delta_auc_student_vs_human": (0.108, 0.001),
        "delta_auc_student_vs_human_lo": (-0.028, 0.001),
        "delta_auc_student_vs_human_hi": (0.241, 0.001),
        # Same wall as Ch. 27's: DESI grade-C at 1.3" seeing re-scored at
        # Euclid's 0.1" (reproductions/lensjudge/papers/main.tex:571-575).
        "euclid_c_regrade_frac": (0.53, 0.001),
        "euclid_c_to_a_frac": (6 / 17, 0.001),
        # The chapter's own from-scratch derivation of the intergrader QWK,
        # built ONLY from the seven published unordered score-pair counts
        # (FINDINGS.md:30) -- must land within rounding of the reported 0.420.
        "qwk_derived_from_pairs": (0.4198, 0.001),
        "pair_exact_frac": (0.5542, 0.001),
        "pair_one_step_frac": (0.3832, 0.001),
        "pair_two_step_frac": (0.0626, 0.001),
    },
    note="LensJudge: purity is flat across grades and the grade is ~chance vs truth",
)
def ch28_labels():
    # reproductions/lensjudge/parity/FINDINGS.md
    grades = {"A": (79, 83), "B": (23, 25), "C": (20, 22)}
    v = {f"purity_{g}": k / n for g, (k, n) in grades.items()}
    v["purity_spread"] = max(v.values()) - min(v.values())
    v["human_grade_auc"] = 0.577          # [0.396, 0.757] on 130 decided
    v["human_grade_auc_lo"] = 0.396
    v["human_grade_auc_hi"] = 0.757
    # The CI contains 0.5: "consistent with chance", NOT "proven to be chance".
    v["ci_contains_chance"] = float(v["human_grade_auc_lo"] < 0.5)
    v["intergrader_qwk"] = 0.420          # two graders' mutual agreement
    v["grader_vs_consensus_qwk"] = 0.776  # an upper bound: each is half the consensus

    # --- Phase C gate scoreboard: HARD (A/B lenses vs. grade-D human rejects) --
    # three different machine architectures, same frozen 259-row gate, one-shot.
    v["rep_probe_wall_auc"] = 0.425           # [0.330, 0.517] -- engineered features
    v["cnn_wall_auc"] = 0.646                 # [0.557, 0.734] -- published CNN ensemble
    v["student_wall_auc"] = 0.644             # [0.550, 0.733] -- fine-tuned 27B VLM

    # --- E1: agreement with the LABEL itself (QWK vs. published consensus) ----
    v["student_qwk_vs_consensus"] = 0.044     # n=162, fine-tuned 27B student
    v["claude_matched_qwk_gate"] = 0.025      # C3 gate, Claude Sonnet 5, matched grader

    # --- E2: agreement with TRUTH (confirmed vs. refuted, paired vs. human) ---
    v["student_truth_auc"] = 0.685
    v["student_truth_auc_lo"] = 0.538
    v["student_truth_auc_hi"] = 0.819
    v["delta_auc_student_vs_human"] = 0.108       # student - human, paired
    v["delta_auc_student_vs_human_lo"] = -0.028
    v["delta_auc_student_vs_human_hi"] = 0.241

    # --- the same wall Ch. 27 derives from the pixel scale, seen from the
    # label side: 17 DESI grade-C candidates independently rediscovered by
    # Euclid Q1 (0.1" VIS, ~10-expert panel); 53% re-grade to A or B once the
    # resolution improves, 6 of the 17 jumping all the way to A.
    v["euclid_c_regrade_n"] = 17
    v["euclid_c_to_a_count"] = 6
    v["euclid_c_regrade_frac"] = 0.53
    v["euclid_c_to_a_frac"] = 6 / 17

    # --- derive the symmetrized quadratic-weighted kappa FROM SCRATCH, using
    # only the seven unordered score-pair counts human_baseline.tex publishes
    # (FINDINGS.md:30 / Table tab:reliability). Grader identity is lost -- an
    # unordered pair {a,b} is symmetrized by letting it stand for BOTH rater
    # orderings (a,b) and (b,a), which is what "symmetrized" means and why
    # this equals Krippendorff's interval alpha for two coders.
    pair_counts = {(2, 2): 461, (2, 3): 387, (3, 3): 165, (3, 4): 115,
                   (4, 4): 100, (2, 4): 34, (1, 3): 48}
    scores4 = [1, 2, 3, 4]
    idx4 = {s: i for i, s in enumerate(scores4)}
    K4 = len(scores4)
    Ocount = np.zeros((K4, K4))
    for (a, b), n in pair_counts.items():
        Ocount[idx4[a], idx4[b]] += n
        Ocount[idx4[b], idx4[a]] += n
    n_pairs = sum(pair_counts.values())
    N4 = Ocount.sum()
    row = Ocount.sum(axis=1)
    col = Ocount.sum(axis=0)
    Wq = np.array([[(scores4[i] - scores4[j]) ** 2 / 9.0
                     for j in range(K4)] for i in range(K4)])  # /(4-1)^2
    Ecount = np.outer(row, col) / N4
    v["qwk_derived_from_pairs"] = 1.0 - (Wq * Ocount).sum() / (Wq * Ecount).sum()
    v["pair_n_candidates"] = n_pairs
    # delSc=0 (exact), 1, 2 collapse several pairs each -- sum by hand:
    exact_n = sum(n for (a, b), n in pair_counts.items() if a == b)
    one_n = sum(n for (a, b), n in pair_counts.items() if abs(a - b) == 1)
    two_n = sum(n for (a, b), n in pair_counts.items() if abs(a - b) == 2)
    v["pair_exact_frac"] = exact_n / n_pairs
    v["pair_one_step_frac"] = one_n / n_pairs
    v["pair_two_step_frac"] = two_n / n_pairs
    return v


# --------------------------------------------------------------------------- #
# Part III — Cosmology (ch14, appended after ch15 was already registered above;
# dict order is cosmetic and does not affect anything).
# --------------------------------------------------------------------------- #
@example(
    "ch14",
    expect={
        "flatness_sum": (1.0, 1e-9),
        "hz_match_diff": (0.0, 1e-9),
        "hz_match_diff_z1": (0.0, 1e-9),
        "hubble_distance_mpc": (4282.75, 0.5),
        "age_today_gyr": (13.467, 0.01),
    },
    note="FlatLambdaCDM(70, 0.3): what it asserts, and the Friedmann equation built by hand",
)
def ch14_frw_friedmann():
    import astropy.units as u
    from astropy import constants as const

    C = cosmo.COSMO  # the repo-wide FlatLambdaCDM(H0=70, Om0=0.3)
    c_kms = const.c.to_value(u.km / u.s)
    v = dict(
        H0=C.H0.value,
        Om0=C.Om0,
        Ode0=C.Ode0,     # astropy sets this to 1 - Om0 - Ok0 for a FLAT model
        Ok0=C.Ok0,       # forced to exactly 0 by the class name "Flat..."
        flatness_sum=C.Om0 + C.Ode0 + C.Ok0,
        hubble_distance_mpc=c_kms / C.H0.value,          # D_H = c / H0
        age_today_gyr=C.age(0).to_value(u.Gyr),
        rho_crit0_msun_mpc3=C.critical_density0.to_value(u.Msun / u.Mpc**3),
    )
    # The Friedmann equation, assembled here from nothing but Om0, Ode0, and the
    # definition H(z)^2 = H0^2 [Om0 (1+z)^3 + Ode0], must reproduce astropy's own
    # H(z) exactly -- that reproduction IS the content of the equation. z=0.5 is
    # worked in the chapter text; z=1.0 is held back for the reader to do first
    # and check against, in the exercises.
    def hz_manual(z):
        return C.H0.value * math.sqrt(C.Om0 * (1 + z) ** 3 + C.Ode0)

    v["hz_manual_z0p5"] = hz_manual(0.5)
    v["hz_astropy_z0p5"] = C.H(0.5).value
    v["hz_match_diff"] = abs(v["hz_manual_z0p5"] - v["hz_astropy_z0p5"])
    v["hz_manual_z1"] = hz_manual(1.0)
    v["hz_astropy_z1"] = C.H(1.0).value
    v["hz_match_diff_z1"] = abs(v["hz_manual_z1"] - v["hz_astropy_z1"])
    return v


# --------------------------------------------------------------------------- #
# Part I — the mathematical spine
# --------------------------------------------------------------------------- #
@example(
    "ch05",
    expect={
        "eig_tangential": (0.4, 1e-6),
        "eig_radial": (1.0, 1e-6),
        "saddle_logp_gain": (74.0, 0.5),
        "stable_digits_remaining": (1.653, 0.01),
    },
    note="Symmetric 2x2 -> kappa/shear eigenvalues; the v3b-low saddle; cond ~ 1e14",
)
def ch05_eigen_saddle_cond():
    # --- symmetric 2x2: A = (1-kappa) I - Gamma, eigenvalues (1-kappa) -+ |shear| ---
    # Matches figures.py:kappa_gamma_eigen's third panel (kappa=0.3, both nonzero).
    kappa, g1, g2 = 0.3, 0.3, 0.0
    A = np.array([[1 - kappa - g1, -g2], [-g2, 1 - kappa + g1]])
    eigvals = np.linalg.eigvalsh(A)
    v = dict(
        toy_kappa=kappa, toy_gamma1=g1, toy_gamma2=g2,
        toy_shear_mag=math.hypot(g1, g2),
        eig_tangential=float(eigvals[0]),
        eig_radial=float(eigvals[1]),
        symmetric_off_diag_gap=float(A[0, 1] - A[1, 0]),  # exactly 0 -- A is symmetric
        stretch_tangential=1.0 / float(eigvals[0]),  # 1/lambda_t: the A^-1 image stretch
        stretch_radial=1.0 / float(eigvals[1]),      # 1/lambda_r: the other axis
    )

    # --- definiteness / saddle: the SAME real saddle Ch. 2 previews, cgl/e2.py:544-558
    # (laplace_evidence: H = -jax.hessian(logpost), w = eigh(H), n_neg = sum(w<=0)).
    # CAMPAIGN.md:208-209 / main.tex sec:samplersaga: the v3b-low (T2, 46-dim) MAP.
    v["saddle_ndim"] = 46
    v["saddle_min_eig"] = -14.85
    v["saddle_n_negative"] = 5
    # Two candidate points on the SAME real fit: the saddle-consistent MAP
    # (gamma_map=1.27) versus the point map_polish never reached (gamma_best=1.10).
    # Both numbers are a mid-campaign snapshot, superseded by Ch. 25's converged
    # answer -- quoted here only to make "a saddle is not the top" concrete.
    v["gamma_at_saddle_map"] = 1.27
    v["logp_at_saddle_map"] = -4757.0
    v["gamma_at_true_peak"] = 1.10
    v["logp_at_true_peak"] = -4683.0
    v["saddle_logp_gain"] = v["logp_at_true_peak"] - v["logp_at_saddle_map"]

    # --- conditioning: the same T2 posterior, cond ~ 1e14 (foundry-i/README.md:114-117;
    # the same order of magnitude survives marginalization: papers/main.tex:352) ---
    v["hessian_diag_max"] = 1e12   # lens-light companion Sersic centers, H_ii
    v["hessian_diag_min"] = 1e-2   # Sersic index n_sersic, H_ii
    cond = v["hessian_diag_max"] / v["hessian_diag_min"]
    v["cond_ill_conditioned"] = cond
    eps64 = float(np.finfo(np.float64).eps)
    v["float64_eps"] = eps64
    digits_total = -math.log10(eps64)
    v["float64_digits_total"] = digits_total
    v["stable_digits_remaining"] = digits_total - math.log10(cond)

    # --- the fix: the ridge-regularised marg normal matrix, cgl/marg.py:48 ---
    v["cond_marg_normal_matrix"] = 1.37e4     # CAMPAIGN.md:702 parity-gate measurement
    v["cond_improvement_factor"] = cond / v["cond_marg_normal_matrix"]
    v["stable_digits_remaining_marg"] = digits_total - math.log10(v["cond_marg_normal_matrix"])
    return v


# --------------------------------------------------------------------------- #
# Part I -- det J as a literal area ratio (appended after other Part I entries
# above; dict order is cosmetic and does not affect anything).
# --------------------------------------------------------------------------- #
@example(
    "ch04",
    expect={
        "det_A": (0.485, 1e-9),
        "image_area_shoelace": (0.485, 1e-9),
        "mu": (2.061855670103093, 1e-9),
    },
    note="det J as a literal area ratio: the same 2x2 map figures/ch04-det-j-area.svg "
    "draws, checked two independent ways -- the determinant formula, and the "
    "shoelace-formula area of the unit square's image.",
)
def ch04_det_j_area():
    A = np.array([[0.6, 0.25], [0.1, 0.85]])
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    im = sq @ A.T  # the unit square, pushed through A

    def shoelace(pts):
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    det = float(np.linalg.det(A))
    return dict(
        det_A=det,
        unit_square_area=shoelace(sq),
        image_area_shoelace=shoelace(im),
        mu=1.0 / abs(det),
    )


# --------------------------------------------------------------------------- #
# Part I — Fourier, PSD, whitening (ch07). Reads the P1a artifacts directly
# rather than recomputing them: e_op/M/kept-pixel numbers are jax-built
# (cgl/whiten.py) on real data this repo does not carry, but the JSON reports
# those calls wrote are checked into the repo, so this is the honest way for
# a numpy-only guide script to reproduce them exactly.
# --------------------------------------------------------------------------- #
@example(
    "ch07",
    expect={
        # data/whitener_report.json products.v3b (main.tex Table tab:whiten)
        "eop_v3b": (0.012425, 0.0001),
        "s_raw_min_over_mean_v3": (0.052667, 0.0005),
        "s_raw_min_over_mean_v3b": (0.024139, 0.0005),
        # data/noise_kernel_report.json products.v3.rho_axis_meas, reduced to
        # a truncated integrated-autocorrelation-time / N_eff estimate
        "tau_int_v3_11lag": (7.453, 0.01),
        "n_eff_over_n_1d_v3": (0.1342, 0.001),
        "n_eff_over_n_2d_estimate": (0.0180, 0.001),
        "ar1_s0_over_spi_v3": (95.87, 0.05),
        "tau_int_v3b_7lag": (4.156, 0.01),
        "n_eff_over_n_1d_v3b": (0.2406, 0.001),
    },
    note="Wiener-Khinchin and the e_op whitening gate, read from the campaign's own P1a artifacts",
)
def ch07_fourier_whitening():
    root = HERE.parent.parent
    data = root / "reproductions" / "claude-giga-lens" / "data"
    nk = json.loads((data / "noise_kernel_report.json").read_text())
    wr = json.loads((data / "whitener_report.json").read_text())

    # Fine product's measured along-axis autocorrelation rho[0..11].
    # Wiener-Khinchin, truncated: tau_int = 1 + 2 sum_{k=1}^{K} rho(k);
    # N_eff/N = 1/tau_int for a 1-D stationary sequence (the same ratio
    # Ch. 23 will call ESS/N for a Markov chain).
    rho_v3 = nk["products"]["v3"]["rho_axis_meas"]
    tau_int = 1.0 + 2.0 * sum(rho_v3[1:])
    n_eff_1d = 1.0 / tau_int
    # Same reduction on the binned product (held back for the reader to redo
    # in the exercises): its shorter, faster-decaying tail (7 measured lags,
    # not 11) means less information is lost to 2x2 rebinning than to the
    # raw upsampling that makes v3 "fine" in the first place.
    rho_v3b_axis = nk["products"]["v3b"]["rho_axis_meas"]
    tau_int_v3b = 1.0 + 2.0 * sum(rho_v3b_axis[1:])

    # Toy AR(1) model rho(Delta) = r^|Delta| with r = the measured lag-1
    # correlation: closed-form PSD S(omega) = (1-r^2)/(1-2r cos(omega)+r^2)
    # (a geometric series summed both directions), evaluated at the two
    # frequency extremes.
    r = rho_v3[1]
    s0 = (1.0 + r) / (1.0 - r)     # S(0), omega=0 (DC)
    spi = (1.0 - r) / (1.0 + r)    # S(pi), omega=pi (Nyquist)

    v3, v3b = wr["products"]["v3"], wr["products"]["v3b"]
    return dict(
        rho1_v3_fine=rho_v3[1],
        tau_int_v3_11lag=tau_int,
        n_eff_over_n_1d_v3=n_eff_1d,
        # Noise is close to separable across the two pixel axes, so a 2-D
        # pixel's independent fraction is (roughly) the 1-D fraction squared.
        n_eff_over_n_2d_estimate=n_eff_1d ** 2,
        # main.tex:213/:318 headline figure, quoted for comparison — it comes
        # from a fuller 2-D radial audit (reproductions/foundry-i/46_noise_audit.py)
        # this guide does not re-run, not from the estimate above.
        n_eff_over_n_reported=0.017,
        tau_int_v3b_7lag=tau_int_v3b,
        n_eff_over_n_1d_v3b=1.0 / tau_int_v3b,
        ar1_s0_v3=s0,
        ar1_spi_v3=spi,
        ar1_s0_over_spi_v3=s0 / spi,
        eop_v3b=v3b["e_op"],
        m_v3b=v3b["M"],
        m_v3=v3["M"],
        n_keep_v3b=v3b["n_keep"],
        n_eroded_v3b=v3b["n_eroded"],
        s_floor_v3b=v3b["s_floor"],
        s_floor_default=0.05,
        s_raw_min_over_mean_v3=v3["s_raw_min_over_mean"],
        s_raw_min_over_mean_v3b=v3b["s_raw_min_over_mean"],
        logdet_per_pix_v3b=v3b["logdet_per_pix"],
        var_u_dense_v3b=v3b["dense_sigma_u"]["var_mean"],
        # main.tex:428 / CAMPAIGN.md:622-623: the REJECTED fixed-0.05-floor
        # result, kept on record as the reason the floor became adaptive.
        var_u_hard_floor_biased=0.981,
    )


# --------------------------------------------------------------------------- #
# Part IV — Gravitational lensing: Newtonian vs. GR deflection, and the
# thin-lens projection (ch16). Uses astropy's own constants, not hand-typed
# ones, so this reproduces to machine precision against anyone else's copy.
# --------------------------------------------------------------------------- #
@example(
    "ch16",
    expect={
        # 2GM/(c^2 R_sun) and 4GM/(c^2 R_sun): the classical and GR deflection
        # of starlight grazing the Sun's limb. GR is EXACTLY double, by
        # construction (alpha_gr = 2 * alpha_newton in the code below) --
        # the physics content is that this ratio, not either number alone,
        # is what Eddington's 1919 expedition existed to measure.
        "alpha_newton_arcsec": (0.8756, 0.001),
        "alpha_gr_arcsec": (1.7512, 0.001),
        "gr_over_newton_ratio": (2.0, 1e-9),
        "schwarzschild_radius_sun_km": (2.9533, 0.001),
        # Why a galaxy "collapses" to a sheet: its own line-of-sight depth
        # against the angular-diameter distance to a fiducial lens at z=0.5
        # (the same fiducial system ch10/ch19 use for sigma_v=250 km/s).
        "depth_over_distance_ratio": (7.94e-6, 0.5e-6),
    },
    note="Newtonian vs GR deflection at the solar limb (the Eddington 1919 test), "
    "and the depth/distance ratio behind the thin-lens approximation",
)
def ch16_deflection():
    from astropy import constants as const

    GM_over_c2_m = (const.G * const.M_sun / const.c ** 2).to_value(u.m)
    R_sun_m = const.R_sun.to_value(u.m)

    alpha_newton_rad = 2.0 * GM_over_c2_m / R_sun_m   # Soldner 1801's estimate
    alpha_gr_rad = 4.0 * GM_over_c2_m / R_sun_m       # Einstein 1915, full GR

    v = dict(
        GM_over_c2_m=GM_over_c2_m,
        R_sun_m=R_sun_m,
        schwarzschild_radius_sun_km=2.0 * GM_over_c2_m / 1000.0,
        alpha_newton_rad=alpha_newton_rad,
        alpha_gr_rad=alpha_gr_rad,
        alpha_newton_arcsec=alpha_newton_rad * L.ARCSEC_PER_RAD,
        alpha_gr_arcsec=alpha_gr_rad * L.ARCSEC_PER_RAD,
        gr_over_newton_ratio=alpha_gr_rad / alpha_newton_rad,
    )

    # Dyson, Eddington & Davidson (1920), Phil. Trans. R. Soc. A 220, 291-333:
    # the two 1919 eclipse-expedition results, kept as literal historical
    # record (like ch13's z_cikota_lens), not derived from anything above.
    v["eddington_sobral_arcsec"] = 1.98
    v["eddington_sobral_err"] = 0.16
    v["eddington_principe_arcsec"] = 1.61
    v["eddington_principe_err"] = 0.40

    # Thin-lens plausibility check: a giant elliptical's own line-of-sight
    # depth (~10 kpc, order-of-magnitude only) against the angular-diameter
    # distance to the SAME fiducial lens plane ch10/ch19 use (z_l=0.5).
    depth_kpc = 10.0
    D_d_mpc = cosmo.d_a(0.5)
    v["depth_kpc"] = depth_kpc
    v["D_d_fiducial_mpc"] = D_d_mpc
    v["depth_over_distance_ratio"] = (depth_kpc / 1000.0) / D_d_mpc
    return v


# --------------------------------------------------------------------------- #
# Part IV — the lens equation (ch17): the SIS root-find behind figures.py's
# lens_equation_1d, reproduced here two ways -- closed form and a numerical
# root-finder on the SAME L.sis_deflection the rest of the guide differentiates
# -- plus the SIS's doubles-vs-singles multiplicity rule.
# --------------------------------------------------------------------------- #
@example(
    "ch17",
    expect={
        # figures.json "ch17-lens-equation".worked_values: the SAME figure,
        # computed independently here by root-finding rather than algebra.
        "image_1": (1.4, 1e-6),
        "image_2": (-0.6, 1e-6),
        "separation": (2.0, 1e-6),
        # scipy.brentq, bracketing each branch of L.sis_deflection, must land
        # on the closed-form roots to solver tolerance, not just "close."
        "root_diff_1": (0.0, 1e-8),
        "root_diff_2": (0.0, 1e-8),
        # Multiplicity: push the source outside the Einstein radius
        # (beta0=1.5 > theta_E=1.0). The major image persists; the minor
        # image's algebraic candidate lands at a NON-negative theta, which
        # contradicts the theta<0 assumption it was derived under -- so no
        # second root exists, and brentq on that bracket must fail to bracket
        # one (a genuine sign-change error, not a numerical near-miss).
        "single_image_theta": (2.5, 1e-6),
        "invalid_branch_candidate": (0.5, 1e-6),
        "second_root_correctly_absent": (1.0, 0.0),
    },
    note="The SIS lens equation beta=theta-alpha (Fig. 17.1), solved graphically "
    "then numerically, and the SIS's own doubles-vs-singles multiplicity rule.",
)
def ch17_lens_equation():
    from scipy.optimize import brentq

    theta_E = 1.0

    def g(theta, beta0):
        # beta(theta) - beta0, restricted to the x-axis (y=0), using the SAME
        # 2-D deflection field lensing.py exports -- not a reimplementation.
        ax, _ = L.sis_deflection(np.array([theta]), np.array([0.0]), theta_E)
        return float(theta - ax[0] - beta0)

    beta0 = 0.4  # matches figures.py's lens_equation_1d exactly
    image_1_analytic = beta0 + theta_E
    image_2_analytic = beta0 - theta_E
    image_1_numeric = brentq(g, 0.05, 5.0, args=(beta0,))
    image_2_numeric = brentq(g, -5.0, -0.05, args=(beta0,))

    v = dict(
        theta_E=theta_E,
        beta0=beta0,
        image_1=image_1_analytic,
        image_2=image_2_analytic,
        separation=abs(image_1_analytic - image_2_analytic),
        image_1_numeric=image_1_numeric,
        image_2_numeric=image_2_numeric,
        root_diff_1=abs(image_1_numeric - image_1_analytic),
        root_diff_2=abs(image_2_numeric - image_2_analytic),
    )

    # Multiplicity: push the source outside the Einstein radius.
    beta_outside = 1.5
    v["beta_outside"] = beta_outside
    v["single_image_theta"] = beta_outside + theta_E
    v["invalid_branch_candidate"] = beta_outside - theta_E  # >=0: violates theta<0
    try:
        brentq(g, -5.0, -0.05, args=(beta_outside,))
        v["second_root_correctly_absent"] = 0.0  # a root WAS found: unexpected
    except ValueError:
        v["second_root_correctly_absent"] = 1.0  # no sign change: correctly absent
    return v


# --------------------------------------------------------------------------- #
# Part V — the correlated-noise likelihood (ch24). Reads the P1a/P1c JSON
# artifacts directly (same honest strategy as ch07_fourier_whitening): these
# numbers come from real jax/scipy runs on data this guide script does not
# carry, but the reports those runs wrote are checked into the repo.
# --------------------------------------------------------------------------- #
@example(
    "ch24",
    expect={
        # data/noise_kernel_report.json: two-component kernel gate (<=0.05)
        # PASSES on all three real products; the pre-registered single-Gaussian
        # family FAILS on all three (main.tex:384-397).
        "kernel_two_component_v2d": (0.044783, 0.0005),
        "kernel_two_component_v3": (0.027006, 0.0005),
        "kernel_two_component_v3b": (0.032634, 0.0005),
        "kernel_single_family_v2d": (0.089539, 0.0005),
        "kernel_single_family_v3": (0.311015, 0.0005),
        "kernel_single_family_v3b": (0.166392, 0.0005),
        "w_sum_v3b": (0.981648, 0.0005),
        "blocksum_crosscheck_diff": (0.030783, 0.0005),
        # data/whitener_report.json: the operator-norm gate e_op<=0.02, all
        # four accepted whiteners (Table tab:whiten).
        "eop_v2d": (0.017672, 0.0005),
        "eop_v3": (0.016013, 0.0005),
        "eop_v3b": (0.012425, 0.0005),
        "eop_v2d_relaxed": (0.031196, 0.0005),
        "n_eroded_vs_strict_v2d": (3.0103, 0.001),
        "pixel_loss_frac_v2d_strict": (0.916965, 0.0005),
        # data/parity_report.json gate D: the delta-kernel conv-whitener vs the
        # diagonal path, worst of 4 points -- EXACT, not merely small.
        "gate_D_achieved": (0.0, 1e-12),
        "gate_D_threshold": (1e-10, 1e-14),
        # data/exact_ref_report.json: the SEPARATE dense-covariance Cholesky
        # cross-check (a real, non-delta kernel), and the dropped log|C|
        # constant's Szego-vs-exact gap (main.tex:476-479,494-495).
        "dense_c_worst_dlogl_v2d": (2.794e-9, 1e-11),
        "dense_c_worst_dlogl_v3b": (6.258e-7, 1e-9),
        "szego_gap_v2d": (27.297, 0.01),
        "szego_gap_v3b": (179.213, 0.01),
        # The toy re-derivation of gate D: a 1x1 delta-kernel conv whitener,
        # built by hand in plain numpy, must match the diagonal path bit for
        # bit -- not approximately, exactly, because convolving by a single
        # tap IS elementwise multiplication.
        "toy_delta_kernel_matches_diagonal": (0.0, 1e-12),
        # main.tex:504-512 / CAMPAIGN.md tab:repro #7: the gate-D-adjacent
        # implementation defect (28 per-column convs livelock XLA under grad)
        # and its fix (one grouped/depthwise conv).
        "grouped_conv_compile_s": (13.8, 0.01),
        "conv_fwd_over_diag_ratio": (1.6, 0.001),
        "conv_grad_over_diag_ratio": (1.545, 0.001),
    },
    note="C = D^1/2 K D^1/2: the kernel-fit and whitening gates, and gate D "
    "(the diagonal limit) as both an algebraic identity and a regression test "
    "that caught a real XLA-compile defect",
)
def ch24_correlated_noise():
    root = HERE.parent.parent
    data = root / "reproductions" / "claude-giga-lens" / "data"
    nk = json.loads((data / "noise_kernel_report.json").read_text())
    wr = json.loads((data / "whitener_report.json").read_text())
    xr = json.loads((data / "exact_ref_report.json").read_text())
    pr = json.loads((data / "parity_report.json").read_text())

    v = {}

    # --- the covariance model / fitting the kernel: two-component vs the
    # pre-registered single-Gaussian family, on all three real products ------
    for tag in ("v2d", "v3", "v3b"):
        prod = nk["products"][tag]
        v[f"kernel_two_component_{tag}"] = prod["max_abs_resid"]
        v[f"kernel_two_component_pass_{tag}"] = prod["gate_le_0p05"]
        v[f"kernel_single_family_{tag}"] = prod["fit_single_family"]["max_abs_resid"]
        v[f"kernel_single_family_pass_{tag}"] = prod["fit_single_family"]["gate_le_0p05"]
    v["kernel_gate"] = 0.05
    v3b = nk["products"]["v3b"]
    v["w_d_v3b"] = v3b["w_d"]
    v["w_b_v3b"] = v3b["w_b"]
    v["w_sum_v3b"] = v3b["w_d"] + v3b["w_b"]  # PSD needs this <= 1
    v["blocksum_crosscheck_diff"] = nk["blocksum_crosscheck"]["max_abs_diff"]
    v["blocksum_gate"] = nk["blocksum_crosscheck"]["gate"]

    # --- convolutional whitening: the accepted-whitener table ---------------
    for tag in ("v2d", "v3", "v3b", "v2d_relaxed"):
        prod = wr["products"][tag]
        v[f"eop_{tag}"] = prod["e_op"]
        v[f"M_{tag}"] = prod["M"]
        v[f"n_keep_{tag}"] = prod["n_keep"]
        v[f"n_eroded_{tag}"] = prod["n_eroded"]
    v["eop_gate"] = 0.02
    v["eop_relaxed_gate"] = 0.05
    v["pixel_loss_frac_v2d_strict"] = wr["products"]["v2d"]["pixel_loss_frac"]
    v["n_eroded_vs_strict_v2d"] = wr["products"]["v2d_relaxed"]["n_eroded_vs_strict"]

    # --- the diagonal limit: gate D, straight from the parity harness -------
    gate_D = pr["gates"]["D"]
    v["gate_D_achieved"] = gate_D["achieved"]
    v["gate_D_threshold"] = gate_D["threshold"]

    # --- the toy re-derivation: build the SAME 1x1 delta-kernel conv
    # whitener cgl/whiten.py's docstring describes ("Parity anchor (gate D)"),
    # in plain numpy on a tiny synthetic patch with one masked pixel, and show
    # it matches the diagonal sqrt(W) path bit for bit. h=[[1.0]] makes the
    # SAME-padded convolution degenerate: there is only one tap and no
    # neighbour to sum in, so "convolve by h" IS "multiply by h[0,0]".
    rng = np.random.default_rng(0)
    img = rng.normal(size=(5, 5))
    sigma = rng.uniform(0.5, 2.0, size=(5, 5))
    keep = np.ones((5, 5))
    keep[0, 0] = 0.0  # one masked pixel, to exercise keep_w too
    sqrt_d_inv = 1.0 / sigma
    h = np.array([[1.0]])                      # gate D's delta kernel
    u_conv = keep * (h[0, 0] * sqrt_d_inv * img)   # make_conv_whitener, h=[[1]]
    u_diag = keep * (img * sqrt_d_inv)             # the diagonal sqrt(W) path
    v["toy_delta_kernel_matches_diagonal"] = float(np.max(np.abs(u_conv - u_diag)))
    v["toy_kept_pixels"] = int(keep.sum())

    # --- the dense-covariance reference and the dropped log|C| constant -----
    v["dense_c_gate_nat"] = xr["gate_nats"]
    v["dense_c_worst_dlogl_v2d"] = xr["products"]["v2d"]["gate"]["worst_abs_dlogL"]
    v["dense_c_worst_dlogl_v3b"] = xr["products"]["v3b"]["gate"]["worst_abs_dlogL"]
    v["szego_gap_v2d"] = xr["products"]["v2d"]["constants"]["szego_gap"]
    v["szego_gap_v3b"] = xr["products"]["v3b"]["constants"]["szego_gap"]

    # --- the gate-D-adjacent regression: main.tex:504-512, cgl/e2.py:207-248 -
    v["grouped_conv_compile_s"] = 13.8
    v["diag_grad_compile_s"] = 14.0
    v["diag_fwd_ms"], v["diag_grad_ms"] = 25.0, 55.0
    v["conv_fwd_ms"], v["conv_grad_ms"] = 40.0, 85.0
    v["conv_fwd_over_diag_ratio"] = v["conv_fwd_ms"] / v["diag_fwd_ms"]
    v["conv_grad_over_diag_ratio"] = v["conv_grad_ms"] / v["diag_grad_ms"]
    return v


# --------------------------------------------------------------------------- #
# Part IV — the mass-sheet transformation (ch21). kappa_lam = lam*kappa +
# (1-lam) is checked two ways on the SAME SIS double Ch. 17 solves (theta_E=1,
# beta=0.4 -> images 1.4/-0.6): (1) the two images stay consistent with a
# SINGLE source under any lam (imaging cannot see lam), and (2) the
# Fermat-potential difference between them scales by exactly lam, so a fit
# that assumes lam=1 (no sheet) when the truth is lam=0.8 infers H0 too high
# by exactly 1/lam (imaging-blind, time-delay-visible). The source-centre
# R-hat/cluster numbers are transcribed, not recomputed, from CAMPAIGN.md:179
# / main.tex:914 (Ch. 26 owns the mechanism).
# --------------------------------------------------------------------------- #
@example(
    "ch21",
    expect={
        "beta_consistency_gap": (0.0, 1e-9),
        "beta_lambda_ratio": (0.8, 1e-9),
        "fermat_diff_ratio": (0.8, 1e-9),
        "h0_bias_factor": (1.25, 1e-9),
    },
    note="The mass-sheet transformation on the ch17 SIS double: image "
    "astrometry is exactly invariant under lambda_MST, the Fermat-potential "
    "difference scales by exactly lambda_MST, and an H0 fit that assumes no "
    "sheet is too high by exactly 1/lambda_MST.",
)
def ch21_mass_sheet_degeneracy():
    theta_E, beta0 = 1.0, 0.4              # same SIS as figures/ch17-lens-equation
    theta1, theta2 = beta0 + theta_E, beta0 - theta_E   # 1.4, -0.6
    lam = 0.8                              # a 20% sheet the images cannot see

    def psi_sis(theta, tE=theta_E):
        return tE * abs(theta)

    def alpha_sis(theta, tE=theta_E):
        return tE * (1.0 if theta > 0 else -1.0)

    def psi_mst(theta, lam=lam, tE=theta_E):
        return lam * psi_sis(theta, tE) + (1.0 - lam) * 0.5 * theta ** 2

    def alpha_mst(theta, lam=lam, tE=theta_E):
        return lam * alpha_sis(theta, tE) + (1.0 - lam) * theta

    # (1) beta_lam(theta) = theta - alpha_lam(theta), evaluated at the SAME two
    # true images. A genuine degeneracy means both still point to ONE source.
    beta_lam_1 = theta1 - alpha_mst(theta1)
    beta_lam_2 = theta2 - alpha_mst(theta2)
    beta_consistency_gap = abs(beta_lam_1 - beta_lam_2)
    beta_lambda_ratio = beta_lam_1 / beta0

    # (2) Fermat potential tau(theta;beta) = 0.5*(theta-beta)^2 - psi(theta);
    # the transformed model's own implied source position is lam*beta0.
    def tau(theta, beta, psi_fn):
        return 0.5 * (theta - beta) ** 2 - psi_fn(theta)

    dtau_true = tau(theta1, beta0, psi_sis) - tau(theta2, beta0, psi_sis)
    dtau_lam = (tau(theta1, lam * beta0, psi_mst)
                - tau(theta2, lam * beta0, psi_mst))
    fermat_diff_ratio = dtau_lam / dtau_true

    v = dict(
        theta_E=theta_E, beta0=beta0, theta1=theta1, theta2=theta2, lam=lam,
        beta_lam_1=beta_lam_1, beta_lam_2=beta_lam_2,
        beta_consistency_gap=beta_consistency_gap,
        beta_lambda_ratio=beta_lambda_ratio,
        dtau_true=dtau_true, dtau_lam=dtau_lam,
        fermat_diff_ratio=fermat_diff_ratio,
        # D_dt ~ 1/H0 and Delta-t_obs is fixed data, so a model that ASSUMES
        # lam=1 (no sheet) when the truth is lam=0.8 needs D_dt_assumed =
        # D_dt_true * lam to match the same Delta-t_obs (Delta-t_obs =
        # D_dt/c * dtau, and dtau_assumed = dtau_true/lam is LARGER in
        # magnitude, so D_dt_assumed must be SMALLER) -> H0_assumed =
        # H0_true / lam: a 20% unmodeled sheet gives a 25% high H0.
        h0_bias_factor=1.0 / fermat_diff_ratio,
        # --- the ONE real source-vs-mass ambiguity this campaign's flexible
        # shapelet source exhibits: quoted, not recomputed, and explicitly
        # decoupled from gamma (CAMPAIGN.md:179; main.tex:911-916).
        rhat_source_centre_hi=22.3,
        rhat_source_centre_lo=15.1,
        source_centre_cluster_neg=-0.15,
        source_centre_cluster_pos=0.09,
    )
    return v


# --------------------------------------------------------------------------- #
# Part IV — profiles (ch20): SIS -> SIE -> EPL, the ellipticity half-angle's
# spin-2 signature, external shear recovered as a pure zero-convergence
# Jacobian contribution (the SAME decomposition ch05 defines), and the
# shapelet source-basis count. All from site/guide_src/lensing.py + cgl/e2.py.
# --------------------------------------------------------------------------- #
@example(
    "ch20",
    expect={
        # Local kappa AT theta_E is (3-gamma)/2 -- NOT the Ch.19 mean-interior
        # identity, which is always 1 regardless of gamma. At gamma=1 the two
        # coincide (kappa=1 EVERYWHERE: the mass-sheet-critical uniform sheet),
        # which is the geometric reason the report's prior wall sits at 1.0.
        "kappa_theta_e_wall": (1.0, 1e-9),
        "kappa_theta_e_money": (0.9485, 1e-9),
        "kappa_theta_e_anchor": (0.7835, 1e-9),
        "kappa_theta_e_isothermal": (0.5, 1e-9),
        "kappa_theta_e_artifact": (0.2075, 1e-9),
        "mean_kappa_money": (1.0, 1e-6),
        "mean_kappa_artifact": (0.9938, 0.001),
        "kappa_theta_e_money_gap_to_wall": (0.0515, 1e-9),
        # (e1, e2) = (0.3, 0.4): a 3-4-5 triangle, picked for clean numbers --
        # NOT a typical draw from cgl/e2.py:111's e1, e2 ~ Normal(0, 0.1).
        "ellip_q": (1.0 / 3.0, 1e-9),
        "ellip_phi_deg": (26.565051, 1e-4),
        "spin2_e1_at_phi_plus_pi": (0.3, 1e-9),
        "spin2_e1_at_phi_plus_halfpi": (-0.3, 1e-9),
        # External shear: differentiating shear_deflection ALONE recovers
        # exactly (kappa=0, gamma1, gamma2) -- a Taylor argument confirmed by
        # exact finite differences (the deflection is linear, so there is no
        # truncation error at all).
        "ext_shear_kappa_recovered": (0.0, 1e-9),
        "ext_shear_gamma1_recovered": (0.03, 1e-6),
        "ext_shear_gamma2_recovered": (-0.02, 1e-6),
        "ext_shear_magnitude": (0.036056, 1e-5),
        "shapelet_depth_nmax6": (28, 1e-9),
        # SIE -> SIS as q -> 1 (lensing.py:65-78's docstring claim), checked
        # to a stated tolerance, not exact algebra.
        "sie_sis_diff_q0p9": (0.0569113789, 1e-6),
        "sie_sis_diff_q0p999": (0.000639249781, 1e-6),
    },
    note="SIS -> SIE -> EPL; kappa(theta_E) = (3-gamma)/2 derived from Ch.19's "
    "mean-kappa=1 definition; the ellipticity half-angle's spin-2 signature; "
    "external shear as a constant, zero-convergence Jacobian contribution; "
    "the shapelet source basis's amplitude count.",
)
def ch20_profiles():
    v = {}
    theta_E = 1.0

    # --- EPL local convergence AT theta_E, across the actual gamma bracket
    # this campaign reported for one galaxy (DESI-165.4754-06.0423): the
    # money number, the anchor, the isothermal reference, and the fine-
    # product artifact (main.tex Table tab:crossscale; CAMPAIGN.md:134).
    # kappa(theta_E) = (3-gamma)/2 exactly, from L.epl_kappa at R = theta_E.
    gammas = dict(wall=1.0, money=1.103, anchor=1.433,
                  isothermal=2.0, artifact=2.585)
    for tag, g in gammas.items():
        v[f"kappa_theta_e_{tag}"] = float(
            L.epl_kappa(np.array([theta_E]), np.array([0.0]), theta_E, g, 1.0)[0])

    # The Ch.19 mean-interior identity generalizes across the WHOLE gamma
    # family: kbar(theta_E) = 1 for ANY gamma, not just gamma=2. Checked at
    # the money value (well-behaved quadrature) and the steep artifact value
    # (where a linear grid undersamples the R->0 cusp once gamma>2 -- kept
    # for Exercise 20.4, not claimed as a physical result).
    v["mean_kappa_money"] = L.mean_kappa_within(
        lambda t: L.epl_kappa(t, np.zeros_like(t), theta_E, gammas["money"], 1.0),
        theta_E)
    v["mean_kappa_artifact"] = L.mean_kappa_within(
        lambda t: L.epl_kappa(t, np.zeros_like(t), theta_E, gammas["artifact"], 1.0),
        theta_E)
    # Distance from the money slope's local kappa(theta_E) to the fully
    # degenerate sheet's kappa=1 -- the sharper, structural companion to
    # Ch.10's "3.588 prior-sigma" framing of the same number.
    v["kappa_theta_e_money_gap_to_wall"] = (
        v["kappa_theta_e_wall"] - v["kappa_theta_e_money"])

    # --- Ellipticity: gigalens' epl.py/sie.py convention, exactly as
    # L.ellip_to_q_phi implements it. (0.3, 0.4) is a 3-4-5 triangle -- picked
    # for a clean half-angle, not a typical prior draw (see note above).
    e1, e2 = 0.3, 0.4
    q, phi = L.ellip_to_q_phi(e1, e2)
    v["ellip_q"] = q
    v["ellip_phi_rad"] = phi
    v["ellip_phi_deg"] = math.degrees(phi)
    c = (1.0 - q) / (1.0 + q)
    v["ellip_c"] = c
    # Spin-2 check: a FULL 180-degree rotation of the ellipse (phi -> phi+pi)
    # must reproduce the identical (e1, e2) -- an ellipse has no arrow. A
    # quarter turn (phi -> phi+pi/2) must NOT: it swaps the axes.
    v["spin2_e1_at_phi_plus_pi"] = c * math.cos(2 * (phi + math.pi))
    v["spin2_e2_at_phi_plus_pi"] = c * math.sin(2 * (phi + math.pi))
    v["spin2_e1_at_phi_plus_halfpi"] = c * math.cos(2 * (phi + math.pi / 2))
    v["spin2_e2_at_phi_plus_halfpi"] = c * math.sin(2 * (phi + math.pi / 2))

    # --- External shear: one plausible draw from cgl/e2.py:113-114's
    # gamma1, gamma2 ~ Normal(0, 0.05). Differentiating shear_deflection
    # ALONE, at an arbitrary image-plane point, must recover zero convergence
    # and exactly (gamma1, gamma2) as the Jacobian's OWN traceless part --
    # the same decomposition Ch.05's kappa_gamma_from_jacobian defines.
    g1_ext, g2_ext = 0.03, -0.02
    defl = lambda x, y: L.shear_deflection(x, y, g1_ext, g2_ext)  # noqa: E731
    x0, y0 = np.array([0.4]), np.array([0.7])
    a11, a12, a21, a22 = L.lens_jacobian(defl, x0, y0)
    kappa_ext, g1_rec, g2_rec = L.kappa_gamma_from_jacobian(a11, a12, a21, a22)
    v["ext_shear_kappa_recovered"] = float(kappa_ext[0])
    v["ext_shear_gamma1_recovered"] = float(g1_rec[0])
    v["ext_shear_gamma2_recovered"] = float(g2_rec[0])
    v["ext_shear_magnitude"] = math.hypot(g1_ext, g2_ext)

    # --- SIE -> SIS at q -> 1. lensing.py itself clips q at 0.99999 and
    # carries a nonzero core radius s=1e-4 (its own default), so the
    # reduction is checked to a stated tolerance, not exact algebra.
    x1, y1 = np.array([0.6]), np.array([0.8])   # r = 1 = theta_E
    sis_ax, sis_ay = L.sis_deflection(x1, y1, theta_E)
    for tag, qv in [("q0p9", 0.9), ("q0p999", 0.999), ("q0p999999", 0.999999)]:
        sie_ax, sie_ay = L.sie_deflection(x1, y1, theta_E, qv, 0.0)
        v[f"sie_sis_diff_{tag}"] = float(
            np.hypot(sie_ax - sis_ax, sie_ay - sis_ay)[0])

    # --- Source basis: Sersic + Shapelets(n_max=6), cgl/e2.py:120-131 /
    # vendor shapelets.py:18. Depth is a triangular number, the count of 2-D
    # Hermite basis functions up to total order n_max.
    n_max = 6
    v["shapelet_nmax"] = n_max
    v["shapelet_depth_nmax6"] = (n_max + 1) * (n_max + 2) // 2

    return v


# --------------------------------------------------------------------------- #
# Part V — the forward model's parameter budget (ch22). The 74 -> 46 reduction
# is arithmetic the reader is meant to redo by hand: sum one count per model
# component (cgl/likelihood.py:208-386, cgl/e2.py:12-21), then subtract the
# 28 shapelet amplitudes that cgl/marg.py marginalizes away analytically.
# shapelet_depth_nmax6 itself is NOT recomputed here -- it is ch20's number,
# reused (cross-chapter, same convention as ch08's Occam-term numbers below).
# --------------------------------------------------------------------------- #
@example(
    "ch22",
    expect={
        "n_params_full": (74, 0),
        "n_params_marg": (46, 0),
        "param_count_check": (0, 0),
        "marg_count_check": (0, 0),
        "n_pix_supersampled_v3b": (260, 0),
        "occam_correction_production_nats": (161.6145, 0.001),
    },
    note="The 74-parameter forward model and the 46-parameter arithmetic of "
    "marginalizing its 28 shapelet amplitudes away (cgl/likelihood.py:208-386, "
    "cgl/e2.py:12-21); the render grid for the v3b (binned) product; and what "
    "the audited production Occam term (Ch. 8's parity number) costs in nats "
    "if you drop it, as gigalens's own lstsq path does.",
)
def ch22_forward_model():
    # One count per model component, cgl/likelihood.py:87-134 (_build_prior)
    # and :299-315 (the two PhysicalModel instantiations).
    n_mass = 6              # EPL: theta_E, gamma, e1, e2, center_x, center_y
    n_shear = 2              # external shear: gamma1, gamma2
    n_lens_light_each = 7    # SersicEllipse: R_sersic, n_sersic, e1, e2, cx, cy, Ie
    n_lens_light_components = 4
    n_source_sersic = 7      # the same 7-parameter Sersic, once, for the source
    n_shapelet_pose = 3      # beta, center_x, center_y -- always sampled
    n_shapelet_amps_n6 = 28  # sampled ONLY in the 74-dim model; marginalized
                              # away (not sampled at all) in the 46-dim one

    n_lens_light = n_lens_light_each * n_lens_light_components
    n_mass_shear = n_mass + n_shear
    n_params_full = (n_mass_shear + n_lens_light + n_source_sersic
                      + n_shapelet_pose + n_shapelet_amps_n6)
    n_params_marg = n_params_full - n_shapelet_amps_n6

    # The v3b (binned) product's render grid, cgl/e2.py:47-54: delta_pix is
    # the DATA pixel scale; supersample sets how many sub-pixels per data
    # pixel the lens equation and PSF convolution are evaluated at, before
    # average_pool_2d bins back down (cgl/likelihood.py:329-352).
    delta_pix_v3b, supersample_v3b, n_pix_v3b = 0.08, 2, 130

    # What gigalens's own plain-lstsq path (no Occam term) would over-report
    # at the SAME audited production point Ch. 8 quotes -- half of the real
    # logdetA, not the toy one. (Reuses ch08's own number; not recomputed.)
    occam_logdetA_production = 323.229  # ch08.occam_logdetA_parity

    return dict(
        n_mass=n_mass, n_shear=n_shear, n_mass_shear=n_mass_shear,
        n_lens_light_each=n_lens_light_each,
        n_lens_light_components=n_lens_light_components,
        n_lens_light=n_lens_light,
        n_source_sersic=n_source_sersic,
        n_shapelet_pose=n_shapelet_pose,
        n_shapelet_amps_n6=n_shapelet_amps_n6,
        n_params_full=n_params_full,
        n_params_marg=n_params_marg,
        param_count_check=n_params_full - 74,
        marg_count_check=n_params_marg - 46,
        delta_pix_v3b=delta_pix_v3b,
        supersample_v3b=supersample_v3b,
        n_pix_v3b=n_pix_v3b,
        n_pix_supersampled_v3b=n_pix_v3b * supersample_v3b,
        occam_logdetA_production=occam_logdetA_production,
        occam_correction_production_nats=0.5 * occam_logdetA_production,
    )


# --------------------------------------------------------------------------- #
# Part VI — Discovery (ch27): the DESI Legacy survey's own numbers, a
# closed-form resolution wall derived from the PSF as a Gaussian convolution
# kernel (Ch. 7/11), and the repo's own confirmation that neither bigger
# architectures nor bare AUC move the needle at 53.8M-galaxy scale.
# --------------------------------------------------------------------------- #
@example(
    "ch27",
    expect={
        # tools/desi-dr11-cookbook/README.md:41; huang-2020/papers/main.tex:138
        "cutout_side_arcsec": (26.462, 0.001),
        # dr11-campaign/papers/main.tex:93 (exact parent count after the
        # published Inchausti cuts, applied to the full DR11-south footprint)
        "n_parent_galaxies": (53_809_040, 0),
        "candidates_at_1pct_fpr": (538_090.4, 1.0),
        "candidates_at_0_1pct_fpr": (53_809.04, 0.1),
        "candidates_at_0_01pct_fpr": (5_380.904, 0.01),
        # The Sparrow-criterion resolution floor derived in Ch. 27, at the
        # Legacy g-band coadd's own measured seeing (cikota-2023/README.md:38).
        "theta_e_wall_arcsec": (0.5733, 0.001),
        # Same derivation, at Euclid VIS's much sharper PSF (lensjudge/
        # papers/main.tex:565,589 quote its cutouts as 0.1'' resolution).
        "theta_e_wall_euclid_arcsec": (0.04247, 0.0001),
        "wall_ratio_desi_over_euclid": (13.5, 0.01),
        "typical_over_euclid_wall": (26.9695, 0.01),
        "ring_diameter_over_wall_typical": (1.9977, 0.001),
        "ring_diameter_over_wall_cikota": (3.8954, 0.001),
        # cikota-2023's REAL imaging fit (2.103'', not Ch. 19's illustrative
        # theta_e_cikota) against the SAME wall.
        "ring_diameter_over_wall_cikota_imaging": (3.6683, 0.001),
        # inchausti-2025 params table (papers/main.tex:172-174); the ratio
        # claudenet/README.md:12-13 rounds to "105x" / "within +/-0.003 AUC".
        "param_ratio_effnet_over_shielded194k": (105.6197, 0.001),
        "auc_gap_resnet_effnet_paper": (0.0003, 1e-6),
        "auc_gap_resnet_meta_paper": (0.0005, 1e-6),
        # huang-2021/README.md:34-37: the SAME controlled-comparison move one
        # generation earlier (L18 vs the first 59,905-param shielded net).
        "auc_gap_l18_shielded60k": (0.0006, 1e-6),
        # dr11-campaign/papers/main.tex:124-125,134-138: SAME scores, SAME
        # AUC, only the combiner changes -- recall at a fixed budget swings
        # 54% -> 88% with zero retraining.
        "recall_union_95k": (0.54, 0.0),
        "recall_mean_95k": (0.75, 0.0),
        "recall_mean_150k": (0.80, 0.0),
        "recall_mean_heldout": (0.88, 0.0),
        "auc_mean_combiner": (0.9955, 0.0),
    },
    note="The survey's own numbers (grz, 0.262''/px, 101x101, 53.8M parent "
    "galaxies); a Gaussian-PSF second-derivative test for when two nearby "
    "images merge into one blob (deriving the resolution wall); confirmation "
    "from the finder lineage (huang-2020/21, inchausti-2025, claudenet) that "
    "a 105x parameter increase buys <0.001 AUC; and the DR11-south sweep's "
    "own union-vs-mean-combiner recall swing at fixed AUC.",
)
def ch27_discovery():
    v: dict = {}

    # --- The survey: DESI Legacy Imaging Surveys grz cutouts ---------------
    # tools/desi-dr11-cookbook/README.md:41 ("(101, 101, 3)"); huang-2020/
    # papers/main.tex:138 ("101x101 pixel grz, pixscale 0.262''/pix").
    pixel_scale_arcsec = 0.262
    cutout_side_px = 101
    v["pixel_scale_arcsec"] = pixel_scale_arcsec
    v["cutout_side_px"] = cutout_side_px
    v["cutout_side_arcsec"] = cutout_side_px * pixel_scale_arcsec

    # dr11-campaign/papers/main.tex:93: the parent sample after the published
    # Inchausti cuts (TYPE in {SER,EXP,DEV,REX}, NOBS_grz>=3, m_z<20), swept
    # across the full DR11-south footprint. Also quoted rounded (53.8M) in
    # dr11-campaign-v4/README.md:7 and in the same main.tex's own caption
    # (FPR x 53.8M, papers/main.tex:135).
    n_parent = 53_809_040
    v["n_parent_galaxies"] = n_parent
    v["candidates_at_1pct_fpr"] = n_parent * 0.01
    v["candidates_at_0_1pct_fpr"] = n_parent * 0.001
    v["candidates_at_0_01pct_fpr"] = n_parent * 0.0001

    # Legacy DR10 g-band coadd PSF, the seeing that actually sits on this
    # survey's pixels (cikota-2023/README.md:38, FWHM ~= 1.35"; lensjudge/
    # papers/main.tex:572 rounds the same quantity to "1.3\arcsec" in prose).
    seeing_fwhm_arcsec = 1.35
    v["seeing_fwhm_arcsec"] = seeing_fwhm_arcsec
    v["fwhm_in_pixels"] = seeing_fwhm_arcsec / pixel_scale_arcsec

    # --- Deriving the wall: two point sources under a Gaussian PSF ---------
    # f(x) = G_sigma(x-a) + G_sigma(x+a); f''(0) changes sign at a = sigma,
    # i.e. separation d = 2a = 2*sigma is where the single central hump
    # splits into two -- the Sparrow resolution limit, derived (not looked
    # up) from FWHM = 2*sqrt(2 ln 2)*sigma.
    fwhm_to_sigma = 2.0 * math.sqrt(2.0 * math.log(2.0))
    v["fwhm_to_sigma_factor"] = fwhm_to_sigma
    sigma = seeing_fwhm_arcsec / fwhm_to_sigma
    v["sigma_arcsec"] = sigma
    v["d_min_arcsec"] = 2.0 * sigma          # minimum resolvable separation
    v["theta_e_wall_arcsec"] = sigma          # = d_min / 2, in Einstein-radius terms

    # The identical derivation at Euclid VIS's much sharper PSF (lensjudge/
    # papers/main.tex:565,589: "0.1 arcsec" cutouts) -- Exercise 27.2.
    seeing_fwhm_euclid_arcsec = 0.1
    v["seeing_fwhm_euclid_arcsec"] = seeing_fwhm_euclid_arcsec
    v["theta_e_wall_euclid_arcsec"] = seeing_fwhm_euclid_arcsec / fwhm_to_sigma
    v["wall_ratio_desi_over_euclid"] = v["theta_e_wall_arcsec"] / v["theta_e_wall_euclid_arcsec"]

    # lensjudge/papers/main.tex:571-575: the direct empirical confirmation --
    # DESI grade-C candidates the Euclid Q1 panel independently re-observed,
    # regraded at 0.1'' instead of DESI's ~1.3'' seeing.
    v["lensjudge_gradeC_to_gradeA_num"] = 6
    v["lensjudge_gradeC_to_gradeA_den"] = 17
    v["lensjudge_gradeC_to_gradeA_frac"] = 6 / 17

    # cikota-2023/README.md:28-43, papers/main.tex:41-42,49,213: the SAME real
    # system's Einstein radius, fit three ways -- not the illustrative
    # theta_e_cikota Ch. 19 gets by exercising the sigma_v formula at an
    # arbitrary redshift pair, but the actual imaging fits.
    v["theta_e_cikota_imaging_fit"] = 2.103    # this repo, on the blended 1.35'' data
    v["theta_e_cikota_published"] = 2.520      # paper, on 0.6'' MUSE-derived imaging
    v["theta_e_cikota_ablation_06psf"] = 2.276  # this repo, SAME data, paper's 0.6'' PSF
    v["ring_diameter_over_wall_cikota_imaging"] = (
        2.0 * v["theta_e_cikota_imaging_fit"] / v["d_min_arcsec"]
    )

    # Confirm against this guide's own canonical theta_E values (Ch. 19):
    # theta_e_typical (sigma_v=250 km/s fiducial elliptical) and
    # theta_e_cikota (the repo's own real system, DESI-253.2534+26.8843).
    theta_e_typical = cosmo.theta_e_from_sigma_v(250.0, 0.5, 2.0)
    theta_e_cikota = cosmo.theta_e_from_sigma_v(347.0, 0.271, 0.897)
    v["theta_e_typical_arcsec"] = theta_e_typical
    v["theta_e_cikota_arcsec"] = theta_e_cikota
    v["ring_diameter_over_wall_typical"] = (2.0 * theta_e_typical) / v["d_min_arcsec"]
    v["ring_diameter_over_wall_cikota"] = (2.0 * theta_e_cikota) / v["d_min_arcsec"]
    v["typical_over_euclid_wall"] = theta_e_typical / v["theta_e_wall_euclid_arcsec"]

    # --- The finders: parameter count vs AUC, across the whole lineage -----
    # huang-2020/README.md (Lanusse L18); huang-2021/README.md:28-30 (shielded
    # 60K); inchausti-2025/README.md:44-53 + papers/main.tex:170-179 (the
    # 194K/EfficientNetV2-S/meta table); claudenet/README.md:9-13 rounds the
    # last comparison to "105x" / "within +/-0.003 AUC".
    params_l18 = 3_508_833
    params_shielded_60k = 59_905
    params_shielded_194k = 194_501
    params_effnet = 20_543_145
    v["params_l18"] = params_l18
    v["params_shielded_60k"] = params_shielded_60k
    v["params_shielded_194k"] = params_shielded_194k
    v["params_effnet"] = params_effnet
    v["param_ratio_l18_over_60k"] = params_l18 / params_shielded_60k
    v["param_ratio_effnet_over_shielded194k"] = params_effnet / params_shielded_194k

    # huang-2021/README.md's own controlled DR9 table (L18 vs shielded-60K,
    # identical cutouts/positives/negatives/seed/split -- architecture is the
    # only variable): val AUC 0.9983 (L18) vs 0.9989 (shielded), i.e. the 60K
    # net actually edges the 3.5M-param one at 58.6x fewer parameters.
    auc_l18_dr9_val, auc_shielded60k_dr9_val = 0.9983, 0.9989
    v["auc_l18_dr9_val"] = auc_l18_dr9_val
    v["auc_shielded60k_dr9_val"] = auc_shielded60k_dr9_val
    v["auc_gap_l18_shielded60k"] = auc_shielded60k_dr9_val - auc_l18_dr9_val

    # Inchausti Fig. 6 validation AUCs, reproduced exactly (papers/main.tex's
    # "paper (val)" column, tab:auc): shielded ResNet, EfficientNetV2-S, meta.
    auc_resnet_paper, auc_effnet_paper, auc_meta_paper = 0.9984, 0.9987, 0.9989
    v["auc_resnet_paper"] = auc_resnet_paper
    v["auc_effnet_paper"] = auc_effnet_paper
    v["auc_meta_paper"] = auc_meta_paper
    v["auc_gap_resnet_effnet_paper"] = auc_effnet_paper - auc_resnet_paper
    v["auc_gap_resnet_meta_paper"] = auc_meta_paper - auc_resnet_paper

    # --- The operating point: same scores, same AUC, different recall ------
    # dr11-campaign/papers/main.tex:98-107,124-138: the union-vs-mean combiner
    # swap at the SAME 95k survivor budget, with the AUC held fixed.
    v["n_survivor_budget_union"] = 95_104
    v["n_survivor_budget_mean"] = 150_000
    v["fpr_at_95k_budget"] = 95_104 / n_parent
    v["fpr_at_150k_budget"] = 150_000 / n_parent
    v["auc_mean_combiner"] = 0.9955
    v["recall_union_95k"] = 0.54
    v["recall_mean_95k"] = 0.75
    v["recall_mean_150k"] = 0.80
    v["recall_mean_heldout"] = 0.88

    # inchausti-2025/README.md:150-163: Stage B -> C -> D, recovery @ matched
    # 1% FPR, as the negative-sample composition (not the architecture) is
    # fixed. AUC moves by <0.01 across all three stages.
    v["recovery_storfer_stageB_1pct"] = 0.118
    v["recovery_inchausti_stageB_1pct"] = 0.191
    v["recovery_storfer_stageC_1pct"] = 0.836
    v["recovery_inchausti_stageC_1pct"] = 0.885
    v["recovery_storfer_stageD_1pct"] = 0.908
    v["recovery_inchausti_stageD_1pct"] = 0.968
    v["auc_meta_stageC"] = 0.9876
    v["auc_meta_stageD"] = 0.9919
    return v


# --------------------------------------------------------------------------- #
# Part V — the sampler saga (ch23): the mass-matrix convention trap (M =
# Sigma^-1) worked by hand on a toy anisotropic covariance; the classic
# Gelman-Rubin R-hat, worked by hand on two chains that deliberately never
# overlap; the N_eff/N -> ESS/N reduction ch07 held back for this chapter
# (ch07_fourier_whitening's own comment: "the same ratio Ch. 23 will call
# ESS/N for a Markov chain"); and the real R-hat/ESS numbers behind the
# money-number's own sampler chain (T2's metric fix, the v3b-low saddle's
# metric failure, the 128-particle correlated SMC, NeuTra's collapse on the
# campaign's hardest real system) -- all literal quotes, traced to main.tex /
# CAMPAIGN.md line-by-line in the comments below, not re-derived.
# --------------------------------------------------------------------------- #
@example(
    "ch23",
    expect={
        "toy_mass_tight": (100.0, 1e-9),
        "toy_mass_loose": (0.01, 1e-9),
        "toy_cond": (1e4, 1e-6),
        "toy_rhat": (2.355844, 1e-5),
        "ess_from_tauint": (670.871, 0.01),
        "t2_rhat_twostage_after": (1.003, 1e-9),
    },
    note="The mass-matrix convention trap (M=Sigma^-1) and R-hat, both worked by "
    "hand on toy numbers; ESS via ch07's own N_eff/N reduction; and the real "
    "R-hat/ESS numbers behind the money number's sampler chain (T2's metric "
    "fix, the v3b-low saddle, the 128-particle correlated SMC, NeuTra's "
    "collapse on the hardest real T1 system).",
)
def ch23_sampler_saga():
    v = {}

    # --- the mass-matrix convention trap, by hand: a toy posterior covariance
    # with the same flavour of anisotropy Ch. 5 measured for real (cond ~ 1e4,
    # the same order as marg.py's own ridge-regularized normal matrix,
    # ch05.cond_marg_normal_matrix = 1.37e4) -- one loose direction, one tight
    # one. M = Sigma^-1 is what cgl/e2.py:657-658 calls "the GIGA-Lens
    # convention" and what TFP's PreconditionedHamiltonianMonteCarlo needs
    # passed as momentum_distribution's covariance_matrix (remc_pt.py:125,
    # baseline_gigalens.py -- both invert Sigma explicitly). blackjax's own
    # inverse_mass_matrix parameter, by contrast, wants Sigma UN-inverted
    # (bj_smc.py:72 "imm = ginit.cov_reg  # inverse mass = covariance";
    # bj_nuts.py:116) -- the same physical Sigma, opposite ends of a
    # reciprocal, depending which library's parameter you are filling in.
    Sigma = np.diag([100.0, 0.01])
    M = np.linalg.inv(Sigma)
    v["toy_sigma_loose"] = float(Sigma[0, 0])
    v["toy_sigma_tight"] = float(Sigma[1, 1])
    v["toy_cond"] = float(Sigma[0, 0] / Sigma[1, 1])
    v["toy_mass_loose"] = float(M[0, 0])
    v["toy_mass_tight"] = float(M[1, 1])

    # --- R-hat by hand: two chains that never overlap at all -- the classic
    # (Gelman & Rubin 1992) split-free formula, cgl/metrics.py:23-68 runs the
    # more careful rank-normalized SPLIT version (arviz.rhat(..., "rank")) but
    # the between/within-variance arithmetic is the same idea.
    chain1 = np.array([1.0, 2.0, 3.0, 4.0])
    chain2 = np.array([5.0, 6.0, 7.0, 8.0])
    chains = np.stack([chain1, chain2])            # (m=2 chains, n=4 draws)
    m, n = chains.shape
    chain_means = chains.mean(axis=1)
    grand_mean = chains.mean()
    W = chains.var(axis=1, ddof=1).mean()           # within-chain variance
    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)  # between-chain
    var_hat = (n - 1) / n * W + B / n
    v["toy_rhat_W"] = float(W)
    v["toy_rhat_B"] = float(B)
    v["toy_rhat_varhat"] = float(var_hat)
    v["toy_rhat"] = float(math.sqrt(var_hat / W))

    # --- ESS from an autocorrelation time: the SAME N_eff/N reduction ch07
    # ran on a stationary pixel sequence (worked_examples.py ch07_fourier_
    # whitening, tau_int_v3_11lag), applied here to a single MCMC chain
    # instead of a noise field -- ESS = N / tau_int.
    tau_int_ch07 = 7.453           # ch07.tau_int_v3_11lag, reused verbatim
    v["ess_from_tauint_n_draws"] = 5000
    v["ess_from_tauint"] = 5000 / tau_int_ch07

    # --- T2 (cond~1e14, ch05.cond_ill_conditioned): a POSITIVE-DEFINITE real
    # posterior that a better metric alone fixes. main.tex sec:p2c
    # (lines 1140-1152), Table tab:matrix.
    v["t2_rhat_singlestage"] = 3.10
    v["t2_ess_singlestage"] = 28
    v["t2_rhat_automass_mclmc"] = 5.9
    v["t2_ess_automass_mclmc"] = 9
    v["t2_rhat_twostage_before"] = 2.11
    v["t2_rhat_twostage_after"] = 1.003

    # --- the money-number's own v3b-low posterior is a SADDLE (ch05.
    # saddle_min_eig = -14.85, 5 negative of 46; ch25.rhat_saddle_metric =
    # 22.3 is the number Ch. 25 already quotes for "both repairs stall"): no
    # PD metric repair converges it. CAMPAIGN.md "P1c metric-fix attempts --
    # 2026-07-10" is more granular than Ch. 25's rounded pair -- diagraw and
    # svi_cov are two DIFFERENT repairs, landing at two different R-hats, and
    # ch25's 22.3 is specifically the svi_cov one, traced (main.tex:909-914)
    # to one of two decoupled source-centre sub-modes -- the other cluster's
    # R-hat (15.1) is new here, not quoted anywhere else in this book.
    v["saddle_rhat_diagraw"] = 21.0
    v["source_centre_rhat_b"] = 15.1

    # --- the fix: adaptive tempered SMC, metric-free at the OUTER loop.
    # Ch. 25 already registers the CORRELATED run's own particulars (128
    # particles, 28 lambda-steps, ESS 77-118: ch25.n_smc_particles,
    # ch25.n_smc_lambda_steps, ch25.ess_smc_low/high) -- reused directly in
    # this chapter's prose, not re-pinned here. New here: the DIAGONAL run
    # (Table tab:basinflip caption) used 300 particles, not 128 -- the OTHER
    # half of the 191-nat swing ch08/ch25 already quote used a different
    # particle count entirely.
    v["smc_n_particles_diagonal"] = 300

    # --- flows: NeuTra's own R-hat on t0_illcond46 -- the zoo's own MOCK
    # cond~1e14 target (main.tex line 1016-1025), the same flavour of
    # ill-conditioning ch05/T2 above face on real data -- versus its median
    # across the WHOLE benchmark (Table tab:deveval, main.tex line 1085): an
    # aggregate that hides a single bad failure.
    v["neutra_rhat_illcond46"] = 5.72
    v["neutra_eval_rhat_median"] = 1.160

    # --- why gradients alone are not sufficient: PT-HMC's until-converged
    # reliability versus nautilus's budget-matched efficiency, opposite
    # corners of the same benchmark. main.tex lines 989-991, 1019-1020.
    v["pthmc_hard_converged"] = 5
    v["pthmc_hard_total"] = 6
    v["nautilus_essgrad_lo"] = 2.6
    v["nautilus_essgrad_hi"] = 307.0

    return v


# --------------------------------------------------------------------------- #
# Part V — the saddle's own contrast case (ch26). ch05/ch21/ch25 already carry
# every number this chapter's narrative reuses (the saddle spectrum, the
# source-centre R-hat/clusters, the SMC particulars); the two facts genuinely
# new here are the ONE positive-definite basin in the same campaign (the
# contrast that turns "convergence tracks Hessian definiteness" into a rule
# rather than a one-off anecdote) and the inflation factor the metric-free SMC
# path applies to its own empirical reference covariance.
# --------------------------------------------------------------------------- #
@example(
    "ch26",
    expect={
        "fine_steep_min_eig": (0.108, 0.005),
        "fine_steep_rhat": (1.03, 0.01),
        "smc_cov_inflate": (3.0, 0.0),
    },
    note="The fine-steep basin's POSITIVE-definite Laplace Hessian -- the "
    "contrast case that makes convergence-tracks-definiteness a rule, not a "
    "one-off -- and the metric-free SMC path's reference-covariance inflation.",
)
def ch26_saddle_contrast():
    # main.tex sec:samplersaga: "the fine steep basin (min eigenvalue +0.11,
    # positive-definite) mixes cleanly under PHMC (Rhat=1.03)". CAMPAIGN.md's
    # more precise figure ("P1c v3b MONEY + all products") is +0.108, 0/46
    # eigenvalues floored (is_pd=True) -- the ONE basin this campaign's own
    # Laplace check certified PD, out of every real-data product it tried.
    v = dict(
        fine_steep_min_eig=0.108,
        fine_steep_n_floored=0,
        fine_steep_ndim=46,
        fine_steep_rhat=1.03,
        # Restated for direct, same-function contrast with the saddle Ch. 5
        # first found (ch05.saddle_min_eig / ch05.saddle_n_negative).
        saddle_min_eig=-14.85,
        saddle_n_negative=5,
        # cgl/e2.py:732 fit_gaussian_from_draws / cgl/e2.py:756
        # run_correlated_smc: the "metric-free" SMC path's reference
        # covariance is the POOLED covariance of the earlier (individually
        # unconverged) HMC chains, inflated by this factor before it seeds
        # the 128 SMC particles.
        smc_cov_inflate=3.0,
    )
    return v


# --------------------------------------------------------------------------- #
# Part VII — synthesis (ch29). Pinned constants from the sampler-benchmark
# abstract and the P1c SMC out-of-memory / multi-day agent stall (main.tex
# abstract, Sec. 7; CAMPAIGN.md "P1c SMC OOM + multi-day stall") that no
# earlier chapter's worked example already carries. Transcribed from the
# report and the ledger, not recomputed -- same discipline as ch16's
# Eddington constants.
# --------------------------------------------------------------------------- #
@example(
    "ch29",
    expect={
        "nautilus_ess_ratio_min": (2.6, 0.01),
        "nautilus_ess_ratio_max": (307.0, 0.1),
        "glnt_ess_ratio_min": (0.03, 0.001),
        "glnt_ess_ratio_max": (0.05, 0.001),
        "glnt_target_ratio": (3.0, 0.01),
        "smc_oom_attempted_gb": (120.4, 0.05),
        "smc_particles_oom": (300, 0),
        "smc_particles_fixed": (128, 0),
        "impl_agent_stall_days": (4, 0),
    },
    note="Sampler-benchmark efficiency spread (nautilus vs. GL-NT) and the "
    "P1c implementation-agent stall, cited in Ch. 29's synthesis discussion.",
)
def ch29_synthesis_constants():
    # main.tex abstract: nautilus's ESS/gradient advantage over the GIGA-Lens
    # baseline, and GL-NT's shortfall against its own pre-registered bar.
    return dict(
        nautilus_ess_ratio_min=2.6,
        nautilus_ess_ratio_max=307.0,
        glnt_ess_ratio_min=0.03,
        glnt_ess_ratio_max=0.05,
        glnt_target_ratio=3.0,
        # CAMPAIGN.md "P1c SMC OOM + multi-day stall — 2026-07-13/14": the
        # canary that OOM'd, and the fresh agent's fix that finally worked.
        smc_oom_attempted_gb=120.4,
        smc_particles_oom=300,
        smc_particles_fixed=128,
        # "stalled 07-10->14" -- inclusive span, four calendar days.
        impl_agent_stall_days=4,
    )
