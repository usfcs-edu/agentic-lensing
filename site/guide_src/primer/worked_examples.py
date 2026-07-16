#!/usr/bin/env python
"""Every worked number in the Beginner's Guide, computed here and nowhere else.

    ~/.venvs/lensjudge/bin/python site/guide_src/make_examples.py --guide primer --check

Same rule as the main guide: a number may appear in the prose only if a function
here returns it, tagged `<!-- check: pch08.z2_lookback_gyr = 10.240 ± 0.01 -->`.

Imports the SHARED `cosmo`/`lensing` modules, not copies. A primer that computed
its own Sigma_crit could disagree with the guide's; sharing makes disagreement
impossible rather than unlikely.

That is not hypothetical. On first run this gate caught three numbers written
from memory, all wrong in instructive ways:

  * the H-alpha wavelength quoted as 656.28 nm (the AIR value) against a formula
    that returns 656.470 (VACUUM) — the 0.03% gap is the refractive index of air;
  * the z=2 lookback time quoted as 10.51 Gyr, which is Planck's (67.4, 0.315).
    This repository asserts FlatLambdaCDM(70, 0.3) and gets 10.24. The primer
    must agree with the guide, not with a textbook;
  * a stellar density lifted from "~0.1 per cubic parsec near the Sun" and
    applied to a uniform-disk average, which is 4x higher.

None of those would have been caught by reading.

Keys are `pchNN` — the `p` is not namespacing (guides.py gives each book its own
module and imports exactly one per process, so `ch08` would be safe). It is for
the READER: these tags are visible in the markdown source, and `pch08` says
"primer chapter 8" at a glance.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cosmo  # noqa: E402
import numpy as np  # noqa: E402
from examples_registry import example  # noqa: E402

# --- constants the book quotes, all standard --------------------------------
C_KM_S = 299792.458
AU_KM = 1.495978707e8
LY_KM = 9.4607304725808e12
PC_LY = 3.261563777
MW_STARS = 1e11           # order of magnitude
MW_DIAM_LY = 1.0e5
MW_MASS = 1.5e12          # Msun, incl. dark halo


# --------------------------------------------------------------------------- #
# Part I — Where are we?
# --------------------------------------------------------------------------- #
@example(
    "pch01",
    expect={"au_light_minutes": (8.317, 0.01), "pc_in_ly": (3.2616, 0.001)},
    note="The scale ladder: AU, light-year, parsec, and the Milky Way as ruler",
)
def pch01_scale_ladder():
    v = dict(
        au_km=AU_KM,
        # Light takes this long to reach us from the Sun — the first rung where
        # "distance is time" stops being an abstraction.
        au_light_minutes=AU_KM / C_KM_S / 60.0,
        ly_km=LY_KM,
        pc_in_ly=PC_LY,
        # Proxima Centauri, the nearest star: 4.24 ly. Compare the AU.
        nearest_star_ly=4.24,
        nearest_star_in_au=4.24 * LY_KM / AU_KM,
        mw_diameter_ly=MW_DIAM_LY,
        mw_stars=MW_STARS,
        # Crossing our own galaxy at light speed, in years.
        mw_crossing_yr=MW_DIAM_LY,
        # The observable universe (Ch. 13's Hubble distance) in Milky Ways.
        hubble_distance_mpc=C_KM_S / 70.0,
    )
    v["hubble_distance_in_mw"] = (v["hubble_distance_mpc"] * 1e6 * PC_LY) / MW_DIAM_LY
    return v


@example(
    "pch02",
    expect={"sun_lifetime_gyr": (10.0, 0.5)},
    note="Stellar lifetimes: why massive stars die fast (t ~ M/L, L ~ M^3.5)",
)
def pch02_stars():
    # Main-sequence lifetime scales as fuel/burn-rate = M/L, and L ~ M^3.5,
    # so t ~ M^-2.5. This one exponent is why "old and red" identifies a lens.
    def lifetime_gyr(mass_msun):
        return 10.0 * mass_msun ** -2.5

    # Remnants: which side of the ~8 Msun line a star started on decides
    # whether it ends as a white dwarf or a core-collapse compact object.
    # Sizes/masses are standard textbook benchmarks, not repo-derived; the
    # density comparison is computed here so it is never quoted from memory.
    m_sun_kg = 1.989e30
    earth_radius_km = 6371.0
    wd_mass_msun, wd_radius_km = 0.6, earth_radius_km
    ns_mass_msun, ns_radius_km = 1.4, 11.0

    def density_tonne_per_cm3(mass_msun, radius_km):
        mass_kg = mass_msun * m_sun_kg
        radius_cm = radius_km * 1e5
        volume_cm3 = 4 / 3 * np.pi * radius_cm ** 3
        return (mass_kg / volume_cm3) / 1000.0  # kg/cm^3 -> tonne/cm^3

    # 13.47, not the textbook 13.8: this book quotes the age its OWN cosmology
    # gives, FlatLambdaCDM(70, 0.3), the same object every campaign in this
    # repository asserts. p02-main-sequence draws its "age of the universe"
    # line from this identical call, so the figure and the prose cannot drift.
    # See pch11, which reports the 13.47-vs-13.8 gap and explains it.
    import astropy.units as u
    universe_age_gyr = cosmo.COSMO.age(0).to_value(u.Gyr)

    wd_density = density_tonne_per_cm3(wd_mass_msun, wd_radius_km)
    ns_density = density_tonne_per_cm3(ns_mass_msun, ns_radius_km)
    return dict(
        sun_lifetime_gyr=lifetime_gyr(1.0),
        star_10msun_lifetime_myr=lifetime_gyr(10.0) * 1000.0,
        star_half_msun_lifetime_gyr=lifetime_gyr(0.5),
        # A 10-Msun star lives this fraction of the Sun's life.
        ratio_10msun_to_sun=lifetime_gyr(10.0) / lifetime_gyr(1.0),
        universe_age_gyr=universe_age_gyr,
        # Both lifetimes measured against the universe's own age.
        ratio_10msun_lifetime_to_universe_age=(lifetime_gyr(10.0)) / universe_age_gyr,
        ratio_half_msun_lifetime_to_universe_age=lifetime_gyr(0.5) / universe_age_gyr,
        # The fork in stellar fate.
        core_collapse_threshold_msun=8.0,
        chandrasekhar_mass_msun=1.4,
        earth_radius_km=earth_radius_km,
        wd_mass_msun=wd_mass_msun,
        wd_radius_km=wd_radius_km,
        wd_density_tonne_per_cm3=wd_density,
        ns_mass_msun=ns_mass_msun,
        ns_radius_km=ns_radius_km,
        ns_density_tonne_per_cm3=ns_density,
        ns_wd_density_ratio=ns_density / wd_density,
    )


@example(
    "pch03",
    expect={"stars_per_cubic_pc_mw": (0.442, 0.005),
            "density_ratio_uniform_over_measured": (4.42, 0.05),
            "gas_depletion_gyr": (5.0, 0.01)},
    note="Galaxies: how empty they are, and why ellipticals lens. NB this "
         "density is a crude UNIFORM-disk average (~0.44/pc^3); the measured "
         "density near the Sun is ~0.1/pc^3, because the real disk thins with "
         "radius and height. The 4x is fine for an order-of-magnitude "
         "emptiness argument and the prose must not claim better. The gas "
         "mass/SFR pair are order-of-magnitude Milky-Way benchmarks (standard "
         "textbook figures, not repo-derived), used only for the depletion-"
         "timescale division.",
)
def pch03_galaxies():
    # A disk 1e5 ly across, ~1000 ly thick, with ~1e11 stars.
    r_ly, h_ly = MW_DIAM_LY / 2, 1e3
    vol_ly3 = np.pi * r_ly**2 * h_ly
    vol_pc3 = vol_ly3 / PC_LY**3
    mean_sep_au = (vol_pc3 / MW_STARS) ** (1 / 3) * PC_LY * LY_KM / AU_KM
    sun_diameter_au = 1.392e6 / AU_KM
    # The measured local density near the Sun (order-of-magnitude literature
    # figure, e.g. Holmberg & Flynn); NOT derived from the uniform-disk model
    # above. The ratio is the honesty check the uniform average owes the reader.
    local_star_density_pc3 = 0.1
    # Milky-Way-scale cold-gas reservoir and star-formation rate (order-of-
    # magnitude textbook benchmarks): how long star formation can continue
    # unfed before the gas runs out.
    mw_gas_mass_msun = 1.0e10
    mw_sfr_msun_per_yr = 2.0
    return dict(
        mw_volume_pc3=vol_pc3,
        stars_per_cubic_pc_mw=MW_STARS / vol_pc3,
        # Mean separation between stars, in AU: the number that shows a galaxy
        # is essentially empty and why galaxy collisions pass through.
        mean_star_sep_pc=(vol_pc3 / MW_STARS) ** (1 / 3),
        mean_star_sep_au=mean_sep_au,
        sun_diameter_au=sun_diameter_au,
        mean_star_sep_over_sun_diameter=mean_sep_au / sun_diameter_au,
        local_star_density_pc3=local_star_density_pc3,
        density_ratio_uniform_over_measured=(MW_STARS / vol_pc3) / local_star_density_pc3,
        mw_gas_mass_msun=mw_gas_mass_msun,
        mw_sfr_msun_per_yr=mw_sfr_msun_per_yr,
        gas_depletion_gyr=(mw_gas_mass_msun / mw_sfr_msun_per_yr) / 1e9,
    )


@example(
    "pch04",
    expect={"theta_e_ratio_cluster_to_galaxy": (11.4, 0.3)},
    note="Clusters: why a cluster's Einstein ring is ~10x a galaxy's",
)
def pch04_clusters():
    # theta_E ~ sqrt(M). A cluster is ~100x an elliptical's mass, so its ring is
    # ~10x bigger. The main guide states 1.145" (Ch. 10) and 13.03" (Ch. 15) in
    # different chapters and never connects them. This is the connection.
    theta_e_galaxy = 1.145           # Ch. 10's fiducial elliptical
    theta_e_carousel = 13.03         # Ch. 15's Carousel cluster
    m_carousel = cosmo.mass_within_theta_e(theta_e_carousel, 0.49, 1.432)

    # Groups/clusters/the cosmic web: standard order-of-magnitude astronomy
    # (not a repo measurement), run through the same Milky-Way ruler as
    # everything else in this book rather than quoted from memory.
    n_galaxies_cluster = 1.0e3
    cluster_mass_low_msun = 1.0e14
    cluster_mass_high_msun = 1.0e15
    void_scale_mpc = 100.0
    mw_diam_mpc = MW_DIAM_LY / (PC_LY * 1e6)

    return dict(
        theta_e_galaxy=theta_e_galaxy,
        theta_e_carousel=theta_e_carousel,
        theta_e_ratio_cluster_to_galaxy=theta_e_carousel / theta_e_galaxy,
        carousel_mass_msun=m_carousel,
        # In Milky Ways — the comparison the main guide cannot make.
        carousel_in_milky_ways=m_carousel / MW_MASS,
        # sqrt-scaling sanity check: mass ratio ~ (theta ratio)^2
        implied_mass_ratio=(theta_e_carousel / theta_e_galaxy) ** 2,
        # A cluster, in galaxies and in Milky Ways.
        n_galaxies_cluster=n_galaxies_cluster,
        cluster_mass_low_msun=cluster_mass_low_msun,
        cluster_mass_high_msun=cluster_mass_high_msun,
        cluster_low_in_mw=cluster_mass_low_msun / MW_MASS,
        cluster_high_in_mw=cluster_mass_high_msun / MW_MASS,
        # The cosmic web: a void, in Milky-Way diameters.
        void_scale_mpc=void_scale_mpc,
        mw_diam_mpc=mw_diam_mpc,
        void_in_mw_diameters=void_scale_mpc / mw_diam_mpc,
    )


# --------------------------------------------------------------------------- #
# Part II — How do we know?
# --------------------------------------------------------------------------- #
@example(
    "pch05",
    expect={"sun_peak_nm": (500.0, 5.0)},
    note="Light: Wien's law, the inverse-square law, and lookback time",
)
def pch05_light():
    # Wien: hotter is bluer. This is why "old and red" vs "young and blue" works.
    wien_nm_k = 2.897771955e6
    sun_peak_nm = wien_nm_k / 5772.0
    return dict(
        sun_temp_k=5772.0,
        sun_peak_nm=sun_peak_nm,
        # c = lambda*nu, applied to the Sun's own Wien peak.
        sun_peak_freq_hz=(C_KM_S * 1000.0) / (sun_peak_nm * 1e-9),
        hot_star_peak_nm=wien_nm_k / 30000.0,
        # The book's own figure (p05-blackbody) plots exactly this curve
        # (3000 K / 5772 K / 10000 K); its Wien peak lands in the near-UV,
        # off the left edge of the visible band the figure shades.
        star10000k_peak_nm=wien_nm_k / 10000.0,
        cool_star_peak_nm=wien_nm_k / 3000.0,
        cmb_temp_k=2.725,
        cmb_peak_mm=wien_nm_k / 2.725 * 1e-6,
        # Inverse square: double the distance, quarter the flux; 10x farther,
        # 100x fainter. Same law, two checkpoints.
        flux_ratio_at_2x=1 / 4.0,
        flux_ratio_at_10x=1 / 100.0,
        # Light takes time -- looking far away IS looking into the past.
        # Recomputed here (not imported from pch01) so this chapter's own
        # examples list is self-contained.
        sun_light_travel_min=AU_KM / C_KM_S / 60.0,
        andromeda_ly=2.5e6,
        andromeda_travel_myr=2.5e6 / 1.0e6,
    )


@example(
    "pch06",
    expect={"desi_arcsec_per_px": (0.262, 0.001), "euclid_gain": (13.0, 0.5)},
    note="Telescopes: resolution, and the wall the discovery half runs into",
)
def pch06_telescopes():
    desi_px, desi_seeing = 0.262, 1.3
    euclid_px = 0.1
    theta_e_typical = 1.2
    return dict(
        desi_arcsec_per_px=desi_px,
        desi_seeing_arcsec=desi_seeing,
        # A typical Einstein ring is this many DESI pixels across...
        ring_diameter_px_desi=2 * theta_e_typical / desi_px,
        # ...but the atmosphere smears everything to this width, so the ring is
        # thinner than the blur. That is the resolution wall of Ch. 27.
        seeing_px_desi=desi_seeing / desi_px,
        euclid_arcsec_per_px=euclid_px,
        euclid_gain=desi_seeing / euclid_px,
        hst_arcsec_per_px=0.128,
    )


@example(
    "pch07",
    expect={
        "halpha_nm": (656.470, 0.005),
        "halpha_air_nm": (656.28, 0.02),
        "halpha_air_vacuum_gap_percent": (0.0277, 0.001),
    },
    note="Spectra: the hydrogen fingerprint. The Rydberg formula gives the "
         "VACUUM wavelength (656.470 nm); line lists and older papers often "
         "quote AIR (656.28 nm). The 0.03% gap is exactly the refractive index "
         "of air, and it is a real trap: a redshift measured against the wrong "
         "convention is wrong in the 4th digit.",
)
def pch07_spectra():
    # Balmer lines from one formula: this is why a fingerprint is reproducible.
    R = 1.0967758e7  # m^-1
    def balmer_nm(n):
        return 1.0 / (R * (1 / 4 - 1 / n**2)) * 1e9
    n_air = 1.000277
    halpha_vac = balmer_nm(3)
    halpha_air = halpha_vac / n_air
    return dict(
        halpha_nm=halpha_vac,                 # vacuum, straight from the formula
        halpha_air_nm=halpha_air,              # what a line list usually quotes
        hbeta_nm=balmer_nm(4),
        hgamma_nm=balmer_nm(5),
        # The lens/source line pair the main guide's Ch. 12 selects on.
        ca_ii_k_nm=393.366,
        oii_nm=372.7,
        # Quantifies the air/vacuum trap: the fraction of Halpha's own
        # wavelength that the two conventions disagree by.
        halpha_air_vacuum_gap_percent=(halpha_vac - halpha_air) / halpha_vac * 100.0,
    )


@example(
    "pch08",
    expect={
        # 10.24, NOT the 10.51 you get from Planck's (67.4, 0.315). This repo
        # asserts FlatLambdaCDM(70, 0.3) everywhere and the primer must agree
        # with the guide, not with a textbook. Mixing the two is precisely the
        # silent inconsistency this gate exists for -- it caught this one.
        "z2_lookback_gyr": (10.240, 0.01),
        "halpha_at_z2_nm": (1969.41, 0.5),
    },
    note="Redshift: the chain the main guide never closes",
)
def pch08_redshift():
    from astropy.cosmology import z_at_value  # noqa: F401  (kept: see below)
    import astropy.units as u

    v = {}
    for z in (0.5, 2.0):
        tag = str(z).replace(".", "")
        # 1+z is literally how much the universe has grown since emission.
        v[f"z{tag}_scale_factor"] = 1.0 / (1.0 + z)
        v[f"z{tag}_stretch_percent"] = z * 100.0
        v[f"z{tag}_lookback_gyr"] = cosmo.COSMO.lookback_time(z).to_value(u.Gyr)
        v[f"z{tag}_comoving_mpc"] = cosmo.COSMO.comoving_distance(z).to_value(u.Mpc)
        v[f"z{tag}_angular_mpc"] = cosmo.d_a(z)
    # The guide's standard test pair, finally given a physical picture.
    v["z2_lookback_gyr"] = cosmo.COSMO.lookback_time(2.0).to_value(u.Gyr)
    v["z05_lookback_gyr"] = cosmo.COSMO.lookback_time(0.5).to_value(u.Gyr)
    v["universe_age_gyr"] = cosmo.COSMO.age(0).to_value(u.Gyr)
    # H-alpha, emitted green-red, arrives in the infrared from z=2.
    # Vacuum rest wavelength (see pch07), stretched by 1+z = 3.
    v["halpha_rest_nm"] = 656.470
    v["halpha_at_z2_nm"] = 656.470 * 3.0
    # The naive Doppler reading, v = cz -- fine as an intuition pump at z=0.5,
    # openly absurd at z=2 (2x the speed of light). Ch. 13 runs the same trap
    # on a real cluster redshift (z=1.432) and gets the same kind of nonsense.
    v["naive_v_z05_kms"] = C_KM_S * 0.5
    v["naive_v_z2_kms"] = C_KM_S * 2.0
    v["naive_v_z2_over_c"] = 2.0
    # How far away that light source sits today, in Milky Way diameters --
    # the scale-anchor device, applied to a distance instead of a mass.
    mw_diam_mpc = (MW_DIAM_LY / PC_LY) / 1e6
    v["z05_comoving_in_mw"] = v["z05_comoving_mpc"] / mw_diam_mpc
    v["z20_comoving_in_mw"] = v["z20_comoving_mpc"] / mw_diam_mpc
    # What fraction of the universe's own life that lookback time already is.
    v["z05_lookback_frac_of_age"] = v["z05_lookback_gyr"] / v["universe_age_gyr"]
    v["z20_lookback_frac_of_age"] = v["z20_lookback_gyr"] / v["universe_age_gyr"]
    return v


@example(
    "pch09",
    expect={"parsec_definition_check": (1.0, 1e-9)},
    note="The distance ladder: parallax, and why 1 pc is 1 arcsec by definition",
)
def pch09_ladder():
    arcsec_per_rad = 206264.80624709636
    # A parsec IS the distance at which 1 AU subtends 1 arcsec. Ch. 9 derives the
    # arithmetic; this is the story: the baseline is Earth's own orbit.
    d_pc_for_1arcsec = 1.0 / 1.0
    proxima_distance_pc = 1.0 / 0.7687
    gaia_limit_ly = (1.0 / 20e-6) * PC_LY
    d_l_mpc_z05 = cosmo.COSMO.luminosity_distance(0.5).value
    snia_absolute_mag = -19.3
    snia_apparent_mag_at_z05 = snia_absolute_mag + 5 * np.log10(
        d_l_mpc_z05 * 1e6 / 10.0)
    # One Milky Way diameter, in Mpc -- the same ruler as every other chapter,
    # now used to size a cosmological distance instead of a galactic one.
    mw_diameter_mpc = MW_DIAM_LY / (PC_LY * 1e6)
    gaia_reach_mw = ((1.0 / 20e-6) * PC_LY) / MW_DIAM_LY
    d_l_mw_z05 = d_l_mpc_z05 / mw_diameter_mpc
    return dict(
        parsec_definition_check=d_pc_for_1arcsec,
        arcsec_per_rad=arcsec_per_rad,
        # Proxima Centauri's parallax: the largest of any star, and still tiny.
        proxima_parallax_arcsec=0.7687,
        proxima_distance_pc=proxima_distance_pc,
        # Same star, the ladder's own rung-1 number, converted with Ch. 1's
        # pc-to-ly factor -- and checked against Ch. 1's own independent 4.24 ly.
        proxima_distance_ly=proxima_distance_pc * PC_LY,
        # Gaia reaches ~20 microarcsec -> this far, in parsecs. Still only our
        # own galaxy: the ladder must hand off long before a lens.
        gaia_limit_pc=1.0 / 20e-6,
        gaia_limit_ly=gaia_limit_ly,
        mw_diameter_ly=MW_DIAM_LY,
        # Pure geometry, run to its own precision limit, reaches this many
        # Milky-Way diameters -- barely past our own galaxy's far edge.
        gaia_reach_in_mw_diameters=gaia_limit_ly / MW_DIAM_LY,
        # A Type Ia peaks at M = -19.3; at z=0.5 (D_L ~ 2833 Mpc) it appears:
        snia_absolute_mag=snia_absolute_mag,
        d_l_mpc_z05=d_l_mpc_z05,
        snia_apparent_mag_at_z05=snia_apparent_mag_at_z05,
        # The distance modulus itself, m - M -- the number the whole standard-
        # candle argument is built to produce.
        snia_distance_modulus_z05=snia_apparent_mag_at_z05 - snia_absolute_mag,
        # The same z=0.5 supernova, in Milky-Way diameters: the ladder's other
        # end of the scale that started this section at 1.6.
        d_l_in_mw_diameters_z05=d_l_mw_z05,
        # One calibrated standard candle vs. pure geometry run to its own
        # precision limit -- how much farther one rung of the ladder reaches.
        candle_vs_parallax_reach_ratio=d_l_mw_z05 / gaia_reach_mw,
    )


# --------------------------------------------------------------------------- #
# Part III — What is the universe doing?
# --------------------------------------------------------------------------- #
@example(
    "pch10",
    expect={"hubble_time_gyr": (13.97, 0.05)},
    note="Expansion: Hubble's law and the age it implies",
)
def pch10_expansion():
    H0 = 70.0                       # km/s/Mpc
    mpc_km = 3.0856775814913673e19
    hubble_time_s = mpc_km / H0
    return dict(
        H0=H0,
        # 1/H0 — the age the universe would have at constant expansion. It lands
        # within 4% of the real 13.8 Gyr, which is a coincidence worth noticing.
        hubble_time_gyr=hubble_time_s / (3.1557e7 * 1e9),
        hubble_distance_mpc=C_KM_S / H0,
        # A galaxy 100 Mpc away recedes at:
        v_at_100mpc=H0 * 100.0,
        # Naive v = cz would put z=1.432 (the Carousel source) superluminal:
        naive_v_at_z1432=C_KM_S * 1.432,
        naive_v_over_c=1.432,
        # Figure 10.1's simulated peculiar-velocity scatter around the v=H0*d
        # line — not a measurement, the same noise amplitude figures.py adds
        # (primer/figures.py:258), quoted here so the caption can cite it.
        scatter_kms=900.0,
    )


@example(
    "pch11",
    expect={"cmb_redshift": (1100.0, 50.0)},
    note="Big Bang and CMB: the numbers that name a=0",
)
def pch11_big_bang():
    """Two places where this book's own cosmology cannot produce the famous number.

    Both are kept visible rather than papered over, because Ch. 11's prose is
    about exactly this and the book's rule is to agree with the guide, not with
    a textbook:

    1. **13.47, not 13.8.** FlatLambdaCDM(70, 0.3) — what every campaign in this
       repository asserts — gives 13.47 Gyr. The famous 13.8 is Planck's
       (67.4, 0.315). The 2.5% gap is not a rounding error; it IS the H0
       tension of Ch. 14, showing up as an age. A lower H0 means a slower
       expansion means a longer time to reach today's size.
    2. **380 kyr is measured, not derived.** This cosmology has Tcmb0 = 0: it
       models matter and Lambda and no radiation at all. Its own age at
       z = 1100 is 465 kyr, because the real radiation-dominated early universe
       expanded faster than a matter-only model knows how to. So 380 kyr is
       pinned from measurement and the model's own 465 is reported next to it,
       rather than quietly quoting one and computing with the other.

    The previous version of this function divided 380 kyr by 13.8e9 while
    reporting 13.47 Gyr as the age two lines above — a fraction against an age
    the same dict contradicted.
    """
    import astropy.units as u

    age_now_gyr = cosmo.COSMO.age(0).to_value(u.Gyr)
    z_cmb = 1100.0
    cmb_age_measured_kyr = 380.0
    return dict(
        universe_age_gyr=age_now_gyr,
        # The textbook/Planck value, for the comparison Ch. 11 draws explicitly.
        planck_age_gyr=13.8,
        age_gap_percent=100.0 * (13.8 - age_now_gyr) / age_now_gyr,
        cmb_redshift=z_cmb,
        cmb_temp_k=2.725,
        # The CMB was emitted at ~3000 K and has been stretched by 1+z since.
        cmb_emission_temp_k=2.725 * (1.0 + z_cmb),
        cmb_scale_factor=1.0 / (1.0 + z_cmb),
        cmb_age_kyr=cmb_age_measured_kyr,
        # What THIS cosmology says, radiation-free and therefore too slow early.
        cmb_age_this_model_kyr=cosmo.COSMO.age(z_cmb).to_value(u.Gyr) * 1e6,
        # Self-consistent: the measured CMB age against the age this book quotes.
        cmb_age_fraction=cmb_age_measured_kyr * 1e3 / (age_now_gyr * 1e9),
    )


@example(
    "pch12",
    expect={"dm_to_baryon_ratio": (5.4, 0.2)},
    note="Dark matter: the budget, and what CDM stands for",
)
def pch12_dark_matter():
    omega_b, omega_dm, omega_lambda = 0.049, 0.265, 0.685
    return dict(
        omega_baryon=omega_b,
        omega_dark_matter=omega_dm,
        omega_matter_total=omega_b + omega_dm,
        omega_lambda=omega_lambda,
        dm_to_baryon_ratio=omega_dm / omega_b,
        # Om0=0.3 in FlatLambdaCDM(70, 0.3) is baryons + dark matter. This is
        # the fraction of that 0.3 which is invisible.
        invisible_fraction_of_om0=omega_dm / (omega_b + omega_dm),
        sum_check=omega_b + omega_dm + omega_lambda,
    )


@example(
    "pch13",
    expect={"omega_sum": (1.0, 0.005)},
    note="Dark energy: the budget and the acceleration",
)
def pch13_dark_energy():
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM

    omega_b, omega_dm, omega_lambda = 0.049, 0.265, 0.685
    H0 = cosmo.COSMO.H0.value
    # Matter dilutes as a^-3; Lambda does not dilute at all. So they cross.
    # Solve Om(1+z)^3 = OL for the redshift where the universe stops decelerating.
    z_cross = (omega_lambda / (omega_b + omega_dm)) ** (1 / 3) - 1

    # The 1998 surprise, quantified: how much farther (fainter) a z=0.5 Type Ia
    # sits in the repo's actual (accelerating) cosmology than in a matter-only,
    # decelerating one of the same H0. Same comparison Perlmutter/Riess made.
    decel_only = FlatLambdaCDM(H0=H0, Om0=1.0)  # flat, Lambda=0 -- the pre-1998 prior
    z_sn = 0.5
    dl_actual_mpc = cosmo.COSMO.luminosity_distance(z_sn).to_value(u.Mpc)
    dl_decel_mpc = decel_only.luminosity_distance(z_sn).to_value(u.Mpc)

    return dict(
        omega_baryon=omega_b,
        omega_dark_matter=omega_dm,
        omega_lambda=omega_lambda,
        omega_sum=omega_b + omega_dm + omega_lambda,
        z_matter_lambda_equality=z_cross,
        lookback_at_crossover_gyr=cosmo.COSMO.lookback_time(z_cross).to_value(u.Gyr),
        age_at_crossover_gyr=cosmo.COSMO.age(z_cross).to_value(u.Gyr),
        universe_age_gyr=cosmo.COSMO.age(0).to_value(u.Gyr),
        # We live after the crossover: expansion is accelerating now.
        omega_ratio_today=omega_lambda / (omega_b + omega_dm),
        dl_actual_z05_mpc=dl_actual_mpc,
        dl_decel_only_z05_mpc=dl_decel_mpc,
        dl_ratio_z05=dl_actual_mpc / dl_decel_mpc,
        mag_diff_z05=5.0 * np.log10(dl_actual_mpc / dl_decel_mpc),
        # As matter dilutes to nothing, H(z) -> H0*sqrt(Omega_Lambda): the
        # constant rate the expansion is heading toward, not zero.
        h_de_sitter_kms_mpc=H0 * np.sqrt(omega_lambda),
        # Of the universe's life so far, this fraction was spent decelerating
        # (before the crossover) versus accelerating (after it).
        fraction_of_age_decelerating=(
            cosmo.COSMO.age(z_cross).to_value(u.Gyr) / cosmo.COSMO.age(0).to_value(u.Gyr)),
        fraction_of_age_accelerating=1.0 - (
            cosmo.COSMO.age(z_cross).to_value(u.Gyr) / cosmo.COSMO.age(0).to_value(u.Gyr)),
        # The repo's OWN cosmology object rounds Omega_m to 0.3 exactly (not the
        # measured 0.314), so Omega_Lambda rounds to 0.7 (not 0.685). Harmless
        # for lensing (Ch. 14 of the main guide), but this chapter cares about
        # the crossover epoch specifically, so show what the rounding costs.
        z_cross_repo_om03=(0.7 / 0.3) ** (1 / 3) - 1,
    )


@example(
    "pch14",
    expect={"tension_sigma": (5.0, 1.5)},
    note="The Hubble tension: two answers that do not overlap",
)
def pch14_tension():
    # SH0ES (distance ladder) vs Planck (CMB). Representative published values.
    h0_ladder, e_ladder = 73.0, 1.0
    h0_cmb, e_cmb = 67.4, 0.5
    diff = h0_ladder - h0_cmb
    sig = np.hypot(e_ladder, e_cmb)
    return dict(
        h0_ladder=h0_ladder,
        h0_ladder_err=e_ladder,
        h0_cmb=h0_cmb,
        h0_cmb_err=e_cmb,
        h0_difference=diff,
        tension_sigma=diff / sig,
        # The repo asserts 70 — almost exactly between them, and committed to
        # neither. Percent difference between the two camps:
        percent_difference=100.0 * diff / h0_cmb,
        repo_h0=70.0,
        # Each camp's own claimed precision, as a percent of its own value —
        # for comparing against the 8.3% gap between the two camps above.
        h0_ladder_relerr_pct=100.0 * e_ladder / h0_ladder,
        h0_cmb_relerr_pct=100.0 * e_cmb / h0_cmb,
        # The gap, expressed as a multiple of each camp's own claimed precision.
        gap_vs_ladder_relerr=(100.0 * diff / h0_cmb) / (100.0 * e_ladder / h0_ladder),
        gap_vs_cmb_relerr=(100.0 * diff / h0_cmb) / (100.0 * e_cmb / h0_cmb),
        # What the repo asserts sits almost exactly halfway between the two camps.
        h0_midpoint=(h0_ladder + h0_cmb) / 2.0,
    )


# --------------------------------------------------------------------------- #
# Part IV — Why lensing?
# --------------------------------------------------------------------------- #
@example(
    "pch15",
    expect={"sun_deflection_arcsec": (1.75, 0.01)},
    note="Deflection: the Eddington number, and why the factor of two mattered",
)
def pch15_deflection():
    G = 6.674e-11
    c = 2.998e8
    m_sun = 1.989e30
    r_sun = 6.957e8
    rad_to_arcsec = 206264.80624709636
    newton = 2 * G * m_sun / (c**2 * r_sun) * rad_to_arcsec
    einstein = 2 * newton
    return dict(
        newtonian_deflection_arcsec=newton,
        sun_deflection_arcsec=einstein,
        factor=einstein / newton,
        # 1919 measured ~1.98 +/- 0.16 (Sobral) and ~1.61 +/- 0.40 (Principe):
        eddington_measured=1.98,
        eddington_error=0.16,
        # Newton's prediction sits this many sigma from what they measured.
        newton_sigma_off=(1.98 - newton) / 0.16,
    )


@example(
    "pch16",
    expect={"theta_e_typical_arcsec": (1.145, 0.01)},
    note="What a strong lens is: the geometry and how rare it is",
)
def pch16_strong_lens():
    theta_e = cosmo.theta_e_from_sigma_v(250.0, 0.5, 2.0)
    v = dict(
        theta_e_typical_arcsec=theta_e,
        sigma_v_kms=250.0,
        z_lens=0.5,
        z_source=2.0,
        # Alignment must be within ~theta_E. The sky is 41253 sq deg; a
        # theta_E=1.145" disc is this fraction of it -- the rarity, quantified.
        sky_sq_deg=41253.0,
        lens_disc_sq_deg=np.pi * (theta_e / 3600.0) ** 2,
    )
    v["alignment_fraction"] = v["lens_disc_sq_deg"] / v["sky_sq_deg"]
    # Roughly one in this many random sightlines lands inside a given lens.
    v["one_in_n_sightlines"] = 1.0 / v["alignment_fraction"]
    # The DESI sweep that found them: 53.8M galaxies scored, ~5000 candidates.
    v["dr11_galaxies_scored"] = 5.38e7
    return v
