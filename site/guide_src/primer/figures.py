"""Figure builders for the Beginner's Guide.

Same contract as the main guide's `figures.py`: each function is registered with
@figure, called once per scheme, and returns (fig, values). Same shared `_style`
(so the two books look like one system) and the same shared `lensing`/`cosmo`
(so they cannot disagree about a number).

House rule, unchanged: compute, don't draw. A beginner's figure has to be MORE
honest than an expert's, not less — the reader has no way to catch a fudge.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _style  # noqa: E402
import cosmo  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from registry import figure  # noqa: E402


# --------------------------------------------------------------------------- #
# Part I — Where are we?
# --------------------------------------------------------------------------- #
@figure("p01-scale-ladder", chapter=1,
        caption_hint="Every rung from the Earth to the observable universe, on one log axis")
def scale_ladder(scheme):
    """The book's ruler. The main guide has no rung of this except the parsec."""
    rungs = [
        ("Earth", 1.3e4 / 9.461e12, ""),
        ("Earth–Sun (1 AU)", 1.496e8 / 9.461e12, ""),
        ("Solar system (Neptune)", 4.5e9 / 9.461e12, ""),
        ("Nearest star", 4.24, "Proxima"),
        ("1 parsec", 3.2616, ""),
        ("Milky Way", 1e5, "our ruler"),
        ("To Andromeda", 2.5e6, ""),
        ("Local Group", 1e7, ""),
        ("A galaxy cluster", 2e7, "~1000 galaxies"),
        ("Cosmic web filament", 3e8, ""),
        ("Observable universe", 9.3e10, ""),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for i, (name, ly, note) in enumerate(rungs):
        y = len(rungs) - i
        col = (_style.ACCENT[scheme] if "ruler" in note
               else _style.GOOD[scheme])
        ax.plot([ly], [y], "o", color=col, ms=7)
        ax.hlines(y, 1e-9, ly, color=_style.MUTED[scheme], lw=0.6, alpha=0.5)
        label = f"{name}" + (f"  ({note})" if note else "")
        ax.text(ly * 2.2, y, label, va="center", fontsize=9, color=col)
    ax.set_xscale("log")
    ax.set_xlim(1e-9, 1e14)
    ax.set_ylim(0, len(rungs) + 1)
    ax.set_yticks([])
    ax.set_xlabel("distance (light-years, log scale)")
    ax.set_title("Twenty-three orders of magnitude, one axis", fontsize=11)
    ax.grid(axis="y", visible=False)
    return fig, dict(n_rungs=len(rungs), span_decades=np.log10(9.3e10 / (1.3e4 / 9.461e12)))


@figure("p02-main-sequence", chapter=2,
        caption_hint="Why massive stars die young: lifetime falls as M^-2.5")
def main_sequence(scheme):
    """The one exponent behind 'old and red' — the guide's unexplained premise."""
    m = np.logspace(-1, 1.5, 300)
    t = 10.0 * m ** -2.5          # Gyr
    # The repo's own cosmology, NOT the textbook 13.8: FlatLambdaCDM(70, 0.3)
    # gives 13.47 Gyr. p08 already draws this line from cosmo.COSMO.age(0), and
    # two figures in one book must not label the same line with two numbers.
    # Ch. 11 explains the 13.47-vs-13.8 gap; it is the H0 tension in disguise.
    import astropy.units as u
    age_gyr = cosmo.COSMO.age(0).to_value(u.Gyr)

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ax.loglog(m, t, color=_style.GOOD[scheme])
    ax.axhline(age_gyr, color=_style.WARN[scheme], ls="--", lw=1.2,
               label=f"age of the universe ({age_gyr:.1f} Gyr)")
    for mm, lab in ((1.0, "the Sun"), (10.0, "10 $M_\\odot$"), (0.5, "0.5 $M_\\odot$")):
        ax.plot([mm], [10 * mm ** -2.5], "o", color=_style.ACCENT[scheme])
        ax.annotate(lab, (mm, 10 * mm ** -2.5), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color=_style.ACCENT[scheme])
    ax.set_xlabel("stellar mass ($M_\\odot$)")
    ax.set_ylabel("main-sequence lifetime (Gyr)")
    ax.legend(fontsize=9)
    ax.set_title("A star 10× the Sun's mass lives 0.03% as long")
    return fig, dict(sun_gyr=10.0, m10_myr=10 * 10 ** -2.5 * 1000,
                     age_gyr=age_gyr)


@figure("p03-galaxy-types", chapter=3,
        caption_hint="A spiral's disk and arms vs an elliptical's smooth, compact blob")
def galaxy_types(scheme):
    """The visible split this chapter draws on: structure and light, not orbits.

    Ch. 10 of the main guide earns the DYNAMICAL distinction (ordered rotation
    vs disordered stellar orbits) properly; repeating it here would be the
    outline's explicit anti-goal. This figure only shows what a telescope
    sees: an exponential disk traced by two spiral arms, next to a smooth,
    centrally concentrated de Vaucouleurs blob with a much smaller footprint
    for the same total light budget.
    """
    n, half = 220, 6.0
    g = np.linspace(-half, half, n)
    X, Y = np.meshgrid(g, g)
    R = np.hypot(X, Y)
    THETA = np.arctan2(Y, X)

    # Spiral: exponential disk, brightened along two logarithmic-spiral arms.
    r_d = 2.2
    disk = np.exp(-R / r_d)
    pitch = 0.30
    arm = np.zeros_like(R)
    for offset in (0.0, np.pi):
        phase = THETA - offset - np.log(np.maximum(R, 0.05) / r_d) / pitch
        arm += np.exp(-3.0 * (1 - np.cos(phase))) * np.exp(-R / (1.6 * r_d))
    spiral_img = disk + 1.3 * arm
    spiral_img /= spiral_img.max()

    # Elliptical: de Vaucouleurs (n=4, Ch. 10's b_n), a third the footprint,
    # no arms, no structure — smooth all the way down.
    r_e = 1.1
    b_n = 7.6697
    ell_img = np.exp(-b_n * ((R / r_e) ** 0.25 - 1))
    ell_img /= ell_img.max()

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for ax, img, title in zip(
        axes, (spiral_img, ell_img),
        ("A spiral — disk, arms, still forming stars",
         "An elliptical — smooth, compact, quiescent"),
    ):
        ax.imshow(np.clip(img, 0, 1) ** 0.5, extent=[-half, half, -half, half],
                  cmap=_style.CMAP_SEQ, origin="lower", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("Same class of object, two very different shapes", fontsize=11)
    return fig, dict(disk_scale_length=r_d, elliptical_effective_radius=r_e)


@figure("p04-theta-e-ladder", chapter=4,
        caption_hint="Why a cluster's Einstein ring is ~10x a galaxy's: theta_E ~ sqrt(M)")
def theta_e_ladder(scheme):
    """Connects three numbers the main guide states in three chapters and never joins."""
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    m = np.logspace(11, 15.2, 200)
    # theta_E ~ sqrt(M), anchored on the Carousel's measured pair.
    m_car = cosmo.mass_within_theta_e(13.03, 0.49, 1.432)
    th = 13.03 * np.sqrt(m / m_car)
    ax.loglog(m, th, color=_style.GOOD[scheme], lw=1.6,
              label=r"$\theta_{\rm E} \propto \sqrt{M}$")
    pts = [(1.5e12, None, "Milky Way\n(mass only)", _style.MUTED[scheme]),
           (m_car / 130, 1.145, "a massive elliptical\n(Ch. 10)", _style.ACCENT[scheme]),
           (m_car, 13.03, "the Carousel cluster\n(Ch. 15)", _style.WARN[scheme])]
    for mm, tt, lab, col in pts:
        yy = tt if tt else 13.03 * np.sqrt(mm / m_car)
        ax.plot([mm], [yy], "o", color=col, ms=8)
        ax.annotate(lab, (mm, yy), textcoords="offset points", xytext=(-14, -34),
                    fontsize=8, color=col, ha="center")
    ax.set_xlabel(r"mass inside the ring ($M_\odot$)")
    ax.set_ylabel(r"Einstein radius $\theta_{\rm E}$ (arcsec)")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("100× the mass buys 10× the ring", fontsize=11)
    return fig, dict(carousel_mass=m_car, ratio=13.03 / 1.145)


# --------------------------------------------------------------------------- #
# Part II — How do we know?
# --------------------------------------------------------------------------- #
@figure("p05-blackbody", chapter=5,
        caption_hint="Hotter is bluer: Planck curves and Wien's law")
def blackbody(scheme):
    """Why 'old and red' vs 'young and blue' is a temperature statement."""
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    lam = np.linspace(50e-9, 2000e-9, 800)

    def planck(T):
        return (2 * h * c**2 / lam**5) / (np.exp(h * c / (lam * k * T)) - 1)

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for T, lab in ((3000, "3000 K — a cool red star"),
                   (5772, "5772 K — the Sun"),
                   (10000, "10000 K — a hot blue star")):
        b = planck(T)
        ax.plot(lam * 1e9, b / b.max(), label=lab)
        ax.axvline(2.8978e6 / T, ls=":", lw=0.8, color=_style.MUTED[scheme])
    ax.axvspan(380, 750, color=_style.MUTED[scheme], alpha=0.12)
    ax.text(565, 1.04, "visible", ha="center", fontsize=8,
            color=_style.MUTED[scheme])
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("brightness (each curve scaled to its own peak)")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.set_title("Wien's law: the peak moves left as it heats up")
    return fig, dict(sun_peak_nm=2.8978e6 / 5772)


@figure("p06-resolution-ladder", chapter=6,
        caption_hint="The same lens as DESI, HST and Euclid see it")
def resolution_ladder(scheme):
    """The resolution wall of Ch. 27, drawn: a ring thinner than the blur."""
    theta_e = 1.2
    n, half = 200, 3.0
    g = np.linspace(-half, half, n)
    X, Y = np.meshgrid(g, g)
    R = np.hypot(X, Y)
    ring = np.exp(-((R - theta_e) ** 2) / (2 * 0.12 ** 2))

    def blur(img, fwhm_arcsec):
        from scipy.ndimage import gaussian_filter
        sig_px = (fwhm_arcsec / 2.3548) / (2 * half / n)
        return gaussian_filter(img, sig_px)

    cases = [("truth", 0.0), ("Euclid  $0.1''$", 0.1),
             ("HST  $0.13''$", 0.13), ("DESI  $1.3''$ seeing", 1.3)]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.3))
    for ax, (lab, fw) in zip(axes, cases):
        im = ring if fw == 0 else blur(ring, fw)
        ax.imshow(im, extent=[-half, half, -half, half],
                  cmap=_style.CMAP_SEQ, origin="lower")
        ax.set_title(lab, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("One $1.2''$ Einstein ring, four instruments", fontsize=10)
    return fig, dict(theta_e=theta_e, desi_seeing=1.3)


@figure("p07-spectrum-lines", chapter=7,
        caption_hint="A continuum with absorption lines — the fingerprint")
def spectrum_lines(scheme):
    """Why a spectrum identifies an element AND a redshift, in one picture."""
    lam = np.linspace(350, 750, 2000)
    cont = 1.0 - 0.35 * (lam - 350) / 400          # a tilted continuum
    lines = {"Ca II K": 393.4, "Ca II H": 396.8, r"H$\beta$": 486.1,
             "Mg b": 517.3, r"H$\alpha$": 656.5}
    flux = cont.copy()
    for _, c in lines.items():
        flux -= 0.30 * np.exp(-((lam - c) ** 2) / (2 * 1.6 ** 2))

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(lam, flux, color=_style.GOOD[scheme], lw=1.2)
    for name, c in lines.items():
        ax.annotate(name, (c, flux[np.argmin(abs(lam - c))] - 0.05),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color=_style.ACCENT[scheme])
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("flux (arbitrary)")
    ax.set_ylim(0.1, 1.1)
    ax.set_title("An elliptical galaxy's spectrum: old stars, no emission")
    return fig, dict(n_lines=len(lines))


@figure("p08-redshift-chain", chapter=8,
        caption_hint="One redshift, four readings: stretch, scale, lookback, distance")
def redshift_chain(scheme):
    """The chain the main guide never closes, on one axis."""
    import astropy.units as u
    z = np.linspace(0.001, 3.0, 300)
    look = cosmo.COSMO.lookback_time(z).to_value(u.Gyr)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(z, look, color=_style.GOOD[scheme], lw=1.8)
    for zz, lab in ((0.5, "$z=0.5$\na typical lens"), (2.0, "$z=2.0$\na typical source")):
        ll = cosmo.COSMO.lookback_time(zz).to_value(u.Gyr)
        ax.plot([zz], [ll], "o", color=_style.ACCENT[scheme], ms=8)
        ax.vlines(zz, 0, ll, color=_style.MUTED[scheme], ls=":", lw=1)
        ax.annotate(f"{lab}\n{ll:.1f} Gyr ago\nuniverse was {1/(1+zz)*100:.0f}% its size",
                    (zz, ll), textcoords="offset points", xytext=(10, -46),
                    fontsize=8, color=_style.ACCENT[scheme])
    ax.axhline(cosmo.COSMO.age(0).to_value(u.Gyr), color=_style.WARN[scheme],
               ls="--", lw=1, label="age of the universe")
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel("how long ago the light left (Gyr)")
    ax.set_ylim(0, 14)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Bigger $z$ = longer ago = farther away")
    return fig, dict(z05_gyr=cosmo.COSMO.lookback_time(0.5).to_value(u.Gyr),
                     z2_gyr=cosmo.COSMO.lookback_time(2.0).to_value(u.Gyr))


@figure("p09-distance-ladder", chapter=9,
        caption_hint="Each rung calibrates the next, and errors compound")
def distance_ladder(scheme):
    """Names the ladder the main guide climbs without naming."""
    rungs = [("Parallax\n(geometry)", 1e-5, 1e-2),
             ("Cepheids\n(period–luminosity)", 1e-3, 30),
             ("Type Ia SNe\n(standard candle)", 1, 3000),
             ("Redshift\n($v = H_0 d$)", 30, 1e5)]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for i, (name, lo, hi) in enumerate(rungs):
        y = len(rungs) - i
        ax.hlines(y, lo, hi, color=_style.SERIES[scheme][i], lw=6, alpha=0.85)
        ax.text(np.sqrt(lo * hi), y + 0.28, name, ha="center", fontsize=8.5,
                color=_style.SERIES[scheme][i])
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1e6)
    ax.set_ylim(0.3, len(rungs) + 0.9)
    ax.set_yticks([])
    ax.set_xlabel("distance (Mpc, log scale)")
    ax.set_title("Every rung is calibrated by the one below it", fontsize=11)
    ax.grid(axis="y", visible=False)
    return fig, dict(n_rungs=len(rungs))


# --------------------------------------------------------------------------- #
# Part III — What is the universe doing?
# --------------------------------------------------------------------------- #
@figure("p10-expansion", chapter=10,
        caption_hint="Hubble's law: recession speed rises linearly with distance")
def expansion(scheme):
    """Every galaxy sees itself at the centre — that IS the point."""
    rng = np.random.default_rng(7)          # fixed: figures must be reproducible
    d = rng.uniform(5, 400, 40)
    v = 70.0 * d + rng.normal(0, 900, 40)   # H0*d plus peculiar motions
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.scatter(d, v, s=22, color=_style.GOOD[scheme], alpha=0.8,
               label="galaxies (with peculiar motions)")
    dd = np.linspace(0, 420, 10)
    ax.plot(dd, 70 * dd, color=_style.ACCENT[scheme], lw=1.8,
            label=r"$v = H_0 d$,  $H_0 = 70$")
    ax.set_xlabel("distance (Mpc)")
    ax.set_ylabel("recession speed (km/s)")
    ax.legend(fontsize=9)
    ax.set_title("The farther away, the faster it recedes — from everywhere")
    return fig, dict(H0=70.0, scatter_kms=900)


@figure("p11-cosmic-history", chapter=11,
        caption_hint="Running the expansion backwards: the scale factor and the temperature that goes with it")
def cosmic_history(scheme):
    """The whole history on one log axis, computed from the repo's own cosmology.

    Two honest caveats, both carried by the caption rather than hidden:

    1. FlatLambdaCDM(70, 0.3) has Tcmb0 = 0 — it models matter and Lambda and
       NO radiation. So its own age at z = 1100 is 0.47 Myr, where the measured
       value is 0.38 Myr: the real early universe was radiation-dominated and
       expanded faster than this model knows. The marker is placed at the CMB's
       SCALE FACTOR (a = 1/1101, which is a measurement, not a model output) and
       the curve puts it in time, so the figure stays self-consistent with the
       cosmology it is drawn from instead of splicing in a foreign number.
    2. a -> 0 is where the model's own extrapolation ends, not a depiction of
       t = 0. Nothing here claims to draw the singularity.

    The two panels are the SAME CURVE and that is the entire point: on linear
    axes you can see a -> 0, and the CMB is invisible because it sits in the
    first 0.003% of the x-axis; on log axes the CMB is visible and a = 0 is
    gone, because log(0) does not exist. Cosmology is read on log axes for this
    reason, and a beginner who has only ever seen the log version has never been
    shown that a -> 0 is what "the Big Bang" names. Plotting T alongside would
    be redundant: on log axes T = 2.725/a is a(t) flipped, the same line twice.
    """
    import astropy.units as u

    z = np.logspace(np.log10(3000.0), -3, 600)
    a = 1.0 / (1.0 + z)
    t_gyr = cosmo.COSMO.age(z).to_value(u.Gyr)

    z_cmb, T_now = 1100.0, 2.725
    a_cmb = 1.0 / (1.0 + z_cmb)
    t_cmb = cosmo.COSMO.age(z_cmb).to_value(u.Gyr)
    age_now = cosmo.COSMO.age(0).to_value(u.Gyr)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))

    # -- linear: a -> 0 is visible, the CMB is not ---------------------------
    ax = axes[0]
    ax.plot(t_gyr, a, color=_style.GOOD[scheme], lw=1.8)
    ax.plot([age_now], [1.0], "o", color=_style.WARN[scheme], ms=7)
    ax.annotate(f"now — $a=1$, {age_now:.1f} Gyr", (age_now, 1.0),
                textcoords="offset points", xytext=(-8, 7), fontsize=8,
                ha="right", va="bottom", color=_style.WARN[scheme])
    ax.annotate("$a\\to0$:\nthe Big Bang", (0.0, 0.0),
                textcoords="offset points", xytext=(16, 30), fontsize=8,
                color=_style.ACCENT[scheme],
                arrowprops=dict(arrowstyle="->", lw=1.0,
                                color=_style.ACCENT[scheme]))
    ax.plot([t_cmb], [a_cmb], "o", color=_style.ACCENT[scheme], ms=6)
    ax.set_xlim(-0.4, 14.2)
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlabel("time since the Big Bang (Gyr)")
    ax.set_ylabel("scale factor $a$")
    ax.set_title("Linear: you can see $a\\to0$", fontsize=9.5)

    # -- log: the CMB is visible, a = 0 is not -------------------------------
    ax = axes[1]
    ax.loglog(t_gyr, a, color=_style.GOOD[scheme], lw=1.8)
    ax.plot([t_cmb], [a_cmb], "o", color=_style.ACCENT[scheme], ms=7)
    # 1/1101, not 1/1100: a = 1/(1+z) with z = 1100. The chapter's arithmetic
    # (3000/1101, "one part in eleven hundred") depends on that distinction, and
    # the marker is plotted at 1/1101 — the label must not say something else.
    # va="center" keeps the three-line block off the bottom spine; with va="top"
    # the "3000 K" line collided with the x-axis.
    ax.annotate("the CMB\n$a=1/1101$\n$3000$ K", (t_cmb, a_cmb),
                textcoords="offset points", xytext=(12, 0), fontsize=8,
                ha="left", va="center", color=_style.ACCENT[scheme])
    ax.plot([age_now], [1.0], "o", color=_style.WARN[scheme], ms=7)
    ax.annotate("now — $2.7$ K", (age_now, 1.0), textcoords="offset points",
                xytext=(-8, 8), fontsize=8, ha="right", va="bottom",
                color=_style.WARN[scheme])
    ax.set_ylim(2e-4, 3.0)
    ax.set_xlabel("time since the Big Bang (Gyr)")
    ax.set_ylabel("scale factor $a$")
    ax.set_title("Log: you can see the CMB", fontsize=9.5)

    fig.tight_layout()
    return fig, dict(
        a_cmb=a_cmb,
        t_cmb_myr=t_cmb * 1e3,
        age_now_gyr=age_now,
        T_cmb_emission_k=T_now / a_cmb,
        T_now_k=T_now,
    )


@figure("p12-rotation-curve", chapter=12,
        caption_hint="The first evidence: rotation curves that refuse to fall")
def rotation_curve(scheme):
    """Kepler's prediction vs what is measured. The gap is the dark matter."""
    r = np.linspace(0.5, 30, 300)
    r_vis = 12.0
    # Visible mass only: flat inside, Keplerian v ~ 1/sqrt(r) outside.
    v_kep = np.where(r < r_vis, 200 * np.sqrt(r / r_vis), 200 * np.sqrt(r_vis / r))
    v_obs = 200 * np.ones_like(r)
    v_obs[r < r_vis] = 200 * np.sqrt(r[r < r_vis] / r_vis)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(r, v_obs, color=_style.GOOD[scheme], lw=2, label="measured — stays flat")
    ax.plot(r, v_kep, color=_style.WARN[scheme], lw=1.8, ls="--",
            label="predicted from the light alone")
    ax.fill_between(r, v_kep, v_obs, where=(v_obs > v_kep),
                    color=_style.ACCENT[scheme], alpha=0.18,
                    label="the missing mass")
    ax.axvline(r_vis, color=_style.MUTED[scheme], ls=":", lw=1)
    ax.text(r_vis + 0.4, 40, "edge of\nthe visible disk", fontsize=8,
            color=_style.MUTED[scheme])
    ax.set_xlabel("radius (kpc)")
    ax.set_ylabel("orbital speed (km/s)")
    ax.set_ylim(0, 260)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Stars at the edge orbit far too fast for the light we see")
    return fig, dict(v_flat=200.0, r_visible_kpc=r_vis)


@figure("p13-energy-budget", chapter=13,
        caption_hint="What the universe is made of — and what Om0=0.3 asserts")
def energy_budget(scheme):
    """What FlatLambdaCDM(70, 0.3) actually claims, drawn."""
    parts = [("Dark energy", 68.5), ("Dark matter", 26.5), ("Ordinary matter", 4.9)]
    cols = [_style.SERIES[scheme][4], _style.SERIES[scheme][0],
            _style.ACCENT[scheme]]
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    wedges, _ = ax.pie([p[1] for p in parts], colors=cols, startangle=90,
                       wedgeprops=dict(width=0.42, edgecolor="none"))
    ax.legend(wedges, [f"{n} — {v}%" for n, v in parts],
              loc="center", fontsize=9, frameon=False)
    ax.set_title("Everything you have ever seen is the 4.9%", fontsize=11)
    return fig, dict(omega_lambda=0.685, omega_dm=0.265, omega_b=0.049)


@figure("p14-h0-tension", chapter=14,
        caption_hint="Two measurements of H0 that do not overlap")
def h0_tension(scheme):
    """The controversy the main guide sidesteps by asserting 70."""
    rows = [("Planck CMB", 67.4, 0.5, _style.SERIES[scheme][2]),
            ("SH0ES ladder", 73.0, 1.0, _style.WARN[scheme]),
            ("this repo asserts", 70.0, None, _style.MUTED[scheme])]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for i, (name, v, e, col) in enumerate(rows):
        y = len(rows) - i
        if e:
            ax.errorbar([v], [y], xerr=[e], fmt="o", color=col, capsize=4, lw=2)
        else:
            ax.plot([v], [y], "|", ms=18, mew=2, color=col)
        ax.text(75.2, y, name, fontsize=9, va="center", color=col)
    ax.set_xlim(65, 79)
    ax.set_ylim(0.3, len(rows) + 0.7)
    ax.set_yticks([])
    ax.set_xlabel("$H_0$ (km/s/Mpc)")
    ax.set_title("The error bars do not touch. Somebody is wrong.", fontsize=11)
    ax.grid(axis="y", visible=False)
    return fig, dict(gap=73.0 - 67.4, sigma=(73.0 - 67.4) / np.hypot(1.0, 0.5))


# --------------------------------------------------------------------------- #
# Part IV — Why lensing?
# --------------------------------------------------------------------------- #
@figure("p15-deflection", chapter=15,
        caption_hint="Why 1919 settled it: Newton and Einstein differ by exactly 2x, and the eclipse could tell")
def deflection(scheme):
    """The factor of two, drawn as the thing Eddington actually had to resolve.

    Both curves are the SAME 1/b law — this is not Einstein predicting a
    different shape, which is exactly why the measurement had to be a factor of
    two in amplitude at one impact parameter rather than a curve shape anyone
    could fit. Grazing the Sun's limb (b = R_sun) is the largest deflection
    available in the solar system, and it is still 1.75 arcsec: the entire
    argument between two theories of gravity came down to half an arcsecond.

    The 1919 point is Sobral's 1.98 +/- 0.16 (see pch15). It sits 1.4 sigma from
    Einstein and 6.9 sigma from Newton. Deliberately NOT drawn as a clean win:
    the error bar is wide, Principe's was wider (1.61 +/- 0.40), and the honest
    statement is that 1919 excluded Newton, not that it confirmed Einstein to
    any precision. Ch. 15's prose says exactly that.
    """
    G, c = 6.674e-11, 2.998e8
    m_sun, r_sun = 1.989e30, 6.957e8
    rad_to_arcsec = 206264.80624709636

    b = np.linspace(1.0, 4.0, 300)                    # impact parameter, R_sun
    einstein = 4 * G * m_sun / (c**2 * b * r_sun) * rad_to_arcsec
    newton = einstein / 2.0

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(b, einstein, color=_style.GOOD[scheme], lw=1.8,
            label=r"Einstein:  $\alpha = 4GM/c^2 b$")
    ax.plot(b, newton, color=_style.WARN[scheme], lw=1.8, ls="--",
            label=r"Newton:  $\alpha = 2GM/c^2 b$")
    ax.fill_between(b, newton, einstein, color=_style.MUTED[scheme], alpha=0.12)
    ax.annotate("the factor of two —\nthe whole argument", xy=(3.1, 0.33),
                xytext=(2.35, 1.02), fontsize=8, ha="center",
                color=_style.MUTED[scheme],
                arrowprops=dict(arrowstyle="->", lw=0.9,
                                color=_style.MUTED[scheme]))

    ax.errorbar([1.0], [1.98], yerr=[0.16], fmt="o", ms=7,
                color=_style.ACCENT[scheme], capsize=4, lw=1.8, zorder=5,
                label="Sobral, 29 May 1919:  $1.98'' \\pm 0.16''$")
    ax.axvline(1.0, color=_style.MUTED[scheme], ls=":", lw=1)
    ax.text(1.08, 0.09, "grazing the Sun's limb —\nthe biggest there is",
            fontsize=8, color=_style.MUTED[scheme])

    ax.set_xlabel("impact parameter $b$  (solar radii)")
    ax.set_ylabel(r"deflection $\alpha$ (arcsec)")
    ax.set_xlim(0.9, 4.05)
    ax.set_ylim(0, 2.35)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("One measurement, half an arcsecond apart", fontsize=11)
    return fig, dict(
        einstein_at_limb=4 * G * m_sun / (c**2 * r_sun) * rad_to_arcsec,
        newton_at_limb=2 * G * m_sun / (c**2 * r_sun) * rad_to_arcsec,
        measured=1.98,
        measured_err=0.16,
    )


@figure("p16-lens-geometry", chapter=16,
        caption_hint="The geometry: two galaxies, wildly different distances, one sightline")
def lens_geometry(scheme):
    """The picture Ch. 12 calls 'the geometric definition of a lens candidate'."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    obs, lens, src = 0.0, 6.0, 10.0
    ax.plot([obs], [0], "o", ms=9, color=_style.GOOD[scheme])
    ax.text(obs, -0.85, "you\n$z=0$", ha="center", fontsize=9)
    ax.plot([lens], [0], "o", ms=13, color=_style.ACCENT[scheme])
    ax.text(lens, -0.95, "lens galaxy\n$z=0.5$\n5 Gyr ago", ha="center", fontsize=9,
            color=_style.ACCENT[scheme])
    ax.plot([src], [0], "*", ms=15, color=_style.WARN[scheme])
    ax.text(src, -0.95, "source galaxy\n$z=2.0$\n10 Gyr ago", ha="center",
            fontsize=9, color=_style.WARN[scheme])
    # Two bent rays, deflected at the lens plane.
    for sgn in (+1, -1):
        b = sgn * 0.9
        ax.plot([obs, lens], [0, b], color=_style.WARN[scheme], lw=1.3)
        ax.plot([lens, src], [b, 0], color=_style.WARN[scheme], lw=1.3)
    ax.plot([obs, src], [0, 0], color=_style.MUTED[scheme], ls=":", lw=1)
    ax.annotate("light bends here", (lens, 0.95), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=8,
                color=_style.WARN[scheme])
    ax.text(3.0, 0.62, "you see the source HERE", fontsize=8,
            color=_style.WARN[scheme])
    ax.set_xlim(-1, 11.5)
    ax.set_ylim(-2.1, 1.6)
    ax.axis("off")
    ax.set_title("A strong lens is an accident of alignment", fontsize=11)
    return fig, dict(z_lens=0.5, z_source=2.0)
