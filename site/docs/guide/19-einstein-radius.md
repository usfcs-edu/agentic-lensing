# The Einstein radius: the one thing lensing measures cleanly

Every number this book eventually argues about — the EPL slope $\gamma$, the
external shear, the source's shape — is a fit: a quantity pulled out of noisy
pixels by trusting a model of how the light got there. This chapter is about
the one number that is not. The Einstein radius $\theta_{\mathrm{E}}$ falls out
of pure geometry — where the images are, and how far the lens and source sit
from us and each other — and it does so *before* you commit to any assumption
about how mass is arranged inside the lens galaxy. That is why it is the
number every lensing paper leads with, the number a photometric pipeline can
estimate from a spectrum alone, and the number this repository's own
reproductions reconstruct to a percent or two even from an independent code
path. By the end of this chapter you can derive $\theta_{\mathrm{E}}$ from a
galaxy's velocity dispersion, explain in one line why the enclosed mass
$M(<\theta_{\mathrm{E}})$ does not care what density profile you assumed, and
see both claims checked against this repository's own numbers. Chapters 20
through 26 spend the rest of the book on $\gamma$, which is a much harder
measurement precisely *because* it does not have this chapter's kind of
protection.

!!! abstract "What you can skip"
    If you already know that an Einstein ring is what a circularly symmetric
    lens makes of a perfectly aligned source, skip [What $\theta_{\mathrm{E}}$
    is](#what-theta-e-is) and start at [The mean-convergence
    identity](#the-mean-convergence-identity). If you are comfortable setting
    up and evaluating a definite integral, you can take the projection
    integral in [$\theta_{\mathrm{E}}$ from $\sigma_v$](#theta-e-from-sigma-v)
    on faith and skip straight to the boxed result.

## What $\theta_{\mathrm{E}}$ is { #what-theta-e-is }

Recall the lens equation from [Ch. 17](17-lens-equation.md#the-lens-equation):
$\boldsymbol\beta = \boldsymbol\theta - \boldsymbol\alpha(\boldsymbol\theta)$,
mapping an image-plane position $\boldsymbol\theta$ to the source-plane
position $\boldsymbol\beta$ it came from. Now specialize to the cleanest
possible geometry: a lens whose mass distribution is circularly symmetric
(a good approximation for a single massive elliptical viewed close to
face-on), with the source sitting exactly on the optical axis,
$\boldsymbol\beta = \mathbf{0}$.

Because the lens is circularly symmetric, $\boldsymbol\alpha(\boldsymbol\theta)$
points radially and its magnitude depends only on $\theta = |\boldsymbol\theta|$.
Restrict the lens equation to any fixed position angle and it becomes a
single scalar equation in the radius alone,
$\theta - \alpha(\theta) = 0$. Nothing in that equation refers to which
direction you picked — so if $\theta_{\mathrm{E}}$ solves it, it solves it at
*every* position angle simultaneously. The solution set is not a point; it is
an entire circle. That circle is the **Einstein ring**, and its radius is the
**Einstein radius** $\theta_{\mathrm{E}}$.

Break the alignment even slightly and the ring splits into the discrete
images [Ch. 17](17-lens-equation.md#multiple-images) already introduced —
two for a source outside the lens's inner caustic, four for a source inside
it — with typical image separations and arc lengths set by $\theta_{\mathrm{E}}$
itself. $\theta_{\mathrm{E}}$ is the master angular scale of a lensing system:
everything else in the image — the separation of a double, the diameter of a
quad, the length of an arc — is a number of order $\theta_{\mathrm{E}}$, not
an independent scale.

## The mean-convergence identity { #the-mean-convergence-identity }

The defining condition $\theta_{\mathrm{E}} - \alpha(\theta_{\mathrm{E}}) = 0$
looks like it depends on the full shape of $\alpha(\theta)$, which in turn
depends on the full radial mass profile. It does not. What it depends on is a
single number: the *average* convergence enclosed within
$\theta_{\mathrm{E}}$, regardless of how that convergence is arranged inside.

Start from [Ch. 6](06-vector-calculus.md#poisson-for-lensing)'s statement of
the lens equation trio, $\nabla^2\psi = 2\kappa$, $\boldsymbol\alpha =
\nabla\psi$. For a circularly symmetric $\psi(\theta)$, the 2-D Laplacian in
polar coordinates acting on a radius-only function is
$\nabla^2\psi = \theta^{-1}\, d(\theta\, d\psi/d\theta)/d\theta$, and the
radial deflection is $\alpha(\theta) = d\psi/d\theta$. So

$$
\frac{1}{\theta}\frac{d}{d\theta}\big(\theta\,\alpha(\theta)\big) = 2\kappa(\theta)
\quad\Longrightarrow\quad
\frac{d}{d\theta}\big(\theta\,\alpha(\theta)\big) = 2\,\theta\,\kappa(\theta).
$$

Integrate from $0$ to $\theta$ — $\theta\alpha(\theta)$ starts at $0$ for any
profile without a point mass at the center —

$$
\theta\,\alpha(\theta) = 2\int_0^\theta \kappa(t)\, t\, dt
\quad\Longrightarrow\quad
\alpha(\theta) = \theta\,\bar\kappa(\theta), \qquad
\bar\kappa(\theta) \equiv \frac{2}{\theta^2}\int_0^\theta \kappa(t)\,t\,dt.
\label{eq:alpha-kbar}
$$

$\bar\kappa(\theta)$ is the mean convergence enclosed within radius $\theta$ —
exactly the quantity `mean_kappa_within` computes at
`site/guide_src/lensing.py:179`. Equation $\eqref{eq:alpha-kbar}$ says the
deflection at any radius is fixed entirely by how much convergence sits
*inside* that radius; nothing outside it contributes. (This is the lensing
analogue of Newton's shell theorem: a spherical shell of mass exerts no net
force on anything inside it, so only the enclosed mass matters. Same
statement, one dimension flatter.)

Now substitute $\theta = \theta_{\mathrm{E}}$ into the Einstein-ring condition
$\theta_{\mathrm{E}} = \alpha(\theta_{\mathrm{E}})$ from the previous section,
and use $\eqref{eq:alpha-kbar}$:

$$
\theta_{\mathrm{E}} = \theta_{\mathrm{E}}\,\bar\kappa(\theta_{\mathrm{E}})
\quad\Longrightarrow\quad
\bar\kappa(\theta_{\mathrm{E}}) = 1.
$$

This is the whole content of the phrase "Einstein radius": **the radius at
which the mean interior convergence is exactly critical.** It is a
*definition*, derived from nothing but the Poisson equation and the ring
condition — it says nothing about whether the lens is a point mass, an SIS,
an SIE, or an EPL with some particular $\gamma$. Any circularly symmetric
profile has *some* radius where its own enclosed mean reaches 1, and that
radius is, by construction, what everyone calls $\theta_{\mathrm{E}}$.

Check it numerically on the simplest case, the singular isothermal sphere
(SIS), whose convergence is $\kappa(\theta) = \theta_{\mathrm{E}}/(2\theta)$
(derived from first principles in the next section). Evaluating $\bar\kappa$
by direct quadrature at $\theta = \theta_{\mathrm{E}} = 1$ gives $0.999995$
<!-- check: ch19.sis_mean_kappa_at_theta_e = 0.999995 ± 0.0001 -->
— not exactly $1$ only because the numerical integral starts a hair off the
origin rather than because the identity is approximate.

<figure markdown="span">
  ![Mean interior convergence for an SIS as a function of radius, crossing the critical value 1 exactly at the Einstein radius](figures/ch19-mean-kappa-light.svg#only-light){ width="90%" }
  ![Mean interior convergence for an SIS as a function of radius, crossing the critical value 1 exactly at the Einstein radius](figures/ch19-mean-kappa-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 19.1.** $\bar\kappa(<\theta)$ for an SIS,
  computed by the quadrature in $\eqref{eq:alpha-kbar}$. It crosses the critical value
  $\bar\kappa = 1$ at $\theta = \theta_{\mathrm{E}}$ by construction — that
  crossing point *is* the definition of $\theta_{\mathrm{E}}$, not a fitted
  feature of this particular curve.</figcaption>
</figure>

## $\theta_{\mathrm{E}}$ from $\sigma_v$ { #theta-e-from-sigma-v }

[Ch. 10](10-galaxies.md#velocity-dispersion) derived the isothermal sphere's
3-D density from hydrostatic equilibrium: $\rho(r) = \sigma_v^2/(2\pi G r^2)$,
where $\sigma_v$ is the galaxy's stellar velocity dispersion — a number a
spectrum measures directly, with no imaging required. Project that to a
surface density with the line-of-sight integral
[Ch. 3](03-integrals.md#the-abel-projection) introduced in general:

$$
\Sigma(R) = \int_{-\infty}^{\infty} \rho\!\left(\sqrt{R^2+z^2}\right) dz
= \frac{\sigma_v^2}{2\pi G}\int_{-\infty}^{\infty} \frac{dz}{R^2+z^2}
= \frac{\sigma_v^2}{2\pi G}\cdot\frac{\pi}{R}
= \frac{\sigma_v^2}{2GR}.
$$

The middle integral is $\left[\theta \mapsto \arctan(z/R)/R\right]_{-\infty}^{\infty} = \pi/R$
— a table integral, no new machinery. This is exactly the $\rho \sim r^{-\gamma}
\Rightarrow \Sigma \sim R^{1-\gamma}$ scaling of Ch. 3 at $\gamma = 2$, with
the constant filled in.

Convert to convergence with $\Sigma_{\mathrm{cr}}$
([Ch. 15](15-distances.md#sigma-crit)) and physical radius
$R = D_{\mathrm{d}}\theta$: $\kappa(\theta) = \sigma_v^2 / (2GD_{\mathrm{d}}\theta\Sigma_{\mathrm{cr}})$,
which has the form $A/\theta$. Plugging $\kappa(t)=A/t$ into
$\eqref{eq:alpha-kbar}$ gives $\bar\kappa(\theta) = 2A/\theta$ for *any*
$\theta$ — for this particular profile the mean is exactly twice the local
value everywhere, not just at $\theta_{\mathrm{E}}$. Setting
$\bar\kappa(\theta_{\mathrm{E}}) = 1$ gives $\theta_{\mathrm{E}} = 2A =
\sigma_v^2/(GD_{\mathrm{d}}\Sigma_{\mathrm{cr}})$. Substitute
$\Sigma_{\mathrm{cr}} = c^2D_{\mathrm{s}}/(4\pi GD_{\mathrm{d}}D_{\mathrm{ds}})$
([Ch. 15](15-distances.md#sigma-crit)) and $D_{\mathrm{d}}$ cancels outright:

$$
\theta_{\mathrm{E}} = 4\pi\left(\frac{\sigma_v}{c}\right)^2\frac{D_{\mathrm{ds}}}{D_{\mathrm{s}}}.
\label{eq:thetaE-sv}
$$

This is Hsu+2025's Eq. 1, used verbatim at
`reproductions/hsu-2025/07_classify_einstein_dimple.py:116` and
`site/guide_src/lensing.py:170`. It is quadratic in $\sigma_v$ and linear in
the distance ratio — and $D_{\mathrm{ds}} \neq D_{\mathrm{s}} - D_{\mathrm{d}}$
([Ch. 15](15-distances.md#distances-do-not-add)) matters here exactly as much
as it did there, since $D_{\mathrm{ds}}$ enters $\eqref{eq:thetaE-sv}$
directly.

For a fiducial massive elliptical — $\sigma_v = 250$ km/s at $z_l=0.5$,
$z_s=2.0$, the same system [Ch. 10](10-galaxies.md#velocity-dispersion)
already quoted — $\eqref{eq:thetaE-sv}$ gives $\theta_{\mathrm{E}} = 1.145''$
<!-- check: ch19.theta_e_typical = 1.145 ± 0.001 -->.

Now apply it to real data. `reproductions/cikota-2023` reproduces
Cikota et al. 2023's Einstein cross DESI-253.2534+26.8843 — "a massive
elliptical lens galaxy (L1)" (`reproductions/cikota-2023/papers/main.tex:69`)
— on public DESI Legacy imaging, fitting an imaging Einstein radius of
$2.103''$ against the paper's own, higher-resolution $2.520''$
(`reproductions/cikota-2023/papers/main.tex:213`), and inverting $\eqref{eq:thetaE-sv}$ to
read off an implied $\sigma_{\mathrm{SIE}} = 347$ km/s against the paper's
spectroscopic $379 \pm 2$ km/s
(`reproductions/cikota-2023/papers/main.tex:214`). Feed that recovered
$347$ km/s back into $\eqref{eq:thetaE-sv}$ at an illustrative lens/source redshift pair —
$z_l = 0.271$, $z_s = 0.897$, chosen only to exercise the formula at some
concrete geometry, *not* this system's own spectroscopic redshifts (those
are $z_{\mathrm{L1}} = 0.636$, $z_s = 2.597$; see Exercise 19.3 for what
using them instead does) — and it returns

$$
\theta_{\mathrm{E}} = 2.233''
$$

<!-- check: ch19.theta_e_cikota = 2.233 ± 0.01 -->,

landing between the repository's own imaging fit ($2.103''$) and the
published, sharper-PSF fit ($2.520''$). That is not a coincidence of the
particular redshift pair chosen — $347$ km/s was itself backed out of a
$\theta_{\mathrm{E}}$ in roughly that range — but it is a genuine sanity
check that $\eqref{eq:thetaE-sv}$ is behaving the way a quadratic-in-$\sigma_v$,
linear-in-geometry relation should: neither wildly off nor suspiciously
exact.

Why does the campaign fit the full image rather than stopping at $\eqref{eq:thetaE-sv}$?
Because $\sigma_v$ alone is a single scalar — it fixes $\theta_{\mathrm{E}}$
but says nothing about ellipticity, position angle, external shear, or the
density slope $\gamma$ ([Ch. 20](20-profiles.md#the-epl-and-gamma)), all of
which the pixels constrain and a spectrum does not. $\eqref{eq:thetaE-sv}$ is the
back-of-the-envelope check every lensing paper runs before trusting an
imaging fit — not a replacement for one.

## Mass inside $\theta_{\mathrm{E}}$ { #mass-inside-theta-e }

Equation $\eqref{eq:alpha-kbar}$'s defining fact, $\bar\kappa(\theta_{\mathrm{E}}) = 1$,
means the *average* surface density inside the Einstein radius is exactly
$\Sigma_{\mathrm{cr}}$ — regardless of whether the true profile is an SIS, an
SIE, a cored halo, or anything else circularly averaged. So the enclosed mass
follows from geometry alone:

$$
M(<\theta_{\mathrm{E}}) = \Sigma_{\mathrm{cr}}\cdot\pi R_{\mathrm{E}}^2,
\qquad R_{\mathrm{E}} = D_{\mathrm{d}}\,\theta_{\mathrm{E}},
$$

implemented at `site/guide_src/cosmo.py:66`. This is the reason
$M(<\theta_{\mathrm{E}})$, not the density slope, is what every lensing paper
reports with confidence.

!!! tip "You already know this"
    $M(<\theta_{\mathrm{E}})$ is a checksum, not a fit. It depends only on the
    *total* enclosed inside a fixed boundary, never on how that total is
    arranged within it — the same way a histogram's cumulative count up to a
    bin edge is fixed by the counts below it and is completely blind to which
    bin each of those counts landed in. $\gamma$ is a claim about the shape
    inside that boundary; $M(<\theta_{\mathrm{E}})$ is a claim about the sum,
    and sums are far more robust to being wrong about the shape.

[Ch. 15](15-distances.md#the-carousel) already computed this for the Carousel
cluster lens (Sheu et al. 2024): $z_l=0.49$, $\theta_{\mathrm{E}}=13.03''$
with respect to $z_s=1.432$, giving $\Sigma_{\mathrm{cr}} = 2.376\times10^{15}\ M_\odot/\mathrm{Mpc}^2$
<!-- check: ch15.sigma_crit_carousel = 2.376e15 ± 5e12 --> and
$M(<\theta_{\mathrm{E}}) = 4.621\times10^{13}\ M_\odot$
<!-- check: ch15.mass_within_theta_e_carousel = 4.621e13 ± 1e11 -->. Sheu et
al.'s own published fit reports $4.78\times10^{13}\ M_\odot$
(`reproductions/sheu-2024b/README.md:24`) — a ratio of $0.9667$
<!-- check: ch15.mass_repro_vs_paper_ratio = 0.9667 ± 0.001 -->, $3.3\%$ low.
That residual is not evidence against the identity; it is the price of the
formula above assuming a *circular* aperture, while the real lens is an
elliptical EPL with $q=0.87$ and external shear
$\gamma_{\mathrm{ext}} = 0.11$ (`reproductions/sheu-2024b/README.md:20`–`22`) —
exactly what the repository's own reproduction says drives the gap
(`reproductions/sheu-2024b/README.md:46`). Feed the *published* best-fit
parameters, ellipticity and all, into a real ray-traced lens model instead of
the circular formula, and the same reproduction's own deflection-based check
of $\bar\kappa(\theta_{\mathrm{E}})$ — literally $\alpha(\theta_{\mathrm{E}})/\theta_{\mathrm{E}}$,
the left-hand side of this chapter's central identity, read off a fitted
`lenstronomy` model's own analytic EPL deflection rather than the closed-form
SIS quadrature above — returns $1.001$
(`reproductions/sheu-2024b/README.md:45`; source at
`reproductions/sheu-2024b/04_setup_multiplane.py:132`–`134`). Two independent
numerical routes, a hand-rolled SIS quadrature and a fitted multi-galaxy
`lenstronomy` model, land on the same identity to three significant figures.

## Connect to the repo { #connect }

- `site/guide_src/lensing.py:170` (`theta_e_sis`) and `:179`
  (`mean_kappa_within`) — $\eqref{eq:thetaE-sv}$ and the mean-convergence quadrature behind
  Figure 19.1, implemented from scratch to match the repository's own
  conventions.
- `site/guide_src/cosmo.py:45` (`sigma_crit`), `:66` (`mass_within_theta_e`),
  `:79` (`theta_e_from_sigma_v`) — the three cosmology-dependent quantities
  this chapter derives.
- [`reproductions/hsu-2025/07_classify_einstein_dimple.py:103`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/hsu-2025/07_classify_einstein_dimple.py#L103)–`118` —
  $\eqref{eq:thetaE-sv}$ applied at survey scale to 13,530 DESI galaxy pairs
  (`reproductions/hsu-2025/README.md:24`), no imaging fit required.
- [`reproductions/cikota-2023/papers/main.tex:171`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/cikota-2023/papers/main.tex#L171)–`184`
  and [`reproductions/cikota-2023/README.md`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/cikota-2023/README.md) —
  the Einstein cross $\theta_{\mathrm{E}} \leftrightarrow \sigma_{\mathrm{SIE}}$
  round trip.
- [`reproductions/sheu-2024b/04_setup_multiplane.py:120`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/sheu-2024b/04_setup_multiplane.py#L120)–`139`
  and [`reproductions/sheu-2024b/README.md`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/sheu-2024b/README.md) —
  the mean-convergence identity checked against a real fitted multi-plane
  model, both by mass and by deflection.

## Exercises { #exercises }

??? question "Exercise 19.1 — the identity holds for every $\gamma$, not just the SIS"
    The general EPL convergence (`site/guide_src/lensing.py:81`, circular
    case $q=1$) is $\kappa(\theta) = \frac{3-\gamma}{2}\left(\theta_{\mathrm{E}}/\theta\right)^{\gamma-1}$.
    Using $\eqref{eq:alpha-kbar}$, show that $\bar\kappa(\theta_{\mathrm{E}}) = 1$ for
    *every* value of $\gamma < 3$ — not just the isothermal $\gamma=2$ case
    worked out in the main text — and say in one sentence what this implies
    about how much information $\theta_{\mathrm{E}}$ alone carries about
    $\gamma$.

    ??? success "Solution"

        $$
        \bar\kappa(\theta) = \frac{2}{\theta^2}\int_0^\theta \frac{3-\gamma}{2}
        \left(\frac{\theta_{\mathrm{E}}}{t}\right)^{\gamma-1} t\, dt
        = \frac{(3-\gamma)\,\theta_{\mathrm{E}}^{\gamma-1}}{\theta^2}
        \int_0^\theta t^{2-\gamma}\, dt
        = \frac{(3-\gamma)\,\theta_{\mathrm{E}}^{\gamma-1}}{\theta^2}\cdot
        \frac{\theta^{3-\gamma}}{3-\gamma}
        = \left(\frac{\theta_{\mathrm{E}}}{\theta}\right)^{\gamma-1}.
        $$

        At $\theta = \theta_{\mathrm{E}}$ this is $1^{\gamma-1} = 1$,
        independent of $\gamma$. (Setting $\gamma=2$ recovers
        $\bar\kappa(\theta) = \theta_{\mathrm{E}}/\theta$, exactly twice the
        SIS's own $\kappa(\theta) = \theta_{\mathrm{E}}/(2\theta)$ from the
        main text — consistent with what the $\theta_{\mathrm{E}}$-from-$\sigma_v$
        derivation found directly.) Because *every* circular EPL, whatever
        its $\gamma$, satisfies $\bar\kappa(\theta_{\mathrm{E}})=1$ at its own
        $\theta_{\mathrm{E}}$, a measurement of $\theta_{\mathrm{E}}$ alone
        carries essentially zero information about $\gamma$ — the two are
        cleanly separated by construction. That separation is exactly why
        [Ch. 20](20-profiles.md#the-epl-and-gamma) onward needs the full
        image, not just the ring radius, to pin $\gamma$ down.

??? question "Exercise 19.2 — rebuild $\theta_{\mathrm{E}} = 1.145''$ by hand"
    Using only [Ch. 15](15-distances.md#the-carousel)'s already-computed
    distances $D_{\mathrm{d}}(z=0.5)=1259.08$ Mpc
    <!-- check: ch15.d_d_05 = 1259.08 ± 0.01 -->,
    $D_{\mathrm{s}}(z=2.0)=1726.62$ Mpc
    <!-- check: ch15.d_s_20 = 1726.62 ± 0.01 -->, and
    $D_{\mathrm{ds}}(0.5,2.0)=1097.08$ Mpc
    <!-- check: ch15.d_ds_05_20 = 1097.08 ± 0.01 -->, plug $\sigma_v=250$
    km/s into $\eqref{eq:thetaE-sv}$ and reproduce this chapter's $\theta_{\mathrm{E}} = 1.145''$
    worked example without calling `cosmo.py` at all.

    ??? success "Solution"

        $$
        \frac{D_{\mathrm{ds}}}{D_{\mathrm{s}}} = \frac{1097.08}{1726.62} \approx 0.6354,
        \qquad
        \left(\frac{\sigma_v}{c}\right)^2 = \left(\frac{250}{299{,}792.458}\right)^2
        \approx 6.954\times10^{-7}.
        $$

        $$
        \theta_{\mathrm{E}} = 4\pi \times 6.954\times10^{-7} \times 0.6354\ \text{rad}
        \approx 5.553\times10^{-6}\ \text{rad}.
        $$

        Multiply by $206{,}264.806$ arcsec/rad
        ([Ch. 9](09-units.md#angles-on-the-sky)) to get $\theta_{\mathrm{E}}
        \approx 1.145''$ <!-- check: ch19.theta_e_typical = 1.145 ± 0.001 -->,
        matching the worked example exactly — the whole formula, run by hand
        with a calculator, using nothing this chapter has not already derived.

??? question "Exercise 19.3 — the Einstein cross's own redshifts"
    Repeat the Cikota calculation from the main text, but with the system's
    own spectroscopic redshifts, $z_{\mathrm{L1}}=0.636$, $z_s=2.597$
    (`reproductions/cikota-2023/papers/main.tex:166`), instead of the
    illustrative pair used there. Run
    `cosmo.theta_e_from_sigma_v(347.0, 0.636, 2.597)` and compare the result
    to the repository's own imaging fit of $2.103''$. Then explain why it
    does not land *exactly* on $2.103''$, given that $347$ km/s was itself
    obtained by inverting $\eqref{eq:thetaE-sv}$ at essentially this same geometry.

    ??? success "Solution"
        ```python
        import cosmo
        cosmo.theta_e_from_sigma_v(347.0, 0.636, 2.597)   # roughly 2.12"
        ```
        That lands far closer to the repository's own $2.103''$ imaging fit
        than the $2.233''$ illustrative-pair result did — as it should, since
        this is now the right geometry. It is still not an exact match,
        because `cosmo.py` fixes cosmology at $H_0=70$, $\Omega_m=0.3$
        throughout (`site/guide_src/cosmo.py:29`) — the convention the
        repository's other cosmology-dependent reproductions share — while
        `reproductions/cikota-2023` follows its own paper in adopting Planck18
        ($H_0=67.4$, $\Omega_m=0.315$) for this specific conversion. Swap in
        $H_0=67.4$ and the small residual gap closes almost entirely. The
        lesson is not that $\eqref{eq:thetaE-sv}$ is imprecise; it is that *which*
        cosmology you plug in is itself an input worth tracking, the same
        point [Ch. 14](14-frw.md#the-density-parameters) makes about
        $\Omega_m$ in general.

??? question "Exercise 19.4 — is the 3.3% mass gap a problem?"
    [Ch. 15](15-distances.md#the-carousel)'s circular-aperture estimate,
    $M(<\theta_{\mathrm{E}}) = 4.621\times10^{13}\ M_\odot$, sits $3.3\%$
    below Sheu et al.'s published $4.78\times10^{13}\ M_\odot$. Using this
    chapter's mean-convergence identity, argue whether that gap should worry
    you about the identity itself, and what it *would* mean if the gap were
    instead $30\%$.

    ??? success "Solution"
        The identity $\bar\kappa(\theta_{\mathrm{E}})=1$ was derived for a
        *circularly symmetric* lens; the real Carousel lens is an elliptical
        EPL ($q=0.87$) plus external shear ($\gamma_{\mathrm{ext}}=0.11$), so
        $\pi R_{\mathrm{E}}^2$ is only an approximation to the true enclosed
        area, and a few percent of discrepancy is exactly what that
        approximation should cost — confirmed by the repository's own
        independent deflection-based check landing at $1.001$
        (`reproductions/sheu-2024b/README.md:45`), which uses the actual
        elliptical model rather than the circular formula and closes almost
        all of the gap. A $3.3\%$ mismatch is the ellipticity correction, not
        a crack in the identity. A $30\%$ mismatch would be a different
        matter — ellipticity this modest cannot move a profile-independent
        geometric quantity that far, so a gap that large would point at a
        wrong redshift, a wrong cosmology, or a real error in either
        $\theta_{\mathrm{E}}$ or $\Sigma_{\mathrm{cr}}$, not at the assumption
        of circularity.
