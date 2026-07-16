# FRW and the Friedmann equations

Chapter 13 gave you the scale factor $a(t)$ and showed that redshift is a
record of how much $a$ has grown since a photon left its source. This chapter
gives you the equation that decides how $a(t)$ actually evolves — the
Friedmann equation — and the handful of numbers, $\Omega_m$ and
$\Omega_\Lambda$, that fix its trajectory. By the end you can write down
`FlatLambdaCDM(H0=70, Om0=0.3)`, the one line this repository's cosmology
begins and ends with, and say exactly what each of its two arguments asserts,
including the third number it fixes that you never typed.

!!! abstract "What you can skip"
    If you already have a GR or cosmology course behind you: skip the
    Newtonian derivation of the Friedmann equation below (you already have the
    real one from Einstein's field equations) and the review of $F = GMm/r^2$.
    Go straight to [The density parameters](#the-density-parameters), where the only repo-specific
    content lives — what `FlatLambdaCDM(70, 0.3)` fixes, and the fact that
    $\Omega_m + \Omega_\Lambda = 1$ is a *definition*, not a measurement (the
    measurement is that $\Omega_k = 0$). If you're willing to take the
    Friedmann equation on faith entirely, skip to
    [Connect to the repo](#connect): none of this chapter touches $\gamma$, and
    that fact is worth knowing on its own.

## The FRW metric { #the-frw-metric }

A metric is a distance formula: a rule for turning a small coordinate
separation into a physical (proper) length. In flat 2-D Euclidean space you
already know it as $ds^2 = dx^2 + dy^2$ — Pythagoras, applied to
infinitesimally close points. General relativity's only real complication is
that the coefficients in that formula are allowed to depend on where, and
when, you are.

!!! tip "You already know this"
    A metric is a (possibly position- or time-dependent) quadratic form that
    turns coordinate differences into a distance. That is exactly the job a
    covariance-weighted norm does when you whiten a feature vector, and it is
    exactly the job the mass matrix does in [Ch. 23](23-samplers.md#hmc-and-the-metric)'s
    HMC sampler, which turns a gradient into a step in parameter space. The FRW
    metric below is the same kind of object, applied to spacetime.

The **cosmological principle** — on large scales the universe looks the same
at every point (homogeneous) and the same in every direction from that point
(isotropic) — is an assumption, well tested but unprovable in general. It is
also an enormous constraint: it says the distance formula for the whole
universe can depend on time through only *one* function and on space through
only *one* constant. That function is $a(t)$. The constant is a curvature
index $k$. The result is the **Friedmann–Robertson–Walker (FRW) metric**:

$$
ds^2 = -c^2\,dt^2 + a(t)^2\left[\frac{dr^2}{1-kr^2} + r^2\,d\Omega^2\right]
$$

Here $t$ is cosmic time — the clock every observer at rest relative to the
overall expansion agrees on — and $r$ is a *comoving* radial coordinate: fixed
for an object that moves only with the Hubble flow, exactly the coordinate
whose separation $a(t)$ stretches into a physical one in
[Ch. 13](13-expansion.md#scale-factor). The term $d\Omega^2$ is the ordinary
solid-angle piece from spherical coordinates — the same combination you'd get
unrolling latitude and longitude on a globe. This guide does not write out its
two angles individually: those two symbols are reserved elsewhere for the
lens-plane position ([Ch. 17](17-lens-equation.md#the-lens-equation)), and
this repo's overloading habit is exactly what this notation is trying to
avoid repeating.

$k$ takes one of three values (after an overall rescaling of $r$): $k=-1$
(open, hyperbolic, infinite volume), $k=0$ (flat, ordinary Euclidean
geometry), or $k=+1$ (closed, a 3-sphere, finite volume). This repository
never considers the first or the third: every cosmology object it builds is
`FlatLambdaCDM`, which hard-codes $k=0$ and does not expose a curvature
argument at all. Comoving coordinates are the ones a galaxy catalog is built
in; the *proper* separation between two comoving points at a fixed time is
$a(t)$ times their (constant) comoving separation — the same idea Ch. 13
introduced for a single photon's wavelength, now embedded in an actual
distance formula. [Ch. 15](15-distances.md#three-distances) turns this into
the angular-diameter distance the lensing code calls directly.

Deriving this metric from Einstein's field equations requires the tensor
calculus this guide deliberately skips. You do not need that derivation to
use the metric — you need it to know what makes $a(t)$ move, which is the
subject of the next section.

## The Friedmann equation, derived as energy conservation { #friedmann }

You can get the right equation for $\dot a$ without any tensor calculus, using
nothing but Newtonian energy conservation — a genuine, standard trick, not a
hand-wave, though it comes with a caveat stated at the end of this section.

Pick any comoving observer as an origin and consider a sphere around them of
fixed comoving radius $r_c$, small enough that space inside it looks
uniform. Its physical radius is $R(t) = a(t)\,r_c$, and by homogeneity it
encloses a uniform matter density $\rho(t)$, hence a mass
$M(R) = \frac{4}{3}\pi R^3 \rho$. Put a test particle of mass $m$ on its
surface. By Newton's shell theorem the sphere pulls on that particle exactly
as if all its mass sat at the center, so ordinary energy conservation gives

$$\frac{1}{2}m\dot R^2 - \frac{GM(R)m}{R} = E$$

with $E$ constant — nothing is adding or removing energy from the particle.
Substituting $M(R)$ and writing the (still-constant) $E$ in the suggestive
form $E \equiv -\frac{1}{2}mkc^2 r_c^2$:

$$\frac{1}{2}\dot R^2 - \frac{4}{3}\pi G\rho R^2 = -\frac{1}{2}kc^2 r_c^2$$

Divide through by $\frac{1}{2}R^2 = \frac{1}{2}a^2 r_c^2$. The comoving radius
$r_c$ — which was arbitrary, the radius of a sphere you drew, not a physical
scale — cancels completely, because $\dot R/R = \dot a/a$ for *any* $r_c$.
That cancellation is not a bookkeeping convenience; it is required, because
the resulting law has to hold for every comoving observer, not just the one
who happens to sit at the center of your particular sphere. What survives is
the first **Friedmann equation**:

$$
\begin{equation}\label{eq:friedmann}
H^2 \equiv \left(\frac{\dot a}{a}\right)^2 = \frac{8\pi G}{3}\rho - \frac{kc^2}{a^2}
\end{equation}
$$

with the same curvature index $k$ that appears in the metric. A denser
universe ($\rho$ larger) expands faster; positive curvature ($k=+1$) is a
literal drag term that can eventually halt and reverse the expansion, exactly
as positive total energy versus a negative one decides whether a thrown ball
escapes to infinity or falls back.

**The honest caveat.** This derivation is Newtonian, and Newtonian gravity has
no curvature and no dynamical role for pressure. That $\eqref{eq:friedmann}$
comes out *exactly* right is a genuine feature of general relativity applied
to a homogeneous, isotropic spacetime (a cousin of Birkhoff's theorem), not a
coincidence you should distrust — but it is also not a proof, and it does not
extend to the *second* Friedmann equation, the one governing $\ddot a$, which
needs pressure to source gravity. Pressure gravitating at all has no
Newtonian analogue; it is a genuinely relativistic effect, in the same family
as the factor of two you'll meet in
[Ch. 16](16-deflection.md#the-factor-of-two) when light bends twice as far as
Newton predicts. This repo never needs the second equation: it treats dark
energy as a fixed extra density, not as the solution of a dynamical equation,
which is exactly the next paragraph.

**Adding dark energy and matter dilution.** Treat the cosmological constant
$\Lambda$ as an extra density component that never dilutes,
$\rho_\Lambda \equiv \Lambda c^2/(8\pi G) = \text{const}$ — this capital
$\Lambda$ is unrelated to the lowercase tempering parameter $\lambda$ you'll
meet in [Ch. 23](23-samplers.md#tempering-and-smc); case is the only thing
distinguishing them, so read carefully. Ordinary matter, by contrast, does
dilute: mass conservation in a fixed comoving sphere,
$M = \rho_m(t)\cdot\frac{4}{3}\pi R(t)^3 = \text{const}$ (nothing creates or
destroys matter, only accumulation the way [Ch. 3](03-integrals.md#accumulation)
already had you doing over a volume), together with $R \propto a$, forces
$\rho_m(t) \propto a(t)^{-3}$. Put both pieces into
$\eqref{eq:friedmann}$ with $\rho = \rho_m + \rho_\Lambda$.

**A reference density.** At any moment, relabel the right-hand side's overall
scale as a density: $\rho_{\mathrm{crit}}(t) \equiv \dfrac{3H(t)^2}{8\pi G}$. This
is not a special density in the physics — it is what $\rho$ would have to
equal for $k=0$ to hold exactly at that instant, given the $H$ that instant
actually has. It is the natural yardstick for the next section, and it is
already enough to compute two numbers with only $H_0$ in hand. This
repository fixes $H_0 = 70$ km/s/Mpc in every one of the three places
cosmology appears (the next section names them), giving

$$\rho_{\mathrm{crit,0}} = \frac{3H_0^2}{8\pi G} \approx 1.360\times10^{11}\ M_\odot / \text{Mpc}^3$$

<!-- check: ch14.rho_crit0_msun_mpc3 = 1.360e11 ± 0.001e11 -->

— roughly one solar mass smeared over a cube 634 light-years on a side, which
is the density the entire observable universe would need, on average, to be
spatially flat. [Ch. 13](13-expansion.md#hubbles-law) already showed you the
companion length and time scales that come from $H_0$ alone, the Hubble
distance and Hubble time; $\rho_{\mathrm{crit,0}}$ is the third such scale, and it
is the one that needs a density, not just a rate, to define.

## Omega_m, Omega_Lambda, and what flat means { #the-density-parameters }

Normalize any density by $\rho_{\mathrm{crit}}$ and you get a dimensionless
**density parameter**: $\Omega_m \equiv \rho_{m,0}/\rho_{\mathrm{crit,0}}$,
$\Omega_\Lambda \equiv \rho_{\Lambda}/\rho_{\mathrm{crit,0}}$, and, folding the
curvature term into the same units,
$\Omega_k \equiv -kc^2/(H_0^2 a_0^2)$ (with the convention $a_0 \equiv a(t_0) = 1$
today). Evaluate $\eqref{eq:friedmann}$ at $t=t_0$ and divide every term by
$H_0^2$:

$$\Omega_m + \Omega_\Lambda + \Omega_k = 1$$

This identity costs you nothing: it is what dividing the Friedmann equation by
$H_0^2$ *always* produces, for any universe, whatever its actual curvature. It
is not evidence that the universe is flat. **Flatness is the separate,
empirical claim that $\Omega_k = 0$** — that curvature's contribution
happens to vanish. Once you assert that, the identity above collapses to
$\Omega_m + \Omega_\Lambda = 1$, and specifying $\Omega_m$ alone fixes
$\Omega_\Lambda$ for free.

**What `FlatLambdaCDM(70, 0.3)` actually asserts.** Three independent places in
this repository build the identical cosmology object:
`FlatLambdaCDM(H0=70, Om0=0.3)`, at `reproductions/hsu-2025/07_classify_einstein_dimple.py:50`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/hsu-2025/07_classify_einstein_dimple.py#L50)),
`reproductions/sheu-2024b/04_setup_multiplane.py:35`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/sheu-2024b/04_setup_multiplane.py#L35)),
and `reproductions/sheu-2023/05_lightcurve_salt3.py:57`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/sheu-2023/05_lightcurve_salt3.py#L57)).
That one line asserts three numbers, only one of which you typed:

<!-- check: ch14.H0 = 70.0 ± 0.01 -->
<!-- check: ch14.Om0 = 0.3 ± 0.001 -->
<!-- check: ch14.Ode0 = 0.7 ± 0.001 -->
<!-- check: ch14.Ok0 = 0.0 ± 1e-9 -->
<!-- check: ch14.flatness_sum = 1.0 ± 1e-9 -->

$H_0 = 70$ and $\Omega_{m,0} = 0.3$ are the two you chose. `astropy`'s
`FlatLambdaCDM` class does not even accept a curvature argument — it forces
$\Omega_{k,0}=0$ by construction — so $\Omega_{\Lambda,0}$ is not independently
set, it is *computed*, as $1 - \Omega_{m,0} - \Omega_{k,0} = 0.7$. Check the
sum yourself and you get exactly $1$, as the identity above guarantees it must.

With $\Omega_m$ and $\Omega_\Lambda$ in hand, $\eqref{eq:friedmann}$ becomes a
concrete, checkable function of redshift. Using $1+z = 1/a$
([Ch. 13](13-expansion.md#redshift-is-expansion)) and the matter-dilution
result from the previous section,

$$H(z)^2 = H_0^2\left[\Omega_m(1+z)^3 + \Omega_\Lambda\right]$$

Build the right-hand side yourself from nothing but $H_0=70$, $\Omega_m=0.3$,
$\Omega_\Lambda=0.7$, at $z=0.5$, and you should get

<!-- check: ch14.hz_manual_z0p5 = 91.604 ± 0.01 -->

in km/s/Mpc. That is not a number this guide asserts and asks you to trust —
it is `astropy`'s own `FlatLambdaCDM.H(0.5)`, computed independently from the
metric machinery the library actually runs, and the two agree to machine
precision:

<!-- check: ch14.hz_match_diff = 0.0 ± 1e-9 -->

The same $H_0$, $\Omega_m$, $\Omega_\Lambda$ also fix the age of the universe
— the time integral of $1/[(1+z)H(z)]$ back from today to $a=0$ — at

<!-- check: ch14.age_today_gyr = 13.467 ± 0.01 -->

billion years. None of the three numbers above, nor the age, nor
$\rho_{\mathrm{crit,0}}$, ever enters the lens-modelling likelihood this guide
spends Part V decoding: `site/guide_src/lensing.py` runs entirely in arcsec,
with no distances at all, and this chapter's whole apparatus enters this
repository in exactly the three files cited above — Einstein radii from
velocity dispersions, $\Sigma_{\mathrm{cr}}$ and enclosed mass, and one
supernova's distance modulus. The money number
$\gamma_{\mathrm{binned}}(\text{corr, low}) = 1.103 \pm 0.008$ does not know that
$H_0=70$; it would come out identical if this repository had asserted $H_0=67$
instead. That scoping is worth holding onto precisely because it tells you
where a wrong cosmology *would* and would not bite.

## Connect to the repo { #connect }

`site/guide_src/cosmo.py:29`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/cosmo.py#L29))
is the one line this chapter's whole worked example runs against:
`COSMO = FlatLambdaCDM(H0=70, Om0=0.3)`. The module's own docstring, starting
at `site/guide_src/cosmo.py:5`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/cosmo.py#L5)),
makes the same scoping argument as the
paragraph above — cosmology enters this repository in exactly three places —
and is the reason this guide's cosmology Part is three chapters, not ten:
that proportion is itself a finding about where to spend your attention, not
an omission.

This repository never builds its own Friedmann-equation solver. All three
files construct `astropy.cosmology.FlatLambdaCDM`; two of them call
`.angular_diameter_distance` directly —
`reproductions/hsu-2025/07_classify_einstein_dimple.py:50` and
`reproductions/sheu-2024b/04_setup_multiplane.py:35`, cited above — and the
third, `reproductions/sheu-2023/05_lightcurve_salt3.py:57`, hands the object
straight to `sncosmo`, which draws the supernova's luminosity distance from it
internally. What this chapter buys you is the ability to read `.H`, `.age`,
`.critical_density0`, and `.angular_diameter_distance` — the four methods this
chapter's own worked example (`ch14_frw_friedmann` in
`site/guide_src/worked_examples.py`) exercises directly — as physics rather
than as API surface: `.H(z)` is $\eqref{eq:friedmann}$ evaluated at a
redshift, `.age(0)` is that same equation integrated back to $a=0$, and
`.critical_density0` is the reference density this chapter derived from $H_0$
alone. [Ch. 15](15-distances.md#three-distances) is next: it turns the same
`COSMO` object into the two distances,
$D_{\mathrm{d}}, D_{\mathrm{s}}, D_{\mathrm{ds}}$, that this repository's
lensing calculations actually consume, and into the non-additivity gotcha
that bites everyone once.

## Exercises { #exercises }

??? question "Exercise 14.1 — Dimensional analysis on the Friedmann equation"
    Confirm, from units alone, that $\frac{8\pi G}{3}\rho$ in
    $\eqref{eq:friedmann}$ has the same units as $H^2$. You will need
    Newton's constant's units, $[G] = \text{m}^3\,\text{kg}^{-1}\,\text{s}^{-2}$,
    and a mass density's units, $[\rho] = \text{kg}\,\text{m}^{-3}$.

    ??? success "Solution"
        $$[G][\rho] = \left(\text{m}^3\,\text{kg}^{-1}\,\text{s}^{-2}\right)\left(\text{kg}\,\text{m}^{-3}\right) = \text{s}^{-2}$$

        which is exactly $[H]^2$, since $H = \dot a/a$ is a rate (one over a
        time). The $8\pi/3$ is dimensionless, so it does not affect the check.
        The $kc^2/a^2$ term must independently carry units of $\text{s}^{-2}$
        as well: $[c^2] = \text{m}^2\text{s}^{-2}$, so $k$ itself must carry
        units of inverse length squared unless $a$ is defined with units of
        length (some textbooks set $a_0 \ne 1$ precisely to make $k$
        dimensionless $\pm 1$ or $0$; this guide follows the convention
        $a_0 = 1$, in which case $k$ silently absorbs the missing length
        dimension). Either convention gives the same physics; just be
        consistent about which one a given equation is using.

??? question "Exercise 14.2 — Flatness is a measurement, not an identity"
    Explain, in a sentence or two, why $\Omega_m + \Omega_\Lambda + \Omega_k = 1$
    can be stated with total confidence before a single measurement is made,
    while $\Omega_k = 0$ cannot.

    ??? success "Solution"
        The sum identity falls straight out of *dividing the Friedmann
        equation by $H_0^2$* — it is true by the definitions of $\Omega_m$,
        $\Omega_\Lambda$, and $\Omega_k$ as $H_0^2$-normalized pieces of an
        equation that already balances. It holds for any values those three
        symbols take, in any universe, flat or not. $\Omega_k = 0$ is a
        different kind of claim entirely: it says the *actual* curvature
        term is zero, which is a statement about which universe we are in,
        checkable only by measuring $\Omega_m$, $\Omega_\Lambda$ (or $H(z)$
        directly) independently and seeing whether they leave any room for a
        curvature term. `FlatLambdaCDM` doesn't measure this — it assumes it,
        by refusing to expose a curvature argument at all.

??? question "Exercise 14.3 — H(z) at a redshift this chapter didn't show you"
    Using only $H_0 = 70$, $\Omega_m = 0.3$, $\Omega_\Lambda = 0.7$, and
    $H(z)^2 = H_0^2\left[\Omega_m(1+z)^3+\Omega_\Lambda\right]$, compute $H(z=1)$
    by hand. Then check your answer against `astropy`'s own value.

    ??? success "Solution"
        $$H(1)^2 = 70^2\left[0.3\cdot 2^3 + 0.7\right] = 4900 \times 3.1 = 15190$$

        $$H(1) = \sqrt{15190} \approx 123.25\ \text{km/s/Mpc}$$

        <!-- check: ch14.hz_manual_z1 = 123.248 ± 0.01 -->

        which matches `cosmo.COSMO.H(1.0)` to machine precision:

        <!-- check: ch14.hz_match_diff_z1 = 0.0 ± 1e-9 -->

        Notice $H$ nearly *doubled* between $z=0.5$ (91.6 km/s/Mpc, in the main
        text) and $z=1$ (123.25 km/s/Mpc) — the matter term $\Omega_m(1+z)^3$
        is starting to dominate over the constant $\Omega_\Lambda$ term as you
        look back in time, exactly as $\rho_m \propto a^{-3}$ says it must.

??? question "Exercise 14.4 — Where pressure would have to enter"
    This chapter's derivation of $\eqref{eq:friedmann}$ never once used
    pressure, and it doesn't need to: the equation it produces is exactly
    right. Where, physically, would you expect a Newtonian argument like this
    one to start failing if you tried to extend it to the *second* Friedmann
    equation (the one for $\ddot a$)?

    ??? success "Solution"
        Newton's shell theorem and $F=GMm/r^2$ know about mass, full stop —
        pressure does not appear anywhere in Newtonian gravity as a source
        of attraction. In general relativity it does: the source of gravity
        is (loosely) $\rho + 3p/c^2$, not $\rho$ alone, so a medium under
        positive pressure gravitates *more*, and a negative-pressure
        component (which is exactly what makes something behave like a
        cosmological constant) can drive $\ddot a$ positive — accelerated
        expansion — with no Newtonian counterpart at all. This is the same
        family of "GR adds a term Newton has no room for" as the factor of
        two in [Ch. 16](16-deflection.md#the-factor-of-two), where light
        bends twice as far as a naive Newtonian calculation predicts. Neither
        this chapter nor this repository ever needs the pressure-sourced
        equation explicitly — dark energy is supplied as a fixed density,
        not derived from an equation of state — but it is worth knowing
        where the Newtonian trick this section leaned on would have quietly
        stopped working.
