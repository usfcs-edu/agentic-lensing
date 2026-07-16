# 15. Distances that do not add, and $\Sigma_{\mathrm{cr}}$

Chapter 14 gave you one object, `FlatLambdaCDM(H0=70, Om0=0.3)`, and the
equation that makes it tick. This chapter turns that object into distances you
can act on — and there turn out to be three of them, not one, because "how far
away" depends on whether you mean light-travel geometry, angular size, or
flux. You will derive all three from the same flat FRW metric Chapter 14 wrote
down, meet the one operation on them that is silently illegal — subtracting
two angular-diameter distances — and use the survivor, $\Sigma_{\mathrm{cr}}$,
to reproduce this repository's own published numbers for a real cluster lens:
its critical surface density and the mass enclosed inside its Einstein radius,
both computed from nothing but two redshifts and five lines of code.

!!! abstract "What you can skip"
    If comoving, angular-diameter, and luminosity distance are already second
    nature — including the fact that they differ by explicit factors of
    $(1+z)$, not by anything mysterious — skip **Three distances, one
    universe** and go straight to **D_ds != D_s - D_d**. That section is not a
    subtle GR effect; it is an algebra trap this repository's own code has to
    dodge every time it builds a lens model, and it is worth reading even if
    you could write down $D_{\mathrm{A}}(z)$ from memory. If you also already
    know $\Sigma_{\mathrm{cr}}$, skip straight to **The Carousel**, where the
    payoff is two real, checkable numbers.

## Three distances, one universe { #three-distances }

In a flat, static, Euclidean space "distance" needs no qualifier. In an
expanding one it does, because the three operations you might use a distance
for — timing a light ray, measuring an angular size, and measuring a flux —
stop agreeing with each other once redshift is not tiny. This section derives
the three you need, in order, each one built on the last.

**Comoving distance.** [Ch. 14](14-frw.md#the-frw-metric) wrote down the FRW
metric; this repository only ever uses its flat ($k=0$) case, where the radial
part simplifies to $ds^2 = -c^2dt^2 + a(t)^2\left[dr^2 + r^2\,d\Omega^2\right]$.
A photon travels on a null geodesic, $ds^2=0$, and along a purely radial path
($d\Omega=0$) that forces $c\,dt = a(t)\,dr$. Define the **comoving distance**
as the coordinate distance such a photon covers:

$$
D_{\mathrm{C}}(z) \;\equiv\; \int_{t_e}^{t_0} \frac{c\,dt}{a(t)}
$$

Change the integration variable from cosmic time to redshift using
[Ch. 13](13-expansion.md#redshift-is-expansion)'s $a(t_e) = 1/(1+z)$ and
[Ch. 14](14-frw.md#friedmann)'s $H \equiv \dot a/a$: differentiating
$a = (1+z)^{-1}$ gives $da/dz = -(1+z)^{-2}$, and $dt = da/\dot a = da/(aH)$,
so $dt = -dz/[(1+z)H(z)]$. Absorbing the sign by integrating from $0$ to $z$
instead of $t_e$ to $t_0$:

$$
D_{\mathrm{C}}(z) \;=\; c\int_0^z \frac{dz'}{H(z')}
\label{eq:dc}
$$

That is [Ch. 14](14-frw.md#friedmann)'s Friedmann equation, integrated along
the line of sight. `astropy` performs $\eqref{eq:dc}$ numerically — there is
no closed form once both $\Omega_m$ and $\Omega_\Lambda$ are nonzero — and its
`comoving_distance` method is exactly this integral. At $z=2.0$:

$$
D_{\mathrm{C}}(2.0) \approx 5179.862\ \mathrm{Mpc}
$$

<!-- check: ch15.d_c_20 = 5179.862 ± 0.01 -->

**Angular-diameter distance.** The metric's transverse piece,
$a(t)^2 r^2\,d\Omega^2$, is a genuine 2-sphere of *physical* (proper) radius
$a(t)\,r$ at fixed $t$ — this is [Ch. 13](13-expansion.md#scale-factor)'s
"proper distance $=a(t)\times$ comoving distance" rule, now applied to the
transverse direction instead of the radial one. An object of physical
transverse size $\ell$, sitting at comoving radius $r$ at the moment $t_e$ its
light was emitted, therefore subtends
$\delta\theta = \ell / [a(t_e)\,r]$ — plain Euclidean geometry on a sphere of
that radius, nothing relativistic beyond the metric itself. Define
$D_{\mathrm{A}} \equiv \ell/\delta\theta = a(t_e)\,r$. In a flat universe the
comoving radial coordinate $r$ *is* $D_{\mathrm{C}}(z)$ — no curvature-dependent
$\sin$ or $\sinh$ correction, since this repository never leaves $k=0$ — so,
using $a(t_e) = 1/(1+z)$:

$$
D_{\mathrm{A}}(z) \;=\; \frac{D_{\mathrm{C}}(z)}{1+z}
\label{eq:da}
$$

This is `astropy`'s `.angular_diameter_distance`, and the one function this
repository actually calls: `cosmo.d_a` at `site/guide_src/cosmo.py:32`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/cosmo.py#L32)).
At $z=2.0$: $D_{\mathrm{A}}(2.0) = D_{\mathrm{C}}(2.0)/3 \approx 1726.621\
\mathrm{Mpc}$.

<!-- check: ch15.d_s_20 = 1726.621 ± 0.01 -->

**Luminosity distance.** Put a source of luminosity $L$ at comoving distance
$D_{\mathrm{C}}(z)$, radiating isotropically. By the time its light is observed
it has spread over a sphere of *physical* area $4\pi D_{\mathrm{C}}(z)^2$
(using $a_0\equiv a(t_0)=1$, today). Ordinary inverse-square dilution alone
would give $F = L/[4\pi D_{\mathrm{C}}(z)^2]$, but two more factors of $(1+z)$
cut the observed flux further: each photon's energy is itself redshifted,
$E_{\mathrm{obs}} = E_{\mathrm{emit}}/(1+z)$, and the arrival *rate* of photons
is time-dilated by the same factor — photons emitted a proper time $\delta t_e$
apart arrive spaced by $\delta t_e\,(1+z)$. Combining all three:

$$
F = \frac{L}{4\pi D_{\mathrm{C}}(z)^2 (1+z)^2}
$$

Flux is *defined* via $F \equiv L/(4\pi D_{\mathrm{L}}^2)$, so matching the two
expressions gives $D_{\mathrm{L}}(z) = (1+z)\,D_{\mathrm{C}}(z) = (1+z)^2\,D_{\mathrm{A}}(z)$.
At $z=2.0$: $D_{\mathrm{L}}(2.0) = 9 \times 1726.621 \approx 15539.586\
\mathrm{Mpc}$

<!-- check: ch15.d_l_20 = 15539.586 ± 0.01 -->

— matching `astropy`'s own `luminosity_distance` to better than a millimeter
in $15.5$ billion parsecs:

<!-- check: ch15.d_l_da_diff = 0.0 ± 1e-6 -->

$D_{\mathrm{L}}$ is what a Type Ia supernova's distance modulus,
$m - M = 5\log_{10}(D_{\mathrm{L}}/10\,\mathrm{pc})$, actually consumes — the
third and final place cosmology enters this repository,
`reproductions/sheu-2023/05_lightcurve_salt3.py:57`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/sheu-2023/05_lightcurve_salt3.py#L57)),
a lensed-supernova campaign this guide does not otherwise follow. Everything
else in this guide — every $\theta_{\mathrm{E}}$, every $\Sigma_{\mathrm{cr}}$
— uses only $D_{\mathrm{A}}$, under the repository's own subscripted names
$D_{\mathrm{d}}$ (to the lens), $D_{\mathrm{s}}$ (to the source), and
$D_{\mathrm{ds}}$ (between them) — never $D_{\mathrm{s}}-D_{\mathrm{d}}$, which
is the next section's whole point.

## D_ds != D_s - D_d { #distances-do-not-add }

State it as plainly as the notation contract this guide follows demands:
$D_{\mathrm{ds}}$ is **not** $D_{\mathrm{s}}-D_{\mathrm{d}}$. Angular-diameter
distances do not add, and the reason is entirely algebraic, not a mystery of
curved spacetime — you already have every piece needed to derive it.

The general angular-diameter distance between two redshifts, in a **flat**
universe, is

$$
D_{\mathrm{A}}(z_1,z_2) \;=\; \frac{D_{\mathrm{C}}(z_2)-D_{\mathrm{C}}(z_1)}{1+z_2}
\label{eq:dds}
$$

— the formula `astropy`'s `angular_diameter_distance_z1z2` evaluates for
`FlatLambdaCDM`, and exactly what `cosmo.d_ds` at `site/guide_src/cosmo.py:37`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/cosmo.py#L37))
calls. Two things about $\eqref{eq:dds}$ are worth noticing before touching a
number. First, it reduces correctly at $z_1=0$ (Exercise 15.2 makes you check
this): $D_{\mathrm{A}}(0,z_2) = D_{\mathrm{C}}(z_2)/(1+z_2) = D_{\mathrm{A}}(z_2)$,
exactly $\eqref{eq:da}$. Second — and this is the trap — $D_{\mathrm{ds}}$
divides by $(1+z_s)$ *once*, applied to the *difference* of two
comoving distances. It does not divide $D_{\mathrm{C}}(z_l)$ by
$(1+z_l)$ and $D_{\mathrm{C}}(z_s)$ by $(1+z_s)$
separately and then subtract. But that second, wrong operation is exactly what
$D_{\mathrm{s}}-D_{\mathrm{d}}$ is:

$$
D_{\mathrm{s}}-D_{\mathrm{d}} \;=\; \frac{D_{\mathrm{C}}(z_s)}{1+z_s} - \frac{D_{\mathrm{C}}(z_l)}{1+z_l}
$$

!!! tip "You already know this"
    Two quantities that were each independently rescaled by a *different*
    factor do not combine correctly under subtraction — this is the same
    reason you cannot take two activations that were each standardized by a
    different layer's own running mean and variance, subtract them, and
    expect the result to be meaningfully normalized. $D_{\mathrm{d}}$ and
    $D_{\mathrm{s}}$ are each already divided by their *own* $(1+z)$; the fix
    is to undo that per-term rescaling — work in $D_{\mathrm{C}}$, which really
    is additive — and apply a single, shared rescaling only at the end.

Subtracting $\eqref{eq:dds}$ and the naive expression above, the $D_{\mathrm{C}}(z_s)$
terms cancel identically, leaving a closed form entirely in terms of the lens
distance:

$$
D_{\mathrm{ds}} - \left(D_{\mathrm{s}}-D_{\mathrm{d}}\right)
\;=\; D_{\mathrm{C}}(z_l)\left[\frac{1}{1+z_l}-\frac{1}{1+z_s}\right]
\;=\; D_{\mathrm{d}}\,\frac{z_s-z_l}{1+z_s}
$$

(using $D_{\mathrm{C}}(z_l) = D_{\mathrm{d}}(1+z_l)$ in the
last step). Now the numbers, for $z_l=0.5$, $z_s=2.0$ — the
same pair [Ch. 13](13-expansion.md#redshift-is-expansion)'s exercises already
flagged as this chapter's test case, chosen specifically because $(1+z)$ is
nowhere near $1$ for either redshift:

$$
D_{\mathrm{d}} \approx 1259.084\ \mathrm{Mpc}, \qquad D_{\mathrm{s}} \approx 1726.621\ \mathrm{Mpc}
$$

<!-- check: ch15.d_d_05 = 1259.084 ± 0.01 -->

The correct $D_{\mathrm{ds}}$, from $\eqref{eq:dds}$ or equivalently
`cosmo.d_ds(0.5, 2.0)`:

$$
D_{\mathrm{ds}} \approx 1097.079\ \mathrm{Mpc}
$$

<!-- check: ch15.d_ds_05_20 = 1097.079 ± 0.01 -->

Build it the long way, from the comoving distances directly —
$D_{\mathrm{C}}(0.5)\approx1888.625$ Mpc and $D_{\mathrm{C}}(2.0)\approx5179.862$
Mpc, subtract, then divide once by $1+z_s=3$:

<!-- check: ch15.d_c_05 = 1888.625 ± 0.01 -->

$$
\frac{5179.862-1888.625}{3} \approx 1097.079\ \mathrm{Mpc}
$$

and it lands on the same number to machine precision:

<!-- check: ch15.d_ds_dc_diff = 0.0 ± 1e-6 -->

The naive subtraction, by contrast:

$$
D_{\mathrm{s}}-D_{\mathrm{d}} \approx 467.537\ \mathrm{Mpc}
$$

<!-- check: ch15.naive_subtraction = 467.537 ± 0.01 -->

— less than half of $D_{\mathrm{ds}}$, a factor of

<!-- check: ch15.subtraction_error_ratio = 2.3465 ± 0.001 -->

$2.3465$ off, silently. The closed-form gap derived above,
$D_{\mathrm{d}}\,(z_s-z_l)/(1+z_s) = 1259.084
\times 1.5/3 \approx 629.542\ \mathrm{Mpc}$

<!-- check: ch15.additivity_gap_closed_form = 629.542 ± 0.01 -->

matches the actual gap between the two numbers above exactly:

<!-- check: ch15.additivity_gap_diff = 0.0 ± 1e-6 -->

This is not a rounding error or a subtle relativistic correction to chase down
— it is what happens whenever you subtract two numbers that were each divided
by something different before you got them.

## Sigma_crit: the natural unit of surface density { #sigma-crit }

Convergence, $\kappa \equiv \Sigma/\Sigma_{\mathrm{cr}}$, is the surface mass
density $\Sigma$ of a lens measured in units of a reference density
$\Sigma_{\mathrm{cr}}$ built entirely from $c$, $G$, and the three distances
of the previous section:

$$
\Sigma_{\mathrm{cr}} \;=\; \frac{c^2}{4\pi G}\,\frac{D_{\mathrm{s}}}{D_{\mathrm{d}}D_{\mathrm{ds}}}
$$

Dimensional analysis alone forces this shape to be plausible before you know
anything about *why* it is the right one: $[c^2/G] = (\mathrm{m^2\,s^{-2}})/(\mathrm{m^3\,kg^{-1}\,s^{-2}}) = \mathrm{kg/m}$,
and the distance ratio $D_{\mathrm{s}}/(D_{\mathrm{d}}D_{\mathrm{ds}})$ carries
units $\mathrm m/\mathrm m^2 = \mathrm m^{-1}$, so the product is
$\mathrm{kg/m^2}$ — a surface density, exactly as the name promises.
Dimensional analysis cannot tell you *why* this particular combination of
distances is the physically correct one, only that whatever the correct
formula turns out to be, it had better have this shape — the actual physics
(the point-mass deflection law and the geometry of a lens, source, and
observer in a line) is [Ch. 16](16-deflection.md#the-factor-of-two)–[Ch. 17](17-lens-equation.md#the-lens-equation)'s
job, and the fact that $\kappa=1$ marks the boundary between one image and
several is [Ch. 19](19-einstein-radius.md#the-mean-convergence-identity)'s.
What matters here is narrower: $\Sigma_{\mathrm{cr}}$ is a well-defined,
computable number the moment you have $D_{\mathrm{d}}$, $D_{\mathrm{s}}$, and
$D_{\mathrm{ds}}$ in hand, and — because $D_{\mathrm{ds}}$ appears in the
denominator — it inherits the previous section's non-additivity trap directly:
get $D_{\mathrm{ds}}$ wrong by a factor of $2.3$ and $\Sigma_{\mathrm{cr}}$,
and every mass this guide will ever quote, is wrong by the same factor.

## The Carousel: reproducing the repo's own numbers { #the-carousel }

The Carousel Lens (Sheu et al. 2024, `reproductions/sheu-2024b/README.md`) is
a cluster-scale strong lens at $z_l=0.49$, with its Einstein radius
quoted relative to one of five spectroscopically-confirmed source planes,
$z_s=1.432$: $\theta_{\mathrm{E}}=13.03''$ for the primary deflector.
Plug those two redshifts into $\Sigma_{\mathrm{cr}}$:

$$
\Sigma_{\mathrm{cr}}(0.49,\,1.432) \;\approx\; 2.376\times10^{15}\ M_\odot/\mathrm{Mpc}^2
$$

<!-- check: ch15.sigma_crit_carousel = 2.376e15 ± 1e12 -->

— equivalently $2376.07\ M_\odot/\mathrm{pc}^2$

<!-- check: ch15.sigma_crit_carousel_msun_pc2 = 2376.07 ± 0.5 -->

which is `reproductions/sheu-2024b/04_setup_multiplane.py:128`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/sheu-2024b/04_setup_multiplane.py#L128))
computed independently and, per that script's own printed output
(`f"{sigma_crit:.3e}"`), to four significant figures.

To turn $\theta_{\mathrm{E}}$ into a physical size you need one more
conversion: how much proper transverse distance one arcsecond spans at
$z_l=0.49$, which is $D_{\mathrm{d}}$ itself times $1''$ in radians —
[Ch. 9](09-units.md#angles-on-the-sky)'s $206264.8$ factor, applied here to a
cosmological rather than a local distance:

$$
1'' \times D_{\mathrm{d}}(0.49) \;\approx\; 6.038\times10^{-3}\ \mathrm{Mpc}
$$

<!-- check: ch15.arcsec_to_mpc_carousel = 6.038e-3 ± 1e-6 -->

so $\ell_{\mathrm{E}} = 6.038\times10^{-3}\times 13.03 \approx 0.07868\
\mathrm{Mpc} = 78.68$ kpc

<!-- check: ch15.r_e_carousel_mpc = 0.07868 ± 0.0001 -->

and, taking $M(<\theta_{\mathrm{E}}) = \Sigma_{\mathrm{cr}}\,\pi \ell_{\mathrm{E}}^2$
on faith for now (why this particular formula gives the enclosed mass
regardless of the density profile's shape is
[Ch. 19](19-einstein-radius.md#mass-inside-theta-e)'s identity):

$$
M(<\theta_{\mathrm{E}}) \;\approx\; 4.621\times10^{13}\ M_\odot
$$

<!-- check: ch15.mass_within_theta_e_carousel = 4.621e13 ± 1e11 -->

Sheu et al. (2024)'s own Table 2 quotes $M(<\theta_{\mathrm{E}}) = 4.78\times10^{13}\
M_\odot$ for this deflector — this chapter's from-scratch, geometry-only
estimate lands $3.3\%$ low:

<!-- check: ch15.mass_repro_vs_paper_ratio = 0.9667 ± 0.001 -->

Not a bug. The formula used here assumes a perfectly circular lens (mean
$\kappa=1$ inside a circle of radius $\theta_{\mathrm{E}}$); the published
figure comes from the paper's full elliptical mass model, and the primary
deflector's axis ratio is $q=0.87$ — close to circular but not exactly. A
small, well-understood correction, and Exercise 15.3 asks you to check that
its size moves in the direction the ellipticity story predicts once you apply
the identical method to the Carousel's more elongated second deflector.

## Connect to the repo { #connect }

Every distance in this chapter is one of five short functions in
`site/guide_src/cosmo.py`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/cosmo.py)):
`d_a` (`cosmo.py:32`) wraps `.angular_diameter_distance` for $\eqref{eq:da}$;
`d_ds` (`cosmo.py:37`) wraps `.angular_diameter_distance_z1z2` for
$\eqref{eq:dds}$ and is annotated, in its own docstring, with exactly this
chapter's warning — never a subtraction; `sigma_crit` (`cosmo.py:45`) is
$\Sigma_{\mathrm{cr}}$ verbatim; `arcsec_to_mpc` (`cosmo.py:61`) is the
$1''\times D_{\mathrm{d}}$ conversion behind $\ell_{\mathrm{E}}$; and
`mass_within_theta_e` (`cosmo.py:66`) is the whole Carousel worked example in
four lines. `site/guide_src/worked_examples.py`'s `ch15_distances_and_sigma_crit`
reproduces `reproductions/sheu-2024b/04_setup_multiplane.py:125`–`131`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/sheu-2024b/04_setup_multiplane.py#L125))
line for line, against the published Table 2 numbers printed at that script's
own `:135`–`138`.

[Ch. 13](13-expansion.md#connect) and [Ch. 14](14-frw.md#connect) already
made the scoping argument this chapter's numbers confirm from the distance
side: `reproductions/claude-giga-lens/cgl/` never imports
`astropy.cosmology` at all, so none of $D_{\mathrm{d}}$, $D_{\mathrm{s}}$,
$D_{\mathrm{ds}}$, or $\Sigma_{\mathrm{cr}}$ enters the money-number pipeline.
This repository's two remaining cosmology campaigns need nothing else this
chapter didn't already derive:
`reproductions/hsu-2025/07_classify_einstein_dimple.py:114`–`116`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/hsu-2025/07_classify_einstein_dimple.py#L114))
uses $D_{\mathrm{s}}$ and $D_{\mathrm{ds}}$ to turn a velocity dispersion into
an Einstein radius, derived in full in
[Ch. 19](19-einstein-radius.md#theta-e-from-sigma-v); and
`reproductions/sheu-2023/05_lightcurve_salt3.py:57` uses $D_{\mathrm{L}}$, this
chapter's third distance, for a lensed supernova's magnitude — the one place
in this repository a distance sets a *brightness* rather than a length or an
angle.

## Exercises { #exercises }

??? question "Exercise 15.1 — Comoving distance at low redshift"
    Chapter 13 derived the linear Hubble law, $v \approx H_0 d$, valid for
    $z \ll 1$. Show that the comoving-distance integral $\eqref{eq:dc}$
    reduces to this law in the same limit, and that $D_{\mathrm{A}}(z)\approx
    D_{\mathrm{C}}(z)$ there too. (You need only that $H(z')\to H_0$ as
    $z'\to0$ — not $\Omega_m$ or $\Omega_\Lambda$ individually.)

    ??? success "Solution"
        For $z\ll1$, $H(z')$ barely moves from $H_0$ across the whole
        integration range $[0,z]$ in $\eqref{eq:dc}$, so to leading order it
        can be pulled outside the integral:
        $D_{\mathrm{C}}(z)\approx c\int_0^z dz'/H_0 = cz/H_0$ — exactly the
        linear Hubble law solved for distance, $d = v/H_0$ with $v\approx cz$.
        And since $D_{\mathrm{A}}=D_{\mathrm{C}}/(1+z)$, at $z\ll1$ the factor
        $(1+z)\approx1$ to first order, so $D_{\mathrm{A}}(z)\approx
        D_{\mathrm{C}}(z)$ as well — all three distances of this chapter agree
        in the nearby universe (since $D_{\mathrm{L}}=(1+z)^2D_{\mathrm{A}}$
        collapses to the same limit too). They only pull apart once $(1+z)$
        stops being close to $1$, which is exactly why this chapter's worked
        example uses $z_l=0.5$, $z_s=2.0$ rather than
        anything small.

??? question "Exercise 15.2 — The two-redshift formula, checked at its own boundary"
    Show that $\eqref{eq:dds}$, $D_{\mathrm{A}}(z_1,z_2) =
    [D_{\mathrm{C}}(z_2)-D_{\mathrm{C}}(z_1)]/(1+z_2)$, reduces correctly to the
    single-redshift formula $\eqref{eq:da}$ when $z_1=0$, and explain in one
    sentence why that check has to pass.

    ??? success "Solution"
        Set $z_1=0$: $D_{\mathrm{C}}(0)=0$ trivially — there is no distance to
        travel to reach redshift zero, since that is *here* — so
        $D_{\mathrm{A}}(0,z_2) = D_{\mathrm{C}}(z_2)/(1+z_2)$, exactly
        $\eqref{eq:da}$. It has to pass because $D_{\mathrm{A}}(0,z)$ and "the
        angular-diameter distance to $z$" are not two different quantities
        that happen to agree — they are the same quantity under two names, an
        observer at redshift zero looking out to $z$. A two-redshift formula
        that failed this check would be wrong for every pair of redshifts, not
        just the edge case $z_1=0$.

??? question "Exercise 15.3 — The Carousel's other deflector"
    The Carousel's own Table 2 lists a second deflector, Ld, with
    $\theta_{\mathrm{E}}=0.99''$ (also quoted relative to $z_s=1.432$)
    and a more elongated shape than the primary — axis ratio $q=0.69$, versus
    $0.87$ for the primary. Using the *same* $\Sigma_{\mathrm{cr}}$ already
    computed for this system, compute $M(<\theta_{\mathrm{E}})$ for Ld and
    compare it to the paper's published $2.77\times10^{11}\ M_\odot$.

    ??? success "Solution"
        $\Sigma_{\mathrm{cr}}$ depends only on the redshifts, which are
        unchanged (both deflectors sit at $z_l=0.49$, both
        $\theta_{\mathrm{E}}$ values are quoted relative to the same
        $z_s=1.432$), so only $\ell_{\mathrm{E}}$ changes:
        $\ell_{\mathrm{E},\mathrm{Ld}} = 6.038\times10^{-3}\times0.99 \approx
        5.978\times10^{-3}\ \mathrm{Mpc}$. Then
        $M(<\theta_{\mathrm{E}}) = \Sigma_{\mathrm{cr}}\,\pi \ell_{\mathrm{E},\mathrm{Ld}}^2
        \approx 2.668\times10^{11}\ M_\odot$
        <!-- check: ch15.mass_within_theta_e_carousel_ld = 2.668e11 ± 1e9 -->
        against the paper's $2.77\times10^{11}$: a ratio of
        <!-- check: ch15.mass_repro_vs_paper_ratio_ld = 0.963 ± 0.001 -->
        $0.963$ — a slightly *larger* shortfall than the primary deflector's
        $3.3\%$. That is exactly the direction the ellipticity story predicts:
        Ld's axis ratio ($q=0.69$) is farther from circular than the
        primary's ($q=0.87$), so the circular-equivalent approximation this
        chapter's formula makes should miss by more, not less. It does.

??? question "Exercise 15.4 — Dimensional analysis of Sigma_crit"
    Confirm, from units alone, that
    $\dfrac{c^2}{4\pi G}\dfrac{D_{\mathrm{s}}}{D_{\mathrm{d}}D_{\mathrm{ds}}}$
    has units of mass per area, using $[c]=\mathrm{m/s}$,
    $[G]=\mathrm{m^3\,kg^{-1}\,s^{-2}}$, and $[D]=\mathrm{m}$.

    ??? success "Solution"
        $[c^2/G] = \dfrac{\mathrm{m^2\,s^{-2}}}{\mathrm{m^3\,kg^{-1}\,s^{-2}}}
        = \dfrac{\mathrm{kg}}{\mathrm{m}}$. The distance ratio
        $D_{\mathrm{s}}/(D_{\mathrm{d}}D_{\mathrm{ds}})$ carries units
        $\mathrm m/\mathrm m^2 = \mathrm m^{-1}$. Multiplying:
        $\dfrac{\mathrm{kg}}{\mathrm m}\times\dfrac1{\mathrm m} =
        \dfrac{\mathrm{kg}}{\mathrm m^2}$ — mass per area, exactly a surface
        density; $4\pi$ is dimensionless and does not affect the check. This
        argument alone cannot tell you *why* this particular combination of
        $c$, $G$, and three distances is the physically relevant one — only
        that whatever the correct formula turns out to be, it had better have
        this shape, because nothing else built from $c$, $G$, and three
        lengths has the right units to be a surface density at all.
