# Integrals, projection, and where the logarithm comes from

By the end of this chapter you can derive, from nothing but the definition of
an integral, two facts that look like they come from nowhere in every lensing
paper you will read: why a mass profile's density slope $\gamma$ shifts by
exactly one when you project it from three dimensions to two, and why the
potential of a point mass sitting in a two-dimensional world is a logarithm
rather than the familiar $1/r$. Both facts are downstream of doing an integral
as an *accumulation* — stacking density along a sightline, summing flux around
a loop — rather than treating "integral" as a symbol you look up a closed form
for. By the last section you will have re-derived, from those two facts alone,
why `lensing.py`'s singular-isothermal-sphere deflection has constant
magnitude: a one-line code comment turns into a two-line proof.

!!! abstract "What you can skip"
    You already know how to evaluate a definite integral, substitute
    $u = g(x)$, and check an improper integral for convergence with the
    $p$-test ($\int^\infty x^{-p}\,dx$ converges iff $p>1$). None of that
    machinery is new here. What is new is what the integral *means* in this
    setting — a running total of mass, not an area under a curve — and two
    consequences that follow from computing it: an exponent that shifts by
    one, and a logarithm that appears in exactly one dimension count.

## Integrals as accumulation { #accumulation }

A Riemann sum partitions an interval, evaluates a function on each piece, and
adds up the pieces times their width. Taking the partition to zero width gives
the integral. Nothing about that procedure requires the function to be a curve
on a page — it works exactly as well when the function is a mass density and
the "area under the curve" is a mass.

The accumulation this guide returns to most often is a radial one: given a
surface (2-D, sky-plane) mass density $\Sigma(R)$, the mass enclosed inside
radius $R$ is

$$
M(<R) \;=\; \int_0^R \Sigma(R')\, \bigl(2\pi R'\bigr)\, dR'.
$$

The factor $2\pi R'\,dR'$ is itself a small accumulation: it is the area of a
thin ring of radius $R'$ and width $dR'$, obtained by unrolling the ring into
a rectangle of length (circumference) $2\pi R'$ and height $dR'$ — the
same "slice it thin, add up the slices" move as the outer integral, one level
down. You will meet this exact integral again in
[Ch. 19](19-einstein-radius.md#the-mean-convergence-identity), where dividing
$M(<\theta)$ by the enclosed area turns it into the mean-convergence identity
that defines the Einstein radius; the implementation,
`mean_kappa_within` (`site/guide_src/lensing.py:179`), is this formula with
$\Sigma$ replaced by $\kappa$ and normalized by the enclosed area.

That is accumulation along one direction (radius, in the sky plane). The next
section needs accumulation along a second, orthogonal direction: straight
through the galaxy, along the line of sight. That is where $\gamma$ picks up
its first concrete consequence.

## The Abel projection { #the-abel-projection }

Take a spherically symmetric 3-D mass density that is a pure power law,

$$
\rho(r) = A\,r^{-\gamma}, \qquad \gamma > 0,
$$

with $A$ collecting whatever normalization and scale radius the profile
carries — $\gamma = 2$ is the isothermal sphere, the campaign's reference
slope. A telescope cannot see $\rho(r)$; it sees the sky-plane surface density
$\Sigma(R)$ obtained by adding up every bit of mass along the sightline at
fixed projected radius $R$. Writing the sightline coordinate as $z$, so that
the true 3-D radius is $r = \sqrt{R^2+z^2}$, that accumulation is the integral

$$
\Sigma(R) \;=\; \int_{-\infty}^{\infty} \rho\!\left(\sqrt{R^2+z^2}\right) dz
\;=\; 2A \int_0^{\infty} \left(R^2+z^2\right)^{-\gamma/2} dz.
$$

!!! tip "You already know this"
    Integrating a joint density over one of its coordinates to get the
    density of the rest is marginalization. Integrating $\rho(x,y,z)$ over $z$
    to get $\Sigma(x,y)$ is exactly that operation, with the line-of-sight
    coordinate playing the role of a nuisance variable. The campaign performs
    the identical move analytically on 28 shapelet source-light amplitudes
    (`reproductions/claude-giga-lens/cgl/likelihood.py:8`) —
    `reproductions/claude-giga-lens/cgl/marg.py:9` marginalizes them out of the posterior in closed form,
    via a Cholesky solve rather than a sightline integral, but it is the same
    integral in spirit: sum over what you cannot measure directly, keep what
    you can.

The integral over $z$ still has $R$ tangled up inside it, which hides the
point. Substitute $z = Ru$ — a pure rescaling of the integration variable,
nothing more:

$$
dz = R\,du, \qquad R^2+z^2 = R^2(1+u^2),
$$

$$
\Sigma(R) = 2A\int_0^\infty \bigl(R^2(1+u^2)\bigr)^{-\gamma/2}\, R\, du
$$

which needs a moment of care with the algebra: $\bigl(R^2(1+u^2)\bigr)^{-\gamma/2} = R^{-\gamma}(1+u^2)^{-\gamma/2}$, and there is one more factor of $R$ from $dz$, so

$$
\begin{equation}
\label{eq:abel}
\Sigma(R) \;=\; 2A\, I(\gamma)\; R^{\,1-\gamma},
\qquad
I(\gamma) \;=\; \int_0^\infty (1+u^2)^{-\gamma/2}\, du.
\end{equation}
$$

Every trace of $R$ inside the integral is gone — the substitution swept all
of it out front as the single factor $R^{1-\gamma}$, and $I(\gamma)$ is just a
number (a standard Beta-function integral once you evaluate it, though you do
not need its value for anything that follows). $\eqref{eq:abel}$ is the whole
content of the Abel projection: **a 3-D power law of slope $\gamma$ projects
to a 2-D power law of slope $\gamma - 1$.** The exponent shifts by exactly
one, every time, for any pure power-law $\rho$, no matter what $A$ or the
scale radius are.

The convergence of $I(\gamma)$ is not a technicality. For large $u$ the
integrand behaves like $u^{-\gamma}$, and by the $p$-test you already own,
$\int^\infty u^{-\gamma}\,du$ converges only for $\gamma > 1$. Below that, the
sightline integral does not merely get large — it is infinite: an idealized,
untruncated power-law halo with $\gamma \le 1$ has infinite surface density at
*every* radius. That is not a statement about real galaxies (which flatten
and truncate at large $r$); it is a statement about this exact scale-free
parameterization, which is precisely what the campaign's EPL mass profile is.
The model's own prior on $\gamma$,
`gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 2.7)`
(`reproductions/claude-giga-lens/cgl/e2.py:110`), happens to put its hard
lower wall at exactly the boundary this section's math identifies as fatal.
Nothing in the campaign's own notes says that convergence argument was the
deliberate reason for choosing $1.0$ rather than some other round number
below the isothermal value — but the mathematics does not care why the wall
was placed there: below $\gamma=1$, the projection this section just derived
stops being well-defined, not merely improbable, so a wall somewhere at or
above $1.0$ is not an arbitrary modeling choice, whether or not it was
chosen for that reason.

That wall is worth a number. The campaign's headline result is
<!-- check: ch25.gamma_money = 1.103 ± 0.001 -->
$\gamma_{\mathrm{binned}}(\mathrm{corr,low}) = 1.103\pm0.008$
(`reproductions/claude-giga-lens/CAMPAIGN.md:134`) — a value this guide's
later chapters spend real effort auditing
([Ch. 20](20-profiles.md#the-epl-and-gamma),
[Ch. 25](25-money-number.md#the-verdict)). For now, treat it only as a number
to plug into $\eqref{eq:abel}$: it sits
<!-- check: ch25.distance_above_wall_in_own_sigma = 12.9 ± 0.05 -->
12.9 of its own posterior widths above that hard wall — comfortably clear of
it, but only $\eqref{eq:abel}$ tells you *why* the wall is at 1.0 and not, say,
0.5 or 1.5.

Here is the same relationship, checked against real code rather than left as
algebra. `epl_kappa` (`site/guide_src/lensing.py:81`) is the guide's
implementation of the campaign's actual EPL convergence law — the same
function [Ch. 18](18-magnification.md#magnification-is-a-jacobian) and
[Ch. 20](20-profiles.md#the-epl-and-gamma) use:

```
kappa(R) = 0.5 * (3 - gamma) * (theta_E / R)**(gamma - 1)
```

The prefactor $(3-\gamma)/2$ and the scale $\theta_{\mathrm{E}}$ are this
formula's version of $2A\,I(\gamma)$ in $\eqref{eq:abel}$ — where that
prefactor comes from is [Ch. 20](20-profiles.md#the-epl-and-gamma)'s job, not
this chapter's. What this chapter derived is the *exponent*, and a ratio at
two radii cancels every multiplicative constant, prefactor included, and
tests the exponent alone:

$$
\frac{\kappa(R_2)}{\kappa(R_1)} = \left(\frac{R_1}{R_2}\right)^{\gamma-1}.
$$

For the isothermal slope $\gamma=2$, doubling the radius from
$R_1=0.5''$ to $R_2=1.0''$ must exactly halve $\kappa$:

<!-- check: ch03.sigma_ratio_isothermal = 0.5 ± 1e-9 -->
<!-- check: ch03.exponent_match_isothermal = 0.0 ± 1e-9 -->

$$
\kappa(1.0'') / \kappa(0.5'') = 0.5, \qquad \text{predicted } (0.5/1.0)^{2-1} = 0.5.
$$

For the campaign's money number $\gamma = 1.103$, the same doubling barely
moves $\kappa$ at all:

<!-- check: ch03.sigma_ratio_money = 0.9311 ± 0.001 -->
<!-- check: ch03.exponent_match_money = 0.0 ± 1e-9 -->

$$
\kappa(1.0'') / \kappa(0.5'') = 0.9311, \qquad \text{predicted } (0.5/1.0)^{1.103-1} = 0.9311.
$$

An isothermal halo's surface density drops by half every time the radius
doubles; a $\gamma = 1.103$ halo's surface density drops by seven percent.
$\eqref{eq:abel}$ says that difference is not a subtlety of the fit — it is
the entire content of what $\gamma$ means. Whether $1.103$ is a trustworthy
measurement of a real galaxy, or an artifact of a mis-specified likelihood, is
exactly the question [Ch. 25](25-money-number.md#the-verdict) spends its
report card on; this chapter only establishes what the number would mean if
you believed it.

## Why a 2-D point mass has a logarithmic potential { #why-log-potential }

Gauss's law is itself an accumulation statement — it says the net flux
through a closed surface equals (a constant times) the mass enclosed. In 3-D,
for a point mass and a spherical surface of radius $r$, spherical symmetry
makes the flux integral trivial: the field $g(r)$ is the same everywhere on
the sphere, so the flux is just $g(r)$ times the sphere's area,

$$
g(r)\,(4\pi r^2) = -4\pi G M \quad\Longrightarrow\quad g(r) = -\frac{GM}{r^2},
$$

the inverse-square law. Integrating $g = -d\Phi/dr$ gives the familiar
Newtonian potential $\Phi(r) = -GM/r$.

Now run the identical argument with the enclosing surface confined to two
dimensions — a circle instead of a sphere. This is not a toy: the lensing
potential $\psi$ genuinely lives in the 2-D sky plane, sourced by exactly the
projected surface density $\Sigma(R)$ the previous section built, so a 2-D
Gauss's law is the correct tool, not an analogy borrowed from one. A circle of
radius $R$ has circumference $2\pi R$ instead of a sphere's area $4\pi r^2$,
so enclosing a mass $M$ gives

$$
g(R)\,(2\pi R) = -(\text{const})\,M \quad\Longrightarrow\quad
g(R) \propto -\frac{M}{R},
$$

**inverse-linear, not inverse-square.** Integrating $g = -d\Phi/dR$ against a
$1/R$ field gives $\Phi(R) \propto \ln R$: a **logarithmic** potential —
note the sign flip relative to 3-D: $\int R^{-2}\,dR = -R^{-1}$ carries a
minus sign that $\int R^{-1}\,dR = \ln R$ does not, so the logarithm comes
out positive even though $g$ itself is negative (attractive). That
is the entire reason a lens's potential shows up wearing a logarithm in every
textbook derivation you will meet later — it is the fundamental (Green's)
solution of the 2-D Laplacian. [Ch. 6](06-vector-calculus.md#greens-function)
derives it formally from the Laplacian directly, and checks numerically that
it is harmonic everywhere except at the source, which is this section's
result stated the other way around.

The two derivations in this chapter now click together. Take the isothermal
case, $\gamma = 2$, where the previous section gave $\Sigma(R) \propto 1/R$.
Feed that into the ring-accumulation integral from the first section:

$$
M(<R) \;\propto\; \int_0^R \frac{1}{R'}\cdot 2\pi R'\, dR' \;=\; 2\pi \int_0^R dR' \;\propto\; R.
$$

The enclosed mass of an isothermal profile grows *linearly* with radius —
each new ring you add contributes the same amount of mass as the last one,
because the ring's shrinking density ($1/R'$) exactly cancels its growing
area ($R'$). Now put that into the 2-D field law just derived:

$$
g(R) \;\propto\; -\frac{M(<R)}{R} \;\propto\; -\frac{R}{R} \;=\; \text{const}.
$$

The radius cancels completely. An isothermal sphere's deflection has the same
magnitude at every radius — the exact fact `sis_deflection`
(`site/guide_src/lensing.py:54`) states as a one-line comment: "the flat
rotation curve of a galaxy in one line: the deflection does not care how far
out you are, only which way." You have now derived that sentence from two
integrals, without reading a single line of the code that implements it. (Two
honesty notes for later chapters: this argument is Newtonian, and general
relativity multiplies the whole thing by a factor of two that
[Ch. 16](16-deflection.md#the-factor-of-two) derives; and turning $g$ into an
actual lens *equation* — a map from source position to image position —
is [Ch. 17](17-lens-equation.md#the-lens-equation)'s job.)

## Connect to the repo { #connect }

- `site/guide_src/lensing.py:81` — `epl_kappa`, the EPL convergence law whose
  exponent this chapter derived; `site/guide_src/lensing.py:54` —
  `sis_deflection`, the constant-magnitude deflection this chapter's two
  derivations combine to explain; `site/guide_src/lensing.py:179` —
  `mean_kappa_within`, the ring-accumulation integral of the first section,
  reused for the Einstein-radius identity in Ch. 19.
- `reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/profiles/mass/epl.py:24` —
  gigalens' own EPL code sets `t = gamma - 1`, the same exponent shift derived
  here, in the production profile this campaign actually fits.
- `reproductions/claude-giga-lens/cgl/e2.py:110` — the campaign's prior on
  $\gamma$, `TruncatedNormal(2.0, 0.25, 1.0, 2.7)`; its low wall at $1.0$ is
  where the projection integral in $\eqref{eq:abel}$ stops converging, not an
  arbitrary modeling choice.
- `reproductions/claude-giga-lens/CAMPAIGN.md:134` — the money number this
  chapter used as a worked example, $\gamma_{\mathrm{binned}}(\mathrm{corr,low})=1.103\pm0.008$.
- `reproductions/claude-giga-lens/cgl/marg.py:9` — the campaign's other
  integral-as-marginalization, done analytically over the 28 shapelet
  amplitudes counted at `reproductions/claude-giga-lens/cgl/likelihood.py:8`,
  rather than numerically over a sightline.

## Exercises { #exercises }

??? question "Exercise 3.1 — the exponent, from scratch"
    Starting from $\Sigma(R) = 2A\int_0^\infty (R^2+z^2)^{-\gamma/2}\,dz$,
    carry out the substitution $z = Ru$ yourself and confirm
    $\eqref{eq:abel}$. Then state, in one sentence, why the integral
    $I(\gamma)$ requires $\gamma > 1$ to exist at all.

    ??? success "Solution"
        $R^2 + z^2 = R^2(1+u^2)$ and $dz = R\,du$, so $(R^2+z^2)^{-\gamma/2}dz = R^{-\gamma}(1+u^2)^{-\gamma/2}\cdot R\,du = R^{1-\gamma}(1+u^2)^{-\gamma/2}\,du$. The factor $R^{1-\gamma}$ does not depend on $u$, so it comes straight out of the integral, leaving $\Sigma(R) = 2A\,R^{1-\gamma}\int_0^\infty(1+u^2)^{-\gamma/2}\,du$.

        That is $\eqref{eq:abel}$. For large $u$ the integrand behaves like $u^{-\gamma}$; by the $p$-test, $\int^\infty u^{-\gamma}\,du$ converges only for $\gamma>1$, so a pure power-law density with $\gamma \le 1$ has infinite surface density at every projected radius.

??? question "Exercise 3.2 — an intermediate slope"
    Without running any code, predict the ratio $\kappa(1.0'')/\kappa(0.5'')$
    for $\gamma = 1.5$ — a slope roughly midway between the isothermal value
    and the money number. Then check it.

    ??? success "Solution"
        $\eqref{eq:abel}$'s exponent gives $(0.5/1.0)^{1.5-1} = (0.5)^{0.5} =
        1/\sqrt{2} \approx 0.7071$.
        <!-- check: ch03.sigma_ratio_mid = 0.7071 ± 0.001 -->
        <!-- check: ch03.exponent_match_mid = 0.0 ± 1e-9 -->
        `epl_kappa` at $\gamma=1.5$ gives the same ratio, $0.7071$, to
        machine precision — the exponent match is $0$ to $10^{-9}$. A slope
        between $1.103$ and $2$ produces surface-density falloff between
        "barely drops" and "halves": the exponent interpolates exactly the
        way $\eqref{eq:abel}$ says it must.

??? question "Exercise 3.3 — the prior wall, quantified"
    The campaign's $\gamma$ prior is `TruncatedNormal(2.0, 0.25, 1.0, 2.7)`
    (`reproductions/claude-giga-lens/cgl/e2.py:110`). Using the money number
    <!-- check: ch25.gamma_money = 1.103 ± 0.001 -->
    $\gamma=1.103\pm0.008$, how many
    of its *own* posterior widths separate it from the wall at $1.0$? Is the
    wall clipping the posterior, or just sitting nearby?

    ??? success "Solution"
        $(1.103 - 1.0)/0.008 \approx 12.9$
        <!-- check: ch25.distance_above_wall_in_own_sigma = 12.9 ± 0.05 -->
        posterior widths. A boundary that far from the bulk of the posterior
        mass is not clipping it — the posterior would look identical if the
        wall were moved further down. What the wall *does* constrain is the
        model itself: it is the smallest $\gamma$ for which
        $\eqref{eq:abel}$'s projection integral converges, so no amount of
        data pressure could ever push a fit past it in a way that made
        mathematical sense.

??? question "Exercise 3.4 — deriving the flat rotation curve"
    Using only (a) $\Sigma(R)\propto 1/R$ for an isothermal profile and (b)
    the 2-D field law $g(R)\propto -M(<R)/R$, show that the isothermal
    deflection magnitude is independent of radius, and identify which of the
    two facts would have to change for that to fail.

    ??? success "Solution"
        $M(<R) = \int_0^R \Sigma(R')\,2\pi R'\,dR' \propto \int_0^R dR' \propto R$ because the $1/R'$ density and the $R'$ ring area cancel inside the integrand. Then $g(R)\propto -M(<R)/R \propto -R/R =$ constant. It fails for any $\gamma \ne 2$: the enclosed-mass integral would give $M(<R)\propto R^{2-\gamma}$ (from $\Sigma\propto R^{1-\gamma}$ and one more power of $R$ from the ring area), and $g(R)\propto R^{2-\gamma}/R = R^{1-\gamma}$ would depend on radius for every slope except $\gamma=2$. Isothermal is not "a" profile with
        a flat rotation curve — it is *the* profile whose projected exponent
        exactly cancels the ring-area exponent, and that cancellation is a
        single-value coincidence in $\gamma$.
