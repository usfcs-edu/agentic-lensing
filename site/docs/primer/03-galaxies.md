# 3. Galaxies, and which ones bend light

[Chapter 10 of the main guide](../guide/10-galaxies.md) opens with the best
one-line statement of what this program is: "Every lens in this repository is
a galaxy first and a set of equations second." It then spends the rest of the
chapter on the equations, because it is entitled to assume you already hold
the galaxy. This chapter is the galaxy. By the end you will know what one is,
why the two families in any survey image look as different as they do, why
one of them stopped making stars and went red, and why that same family —
massive ellipticals — is what every campaign here points a telescope at.
Chapter 2 gave you a star. A galaxy is $10^{11}$ of them, some gas, and one
further ingredient that outweighs all of it.

!!! abstract "What you can skip"
    You do not need galaxy-formation simulations, orbit integration, or the
    taxonomy of Hubble's tuning fork. You need no stellar dynamics either, and
    this chapter gives you none: the *dynamical* distinction between the two
    families — ordered rotation versus disordered orbits — is handed off
    rather than explained, because
    [Ch. 10](../guide/10-galaxies.md#ellipticals) does it in its opening
    paragraph and better than a restatement would. What is left is the part
    Ch. 10 assumes and never states: what these objects are, what they look
    like, and why one kind is red.

## What a galaxy is { #what-a-galaxy-is }

A galaxy is a gravitationally bound system of stars, gas, dust and dark
matter, separated from the next by a gap far larger than itself. That is the
whole definition. Nothing enforces it and nothing maintains it; it is what a
large enough overdensity of matter settles into, given time.

Take the Milky Way as the ruler ([Ch. 1](01-scale-ladder.md#our-galaxy)):
$10^{11}$ stars in a disk about $10^{5}$ light-years across. Ch. 1 gives no
thickness, so take a further $10^{3}$ light-years on that axis — a round
figure rather than a measurement, and nothing below leans on its second
digit. Those three numbers answer the first question worth asking, which is
how full it is. Model the disk as a uniform cylinder, radius
$R = 5\times10^{4}$ ly, thickness $h = 10^{3}$ ly. Its volume is

<!-- check: pch03.mw_volume_pc3 = 2.264e11 ± 0.01e11 -->

$$
V = \pi R^{2} h \approx 2.26\times10^{11}\ \mathrm{pc}^{3},
$$

so spreading $10^{11}$ stars through it uniformly gives a stellar number
density of

<!-- check: pch03.stars_per_cubic_pc_mw = 0.442 ± 0.001 -->

$$
n = \frac{10^{11}}{2.26\times10^{11}\ \mathrm{pc}^{3}} \approx 0.442\ \mathrm{stars\ pc}^{-3}.
$$

Roughly one star per two cubic parsecs. Invert that for a typical spacing —
$n^{-1/3}$, the side of the cube each star gets to itself — and you get
$1.313$ pc <!-- check: pch03.mean_star_sep_pc = 1.313 ± 0.001 -->, about
$2.7\times10^{5}$ AU
<!-- check: pch03.mean_star_sep_au = 2.708e5 ± 100 -->. The Sun is $0.0093$ AU
across <!-- check: pch03.sun_diameter_au = 0.0093 ± 0.0001 -->, so the gap
between neighbours is around

<!-- check: pch03.mean_star_sep_over_sun_diameter = 2.91e7 ± 0.01e7 -->

$$
\frac{2.7\times10^{5}\ \mathrm{AU}}{0.0093\ \mathrm{AU}} \approx 2.9\times10^{7}
$$

times the diameter of the object doing the occupying. A galaxy is not a
crowd. It is $10^{11}$ objects separated by thirty million of their own
diameters.

That number needs an immediate correction, because the model behind it is
crude in a known direction. A real disk is not uniform: it is centrally
concentrated, and it thins with radius and with height above the midplane.
The Sun sits well out in it, where the measured density is about $0.1$ stars
per cubic parsec
<!-- check: pch03.local_star_density_pc3 = 0.1 ± 0.001 -->, making the
uniform-disk average above too high by a factor of $4.42$
<!-- check: pch03.density_ratio_uniform_over_measured = 4.42 ± 0.01 -->. That
is a real error, not a rounding difference. It does not touch the conclusion,
for one reason worth stating: it runs in the conservative direction. The true
local density is *lower*, so the true spacing is *larger*, and the galaxy is
emptier than the calculation above claims rather than fuller. That factor is
the entire warranty on this density — an order-of-magnitude statement about
emptiness, nothing more. Do not carry it into anything needing a second
significant figure.

Emptiness has one consequence, and
[the section below](#why-star-formation-stops) turns on it. When two galaxies
pass through each other — and they do, routinely — essentially no two stars
collide. Gas fills a far larger fraction of the volume, so two galaxies'
reservoirs do run into each other, shock and heat.

!!! tip "You already know this"
    A galaxy is $10^{11}$ discrete point sources, and the main guide models
    its light with one smooth analytic function of radius — the Sersic
    profile, [Ch. 10](../guide/10-galaxies.md#the-sersic-profile). That is
    legitimate for the reason a kernel density estimate stops looking like a
    histogram once the sample is large enough: at these distances no telescope
    resolves an individual star, every pixel integrates millions, and the
    discreteness sits orders of magnitude below the photon noise. The smooth
    profile is not a concession to tractability; at that sampling density it
    is what the data are. It breaks where a KDE breaks — in the sparse
    outskirts, where the count per pixel gets small.

## Spirals and ellipticals { #spirals-and-ellipticals }

Point a telescope at enough galaxies and they sort, imperfectly, into two
families. The split is visible before it is understood, so it is worth
separating what you *see* from what it *means*.

A **spiral** is a flattened rotating disk with a central bulge, traced by
arms winding out from the middle. It holds cold gas and dust, often in dark
lanes along the arms. Its light is comparatively blue, concentrated in the
arms. The Milky Way is one.

An **elliptical** is a smooth ellipse of light — no arms, no dust lanes, no
disk, no visible structure at any radius, brightest in the middle and falling
off monotonically outward. It is red, and it has little cold gas. Ch. 10
describes exactly this and gives the reason in the same breath: "The light is
correspondingly featureless — smooth, centrally concentrated, no spiral arms,
no ongoing star formation, an old and red stellar population — because there
is no organizing structure left to draw a pattern with."

<figure markdown="span">
  ![A spiral galaxy's disk and arms beside an elliptical galaxy's smooth compact blob, drawn at the same scale](figures/p03-galaxy-types-light.svg#only-light){ width="90%" }
  ![A spiral galaxy's disk and arms beside an elliptical galaxy's smooth compact blob, drawn at the same scale](figures/p03-galaxy-types-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 3.1.** The visible split, drawn: an
  exponential disk traced by two logarithmic spiral arms (left) against a
  de Vaucouleurs profile — the Sersic index $n=4$ of
  [Ch. 10](../guide/10-galaxies.md#the-sersic-profile) — with no structure at
  any radius (right). Both panels cover the same field of view, each
  normalized to its own peak under a square-root stretch, so the comparison
  to make is of *shape*, not brightness. Two things the figure does not say:
  the relative sizes are a choice of the illustration, not a law — real
  ellipticals span a wide range, and the most massive are far larger than a
  Milky Way-sized disk — and neither panel shows the dark matter, which is
  most of what is there and none of what is drawn.</figcaption>
</figure>

The colour is the informative part, and Chapter 2 gave you why. Blue light
means massive stars, and massive stars burn out fast enough that their
presence is a statement in the present tense
([Ch. 2](02-stars.md#why-massive-means-short)): a galaxy shining blue is
making stars *now*, because the blue ones cannot be old. Red means the
massive stars are gone and only the long-lived low-mass ones remain — past
tense, a galaxy that made its stars long ago and none since. Colour is a
clock you can read off a single image. It is not a claim about composition;
the elements are much the same either way.

Two honesty notes. First, the split is a continuum, not a partition:
lenticulars sit between the two, irregulars sit off the diagram entirely, and
mergers in progress belong to no class at all. "Spiral" and "elliptical" are
the ends of a distribution, and the useful statements are about the ends.
Second, morphology is what a telescope measures, and it is the *less*
fundamental distinction. The one that matters downstream — for the mass
model, the Einstein radius, the whole apparatus — is dynamical: whether the
system is held up against its own gravity by ordered rotation or by
disordered motion. That is
[Ch. 10's opening section](../guide/10-galaxies.md#ellipticals) — its first
paragraph — and where you should read it rather than here.

## Why star formation stops { #why-star-formation-stops }

Stars form out of cold, dense gas and out of nothing else. A galaxy with no
cold gas makes no stars, and one that has run out fades from blue to red on
the timescale its massive stars take to die — fast. "Elliptical," "red," "old
stellar population" and "quiescent" are, to first order, four names for the
same fact: the gas is gone.

Does the gas going away even need an explanation? No. The Milky Way's cold
gas reservoir is of order
$10^{10}$ solar masses
<!-- check: pch03.mw_gas_mass_msun = 1e10 ± 1 --> and it converts that gas
into stars at roughly $2$ solar masses per year
<!-- check: pch03.mw_sfr_msun_per_yr = 2 ± 0.001 -->. Divide:

<!-- check: pch03.gas_depletion_gyr = 5 ± 0.01 -->

$$
t_{\mathrm{depl}} = \frac{10^{10}\ M_{\odot}}{2\ M_{\odot}\,\mathrm{yr}^{-1}}
= 5\times10^{9}\ \mathrm{yr} = 5\ \mathrm{Gyr}.
$$

Five billion years — less than the age of the universe
([Ch. 11](11-big-bang.md#running-it-backwards)). Both inputs are
order-of-magnitude benchmarks and the quotient inherits that, so read it as
"a few Gyr," not $5.0$. The lesson survives the imprecision: a star-forming
galaxy left alone shuts down within a time comparable to its own history,
purely by consuming its supply. Star formation is not a steady state that
needs breaking; it is a transient that needs *feeding*, and the Milky Way is
still blue because gas keeps falling in to refill the tank.

So the question is not "why did ellipticals stop" but "why did their supply
stop being replenished," and the honest answer is that this is an open
problem. Astronomers call the shutdown **quenching**. The candidate
mechanisms are known, individually plausible, and collectively unsettled:

- **Mergers.** Two gas-rich spirals colliding destroy each other's disks —
  each one's ordered rotation scrambled into disordered orbits — and the gas,
  which unlike the stars *does* collide, is driven to the centre, burned in a
  violent burst of star formation, and largely used up or blown out. The
  remnant is a smooth, gas-poor, dispersion-supported blob: an elliptical.
  The emptiness argument licenses this picture — only because stars never hit
  each other can two galaxies interpenetrate and leave $10^{11}$ intact stars
  in a rearranged configuration.
- **Feedback.** An accreting supermassive black hole at the centre dumps
  enough energy into the surrounding gas to heat it or eject it outright.
- **Environment.** A galaxy falling into a cluster ([Ch. 4](04-clusters.md#groups-and-clusters))
  moves through hot intracluster gas that can strip its reservoir away, and
  its halo may be too hot for fresh gas to cool and settle in at all.

Which of these dominates, in which galaxies, at which epoch, is genuinely
unsettled, and this book will not pretend otherwise. What *is* settled is the
observation: massive galaxies are overwhelmingly red, quiescent and gas-poor,
and have been for most of cosmic history. That observation is what the lens
model consumes — the input, not the explanation.

## Which ones lens { #which-ones-lens }

Now the payoff. Bending light into a visible arc takes mass, concentrated, in
a small patch of sky. Ellipticals are characteristically the most massive
galaxies in their environment — Ch. 10 calls them "the deepest potential
wells around" and notes that their ubiquity among lenses "is close to a
selection effect." That is the sentence to hold. This repository does not
study ellipticals because someone chose to, but because a galaxy that bends
light detectably is, with high probability, an elliptical. The universe did
the filtering before the sample was drawn.

The quantitative version is one line and it belongs to the main guide. An
elliptical's stars move on disordered orbits with a velocity spread
$\sigma_v$ — a standard deviation, measurable from the width of an absorption
line in its spectrum — and the Einstein radius scales as
$\theta_{\mathrm{E}} \propto \sigma_v^{2}$. Quadratic, so the most massive
ellipticals do not merely lens better than everything else; they dominate.
[Ch. 10 derives that relation](../guide/10-galaxies.md#velocity-dispersion)
from a single equilibrium condition and hands it to
[Ch. 19](../guide/19-einstein-radius.md#theta-e-from-sigma-v);
[Ch. 16](16-what-is-a-strong-lens.md#why-anyone-cares) of this book draws the
geometry it implies.

Three things follow, each of which the main guide uses without pausing:

- **The red, featureless light is a modelling asset.** A galaxy with no arms,
  no dust lanes and no star-forming knots has a surface brightness a smooth
  analytic profile describes well. That is why
  [Ch. 10's Sersic profile](../guide/10-galaxies.md#the-sersic-profile) works
  at all, and why the campaign can subtract the lens's own light cleanly
  enough to see the arcs behind it. A lensing program built around spirals
  would be fighting its own foreground.
- **The mass doing the bending is mostly not the stars.** The old red
  population is the visible marker of a potential well that is predominantly
  dark matter ([Ch. 12](12-dark-matter.md#halos)). Everything this chapter
  said about stars describes the label on the well, not its depth.
- **Nobody fully understands the profile that results.** Stars and dark
  matter have unrelated distributions, and yet the *total* mass profile of a
  real massive elliptical comes out close to isothermal, over exactly the
  radii lensing probes, for reasons no one derives from first principles.
  Ch. 10 calls it [the isothermal conspiracy](../guide/10-galaxies.md#the-isothermal-conspiracy),
  and it is no footnote: it is why this repository's prior on the slope
  $\gamma$ is centred where it is, and part of why the campaign's headline
  number is worth being suspicious of.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 10 — Galaxies, Sersic profiles, and velocity dispersion](../guide/10-galaxies.md#ellipticals)**
      is the chapter this one exists for. It opens on "a massive elliptical
      lens galaxy," describes "an old and red stellar population" with "no
      ongoing star formation," and calls ellipticals "the most massive
      galaxies in their environment: the deepest potential wells around" — all
      correct, none explained, because its job is the mathematics that
      follows. You now hold each of those assumptions: what a galaxy is, why
      one family is red (its massive stars died and nothing replaced them,
      [Ch. 2](02-stars.md#why-massive-means-short)), why star formation
      stopped (the gas ran out within a depletion time and was never
      replenished), and why the lens sample is ellipticals by construction
      rather than by choice.
    - Its skip box says "skip past the notation, not the physics." That now
      costs you nothing: the physics it means is
      [the dynamical distinction](../guide/10-galaxies.md#ellipticals) —
      ordered rotation versus disordered motion — which this chapter left for
      it, having given you the visible and historical halves first. And the
      one thing Ch. 10 does *not* resolve,
      [the isothermal conspiracy](../guide/10-galaxies.md#the-isothermal-conspiracy),
      you can now read as a real open problem rather than as notation.

## Exercises { #exercises }

??? question "Exercise 3.1 — how empty is a galaxy?"
    Model the Milky Way's disk as a uniform cylinder, radius
    $5\times10^{4}$ ly, thickness $10^{3}$ ly, containing $10^{11}$ stars.
    Compute the volume in cubic parsecs, the mean stellar number density, and
    the typical spacing $n^{-1/3}$ in units of the Sun's diameter. Then say
    what the answer implies about a collision between two galaxies.

    ??? success "Solution"
        $V = \pi R^{2}h \approx 2.26\times10^{11}\ \mathrm{pc}^{3}$
        <!-- check: pch03.mw_volume_pc3 = 2.264e11 ± 0.01e11 -->, so
        $n \approx 0.442$ stars per cubic parsec
        <!-- check: pch03.stars_per_cubic_pc_mw = 0.442 ± 0.001 --> and
        $n^{-1/3} \approx 1.313$ pc
        <!-- check: pch03.mean_star_sep_pc = 1.313 ± 0.001 -->, or
        $2.7\times10^{5}$ AU
        <!-- check: pch03.mean_star_sep_au = 2.708e5 ± 100 -->. Against the
        Sun's diameter of $0.0093$ AU
        <!-- check: pch03.sun_diameter_au = 0.0093 ± 0.0001 --> that is
        $2.9\times10^{7}$ solar diameters
        <!-- check: pch03.mean_star_sep_over_sun_diameter = 2.91e7 ± 0.01e7 -->.
        At thirty million diameters of separation the probability any two
        stars meet is negligible, so the stellar components of two colliding
        galaxies pass straight through each other. Only the gas, not
        point-like at these scales, actually collides.

??? question "Exercise 3.2 — what the uniform disk gets wrong"
    The density above is a uniform-disk average. The measured density near the
    Sun is about $0.1$ stars per cubic parsec. By what factor does the model
    miss, in which direction, and why? Then the part that matters: does the
    emptiness conclusion survive the error?

    ??? success "Solution"
        The ratio is $0.442/0.1 = 4.42$
        <!-- check: pch03.density_ratio_uniform_over_measured = 4.42 ± 0.01 -->
        <!-- check: pch03.local_star_density_pc3 = 0.1 ± 0.001 -->, so the
        uniform model is high by about four and a half. The model is exactly
        the one thing a real disk is not: uniform. Stellar density falls with
        distance from the centre and with height above the midplane, and the
        Sun sits far out in the disk — well below the average the model
        spreads evenly everywhere.

        The conclusion survives because the bias runs the safe way. A *lower*
        true density means a *larger* true spacing, so the real neighbourhood
        is emptier than the uniform estimate says, not fuller. An error with a
        known sign, order-unity magnitude and an insensitive conclusion is
        fine to use provided you say so out loud. It would not be fine if
        anything downstream needed the density itself; nothing does.

??? question "Exercise 3.3 — the gas runs out"
    A galaxy holds $10^{10}\,M_{\odot}$ of cold gas and forms stars at
    $2\,M_{\odot}$ per year, with no new gas arriving. How long until it
    stops? Compare to the age of the universe, and say what that implies about
    why the Milky Way is still blue.

    ??? success "Solution"

        $$
        t_{\mathrm{depl}} = \frac{10^{10}\ M_{\odot}}{2\ M_{\odot}\,\mathrm{yr}^{-1}} = 5\ \mathrm{Gyr}.
        $$

        <!-- check: pch03.gas_depletion_gyr = 5 ± 0.01 -->
        <!-- check: pch03.mw_gas_mass_msun = 1e10 ± 1 -->
        <!-- check: pch03.mw_sfr_msun_per_yr = 2 ± 0.001 -->
        That is shorter than the age of the universe
        ([Ch. 11](11-big-bang.md#running-it-backwards)), which is the point:
        an unfed star-forming galaxy shuts itself down within a time
        comparable to its own history, with no external mechanism required.
        The Milky Way is still forming stars not because its original gas
        lasted but because gas keeps falling in to replace what was consumed.
        Quenching is therefore not about *destroying* a steady state but about
        *interrupting a resupply* — which is why the candidate mechanisms all
        concern cutting off or heating the inflow rather than the gas already
        present.

??? question "Exercise 3.4 — why the lens sample is what it is"
    Give the two independent reasons a lens-finding campaign's lenses are
    mostly ellipticals — one about physics, one about measurement — and say
    which one Ch. 10 means by "close to a selection effect."

    ??? success "Solution"
        Physics: a resolvable Einstein radius needs a deep, compact potential
        well, and $\theta_{\mathrm{E}} \propto \sigma_v^{2}$ makes that
        steeply preferential — the ellipticals, being the most massive
        galaxies around, dominate the lensing cross-section. Measurement: a
        smooth, red, dust-free galaxy is fit well by a simple analytic profile
        ([Ch. 10](../guide/10-galaxies.md#the-sersic-profile)), so its light
        subtracts cleanly enough to reveal the arcs behind it.

        "Close to a selection effect" means the first. Nobody chose these
        ellipticals — a detectable deflection selected them before any human
        made a decision, so "the lenses are ellipticals" is a property of the
        population that survives the filter, not an assumption the campaign
        makes.
