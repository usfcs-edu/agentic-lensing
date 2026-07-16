# 2. Stars, and why old means red

The main guide describes every lens galaxy it touches, in one clause, as having
"an old and red stellar population," and never returns to it. Chapter 12 then
builds a measurement on that unexplained premise, reading the lens's redshift off
a calcium line and the source's off an oxygen line: two galaxies, one sightline,
two atomic transitions, and the reason the choice works is nowhere in the book.
It comes down to a single exponent, derived here in one line of algebra.

!!! abstract "What you can skip"
    You do not need nuclear physics: no cross-sections, no reaction networks, no
    Gamow peak. Take "four hydrogen nuclei fuse into one helium nucleus and
    release energy" as given. You do not need the equations of stellar structure
    either — this chapter states the one result it needs from them, the
    mass–luminosity relation, as an empirical power law. If you already know why
    the main sequence is a locus and not a track, skip to
    [Why massive means short](#why-massive-means-short).

## What a star is { #what-a-star-is }

A star is a ball of plasma massive enough that its own gravity would crush it,
held up by the pressure of its own heat. The tension *is* the star: at every
radius, the weight of everything above presses down and the pressure of the gas
below pushes back, exactly hard enough to cancel it. Astronomers call that
balance **hydrostatic equilibrium**. Writing it down takes calculus, so this book
does not: [Ch. 10](../guide/10-galaxies.md#velocity-dispersion) of the main guide
has the equation. Its content is the sentence above — pressure falls off outward
at whatever rate the enclosed weight demands — and Ch. 10 reuses that
same statement for an entire galaxy, with the random motion of stars playing the
role gas pressure plays inside a star. Same balance both places; only the source
of the pressure changes.

Pressure requires heat, and heat leaks out — a star's luminosity *is* heat
escaping. So the balance is not free: something must resupply the heat, or the
star cools, the pressure falls, and gravity wins. The resupply is **fusion**. In
the core, at temperatures of order ten million kelvin, hydrogen nuclei collide
hard enough to overcome their electrostatic repulsion and stick; four end up as
one helium nucleus weighing slightly less than the four did, and the missing
mass leaves as energy at the rate $E = mc^2$. That rate is favourable enough that
a small mass deficit funds a very long time — the only reason "how long does a
star live?" has an interesting answer.

!!! tip "You already know this"
    A main-sequence star is a negative-feedback loop, stable for the reason any
    such loop is. Perturb the fusion rate upward: the core heats, hot gas
    expands, expansion cools it, and the fusion rate — steeply
    temperature-dependent — falls back. Gravity sets the gain. Worth stating
    rather than waving at, because the loop can be **broken**, and the break is
    the plot of the [last section](#what-is-left-behind): in matter whose
    pressure does not depend on temperature, heating no longer causes expansion,
    the negative feedback term vanishes, and the loop runs away. Stars do not
    merely run out of fuel. They die of control failures.

## The main sequence { #the-main-sequence }

Plot a few thousand stars with luminosity on one axis and surface temperature on
the other, and the great majority fall along a single diagonal band: the **main
sequence**. Be precise about what that band is, because the name invites a wrong
reading. It is not a track a star travels along. It is a *locus* — the set of
points occupied, at any instant, by stars fusing hydrogen in their cores. A star
arrives at one point, sits essentially still for nearly all of its life, then
leaves the band entirely when the core hydrogen runs out. Nothing slides down it.
The band is crowded for that reason: it is where a star spends almost all its
time, so it is where a star is whenever you happen to look.

What sets the point is mass, and to a good approximation only mass. Hand the
equations of stellar structure a birth mass and they return luminosity, radius,
surface temperature, lifetime and eventual fate. The one relation this chapter
needs is the empirical **mass–luminosity relation**:

$$
\frac{L}{L_\odot} \approx \left(\frac{M}{M_\odot}\right)^{3.5}.
$$

The exponent is superlinear for a mechanical reason. More mass means more weight
bearing down on the core, so the core must run at higher pressure and
temperature to hold it up — and the fusion rate is a very steep function of
temperature. A modest increase in mass buys a large increase in burn rate. The
star pays for its extra fuel by consuming it out of all proportion.

Be honest about the $3.5$: it is a mid-range fit, not a law — nearer $4$ below a
solar mass, flattening toward $3$ and below at the top end where radiation
pressure starts to matter. This chapter uses it because the argument it supports
is an order-of-magnitude argument, and [Exercise 2.1](#exercises) shows that
argument gets *stronger*, not weaker, under the accurate exponents.

Now colour. Surface temperature sets a star's colour through Wien's law
([Ch. 5](05-light.md#why-hot-means-blue)): hot peaks blue, cool peaks red.
Massive stars are hot at the surface as well as the core, so the main sequence
runs from hot, luminous, blue stars at the massive end to cool, faint, red ones
at the low-mass end. **Blue and massive and luminous** are one thing; **red and
low-mass and faint** are another. The rest of this chapter is that fact combined
with a lifetime.

## Why massive means short { #why-massive-means-short }

A star's main-sequence lifetime is the time it takes to burn its fuel:

$$
t \sim \frac{\text{fuel}}{\text{burn rate}}.
$$

The fuel is the hydrogen the star can get into its core, a roughly fixed
fraction of its total mass, so fuel $\propto M$. The burn rate is the
luminosity, because the luminosity *is* the rate at which fusion energy leaves.
So burn rate $\propto M^{3.5}$, and

$$
t \;\propto\; \frac{M}{M^{3.5}} \;=\; M^{-2.5}.
$$

That is the whole derivation. That exponent carries this book's entire account
of galaxy colour, and two chapters of the main guide that never mention it.
Calibrate on the one star whose lifetime we have an independent handle on — the
Sun's main-sequence lifetime is about 10 Gyr:

<!-- check: pch02.sun_lifetime_gyr = 10 ± 0.01 -->

$$
t(M) \approx 10\ \text{Gyr} \times \left(\frac{M}{M_\odot}\right)^{-2.5}.
$$

Measure the results against the age of the universe, which this book quotes as
its *own* cosmology returns it — the `FlatLambdaCDM(70, 0.3)` every campaign in
this repository asserts —
$13.47$ Gyr<!-- check: pch02.universe_age_gyr = 13.47 ± 0.01 -->. That is not the
figure you will find in a textbook; the gap is real, is not a rounding error, and
[Ch. 11](11-big-bang.md#running-it-backwards) explains it rather than smoothing
it over. Figure 2.1 draws its line from the identical number.

**A star of 10 $M_\odot$.** Ten times the Sun's mass, so $10^{-2.5}$ times its
lifetime:

<!-- check: pch02.star_10msun_lifetime_myr = 31.62 ± 0.01 -->

$$
t = 10\ \text{Gyr} \times 10^{-2.5} = 0.0316\ \text{Gyr} = 31.62\ \text{Myr}.
$$

That is a fraction
$0.00316$<!-- check: pch02.ratio_10msun_to_sun = 0.00316 ± 0.0001 --> of the
Sun's life, and
$0.00235$<!-- check: pch02.ratio_10msun_lifetime_to_universe_age = 0.00235 ± 0.0001 -->
of cosmic history — two-tenths of one percent. Read it as a statement about
evidence rather than about stars: if you are looking at a 10 $M_\odot$ star, it
was made essentially *now*. Blue light from a galaxy is not a fact about its
past; it is a report on its present.

**A star of 0.5 $M_\odot$.** Half the Sun's mass, so $0.5^{-2.5}$ times its
lifetime:

<!-- check: pch02.star_half_msun_lifetime_gyr = 56.57 ± 0.01 -->

$$
t = 10\ \text{Gyr} \times 0.5^{-2.5} = 56.57\ \text{Gyr},
$$

which is
$4.2$<!-- check: pch02.ratio_half_msun_lifetime_to_universe_age = 4.2 ± 0.01 -->
times the age of the universe. Read that one plainly: **no half-solar-mass star
has ever died.** Not one, anywhere — the universe has not run long enough for the
first one to finish. Every low-mass star ever formed, and they are the
overwhelming majority of the Milky Way's $10^{11}$
([Ch. 1](01-scale-ladder.md#our-galaxy)), is on the main sequence right now,
still burning, still red.

Treat $56.57$ Gyr as a lower bound rather than a measurement. Two independent
corrections both push it upward, and [Exercise 2.1](#exercises) is where you work
out which. The conclusion does not depend on the number being right — only on its
being enormous, and both corrections make it more so.

<figure markdown="span">
  ![Main-sequence lifetime against stellar mass on log-log axes, a straight line falling as M to the minus 2.5, with the Sun, a 10 solar-mass star and a 0.5 solar-mass star marked, and the age of the universe drawn as a dashed horizontal line](figures/p02-main-sequence-light.svg#only-light){ width="90%" }
  ![Main-sequence lifetime against stellar mass on log-log axes, a straight line falling as M to the minus 2.5, with the Sun, a 10 solar-mass star and a 0.5 solar-mass star marked, and the age of the universe drawn as a dashed horizontal line](figures/p02-main-sequence-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 2.1.** Main-sequence lifetime against birth
  mass, both axes logarithmic, so the power law
  $t = 10\,\mathrm{Gyr}\,(M/M_\odot)^{-2.5}$ plots as the straight line its
  exponent promises. The dashed horizontal line is the age of the universe on
  this book's own cosmology. Everything *above* that line has never happened: a
  star whose lifetime plots above it cannot have finished, whenever it formed.
  The Sun sits a little below the line; $0.5\,M_\odot$ sits well above it;
  $10\,M_\odot$ sits two and a half decades below the Sun, which is the exponent
  $2.5$ read straight off the axes.</figcaption>
</figure>

!!! tip "You already know this"
    A lens galaxy is never resolved into stars. What a spectrograph gets is one
    integrated spectrum — [Ch. 12](../guide/12-spectroscopy.md#spectral-lines)
    calls it "the light of a few hundred billion stellar photospheres, each
    close to a blackbody" — a mixture weighted by each component's luminosity.
    Notice what $L \propto M^{3.5}$ and $t \propto M^{-2.5}$ do to that mixture
    *jointly*: the massive components are wildly over-weighted, being the
    luminous ones, and wildly short-lived. A population's integrated colour is
    dominated by exactly the components that disappear first. Stop feeding it new
    massive components and it does not fade uniformly — it **reddens**, fast,
    because the bluest and brightest terms drop out of the sum while the red ones
    stay, and will stay for longer than the universe has existed. That is the
    entire content of "old and red."

This is also, directly, the main guide's line selection. The [OII] doublet is an
emission feature of ionized gas, and the only thing that ionizes gas on a
galactic scale is ultraviolet from hot, massive, blue stars — the ones that last
$31.62$ Myr<!-- check: pch02.star_10msun_lifetime_myr = 31.62 ± 0.01 -->. So
[OII] never says a galaxy *has* stars; it says a galaxy made stars within the
last few tens of Myr. Ca II K is an absorption feature of calcium in cool stellar
atmospheres, which is what an old population is made of. Chapter 12 asserts that
split in two parentheses and moves on; the reason it holds is one exponent.

## What is left behind { #what-is-left-behind }

When the core hydrogen is gone, the loop from the first section loses its heat
source. The core contracts, the outer layers expand and cool, and the star
leaves the main sequence to become a giant: large, cool, and red for a reason
that has nothing to do with being low-mass.

Flag that, because this chapter's title is a trap taken too literally. An
individual star is red either because it is small or because it is evolved —
opposite ends of a life. It is *populations* that are red because they are old.
A red dwarf is not an old star; it is a low-mass one.

What happens after the giant phase forks on birth mass, at roughly
$8\,M_\odot$<!-- check: pch02.core_collapse_threshold_msun = 8 ± 0.01 -->.

**Below the threshold.** The star sheds its envelope and leaves its bare core
behind: a **white dwarf**, fusing nothing, only cooling. A typical one carries
about $0.6\,M_\odot$<!-- check: pch02.wd_mass_msun = 0.6 ± 0.01 --> inside a
radius of roughly $6371$ km<!-- check: pch02.wd_radius_km = 6371 ± 1 --> — the
Earth's radius<!-- check: pch02.earth_radius_km = 6371 ± 1 -->, which is the
point of quoting it that way — for a density of
$1.10$ tonne/cm³<!-- check: pch02.wd_density_tonne_per_cm3 = 1.10 ± 0.01 -->: a
sugar cube of the stuff outweighs a person. Nothing holds it up but electron
degeneracy pressure — a quantum-mechanical refusal of electrons to share a state
— and that pressure does not depend on temperature. The thermostat is gone,
which is why there is a ceiling: above the **Chandrasekhar mass**,
$1.4\,M_\odot$<!-- check: pch02.chandrasekhar_mass_msun = 1.4 ± 0.01 -->,
degeneracy pressure cannot carry the weight and the object cannot exist as a
white dwarf. That ceiling is what makes Type Ia supernovae *usable* as standard
candles — an explosion triggered at a fixed mass should release a fixed energy —
and so a rung of the distance ladder ([Ch. 9](09-distance-ladder.md#supernovae))
and the discovery of dark energy
([Ch. 13](13-dark-energy.md#the-supernova-surprise)). "Usable" is doing work
there: real Type Ia's are not identical, only alike enough that the residual
spread calibrates out of the light-curve shape, and what tips a white dwarf over
the ceiling — accretion from a companion, or a merger — is unsettled. Ch. 9
inherits both; this chapter needs only the ceiling.

**Above the threshold.** The star fuses its way up the periodic table to iron,
which yields no energy when fused. The heat source stops abruptly, the core
collapses, and the rest is blown off as a supernova. What remains is a **neutron
star** — $1.4\,M_\odot$<!-- check: pch02.ns_mass_msun = 1.4 ± 0.01 --> inside a
radius of $11$ km<!-- check: pch02.ns_radius_km = 11 ± 0.01 -->, a sphere you
could drive across — or, above some mass, a black hole. Its mean density is
$4.99\times10^{8}$ tonne/cm³<!-- check: pch02.ns_density_tonne_per_cm3 = 4.99e8 ± 0.01e8 -->,
which is $4.53\times10^{8}$ times the white dwarf's
<!-- check: pch02.ns_wd_density_ratio = 4.53e8 ± 0.01e8 -->. Both densities are
computed from the masses and radii above, which are textbook benchmarks rather
than repo measurements — [Exercise 2.2](#exercises) is where that distinction
earns its keep.

None of these remnants shines the way a main-sequence star does, and none makes
[OII]. A galaxy that stopped forming stars long ago is left with exactly this:
low-mass stars still on the main sequence, the giants some have become, and a
growing population of dark remnants. Old, red, and — for a lens model — massive
without being especially bright.
[Ch. 3](03-galaxies.md#why-star-formation-stops) takes up the question this
chapter leaves open: what stops a galaxy making new blue stars at all.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 10 — Galaxies, Sersic profiles, and velocity dispersion](../guide/10-galaxies.md#ellipticals)**
      characterises every lens in the repository in one sentence: an
      elliptical's light is "smooth, centrally concentrated, no spiral arms, no
      ongoing star formation, an old and red stellar population." Five clauses
      presented as one observation — but "no ongoing star formation" is the
      *cause* and "old and red" is its *consequence*, and the mechanism joining
      them is this chapter's $t \propto M^{-2.5}$: cut off the supply of new
      massive stars and the blue light drains out of the population within tens
      of Myr while the red light stays for longer than the universe has existed.
      Ch. 10 spends its effort on the *dynamical* half of that story —
      disordered orbits, the Jeans equation, the isothermal conspiracy — and
      never says why the colour comes along with it.
    - **[Ch. 12 — Redshifts and what a spectrum tells you](../guide/12-spectroscopy.md#measuring-redshift)**
      measures two redshifts down one sightline using two different lines, and
      justifies the choice entirely inside two parentheses: Ca II K because "its
      depth traces old stars, so it shows up in a quiescent lens galaxy, not a
      star-forming background source," and the [OII] doublet because it is
      "present in a star-forming galaxy, essentially absent in an old
      elliptical." That is the whole argument, asserted twice and never derived.
      You now have the mechanism under it, and the line selection reads as what
      it is: the $M^{-2.5}$ law used as an instrument.

## Exercises { #exercises }

??? question "Exercise 2.1 — How much does the 3.5 matter?"
    The mass–luminosity exponent is not really $3.5$; below a solar mass it is
    closer to $4$. Redo the 0.5 $M_\odot$ lifetime with $L \propto M^{4}$ — you
    need the direction, not the arithmetic — and say whether the chapter's
    conclusion survives. Then name the second, independent reason the
    $56.57$ Gyr figure is too small.

    ??? success "Solution"
        With $L \propto M^{4}$, $t \propto M/M^{4} = M^{-3}$. For
        $M < M_\odot$ the base is below 1, so a more negative power makes the
        result *larger*: the lifetime goes up. The conclusion was already safe at
        $4.2$
        <!-- check: pch02.ratio_half_msun_lifetime_to_universe_age = 4.2 ± 0.01 -->
        times the age of the universe, and gets safer.

        The second reason is the fuel step, not the burn step. Fuel $\propto M$
        counts only the core's hydrogen; a fully convective 0.5 $M_\odot$ star
        circulates essentially all of its hydrogen through the burning region, so
        its usable fuel fraction beats the Sun's. Both corrections carry the same
        sign, which is the point worth keeping: the value of an
        order-of-magnitude estimate is not that it is accurate, it is that you
        can bound which way it is wrong.

??? question "Exercise 2.2 — A sugar cube of each"
    Using $\rho = M/(\tfrac{4}{3}\pi R^{3})$ and
    $M_\odot = 1.989\times10^{30}$ kg, compute the mean density in tonnes per
    cm³ of a 0.6 $M_\odot$ white dwarf of radius 6371 km and a 1.4 $M_\odot$
    neutron star of radius 11 km, and take the ratio. Then: which of those four
    inputs does this repository actually stand behind?

    ??? success "Solution"
        White dwarf: $0.6 \times 1.989\times10^{30}$ kg over
        $\tfrac{4}{3}\pi(6.371\times10^{8}\,\mathrm{cm})^{3}$ gives
        $1.10$ tonne/cm³
        <!-- check: pch02.wd_density_tonne_per_cm3 = 1.10 ± 0.01 -->. Neutron
        star: $1.4 \times 1.989\times10^{30}$ kg over
        $\tfrac{4}{3}\pi(1.1\times10^{6}\,\mathrm{cm})^{3}$ gives
        $4.99\times10^{8}$ tonne/cm³
        <!-- check: pch02.ns_density_tonne_per_cm3 = 4.99e8 ± 0.01e8 -->, a ratio
        of $4.53\times10^{8}$
        <!-- check: pch02.ns_wd_density_ratio = 4.53e8 ± 0.01e8 -->.

        None of the four inputs is repo-derived — they are textbook benchmarks,
        and the white dwarf's "6371 km" is Earth's radius standing in for "about
        Earth-sized." Only the arithmetic on top of them is computed, so only the
        arithmetic is checked. That is the habit worth taking from this exercise:
        a number that reproduces is not the same as a number that is measured,
        and a build that verifies the first says nothing about the second.

??? question "Exercise 2.3 — Which one is the lens?"
    Two spectra come from the same arcsecond of sky. Spectrum A shows a strong
    [OII] emission doublet and weak calcium absorption. Spectrum B shows deep
    Ca II H & K absorption and essentially no [OII]. Which is the better
    candidate for the foreground lens galaxy? What timescale is spectrum A
    actually reporting on? And what would you need before concluding anything
    about which galaxy is in front?

    ??? success "Solution"
        B is the lens candidate: old population, no massive stars, therefore no
        ionizing ultraviolet, therefore no star formation for a long time — a
        quiescent elliptical, the class of galaxy massive enough to lens at all,
        the [deepest potential wells around](../guide/10-galaxies.md#ellipticals).
        A's [OII] is present-tense: it reports on the last few tens of Myr
        specifically, because that is how long the stars supplying the ionizing
        photons last — $31.62$ Myr
        <!-- check: pch02.star_10msun_lifetime_myr = 31.62 ± 0.01 --> for a
        10 $M_\odot$ star. It is not a history of the galaxy.

        The trap is the last question. Colour and line type are not geometry.
        Nothing in either spectrum says which galaxy is *in front* — that is an
        inference from a prior about which galaxies lens, not a measurement. What
        settles it is the redshifts, one wavelength ratio per spectrum, and the
        fact that the higher-$z$ object is the more distant one. That chain is
        [Ch. 8](08-redshift.md#bigger-z-means-farther), the one the main guide
        never closes.
