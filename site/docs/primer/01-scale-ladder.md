# 1. The scale ladder

This chapter buys you a ruler. By the end of it you will carry two numbers —
the Milky Way holds roughly $10^{11}$ stars<!-- check: pch01.mw_stars = 1e11 ± 1e9 -->
and is roughly $10^5$ light-years across<!-- check: pch01.mw_diameter_ly = 100000 ± 1 -->
— and every mass, size and distance in the rest of this book gets quoted
against them at least once. That is not decoration. The main guide reproduces a
galaxy cluster's mass to four significant figures and cannot tell you whether
that is a lot, because the Milky Way appears nowhere in it. The ladder below is
the fix: one rung at a time, from the ground under you to the edge of what
light has had time to reach, with the units changing underfoot because no
single unit survives the climb.

!!! abstract "What you can skip"
    You own logarithms, orders of magnitude and log axes; nothing here explains
    a log scale, it only uses one. If you have read
    [Ch. 9 of the main guide](../guide/09-units.md#angles-on-the-sky) you also
    own the parsec's *arithmetic*, which it derives from one triangle and the
    small-angle approximation. What that derivation withholds, and this
    chapter supplies, is the ladder the parsec is a rung of — and the fact
    that a parsec is not a large distance. The one section to read slowly is
    [Our galaxy](#our-galaxy); everything later refers back to it.

## Powers of ten { #powers-of-ten }

The physics in this chapter is nothing. The difficulty is entirely that the
numbers span more decades than intuition carries, and intuition does not
degrade gracefully at the edge of its range — it fails silently and returns a
confident answer that is wrong by orders of magnitude.

So the ladder gets climbed rung by rung, each one quoted as a ratio to the one
below it, with the unit swapped whenever the digits get out of hand. Astronomy's
pile of distance units is not historical clutter: each exists because the
previous one produced numbers nobody could hold, and the swap points are the
rungs of this ladder.

!!! tip "You already know this"
    The unit ladder is exponent normalisation. km, AU, ly, pc and Mpc are not
    five physical ideas; they are five choices of exponent, each defined from
    its neighbours by a fixed constant, picked so the mantissa stays short.
    Astronomy renormalises its units for the same reason a floating-point
    format carries a separate exponent field: the alternative is dragging a
    long tail of zeros through every statement, most of them digits you do not
    know. When a paper quotes a distance in Mpc, read it as scientific notation
    with the exponent hidden in the unit name.

## The solar system, and where the AU runs out { #the-solar-system }

Start with the Earth: the only rung you have direct physical access to, and the
last one you can picture honestly — a ball you could walk around. The next rung
up is the Earth–Sun distance, which astronomy names the **astronomical unit**
(AU):

<!-- check: pch01.au_km = 1.496e8 ± 0.001e8 -->

$$
1\ \mathrm{AU} \approx 1.496 \times 10^{8}\ \mathrm{km},
$$

the mean radius of the Earth's orbit. This is the first rung where distance and
time stop being separate statements. Light covers that gap in

<!-- check: pch01.au_light_minutes = 8.32 ± 0.01 -->

$$
\frac{1\ \mathrm{AU}}{c} \approx 8.32\ \text{minutes},
$$

so the Sun you observe is always the Sun as it was eight minutes ago.
[Ch. 5](05-light.md#looking-back-in-time) runs that fact out to its limit.

The AU is the right unit for the solar system and for nothing else. The planets
all sit within a modest multiple of it — Neptune, the outermost, is the last
rung on Figure 1.1 inside our own system. Then the planets stop, and there is a
gap, and the gap is the point of this section.

The nearest star to the Sun is Proxima Centauri, at

<!-- check: pch01.nearest_star_ly = 4.24 ± 0.01 -->

$$
4.24\ \text{light-years} = 268{,}142\ \mathrm{AU}.
$$

<!-- check: pch01.nearest_star_in_au = 268142 ± 1 -->

Sit with the ratio, not the two numbers. The distance to the *nearest other
star* is a quarter of a million times the distance from here to the Sun. The
solar system is not a scaled-down galaxy; it is a speck with an enormous amount
of nothing around it. That emptiness is why two galaxies almost never line up
well enough to make a lens
([Ch. 16](16-what-is-a-strong-lens.md#why-it-is-rare)).

The AU has run out. Expressing interstellar distances in AU is exactly the
failure mode the previous section named: correct, useless, and six digits long.

## The light-year and the parsec { #the-light-year }

The next unit up is the **light-year**: the distance light covers in one year,

<!-- check: pch01.ly_km = 9.461e12 ± 0.001e12 -->

$$
1\ \text{light-year} \approx 9.461 \times 10^{12}\ \mathrm{km}.
$$

It is a distance, not a duration, despite the name — the most common misreading
of the unit, and worth killing on sight. But the name is not an accident: since
the unit is defined *from* a travel time, an object's distance in light-years
and the age of its arriving light are numerically the same. One number does two
jobs, and [Ch. 5](05-light.md#looking-back-in-time) is where that stops being a
convenience and becomes the foundation of every claim in the book.

Now the awkward part. Professional astronomy does not publish in light-years.
It publishes in **parsecs**, and

<!-- check: pch01.pc_in_ly = 3.2616 ± 0.001 -->

$$
1\ \mathrm{pc} \approx 3.2616\ \text{light-years}.
$$

Both units survive because they are built from different things. The light-year
is built from a physical constant, the speed of light. The parsec is built from
an *instrument*: the distance at which one AU subtends one arcsecond, which is
exactly the geometry of parallax — the only distance measurement that requires
no astrophysical assumptions at all, using the Earth's orbit as a baseline.
[Ch. 9 of the main guide](../guide/09-units.md#angles-on-the-sky) derives the
parsec from that triangle in two lines and reproduces astropy's constant to the
last bit, but never mentions parallax and never says what the unit is *for*.
[Ch. 9 of this book](09-distance-ladder.md#parallax) supplies the measurement
the guide's triangle is quietly a picture of.

Read one thing off Figure 1.1 here: the dot for "1 parsec" and the dot for the
nearest star sit almost on top of each other. Nothing in the parsec's
definition knows about Proxima Centauri, so that is a coincidence — and it is
the coincidence the distance ladder rests on. "Stars are spaced about a parsec
apart" is the same statement as "the nearest stars have parallaxes of order an
arcsecond", which is the same statement as "the measurement is barely
achievable with nineteenth-century optics". It was, barely. Had the geometry
landed the other side of that, the ladder would have had no first rung.

Above the parsec the ladder only adds prefixes: kiloparsec inside a galaxy,
**megaparsec** between galaxies, gigaparsec for the cosmological scale. The main
guide names the AU, the parsec and the Mpc, but only ever as definitions to
convert with; it never attaches a distance to any of them. That is why you can
read all 29 of its chapters and still not know whether a megaparsec is far.

## Our galaxy — the ruler { #our-galaxy }

The Sun is one star in the **Milky Way**, a barred spiral galaxy: a flattened
rotating disk of stars and gas with a central bulge, seen edge-on from inside,
which is why it looks like a band across the sky rather than a pinwheel. The
Sun sits in the disk, roughly halfway out — a bad seat, since half the galaxy
is behind dust from here.

Two numbers. This is the ruler, and it is built to be memorable.

<!-- check: pch01.mw_stars = 1e11 ± 1e9 -->
<!-- check: pch01.mw_diameter_ly = 100000 ± 1 -->

$$
N_\star \approx 10^{11}\ \text{stars}, \qquad
D \approx 10^{5}\ \text{light-years across}.
$$

A hundred billion stars, a hundred thousand light-years wide. One significant
figure each, both an exact power of ten, which is exactly the precision this
book needs from them. Light crosses the galaxy in

<!-- check: pch01.mw_crossing_yr = 100000 ± 1 -->

$$
t_{\text{cross}} = 10^{5}\ \text{years},
$$

the same number as the diameter, because a light-year is a light-year. The
reflex worth keeping: any picture of the far side of the Milky Way is a hundred
thousand years out of date, and no instrument will fix that.

Both numbers are softer than the notation suggests, and pretending otherwise
would be a simplification you would later have to unlearn. Nobody has counted
the stars. $10^{11}$ is an estimate: measure the galaxy's mass dynamically,
subtract your best guess at the gas and the dark matter
([Ch. 12](12-dark-matter.md#halos)), divide by an average stellar mass drawn
from a fitted distribution. The published range spans a factor of a few, and
that spread is a modelling choice, not a counting error. The diameter is softer
still: the stellar disk has no edge, it fades, so "$10^5$ light-years across"
encodes a convention about where you stop caring, with roughly a factor of two
of slack. And the *matter* reaches much further than the light does — the dark
halo is several times the visible disk, which is
[Ch. 12](12-dark-matter.md#rotation-curves)'s whole subject.

None of that damages the ruler, because a ruler has to be *stable and shared*,
not accurate. A cluster mass quoted in Milky Ways carries exactly the precision
of the comparison and no more — and it is still far more informative than the
same mass quoted in bare solar masses, which is what the main guide does.

## Beyond the galaxy { #beyond-the-galaxy }

The Milky Way is not alone, and the pattern from the last rung repeats: things
come in bound groups, separated by far more emptiness than the groups occupy.

The **Local Group** is our galaxy's own neighbourhood: the Milky Way,
Andromeda (the nearest large galaxy, and visible to an unaided eye from a dark
northern site), and several dozen much smaller dwarf galaxies orbiting the two
of them. [Ch. 5](05-light.md#looking-back-in-time) puts a number on how far
Andromeda is, and therefore how old its photograph is; for the ladder it is
enough that the Local Group sits a couple of decades above the ruler.

Above that are **clusters** — bound collections of galaxies, far more crowded
than the Local Group and far heavier. This is the rung the main guide does much
of its work on without ever introducing it, so
[Ch. 4](04-clusters.md#groups-and-clusters) takes it whole rather than a
paragraph here. What matters for the ladder is that a cluster sits *above* a
galaxy and is not a synonym for one. The main guide quotes an Einstein radius
for a galaxy and one for a cluster, chapters apart, and they differ for
precisely that reason — a fact no sentence in it states and
[Ch. 4](04-clusters.md#why-cluster-rings-are-bigger) exists to state.

Above clusters is the **cosmic web**: filaments of galaxies strung between
dense knots, wrapped around mostly empty voids. This is not an artist's
impression or a simulation passed off as an observation — the web is measured,
by taking redshifts for enormous numbers of galaxies and turning each into a
distance ([Ch. 8](08-redshift.md#bigger-z-means-farther)'s job). Its largest
structures are the largest structures there are. Past the web's own scale the
universe stops having features and becomes, statistically, the same everywhere.

## The whole thing { #the-whole-thing }

The top rung needs a distinction most accounts blur, so take it first: the
**observable** universe and the universe are different claims. The observable
universe is the region light has had time to reach us from since the beginning
([Ch. 11](11-big-bang.md#running-it-backwards)); it is bounded because the
universe has a finite age, not because space stops. What lies past that
boundary, and whether the universe is finite at all, is not known and probably
not knowable. This book will not pretend otherwise.

For the size of the observable part, this program's own cosmology hands you a
number: the **Hubble distance**, $c/H_0$ — the speed of light over the
expansion rate ([Ch. 10](10-expansion.md#hubbles-law)). Using the $H_0$ this
repository asserts throughout, and which
[Ch. 14](14-hubble-tension.md#two-answers) will show is actively disputed:

<!-- check: pch01.hubble_distance_mpc = 4283 ± 1 -->

$$
D_{\mathrm{H}} = \frac{c}{H_0} \approx 4283\ \mathrm{Mpc}.
$$

Be careful with what that is. $c/H_0$ is a characteristic scale for the
observable universe, not its radius. Light emitted at the beginning has been
travelling through *expanding* space the whole time, so whatever emitted it is
now considerably further away than $c$ times the travel time; the true comoving
radius is several times $D_{\mathrm{H}}$.
[Ch. 15 of the main guide](../guide/15-distances.md#three-distances) is where
that becomes an integral and three rival definitions of "distance";
[Ch. 10](10-expansion.md#what-is-actually-expanding) of this book is where it
becomes a picture. Quote $D_{\mathrm{H}}$ as the scale of the thing, not the
edge of it.

Now convert it into the only unit you now have a feel for:

<!-- check: pch01.hubble_distance_in_mw = 139685 ± 1 -->

$$
\frac{D_{\mathrm{H}}}{D_{\text{Milky Way}}} = 139{,}685\ \text{Milky Ways}.
$$

That is the ladder's punchline: the observable universe, measured in galaxies
laid end to end, is of order $10^5$ of them — the same exponent as the Milky
Way's diameter in light-years. The repetition is a coincidence of unit choice
and means nothing physical, but it is a free mnemonic to carry.

<figure markdown="span">
  ![Every rung from Earth to the observable universe on a single logarithmic distance axis in light-years, with the Milky Way marked as the book's ruler](figures/p01-scale-ladder-light.svg#only-light){ width="90%" }
  ![Every rung from Earth to the observable universe on a single logarithmic distance axis in light-years, with the Milky Way marked as the book's ruler](figures/p01-scale-ladder-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 1.1.** The whole ladder on one axis:
  distance in light-years, log scale, from the Earth's own diameter to the
  observable universe. Labelled ticks step three decades at a time, so a rung a
  finger-width from its neighbour is a thousandfold away from it.
  The highlighted rung is the Milky Way, marked *our ruler*, and it is not
  centred: about fourteen decades of ladder run below it and six above. Read
  the gaps rather than the dots: "Nearest star" is a quarter of a million AU
  from "Earth–Sun", and "1 parsec" lands almost on top of it, which is why
  parallax was measurable at all.</figcaption>
</figure>

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 9 — Arcseconds, magnitudes, and the units of the
      sky](../guide/09-units.md#angles-on-the-sky)** opens with the right
      sentence — *"Nothing astronomical comes with a ruler. Every telescope
      measures a direction, not a distance"* — and then derives the parsec
      from *"the distance at which one astronomical unit — Earth's orbital
      radius, roughly $1.496\times10^8$ km — subtends an angle of exactly
      $1''$."*<!-- check: pch01.au_km = 1.496e8 ± 0.001e8 -->
      That is the whole treatment. It never says why anyone would build a unit
      out of that triangle (the answer is parallax, a word absent from all of
      the main guide), and it never says how far a parsec gets you. You now
      have both: the AU as a physical gap light crosses in
      $8.32$ minutes<!-- check: pch01.au_light_minutes = 8.32 ± 0.01 -->, and
      the parsec as $3.2616$ light-years<!-- check: pch01.pc_in_ly = 3.2616 ± 0.001 -->
      — a distance that does not reach the nearest star.
    - **[The same chapter's closing move](../guide/09-units.md)** — *"You need
      the parsec to read an abstract, not to run this repo's own fits"* — now
      reads as what it is: the guide stepping over the ladder because its
      likelihood is "arcsec in, arcsec out" and genuinely does not need it. A
      fair engineering call, and a poor way to learn what the numbers mean.
      When its Part III starts quoting Mpc, you know
      $4283$ Mpc<!-- check: pch01.hubble_distance_mpc = 4283 ± 1 --> is roughly
      the scale of everything observable.

## Exercises { #exercises }

??? question "Exercise 1.1 — where the AU runs out"
    Using $1$ light-year $\approx 9.461\times10^{12}$ km and $1\ \mathrm{AU}
    \approx 1.496\times10^{8}$ km, convert Proxima Centauri's distance of
    $4.24$ light-years into AU. Then answer the question the number is
    actually about: is the solar system a reasonable scale model of the
    galaxy?

    ??? success "Solution"
        One multiplication:

        $$
        4.24 \times \frac{9.461\times10^{12}}{1.496\times10^{8}}
        \approx 268{,}142\ \mathrm{AU}.
        $$

        <!-- check: pch01.nearest_star_in_au = 268142 ± 1 -->
        <!-- check: pch01.nearest_star_ly = 4.24 ± 0.01 -->

        No. A scale model of the galaxy built from the solar system would put
        the next star a quarter of a million Earth–Sun distances away, and then
        $10^{11}$ of them<!-- check: pch01.mw_stars = 1e11 ± 1e9 --> at
        comparable spacing. Every diagram of "the Sun's neighbourhood"
        compresses that gap by orders of magnitude, because the honest version
        is a blank page — which is why lensing alignments are rare
        ([Ch. 16](16-what-is-a-strong-lens.md#why-it-is-rare)).

??? question "Exercise 1.2 — the galaxy is old news"
    The Milky Way is about $10^5$ light-years across. How long does light take
    to cross it? Why is that answer the same number as the diameter, and what
    does it imply about any image of the far side of the galaxy?

    ??? success "Solution"
        $10^{5}$ years.
        <!-- check: pch01.mw_crossing_yr = 100000 ± 1 -->
        It is the same number as the diameter in light-years
        <!-- check: pch01.mw_diameter_ly = 100000 ± 1 -->
        by construction, not by coincidence: the light-year is *defined* as
        the distance light covers in a year, so distance-in-light-years and
        travel-time-in-years are one quantity read in two units. So a
        photograph of the far side of our own galaxy shows it as it was $10^5$
        years ago, and no instrument can change that — the delay is a property
        of the distance, not the telescope
        ([Ch. 5](05-light.md#looking-back-in-time)).

??? question "Exercise 1.3 — the universe, in Milky Ways"
    Take the Hubble distance $D_{\mathrm{H}} = c/H_0 \approx 4283$ Mpc, with
    $1\ \mathrm{pc} \approx 3.2616$ light-years and a Milky Way $10^5$
    light-years across. How many Milky Ways across is $D_{\mathrm{H}}$? Then
    state, in one sentence, why this is *not* the radius of the observable
    universe.

    ??? success "Solution"
        Convert Mpc to pc, pc to light-years, then divide by the ruler:

        $$
        \frac{4283 \times 10^{6} \times 3.2616}{10^{5}}
        \approx 139{,}685\ \text{Milky Ways}.
        $$

        <!-- check: pch01.hubble_distance_mpc = 4283 ± 1 -->
        <!-- check: pch01.pc_in_ly = 3.2616 ± 0.001 -->
        <!-- check: pch01.hubble_distance_in_mw = 139685 ± 1 -->

        $c/H_0$ is a characteristic scale, not a radius: space expanded while
        the light was in transit, so whatever emitted the oldest light we
        receive is now several times $D_{\mathrm{H}}$ away, and "distance"
        itself splits into several inequivalent definitions
        ([Ch. 15 of the main guide](../guide/15-distances.md#three-distances)).
        Note what the answer does *not* hinge on: the two best measurements of
        $H_0$ disagree ([Ch. 14](14-hubble-tension.md#two-answers)) by a few
        percent, which moves this number by a few percent and the exponent not at
        all.

??? question "Exercise 1.4 — why two units for one distance"
    A paper reports a distance in Mpc; a press release reports the same
    distance in light-years, and $1\ \mathrm{pc} \approx 3.2616$ light-years
    converts between them. Why does the field carry both?

    ??? success "Solution"
        <!-- check: pch01.pc_in_ly = 3.2616 ± 0.001 -->
        Each is built from something different: the light-year from a physical
        constant, reporting a distance and the age of the light in one number
        ([Ex. 1.2](#exercises)); the parsec from an *instrument*, being the
        distance at which $1\ \mathrm{AU}$ subtends $1''$. A field publishes in
        the unit its measurement is natively in, so the parsec wins in papers
        and the light-year wins in prose.
