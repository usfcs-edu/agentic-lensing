# 9. The distance ladder

Chapter 5 left an ambiguity it could not resolve: flux alone never separates
"dim and nearby" from "luminous and far away." Chapter 8 gave you redshift,
read straight off a spectrum and needing no distance at all — but turning a
redshift into a distance requires the expansion rate, and measuring the
expansion rate requires a distance obtained some other way first. This chapter
is the other way, and it is not one method but four, stacked: geometry for the
nearest stars, a class of pulsating star for nearby galaxies, exploding white
dwarfs for most of the observable universe, redshift for everything beyond.
Each is calibrated by the one below it and reaches farther — which is what the
word *ladder* is doing, and also the field's central liability: every rung
inherits the rung below's zero point, and its mistakes along with it.

!!! abstract "What you can skip"
    You do not need stellar-interior physics to accept that a certain kind of
    star pulsates with a period set by its luminosity, or nuclear physics to
    accept that white dwarfs detonate at a near-fixed mass. Take both as
    empirical, the way you'd take a calibration curve as given. You also
    already own everything structural here: a ladder is a bootstrap, each rung
    is an estimator fitted against labels the previous rung produced, and the
    failure mode follows from that description alone. The astronomy is only
    *which* four estimators, and what each one costs.

## Nothing comes with a ruler { #nothing-comes-with-a-ruler }

[Ch. 9 of the main guide](../guide/09-units.md#angles-on-the-sky) opens with
the sentence this section is named for: *"Nothing astronomical comes with a
ruler. Every telescope measures a direction, not a distance."* That is not a
limitation of current instruments. A detector records where on the sky a photon
arrived and how many arrived — a direction and a brightness. Neither is a
distance, and no improvement converts one into the other, because the
information is not in the light: a sky image is a projection, and the
coordinate you want is the one the projection threw away.

So distance is never measured directly beyond the solar system. It is
*inferred*, from some other observable that distance controls, and only two are
in wide use. The first is geometry: look at the same object from two places and
its apparent direction shifts by an amount that depends on how far away it is.
The second is the inverse-square law of
[Ch. 5](05-light.md#the-inverse-square-law): if you know an object's intrinsic
luminosity $L$ independently, a measured flux $F$ pins the distance, because
$F = L/(4\pi d^2)$ then has one unknown left. Geometry is the honest one — it
assumes nothing about the object — and it runs out almost immediately.
Everything above it trades self-sufficiency for reach.

## Parallax { #parallax }

Hold a finger at arm's length and look at it with one eye, then the other. It
jumps against the background, and the jump is bigger when the finger is closer.
That is **parallax**, the only distance measurement in astronomy that is pure
geometry.

!!! tip "You already know this"
    Parallax is stereo triangulation with a known baseline — structure from
    motion, where the camera motion is given rather than estimated. The depth
    solve is the one your rectified-stereo code does, with the same pathology:
    disparity goes as $1/d$, so depth error grows as $d^2$ and the estimator
    degrades catastrophically, not gracefully, with range.

The baseline is Earth's orbital radius, one **astronomical unit** (1 AU): over
six months Earth moves to the opposite side of its orbit, and a nearby star
shifts against the far more distant background. Half that shift, as an angle,
is the star's **parallax angle** $p$, and because $p$ is always minuscule the
small-angle relation reduces the triangle to a division — $p$ in radians is
(1 AU)$/d$.

What astronomers did with that division is the origin story the main guide
omits while deriving its arithmetic. Rather than carry radians, define the
distance unit so the formula has no constant in it. A **parsec** — a
contraction of *parallax* and *arcsecond* — is *defined* as the distance at
which 1 AU subtends exactly one arcsecond ($1''$, one 3600th of a degree).
Feed that back into the triangle and it returns itself:

<!-- check: pch09.parsec_definition_check = 1.0 ± 1e-9 -->

$$
d\,[\mathrm{pc}] = \frac{1}{p\,['']}.
$$

A parsec is not a length someone chose; it is the reciprocal of the angle
astronomers actually measure. The factor the main guide leans on so heavily,
$1\ \mathrm{rad} \approx 206{,}265''$
<!-- check: pch09.arcsec_per_rad = 206264.806 ± 0.001 -->,
only restates that in radians and carries no physics.

Run it on Proxima Centauri, the nearest star to the Sun, which has the largest
parallax of any star:

<!-- check: pch09.proxima_parallax_arcsec = 0.7687 ± 1e-6 -->

$$
p = 0.7687'' \quad\Longrightarrow\quad
d = \frac{1}{0.7687} \approx 1.30\ \mathrm{pc}.
$$

<!-- check: pch09.proxima_distance_pc = 1.301 ± 0.001 -->

In [Ch. 1](01-scale-ladder.md#the-light-year)'s units that is
$4.24$ light-years
<!-- check: pch09.proxima_distance_ly = 4.243 ± 0.001 -->
— out of one measured angle and one definition, with nothing assumed about
Proxima itself. Note how small the angle is: the *largest* stellar parallax in
the sky is under one arcsecond, roughly what a coin subtends from several
kilometers away.

That is why this rung ends where it does. Gaia measures parallaxes to about
$20$ millionths of an arcsecond for its best targets, which puts the ceiling at

<!-- check: pch09.gaia_limit_pc = 50000 ± 1 -->

$$
d = \frac{1}{20\times10^{-6}} = 50{,}000\ \mathrm{pc},
$$

or about $163{,}000$ light-years
<!-- check: pch09.gaia_limit_ly = 163078 ± 1 -->.
Against this book's ruler — a Milky Way $100{,}000$ light-years across
<!-- check: pch09.mw_diameter_ly = 100000 ± 1 -->
— pure geometry, pushed to the limit of the best instrument ever built for it,
reaches
<!-- check: pch09.gaia_reach_in_mw_diameters = 1.63 ± 0.01 -->
$1.63$ Milky Way diameters. Barely past our own galaxy's far edge, and not one
step toward any other — and that ceiling is generous, since it is where the
parallax merely *equals* its own error bar. Every distance in this program
therefore rests on something less trustworthy than geometry.

## Standard candles { #standard-candles }

An object whose $L$ is knowable independently — not from its brightness, which
is the thing you are trying to explain, but from something else about it — is a
**standard candle**. Finding them is most of what the ladder is.

First the unit needs fixing, because the main guide hands you half of it.
[Ch. 9](../guide/09-units.md#magnitudes) defines the **apparent magnitude** $m$
— Pogson's backwards logarithmic flux scale, in which
$m_1 - m_2 = -2.5\log_{10}(F_1/F_2)$ and brighter means *smaller*. It describes
arriving photons, so it mixes luminosity and distance together and is not a
property of the object at all.

The fix is a convention, and it is the piece the main guide never states.
Define the **absolute magnitude** $M$ as the apparent magnitude the object
*would* have if it sat at a fixed reference distance of exactly $10$ parsecs.
That makes $M$ a restatement of $L$ in magnitude units — a property of the
object, distance divided out by fiat. Now put the two together. The flux at
distance $d$ and the flux at $10$ pc are in the ratio $(10\,\mathrm{pc}/d)^2$
by inverse square, so Pogson's definition gives

$$
m - M = -2.5\log_{10}\!\left(\frac{10\,\mathrm{pc}}{d}\right)^{\!2}
      = 5\log_{10}\!\left(\frac{d}{10\,\mathrm{pc}}\right).
$$

That quantity, $m - M$, is the **distance modulus**: one number, in magnitudes,
that *is* the distance. It is the formula
[Ch. 15](../guide/15-distances.md#three-distances) drops on you fully formed,
$M$ and all, having defined neither — and there is nothing in it beyond the
reference-distance convention above.

The first real candle was found by Henrietta Leavitt in 1912. **Cepheids** are
stars that pulsate, brightening and dimming over days to months, and Leavitt
noticed that in a population all at one distance, the slow pulsators were
uniformly the bright ones. The period tracks the luminosity — and a period is a
*clock*, immune to distance, to dust, and to everything else that corrupts a
brightness. So a Cepheid announces its own $L$, hence its own $M$, and its
measured $m$ gives $d$.

Except that the period–luminosity relation yields only a *relative* luminosity
until someone supplies the zero point, and that is where the ladder becomes a
ladder: the zero point comes from Cepheids near enough for parallax. Rung 1
calibrates rung 2. It is also where honesty is owed — that zero point depends
on the star's chemical composition and on how much dust has reddened it, both
imperfectly known, and arguments over exactly these corrections are a live part
of why the numbers at the top are still disputed.

## Supernovae { #supernovae }

Cepheids are stars, so they run out too: past a few tens of megaparsecs no
telescope resolves an individual one. The next rung has to be brighter, by a
margin no ordinary star can supply.

A **Type Ia supernova** is a white dwarf — the burnt-out core of a star like
the Sun, held up not by fusion but by electron degeneracy pressure — that
accretes matter from a companion until it crosses a critical mass and
detonates. That near-fixed detonation mass is the entire point: the fuel is
the same every time, so the explosion is nearly the same brightness every
time. At peak, a Type Ia reaches

<!-- check: pch09.snia_absolute_mag = -19.3 ± 0.01 -->

$$
M \approx -19.3,
$$

which is a single exploding star briefly outshining all
[$10^{11}$ ordinary stars](03-galaxies.md#what-a-galaxy-is) of its host
galaxy. That is what buys the reach.

"Nearly the same" is carrying weight. Type Ia peak luminosities scatter, and
the scatter is real. What rescues them is an empirical correlation —
intrinsically brighter ones fade more slowly — so the light curve's own decline
rate predicts the offset and lets you correct it out. A Type Ia is not a
standard candle so much as a *standardizable* one. And as with Cepheids the
absolute scale is not self-supplied: $M \approx -19.3$ is known only because
Cepheids were found in galaxies that also hosted Type Ia supernovae. Rung 2
calibrates rung 3.

Run it at this program's fiducial lens redshift, $z=0.5$, which
[Ch. 8](08-redshift.md#z-in-real-numbers) put in physical terms. The luminosity
distance there, in the cosmology this repository fixes by fiat, is

<!-- check: pch09.d_l_mpc_z05 = 2832.94 ± 0.01 -->

$$
D_{\mathrm{L}}(z=0.5) \approx 2833\ \mathrm{Mpc}.
$$

Convert to parsecs, take the modulus, and add it to $M$:

$$
m = M + 5\log_{10}\!\left(\frac{2.833\times10^{9}\,\mathrm{pc}}{10\,\mathrm{pc}}\right)
  = -19.3 + 42.26 = 22.96.
$$

<!-- check: pch09.snia_distance_modulus_z05 = 42.2612 ± 0.001 -->
<!-- check: pch09.snia_apparent_mag_at_z05 = 22.9612 ± 0.001 -->

The brightest standard object in astronomy, at the redshift of an ordinary lens
galaxy, arrives at magnitude $22.96$ — far below naked-eye visibility, and a
fair reminder of what the inverse-square law does over $2833$ Mpc. On this
book's ruler that distance is
<!-- check: pch09.d_l_in_mw_diameters_z05 = 92398 ± 1 -->
about $92{,}000$ Milky Way diameters, against the $1.63$ where geometry gave
out (Exercise 9.4 makes you form the ratio). The entire cost of that factor is
the phrase "we knew $M$ from somewhere else."

One caveat, not pedantic. $D_{\mathrm{L}}$ is a *luminosity* distance, not a
length you could lay a tape along: by $z=0.5$ the expansion has already made
"the distance" ambiguous — the main guide's
[Ch. 15](../guide/15-distances.md#three-distances) carries three of them,
disagreeing by factors of $(1+z)$ — and the modulus consumes exactly the one
that makes $F = L/(4\pi D_{\mathrm{L}}^2)$ true. Exercise 9.3 takes up the
second caveat: that number came *from* a cosmology.

## Errors compound { #errors-compound }

<figure markdown="span">
  ![Four horizontal bars on a logarithmic distance axis in megaparsecs — parallax, Cepheids, Type Ia supernovae, and redshift, top to bottom — each bar starting inside the span of the bar above it](figures/p09-distance-ladder-light.svg#only-light){ width="90%" }
  ![Four horizontal bars on a logarithmic distance axis in megaparsecs — parallax, Cepheids, Type Ia supernovae, and redshift, top to bottom — each bar starting inside the span of the bar above it](figures/p09-distance-ladder-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 9.1.** The four rungs and the distances
  each works over, on a logarithmic axis in megaparsecs. The ladder is drawn
  top-down — parallax, rung 1, is the *top* bar — so the panel title's "the one
  below it" means earlier in the chain, not lower on the page. There is no
  vertical axis because there is no vertical quantity; the stacking is only the
  ladder's order. What matters is that consecutive bars *overlap*: each
  method's near end reaches back into territory the rung before it already
  covers. That overlap is not redundancy; it is where the calibration
  happens.</figcaption>
</figure>

Read the figure as a dependency graph and the liability is immediate. Each
overlap is a zero-point transfer: parallax fixes the period–luminosity zero
point, Cepheids in galaxies that hosted a Type Ia fix $M \approx -19.3$, and
supernovae far enough out that expansion dominates their redshift fix the
expansion rate. Nothing at the top was ever measured against a ruler. It was
measured against something that was measured against something that was
measured against Earth's orbit.

Magnitudes at least make the propagation easy to see. $M$ enters the modulus
additively, so a zero-point error is a constant offset in magnitudes — and by
$m - M = 5\log_{10}(d/10\,\mathrm{pc})$, that offset is a constant
*multiplicative* error in every distance the rung reports. It does not average
down and it does not show in the scatter (Exercise 9.4).

!!! tip "You already know this"
    This is self-training, exactly. Rung 1 is your labelled set — trustworthy,
    and far too small to cover the domain you care about. Each higher rung is a
    model trained on pseudo-labels the rung below emitted, then deployed on a
    distribution the labelled set never reached. You know what happens:
    variance shrinks with each generation and bias does not, because a
    systematic in the teacher is not noise the student can see. Every
    diagnostic the student has is computed against inherited labels, so it
    comes back clean. Only a measurement from outside the chain catches it.

That last sentence is what [Ch. 14](14-hubble-tension.md#two-answers) is about.
A second, independent route reads the expansion rate out of the early universe
and never touches this ladder — and the two answers disagree by more than
either side's quoted uncertainties allow. The ladder has been rebuilt end to
end by people hunting for exactly the compounding error above, and it has not
been found; that failure is why the disagreement is taken seriously rather than
dismissed. The punchline for this reader is
[Ch. 14](14-hubble-tension.md#the-third-method): strong lensing gets a distance
from geometry and a light-travel-time difference, borrowing nothing from
Cepheids and nothing from supernovae — a third witness, outside both chains,
which makes this program's own field an arbiter of the argument rather than a
spectator to it.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 9 — Arcseconds, magnitudes, and the units of the sky](../guide/09-units.md#angles-on-the-sky)**
      defines the parsec as *"the distance at which one astronomical unit —
      Earth's orbital radius … — subtends an angle of exactly $1''$"*, and its
      own exercises call that *"purely a statement about the geometry of one
      right triangle."* Both true, and neither says whose triangle, why anyone
      would measure that angle, or what the unit is named after — the word
      *parallax* does not occur in the chapter. You now have the missing half:
      the angle is a parallax, the baseline is Earth's own orbit, and $d$ in
      parsecs is $1/p$ in arcseconds by construction. Its
      [magnitudes section](../guide/09-units.md#magnitudes) has the same gap: it
      defines apparent magnitude $m$ from Pogson's law in full, and stops
      there.
    - **[Ch. 15 — Three distances, and why they do not add](../guide/15-distances.md#three-distances)**
      states that *"$D_{\mathrm{L}}$ is what a Type Ia supernova's distance
      modulus, $m - M = 5\log_{10}(D_{\mathrm{L}}/10\,\mathrm{pc})$, actually
      consumes"* — the only place an absolute magnitude appears in twenty-nine
      chapters, the guide's $M$ being a mass everywhere else, inside a formula
      whose other two ingredients it likewise never explains. You now have all
      three, plus the question that formula raises and does not touch: where a
      number like $-19.3$ comes from, and why the answer is a chain of
      calibrations rather than a measurement.

## Exercises { #exercises }

??? question "Exercise 9.1 — parallax, from the nearest star to the wall"
    Proxima Centauri's parallax is $0.7687''$; Gaia's best precision is about
    $20$ millionths of an arcsecond. Give Proxima's distance in parsecs and
    light-years, then the distance at which a parallax angle equals Gaia's
    precision, and compare that to the Milky Way's $100{,}000$-light-year
    diameter. Two follow-ups: why did the parsec answer need no astronomical
    constant while the light-year answer did, and why is the Gaia reach
    optimistic?

    ??? success "Solution"
        Proxima: $d = 1/p = 1/0.7687 \approx 1.30$ pc
        <!-- check: pch09.proxima_distance_pc = 1.301 ± 0.001 -->, or $4.24$
        light-years
        <!-- check: pch09.proxima_distance_ly = 4.243 ± 0.001 -->. Gaia's
        ceiling: $d = 1/(20\times10^{-6}) = 50{,}000$ pc
        <!-- check: pch09.gaia_limit_pc = 50000 ± 1 -->, or $163{,}000$
        light-years <!-- check: pch09.gaia_limit_ly = 163078 ± 1 -->, which is
        $1.63$ Milky Way diameters
        <!-- check: pch09.gaia_reach_in_mw_diameters = 1.63 ± 0.01 -->.

        The parsec answer is free because the unit was *defined* to make it
        free: no physical quantity — not Earth's orbit, not the speed of light
        — need be known numerically. Light-years and orbital radii are
        unrelated, so bridging them takes a measured constant. The Gaia reach
        is optimistic twice over: the parallax there merely equals its own
        uncertainty, and the quoted precision is a best case, not a typical
        one. Either way, geometry alone cannot leave the Milky Way.

??? question "Exercise 9.2 — derive the distance modulus"
    Starting only from Pogson's definition,
    $m_1 - m_2 = -2.5\log_{10}(F_1/F_2)$, the inverse-square law, and the
    convention that $M$ is the apparent magnitude an object would have at
    $10$ pc, derive $m - M = 5\log_{10}(d/10\,\mathrm{pc})$. Then say what
    breaks if you pick a different reference distance.

    ??? success "Solution"
        Inverse square gives
        $F(d)/F(10\,\mathrm{pc}) = (10\,\mathrm{pc}/d)^2$. Pogson with
        $m_1 = m$, $m_2 = M$:

        $$
        m - M = -2.5\log_{10}\!\left(\frac{10\,\mathrm{pc}}{d}\right)^{\!2}
              = -5\log_{10}\!\left(\frac{10\,\mathrm{pc}}{d}\right)
              = 5\log_{10}\!\left(\frac{d}{10\,\mathrm{pc}}\right).
        $$

        The $-2.5$ times the exponent $2$ is where the $5$ comes from; the sign
        flips when the ratio is inverted. Nothing breaks with a different
        reference distance — the derivation goes through for any $d_0$, giving
        $m - M = 5\log_{10}(d/d_0)$, shifting every published $M$ by a
        constant. The $10$ pc is convention, not physics.

??? question "Exercise 9.3 — a supernova at the guide's own lens redshift"
    A Type Ia peaks at $M \approx -19.3$. At $z=0.5$ the luminosity distance
    is $D_{\mathrm{L}} \approx 2833$ Mpc. Compute the distance modulus and the
    apparent magnitude. Then explain why this calculation, unlike Exercise
    9.1, is not independent of cosmology.

    ??? success "Solution"
        In parsecs, $D_{\mathrm{L}} = 2.833\times10^{9}$ pc
        <!-- check: pch09.d_l_mpc_z05 = 2832.94 ± 0.01 -->, so
        $m - M = 5\log_{10}(2.833\times10^{9}/10) \approx 42.26$
        <!-- check: pch09.snia_distance_modulus_z05 = 42.2612 ± 0.001 -->
        and $m = -19.3 + 42.26 \approx 22.96$
        <!-- check: pch09.snia_apparent_mag_at_z05 = 22.9612 ± 0.001 -->.
        The dependence enters through $D_{\mathrm{L}}$: a redshift is not a
        distance, and turning $z=0.5$ into $2833$ Mpc requires an assumed
        expansion history — here the guide's fixed
        $H_0 = 70$, $\Omega_{\mathrm{m}} = 0.3$ model. Exercise 9.1 assumed
        nothing beyond one triangle; this one assumes a cosmology. Which is why
        supernovae get used in the *other* direction in practice: measure $m$,
        take $M$ from the ladder below, and let $D_{\mathrm{L}}$ against $z$
        *determine* the expansion history rather than presuppose it.

??? question "Exercise 9.4 — what the hand-off buys, and what it costs"
    Parallax pushed to Gaia's precision reaches $1.63$ Milky Way diameters; one
    Type Ia at $z=0.5$ sits at $92{,}000$. Form the ratio, then state exactly
    what had to be assumed to buy that factor — and why a zero-point error in
    that assumption would survive any amount of extra data.

    ??? success "Solution"
        The ratio is
        <!-- check: pch09.candle_vs_parallax_reach_ratio = 56659 ± 1 -->
        about $57{,}000$
        <!-- check: pch09.gaia_reach_in_mw_diameters = 1.63 ± 0.01 -->
        <!-- check: pch09.d_l_in_mw_diameters_z05 = 92398 ± 1 -->,
        bought with exactly one assumption: that $M$ was already known, from a
        chain of calibrations terminating at Earth's orbit. An error in that
        $M$ is an offset, not noise — it moves every distance the rung reports
        by the same factor, so averaging cannot touch it and the residuals look
        exactly as they should.
