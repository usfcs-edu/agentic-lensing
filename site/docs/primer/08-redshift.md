# 8. Redshift — what it is and why it means distance

This is the chapter the book was built around. Every claim the main guide makes
about a real system — which galaxy is in front, which is behind, how much mass
sits inside a ring — rests on one measured quantity, $z$, and the guide never
says what it means. Its Ch. 12 defines $z$ and immediately withdraws the
definition ("a definition, not yet a physical claim"); its Ch. 13 says what $z$
is *not* (a velocity); its Ch. 15 hands you an integral. Nobody joins them up.
This chapter does — including the sentence nothing in the corpus says out loud,
why a bigger $z$ means farther away — and closes by locating the guide's
standard test pair, $z = 0.5$ and $z = 2.0$.

!!! abstract "What you can skip"
    You do not need the derivation of $1+z = 1/a$ from the FRW metric; it is
    one line in [Ch. 13](../guide/13-expansion.md#redshift-is-expansion) of the
    main guide, it needs a metric to write down, and this chapter states the
    result instead. You do not need the integral that turns $z$ into a distance
    either — that is [Ch. 15](../guide/15-distances.md#three-distances), and
    `astropy` evaluates it numerically because there is no closed form. What you
    *do* need is one sentence each from
    [Ch. 5](05-light.md#looking-back-in-time) (light takes time) and
    [Ch. 7](07-spectra.md#why-elements-have-fingerprints) (a line's rest
    wavelength is a constant of the transition, not a property of the object).

## What redshift measures { #what-redshift-measures }

[Chapter 7](07-spectra.md#why-elements-have-fingerprints) established the fact
that makes this measurement possible: the wavelength at which a given atomic
transition emits or absorbs is fixed by quantum mechanics and nothing else.
Every hydrogen atom anywhere emits Balmer-alpha at the same wavelength,
whatever galaxy it sits in. So when you find that entire fixed pattern
displaced in a real spectrum — every line moved by the *same* multiplicative
factor — the displacement is a fact about the light's journey, not about the
atom.

Define that factor. With $\lambda_{\mathrm{emit}}$ the known rest wavelength
and $\lambda_{\mathrm{obs}}$ where you actually find the line:

$$
z \;\equiv\; \frac{\Delta\lambda}{\lambda_{\mathrm{emit}}}
\;=\; \frac{\lambda_{\mathrm{obs}} - \lambda_{\mathrm{emit}}}{\lambda_{\mathrm{emit}}},
\qquad\text{equivalently}\qquad
1 + z \;=\; \frac{\lambda_{\mathrm{obs}}}{\lambda_{\mathrm{emit}}}.
$$

$z$ is dimensionless — a wavelength over a wavelength. It is not a distance,
not a speed, and not, yet, anything about the universe. It is a number you read
off a plot.

The arithmetic is as light as it looks. A redshift of $z = 0.5$ means every
wavelength arrives
50%<!-- check: pch08.z05_stretch_percent = 50 ± 0.001 --> longer than it left;
$z = 2.0$ means
200%<!-- check: pch08.z20_stretch_percent = 200 ± 0.001 --> longer, three times
the emitted wavelength, since $1+z = 3$. Take the line
[Ch. 7](07-spectra.md#why-elements-have-fingerprints) derived from the Rydberg
formula, Balmer-alpha at
656.47 nm<!-- check: pch08.halpha_rest_nm = 656.47 ± 0.001 -->, and put it in a
galaxy at $z = 2.0$:

<!-- check: pch08.halpha_at_z2_nm = 1969.41 ± 0.01 -->

$$
\lambda_{\mathrm{obs}} = 656.47\ \mathrm{nm} \times 3 = 1969.41\ \mathrm{nm}.
$$

That line left its galaxy at the red edge of the visible band
([Ch. 5](05-light.md#the-spectrum)) and arrives at nearly two micrometres —
near-infrared, invisible to the eye and off the red end of a ground-based
optical spectrograph. [Ch. 12](../guide/12-spectroscopy.md#measuring-redshift)
of the main guide makes the identical observation about its own source, and it
is why a redshift pipeline cross-correlates a whole template library rather
than hunting for one hard-coded line: which lines are even *observable* depends
on the $z$ you are trying to measure.

Two naming traps, cleared now. "Redshift" does not mean the light turns red; it
means every wavelength gets longer. A galaxy's ultraviolet arrives as visible
blue, its visible red as infrared — "red" names the *direction* of the shift,
not the colour of what lands in the detector. And the $z$ in "$z$-band"
([Ch. 6](06-telescopes.md#bands-and-filters)) is an unrelated use of the same
letter.

Everything above is bookkeeping. It says nothing about how far away the galaxy
is — which is exactly why the main guide's Ch. 12 defines $z$ and then declines
to interpret it. The rest of this chapter is the physical claim it deferred.

## The Doppler picture { #the-doppler-picture }

Everyone's first explanation of redshift is the ambulance siren, and it is
worth having provided you know how far it goes. A source moving away emits each
successive wave crest from slightly farther off, so the crests arrive spaced
wider than they were emitted: longer wavelength, lower pitch. For a source
receding at $v \ll c$, the stretch is proportional to the speed:

$$
\frac{\Delta\lambda}{\lambda} = \frac{v}{c},
\qquad\text{so}\qquad
z \approx \frac{v}{c},
\qquad
v \approx cz.
$$

This is a real effect, and this program measures it. The stars inside a lens
galaxy orbit in every direction at once, so each one's light picks up its own
small Doppler shift and the summed light of the population has its lines
*smeared* rather than displaced. That smear's width is the velocity dispersion
$\sigma_v$ of [Ch. 12](../guide/12-spectroscopy.md#sigma-v-from-lines) — a
genuine Doppler measurement, at speeds three orders of magnitude below $c$
where the formula above is excellent.

Now run the same formula on redshifts of the size this program catalogues,
using the exact speed of light from [Ch. 5](05-light.md#the-spectrum),
$c = 299{,}792.458$ km/s. At $z = 0.5$:

<!-- check: pch08.naive_v_z05_kms = 149896 ± 1 -->

$$
v = cz = 299{,}792.458 \times 0.5 = 149{,}896\ \mathrm{km/s}.
$$

Half the speed of light. Uncomfortable, but not yet impossible. At $z = 2.0$:

<!-- check: pch08.naive_v_z2_kms = 599585 ± 1 -->
<!-- check: pch08.naive_v_z2_over_c = 2 ± 0.001 -->

$$
v = cz = 299{,}792.458 \times 2.0 = 599{,}585\ \mathrm{km/s} = 2c.
$$

Twice the speed of light. No care with the measurement would change that
answer, because the answer is not a measurement — it is a formula reporting an
impossibility. The impossibility arrives the moment $z$ exceeds one, since
$cz > c$ exactly when $z > 1$, and this program routinely works past $z = 1$.

So say it plainly, because the main guide will contradict you if you do not
hear it here: **cosmological redshift is not a Doppler shift.** The Doppler
picture is a serviceable intuition for what a stretched wave *is*, and at small
$z$ it returns the right number, but it is the low-$z$ limit of something else,
and the something else is what is true.
[Ch. 13](../guide/13-expansion.md#redshift-is-expansion) runs this identical
trap on a real system — the Carousel cluster's reference plane at $z = 1.432$ —
gets $1.432c$, and concludes: *"The correct statement was never a velocity."*

Name where the analogy breaks, since a simplification you have to unlearn is
worse than none. The siren's shift is imprinted at emission and set by how fast
the ambulance moves *through the air* relative to you; nothing further happens
to the sound in transit. The cosmological shift is the opposite on both counts.
It is not imprinted at emission — it accumulates over the entire journey — and
no galaxy is moving through space at anything like these speeds. The galaxies
are, to a good approximation, sitting still.

## The cosmological picture { #the-cosmological-picture }

Here is the actual cause. While the light was in flight, space itself expanded,
and the wave was stretched by exactly the same factor as everything else.
[Chapter 10](10-expansion.md#the-raisin-bread) paints that expansion properly —
no centre, nothing moving *through* space, the ruler itself growing. The one
piece of bookkeeping needed here is the **scale factor** $a$: a single
dimensionless number for the size of the universe at a given moment, normalised
so that $a = 1$ today. It has no meaning in isolation; only the *ratio* of $a$
at two times means anything, and a measured redshift is exactly that ratio:

$$
1 + z \;=\; \frac{\lambda_{\mathrm{obs}}}{\lambda_{\mathrm{emit}}}
\;=\; \frac{1}{a(t_{\mathrm{emit}})},
\qquad\text{so}\qquad
a(t_{\mathrm{emit}}) \;=\; \frac{1}{1+z}.
$$

That is the entire physical content of a redshift. It reports how large the
universe was when the light left, relative to how large it is now. Not a speed
— a size then, against a size now. For the pair this chapter is chasing:

<!-- check: pch08.z05_scale_factor = 0.667 ± 0.001 -->
<!-- check: pch08.z20_scale_factor = 0.333 ± 0.001 -->

$$
a(z=0.5) = \frac{1}{1.5} = 0.667,
\qquad
a(z=2.0) = \frac{1}{3} = 0.333.
$$

The $z = 0.5$ light left when the universe was two-thirds its present size; the
$z = 2.0$ light when it was one-third. Notice already that four times the
redshift is not four times anything: $1/(1+z)$ is not linear in $z$, and those
two scale factors differ by a factor of two, not four.

**Only the endpoints matter.** The stretch accumulates continuously across a
journey of billions of years, but the total is exactly the ratio of the size
then to the size now. Whatever the expansion did in between — sped up, slowed
down, both — leaves no trace in $z$. That is why $z$ is such a clean thing to
measure, and why a survey catalogue stores $z$ and not a history. (The history
comes back later: it is what makes the conversion from $z$ to a distance an
integral rather than a multiplication.)

**And a caveat this book will not paper over.** You will find working
physicists who maintain that the cosmological redshift *can* be described as an
accumulated Doppler shift — the total stretch decomposed into a chain of
infinitesimal shifts between neighbouring observers along the photon's path. In
a strict technical sense they are not wrong; this is a live pedagogical
argument, not a settled misconception. What that description cannot do is hand
you a single recession velocity to put into $v = cz$. The expansion description
gives one unambiguous number, $a = 1/(1+z)$, at every redshift; the velocity
description needs a convention for comparing velocities at two widely separated
points in a curved spacetime, and general relativity does not supply one. That
asymmetry is why the guide states the size ratio and refuses the speed.

## Bigger z means farther { #bigger-z-means-farther }

Here is the sentence this book exists to say, and that no chapter of the main
guide ever says:

**A bigger $z$ means the universe was smaller when the light left, which means
the light left longer ago, which means the source is farther away.**

Three links, four quantities. Take the links one at a time, because exactly one
of them is doing work you might not expect.

1. **Bigger $z$ → smaller universe then.** Immediate from the previous section:
   $a = 1/(1+z)$ falls as $z$ rises. No physics beyond the definition.
2. **Smaller then → longer ago.** This link needs a fact about the universe's
   *history* rather than about light: $a$ has been increasing throughout the
   entire span this book covers. Given that, "the universe was one-third its
   present size" names exactly one moment in time. Had the universe ever
   contracted and re-expanded, a given $a$ would name two moments and the chain
   would break here. It has not, over any epoch we can see, and
   [Ch. 11](11-big-bang.md#running-it-backwards) runs $a$ back to zero.
3. **Longer ago → farther away.** Light covers $c$ per unit time and no more,
   so more time in flight is more distance covered. This is
   [Ch. 5](05-light.md#looking-back-in-time)'s point — looking far away *is*
   looking into the past — read in the other direction.

Now the numbers, for the pair the main guide uses as its standard test case and
never once locates. The light from a $z = 0.5$ lens galaxy left
5.04 Gyr<!-- check: pch08.z05_lookback_gyr = 5.04 ± 0.01 --> ago; the light from
a $z = 2.0$ source left
10.24 Gyr<!-- check: pch08.z20_lookback_gyr = 10.24 ± 0.01 --> ago. Set those
against the age of the universe, which in the cosmology this repository fixes
by fiat — `FlatLambdaCDM(H0=70, Om0=0.3)`, asserted everywhere — is
13.47 Gyr<!-- check: pch08.universe_age_gyr = 13.47 ± 0.01 -->. (That is
shorter than the age you will see quoted from CMB-derived parameters. The gap
is not a rounding error: it is the Hubble tension in disguise, and
[Ch. 14](14-hubble-tension.md#two-answers) is where it stops being a footnote.)
So the lens light left
37.4%<!-- check: pch08.z05_lookback_frac_of_age = 0.374 ± 0.001 --> of the
universe's whole life ago, and the source light
76.0%<!-- check: pch08.z20_lookback_frac_of_age = 0.760 ± 0.001 --> of it ago.
The source you are modelling is not a contemporary object photographed at
distance; it is a galaxy as it existed before three-quarters of cosmic history
had happened, and no telescope can show you what became of it.

Notice the compression: four times the redshift bought only about twice the
lookback time. It has to. Lookback time is bounded above by the age of the
universe, so as $z$ climbs the curve flattens against that ceiling and never
touches it. $z$ is not a linear odometer.

!!! tip "You already know this"
    Each arrow in that chain is a deterministic function of the one before, so
    everything downstream is a function of $z$ alone. That is why a catalogue of
    a million galaxies stores $z$ and not four columns: $z$ is the primary key,
    and scale factor, lookback time and every flavour of distance are derived
    views, computed on demand from that one number. A pipeline that reports $z$
    has already committed to all four, whether or not it says so. Note the limit
    of that framing, though: being a function of $z$ is all a derived view
    needs, and it does not make a column *monotone* in $z$. The last section of
    this chapter exhibits one that is not.

<figure markdown="span">
  ![Lookback time plotted against redshift from 0 to 3, with z=0.5 and z=2.0 marked and the age of the universe drawn as a horizontal ceiling the curve approaches](figures/p08-redshift-chain-light.svg#only-light){ width="90%" }
  ![Lookback time plotted against redshift from 0 to 3, with z=0.5 and z=2.0 marked and the age of the universe drawn as a horizontal ceiling the curve approaches](figures/p08-redshift-chain-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 8.1.** The chain on one axis.
  Horizontal: a redshift, the thing you actually measure. Vertical: how long ago
  the light left. The marked points are the main guide's standard test pair —
  $z=0.5$, a typical lens, and $z=2.0$, a typical source — each annotated with
  its lookback time and with the size the universe had when that light departed
  (67%<!-- check: pch08.z05_scale_factor = 0.667 ± 0.001 --> and
  33%<!-- check: pch08.z20_scale_factor = 0.333 ± 0.001 --> of today's). The
  dashed line is the age of the universe in this
  repository's assumed cosmology; the curve rises toward it and never reaches
  it, which is why a fourfold increase in redshift does not buy a fourfold
  increase in lookback time. Nothing here is drawn by hand: the curve is
  `astropy` integrating the expansion history, the calculation
  [Ch. 15](../guide/15-distances.md#three-distances) writes as an
  integral.</figcaption>
</figure>

## z in real numbers { #z-in-real-numbers }

One honest complication first. Once space has been expanding while the light
was in flight, "how far away" needs a qualifier, because the source has kept
receding since its light left. There is no single "the distance," and this is
not a technicality — the main guide's
[Ch. 15](../guide/15-distances.md#three-distances) derives three distances from
one metric because three different questions get three different answers. Two
of them matter here.

**Comoving distance** is how far apart you and the source are *today*, measured
on a grid that expands along with the universe. Getting it from $z$ needs the
integral this book hands off; here are its answers:

<!-- check: pch08.z05_comoving_mpc = 1888.63 ± 0.01 -->
<!-- check: pch08.z20_comoving_mpc = 5179.86 ± 0.01 -->

$$
D_{\mathrm{C}}(z=0.5) = 1888.63\ \mathrm{Mpc},
\qquad
D_{\mathrm{C}}(z=2.0) = 5179.86\ \mathrm{Mpc}.
$$

The second is, to the digit, the main guide's own
$D_{\mathrm{C}}(2.0) = 5179.862$ Mpc<!-- check: pch08.z20_comoving_mpc = 5179.862 ± 0.01 -->
— arrived at there by an integral.

A megaparsec is an abstraction; the Milky Way is not. Lay Milky Ways edge to
edge and count:

<!-- check: pch08.z05_comoving_in_mw = 61599 ± 1 -->
<!-- check: pch08.z20_comoving_in_mw = 168945 ± 1 -->

$$
z = 0.5:\ 61{,}599\ \text{Milky Ways},
\qquad
z = 2.0:\ 168{,}945\ \text{Milky Ways}.
$$

Read those counts to one digit and no further. The distances are solid; the
ruler is not. [Ch. 1](01-scale-ladder.md#our-galaxy) was explicit that the
stellar disk has no edge — it fades — so "$10^5$ light-years across" is a
convention carrying about a factor of two of slack. The trailing digits are the
division's, not the universe's; what survives is the exponent, and the exponent
is the point. It is still the picture the main guide never provides for the two
numbers it leans on hardest: the lens sits some tens of thousands of Milky Ways
away, the source a few times farther, in a direction that happens to have a
galaxy in the way. None of that separation is visible in the data — both land
in the same handful of pixels.

**Angular-diameter distance** is the other one, and it is what every lensing
formula in the corpus actually consumes: the distance that makes ordinary
Euclidean angle arithmetic come out right, physical size divided by observed
angle. It is not the comoving distance:

<!-- check: pch08.z05_angular_mpc = 1259.08 ± 0.01 -->
<!-- check: pch08.z20_angular_mpc = 1726.62 ± 0.01 -->

$$
D_{\mathrm{A}}(z=0.5) = 1259.08\ \mathrm{Mpc},
\qquad
D_{\mathrm{A}}(z=2.0) = 1726.62\ \mathrm{Mpc}.
$$

Look at what happened to the gap. Measured comoving, the source is far more
distant than the lens; measured by angle, the two are much closer. The relation
is $D_{\mathrm{A}} = D_{\mathrm{C}}/(1+z)$, and the divisor is larger for the
more distant object, so the angular ladder is squeezed relative to the comoving
one. Physically: the source's light left when everything was closer together,
so the galaxy subtends a *larger* angle than its present-day distance would
suggest.

Push that far enough and the squeeze becomes a reversal. Angular-diameter
distance does not grow without bound with redshift — it rises, peaks, and then
declines, so a galaxy at very high $z$ can subtend a larger angle than a nearer
one. This does not damage the chain: "bigger $z$ means farther away" holds
without exception for lookback time and comoving distance, the two senses in
which anyone means it. But you do have to say which distance you mean, every
time. Ch. 15 is where that discipline gets enforced, including its sharpest
consequence — that these distances do not subtract, so the lens-to-source
distance is emphatically *not* $D_{\mathrm{s}} - D_{\mathrm{d}}$
([Ch. 15](../guide/15-distances.md#distances-do-not-add)).

That is the whole chain. A spectrum gives one dimensionless ratio; the ratio
gives the size of the universe at emission; the size gives the moment; the
moment gives the distance. Four readings, one measurement, and nothing in
between that needs a velocity.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 12 — Redshifts and what a spectrum tells you](../guide/12-spectroscopy.md#measuring-redshift)**
      opens by promising that $z$ *"tells you how far away something is"* and
      then takes the promise straight back: *"This is a definition, not yet a
      physical claim… treat $z$ as a wavelength ratio you read off a spectrum,
      full stop."* It never returns to collect. You now have the physical claim
      it deferred, and when it puts a redshifted Balmer-alpha *"deep in the
      near-infrared,"* you have already run that calculation yourself.
    - **[Ch. 13 — The expanding universe and redshift](../guide/13-expansion.md#redshift-is-expansion)**
      states the conclusion — *"Not a velocity through space — a size, then,
      compared against a size, now"* — and runs the naive-$cz$ trap on the
      Carousel cluster's reference plane at $z = 1.432$ to get $1.432\,c$. You
      met that trap here at $z = 2.0$, where it returns $2c$, so the chapter's
      punchline lands as confirmation rather than as a surprise to take on
      trust. Its Exercise 13.2 computes $a = 1/(1+z)$ for $z = 0.5$ and
      $z = 2.0$ and gets $0.667$<!-- check: pch08.z05_scale_factor = 0.667 ± 0.001 -->
      and $0.333$<!-- check: pch08.z20_scale_factor = 0.333 ± 0.001 --> — your
      pair, your numbers.
    - **[Ch. 15 — Distances that do not add](../guide/15-distances.md#three-distances)**
      derives comoving, angular-diameter and luminosity distance from the FRW
      metric and quotes $D_{\mathrm{C}}(2.0) = 5179.862$ Mpc
      <!-- check: pch08.z20_comoving_mpc = 5179.862 ± 0.01 --> as the output of
      an integral. You already know what that number *is*: of order $10^5$
      <!-- check: pch08.z20_comoving_in_mw = 168945 ± 1 --> Milky Ways, and
      light that left when the universe was a third its present size. You also
      know why the chapter needs three distances instead of one — the piece it
      assumes rather than motivates.

## Exercises { #exercises }

??? question "Exercise 8.1 — Where does the line land?"
    Balmer-alpha's rest wavelength is $656.47$ nm. Compute where it arrives
    from a galaxy at $z = 2.0$, name the part of the spectrum it lands in, and
    say why a pipeline cannot hunt for one named line.

    ??? success "Solution"
        $1 + z = 3$, so
        $\lambda_{\mathrm{obs}} = 656.47 \times 3 = 1969.41$ nm
        <!-- check: pch08.halpha_at_z2_nm = 1969.41 ± 0.01 -->
        <!-- check: pch08.halpha_rest_nm = 656.47 ± 0.001 -->
        — near-infrared, past the red edge of the visible band and outside a
        ground-based optical spectrograph's range. Which lines are observable
        depends on $z$, the very quantity being measured, so the search runs
        over a whole template pattern: at any $z$ where one line has left the
        band, another has usually entered it.

??? question "Exercise 8.2 — The naive velocity, and where it dies"
    Read $z$ as a recession speed via $v = cz$, with $c = 299{,}792.458$ km/s.
    Compute $v$ at $z = 0.5$ and $z = 2.0$. At what redshift does this reading
    first predict something impossible, and what exactly is wrong — the
    measurement, the physics, or the formula?

    ??? success "Solution"
        At $z = 0.5$: $v = 299{,}792.458 \times 0.5 = 149{,}896$ km/s
        <!-- check: pch08.naive_v_z05_kms = 149896 ± 1 -->, half the speed of
        light. At $z = 2.0$: $v = 599{,}585$ km/s
        <!-- check: pch08.naive_v_z2_kms = 599585 ± 1 -->, which is exactly
        $2c$<!-- check: pch08.naive_v_z2_over_c = 2 ± 0.001 -->. The reading
        first predicts a superluminal speed at $z > 1$, since $cz > c$ exactly
        when $z > 1$ — no computation needed. What is wrong is the formula, and
        specifically the domain it is used on: $v \approx cz$ is only ever the
        small-$v$ approximation to something else. Nothing is measured wrongly
        and no physics is violated, because no galaxy is moving through space at
        these speeds at all. The correct statement is $a = 1/(1+z)$: exact at
        every $z$, and not a speed.

??? question "Exercise 8.3 — Close the chain yourself"
    For $z = 0.5$ and $z = 2.0$: compute the scale factor at emission, state in
    one sentence what each one says about the universe, and then read the
    lookback time for each off Figure 8.1. Finally, explain why quadrupling the
    redshift did not quadruple the lookback time.

    ??? success "Solution"
        $a = 1/(1+z)$ gives $1/1.5 = 0.667$
        <!-- check: pch08.z05_scale_factor = 0.667 ± 0.001 --> and $1/3 = 0.333$
        <!-- check: pch08.z20_scale_factor = 0.333 ± 0.001 -->: the $z=0.5$
        light left when the universe was two-thirds its present size, the
        $z=2.0$ light when it was one-third. The lookback times are
        5.04 Gyr<!-- check: pch08.z05_lookback_gyr = 5.04 ± 0.01 --> and
        10.24 Gyr<!-- check: pch08.z20_lookback_gyr = 10.24 ± 0.01 --> — that
        is 37.4%<!-- check: pch08.z05_lookback_frac_of_age = 0.374 ± 0.001 -->
        and 76.0%<!-- check: pch08.z20_lookback_frac_of_age = 0.760 ± 0.001 -->
        of the universe's whole
        13.47 Gyr<!-- check: pch08.universe_age_gyr = 13.47 ± 0.01 --> life ago.
        Quadrupling $z$ roughly doubled the lookback time because lookback time
        is bounded above by the age of the universe: the curve has a ceiling it
        approaches and cannot cross, so equal steps in $z$ buy ever-smaller
        steps in time. $a = 1/(1+z)$ is not linear, and neither is anything
        downstream of it.

??? question "Exercise 8.4 — Two distances, one source"
    For the $z = 2.0$ source, the comoving distance is $5179.86$ Mpc and the
    angular-diameter distance is $1726.62$ Mpc. Compute their ratio and
    identify it. Then explain, without any integral, why the angular-diameter
    distance is the smaller of the two.

    ??? success "Solution"
        $5179.86 / 1726.62 = 3.000$
        <!-- check: pch08.z20_comoving_mpc = 5179.86 ± 0.01 -->
        <!-- check: pch08.z20_angular_mpc = 1726.62 ± 0.01 -->, which is exactly
        $1 + z$: the relation is $D_{\mathrm{A}} = D_{\mathrm{C}}/(1+z)$, and
        nothing else. $D_{\mathrm{A}}$ is the smaller one for the same reason
        $z$ exists at all. The light left when the universe was a third its
        present size, so the source was physically much closer to us *then*,
        and the angle it subtends was set at that moment, not this one. An
        angular-diameter distance answers "what distance makes the observed
        angle come out right," and that answer is anchored to the geometry at
        emission.
