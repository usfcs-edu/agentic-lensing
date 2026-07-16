# 10. The expanding universe

[Chapter 8](08-redshift.md#the-cosmological-picture) left you holding a claim
without a mechanism: a distant galaxy's spectrum comes back with every line
shifted toward longer wavelengths, the shift grows with distance, and the reason
is *not* that the galaxy is flying away from us through space. This chapter
supplies the mechanism. It is one formula — $v = H_0 d$ — and one picture, and
the picture does more work than the formula. By the end you should be able to say
why a law under which everything recedes from us is not a claim that we sit
anywhere special, what is expanding given that it is not the galaxies, and why no
number this law returns ever breaks a speed limit.
[Chapter 13 of the main guide](../guide/13-expansion.md#scale-factor) writes all
of it down as a function $a(t)$ and extracts Hubble's law from a one-line Taylor
expansion. That derivation means nothing without this chapter.

!!! abstract "What you can skip"
    No general relativity, no metric, no Friedmann equation — [Ch. 14 of the main
    guide](../guide/14-frw.md#friedmann) does all three and nothing here needs
    any of them. Skip the fitting machinery too: Figure 10.1 is a straight line
    through scattered points. What is worth your attention is what the slope
    *means*, why the scatter is there and is not instrument noise, and the line
    of algebra in "There is no centre", which is the whole argument.

## Hubble's law { #hubbles-law }

Measure a galaxy's distance by one of the rungs in
[Ch. 9](09-distance-ladder.md#standard-candles), measure its redshift off its
spectrum, read that redshift as a speed for the moment, and plot one against the
other. The points fall on a line through the origin:

$$
v = H_0\, d.
$$

That is **Hubble's law** — named for Edwin Hubble, who measured it, though the
IAU now calls it the Hubble–Lemaître law, since Georges Lemaître wrote it down
first. Everywhere this program touches cosmology it fixes the slope at

<!-- check: pch10.H0 = 70 ± 0.1 -->

$$
H_0 = 70\ \text{km/s per Mpc},
$$

megaparsecs being millions of parsecs, the unit
[Ch. 1](01-scale-ladder.md#the-light-year) built. Where that value comes from,
and the fact that the two best ways of measuring it disagree by more than
either one's error bars, is [Ch. 14](14-hubble-tension.md#two-answers). Take it
as given here.

Read the units before anything else, because they are the tell. Velocity over
distance is $1/\text{time}$. $H_0$ is **a rate, not a speed and not a length** —
the fractional amount by which a distance grows per unit time. It has no
direction and is attached to no object. That single observation is what the rest
of this chapter unpacks. Three readings of it, all arithmetic:

- A galaxy at $100$ Mpc recedes at $70 \times 100 = 7000$ km/s.
  <!-- check: pch10.v_at_100mpc = 7000 ± 1 -->
- **Hubble time**, $1/H_0 = 13.97$ Gyr — invert a rate, get a time.
  <!-- check: pch10.hubble_time_gyr = 13.97 ± 0.01 -->
- **Hubble distance**, $c/H_0 = 4283$ Mpc — the km/s cancels exactly, leaving
  a pure length: the characteristic scale of the observable universe, the top
  rung of Ch. 1's ladder, which
  [Ch. 1](01-scale-ladder.md#the-whole-thing) converts into Milky Ways laid end
  to end so that it means something.
  <!-- check: pch10.hubble_distance_mpc = 4283 ± 1 -->

The Hubble time is the most over-read number in introductory cosmology. If the
expansion had always run at today's rate, everything would have been on top of
everything else exactly $1/H_0$ ago, and $13.97$ Gyr would be the age of the
universe. It lands within a few percent of the real answer, which is what makes
it tempting. It is **not** the age. The rate has not been constant: for most of
cosmic history gravity pulled the expansion back, and more recently something has
been pushing it faster ([Ch. 13](13-dark-energy.md#the-supernova-surprise)). A
rate that changes cannot be inverted into an age by dividing once.
[Ch. 11](11-big-bang.md#running-it-backwards) does the honest version and gets a
number close to $13.97$ but not equal to it; the gap is physics, not rounding.

<!-- check: pch10.scatter_kms = 900 ± 1 -->

Real galaxies do not sit exactly on the line, and the reason is not measurement
error. Each also carries a **peculiar velocity** — ordinary motion through space,
the accumulated result of gravitational tugs from its neighbours — of order $900$
km/s. That wobble is the same size no matter how far away the galaxy is, so its
*fractional* importance collapses as $d$ grows. Put the two side by side: $900$
km/s of peculiar motion against $7000$ km/s of Hubble flow at $100$ Mpc. Nearby,
the wobble wins outright.

<figure markdown="span">
  ![Forty mock galaxies scattered around the line v = H0 d, with recession speed rising linearly with distance out to 400 Mpc](figures/p10-expansion-light.svg#only-light){ width="90%" }
  ![Forty mock galaxies scattered around the line v = H0 d, with recession speed rising linearly with distance out to 400 Mpc](figures/p10-expansion-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 10.1.** Forty mock galaxies: distances
  drawn uniformly across the plotted range, speeds computed as $H_0 d$ and then
  perturbed by a Gaussian peculiar motion of $900$ km/s. A simulation, not a
  measurement — drawn *from* the law, not evidence *for* it. Its job is the
  scatter, which is the same size everywhere: invisible at the right-hand edge,
  larger than the signal itself at the left, where two points land below zero
  because those galaxies are approaching. The title is the claim this chapter has to
  earn — every galaxy plotted here would draw the same plot, with itself at the
  origin.</figcaption>
</figure>

The left edge is not a defect of the figure. It is why you cannot measure $H_0$
from your own neighbourhood, and why the distance ladder has to reach as far as
it does. Andromeda, the nearest large galaxy to the Milky Way, is *blueshifted* —
falling toward us, because at that separation the Local Group's own gravity beats
the expansion outright. If you believed the expansion were everything flying
apart, that one fact would refute you.

## The raisin bread { #the-raisin-bread }

Here is the standard analogy, which is good as far as it goes. Put bread dough
with raisins scattered through it in the oven. The dough rises: every part of it
swells by the same fractional amount, all at once. Sit on any raisin and watch.
Every other raisin moves away from you, and a raisin twice as far away moves away
twice as fast — because there is twice as much dough between it and you, and
every bit of that dough is swelling at the same rate. That is Hubble's law, and
the loaf has derived the *linearity* of it, the feature that matters most, from
something you can hold in your hands.

Two things the analogy gets right, and they are the two worth keeping:

- **No raisin moves through the dough.** Each sits in the same place relative to
  the dough around it, start to finish. Separations grow because the dough
  between them grows. There is no propulsion anywhere in this story.
- **The raisins do not expand.** They are held together by something stronger
  than the dough's swelling — which maps exactly onto the real thing: atoms, the
  Solar System, the Milky Way, the Local Group are all bound by gravity or by
  forces stronger still, and none of them grows.

Now the part every popular account leaves out. **The analogy breaks, and it
breaks at the centre.**

A loaf has an edge, and it has a middle. The raisins near the crust really are
moving outward through the oven's air, away from a particular raisin near the
middle — and a raisin could work out where that middle is by looking around,
because raisins near the edge see dough on one side and empty oven on the other.
That asymmetry is a compass. The loaf has a preferred point and a boundary, and
its expansion is an expansion *into* something: the oven, already there.

The universe has none of that. No edge to find, no middle to be displaced from,
no oven to grow into. The observable universe does have a horizon — a limit set
by how far light has had time to reach us since the beginning — but that is a
fact about *our vantage point*, not a wall, and every observer has their own.
Do not read the Hubble distance as that horizon: it is a characteristic scale
for the observable region, not its radius, because light has been crossing
*expanding* space the whole way and whatever emitted it now sits several times
$c/H_0$ from us ([Ch. 1](01-scale-ladder.md#the-whole-thing) draws the same
line). Whether the universe as a whole is finite or infinite is
a question astronomy genuinely cannot answer: the measurements are consistent
with infinite, and equally consistent with finite but far larger than anything
we can see. The loaf, by construction, has an answer. That is where you put the
bread down.

## There is no centre { #there-is-no-centre }

If everything recedes from us, and the far things recede faster, why are we not
at the centre? The answer is not philosophical and it is not an appeal to
modesty. It is one line of algebra, and it works because the law is linear.

Put yourself at the origin. A galaxy at position $\mathbf{d}$ recedes at
$\mathbf{v} = H_0\mathbf{d}$ — the law in vector form, $H_0$ acting as a scalar
on the position. Now stand on a galaxy at $\mathbf{b}$, itself receding from you
at $H_0\mathbf{b}$. What does an astronomer there measure looking at the first
galaxy? Subtract:

$$
\mathbf{v}' = H_0\mathbf{d} - H_0\mathbf{b} = H_0(\mathbf{d} - \mathbf{b})
= H_0\,\mathbf{d}',
$$

where $\mathbf{d}'$ is the first galaxy's position *as measured from
$\mathbf{b}$*. Same law, same $H_0$, no trace of the origin you started from.
Every observer recovers the identical constant of proportionality, so every one
of them is equally entitled, and equally unentitled, to call themselves the
centre: measuring Hubble's law from Earth tells you nothing about where Earth
is. That is what Figure 10.1's title asserts, and the algebra above is the
whole proof.

!!! tip "You already know this"
    The argument turns entirely on the law being **linear** — $H_0$ acting as a
    scalar on the position, with no other dependence on $\mathbf{d}$. Linearity
    is what makes the subtraction close: the same function, applied to the new
    separation, comes back. Try anything else — a law where speed went as
    distance *squared*, say — and it does not close: the law an observer at
    $\mathbf{b}$ writes down has a form that depends on $\mathbf{b}$, so the
    point where it takes its simplest form is a centre everybody can find and
    agree on. "The universe has no centre" is a claim about function
    composition, not about modesty. It is the translation-equivariance argument
    you already make when you reach for a convolution over a dense layer.

Two caveats, because this uniformity is an assumption doing real work. It holds
only *on average and on large scales* — close up the universe is emphatically
lumpy, which is what galaxies, clusters and the cosmic web
([Ch. 4](04-clusters.md#the-cosmic-web)) *are*, and Figure 10.1's $900$ km/s of
peculiar motion is that lumpiness talking. And that it holds at all is measured,
not assumed: the cosmic microwave background ([Ch. 11](11-big-bang.md#the-cmb))
is the same temperature in every direction to a few parts in a hundred thousand.

## What is actually expanding { #what-is-actually-expanding }

Strip the bread away and here is what survives. **Nothing is moving through
space.** What grows is the distance between things that are not moving: the ruler
itself is getting longer, everywhere, at the same fractional rate, at once.
Galaxies keep their addresses; the map's scale bar changes underneath them. This
is why $H_0$ came out of the units as a rate — it describes nobody's motion, it
describes how fast a distance inflates.

The main guide names this in one line. Give every galaxy a fixed **comoving**
coordinate that never changes, and let a single global number $a(t)$ — the
**scale factor** — convert comoving coordinates into physical distances at time
$t$. All the time evolution lives in that one multiplier, and Hubble's law is
what you get by differentiating it once. That is
[Ch. 13 of the main guide](../guide/13-expansion.md#scale-factor); it takes half
a page, and you now know what it is a picture *of*. Two consequences follow, and
the "everything is flying apart" picture gets both wrong.

**Bound things do not expand.** The raisins were right. Your desk, the Earth's
orbit, the Milky Way's hundred billion stars, the Local Group — none is growing,
because each is held together by something that beats a fractional stretch of
$H_0$ per unit time. Expansion is not a force things resist; it is what distances
between unheld objects do.

**There is no speed limit on it.** This is the trap
[Ch. 8](08-redshift.md#the-doppler-picture) flagged, and the main guide works it
on the same number, so work it once. This program's Carousel cluster has a
background reference plane at redshift $z = 1.432$. Read that as a velocity the
naive way — $v = cz$, the Doppler reading — and you get

<!-- check: pch10.naive_v_at_z1432 = 429303 ± 50 -->

$$
v = c \times 1.432 = 429{,}303\ \text{km/s},
$$

which is $1.432$ times the speed of light.
<!-- check: pch10.naive_v_over_c = 1.432 ± 0.001 -->
Nothing is wrong with the observation and nothing is wrong with relativity. What
is wrong is the word *velocity*. Relativity bounds how fast anything moves
**through** space; it says nothing about how fast a purely geometric distance can
grow while nothing moves through the space between. $v = cz$ was only ever the
low-$z$ approximation — the first Taylor term
([Ch. 2 of the main guide](../guide/02-derivatives.md#taylor)) — and pushing it
to $z > 1$ breaks the approximation, not the physics. The always-true statement
is not a velocity at all: $a = 1/(1+z)$, the universe was that fraction of its
present size when the light left.

The same trap sits inside Hubble's law itself, and the Hubble distance is where
it surfaces. $c/H_0 = 4283$ Mpc is, by construction, the distance at which the
linear law's own prediction $H_0 d$ reaches $c$; push $d$ past it and $v = H_0 d$
hands you a recession faster than light. Not a wall, not an edge, not a bug:
recession is not motion, so there is nothing there to forbid, and we do observe
galaxies whose recession velocities exceed $c$ on any consistent accounting. A
number this law returns is geometry, never somebody's speedometer.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 13 — The expanding universe and redshift](../guide/13-expansion.md#scale-factor)**
      defines the scale factor $a(t)$ as a "dimensionless function of time" that
      "carries no units and no meaning in isolation." It then lays "a fixed grid
      over the universe," puts galaxies at fixed comoving coordinates, and tells
      you "nothing on this grid moves. What changes is the ruler." That is this
      chapter's loaf with the dough thrown away and the honest parts kept: you
      know what the grid is a picture of, and why the raisins stay put.
    - The same chapter [derives Hubble's law](../guide/13-expansion.md#hubbles-law)
      as $a(t)$'s first Taylor term, then insists it "is not an empirical fit
      stapled onto the expansion after the fact." That sentence is a rebuttal,
      and you have not until now been shown what it rebuts: galaxies flying apart
      through space with a line fitted to them. It also quotes $1/H_0 = 13.97$
      Gyr as "roughly the age of the universe, though not exactly" and defers the
      *why* to [Ch. 14](../guide/14-frw.md#friedmann) — you have it already.
    - **[Ch. 13's naive-$cz$ trap](../guide/13-expansion.md#redshift-is-expansion)**
      runs $z = 1.432$ to $429{,}303$ km/s, calls it "faster than light," and
      compresses the resolution into one clause: "no object is moving *through*
      space at all, space itself is stretching." That clause is a chapter's worth
      of picture, and this is the chapter it is standing on.

## Exercises { #exercises }

??? question "Exercise 10.1 — Hubble's law, both directions"
    Using $H_0 = 70$ km/s/Mpc, compute the recession speed of a galaxy at
    $d = 100$ Mpc. Then, without computing anything new, say what distance the
    same law assigns a recession speed of exactly $c$, and name it.

    ??? success "Solution"
        $v = H_0 d = 70 \times 100 = 7000$ km/s.
        <!-- check: pch10.v_at_100mpc = 7000 ± 1 -->
        Setting $H_0 d = c$ gives $d = c/H_0$: the Hubble distance, $4283$ Mpc
        <!-- check: pch10.hubble_distance_mpc = 4283 ± 1 -->, which is *defined*
        as that distance and so needs no new arithmetic. Nothing physical happens
        there — recession is not motion.

??? question "Exercise 10.2 — The centre test"
    An astronomer on a galaxy at position $\mathbf{b}$ (as measured by you)
    observes a third galaxy at $\mathbf{d}$ (also as measured by you). Write down
    the recession velocity they measure, and say what they conclude about their
    own location. Then repeat for a hypothetical universe obeying
    $\mathbf{v} = k|\mathbf{d}|\mathbf{d}$ — speed going as distance *squared* —
    and say what changes.

    ??? success "Solution"
        They measure $H_0(\mathbf{d} - \mathbf{b})$ against a relative position of
        $\mathbf{d} - \mathbf{b}$, i.e. $\mathbf{v}' = H_0\mathbf{d}'$: identical
        law, identical constant. They conclude exactly what you did, with exactly
        as much evidence for being the centre as you have.

        Under the squared law the subtraction leaves
        $k(|\mathbf{d}|\mathbf{d} - |\mathbf{b}|\mathbf{b})$, which is *not*
        $k|\mathbf{d}'|\mathbf{d}'$ for any $k$: the field measured at
        $\mathbf{b}$ has a form that depends on $\mathbf{b}$, so comparing enough
        galaxies lets that observer solve for $\mathbf{b}$ and locate the origin.
        That universe has a centre and everyone in it can point at it. Linearity
        is the whole difference. What the argument does *not* say: adding the
        same constant velocity to everything is not a centre, it is your own
        motion — real, invisible in relative velocities, and measured instead as
        a direction-dependent warming and cooling of the microwave background
        ([Ch. 11](11-big-bang.md#the-cmb)).

??? question "Exercise 10.3 — The naive-$cz$ trap"
    The Carousel cluster's background reference plane sits at $z = 1.432$.
    Compute the velocity the naive Doppler reading $v = cz$ assigns it, express
    it as a multiple of $c$, and state precisely where the error is: in the
    measurement, in relativity, or somewhere else.

    ??? success "Solution"
        $v = cz$ at $z = 1.432$ gives $429{,}303$ km/s
        <!-- check: pch10.naive_v_at_z1432 = 429303 ± 50 -->,
        i.e. $1.432\,c$
        <!-- check: pch10.naive_v_over_c = 1.432 ± 0.001 -->
        — predictable with no arithmetic at all, since $cz$ exceeds $c$ for any
        $z > 1$.

        The measurement is fine: $z$ is a wavelength ratio read off a spectrum.
        Relativity is fine: it bounds motion *through* space, and nothing here
        moves through space. The error is the interpretive step $v = cz$, the
        first-order approximation to the exact $a = 1/(1+z)$ and valid only while
        $z$ is small. Run any first-order approximation past its regime and the
        absurd number it returns is the approximation's fault. The always-true
        reading: the universe was $1/(1+z)$ of its present size when the light
        left — a size against a size, not a speed.

??? question "Exercise 10.4 — Why you cannot do this from your own street"
    Peculiar velocities run to about $900$ km/s and do not grow with distance.
    Explain why that makes nearby galaxies useless for measuring $H_0$, why two
    of the leftmost points in Figure 10.1 have *negative* recession speeds, and
    what it implies about the effort a real $H_0$ measurement takes.

    ??? success "Solution"
        A measured velocity is $H_0 d$ plus a peculiar term of order $900$ km/s
        <!-- check: pch10.scatter_kms = 900 ± 1 -->, the same size at every
        distance. The signal grows linearly with $d$; the contamination does not.
        So the fractional error on one galaxy's implied $H_0$ goes as $900$ km/s
        over $H_0 d$, which blows up as $d \to 0$. Close enough in, the peculiar
        term exceeds $H_0 d$ outright and flips the sign: a galaxy whose peculiar
        motion happens to point at you reads as approaching. Those are the two
        sub-zero points in Figure 10.1, and it is what Andromeda actually does.

        Hence [Ch. 9](09-distance-ladder.md#errors-compound). To measure $H_0$ you
        must reach out to where $H_0 d \gg 900$ km/s — compare $7000$ km/s at
        $100$ Mpc <!-- check: pch10.v_at_100mpc = 7000 ± 1 --> — which requires
        *independent* distances to galaxies that far out, which requires standard
        candles, which requires a ladder, which compounds calibration error at
        every rung. The wobble is why $H_0$ is hard, and hard is why
        [Ch. 14](14-hubble-tension.md#two-answers) has two answers.
