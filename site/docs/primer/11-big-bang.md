# 11. The Big Bang and the oldest light

[Chapter 10](10-expansion.md#hubbles-law) left the universe expanding: every
distant galaxy receding, the ruler itself getting longer, nothing moving through
space. This chapter runs that film backwards — the universe was once hot, dense
and opaque; it has a finite age you can quote to three digits; and the moment it
stopped being opaque left behind a photograph that is still arriving. It also
names what the main guide uses without ever naming: the state $a = 0$, which the
main guide's [Ch. 14](../guide/14-frw.md#friedmann) integrates back to in a
single clause and never calls anything. Two numbers below disagree with the
ones you will have seen in a magazine, and both disagreements are the point
rather than an embarrassment.

!!! abstract "What you can skip"
    You do not need general relativity, nucleosynthesis, inflation, or a
    Boltzmann code, and you do not need the integral turning
    $\Omega_m,\Omega_\Lambda,H_0$ into an age — that is
    [Ch. 14](../guide/14-frw.md#friedmann)'s job and it does it properly. What
    you already own, and should lean on hard here, is the discipline of knowing
    where a model's domain of validity ends. Every claim below is a fitted model
    extrapolated backwards past the data that constrained it, and the live
    question is how far back it is still entitled to speak.

## Running the expansion backwards { #running-it-backwards }

Reverse the sign of time in [Ch. 10](10-expansion.md#what-is-actually-expanding)'s
picture. If every comoving separation is growing today, each was smaller
yesterday and smaller still the day before. The scale factor $a$ — the single
global multiplier [Ch. 13](../guide/13-expansion.md#scale-factor) will define,
normalised to $a = 1$ today — shrinks as you go back. Fixed matter in a
shrinking volume means the density climbs, and compressed gas heats, so the
temperature climbs with it.

Push that far enough and you reach a regime with no stars, no atoms, eventually
no stable anything: hot, dense, uniform. That is what "the Big Bang" names, and
be blunt about what it does *not* name. It was not an explosion: an explosion
happens at a place and throws debris into surrounding space, and there is
neither here. The objection that killed the centre of the raisin bread in
[Ch. 10](10-expansion.md#there-is-no-centre) kills this too — the expansion has
no centre, so its reversal has none either. The hot dense state was everywhere,
including at the coordinates you are sitting at. What "backwards" contracts is
the ruler, not the contents toward a point.

**The age.** Run the model back until $a$ reaches zero and the elapsed time is
finite. Feed this book's cosmology — `FlatLambdaCDM(H0=70, Om0=0.3)`, the line
every campaign in this repository asserts — into `astropy`, ask for `.age(0)`,
and you get

<!-- check: pch11.universe_age_gyr = 13.467 ± 0.01 -->

$$
t_0 = 13.467\ \text{Gyr},
$$

Against this book's ruler that is long without being incomprehensible: light
takes about $10^5$ years to cross the Milky Way
([Ch. 1](01-scale-ladder.md#our-galaxy)), so the universe has existed for on the
order of a hundred thousand such crossings.

**Why this is not 13.8.** The number in every popular account is

<!-- check: pch11.planck_age_gyr = 13.8 ± 0.01 -->

$13.8$ Gyr, and this book is going to keep saying $13.467$. The gap is

<!-- check: pch11.age_gap_percent = 2.47 ± 0.01 -->

$2.47$ percent, which is not a rounding error. $13.8$ Gyr comes from Planck's
cosmology, whose $H_0$ is lower than this repository's $70$ and whose
$\Omega_m$ is very slightly higher. Those push the age in opposite directions
and $H_0$ wins: a lower $H_0$ is a slower present expansion rate, and a universe
expanding more slowly needs *longer* to have reached its current size. That is
why this book will not print $13.8$ beside a cosmology that does not produce it.
The disagreement between those two values of $H_0$ is not a choice of convention
— it is the open problem this book's [Ch. 14](14-hubble-tension.md#two-answers)
is about, showing up here in the currency of an age.

**What $a = 0$ is, and is not.** The model says $a \to 0$ at a finite time in
the past, with the density diverging. A diverging density does not describe an
event; it is the standard signature of a theory evaluated outside its domain.
General relativity supplies the equation and has no quantum mechanics in it, and
conditions near $a = 0$ are exactly where quantum mechanics cannot be left out.
Nobody has a tested theory of that regime: **"the Big Bang" does not name a
first instant anyone can compute.** It names the hot dense early state the
evidence reaches back to — and that evidence stops a long way short of $a = 0$,
at the moment the next section is about.

## The first light { #the-first-light }

Temperature and size are locked together: $T \propto 1/a$, so halving the size
doubles the temperature. (The reason is
[Ch. 13](../guide/13-expansion.md#redshift-is-expansion)'s, one chapter early:
expansion stretches every wavelength by the same factor $a$, and a blackbody
([Ch. 5](05-light.md#why-hot-means-blue)) with all its wavelengths stretched by
$a$ is still a blackbody, at a temperature $1/a$ times lower.)

Run *forwards* from the hot state, where the physics is easier to state. Above a
few thousand kelvin hydrogen is ionised: protons and electrons wander separately
rather than bound, and the universe is a plasma. Free electrons scatter photons
efficiently, so a photon there travels a short distance, scatters, travels,
scatters again. The universe is *opaque* — not dark but fogged, for the same
reason you cannot see into the interior of the Sun.

As it expands and cools, photons lose the energy needed to keep knocking
electrons off. At around

<!-- check: pch11.cmb_emission_temp_k = 3000 ± 0.5 -->

$3000$ K, electrons and protons bind into neutral hydrogen and stay bound, and
neutral hydrogen does not scatter these photons the way free electrons did. The
fog does not thin gradually over billions of years; it clears over a window
short compared to everything around it, and the universe goes from opaque to
transparent. Every photon bouncing around at that instant stopped, and has
travelled in a straight line ever since.

Two names to flag. This event is universally called **recombination**, which is
wrong — the electrons and protons had never been combined before, so nothing
re-combined. The name is a historical accident from laboratory plasma physics;
read it as "combination" and move on. It is also called **last scattering**,
which is accurate and more useful: the last time those photons interacted with
anything.

$3000$ K is worth pausing on with [Ch. 5](05-light.md#why-hot-means-blue) in
hand — roughly the photosphere temperature of a cool red star. The entire sky
was as hot and bright as a red dwarf's surface, with no dark between the stars
because there were no stars and no dark.

**When this happened, and an honest problem.** The measured answer is

<!-- check: pch11.cmb_age_kyr = 380 ± 1 -->

$380{,}000$ years after $a = 0$ — a few times the $10^5$ years light needs to
cross the Milky Way, and a fraction

<!-- check: pch11.cmb_age_fraction = 2.82e-5 ± 0.01e-5 -->

$2.8\times10^{-5}$ of elapsed history, under three parts in a hundred thousand.
On any clock this book has used, the CMB is the universe's first instant.

That $380$ is **measured, not computed here**, and the distinction matters.
`FlatLambdaCDM(70, 0.3)` models matter and dark energy and *no radiation at
all* — its `Tcmb0` is zero, so the object this book quotes ages from does not
know the CMB exists. Ask it for its own age at last scattering and it answers

<!-- check: pch11.cmb_age_this_model_kyr = 465 ± 1 -->

$465{,}000$ years, not $380{,}000$. The error is neither small nor mysterious:
the real early universe was radiation-dominated, radiation drives a faster
expansion than matter alone, and a model without radiation therefore takes too
long to reach any given early size. The model is fine for everything this
repository asks of it — all of that sits at $z$ below a few, where radiation is
irrelevant — and is outside its domain at $z = 1100$. So this book quotes $380$
from measurement, prints the model's own $465$ beside it, and does not blend the
two.

## The cosmic microwave background { #the-cmb }

Those free-streaming photons are still arriving. They left a $3000$ K
blackbody, and expansion has stretched every wavelength by the same factor the
universe has grown by since — a measured redshift:

<!-- check: pch11.cmb_redshift = 1100 ± 1 -->

$$
z_{\mathrm{CMB}} \approx 1100, \qquad
a = \frac{1}{1+z} = \frac{1}{1101} \approx 0.000908.
$$

<!-- check: pch11.cmb_scale_factor = 0.000908 ± 0.000001 -->

The universe was about one part in eleven hundred of its present size. A
$3000$ K blackbody stretched by $1101$ arrives at $3000/1101$, and measured from
here that is

<!-- check: pch11.cmb_temp_k = 2.725 ± 0.001 -->

$$
T_0 = 2.725\ \text{K}.
$$

Run it back the other way — $2.725 \times 1101$ — and you land on the $3000$ K
this section opened with. Those are not two independent claims but one
measurement read in two directions.

A blackbody at $2.725$ K peaks at a wavelength
[Ch. 5](05-light.md#why-hot-means-blue) already computed for you, well outside
the visible band and squarely in the microwave. Hence the name: the **cosmic
microwave background**. It was found by accident, as an unremovable hiss in a
radio antenna whose operators spent months hunting for the fault before
concluding the noise came from the whole sky at once.

That last part is what the usual shorthand gets wrong. The CMB is not a picture
of a distant *object*; it is a picture of a *time*. The photons arriving now
were emitted at last scattering by whatever material sat at exactly the right
distance for its light to reach us today, and those positions form a sphere
around us: the **surface of last scattering**. It is not an edge of the universe
and not a wall. Every observer has their own, and observers on ours — where the
material long ago cooled into galaxies — see a CMB from a sphere that includes
us.

Which is where "the baby picture of the universe" breaks. An infant photograph
is of a subject, taken from outside it. This one is taken from inside, of
everything, at one moment, and its subject is not a place you could go: that
material is not glowing at $3000$ K today, it is nearby and cold and made into
stars.

<figure markdown="span">
  ![The scale factor from the Big Bang to today, drawn twice from the same cosmology: on linear axes where a goes to zero, and on log axes where the CMB is visible](figures/p11-cosmic-history-light.svg#only-light){ width="100%" }
  ![The scale factor from the Big Bang to today, drawn twice from the same cosmology: on linear axes where a goes to zero, and on log axes where the CMB is visible](figures/p11-cosmic-history-dark.svg#only-dark){ width="100%" }
  <figcaption markdown="span">**Figure 11.1.** The **same curve** in both panels —
  $a(t)$ from `FlatLambdaCDM(70, 0.3)`, nothing drawn by hand — because no single
  pair of axes shows both facts. On linear axes (left) you can see $a\to0$, which
  is what "the Big Bang" names; the CMB marker is there too, crushed against the
  origin, sitting in the first few thousandths of a percent of the axis. On log
  axes (right) the CMB is legible at $a = 1/1101$ and $a = 0$ has vanished,
  because $\log 0$ does not exist. Cosmology is read on log axes for that reason,
  so a reader who has only seen the right-hand panel has never been shown the
  thing this chapter is named after. Two honest notes: the "now" marker rounds
  $13.467$ Gyr to $13.5$, and the CMB marker's *time* coordinate is this model's
  radiation-free $465$ kyr, not the measured $380$ — it sits at the CMB's
  measured *scale factor*, and the model's curve puts it in time.</figcaption>
</figure>

## What the CMB tells us { #what-the-cmb-tells-us }

Calling this the most informative image ever taken is a statement about how easy
the subject is to model, not enthusiasm. Every other astronomical image is of
objects whose appearance depends on gas physics, star formation, feedback and
mergers, none of which anyone computes from first principles. The CMB predates
all of that: nearly uniform, gravitationally linear, governed by physics you can
write down completely — the one cosmological dataset whose forward model is
under control. Three things come out of it.

**The hot dense past is real.** The CMB is the most precise blackbody ever
measured — departures from the ideal curve are too small to draw at the line
width of any published plot. That matters because thermal equilibrium produces a
blackbody and effectively nothing else does. A spectrum that good requires
matter and radiation to have been thoroughly coupled, which requires the
opacity, which requires the density and the temperature.
[The first section](#running-it-backwards)'s reversal argument is an
extrapolation; this is the receipt.

**The numbers Ch. 14 of the main guide asks you to type.** The CMB is not
perfectly uniform: its temperature varies from patch to patch by amounts so
small the maps must be rescaled by orders of magnitude before your eye
registers anything. Those variations are the seeds gravity grew into galaxies,
and their statistics are extraordinarily constraining. Fit the pattern and out
come $\Omega_m$, the flatness of space, and a value of $H_0$. The
$\Omega_m = 0.3$ and $\Omega_\Lambda = 0.7$ you will type into `FlatLambdaCDM`
in [Ch. 14](../guide/14-frw.md#the-density-parameters) without being told where
they came from are that fit rounded off — carrying the same slight excess over
$0.3$ the age gap turned on — and $\Omega_\Lambda$ is not fitted separately at
all: it is what flatness leaves once $\Omega_m$ is fixed. They came from here.
So does the *other* $H_0$, the one producing $13.8$ Gyr rather than $13.467$.

!!! tip "You already know this"
    Cosmologists rarely report the CMB map itself; they report its **power
    spectrum**, a curve of variance against angular scale. That is not a
    compression convenience — it is a claim that the field is very nearly
    Gaussian, and a Gaussian random field's entire information content lives in
    its autocorrelation, equivalently (by
    [Ch. 7](../guide/07-fourier.md#psd-and-autocorrelation)'s Wiener–Khinchin
    relation) in its PSD. The map has more pixels; it has no more information.
    This repository leans on the identical fact when it models correlated image
    noise with a fitted kernel rather than a covariance matrix
    ([Ch. 24](../guide/24-correlated-noise.md#fitting-the-kernel)).

**Dark matter, independently.** The shape of that power spectrum depends on how
much gravitating matter was present versus how much of it could feel the photon
pressure. Ordinary matter can; dark matter cannot. The fit demands far more of
the second kind, with no reference to any galaxy's rotation or any cluster's
dynamics — making the CMB the fourth of the independent witnesses
[Ch. 12](12-dark-matter.md#four-independent-witnesses) lines up, and the one
whose systematics have nothing in common with the other three.

What the CMB does not tell you is what came before it: those photons carry no
information from earlier than last scattering, because earlier than that they
were being scattered. The fog is as opaque looking back through as it was
looking forward. Everything this chapter said about $t < 380{,}000$ years is
inference from a model constrained by lighter-touch evidence — chiefly the
abundances of the light elements — and the first fraction of a second is not
known at all.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 13 — The expanding universe and redshift](../guide/13-expansion.md#scale-factor)**
      defines the scale factor as carrying "no units and no meaning in
      isolation", normalised so $a(t_0) \equiv 1$ today, then says its "actual
      functional form … is set by how much matter and dark energy the universe
      holds". It never mentions the other end of that function. You now have it:
      $a\to0$ is the Big Bang, $13.467$ Gyr back in this book's cosmology, and
      where the model's licence expires rather than an event anyone can compute.
      That also puts Ch. 13's table of scale factors in proportion — its
      smallest entry, the Carousel's reference plane, is a bit under half, while
      the CMB's is $1/1101$. Everything this repository lenses happened in the
      most recent, largest, coldest sliver of that history.
    - **[Ch. 14 — FRW and the Friedmann equations](../guide/14-frw.md#friedmann)**
      tells you `.age(0)` is "that same equation integrated back to $a=0$" — one
      clause, and one of only two mentions of $a = 0$ in the entire guide, both
      of them in that chapter and both in passing. It then quotes the age at
      $13.467$ billion years and moves on, without a word about the number
      everyone else quotes being $13.8$. You now know what the
      integration runs back *to*, why the answer is finite, and why this
      repository's answer and Planck's differ by $2.47$ percent rather than by a
      slip. The same chapter's
      **[density parameters](../guide/14-frw.md#the-density-parameters)**
      section says "**flatness is the separate, empirical claim that
      $\Omega_k = 0$**", and that `FlatLambdaCDM` "doesn't measure this — it
      assumes it". It is right to flag that, and never says who *did* measure it.
      The CMB did, along with the $\Omega_m = 0.3$ you type beside it.

## Exercises { #exercises }

??? question "Exercise 11.1 — The temperature that names the background"
    The CMB is observed today at $2.725$ K and comes from a redshift of about
    $1100$. Using $T \propto 1/a$, compute the scale factor at emission and the
    temperature the radiation had then. Name an astronomical object with roughly
    that surface temperature.

    ??? success "Solution"
        $a = 1/(1+z) = 1/1101 \approx 0.000908$
        <!-- check: pch11.cmb_scale_factor = 0.000908 ± 0.000001 -->.
        Since $T \propto 1/a$, the emission temperature is
        $2.725 \times 1101 \approx 3000$ K
        <!-- check: pch11.cmb_emission_temp_k = 3000 ± 0.5 -->
        — a cool red star's photosphere. The redshift and the temperature ratio
        are the same number read twice.

??? question "Exercise 11.2 — Two ages, one gap"
    This book's cosmology gives $13.467$ Gyr; every popular account says
    $13.8$ Gyr. Compute the percentage gap. Then, without doing an integral:
    Planck's $H_0$ is lower than this repository's $70$ — which way does that
    alone push the age, and why?

    ??? success "Solution"
        $100 \times (13.8 - 13.467)/13.467 = 2.47$ percent
        <!-- check: pch11.age_gap_percent = 2.47 ± 0.01 -->
        <!-- check: pch11.universe_age_gyr = 13.467 ± 0.01 -->
        <!-- check: pch11.planck_age_gyr = 13.8 ± 0.01 -->.
        A lower $H_0$ is a slower expansion, so the universe needed longer to
        reach its current size: lower $H_0$, longer age. The crude version is
        the Hubble time $1/H_0$, which
        [Ch. 13](../guide/13-expansion.md#hubbles-law) computes and correctly
        warns is only "roughly the age of the universe". Planck's $\Omega_m$ is
        also slightly higher, pushing the other way (more matter, more early
        deceleration), but $H_0$ dominates.

??? question "Exercise 11.3 — When the model is outside its domain"
    `FlatLambdaCDM(70, 0.3)` reports its own age at $z = 1100$ as $465{,}000$
    years; the measured value is $380{,}000$. Which is wrong, what is the model
    missing, and why does this not undermine any other number in this book?

    ??? success "Solution"
        The model is wrong, for a nameable reason: `FlatLambdaCDM(70, 0.3)`
        carries matter and dark energy and no radiation — its `Tcmb0` is zero.
        Radiation drives a faster expansion than matter at the same density, so
        the real radiation-dominated early universe reached $a = 1/1101$ sooner
        than a matter-only model can. Its $465{,}000$
        <!-- check: pch11.cmb_age_this_model_kyr = 465 ± 1 -->
        is an overestimate by construction, not a measurement in tension with
        anything.

        It undermines nothing else because radiation's share of the budget falls
        as the universe expands, and at the redshifts this repository works at —
        $z = 0.5$ lenses, $z = 2$ sources — it is negligible. A model can be
        exactly right where it is used and useless three orders of magnitude
        outside it. The failure mode is not the model; it is quoting the model's
        number and the measured one as though they were the same number.
