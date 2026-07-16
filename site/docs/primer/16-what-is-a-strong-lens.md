# 16. What a strong lens is, and where to go next

Fifteen chapters have been building one object.
[Chapter 15](15-gravity-bends-light.md#the-idea) established the physics: mass
curves spacetime, light follows the curve, and Eddington measured the bend.
This chapter puts that physics into the only configuration that makes it
useful — a massive galaxy sitting on the sightline between you and something
far behind it. That is a **strong gravitational lens**, which the main guide
names on its first page and never draws. By the end you will have the geometry,
a number for how rare the accident is, the vocabulary for what it looks like on
a plate, the reasons anyone spends a telescope night on one, and a map into the
main guide. This is the last chapter, and it is the handoff.

!!! abstract "What you can skip"
    You do not need to be told that finding a rare class in a huge candidate
    set is a base-rate problem, or why excellent specificity still delivers
    mostly false positives at a prior this low. You have shipped that argument.
    What is new is the *physical* source of the base rate: where the $10^{-12}$
    comes from, why it is a solid angle rather than a population fraction, and
    why it is not the number that governs a survey's yield. No deflection
    algebra appears here.

## The geometry { #the-geometry }

Three things sit on one line: you, a galaxy, and a second galaxy far behind the
first. Nothing else is required, and nothing else is available — this is not an
experiment anyone set up.

Fix the redshifts, the pair the main guide uses as its standard test case
throughout Part IV without ever saying what they mean physically. The near
galaxy — the **lens**, or **deflector** — sits at
$z_{\mathrm{lens}}=0.5$<!-- check: pch16.z_lens = 0.5 ± 0.001 -->; the
**source** sits at $z_{\mathrm{source}}=2.0$<!-- check: pch16.z_source = 2 ± 0.001 -->.
[Chapter 8](08-redshift.md#z-in-real-numbers) turned those into distances and
lookback times: the source's light left billions of years before the lens's
did, and the gap between them along the sightline dwarfs either galaxy's width.

That gap is the first thing to get right, because the everyday sense of "in
front of" is wrong here. The lens and the source are not neighbours: not
interacting, not bound, not part of the same structure, neither aware the other
exists. They share exactly one property, a **direction**. Light from the source
happens to pass close to the lens on its way to your detector, and close is an
angle on the sky, not a distance in space. Two galaxies "on one sightline" is a
fact about your position, not theirs — and
[Chapter 5](05-light.md#looking-back-in-time) has the sharper version: an image
is never a report of *now*, and these two are not even reports of the same
*then*.

<figure markdown="span">
  ![Schematic of a strong lens: an observer at redshift zero, a lens galaxy at redshift 0.5, and a source galaxy at redshift 2.0 on one sightline, with two light rays bending at the lens plane and arriving from either side of the true direction](figures/p16-lens-geometry-light.svg#only-light){ width="90%" }
  ![Schematic of a strong lens: an observer at redshift zero, a lens galaxy at redshift 0.5, and a source galaxy at redshift 2.0 on one sightline, with two light rays bending at the lens plane and arriving from either side of the true direction](figures/p16-lens-geometry-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 16.1.** The whole geometry: you on the
  left, the lens galaxy at $z=0.5$, the source galaxy at $z=2.0$. The dotted
  line is where the source actually is. The two solid rays are paths light
  really takes — each bends at the lens plane and arrives from a direction that
  is *not* the dotted line, so you see the source displaced, twice. The figure
  lies about two things on purpose. The rays leave you some eight degrees off
  the dotted line and kink through about twenty at the lens plane; both real
  angles, computed below, are of order one arcsecond, so a true-to-scale
  version would show two lines indistinguishable from the dotted one. And the
  horizontal axis is not a distance scale — the lens sits where the labels fit,
  not where $z=0.5$ falls. What it gets right is the topology: two paths, one
  source, and a deflection that happens at one plane rather than gradually
  along the way.</figcaption>
</figure>

The figure's last claim is worth naming. A real galaxy has depth — its mass
spreads over a hundred thousand light-years along the sightline, not a plane.
Treating it as a **thin lens**, a flat sheet of projected mass that bends light
once and instantly, is an approximation, and a good one for the reason
Figure 16.1 makes visible: the galaxy's own depth is negligible against the
distances to it and past it.
[Ch. 16 of the main guide](../guide/16-deflection.md#the-thin-lens) does that
integral and marks the exact line where "thin" enters. You need only know that
the flat sheet in every equation from there on is a real 3-D galaxy, squashed
on purpose, with a stated justification.

One more distinction, since "strong" is doing work. Every mass on every
sightline bends light a little; a background galaxy's shape is distorted by a
percent or so, and measuring that statistically across millions of galaxies is
**weak** lensing — a different discipline with different tools. A lens earns
**strong** only when the alignment is close enough to produce *more than one
image*, or a visibly stretched arc. That threshold has a number, and the number
is why any of this is rare.

## Why it is rare { #why-it-is-rare }

The threshold is an angle called the **Einstein radius**, $\theta_{\mathrm E}$:
the source must lie within roughly $\theta_{\mathrm E}$ of the lens's axis, as
seen by you, for multiple images to appear at all.

$\theta_{\mathrm E}$ depends on the lens's mass and on the two redshifts. Mass,
for an elliptical, is read off its **velocity dispersion** $\sigma_v$ — the
spread of speeds of its stars, which a spectrum measures directly
([Ch. 7](07-spectra.md#continuum-and-lines) is how,
[Ch. 12 of the main guide](../guide/12-spectroscopy.md#sigma-v-from-lines) is
the arithmetic). A fiducial massive elliptical, the kind
[Ch. 3](03-galaxies.md#which-ones-lens) identified as this program's quarry and
heavier than the Milky Way, has
$\sigma_v = 250$ km/s<!-- check: pch16.sigma_v_kms = 250 ± 0.001 -->, and at
$z_{\mathrm{lens}}=0.5$, $z_{\mathrm{source}}=2.0$ that gives

$$
\theta_{\mathrm E} = 1.145''
$$

<!-- check: pch16.theta_e_typical_arcsec = 1.145 ± 0.001 -->

The formula is [Ch. 19's](../guide/19-einstein-radius.md#theta-e-from-sigma-v);
one structural fact is worth carrying there. $\theta_{\mathrm E}$ goes as
$\sigma_v^2$ — mass enters quadratically — and that exponent is what separates a
galaxy's ring from a cluster's, larger by about an order of magnitude
([Ch. 4](04-clusters.md#why-cluster-rings-are-bigger) puts the two side by
side).

Now turn $1.145''$ into a rarity. An arcsecond is $1/3600$ of a degree, so the
disc of sky inside the Einstein radius has area

$$
\pi\,\theta_{\mathrm E}^2 = 3.18\times10^{-7}\ \mathrm{deg}^2
$$

<!-- check: pch16.lens_disc_sq_deg = 3.18e-7 ± 0.01e-7 -->

against a whole sky of $41{,}253\ \mathrm{deg}^2$<!-- check: pch16.sky_sq_deg = 41253 ± 1 -->
— $4\pi$ steradians in square degrees, a definition rather than a measurement.
Divide, and one lens galaxy's disc covers
$7.71\times10^{-12}$<!-- check: pch16.alignment_fraction = 7.71e-12 ± 0.01e-12 -->
of the sky: one part in $1.30\times10^{11}$<!-- check: pch16.one_in_n_sightlines = 1.30e11 ± 0.01e11 -->.
Point somewhere at random and those are the odds it lands inside a given lens
galaxy's disc — near enough the odds of picking one particular star out of the
Milky Way's $10^{11}$ ([Ch. 1](01-scale-ladder.md#our-galaxy)). That is the
target strong lensing has to hit, and nothing aims it.

Two caveats, because this number is easy to overread.

**It is a per-pair figure, not a lens rate.** It is the chance one direction
lands inside one galaxy's disc — not the fraction of galaxies that are lenses,
and not a survey's yield. This book does not compute a yield, and Exercise 16.3
is where you work out why you cannot get one from the figure above.

**A $\sigma_v = 250$ km/s<!-- check: pch16.sigma_v_kms = 250 ± 0.001 --> galaxy
is not a typical galaxy.** It is a fiducial massive elliptical, chosen because
it is the kind that lenses. Most galaxies are far less massive, and since
$\theta_{\mathrm E}\propto\sigma_v^2$ and disc area goes as
$\theta_{\mathrm E}^2$, the target shrinks as the *fourth* power of the
dispersion. Halve the dispersion and the disc is sixteen times smaller, with an
Einstein radius no ground-based survey could resolve even if the alignment
happened. Rarity is not one number; it is a steeply mass-weighted one.

So how does anyone find any? Brute force. The DESI Legacy Surveys' DR11 sweep
scored $5.38\times10^{7}$<!-- check: pch16.dr11_galaxies_scored = 5.38e7 ± 0.01e7 -->
galaxies — one classifier decision each — for a few thousand candidates. That
is what [Ch. 6](06-telescopes.md#what-a-survey-is) meant by a survey: not a
telescope pointed at something interesting, but a telescope pointed at
everything, because the interesting thing is an accident nobody can predict.

!!! tip "You already know this"
    A prior this low is a precision problem before it is a physics problem. At
    $5.38\times10^{7}$<!-- check: pch16.dr11_galaxies_scored = 5.38e7 ± 0.01e7 -->
    decisions and a positive class of order one in ten thousand, a finder with
    a false-positive rate of even one in a thousand returns a list that is
    overwhelmingly contamination, and no AUC on a balanced validation set tells
    you so. This is the half of the program needing no lensing at all:
    [Ch. 27](../guide/27-discovery.md#the-operating-point) derives where the
    threshold has to sit and why the survey's own resolution bounds it;
    [Ch. 28](../guide/28-the-label.md#the-flat-line) asks whether the human
    grades those finders train against carry any information about truth. Both
    are yours already. The physics here supplies their base rate.

## What you see { #what-you-see }

The alignment is invisible. What you see is the light, rearranged, and it comes
in a small vocabulary.

**An arc.** For a circular lens the deflection points straight at the centre:
it changes how far a ray passes from the lens, not its direction around it. The
image therefore keeps the source's azimuth but sits at a larger radius — and
the same azimuthal span is a longer arc the farther out it sits. So the
source's radial extent comes through roughly intact while its tangential extent
is stretched, the more so the tighter the alignment. A small round blob becomes
a curved sliver, concentric with the lens galaxy. Arcs are what a lens
finder — human or convolutional — actually keys on.

**A ring.** Put the source exactly on the axis and the configuration is
rotationally symmetric: every direction around the lens is equally valid, so
the image is a complete circle of radius $\theta_{\mathrm E}$. This is the
**Einstein ring**, which this program's reproductions are named after. Note the
cost: exact alignment is measure-zero, so a perfect ring never happens.
Near-rings do, and they are the best systems in any catalogue, because a ring
displays $\theta_{\mathrm E}$ as a radius you can measure with a caliper.
Figure 16.1 is a slice through this case — its source is on the axis — and its
two rays are two points on a circle the 2-D picture would show as continuous.

**A double.** Move the source off-axis but keep it inside $\theta_{\mathrm E}$
and the ring breaks into two images on opposite sides of the lens. Push it
outside and the second image does not fade — it ceases to exist.
[Ch. 17](../guide/17-lens-equation.md#multiple-images) proves that by trying to
find the root and failing, and derives a result you can use without its
algebra: for an isothermal lens — the profile real ellipticals sit close to,
for reasons the next section admits nobody has explained — the two images of a
double are separated by $2\theta_{\mathrm E}$, and where inside the Einstein
radius the source sits cancels out completely. A separation is a ruler
measurement; it converts to $\theta_{\mathrm E}$ by dividing by two, and
$\theta_{\mathrm E}$ converts to a mass.

**A quad.** Four images. A circular lens cannot make one — the mass has to be
elliptical, which real ellipticals are. The extra pair appears when the source
lands inside a region called a **caustic**, the thing the main guide's landing
page admits it assumes you already know
([Ch. 18](../guide/18-magnification.md#caustics) is where you meet it). Quads
are prized because four images constrain a model far harder than two.

[Ch. 17](../guide/17-lens-equation.md#the-lens-equation) says those three words
"cover almost every strong lens actually observed", and derives all three as
root counts of one equation. You now know what they are as things in a picture,
which is the half that chapter takes for granted.

## Why anyone cares { #why-anyone-cares }

Three reasons, in ascending order of what they are worth.

**Mass, including the mass that does not shine.** Deflection responds to *all*
the mass on the sightline, whether or not it emits light. So a lens weighs a
galaxy without asking what it is made of — and the answer comes out
consistently larger than the stars account for. That is the fourth of the
independent witnesses in
[Ch. 12](12-dark-matter.md#four-independent-witnesses), and the hardest to argue
with, because it assumes nothing about how stars move or how gas heats. The
main guide has a clean identity for it: the mean projected mass density inside
$\theta_{\mathrm E}$, in the natural units of lensing, is exactly $1$ — not
approximately — which turns a measured ring radius into an enclosed mass in one
step ([Ch. 19](../guide/19-einstein-radius.md#the-mean-convergence-identity)
proves it; [Ch. 19](../guide/19-einstein-radius.md#mass-inside-theta-e) cashes
it).

**The shape of the mass, not only its total.** A ring radius gives the mass
inside one circle; an arc gives more, because its brightness and curvature
respond to how the density *falls off* with radius. That slope is the main
guide's destination number, $\gamma$ in $\rho\sim r^{-\gamma}$
([Ch. 20](../guide/20-profiles.md#the-epl-and-gamma) defines it; all of Part V
extracts one galaxy's value from pixels). Why an exponent is worth a campaign:
the stars in an elliptical follow one profile, the dark matter another, and the
*total* lands very close to a specific slope for reasons nobody has explained.
The main guide calls that the isothermal conspiracy
([Ch. 10](../guide/10-galaxies.md#the-isothermal-conspiracy)) and is candid that
it is an unexplained regularity, not a derived one.

**$H_0$.** This one reaches outside the field. When the source varies — a
quasar, a supernova — the images vary too, but not at the same moment, because
the light took paths of different lengths. The **time delay** between images is
measured in days; a delay is a length, a length across a known redshift is a
distance, and a distance against a redshift is $H_0$. That chain is independent
of the distance ladder and of the CMB, which is why it matters:
[Ch. 14](14-hubble-tension.md#the-third-method) laid out the argument between
the ladder's answer and the CMB's, and the reader's own field is the
independent third voice
([Ch. 21](../guide/21-degeneracies.md#time-delays-and-h0) works it).

Now the part that should not be sold. A strong lens does not measure mass
unambiguously. Add a uniform sheet of mass across the image and rescale the
source to compensate, and the result is a *pixel-identical* image — same arcs,
same positions, same brightnesses, different mass. This is the **mass-sheet
degeneracy**; it is exact rather than approximate, no better sampler removes
it, and it is the largest single reason lensing's $H_0$ values move when the
assumptions behind them move.
[Ch. 21](../guide/21-degeneracies.md#the-mass-sheet-degeneracy) states it;
[Ch. 21](../guide/21-degeneracies.md#degeneracy-is-gauge-symmetry) reframes it
as something you recognise from anywhere a model has a direction the data
cannot see. Every claim above is real; each is also conditioned on an
assumption that breaks that degeneracy, and the assumption is where the arguing
happens.

## Where to go next { #where-to-go-next }

The book is finished. Here is the map.

The destination was always one sentence: **you can open the main guide at
Ch. 9 and not be lost.** That chapter's first line is "Nothing astronomical
comes with a ruler" — and you now know what it is failing to rule.

| Main guide part | Open at | What this book gave you |
|---|---|---|
| **I — The mathematical spine** (2–8) | Skim, or skip | Nothing. It is your own field's neighbour. Read it when a later chapter cites it. |
| **II — The physical universe** (9–12) | **[Ch. 9](../guide/09-units.md#angles-on-the-sky)** — start here | Almost the whole book: [Ch. 1](01-scale-ladder.md) (the scale ladder), [Ch. 2](02-stars.md#what-a-star-is) (why old means red — the premise under Ch. 10's stellar populations), [Ch. 3](03-galaxies.md) (which galaxies lens), [Ch. 5](05-light.md) (photons, blackbodies, inverse square), [Ch. 6](06-telescopes.md#the-three-observatories) (HST, DESI and Euclid, none of which the guide introduces), [Ch. 7](07-spectra.md) (lines), [Ch. 8](08-redshift.md) (what $z$ means). |
| **III — Cosmology** (13–15) | **[Ch. 13](../guide/13-expansion.md#scale-factor)** | [Ch. 10](10-expansion.md) (the raisin bread Ch. 13 formalises as $a(t)$), [Ch. 11](11-big-bang.md#running-it-backwards) (what $a=0$ is; the CMB), [Ch. 12](12-dark-matter.md#what-cdm-stands-for) (what the CDM in `FlatLambdaCDM` stands for — you type it in Ch. 14), [Ch. 13](13-dark-energy.md#the-energy-budget) (what $\Omega_m=0.3$ asserts), [Ch. 14](14-hubble-tension.md) (why $H_0$ is contested at all). |
| **IV — Gravitational lensing** (16–21) | **[Ch. 16](../guide/16-deflection.md#newtonian-deflection)** | [Ch. 15](15-gravity-bends-light.md#why-a-factor-of-two-mattered) (why the factor of two was worth an eclipse) and this chapter (what the thing being modelled *is*). |
| **V — This repository's science** (22–26) | **[Ch. 22](../guide/22-inference.md#the-forward-model)** | This chapter's "why anyone cares" — the only reason to model an arc at all. The rest is inference, and it is yours. |
| **VI — Discovery** (27–28) | **[Ch. 28](../guide/28-the-label.md#the-flat-line)** — the guide says read it second | [Ch. 6](06-telescopes.md#what-a-survey-is) (what a survey is) and this chapter's base rate. Ch. 28 needs no physics; it is pure label epistemics. |
| **VII — Synthesis** (29) | **[Ch. 29](../guide/29-how-to-read.md#the-paper-decoder)** | Read it last, as intended. |

Two things to carry across the boundary.

The first: this book gave you nouns, not derivations. When the main guide
integrates something you will not be able to check the integral from anything
here — but you will know what it is integrating, which was never the part that
was missing. [The handoff is the structure](index.md#how-to-read), and every
Unlocks box was a receipt for it.

The second is a habit rather than a fact. The main guide's culture is to
retract its own numbers — its final report carries a "$\sim17\sigma$" claim
that reconciles with none of the uncertainties it quotes, and
[Ch. 25](../guide/25-money-number.md#the-sigma-arithmetic) makes you do the
division yourself instead of telling you the answer. This book ran the same
gate: every number on every page is tagged, recomputed by a script, and fails
the build if it drifts. Nothing here asked to be believed. Take that to Ch. 9.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 1 — What this repository does, and what one number costs](../guide/01-orientation.md#the-two-halves)**
      describes discovery as combing "through tens of millions of galaxy images
      for the rare few whose light has been bent by something massive sitting
      in front of them" — then never says why they are rare, or what "in front
      of" means. It means one sightline and nothing else: two galaxies billions
      of light-years apart sharing a direction and no other property. "Tens of
      millions" is $5.38\times10^{7}$<!-- check: pch16.dr11_galaxies_scored = 5.38e7 ± 0.01e7 -->
      cutouts; "rare" is $7.71\times10^{-12}$<!-- check: pch16.alignment_fraction = 7.71e-12 ± 0.01e-12 -->
      of the sky per lens. Ch. 1 also puts $\gamma = 1.103 \pm 0.008$, "the
      density slope of one galaxy's mass profile", on its first page as the
      book's destination — and you now know what galaxy, and why a slope rather
      than a mass.
    - **[Ch. 9 — Arcseconds, magnitudes, and the units of the sky](../guide/09-units.md#angles-on-the-sky)**
      is this book's destination. It notes in a parenthesis that $1''$ is
      "already large by lensing standards — most Einstein radii are $1$–$2''$"
      and moves on, ten chapters before an Einstein radius exists. You now have
      $\theta_{\mathrm E} = 1.145''$<!-- check: pch16.theta_e_typical_arcsec = 1.145 ± 0.001 -->
      for a $\sigma_v = 250$ km/s<!-- check: pch16.sigma_v_kms = 250 ± 0.001 -->
      elliptical at $z_{\mathrm{lens}} = 0.5$, $z_{\mathrm{source}} = 2.0$, and
      the arithmetic that makes that scale the whole game. Its opening —
      "Nothing astronomical comes with a ruler" — is where
      [Ch. 9 of this book](09-distance-ladder.md#nothing-comes-with-a-ruler)
      stops being a problem statement and becomes a four-rung ladder.
    - **[Ch. 16 — How much light bends, and the factor of two](../guide/16-deflection.md#the-thin-lens)**
      proves that general relativity's deflection is exactly twice Newton's,
      and assumes throughout that you know why anyone wants a deflection angle.
      You do: [Ch. 15](15-gravity-bends-light.md#eddington-1919) gave you the
      eclipse that settled the factor, and this chapter gave you the geometry
      that makes its central approximation reasonable on sight — a galaxy's own
      hundred-thousand-light-year depth is nothing against the gap between
      $z=0.5$ and $z=2.0$, exactly the condition Ch. 16 marks as "where 'thin'
      actually enters".
    - **[Ch. 17 — The lens equation](../guide/17-lens-equation.md#multiple-images)**
      states that "three words cover almost every strong lens actually
      observed: a **double**, a **ring**, or a **quad**", and derives all three
      as root counts of $\beta = \theta - \alpha(\theta)$ without ever saying
      what one looks like. You now have them as objects. And Ch. 17's
      $\boldsymbol\beta$ — "where the source would be if nothing were in the
      way" — is a real galaxy at
      $z=2$<!-- check: pch16.z_source = 2 ± 0.001 -->, ten billion years back,
      with no idea it is being modelled.

## Exercises { #exercises }

??? question "Exercise 16.1 — rebuild the rarity"
    Start from $\theta_{\mathrm E} = 1.145''$. Convert to degrees, compute the
    area of a disc of that radius in square degrees, divide by the
    $41{,}253\ \mathrm{deg}^2$ of the whole sky, and invert.

    ??? success "Solution"
        $1.145/3600 = 3.181\times10^{-4}$ degrees; disc area
        $\pi\,(3.181\times10^{-4})^2 = 3.18\times10^{-7}\ \mathrm{deg}^2$
        <!-- check: pch16.lens_disc_sq_deg = 3.18e-7 ± 0.01e-7 -->; divided by
        the sky<!-- check: pch16.sky_sq_deg = 41253 ± 1 -->,
        $7.71\times10^{-12}$
        <!-- check: pch16.alignment_fraction = 7.71e-12 ± 0.01e-12 -->, or one
        in $1.30\times10^{11}$
        <!-- check: pch16.one_in_n_sightlines = 1.30e11 ± 0.01e11 -->. Note what
        did no work: no cosmology, no dark matter, no general relativity. Once
        $\theta_{\mathrm E}$ is a number, the rarity is geometry — a small disc
        against a sphere.

??? question "Exercise 16.2 — which one is the lens?"
    Two galaxies appear at essentially the same position on the sky; spectra
    give redshifts $0.5$ and $2.0$. Which is the deflector, and how do you know?
    Is there anything in the *image alone* that tells you the same thing?

    ??? success "Solution"
        The $z=0.5$ galaxy<!-- check: pch16.z_lens = 0.5 ± 0.001 --> is the
        lens; the $z=2.0$ one<!-- check: pch16.z_source = 2 ± 0.001 --> is the
        source. Bigger $z$ means the light left when the universe was smaller,
        so longer ago, so farther away
        ([Ch. 8](08-redshift.md#bigger-z-means-farther)) — and light from behind
        is what gets bent by what is in front. Nothing about their *positions*
        decides it: position is an angle and both are at the same angle, which
        is the entire point of a sightline. From the image alone you can guess —
        the deflector is the big red smooth one at the centre
        ([Ch. 3](03-galaxies.md#spirals-and-ellipticals)), the source the blue
        stretched sliver around it. Guessing is what a lens finder does, which
        is why a candidate stays a candidate until somebody takes a spectrum.

??? question "Exercise 16.3 — what the number does not say"
    A colleague reads "one in $1.30\times10^{11}$", multiplies by the
    $5.38\times10^{7}$ galaxies DESI scored, and concludes the survey should
    have found no lenses at all. It found a few thousand. Where is the
    reasoning wrong?

    ??? success "Solution"
        Two errors, pushing the same way. First, the alignment fraction
        <!-- check: pch16.alignment_fraction = 7.71e-12 ± 0.01e-12 --> is the
        fraction of the *sky* one lens's disc covers — the chance a randomly
        thrown direction lands in it. Multiplying it by a count of *galaxies*
        multiplies a solid-angle fraction by a population; the two have nothing
        to divide out, and the product is a rate of nothing. Second, the correct
        calculation asks how many *background sources* are available to fall
        into each foreground disc, and the sky behind a lens is not empty — it
        is crowded with faint high-redshift galaxies. The real yield is a tiny
        per-pair probability times a very large number of pairs, and this book
        does not compute it. What the $7.71\times10^{-12}$ is good for is
        intuition about the *target*: the alignment tolerance is absurdly tight.
        It says nothing about how often nature has hit it.

??? question "Exercise 16.4 — where do you open it?"
    For each question, name the chapter of the main guide you would open and the
    chapter of *this* book that prepared you for it.

    1. What does `pixscale=0.262` in a line of survey code physically assert?
    2. Why does the repository's lens-modeling likelihood contain no cosmology,
       even though its two galaxies are at wildly different redshifts?
    3. Is the human grade on a lens candidate worth training against?
    4. What is $\gamma = 1.103 \pm 0.008$ a measurement *of*, and should you
       believe it?

    ??? success "Solution"
        1. [Ch. 9](../guide/09-units.md#pixel-scale) — pixel scale in
           arcsec/pixel. Prepared by
           [Ch. 6](06-telescopes.md#ground-versus-space) (why the ground/space
           split is a resolution argument) and [Ch. 1](01-scale-ladder.md) (why
           an angle is the only thing a telescope measures).
        2. [Ch. 17](../guide/17-lens-equation.md#the-lens-equation) — every
           distance factor folds once into the scaled deflection, after which
           the problem is purely angular. Prepared by
           [Ch. 8](08-redshift.md#the-cosmological-picture) (what the redshifts
           mean) and this chapter's geometry (why there are exactly three planes
           to carry distances between).
        3. [Ch. 28](../guide/28-the-label.md#the-flat-line) — the most striking
           result in the repository. Prepared by nothing in this book,
           deliberately: it needs no astronomy. Read it second, as Ch. 1 says.
        4. [Ch. 20](../guide/20-profiles.md#the-epl-and-gamma) for what it
           measures — the exponent in $\rho \sim r^{-\gamma}$ — and
           [Ch. 25](../guide/25-money-number.md#the-verdict) for whether to
           believe it. Prepared by [Ch. 12](12-dark-matter.md#halos) (most of
           that mass is dark, so the profile is not the starlight's) and this
           chapter's third section (why an arc constrains a slope, and what the
           mass-sheet degeneracy costs when it does). Do not skip to the
           verdict: Ch. 1 asked you to write down a guess first, and that
           exercise works once.
