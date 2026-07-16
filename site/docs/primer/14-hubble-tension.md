# 14. The Hubble tension, and why lensing arbitrates it

[Chapter 10](10-expansion.md#hubbles-law) handed you $v = H_0 d$ and left
$H_0$ as a number somebody measures. Two groups measure it. They disagree, by
more than either one's error bar permits, and after two decades of people
trying to find the mistake, nobody has found it. This chapter is about that
gap: what the two numbers are, why splitting the difference does not settle it,
and why the method that could arbitrate it is the one this entire research
program is built on. It is the one place in this book where your own field is
not a spectator.

!!! abstract "What you can skip"
    You do not need to be walked through combining independent errors in
    quadrature, or through what "five sigma" asserts. Skip that arithmetic and
    read for the asymmetry instead: one number is a direct measurement of a
    local quantity, the other a model's *prediction* of it, extrapolated from an
    observation of the very early universe. Those are not the same kind of
    object, and that is why the gap is interesting rather than merely
    irritating.

## Two answers { #two-answers }

**The ladder.** [Chapter 9](09-distance-ladder.md#standard-candles) built the
distance ladder: parallax calibrates Cepheid variables, Cepheids calibrate
Type Ia supernovae, and supernovae reach far enough out that the expansion
dominates the random peculiar motions of individual galaxies. Measure a
distance and a redshift for enough supernovae, fit the slope of $v$ against
$d$, and the slope *is* $H_0$. The SH0ES program has been running exactly this
measurement, and re-running it against every systematic anyone could name, for
two decades. Its answer:

<!-- check: pch14.h0_ladder = 73.0 ± 0.05 -->
<!-- check: pch14.h0_ladder_err = 1.0 ± 0.01 -->

$$
H_0^{\mathrm{ladder}} = 73.0 \pm 1.0\ \mathrm{km/s/Mpc}.
$$

Every rung is local, in the sense that matters here: nothing in it assumes what
the universe was doing before the light left. It assumes only that Cepheids are
what we think they are, and that the rungs are calibrated against each other —
the error-compounding problem
[Chapter 9](09-distance-ladder.md#errors-compound) warned you about.

**The CMB.** [Chapter 11](11-big-bang.md#the-cmb) gave you the other one — a
different kind of measurement entirely. The cosmic microwave background
is a photograph of the universe from before any galaxy existed, and its
temperature fluctuations have a characteristic angular size. That size is not
arbitrary: sound waves in the pre-recombination plasma froze out at a physical
length the physics predicts, so the map hands you a known ruler, seen at a
known epoch, subtending a measured angle. What happens next is the important
part: you do **not** read $H_0$ off the map. You fit a model — flat
$\Lambda$CDM, whose ingredients
[Chapter 12](12-dark-matter.md#what-cdm-stands-for) and
[Chapter 13](13-dark-energy.md#the-energy-budget) named — to that angle, then
run it *forward* to today and read off what it says today's expansion rate must
be. Planck's answer:

<!-- check: pch14.h0_cmb = 67.4 ± 0.05 -->
<!-- check: pch14.h0_cmb_err = 0.5 ± 0.01 -->

$$
H_0^{\mathrm{CMB}} = 67.4 \pm 0.5\ \mathrm{km/s/Mpc}.
$$

The gap between them is

<!-- check: pch14.h0_difference = 5.6 ± 0.01 -->

$$
73.0 - 67.4 = 5.6\ \mathrm{km/s/Mpc},
$$

<!-- check: pch14.percent_difference = 8.31 ± 0.01 -->

or $8.31$ percent of the smaller value. Combine the uncertainties in quadrature
— the two sets of errors have nothing to do with each other — and the
disagreement is

<!-- check: pch14.tension_sigma = 5.01 ± 0.01 -->

$$
\frac{73.0 - 67.4}{\sqrt{1.0^2 + 0.5^2}} \approx 5.01\ \sigma.
$$

<figure markdown="span">
  ![Two measurements of the Hubble constant plotted with error bars that do not overlap, plus the value this repository asserts](figures/p14-h0-tension-light.svg#only-light){ width="90%" }
  ![Two measurements of the Hubble constant plotted with error bars that do not overlap, plus the value this repository asserts](figures/p14-h0-tension-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 14.1.** The whole controversy on one axis.
  Planck's CMB value ($67.4 \pm 0.5$) on top, SH0ES's ladder value
  ($73.0 \pm 1.0$) below it, and — with no error bar at all, because it is an
  assertion rather than a measurement — the $70$ this repository fixes by fiat
  everywhere cosmology touches it. The bars are $1\sigma$, and they do not come
  close to touching.</figcaption>
</figure>

## Why it matters { #why-it-matters }

A number without a comparison is not doing its job. Both camps quote km/s/Mpc —
extra recession speed per megaparsec of distance — and a megaparsec dwarfs this
book's ruler: the Milky Way's whole stellar disc fits inside one many times
over, with Andromeda well within the first. That is also where the picture
breaks. Andromeda is not receding at all; it is falling towards us. The Local
Group is bound, and expansion does not operate inside a bound system, so $H_0$
describes the average sightline in the open field, not the neighbourhood. The
ruler that bites here is each camp's own claimed precision, because that is what
the disagreement has to be measured against.

The ladder claims to know its own answer to $1.37$ percent
<!-- check: pch14.h0_ladder_relerr_pct = 1.37 ± 0.01 -->
of itself. The CMB claims $0.74$ percent
<!-- check: pch14.h0_cmb_relerr_pct = 0.74 ± 0.01 -->
— better than one part in a hundred, on a property of the entire universe.
Against those, the $8.31$ percent
<!-- check: pch14.percent_difference = 8.31 ± 0.01 -->
gap between them is $6.07$ times the ladder's own quoted precision
<!-- check: pch14.gap_vs_ladder_relerr = 6.07 ± 0.01 -->
and $11.2$ times the CMB's
<!-- check: pch14.gap_vs_cmb_relerr = 11.2 ± 0.01 -->.
Each side claims a tolerance many times tighter than the thing they disagree
about. At most one can be right, and nobody can rule out that neither is. This
is also why averaging is not an option: the disagreement is itself evidence that
at least one *error model* is wrong, and averaging does not repair one.

!!! tip "You already know this"
    The asymmetry between the two numbers is one you have debugged before. The
    ladder measures $H_0$ where $H_0$ lives — in-distribution, at $z\approx 0$.
    The CMB fits a model to data from one regime, *extrapolates* it across the
    whole history of the universe, and reports the prediction with the fit's
    internal uncertainty attached. When a model with excellent in-sample
    residuals makes a confident out-of-distribution prediction that a direct
    measurement contradicts, the first hypothesis is not "the measurement is
    broken" — it is that the model class is misspecified somewhere between the
    regimes. That is not a metaphor here — it is one of the two live
    explanations, and it is why "new physics before recombination" is a phrase
    people say out loud in seminars.

The stakes are concrete. $H_0$ sets the age of the universe and the size of the
observable one, so [Chapter 11](11-big-bang.md#running-it-backwards) already
showed you this disagreement in different clothes: a gap between the age you
get by running the expansion backwards and the age the CMB fit reports. Age and
expansion rate are two readings of one dial, and a five-sigma disagreement about
the dial is a five-sigma disagreement about how old everything is.

## The third method { #the-third-method }

Suppose you had a way to measure $H_0$ that uses neither ladder nor CMB — no
Cepheids, no supernova calibration, no assumption about what the universe was
doing at recombination. It would inherit neither camp's systematics, so it
could not be wrong in either of the ways they might be. That is what
"independent" has to mean here, and it is a hard bar: most methods fail it by
standing on a rung of the ladder.

Strong gravitational lensing clears it. When a lens produces multiple images
of one background source ([Chapter 16](16-what-is-a-strong-lens.md#what-you-see)
shows the geometry), each image's light took a different route through the
lens's gravitational field, and the routes differ in length and in the depth of
well they cross. So the images do not arrive at the same time. If the source
flickers — a quasar, or a supernova — you see that flicker replayed in each
image, weeks or months apart. The delay is a *measured time*, in days, on a
wall clock.

The delay is a length divided by $c$, and the length is set by the geometry of
the whole configuration — how far away the lens and source are, and how the two
relate. The main guide calls that combination the **time-delay
distance**, and the single fact you need is that it scales as $1/H_0$: a
faster-expanding universe is a smaller one, so its distances shrink together
and its delays get shorter. Measure the delay, model the lens well enough to
predict what the delay *would* be at a reference distance, take the ratio, and
$H_0$ falls out.
[The main guide's Ch. 21](../guide/21-degeneracies.md#time-delays-and-h0) sets
it up in one line — "measure $\Delta t$, model $\Delta \tau$, read off
$D_{\Delta t}$, read off $H_0$" — and then spends the rest of the section on
why that line is harder than it looks.

The method touches no Cepheid and no CMB: one lens, general relativity, and a
stopwatch.

**And now the honest part.** Time-delay lensing has not ended the argument,
because of a problem this program's own guide devotes a chapter to: the
**mass-sheet degeneracy**. Add a uniform sheet of mass to the lens and rescale
the source, and every image position, every arc shape, every flux ratio comes
out *identical*. The imaging cannot see it — not poorly, not noisily, but
exactly not at all, because the transformation is a symmetry of what the image
data constrain
([the main guide's Ch. 21](../guide/21-degeneracies.md#the-mass-sheet-degeneracy)
proves this). The time delays are *not* invariant under it: they scale with the
sheet. So an unmodelled sheet biases the inferred $H_0$ by exactly its own
factor, and the bias runs in the direction that matters — a sheet you failed to
model makes $H_0$ come out too high.

[Ch. 21's time-delay section](../guide/21-degeneracies.md#time-delays-and-h0)
works a representative case, and for a sheet strength nobody would
call implausible the bias exceeds the entire $8.31$ percent
<!-- check: pch14.percent_difference = 8.31 ± 0.01 -->
gap this method is meant to arbitrate. Read that as an order of magnitude, not a
result: it is one illustrative sheet, and nobody knows the true sheet for any
individual lens — which is the problem. The point survives the crudeness. The
systematic lensing must control is larger than the effect it is measuring.

Lensing teams handle it by importing an external constraint that *is* sensitive
to mass rather than light-bending geometry — most often the lens galaxy's
stellar velocity dispersion, which that same time-delay section identifies as
the thing that breaks the degeneracy. How much you assume about the lens's mass
profile
then decides how tight your answer is, and that choice is not a detail. Tighten
the assumptions and lensing lands near the ladder with a small error bar.
Marginalise the mass-sheet freedom fully and the bar swells until the answer is
consistent with both camps and arbitrates nothing.

So the third method is real, genuinely independent, and without the last word.
Its precision is a function of how much you are prepared to believe about
elliptical galaxies — an uncomfortable place for a measurement to sit, and
where this one sits.

## Where it stands { #where-it-stands }

Nobody knows. That is the accurate summary, and it should be said before the
list of candidate explanations rather than after, because the list makes the
field look closer to an answer than it is.

Neither side has cracked under scrutiny. The ladder's most-suspected rung —
whether Cepheids in crowded fields are measured correctly — has been re-examined
with a better telescope, and the ladder still lands near $73.0$
<!-- check: pch14.h0_ladder = 73.0 ± 0.05 -->. The CMB's map
has been independently remade by other experiments, which agree with Planck.
Methods avoiding one rung or another tend to land between the two camps with
error bars wide enough to be consistent with everything, which is not a
resolution.

What is left is uncomfortable in an interesting way. Either one of two
extremely well-audited measurements carries a systematic that twenty years of
adversarial re-analysis has missed, or $\Lambda$CDM — the model
[Chapter 13](13-dark-energy.md#the-energy-budget) told you accounts for the
universe's contents — is missing something between recombination and now. Extra
relativistic species; dark energy that mattered briefly early and then went
away; dark energy whose density is not constant after all. Each proposal fixes
$H_0$ and breaks something else.

**And this repository asserts $70$.**

<!-- check: pch14.repo_h0 = 70.0 ± 0.01 -->
<!-- check: pch14.h0_midpoint = 70.2 ± 0.01 -->

The midpoint of the two camps is $70.2$, so the value hard-coded everywhere
cosmology touches this program sits a fifth of a unit below exactly halfway,
committed to neither side. That is not evasion, and it is worth seeing why. The
main guide's [Ch. 13](../guide/13-expansion.md#hubbles-law) and
[Ch. 14](../guide/14-frw.md#the-density-parameters) establish that this
program's money number never touches $H_0$: the lens-modelling likelihood runs
entirely in arcseconds, and cosmology enters only where a physical scale is
genuinely required — a short, enumerated list. Ch. 14 puts it flatly, that the
money number "would come out identical if this repository had asserted
$H_0=67$ instead."

Read that sentence again now that you know what $67$ is. It is not a rhetorical
alternative pulled from the air — it is the CMB camp's number, rounded. The main
guide names one side of the biggest open argument in cosmology without
mentioning that there is an argument. The scoping claim is true and worth
making; the number it makes the point with is the tension itself.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 13 — The expanding universe and redshift](../guide/13-expansion.md#hubbles-law)**
      states, as a bare repo convention, that *"everywhere this repository
      touches cosmology, it fixes $H_0 = 70$ km/s/Mpc"*, and derives the Hubble
      time and Hubble distance from it. What it never says is that $70$
      <!-- check: pch14.repo_h0 = 70.0 ± 0.01 -->
      is nobody's measurement: the camps sit at $73.0 \pm 1.0$
      <!-- check: pch14.h0_ladder = 73.0 ± 0.05 -->
      and $67.4 \pm 0.5$
      <!-- check: pch14.h0_cmb = 67.4 ± 0.05 -->
      and miss each other by $5.01\sigma$
      <!-- check: pch14.tension_sigma = 5.01 ± 0.01 -->.
      You can now read that chapter's $H_0$ for what it is — a placeholder in an
      argument it does not join.
    - **[Ch. 14 — FRW and the Friedmann equations](../guide/14-frw.md#the-density-parameters)**
      closes its scoping argument with *"it would come out identical if this
      repository had asserted $H_0=67$ instead."* That is the Hubble tension,
      named by its value and nothing else, in a book where the phrase never
      appears. You now know which camp $67$ belongs to, why the two do not
      overlap, and — the part that makes this your problem
      rather than a cosmologist's — that the strong lenses that chapter is a
      prerequisite for are the third method that could tell them apart, if the
      mass-sheet degeneracy
      ([Ch. 21](../guide/21-degeneracies.md#the-mass-sheet-degeneracy)) can be
      controlled.

## Exercises { #exercises }

??? question "Exercise 14.1 — The sigma, by hand"
    The two measurements are $73.0 \pm 1.0$ and $67.4 \pm 0.5$ km/s/Mpc.
    Compute the tension in sigma. Then state the two assumptions that
    arithmetic makes, and say which one you would attack first.

    ??? success "Solution"
        The difference is $5.6$
        <!-- check: pch14.h0_difference = 5.6 ± 0.01 -->
        km/s/Mpc. The errors are independent — nothing in a CMB map shares a
        systematic with a Cepheid photometry pipeline — so they add in
        quadrature, and the tension is

        $$
        \frac{73.0 - 67.4}{\sqrt{1.0^2 + 0.5^2}} \approx 5.01
        $$

        <!-- check: pch14.tension_sigma = 5.01 ± 0.01 -->

        sigma. Two assumptions are buried in it. First, that both error bars
        are Gaussian, so "five sigma" carries its usual tail probability; real
        systematic budgets are sums of terms whose distributions nobody knows,
        and the tails are exactly the part you cannot check. Second, that each
        quoted error is *complete*. Attack the second: the five-sigma result is
        itself the strongest available evidence against it. A disagreement this
        size is not primarily a claim about which value is right — it is a
        claim that at least one error model is incomplete.

??? question "Exercise 14.2 — The gap against each camp's own ruler"
    Express each camp's quoted uncertainty as a percentage of its own central
    value, and the gap between the camps as a percentage too. What is the gap
    in units of each camp's own claimed precision, and what does that tell you
    that the raw sigma does not?

    ??? success "Solution"
        The ladder: $1.0/73.0 = 1.37$ percent
        <!-- check: pch14.h0_ladder_relerr_pct = 1.37 ± 0.01 -->.
        The CMB: $0.5/67.4 = 0.74$ percent
        <!-- check: pch14.h0_cmb_relerr_pct = 0.74 ± 0.01 -->.
        The gap: $5.6/67.4 = 8.31$ percent
        <!-- check: pch14.percent_difference = 8.31 ± 0.01 -->.
        In units of each camp's own precision, that gap is $6.07$
        <!-- check: pch14.gap_vs_ladder_relerr = 6.07 ± 0.01 -->
        and $11.2$
        <!-- check: pch14.gap_vs_cmb_relerr = 11.2 ± 0.01 -->
        respectively. The sigma tells you the disagreement is significant; this
        ratio tells you what the sigma hides — *how far outside each camp's own
        competence claim* the missing systematic would have to live. Whoever is
        wrong is not wrong by a little. They are wrong by many times the
        precision they publish, on the quantity they have spent careers
        measuring.

??? question "Exercise 14.3 — What the repo's number is doing"
    This repository fixes $H_0 = 70.0$ km/s/Mpc. Compute the midpoint of the
    two camps and compare. Then answer the harder question: is choosing $70$
    a scientific decision, an evasion, or neither?

    ??? success "Solution"
        The midpoint is $(73.0 + 67.4)/2 = 70.2$
        <!-- check: pch14.h0_midpoint = 70.2 ± 0.01 -->,
        and the repo asserts $70.0$
        <!-- check: pch14.repo_h0 = 70.0 ± 0.01 -->
        — a fifth of a unit below dead centre, and a value no published
        measurement claims. It is neither, for the reason the main guide's
        Ch. 14 argues: this program's money number comes from a likelihood that
        runs entirely in angles and never multiplies by a distance, so $H_0$ has
        no path into it. When a parameter cannot affect the result, picking a
        round number near the middle of a fight you are not in is correct
        engineering with no scientific content at all. It *would* be an evasion
        in a program that used $H_0$ to set a physical scale — which is why the
        guide enumerates the three files where it does.

??? question "Exercise 14.4 — What disqualifies a third method"
    A "third method" is only useful if it is independent of both existing ones.
    For each of the following, say whether it would break the tie, and why or
    why not: (a) a new supernova sample twice the size, calibrated on the same
    Cepheids; (b) a better CMB map from a different satellite; (c) a
    time-delay measurement of a lensed quasar; (d) that same time-delay
    measurement, where the lens's mass profile was calibrated against a
    Cepheid-based distance.

    ??? success "Solution"
        (a) No. More supernovae shrink the statistical error on a rung that was
        never the suspect; the systematic under suspicion lives in the Cepheid
        calibration, which the new sample inherits unchanged. Doubling $N$ on a
        biased estimator buys a tighter wrong answer.

        (b) No, for the mirror reason. A better map re-measures the same angle
        and feeds it through the same $\Lambda$CDM extrapolation. If the model
        is what is broken, a perfect map does not help — and this has been done,
        by more than one experiment, all agreeing with Planck.

        (c) Yes, in principle: no Cepheid, no supernova calibration, no
        assumption about the universe before the light left. It can be wrong,
        but not *in either of the two ways currently under suspicion* — the
        whole point of an independent method. Its own vulnerability is separate,
        and it is the mass-sheet degeneracy
        ([Ch. 21](../guide/21-degeneracies.md#the-mass-sheet-degeneracy)).

        (d) No, and this is the trap. The moment the lens model's scale is
        pinned by anything standing on a ladder rung, the method inherits the
        ladder's systematic and stops being a third opinion — it becomes a
        correlated restatement of the first. Independence is a property of the
        whole inference chain, not of the headline technique, and auditing that
        chain for a smuggled dependency is the same discipline as checking that
        no validation feature leaked in from training.
