# 15. Gravity bends light

Every arc, ring and multiple image this program hunts is one physical fact
applied at galactic scale: mass bends light, by a specific and computable
amount. The main guide's
[Ch. 16](../guide/16-deflection.md#newtonian-deflection) derives that amount —
$\alpha = 4GM/c^2b$ — in about a page, using an integral this book does not
have. This chapter's job is to make that page worth reading: what the claim
physically says, why the coefficient is *twice* the Newtonian one rather than
approximately twice, and how a single factor of two in a single amplitude became
one of the most famous measurements in astronomy. The derivation belongs to
Ch. 16. The reason to care belongs here.

!!! abstract "What you can skip"
    You do not need general relativity, geodesics, the metric, or the integral
    that produces the coefficient — the main guide does that integral and this
    book never will. Take "mass curves spacetime and light follows the curve" as
    the claim and read on for what it buys. If you already know that GR predicts
    exactly twice the Newtonian deflection and that a 1919 eclipse was the test,
    skip to [Why a factor of two mattered](#why-a-factor-of-two-mattered) — the
    honest version of what 1919 established is less than the story you have
    probably been told.

## The idea { #the-idea }

Start with Newton, who has two answers, and both matter.

The first is zero. Newtonian gravity is a force between masses, $F = GMm/r^2$,
and light has no mass. Put $m = 0$ in and nothing happens: light passes a star
in a straight line and no eclipse is worth mounting. That was a real position,
not a strawman — it is the first thing the formula says.

The second answer is not zero, and comes from noticing that $m$ cancels. Divide
the force by the mass being pulled and you get an *acceleration*, $a = GM/r^2$,
with no reference to the falling body at all: a feather and a cannonball fall
alike. In the eighteenth-century picture light was a stream of tiny corpuscles
travelling at speed $c$, and corpuscles are bodies, and bodies fall. Henry
Cavendish did this calculation privately around 1784 and Johann Georg von
Soldner published it in 1801; both got $\alpha = 2GM/c^2b$, where $b$ is the
**impact parameter** — the closest approach the undeflected ray would have made
to the mass's centre. For light grazing the Sun's visible edge, that is

<!-- check: pch15.newtonian_deflection_arcsec = 0.876 ± 0.001 -->

$$
\alpha_{\mathrm N} = \frac{2GM_\odot}{c^2 R_\odot} \approx 0.876'',
$$

an arcsecond being $1/3600$ of a degree, the unit
[Ch. 9 of the main guide](../guide/09-units.md#angles-on-the-sky) builds.

Now Einstein, whose claim is not "gravity pulls on light harder than Newton
thought". It is a different sentence in a different vocabulary. Mass changes the
*geometry* of spacetime, and light travels the straightest available path
through whatever geometry it finds. Nothing pulls on the ray; the ray does not
turn. It goes straight, and "straight" in a geometry that is not flat does not
look straight to someone drawing flat pictures from far away. The word "bends",
in this chapter's title and in every paper you will read, describes what happens
to the ruler, not to the light.

**Where the analogy breaks, and it does break.** You can carry "the Sun tugs the
photon sideways" around as an intuition, and it will get you the right
functional form and half the right amplitude — which is exactly why it is
dangerous. The Newtonian answer is not a rough approximation that GR nudges; it
is precisely half, for a reason worth holding onto. In the weak-field limit mass
warps two things: the flow of time, and space itself. A slowly moving object
weights the spatial part by something of order $(v/c)^2$ and never notices it,
which is why Newtonian mechanics, built for slow particles, is right about
planets and captures only the time half. Light, at $v = c$, feels both halves
equally — two equal contributions, one of which Newtonian physics has no
vocabulary for. So

<!-- check: pch15.sun_deflection_arcsec = 1.752 ± 0.001 -->

$$
\alpha_{\mathrm{GR}} = \frac{4GM_\odot}{c^2 R_\odot} \approx 1.752'',
$$

and the ratio of the two predictions is

<!-- check: pch15.factor = 2 ± 1e-9 -->

$$
\frac{\alpha_{\mathrm{GR}}}{\alpha_{\mathrm N}} = 2
$$

exactly: the second contribution *equals* the first, it does not resemble it.
[Ch. 16](../guide/16-deflection.md#the-factor-of-two) writes the weak-field
metric down and shows the two terms contributing equally, which is where that
"exactly" is earned. One caveat: both formulas are leading-order results,
dropping corrections suppressed by a further factor of $GM/(c^2b)$ that is
negligible at the Sun's limb. The exactness lives inside a stated limit, not in
the sky.

**How much bending is on offer.** $1.752''$ is the largest gravitational
deflection anywhere in the solar system, and not by a close margin: the Sun
holds essentially all the local mass, and its limb is the nearest a sightline
can be brought to that mass without passing through it. To do better you need
more mass, and the next rung up is a galaxy — where this book's ruler earns its
keep. A Milky Way holds roughly $10^{11}$ stars
([Ch. 1](01-scale-ladder.md#our-galaxy)), so you might expect $10^{11}$ times
the bending. You do not get it: the ray does not graze a stellar surface, and
the impact parameter grows by very nearly the same factor the mass does. The
deflection sees only the combination $M/b$, the two enormous factors cancel, and
a whole galaxy bends light by something in the neighbourhood of the Sun's own
$1.752''$ — [Exercise 15.3](#exercises) is that arithmetic. This
near-cancellation, and nothing deeper, is why every Einstein radius in this
program is around an arcsecond rather than a microarcsecond or a degree.

Treat that as an order-of-magnitude count, not a calculation: a real lens galaxy
is heavier than the Milky Way, holds most of its mass as dark matter rather than
stars ([Ch. 12](12-dark-matter.md#halos)), and — being extended rather than a
point — only puts the mass *inside* the impact parameter to work.
[Ch. 16](16-what-is-a-strong-lens.md#what-you-see) gives the real number.

## Eddington 1919 { #eddington-1919 }

Starlight passing close to the Sun arrives in company with sunlight, and the
daytime sky beside the Sun is overwhelmingly brighter than any star in it. A
total eclipse removes the sunlight for a few minutes and leaves the star field
standing. That is the entire reason this is an eclipse experiment, and why the
test had to wait for a date the sky picked.

The measurement is differential, which is where the difficulty lives. You
photograph the star field around the eclipsed Sun; months later, at night, with
the Sun elsewhere on the sky, you photograph the same field through the same
instrument; then you subtract the two sets of plate positions. If light bends,
the stars nearest the limb will have shifted outward during the eclipse —
radially away from the Sun — by an amount falling off as $1/b$.

29 May 1919 was chosen because the eclipsed Sun would sit in front of the
Hyades, a cluster dense enough to put several usable stars near the limb; the
eclipses either side of it offered a nearly empty field. The Astronomer Royal,
Frank Dyson, organised two expeditions against the weather: Arthur Eddington to
Príncipe, off West Africa, and Andrew Crommelin and Charles Davidson to Sobral,
in Brazil. The number that went into the textbooks came from Sobral's 4-inch
telescope:

<!-- check: pch15.eddington_measured = 1.98 ± 0.001 -->
<!-- check: pch15.eddington_error = 0.16 ± 0.001 -->

$$
\alpha_{\mathrm{measured}} = 1.98'' \pm 0.16''.
$$

That is not the only number the expeditions produced, and here the popular
telling stops. Sobral's *other* instrument, the astrographic telescope, was the
primary one; its mirror appears to have distorted in the heat of the day, its
images came out defocused, and its plates gave a deflection close to the
Newtonian value. Those plates were set aside. Príncipe was clouded out
through most of totality; Eddington came away with a handful of usable plates
whose value was lower than Sobral's and whose error bar was far wider. Taken by
itself, Príncipe excluded nothing.

Was setting the astrographic plates aside legitimate? Most likely yes: the
defocus was a real fault with an identified physical cause rather than a
post-hoc rationalisation, later re-analyses of those plates support the
decision, and a century of better measurements agrees with the answer the
surviving data gave. But the cut was made after the numbers were in hand, by
people who could see which hypothesis each subset favoured and who had a
preferred answer, with no pre-registered criterion to point at. That is a fact
about the result, not an accusation against anyone in it — and it is the sort of
thing you would flag in a colleague's ablation table, so there is no reason to
give it a pass here.

## Why a factor of two mattered { #why-a-factor-of-two-mattered }

Look at what the two theories actually disagree about. Newton's corpuscular
prediction is $2GM/c^2b$; Einstein's is $4GM/c^2b$. Same variables, same linear
dependence on $M$, same $1/b$ falloff: the functional form is *identical*, and
the entire disagreement between two theories of gravity is one multiplicative
constant.

That is why the test was hard, and the difficulty should be familiar. Had the
two predictions differed in *shape* — a different power of $b$, a dependence on
wavelength, anything — you could have measured a spread of stars at a spread of
impact parameters and let the shape decide, no absolute calibration required,
because a shape survives a wrong scale factor. They do not differ in shape.
Divide one by the other and every variable cancels: the ratio is
$2$<!-- check: pch15.factor = 2 ± 1e-9 --> for every $b$, every $M$, every star
on the plate, so on a log axis the two predictions are parallel lines offset by
a constant $\log 2$. No curve fit separates them; only an absolute amplitude
does.

!!! tip "You already know this"
    A constant that multiplies every prediction equally is perfectly confounded
    with any multiplicative error in your instrument. The plate scale — how many
    arcseconds one millimetre of emulsion stands for — enters the measurement in
    the same place the physics does, so a plate scale wrong by $2$ and a theory
    wrong by $2$ produce identical data. The information you need is not in the
    shape of the curve; it is in the units of the axis. Eddington's real
    adversary was not Newton. It was the possibility that his telescope had
    changed its own units between Brazil in daylight and Brazil at night.

And the amplitude on offer is small. The gap between the two theories — the
whole content of the argument, the entire reason for two ships and a year of
preparation — is
$0.876''$<!-- check: pch15.newtonian_deflection_arcsec = 0.876 ± 0.001 -->, the
Newtonian prediction over again, as it must be once the ratio is $2$. Under one
arcsecond, to be resolved on a plate exposed for a few seconds through a
telescope shipped to the tropics, against a comparison plate taken months
earlier at a different temperature.

<figure markdown="span">
  ![Newtonian and Einsteinian deflection versus impact parameter in solar radii: two 1/b curves, the Einstein one twice the Newton one at every impact parameter, with the gap between them shaded and Sobral's 1919 measurement plotted at the Sun's limb](figures/p15-deflection-light.svg#only-light){ width="90%" }
  ![Newtonian and Einsteinian deflection versus impact parameter in solar radii: two 1/b curves, the Einstein one twice the Newton one at every impact parameter, with the gap between them shaded and Sobral's 1919 measurement plotted at the Sun's limb](figures/p15-deflection-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 15.1.** Both theories give the same $1/b$
  falloff, so the two curves have one shape and stand everywhere in the ratio
  $2$<!-- check: pch15.factor = 2 ± 1e-9 -->: no fit to the *shape* tells them
  apart, and the shaded region is the whole argument. These axes are linear, so
  the constant *ratio* is not a constant gap: the curves converge as $b$ grows,
  and only on a log axis would they be parallel. The gap is widest where the
  mass is closest, at the Sun's limb ($b = 1\,R_\odot$), and even there it is
  Newton's
  $0.876''$<!-- check: pch15.newtonian_deflection_arcsec = 0.876 ± 0.001 -->
  against Einstein's
  $1.752''$<!-- check: pch15.sun_deflection_arcsec = 1.752 ± 0.001 -->. The point
  with the error bar is Sobral's 4-inch result of 29 May 1919,
  $1.98'' \pm 0.16''$
  <!-- check: pch15.eddington_measured = 1.98 ± 0.001 -->
  <!-- check: pch15.eddington_error = 0.16 ± 0.001 -->: far above the dashed
  line, and a little above the solid one — the honest shape of the
  result.</figcaption>
</figure>

Now the arithmetic a century of retelling usually skips. Against Newton:

<!-- check: pch15.newton_sigma_off = 6.9 ± 0.01 -->

$$
\frac{1.98'' - 0.876''}{0.16''} \approx 6.9,
$$

standard deviations. Under the Newtonian hypothesis that is not a fluctuation
anyone has to entertain. Newton was excluded — that is the claim 1919 actually
established.

Now the same subtraction against Einstein's $1.752''$ — carefully, because this
is where the story turns into a victory parade. The gap is much smaller, but it
is not zero: Sobral's central value sits above the GR prediction by more than one
standard deviation and less than two, which in the language you would use on your
own results is *consistent with* general relativity and nothing stronger.
Príncipe, lower and wider still, excluded neither theory on its own.

So the honest summary is: **1919 excluded Newton. It did not confirm Einstein to
any precision.** Those are different statements, and it was the second that made
the newspapers. Modern confidence in $4GM/c^2b$ does not rest on those plates and
never should have — radio interferometry measures the same deflection for quasars
near the Sun with no eclipse required, the glare not being a radio problem, and
space astrometry has since pinned the coefficient far beyond anything 1919 could
approach. The eclipse was decisive about Newton, and it was famous, and those are
not the same fact.

**Why this program needs the factor of two.** The coefficient does not stay in
this chapter. [Ch. 16 of the main guide](../guide/16-deflection.md#the-thin-lens)
carries the $4$ — as a $2/c^2$ in the definition of the lensing potential — into
$\Sigma_{\mathrm{cr}}$, then into the convergence $\kappa$, then into
$\theta_{\mathrm E}$, the Einstein radius every strong-lensing paper quotes in
its abstract. Its Exercise 16.4 works out what follows from using the Newtonian
amount instead: $\kappa$ reads at half strength for the same real mass, and
every mass this program measures comes out a factor of $2$ wrong. That is what
the eclipse bought — not a story about a physicist becoming famous, but one
multiplicative constant that every mass in the field is still divided by.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 16 — How much light bends, and the factor of two](../guide/16-deflection.md#the-factor-of-two)**
      does the derivation this chapter refused to do, and does it well — but it
      assumes you already know *why anyone should care about a factor of two*.
      It says the ratio is "exactly $2$ — nothing lets it be anything else,
      since $\alpha_{\mathrm{GR}}$ was *defined* as double $\alpha_{\mathrm N}$
      above. The physics is in which of the two numbers … the sky actually
      shows", and then settles that question in a single clause. Its whole
      account of the *result* is that Dyson, Eddington and Davidson "reported
      results close to the GR prediction and clearly separated from the
      Newtonian one" — no number, no error bar, no mention that the primary
      instrument's plates were cut. You now have
      $1.98'' \pm 0.16''$<!-- check: pch15.eddington_measured = 1.98 ± 0.001 -->,
      excluding Newton by
      $6.9\sigma$<!-- check: pch15.newton_sigma_off = 6.9 ± 0.01 --> while
      agreeing with Einstein only to within its own wide error bar — so you can
      read that sentence as the compression it is.
    - Its **"You already know this"** box says the eclipse "had three sharply
      separated candidates to rule between", and its **skip box** offers to send
      you straight to
      [The thin-lens approximation](../guide/16-deflection.md#the-thin-lens) "if
      you already know the GR light-bending result and where the factor of two
      comes from". You have all three candidates — zero from putting $m = 0$ into
      $F = GMm/r^2$, the Newtonian value from treating light as an ordinary
      falling body, the GR value from letting space curve as well as time — so
      that skip is a real option, and the thin-lens section — the machinery
      every later chapter leans on — is where your attention should go.

## Exercises { #exercises }

??? question "Exercise 15.1 — Why the verdict is asymmetric"
    The text computes that $1.98'' \pm 0.16''$ sits
    $6.9\sigma$<!-- check: pch15.newton_sigma_off = 6.9 ± 0.01 --> from Newton.
    Do the same subtraction against Einstein's $1.752''$, in error-bar units,
    and explain why "excluded Newton" and "confirmed Einstein" are not two
    readings of one result.

    ??? success "Solution"
        The gap to Einstein is more than one error bar and less than two — so
        the measurement is *consistent with* GR, which is the weakest thing a
        measurement can say about a hypothesis it does not reject. Exclusion and
        confirmation are asymmetric because rejection needs only distance, while
        confirmation needs precision, and this measurement had distance without
        precision: it kills Newton outright and pins nothing.

??? question "Exercise 15.2 — Why no curve fit could have settled it"
    Both predictions have the form $\alpha = kGM/c^2b$, differing only in
    $k \in \{2, 4\}$. Suppose you measure many stars at many impact parameters,
    but your plate scale — arcseconds per millimetre of emulsion — carries an
    unknown multiplicative error $s$. Show that the shape of the falloff tells
    you nothing about $k$, and name the quantity you must know to have an
    experiment at all.

    ??? success "Solution"
        What you record is $s \cdot kGM/(c^2b)$: every point is the true
        prediction times the constant $s$, so the *shape* — a $1/b$ falloff,
        slope $-1$ on log-log axes — is identical for $k = 2$, $k = 4$, and any
        $s$ whatever. Fitting the shape recovers only the product $sk$, and
        $s = 1, k = 4$ is indistinguishable from $s = 2, k = 2$: the theories
        disagree by $2$<!-- check: pch15.factor = 2 ± 1e-9 -->, a pure number
        with no $b$ in it, and $s$ is a pure number with no $b$ in it too. What
        you must know independently is the plate scale — the absolute
        calibration of your angular axis. Hence the *differential* design: the
        night-time comparison plate is the calibration, and every worry the
        expeditions had (mirror temperature, focus, emulsion shrinkage) is a
        worry about $s$ drifting between the two exposures, not about physics.

??? question "Exercise 15.3 — Why a galaxy bends light about as much as the Sun"
    A Milky Way holds about $10^{11}$ stars, so about $10^{11}$ times the Sun's
    mass. Why does it not bend light $10^{11}$ times as far as the Sun's limb
    does — and what does the answer say about why every Einstein radius in this
    field is quoted in arcseconds?

    ??? success "Solution"
        The deflection sees $M$ and $b$ only through $M/b$, and $b$ grows by
        nearly the factor $M$ does. A lensed ray passes tens of thousands of
        light-years from a galaxy's centre while an eclipse ray grazes the Sun's
        surface — and the Sun's radius is a couple of light-seconds, while ten
        thousand light-years is a few times $10^{11}$ light-seconds. Numerator
        and denominator both go up by roughly $10^{11}$; they cancel to within
        an order of magnitude; the galaxy lands within a small factor of
        $1.752''$<!-- check: pch15.sun_deflection_arcsec = 1.752 ± 0.001 -->.
        Two enormous numbers dividing out is the whole reason strong lensing is
        an arcsecond-scale phenomenon, and therefore the reason it is
        *measurable* at all.
