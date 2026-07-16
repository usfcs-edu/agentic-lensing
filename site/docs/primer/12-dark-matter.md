# 12. Dark matter

[Chapter 3](03-galaxies.md#what-a-galaxy-is) and
[Chapter 4](04-clusters.md#groups-and-clusters) handed you galaxies and
clusters as objects you can photograph. This chapter is about the part of them
you cannot. Four measurements — a spiral disk, a cluster, the light-bending
this whole program is built around, and a picture of the universe taken before
any galaxy existed — share no instrument, no epoch and no systematics, and
report the same thing: most of the matter does not shine. By the end you can
expand the name `FlatLambdaCDM` a letter at a time, and know which of its
assertions the field can defend and which one it cannot.

!!! abstract "What you can skip"
    You do not need particle physics — nobody has a particle model that works
    here, which is the honest state of the subject. You do not need the N-body
    simulations behind the halo profile named below, only that it is a *fit* to
    simulation output rather than a derived law. You do not need to get $M({<}r)$
    out of $\rho(r)$ by integrating:
    [Ch. 10 of the main guide](../guide/10-galaxies.md#velocity-dispersion)
    does it in three lines. What is left is the evidence, and the evidence is
    arithmetic.

## Rotation curves { #rotation-curves }

Newton, applied to something on a circular orbit: gravity supplies the
centripetal force, so

$$
\frac{G\,M({<}r)\,m}{r^2} = \frac{m\,v^2}{r}
\qquad\Longrightarrow\qquad
v(r) = \sqrt{\frac{G\,M({<}r)}{r}}.
$$

The test mass $m$ cancels, and — by the shell theorem — only the mass
*interior* to the orbit appears. In the solar system essentially all the mass
is the Sun, so $M({<}r)$ is the same constant for every planet and
$v \propto 1/\sqrt{r}$: the outer planets crawl.

A spiral galaxy is not a point mass. Its light is spread through a disk, so
moving outward you enclose more stars, $M({<}r)$ grows, and $v$ rises. Then the
light runs out. Past the outermost stars $M({<}r)$ stops growing and you are back
in the solar-system case: $v$ should turn over and fall as $1/\sqrt{r}$. That
is a prediction with no free parameters — photograph the disk, count the light,
assume the mass is where the light is, and the curve is determined.

That prediction is crude in one identified way, and it is worth naming before
it is used against anything. The shell theorem is exact for a *sphere*, and a
disk is not one: matter outside the orbit does not cancel, and doing the disk
properly lifts the predicted outer speed some tens of percent above the
spherical reading. Tens of percent is not the discrepancy this chapter is
about, which is why the crude version survives — but the curve below is the
crude version.

It is wrong. Measure orbital speeds far enough out, using the 21-cm radio line
from neutral hydrogen that extends well past the last visible star, and the
curve does not turn over. It goes flat and stays flat to the last measurable
point. Vera Rubin and Kent Ford established this through the 1970s, galaxy
after galaxy.

<figure markdown="span">
  ![Orbital speed versus radius for a model spiral galaxy: the measured curve stays flat at 200 km/s past the visible disk edge while the curve predicted from the light alone falls away, with the gap between them shaded as the missing mass](figures/p12-rotation-curve-light.svg#only-light){ width="90%" }
  ![Orbital speed versus radius for a model spiral galaxy: the measured curve stays flat at 200 km/s past the visible disk edge while the curve predicted from the light alone falls away, with the gap between them shaded as the missing mass](figures/p12-rotation-curve-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 12.1.** A model spiral: orbital speed against
  radius, with the visible disk ending at 12 kpc (dotted line). Inside that
  radius the two curves are *identical* — there is no disagreement where the
  light is. Outside it, the curve predicted from the light alone falls as
  $1/\sqrt{r}$ while the measured curve stays flat near 200 km/s, and the
  shaded gap is mass that has to be there and cannot be seen. A schematic, not
  data: a real disk's inner rise has more structure than this smooth ramp, and
  no real galaxy is flat to four significant figures. The argument lives
  entirely in the outer half.</figcaption>
</figure>

Read the flat part back through the equation. If $v$ is constant then
$M({<}r) \propto r$: mass keeps piling up, linearly, past the radius where the
light stopped. The main guide derives the density that does this —
$\rho \propto r^{-2}$ — from the opposite direction, as the profile of a
constant-velocity-dispersion sphere, and notes in passing that it "gives you a
disk galaxy's flat rotation curve for free"
([Ch. 10](../guide/10-galaxies.md#velocity-dispersion)). Same fact, two
kinematic windows.

There are exactly two ways out. Either there is mass out there you cannot see,
or $F = GMm/r^2$ itself fails at the very low accelerations of a galaxy's
outskirts. The second is a real research program — Modified Newtonian Dynamics,
MOND — and should not be waved away: on rotation curves MOND is *good*,
arguably tighter than the halo picture, predicting the curve from the visible
light with one universal constant instead of one fitted halo per galaxy. What
sinks it for most of the field is the other three witnesses.

## Four independent witnesses { #four-independent-witnesses }

Rotation curves alone would be a suspicious result: one class of object, one
technique, one equilibrium assumption, and a proposed fix touching the
least-tested regime of the least-tested law in the argument. Three more
measurements say the same thing from places that share nothing with it.

| Witness | Scale | Epoch | What it weighs |
|---|---|---|---|
| Rotation curves | one spiral disk, tens of kpc | now | $M({<}r)$ from orbital speeds |
| Cluster dynamics | a cluster, several Mpc | now | total mass from the spread of galaxy velocities |
| Gravitational lensing | galaxy through cluster | halfway back | projected mass, with no equilibrium assumed |
| The CMB | the whole sky | before any galaxy | ordinary and dark matter, *separately* |

**Cluster dynamics.** Fritz Zwicky measured how fast Coma's galaxies moved
relative to each other in 1933. Far too fast: a cluster is a bound system, the
visible mass could not have held those galaxies, and Coma should have flown
apart long before he looked at it. He named the discrepancy *dunkle Materie* —
dark matter — and was ignored for forty years, until rotation curves reached
the same conclusion from a different kind of object entirely.

**Lensing.** Your own field, and the cleanest of the four, because it is the
only one that does not assume equilibrium: the deflection depends on the mass
and nothing else ([Ch. 15](15-gravity-bends-light.md#the-idea)). The sharpest
result is the Bullet Cluster, two clusters that passed through each other.
Their hot gas — the majority of the *ordinary* matter, well outnumbering the
stars — collided, shocked and lagged behind, while their galaxies, effectively
collisionless at those densities, sailed through. The lensing mass went with
the galaxies, not the gas: mass and the bulk of visible matter sit in
physically separate places on the sky. Changing the law of gravity does not
help, because a modified law still needs a *source*, and the source is
demonstrably not where the light is.

**The CMB.** Before atoms formed, ordinary matter was ionised and locked to
photons by scattering; the two behaved as one fluid with real pressure, and
pressure means sound waves. Dark matter does not scatter light, so it felt none
of that pressure — it fell into whatever gravitational wells existed and
stayed. Ordinary matter therefore sets the compressions and rarefactions in the
temperature pattern of [the oldest light](11-big-bang.md#the-cmb), dark matter
sets the depth of the wells they oscillate in, and the relative peak heights
pull the two apart ([Ch. 11](11-big-bang.md#what-the-cmb-tells-us)) — each as a
fraction of the density needed for flatness, the $\rho_{\mathrm{crit}}$ of
[Ch. 14](../guide/14-frw.md#friedmann):

<!-- check: pch12.omega_baryon = 0.049 ± 0.001 -->
<!-- check: pch12.omega_dark_matter = 0.265 ± 0.001 -->

$$
\Omega_{\mathrm{b}} = 0.049,
\qquad
\Omega_{\mathrm{dm}} = 0.265.
$$

Divide them:

<!-- check: pch12.dm_to_baryon_ratio = 5.41 ± 0.01 -->

$$
\frac{\Omega_{\mathrm{dm}}}{\Omega_{\mathrm{b}}} = 5.41.
$$

About five and a half times as much dark matter as there are atoms in every
star, planet, dust grain and gas cloud in the universe combined. The Milky
Way's $10^{11}$ stars ([Ch. 1](01-scale-ladder.md#our-galaxy)), and every other
galaxy's, are a minority component of the matter, and a small one.

!!! tip "You already know this"
    The force of this section is not any one measurement's precision — several
    carry systematics you would not accept in a paper on their own. It is that
    the four error models are *uncorrelated*: 21-cm radio astronomy plus an
    equilibrium assumption; optical spectroscopy plus the virial theorem;
    general relativity plus image deconvolution; microwave photometry plus
    linear perturbation theory. A shared bias would have to sit in all four and
    pull them the same way. This is why you ensemble models with decorrelated
    errors rather than four seeds of one architecture, and why a factor of five
    no single method could establish to your satisfaction is nevertheless not
    in doubt.

## Halos { #halos }

The picture that survives all four witnesses: the disk you photograph is the
small bright thing at the centre of a much larger, roughly spherical, invisible
**halo**. The galaxy sits *in* it, and it keeps going for several times the
radius of the visible light. Everything in
[Chapter 3](03-galaxies.md#what-a-galaxy-is) — the $10^{11}$ stars, the
$10^{5}$ light-years of disk — is the ornament. The halo is the object.

Everyone models a halo's density with the **NFW profile**, after Navarro, Frenk
and White. One property, stated without calculus: it is shallower than $r^{-2}$
near the centre and steeper far outside, crossing that slope once in between.
One property about its *status*: it is not derived from anything, but fitted to
what N-body simulations of cold dark matter produce. When
[Ch. 10 of the main guide](../guide/10-galaxies.md#the-isothermal-conspiracy)
sets up the isothermal conspiracy — the puzzle that a Sersic stellar profile
plus an NFW halo, neither isothermal, sum to a total mass profile that *is* —
the NFW half of that puzzle is this paragraph, and the conspiracy is two fitted
forms adding to a third for reasons nobody has derived.

One warning: do not carry the
$5.41$<!-- check: pch12.dm_to_baryon_ratio = 5.41 ± 0.01 --> into a single
galaxy and say the Milky Way has five times more dark matter than stars. That is a global average over the
whole universe. What any one galaxy reports depends on where you stop
measuring, because the stars stop and the halo does not — which is not a defect
of the measurement but what "the light ends and the mass does not" means
quantitatively.

That aperture dependence is your problem directly. Strong lensing weighs the
mass inside the Einstein radius and nothing else
([Ch. 19](../guide/19-einstein-radius.md#mass-inside-theta-e)), and what it
weighs there is stars *and* halo summed, inseparable from the image. Every mass
model in this program — every $\gamma$, every convergence map
([Ch. 20](../guide/20-profiles.md#the-epl-and-gamma)) — models the total, and
the total is mostly dark. That is not a caveat on the program; it is much of
why it is worth running, because lensing weighs mass whether or not it emits
([Ch. 16](16-what-is-a-strong-lens.md#why-anyone-cares)).

## What CDM stands for { #what-cdm-stands-for }

In [Ch. 14](../guide/14-frw.md#the-density-parameters) of the main guide you
will type this line, or read it in three of this repository's files:

```python
FlatLambdaCDM(H0=70, Om0=0.3)
```

Nobody expands the name. Here it is, a piece at a time.

**Flat** asserts $\Omega_k = 0$: Euclidean geometry on large scales. The main
guide is careful and correct here, and `FlatLambdaCDM` does not measure this —
it assumes it, by refusing to expose a curvature argument.

**Lambda** is $\Lambda$, the cosmological constant: dark energy, the thing
making the expansion accelerate, and a completely different mystery from this
chapter's ([Ch. 13](13-dark-energy.md#the-energy-budget)). Its share,

<!-- check: pch12.omega_lambda = 0.685 ± 0.001 -->

$$
\Omega_\Lambda = 0.685,
$$

dwarfs both matter components. Dark matter and dark energy share an adjective
and nothing else; do not let the naming suggest otherwise.

**CDM** is **Cold Dark Matter**, and both words are load-bearing.

*Dark* does not mean black. Soot is black: it absorbs light, which is a way of
interacting with it. Dark matter does not absorb, emit, scatter or reflect — it
does not couple to electromagnetism at all, as far as anyone can measure.
Transparent is closer, and even that misleads, because glass at least refracts.
It has never been observed to do anything except gravitate.

*Cold* means non-relativistic at the epoch when structure was forming — not a
temperature but a statement about the particles' velocity dispersion, and a
falsifiable claim that has so far survived its tests. Were dark matter *hot*
(light,
fast particles, which is what neutrinos would give you) it would stream out of
small overdensities before they could collapse, erasing small-scale structure
and forcing the largest structures to form first and fragment downward. Cold
dark matter does the opposite: small halos form first and merge upward. The
universe we observe is built bottom-up, which rules *hot* out. Be careful how
much further you take that: it does not pin the temperature, it bounds it. An
intermediate *warm* candidate — heavy enough to keep the bottom-up ordering,
light enough to erase the smallest halos — is constrained rather than excluded,
and the count of faint satellite galaxies around rulers like the Milky Way is
where that bound is currently argued. `CDM` is the cold end of a range the data
allow, not a measured value.

!!! tip "You already know this"
    Free-streaming is a low-pass filter: the distance fast particles travel
    before structure collapses is a smoothing kernel on the initial density
    field, erasing power below it. "Hot" and "cold" name the cutoff
    wavenumber, and the observable is the surviving power spectrum of galaxies
    and of the CMB — precisely the object
    [Ch. 7](../guide/07-fourier.md#psd-and-autocorrelation) has you reading as
    a PSD. A detected cutoff would falsify the `C` and leave `Flat` and
    `Lambda` untouched: the three letters are separately testable, not a
    package you take or leave whole.

**And what `Om0=0.3` contains.** $\Omega_m$ is *all* matter — ordinary and
dark, summed — so the measured total, and the invisible part of it, are

<!-- check: pch12.omega_matter_total = 0.314 ± 0.001 -->
<!-- check: pch12.invisible_fraction_of_om0 = 0.844 ± 0.001 -->

$$
\Omega_m = 0.049 + 0.265 = 0.314,
\qquad
\frac{\Omega_{\mathrm{dm}}}{\Omega_m} = 0.844.
$$

The argument you type as `0.3` — a rounded stand-in for $0.314$ — is
<!-- check: pch12.invisible_fraction_of_om0 = 0.844 ± 0.001 -->84 percent
stuff nobody has identified. The main guide walks through what that line
asserts with real care, down to noting that $\Omega_\Lambda$ is computed rather
than supplied, and never mentions that its *other* argument is
<!-- check: pch12.dm_to_baryon_ratio = 5.41 ± 0.01 -->five and a half parts
dark matter to one part atoms.

**And the sum does not come to one:**

<!-- check: pch12.sum_check = 0.999 ± 0.001 -->

$$
\Omega_{\mathrm{b}} + \Omega_{\mathrm{dm}} + \Omega_\Lambda
= 0.049 + 0.265 + 0.685 = 0.999.
$$

Not $1.000$ — but this is not a curvature detection and not a mistake. It is
three measured values, each rounded to three decimals, being added.
`FlatLambdaCDM` sidesteps this by computing $\Omega_\Lambda$ as whatever makes
the sum exactly one. The reason to print the $0.001$ rather than quietly nudge
a digit is that a budget summing to exactly one when its components are quoted
to three places is a budget somebody fudged.

**Nobody knows what it is.** All four witnesses are gravitational. Not one has
ever seen a dark matter particle — they have seen mass, and inferred that the
mass is not made of the stuff we know. WIMPs led for decades and a generation
of underground detectors returned nothing; axions are the current favourite and
are likewise undetected; primordial black holes are largely excluded across
most of their proposed mass range. The answer may be a particle nobody has
thought of, and it remains possible — a minority view, but a live one — that
some of the discrepancy is gravity behaving in a way we have not written down.
What is *not* in doubt is the observation: gravitating mass exists in
quantities the light cannot account for, at four scales, by four methods, at a
ratio of $5.41$<!-- check: pch12.dm_to_baryon_ratio = 5.41 ± 0.01 --> to one.
"Dark matter" names a robust measurement, not an understood substance.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 10 — Galaxies, Sersic profiles, and velocity dispersion](../guide/10-galaxies.md#the-isothermal-conspiracy)**
      contains the main guide's entire treatment of dark matter, and it is one
      subordinate clause: *"The dark matter — inferred, historically, from
      exactly the flat rotation curves derived above staying flat well past the
      edge of a disk galaxy's visible light, which under Newtonian gravity
      requires unseen mass out there — is typically modeled with an NFW
      profile."* That clause compresses this chapter's first section into a
      dash-bounded aside, then names a profile it never explains. You now have
      the rotation curves it gestures at, the three witnesses it does not
      mention, and the standing of NFW as a fit to simulations — which is what
      makes the isothermal conspiracy a conspiracy rather than a theorem.
    - **[Ch. 14 — FRW and the Friedmann equations](../guide/14-frw.md#the-density-parameters)**
      promises you can "write down `FlatLambdaCDM(H0=70, Om0=0.3)` … and say
      exactly what each of its two arguments asserts, including the third
      number it fixes that you never typed." It delivers on $H_0$, on
      $\Omega_k = 0$, and on $\Omega_\Lambda$ being computed — and never expands
      the letters `CDM`, nor opens up $\Omega_m$. You now know the argument
      typed as `0.3` is $0.314$ measured, that $0.844$ of it is invisible, and
      that `CDM` is a falsifiable claim about a free-streaming cutoff which the
      universe's bottom-up structure confirmed.

## Exercises { #exercises }

??? question "Exercise 12.1 — The budget you type without reading"
    From $\Omega_{\mathrm{b}} = 0.049$, $\Omega_{\mathrm{dm}} = 0.265$ and
    $\Omega_\Lambda = 0.685$: compute the dark-matter-to-ordinary-matter ratio,
    the total $\Omega_m$, the invisible fraction of it, and the sum of all
    three components. Why is the last answer not $1.000$, and why is that not
    evidence of curvature?

    ??? success "Solution"
        Two divisions and two additions:

        $$
        \frac{0.265}{0.049} = 5.41,
        \qquad
        0.049 + 0.265 = 0.314,
        $$

        <!-- check: pch12.dm_to_baryon_ratio = 5.41 ± 0.01 -->
        <!-- check: pch12.omega_matter_total = 0.314 ± 0.001 -->

        $$
        \frac{0.265}{0.314} = 0.844,
        \qquad
        0.049 + 0.265 + 0.685 = 0.999.
        $$

        <!-- check: pch12.invisible_fraction_of_om0 = 0.844 ± 0.001 -->
        <!-- check: pch12.sum_check = 0.999 ± 0.001 -->
        The shortfall is rounding: three measured values quoted to three
        decimals, added, give a rounded sum, and $0.001$ is the error to
        expect. Curvature is a separate empirical claim, $\Omega_k = 0$, tested
        against its own error bar and not by this addition. `FlatLambdaCDM`
        sidesteps the issue by computing $\Omega_\Lambda = 1 - \Omega_m$ rather
        than accepting a measured value, which is why the code's budget sums to
        one exactly and the observations' does not.

??? question "Exercise 12.2 — What the light alone predicts"
    Take the model galaxy of Figure 12.1. Outside the visible disk, $M({<}r)$ is
    constant if the mass is where the light is, so $v \propto r^{-1/2}$. By what
    factor should the orbital speed drop between the edge of the visible disk
    and a radius four times larger? What does the measured curve do instead,
    and what does that imply about $M({<}r)$ out there?

    ??? success "Solution"
        With $M({<}r)$ fixed, $v(4r)/v(r) = 4^{-1/2} = 1/2$: the speed should
        halve. The measured curve does not drop at all, so $v(4r)/v(r) = 1$,
        and holding $v$ constant while $r$ quadruples requires $M({<}r)$ to
        quadruple too — mass still accumulating, linearly, in a region with no
        light in it. That is the same statement as $\rho \propto r^{-2}$, the
        profile [Ch. 10](../guide/10-galaxies.md#velocity-dispersion) of the
        main guide derives from the constant-velocity-dispersion condition:
        reached there through stellar kinematics in an elliptical, here through
        gas orbits in a spiral.

??? question "Exercise 12.3 — Why four, and not one"
    Rotation curves can be fitted by MOND without any dark matter, and fitted
    well. Explain why the field nonetheless treats dark matter as established,
    and what specifically a modified-gravity theory would have to reproduce to
    displace it.

    ??? success "Solution"
        Because the claim does not rest on rotation curves. Cluster dynamics,
        lensing and the CMB reach the same conclusion at different scales and
        epochs with disjoint systematics; a bias explaining away all four would
        have to sit in radio astronomy, optical spectroscopy, image
        deconvolution and microwave photometry at once, and pull them the same
        direction. So a modified-gravity theory must do more than beat dark
        matter on one witness: it must fit all four at least as well with no
        more freedom, and in particular explain the Bullet Cluster, where a
        modified force law still needs a source and the source is not
        co-located with the light. Nothing has cleared that bar. Note what the
        argument does *not* do: it does not say what dark matter is made of,
        and it does not make MOND's rotation-curve success go away as something
        a complete theory should eventually explain.
