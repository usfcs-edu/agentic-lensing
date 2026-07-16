# 13. Dark energy and how it ends

[Chapter 11](11-big-bang.md#running-it-backwards) ran the expansion backwards
to a hot, dense start. [Chapter 12](12-dark-matter.md#four-independent-witnesses)
established that most of the matter in that universe is not the kind that
shines. This chapter adds the result neither prepares you for: matter — dark and
ordinary together — is not most of what the universe contains, and the expansion
is not slowing down. It is speeding up, and has been for the last
<!-- check: pch13.lookback_at_crossover_gyr = 3.390 ± 0.01 -->$3.39$ billion
years. By the end you will know what the second argument of
`FlatLambdaCDM(H0=70, Om0=0.3)` — the one line this repository's cosmology
reduces to — asserts about the universe, and where the $0.7$ that `astropy`
computes without your typing it came from.

!!! abstract "What you can skip"
    Nothing here needs general relativity, or the *second* Friedmann equation —
    the one with pressure in it, which is where accelerated expansion formally
    comes from. [Ch. 14 of the main guide](../guide/14-frw.md#friedmann) gets
    the first out of Newtonian energy conservation and explains, in its
    Exercise 14.4, why the second has no Newtonian version. That is the whole
    calculus content of this subject, and this chapter hands it over. If you
    already hold the concordance budget and can say why a constant
    $\rho_\Lambda$ plus $\rho_m \propto a^{-3}$ *forces* a crossover, read only
    [The energy budget](#the-energy-budget), for what this repository's
    rounding of $\Omega_m$ costs.

## The supernova surprise { #the-supernova-surprise }

By the mid-1990s the open question was not *whether* the expansion was slowing
down. Gravity is attractive; a universe full of matter pulls on itself; the
expansion had to be decelerating. The question was by how much — the same
question as how much matter there is. Two teams set out to measure the
deceleration with the tool [Ch. 9](09-distance-ladder.md#supernovae) built:
Type Ia supernovae as standard candles, bright enough to see halfway across the
observable universe and uniform enough to read a distance off.

The measurement needs two independent numbers per supernova. The **redshift**
says how much the universe has expanded since the light left
([Ch. 8](08-redshift.md#the-cosmological-picture)); the **brightness** says how
far the light actually travelled. Different expansion histories relate those two
differently: a universe that decelerated hard was expanding *faster* in the
past, so a photon emitted at a given redshift had less far to travel, and its
source looks nearer and brighter. One whose expansion was slower in the past
puts that same source farther away, and fainter.

Quantify it with this repository's own cosmology, at its standard lens redshift,
$z=0.5$. The repo's `FlatLambdaCDM(H0=70, Om0=0.3)` puts a supernova there at a
luminosity distance of

<!-- check: pch13.dl_actual_z05_mpc = 2832.9 ± 0.1 -->

$$
d_L = 2832.9\ \text{Mpc}.
$$

Now build the universe everyone expected: same $H_0$, flat, all matter and no
dark energy at all, $\Omega_m = 1$ — the model the field called Einstein–de
Sitter and used as its default prior. The same supernova lands at

<!-- check: pch13.dl_decel_only_z05_mpc = 2357.7 ± 0.1 -->

$$
d_L = 2357.7\ \text{Mpc},
$$

<!-- check: pch13.dl_ratio_z05 = 1.2016 ± 0.001 -->

a factor of $1.2016$ nearer. Convert that ratio into the units a telescope
reports. The magnitude system
([Ch. 9 of the main guide](../guide/09-units.md#magnitudes)) is logarithmic in
flux, and flux goes as $1/d^2$, so two distances differ in magnitude by
$\Delta m = 5\log_{10}(d_1/d_2)$:

<!-- check: pch13.mag_diff_z05 = 0.399 ± 0.001 -->

$$
\Delta m = 5\log_{10}\!\left(\frac{2832.9}{2357.7}\right) = 0.399\ \text{mag}.
$$

**Those four tenths of a magnitude are the discovery.** The distant supernovae
came in fainter than a decelerating universe allowed by about that much —
<!-- check: pch13.dl_ratio_z05 = 1.2016 ± 0.001 -->roughly $20\%$ farther than
they had any business being. Both teams reported it in 1998, independently, on
different pipelines. There is no subtler content to the result: the candles were
too dim, and the only way to make them too dim is to have the expansion speed
up.

Two honest caveats, because the above is the *shape* of the argument in the
repo's numbers, not a rerun of the published fits. First, the real analyses fit
$\Omega_m$ and $\Omega_\Lambda$ jointly across many redshifts rather than
comparing two fixed models, and the pre-1998 prior was never as clean as
$\Omega_m = 1$. Second, and more to the point: four tenths of a magnitude is
*not* a large number. It is about what one miscalibrated rung of the distance
ladder could produce by itself
([Ch. 9](09-distance-ladder.md#errors-compound) is where
errors compound), and the alternatives — distant supernovae being intrinsically
dimmer, dust greying them — took years to kill. What settled it was the CMB
([Ch. 11](11-big-bang.md#what-the-cmb-tells-us)) and galaxy clustering
independently landing on the same budget, from physics with nothing to do with a
light curve. Three unrelated measurements agreeing is the argument; four tenths
of a magnitude on its own never was.

## What Lambda is { #what-lambda-is }

What makes the expansion accelerate is written $\Lambda$ — the **cosmological
constant** — and its definition is one property, not a mechanism: an energy
density belonging to space itself that *does not dilute as space expands*.
Matter dilutes. Double every distance and the same atoms occupy eight times the
volume, so $\rho_m \propto a^{-3}$. Double every distance with $\Lambda$ in the
room and there is more $\Lambda$, at the density it always had:
$\rho_\Lambda = \text{const}$. That is the entire model, and
[Ch. 14 of the main guide](../guide/14-frw.md#friedmann) states it in one line —
"treat the cosmological constant $\Lambda$ as an extra density component that
never dilutes" — before moving on.

!!! tip "You already know this"
    Two terms with different growth rates cross exactly once. You reason this
    way whenever you compare an $O(n^3)$ term against an $O(1)$ term: the
    constants set *where* the crossover is, never *whether*, and which one wins
    is a question about the regime. Here $\rho_m \propto a^{-3}$
    races $\rho_\Lambda \propto a^{0}$, so at small $a$ matter dominates by an
    unbounded margin and at large $a$ it loses, guaranteed. The crossover epoch
    was not fitted to the 1998 data; it is forced the moment you write two
    components with different exponents. The data only fixed *when*.

The mechanism, as opposed to the bookkeeping, is where general relativity enters
and this book stops. In GR the source of gravity is density plus three times
pressure, so a component with large *negative* pressure gravitates
*repulsively*. The main guide says this exactly once, in the solution to
Exercise 14.4, where a negative-pressure component — "exactly what makes
something behave like a cosmological constant" — can drive "accelerated
expansion — with no Newtonian counterpart at all".

**What nobody knows.** "Dark energy" labels the observation; it does not explain
it, exactly as "dark matter" does not
([Ch. 12](12-dark-matter.md#what-cdm-stands-for)). The natural candidate — the
vacuum energy of quantum fields — gives the right *kind* of thing, then
overshoots the observed value by many tens of orders of magnitude, depending on
how you count. Calling $\Lambda$ "vacuum energy" in company is a claim, not a
synonym. Whether $\Lambda$ is constant is also open: `FlatLambdaCDM` hard-codes
the constant-density case, and recent large spectroscopic surveys report hints
that the dark-energy density has evolved — at a significance nobody yet calls a
detection, and which may not survive.

## The energy budget { #the-energy-budget }

Divide every energy density in the universe today by the critical density
$\rho_{\mathrm{crit},0}$ that [Ch. 14 of the main
guide](../guide/14-frw.md#the-density-parameters) builds out of $H_0$ alone. The
dimensionless fractions that come back are the **density parameters**, and they
are the answer to "what is the universe made of":

<!-- check: pch13.omega_baryon = 0.049 ± 0.001 -->
<!-- check: pch13.omega_dark_matter = 0.265 ± 0.001 -->
<!-- check: pch13.omega_lambda = 0.685 ± 0.001 -->

- **Ordinary matter** (atoms — "baryons", in the field's loose usage):
  $\Omega_b = 0.049$.
- **Dark matter**: $\Omega_c = 0.265$.
- **Dark energy**: $\Omega_\Lambda = 0.685$.

<figure markdown="span">
  ![Donut chart of the cosmic energy budget: dark energy 68.5 percent, dark matter 26.5 percent, ordinary matter 4.9 percent](figures/p13-energy-budget-light.svg#only-light){ width="80%" }
  ![Donut chart of the cosmic energy budget: dark energy 68.5 percent, dark matter 26.5 percent, ordinary matter 4.9 percent](figures/p13-energy-budget-dark.svg#only-dark){ width="80%" }
  <figcaption markdown="span">**Figure 13.1.** The concordance budget, drawn to
  scale. The thin gold wedge at the top is everything made of atoms; the green
  wedge is the dark matter of [Ch. 12](12-dark-matter.md#halos), the only other
  thing that clusters and dilutes; the purple two-thirds is $\Lambda$, a density
  with no known constituents, inferred entirely from what it does to the
  expansion. `FlatLambdaCDM(H0=70, Om0=0.3)` asserts the purple wedge without
  your typing it, and does not tell the gold from the green at all.</figcaption>
</figure>

That gold sliver contains this book's entire ruler. Every star in the Milky Way
([Ch. 1](01-scale-ladder.md#our-galaxy)), all the gas between them, Andromeda,
every galaxy in every image ever taken and every atom of the telescope that took
it sit inside the
<!-- check: pch13.omega_baryon = 0.049 ± 0.001 -->$4.9\%$ — and not even most of
it, since the majority of ordinary matter is thin intergalactic gas that was
never part of a galaxy. Everything astronomy spent its first three centuries
cataloguing is a rounding error on the contents of the universe.

<!-- check: pch13.omega_ratio_today = 2.1815 ± 0.001 -->

Dark energy today outweighs all matter, dark and ordinary, by a factor of
$2.1815$. And the three numbers sum to

<!-- check: pch13.omega_sum = 0.999 ± 0.001 -->

$$
\Omega_b + \Omega_c + \Omega_\Lambda = 0.999,
$$

not $1.000$. Read nothing into the last digit: these are rounded measured
values, and the missing thousandth is the rounding, not a measured curvature.
Flatness — $\Omega_k = 0$ — is a separate empirical claim, which the main
guide's Exercise 14.2 is precisely about not confusing with the identity
$\Omega_m + \Omega_\Lambda + \Omega_k = 1$, true in any universe.

**What `Om0=0.3` actually asserts.** The object takes one matter parameter, not
two, because the expansion history cannot tell baryons from dark matter: both
dilute as $a^{-3}$, so only their sum enters. That sum is
$\Omega_m = 0.049 + 0.265 = 0.314$. The repo types $0.3$. Rounding the matter
down rounds $\Omega_\Lambda$ up to $0.7$, and moves the one epoch this chapter
cares about — the crossover where $\Lambda$ overtakes matter, from setting
$\Omega_m(1+z)^3 = \Omega_\Lambda$ — from

<!-- check: pch13.z_matter_lambda_equality = 0.297 ± 0.001 -->

$$
z = 0.297
$$

to

<!-- check: pch13.z_cross_repo_om03 = 0.326 ± 0.001 -->

$$
z = 0.326.
$$

Less matter means $\Lambda$ wins sooner, so the repo's universe starts
accelerating slightly *earlier* than the measured budget says. That shift is
real and costs this program nothing: cosmology enters this repository in three
files, none of which asks when the expansion changed gears, and the money number
does not know what $\Omega_m$ is. Name it anyway — it tells you where a rounded
cosmology *would* bite. Two further
omissions in the same object: `site/guide_src/cosmo.py` builds it with `Tcmb0`
at its default, so it contains no radiation at all (harmless at $z \le 2$, wrong
at the CMB), and it carries no baryon fraction, so Figure 13.1's
gold-versus-green split is invisible to it.

## How it ends { #how-it-ends }

Matter keeps diluting and $\Lambda$ does not, so run the same two-term race
forward. In $H(z)^2 = H_0^2\left[\Omega_m(1+z)^3 + \Omega_\Lambda\right]$ — the
main guide's own [Friedmann equation](../guide/14-frw.md#the-density-parameters),
evaluated at a redshift — the matter term goes to zero as the universe expands,
leaving a constant:

<!-- check: pch13.h_de_sitter_kms_mpc = 57.935 ± 0.01 -->

$$
H \longrightarrow H_0\sqrt{\Omega_\Lambda} = 70\sqrt{0.685} = 57.935\
\text{km/s/Mpc}.
$$

Read that carefully: it contains the trap. $H$ is $70$ today and
$57.935$ in the far future: the Hubble parameter is **decreasing**,
monotonically, from here on — while the expansion **accelerates**. The two are
not in tension, because $H$ is a *fractional* growth rate, not a speed.

!!! tip "You already know this"
    This is compound interest, and the confusion dissolves once you say it that
    way. $H$ is an interest rate in units of inverse time. A balance whose rate
    is easing toward a floor of
    <!-- check: pch13.h_de_sitter_kms_mpc = 57.935 ± 0.01 -->$57.935$ still pays
    more interest every year than the last, once the balance has grown enough to
    outrun the falling rate — the separation between two galaxies grows by more
    each year while growing by a smaller *percentage* each year. And a rate that
    stops falling — a *constant* $H$ — is no more a static universe than a fixed
    APR is a static balance: constant fractional growth is exponential growth,
    $a \propto e^{Ht}$. Where it breaks: a bank credits interest from somewhere.
    Nothing is deposited here, and no galaxy moves through space to earn it. The
    identity is in the arithmetic of growth, not the mechanism.

**When the gears changed.** The crossover at
<!-- check: pch13.z_matter_lambda_equality = 0.297 ± 0.001 -->$z = 0.297$ was,
in cosmic time,

<!-- check: pch13.age_at_crossover_gyr = 10.077 ± 0.01 -->
<!-- check: pch13.universe_age_gyr = 13.467 ± 0.01 -->

$10.077$ billion years after the beginning, out of the $13.467$ billion years
the same `FlatLambdaCDM(70, 0.3)` gives for the age of the universe — the
identical number the main guide's Ch. 14 quotes. So the universe spent

<!-- check: pch13.fraction_of_age_decelerating = 0.748 ± 0.001 -->
<!-- check: pch13.fraction_of_age_accelerating = 0.252 ± 0.001 -->

$74.8\%$ of its life so far slowing down, exactly as everyone before 1998
assumed it always had, and only the most recent $25.2\%$ speeding up.

That is itself an open problem, and deserves naming. We live near the
crossover: the $\Lambda$-to-matter ratio is of order one *right now*, at
<!-- check: pch13.omega_ratio_today = 2.1815 ± 0.001 -->$2.1815$, having been
arbitrarily small for most of history and headed for arbitrarily large. Whether
that is coincidence, selection (the era in which
galaxies exist to ask the question need not be uniformly distributed), or a hint
that $\Lambda$ is not constant, nobody knows. It is called the coincidence
problem; it is unsolved.

**How it ends,** given a constant $\Lambda$: exponential expansion, forever.
Galaxies not already bound to us recede faster and faster, redshift toward
invisibility, and cross out of view. This does *not* tear the Milky Way apart,
and that is where the expansion picture must not be over-read: $\Lambda$'s
density is fixed and small, while inside a bound structure the local matter
density is overwhelmingly larger, so the Galaxy, the Local Group and every
cluster of [Ch. 4](04-clusters.md#groups-and-clusters) stay bound and drop out
of the Hubble flow entirely. The Milky Way and Andromeda will merge; that merger
is not in a race with $\Lambda$. What is lost is everything else — and with it,
the evidence: an astronomer in that merged Local Group would have no distant
supernovae and no CMB left to detect, and no way to discover any of this
chapter. That holds if $\Lambda$ is exactly constant. If its density grows,
bound structures do come apart, and nobody has ruled that out.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 14 — FRW and the Friedmann equations](../guide/14-frw.md#friedmann)**
      instructs you to "treat the cosmological constant $\Lambda$ as an extra
      density component that never dilutes", and never says what $\Lambda$ is,
      who measured it, or that anyone was surprised. You now have the
      measurement: <!-- check: pch13.mag_diff_z05 = 0.399 ± 0.001 -->$0.399$
      magnitudes of unexpected faintness in a $z=0.5$ Type Ia, in 1998, and the
      reason it took the CMB to make it stick.
    - **[Ch. 14's density parameters](../guide/14-frw.md#the-density-parameters)**
      says `FlatLambdaCDM(70, 0.3)` "asserts three numbers, only one of which
      you typed", and that $\Omega_{\Lambda,0}$ "is not independently set, it is
      *computed*, as $1 - \Omega_{m,0} - \Omega_{k,0} = 0.7$". That $0.7$ is not
      an API convenience: it is a measured
      <!-- check: pch13.omega_lambda = 0.685 ± 0.001 -->$0.685$, rounded,
      asserting that two thirds of the energy density of the universe has no
      known constituents. The typed $0.3$ likewise rounds $0.049 + 0.265$,
      moving the crossover from
      <!-- check: pch13.z_matter_lambda_equality = 0.297 ± 0.001 -->$z = 0.297$
      to <!-- check: pch13.z_cross_repo_om03 = 0.326 ± 0.001 -->$z = 0.326$
      while touching nothing this repository computes.
    - **Ch. 14's Exercises 14.3 and 14.4** are where this chapter's subject
      surfaces in the main guide, unnamed. 14.3 says $H$ nearly doubles between
      $z = 0.5$ and $z = 1$ because "the matter term $\Omega_m(1+z)^3$ is
      starting to dominate over the constant $\Omega_\Lambda$ term as you look
      back in time" — that sentence *is* the crossover, and the guide never says
      which side of it we are on. We are on the $\Lambda$ side, by
      <!-- check: pch13.omega_ratio_today = 2.1815 ± 0.001 -->$2.1815$ to one,
      for the most recent
      <!-- check: pch13.fraction_of_age_accelerating = 0.252 ± 0.001 -->$25.2\%$
      of the age it quotes. 14.4 is the only place accelerated expansion appears
      at all, and you can now read it as a description of the actual universe,
      not a hypothetical.

## Exercises { #exercises }

??? question "Exercise 13.1 — The whole discovery, in one division"
    A $z = 0.5$ Type Ia sits at $d_L = 2832.9$ Mpc in
    `FlatLambdaCDM(H0=70, Om0=0.3)` and at $d_L = 2357.7$ Mpc in a flat,
    matter-only universe with the same $H_0$. Using
    $\Delta m = 5\log_{10}(d_1/d_2)$, compute how much fainter the supernova is
    in the accelerating universe. Then: why did a result of that size need two
    other kinds of measurement before anyone believed it?

    ??? success "Solution"
        <!-- check: pch13.dl_ratio_z05 = 1.2016 ± 0.001 -->
        The distance ratio is $2832.9/2357.7 = 1.2016$, so

        $$
        \Delta m = 5\log_{10}(1.2016) = 0.399\ \text{mag}.
        $$

        <!-- check: pch13.mag_diff_z05 = 0.399 ± 0.001 -->

        Because every rival explanation makes the same prediction. A bad rung
        on the ladder, evolution in what a Type Ia *is* at earlier times, and
        grey dust all say "distant supernovae look too faint", and no amount of
        care with supernovae separates them. Only an independent systematic can,
        which is what the CMB and galaxy clustering supplied.

??? question "Exercise 13.2 — Where the crossover has to be"
    $\Lambda$'s density is constant; matter's falls as $(1+z)^3$. Set them
    equal — $\Omega_m(1+z)^3 = \Omega_\Lambda$ — and solve for $z$, first with
    the measured budget ($\Omega_m = 0.049 + 0.265$, $\Omega_\Lambda = 0.685$),
    then with what this repository actually types ($0.3$ and $0.7$). Which
    direction does the rounding push the crossover, and why is that the sign
    you should have expected?

    ??? success "Solution"
        Rearranging gives $z = (\Omega_\Lambda/\Omega_m)^{1/3} - 1$. Measured:
        $(0.685/0.314)^{1/3} - 1 = 0.297$.

        <!-- check: pch13.z_matter_lambda_equality = 0.297 ± 0.001 -->

        Repo: $(0.7/0.3)^{1/3} - 1 = 0.326$.

        <!-- check: pch13.z_cross_repo_om03 = 0.326 ± 0.001 -->

        The repo's rounding removes matter and adds $\Lambda$, so $\Lambda$ wins
        sooner — a higher crossover redshift, meaning earlier in cosmic time.
        The sign is forced: less to slow the expansion and more to speed it up
        can only move the switch earlier. Note also how weakly the answer
        depends on the inputs, because the cube root flattens everything: nobody
        needs $\Omega_m$ to three digits to know which side of the crossover we
        are on.

??? question "Exercise 13.3 — How much of the universe's life was the surprise?"
    The crossover at $z = 0.297$ happened when the universe was $10.077$ billion
    years old; its present age in the same cosmology is $13.467$ billion years.
    What fraction of cosmic history was spent decelerating, and how long ago, in
    lookback time, did the switch happen? Then: why does that number make
    $z = 0.5$ a well-chosen place to have looked?

    ??? success "Solution"
        $10.077 / 13.467 = 0.748$, so $74.8\%$ of the universe's life so far was
        spent slowing down and only $25.2\%$ speeding up.

        <!-- check: pch13.age_at_crossover_gyr = 10.077 ± 0.01 -->
        <!-- check: pch13.universe_age_gyr = 13.467 ± 0.01 -->
        <!-- check: pch13.fraction_of_age_decelerating = 0.748 ± 0.001 -->
        <!-- check: pch13.fraction_of_age_accelerating = 0.252 ± 0.001 -->

        The switch is at lookback time $13.467 - 10.077 = 3.390$ billion years.

        <!-- check: pch13.lookback_at_crossover_gyr = 3.390 ± 0.01 -->

        $z = 0.5$ is comfortably beyond the crossover, so a supernova there
        carries light from the decelerating era *and* the accelerating one, and
        its distance accumulates the difference between the two histories. A
        survey confined within $z \approx 0.3$ samples a single era, with far
        less leverage to separate the models. Reaching past the crossover is
        what made the effect visible at all, and the Type Ia is the candle
        bright enough to reach.
