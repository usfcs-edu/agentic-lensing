# 4. Clusters and the cosmic web

The main guide quotes an Einstein radius in three different chapters and never
joins them up. [Ch. 9](../guide/09-units.md#angles-on-the-sky) says a lensed
ring is of order an arcsecond across.
[Ch. 10](../guide/10-galaxies.md#velocity-dispersion) computes
$\theta_{\mathrm{E}} = 1.145''$
<!-- check: pch04.theta_e_galaxy = 1.145 ± 0.001 -->
for its fiducial massive elliptical.
[Ch. 15](../guide/15-distances.md#the-carousel) computes
$\theta_{\mathrm{E}} = 13.03''$
<!-- check: pch04.theta_e_carousel = 13.03 ± 0.001 -->
for the Carousel and describes it, in passing, as "a cluster-scale strong
lens" — without ever saying what a cluster is, or why its ring is ten times
bigger than the elliptical's. That factor of ten is neither a coincidence nor
anything deep. A cluster has roughly a hundred times the mass inside its ring,
and the ring grows as the *square root* of that mass. This chapter supplies
what sits between [Chapter 3](03-galaxies.md)'s galaxy and that square root:
what a cluster is, what it is made of, what larger structure it sits in, and
the one line of arithmetic that closes the gap the guide leaves open.

!!! abstract "What you can skip"
    You do not need the virial theorem, the physics of the hot gas between
    cluster galaxies, or anything about how structure grows out of an initial
    density field. None of it is used downstream in this program, and each is
    a research field of its own. You also do not need to derive
    $M(<\theta_{\mathrm{E}}) = \Sigma_{\mathrm{cr}}\,\pi\,\ell_{\mathrm{E}}^2$
    — [Ch. 19](../guide/19-einstein-radius.md#mass-inside-theta-e) does that,
    and this chapter uses only its *shape*. If you already accept that the mass
    inside a ring scales as the ring's radius squared, the third section is one
    square root: read the figure and move on.

## Groups and clusters { #groups-and-clusters }

Galaxies are not sprinkled at random. Gravity has unlimited range and no
opposite sign to cancel it — electric charge comes in two signs and
neutralizes itself over any large volume, so above the scale of a planet
gravity is the only force with anything left to accumulate. Given time, a
near-uniform distribution of matter with slight density excesses pulls itself
into those excesses, and what starts as a ripple ends as a bound object. The
result is a hierarchy, and every rung of it is real and named.

The Milky Way sits in a **group**: itself, Andromeda, and a few dozen much
smaller dwarf galaxies, gravitationally bound and falling toward each other.
Two large galaxies and a crowd of small ones. Groups are the common case —
most galaxies live in one.

A **cluster** is the next rung and a different animal. Order
$10^{3}$<!-- check: pch04.n_galaxies_cluster = 1000 ± 1 --> galaxies, bound
together in a region a few megaparsecs across, with a total mass between

<!-- check: pch04.cluster_mass_low_msun = 1e14 ± 1e11 -->
<!-- check: pch04.cluster_mass_high_msun = 1e15 ± 1e12 -->

$$
10^{14}\ M_\odot \quad\text{and}\quad 10^{15}\ M_\odot .
$$

Against this book's ruler — one Milky Way, total mass including its dark halo,
which is most of it ([Ch. 12](12-dark-matter.md#halos)) — that range is

<!-- check: pch04.cluster_low_in_mw = 66.7 ± 0.05 -->
<!-- check: pch04.cluster_high_in_mw = 667 ± 0.5 -->

$$
66.7 \quad\text{to}\quad 667 \ \text{Milky Ways}.
$$

Both numbers deserve the same caveat: neither is a measurement of any
particular object. "About a thousand galaxies" and "$10^{14}$–$10^{15}\
M_\odot$" are order-of-magnitude statements about a population with real
scatter, and a cluster's boundary is a convention, not a physical edge — the
density falls off smoothly, so the galaxy count depends on where you stop
counting. Groups and clusters are not two species. They are two regions of one
continuum, and the objects in between get called "poor clusters" or "rich
groups" depending on the author.

The second thing worth fixing early: those thousand galaxies are a *minority*
of the mass. The hot, X-ray-emitting gas filling the space between them
outweighs every star in every one of those galaxies combined, and dark matter
outweighs the gas by a wide margin again. When you read "a cluster of
galaxies," the galaxies are the part that happens to be visible, not the part
that does the gravitating. That is the same statement
[Chapter 12](12-dark-matter.md#four-independent-witnesses) will make from four
independent directions; cluster dynamics is one of the four, and historically
it was the first.

## The cosmic web { #the-cosmic-web }

Zoom out past the cluster and the hierarchy does not continue into
ever-larger blobs. It changes character. Map the positions of a million
galaxies — which is what a redshift survey does, and what
[Chapter 6](06-telescopes.md#what-a-survey-is) is about — and what appears is
not a uniform scatter and not a set of nested spheres. It is a **web**:
clusters sitting at the *nodes*, long thin **filaments** of galaxies running
between them, flatter **sheets** spanning the filaments, and, occupying most
of the volume, **voids**.

A void is a roughly spherical region with very few galaxies in it, and its
characteristic size is about

<!-- check: pch04.void_scale_mpc = 100 ± 0.1 -->

$$
100\ \mathrm{Mpc}.
$$

Put that against the ruler. The Milky Way's disk is

<!-- check: pch04.mw_diam_mpc = 0.03066 ± 0.0001 -->

$$
0.03066\ \mathrm{Mpc}
$$

across, so one void spans

<!-- check: pch04.void_in_mw_diameters = 3262 ± 1 -->

$$
\frac{100}{0.03066} \approx 3262 \ \text{Milky Way diameters}
$$

of mostly nothing. Two honest corrections to that sentence. First, "very few
galaxies" is not "empty": voids hold dwarf galaxies and diffuse gas at much
lower density than the filaments, and a void's interior is a live research
topic rather than a blank. Second, $100$ Mpc is a *characteristic* scale —
voids come in a distribution of sizes, and no single void was measured to
produce that number.

The web is not an artist's impression. It is what surveys measure, and it is
also what you get if you take a nearly uniform early universe with tiny
density fluctuations and let gravity run: overdense regions get denser and
drain the underdense ones, fastest along whichever axis collapses first —
which produces sheets, then filaments, then nodes, in that order. Simulations
reproduce the observed web from initial conditions constrained by
[Chapter 11](11-big-bang.md#the-cmb)'s picture of the early universe. That
agreement is one of cosmology's strongest results, and it is *statistical*:
the simulations reproduce the web's characteristic scales and connectivity,
not the position of any particular filament.

One consequence matters for the main guide. The web means the universe is
lumpy on the scale of $100$ Mpc — that lumpiness is what a void's size
records. Keep zooming out, though, and past a few hundred Mpc the lumps
average away and one large volume looks statistically like any other. That is
what makes [Ch. 14](../guide/14-frw.md#the-frw-metric)'s FRW metric legal: it
assumes a homogeneous, isotropic universe, which is *false* at the scale of
this section and true only above it. The main guide states the cosmological
principle as holding "on large scales" and stops there — it never says how
large, which is the only part of the claim with any content. The answer is:
larger than the web.

## Why cluster rings are bigger { #why-cluster-rings-are-bigger }

Now the arithmetic the guide leaves out.
[Ch. 19](../guide/19-einstein-radius.md#mass-inside-theta-e) proves a fact
this chapter will use and not re-derive: the mass inside the Einstein ring is

$$
M(<\theta_{\mathrm{E}}) = \Sigma_{\mathrm{cr}}\,\pi\,\ell_{\mathrm{E}}^2 ,
$$

where $\ell_{\mathrm{E}}$ is the ring's *physical* radius and
$\Sigma_{\mathrm{cr}}$ is a reference surface density fixed entirely by the
lens and source distances ([Ch. 15](../guide/15-distances.md#sigma-crit)).
Read only the shape: a constant density times the area of a circle. The
physical radius is proportional to the angle, $\ell_{\mathrm{E}} \propto
\theta_{\mathrm{E}}$, so with the distances held fixed,

$$
M \propto \theta_{\mathrm{E}}^2
\qquad\Longleftrightarrow\qquad
\theta_{\mathrm{E}} \propto \sqrt{M}.
$$

That is the whole explanation. A hundred times the mass buys ten times the
ring, because ten is the square root of a hundred. No calculus, no general
relativity — the area of a circle, and one assumption ($\Sigma_{\mathrm{cr}}$
fixed) that the rest of this section will make you pay for.

Run it on the guide's own two numbers. The ratio of the rings is

<!-- check: pch04.theta_e_ratio_cluster_to_galaxy = 11.38 ± 0.005 -->

$$
\frac{13.03''}{1.145''} = 11.38 ,
$$

so the mass ratio the rings *imply* is that squared:

<!-- check: pch04.implied_mass_ratio = 129.5 ± 0.05 -->

$$
11.38^2 = 129.5 .
$$

The Carousel has roughly $130$ times the mass inside its ring that the
fiducial elliptical has inside its own. Going back the other way,
$\sqrt{129.5} = 11.38$, recovers the ring ratio exactly — and that exactness
is not evidence of anything. It is one relation read in two directions. The
content is the exponent $2$, not the arithmetic; the arithmetic cannot fail.

Use the ruler on the mass itself. [Ch. 15](../guide/15-distances.md#the-carousel)
computes, from two redshifts and nothing else,

<!-- check: pch04.carousel_mass_msun = 4.621e13 ± 1e11 -->

$$
M(<\theta_{\mathrm{E}}) \approx 4.621\times10^{13}\ M_\odot
$$

and then says nothing at all about whether that is a lot, because the main
guide has no Milky Way in it to compare against. Divide by one — the Milky
Way's total mass, dark halo included, which this book takes as
$1.5\times10^{12}\ M_\odot$ — and it is

<!-- check: pch04.carousel_in_milky_ways = 30.8 ± 0.05 -->

$$
30.8 \ \text{Milky Ways}
$$

packed inside a circle whose physical radius
[Ch. 15](../guide/15-distances.md#the-carousel) works out to a small fraction
of a megaparsec. Read that third digit as arithmetic, not precision: the halo
has no edge ([Ch. 12](12-dark-matter.md#halos)), so our Galaxy's own total
depends on where you stop counting, and published estimates spread over
roughly a factor of two. Every "in Milky Ways" figure here inherits that. The
ruler is good for orders of magnitude, which is all this chapter asks of it.

<figure markdown="span">
  ![Log-log plot of Einstein radius against mass inside the ring, a straight line of slope one half, with a massive elliptical, the Milky Way's mass, and the Carousel cluster marked on it](figures/p04-theta-e-ladder-light.svg#only-light){ width="90%" }
  ![Log-log plot of Einstein radius against mass inside the ring, a straight line of slope one half, with a massive elliptical, the Milky Way's mass, and the Carousel cluster marked on it](figures/p04-theta-e-ladder-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 4.1.** The relation
  $\theta_{\mathrm{E}} \propto \sqrt{M}$, anchored on the Carousel's measured
  pair, with the main guide's two Einstein radii marked on it about two orders
  of magnitude in mass apart. The Milky Way's point is labelled *mass only*: it
  marks where a Milky-Way-mass deflector would land on this line, not a ring
  anyone has measured — we sit inside our own Galaxy, which is the wrong side
  of it to be lensed by. The elliptical's abscissa is not an independent
  measurement either; it is the Carousel's mass divided by the mass ratio the
  two rings imply, which is why that point sits on the line by construction
  rather than by agreement.</figcaption>
</figure>

!!! tip "You already know this"
    Figure 4.1 is a scaling plot, and it reads exactly like a
    runtime-versus-input-size plot: a power law is a straight line on log-log
    axes, and the exponent is the slope. Here the slope is $1/2$. The same
    caution transfers, too, and it is the reason the caption is worded the way
    it is: two points always fit a straight line. The slope in this figure is
    *imposed* by the relation above, and the labelled systems are placed on
    it. The figure illustrates the law. It does not test it, any more than two
    timing runs establish that your algorithm is $O(n\log n)$.

Two things about $4.621\times10^{13}\ M_\odot$ have to be said plainly,
because the main guide says neither and a reader will otherwise walk away with
a wrong number in his head.

**It is the mass inside the ring, not the cluster.** The Carousel's total mass
is not $4.621\times10^{13}\ M_\odot$; a cluster's total is the
$10^{14}$–$10^{15}\ M_\odot$ of the first section, $66.7$ to $667$ Milky Ways.
The ring at $13.03''$ encloses only the innermost part of the cluster, well
inside its few-megaparsec extent — the Einstein radius is a probe of the core,
which is precisely what makes it useful (it is a mass measurement that needs
no assumption about the profile's shape) and precisely what makes it partial.
$30.8$ Milky Ways is the core. How much more the whole cluster weighs is not
something this ring can tell you: getting from the core out to the total needs
a profile, and refusing to assume a profile is exactly what bought the core
measurement its credibility.

**The square root holds at fixed geometry, and the geometry is not fixed
here.** $\Sigma_{\mathrm{cr}}$ depends on the distances to the lens and the
source ([Ch. 15](../guide/15-distances.md#sigma-crit)), and the two systems
are not at the same redshifts: the fiducial elliptical sits at $z_l=0.5$ with
its source at $z_s=2.0$, while the Carousel sits at $z_l=0.49$ with the source
plane used for this measurement at $z_s=1.432$. The lens redshifts nearly
match; the source redshifts do not. So the two $\Sigma_{\mathrm{cr}}$ values
differ by an order-unity factor, and $129.5$ is the mass ratio the rings imply
*if you hold geometry fixed* — close to the true ratio, not equal to it. That
is the honest status of this chapter's headline: a scaling argument, correct
about the exponent and about the order of magnitude, and not a substitute for
[Ch. 15](../guide/15-distances.md#three-distances)'s machinery, which does the
job exactly and is only a few lines longer.

What survives both caveats is the thing the guide never says: the ring is ten
times bigger because the mass is a hundred times bigger, and rings go as the
square root of mass.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 15 — Distances that do not add, and $\Sigma_{\mathrm{cr}}$](../guide/15-distances.md#the-carousel)**
      opens its worked example with "The Carousel Lens … is a cluster-scale
      strong lens at $z_l=0.49$" and lets the phrase *cluster-scale* carry the
      entire argument for why this system's numbers look nothing like
      [Ch. 10](../guide/10-galaxies.md#velocity-dispersion)'s. You now know
      what that phrase asserts: about $10^{3}$ galaxies, $10^{14}$–$10^{15}\
      M_\odot$ in total, $66.7$ to $667$ Milky Ways. The chapter promises "two
      real, checkable numbers" and delivers
      $M(<\theta_{\mathrm{E}}) \approx 4.621\times10^{13}\ M_\odot$ without a
      word on whether that is large. It is $30.8$ Milky Ways — and it is the
      mass inside a $13.03''$ ring, which is the cluster's core and not the
      cluster.
    - **[Ch. 15's $\Sigma_{\mathrm{cr}}$ section](../guide/15-distances.md#sigma-crit)**
      calls $\Sigma_{\mathrm{cr}}$ "the natural unit of surface density" and
      shows it is computable "the moment you have $D_{\mathrm{d}}$,
      $D_{\mathrm{s}}$, and $D_{\mathrm{ds}}$ in hand". What it never says is
      why anyone would want it: because with it,
      $M(<\theta_{\mathrm{E}}) = \Sigma_{\mathrm{cr}}\pi\ell_{\mathrm{E}}^2$
      turns an angle you can measure on an image into a mass, and that formula
      is the whole reason a $13.03''$ ring and a $1.145''$ ring are the same
      physics with $129.5$ times the mass between them.

## Exercises { #exercises }

??? question "Exercise 4.1 — The square root, both directions"
    The main guide gives $\theta_{\mathrm{E}} = 1.145''$ for a fiducial
    massive elliptical and $13.03''$ for the Carousel. Compute the ratio, and
    from it the mass ratio the two rings imply. Then say what the slogan "a
    hundred times the mass buys ten times the ring" gets right and what it
    hides.

    ??? success "Solution"
        The ring ratio is $13.03/1.145 = 11.38$
        <!-- check: pch04.theta_e_ratio_cluster_to_galaxy = 11.38 ± 0.005 -->,
        and since $M \propto \theta_{\mathrm{E}}^2$ at fixed geometry, the
        implied mass ratio is $11.38^2 = 129.5$
        <!-- check: pch04.implied_mass_ratio = 129.5 ± 0.05 -->. The slogan
        gets the exponent right, which is the only part that matters for
        understanding why the two numbers differ by a factor of ten rather
        than a factor of a hundred. It hides two things. First, the rounding:
        the real numbers are $11.38$ and $129.5$, not $10$ and $100$. Second,
        and more seriously, "at fixed geometry" — $\Sigma_{\mathrm{cr}}$
        depends on the lens and source redshifts, and the two systems do not
        share them ($z_s = 2.0$ against $z_s = 1.432$), so $129.5$ is what the
        rings imply under an assumption that is only approximately true here.

??? question "Exercise 4.2 — Is the Carousel a lightweight cluster?"
    Chapter 15 computes $M(<\theta_{\mathrm{E}}) \approx 4.621\times10^{13}\
    M_\odot$ for the Carousel. Convert it to Milky Ways. Compare against this
    chapter's cluster mass range. The comparison appears to say the Carousel is
    below the low end of the cluster range — is it a runt? Answer carefully.

    ??? success "Solution"
        $4.621\times10^{13}\ M_\odot$
        <!-- check: pch04.carousel_mass_msun = 4.621e13 ± 1e11 --> divided by
        a Milky Way is $30.8$
        <!-- check: pch04.carousel_in_milky_ways = 30.8 ± 0.05 -->, against a
        cluster range of $66.7$
        <!-- check: pch04.cluster_low_in_mw = 66.7 ± 0.05 --> to $667$
        <!-- check: pch04.cluster_high_in_mw = 667 ± 0.5 --> Milky Ways. It
        looks like the Carousel falls short of the low end by a factor of
        about two. It does not, and the reason is that the two quantities are
        not the same quantity. $4.621\times10^{13}\ M_\odot$ is the mass
        *inside the Einstein ring* — the mass in a circle of radius $13.03''$
        projected at the lens, which is the cluster's core and a small
        fraction of its few-megaparsec extent. $10^{14}$–$10^{15}\ M_\odot$ is
        a cluster's *total*. Comparing them directly is a category error, and
        the fact that the core mass lands below the total range is exactly what
        you should expect, not evidence about this cluster's rank. Note that the guide's number is not wrong or misleading — it is
        explicitly written $M(<\theta_{\mathrm{E}})$. The trap is entirely in
        reading it as "the cluster's mass".

??? question "Exercise 4.3 — A void against the ruler"
    Voids are about $100$ Mpc across; the Milky Way's disk is about $0.03066$
    Mpc across. How many Milky Way diameters fit across a void? Then explain
    what is wrong with describing that volume as empty.

    ??? success "Solution"
        $100/0.03066 \approx 3262$
        <!-- check: pch04.void_in_mw_diameters = 3262 ± 1 --> Milky Way
        diameters<!-- check: pch04.mw_diam_mpc = 0.03066 ± 0.0001 -->. "Empty"
        is wrong on two counts. Physically, voids are underdense, not vacant:
        they hold dwarf galaxies and diffuse gas, and what lives in them is an
        active research question rather than a settled nothing. Definitionally,
        $100$ Mpc <!-- check: pch04.void_scale_mpc = 100 ± 0.1 --> is a
        characteristic scale drawn from a distribution of void sizes, not a
        measurement of any particular object, so "a void is 3262 Milky Ways
        across" describes a population's typical member and should be quoted
        with that qualifier attached.

??? question "Exercise 4.4 — Why the Milky Way's point says *mass only*"
    Figure 4.1 marks the Milky Way on the $\theta_{\mathrm{E}}$–$M$ line, but
    labels it *mass only* rather than giving it an Einstein radius like the
    other two points. Give two independent reasons why the Milky Way does not
    appear in this program's lens catalogues.

    ??? success "Solution"
        First, geometry: strong lensing needs the deflector *between* you and
        a distant source, and we are inside the Milky Way. The Galaxy's mass is
        not projected onto a sky position we can view from the far side, so
        there is no configuration in which our own Galaxy produces an Einstein
        ring for us. (Individual stars within it do
        produce microlensing, which is a different regime entirely — no
        resolvable ring.) Second, even for an observer well outside it,
        producing a ring requires a background source close to aligned on the
        line of sight and the right distance ratio to make
        $\Sigma_{\mathrm{cr}}$ small enough
        ([Ch. 15](../guide/15-distances.md#sigma-crit)); alignment that good is
        rare, which is [Chapter 16](16-what-is-a-strong-lens.md#why-it-is-rare)'s
        subject. The point on the figure therefore answers a conditional —
        *if* a Milky-Way-mass deflector were placed in a lensing geometry,
        this is where its ring would sit — and the label says so rather than
        implying a measurement exists.
