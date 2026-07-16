# How much light bends, and the factor of two

This chapter buys you the right to treat every galaxy in this repository as a
flat sheet of surface density $\Sigma(\boldsymbol\theta)$, and every
deflection as the gradient of a 2-D potential — the machinery
[Ch. 17](17-lens-equation.md#the-lens-equation) onward runs without
re-justifying it. It earns that right by settling, from first principles, a
question that took Einstein two tries: exactly how far a ray of light bends
past a mass, and why general relativity's answer is *exactly* twice the naive
Newtonian one, not some approximate correction. Get that factor of two wrong
and $\Sigma_{\mathrm{cr}}$, $\kappa$, and $\theta_{\mathrm E}$ would all be off
by it two chapters from now — everything from Ch. 17's lens equation to the
$\gamma_{\mathrm{binned}}=1.103$
<!-- check: ch25.gamma_money = 1.103 ± 0.008 --> money number this book is
chasing rests on this one multiplicative constant being right.

!!! abstract "What you can skip"
    If you already know the GR light-bending result and where the factor of
    two comes from, skip [Newtonian deflection](#newtonian-deflection) and
    [The factor of two](#the-factor-of-two) and start at
    [The thin-lens approximation](#the-thin-lens) — that section is the one
    piece of new machinery every later chapter leans on: it is where
    $\Sigma_{\mathrm{cr}}$ ([Ch. 15](15-distances.md#sigma-crit)) and $\psi$
    ([Ch. 6](06-vector-calculus.md#poisson-for-lensing)) actually meet. If
    you're already comfortable treating a lens as a flat projected sheet with
    no further justification needed, skip straight to
    [Connect to the repo](#connect). Nothing in this chapter touches $\gamma$
    — the density-slope parameter does not exist until
    [Ch. 20](20-profiles.md#the-epl-and-gamma); this one is about the
    coupling constant $G$, not the slope.

## Newtonian deflection { #newtonian-deflection }

Set up the calculation an 18th-century Newtonian would recognize: a point
mass $M$ sits at the origin, and something travels past it in a straight
line (undeflected, to leading order — an impulse approximation is
self-consistent exactly as long as the true deflection turns out to be
small) at speed $c$, offset by an **impact parameter** $b$. Parametrize
position along the path by $x=ct$, so the distance to the mass at any moment
is $r=\sqrt{b^2+x^2}$.

Newtonian gravity pulls with acceleration $GM/r^2$ directed toward the mass.
Only the component perpendicular to the direction of travel actually bends
the path; by similar triangles that component is $b/r$ of the total:

$$
a_\perp(x) = \frac{GM}{r^2}\cdot\frac{b}{r} = \frac{GMb}{(b^2+x^2)^{3/2}}.
$$

Gravitational acceleration does not depend on the falling body's own mass —
Galileo's observation, built into Newtonian gravity as the equality of
inertial and gravitational mass — so this formula holds for a cannonball, a
comet, or (in the 18th-century corpuscular picture) a particle of light,
provided it moves at $c$. Integrate the perpendicular acceleration over the
whole encounter to get the sideways velocity picked up:

$$
\Delta v_\perp = \int_{-\infty}^{\infty} a_\perp\,dt
= \frac{GMb}{c}\int_{-\infty}^{\infty}\frac{dx}{(b^2+x^2)^{3/2}}.
$$

The integral is elementary — differentiate $x/(b^2\sqrt{b^2+x^2})$ and check
it reproduces the integrand, or substitute $x=b\tan u$ — and comes out to
$2/b^2$ regardless of method:

$$
\int_{-\infty}^{\infty}\frac{dx}{(b^2+x^2)^{3/2}}
= \left[\frac{x}{b^2\sqrt{b^2+x^2}}\right]_{-\infty}^{\infty} = \frac{2}{b^2}.
$$

So $\Delta v_\perp = 2GM/(cb)$. For a small deflection the bend angle is just
the ratio of the sideways kick to the forward speed, $\alpha\approx\Delta
v_\perp/c$:

$$
\alpha_{\mathrm N} = \frac{2GM}{c^2 b}. \label{eq:newton-deflection}
$$

This is precisely the calculation Henry Cavendish worked out privately around
1784 and Johann Georg von Soldner published in 1801, more than a century
before anyone had heard of relativity — nothing here used anything but
$F=GMm/r^2$ and a photon treated as an ordinary, if massless, projectile.

## The factor of two { #the-factor-of-two }

Einstein's own first attempt, in 1911, used only the **equivalence
principle** — the same mass-independence this chapter's derivation leaned on
— applied to how a gravitational potential affects the local speed of light,
rather than to a Newtonian force directly. That calculation reproduced
$\eqref{eq:newton-deflection}$ almost exactly: two conceptually different
arguments, 127 years apart, landing on the same number, because the
equivalence-principle argument (in the weak-field limit) only sees the part
of spacetime's geometry that a slow-moving Newtonian projectile would also
see — the warping of *time*.

The completed field equations of 1915 add a second ingredient with no
Newtonian counterpart at all: mass also curves *space*, not only the flow of
time through it. In the weak-field limit the metric splits cleanly into these
two pieces,

$$
ds^2 = -\left(1+\frac{2\Phi}{c^2}\right)c^2\,dt^2
+ \left(1-\frac{2\Phi}{c^2}\right)\left(dx^2+dy^2+dz^2\right),
\qquad \Phi=-\frac{GM}{r},
$$

and a full derivation — which needs the null-geodesic equation, genuine
tensor calculus this guide skips — shows the two terms contribute *equally*
to the bending of anything moving at $v=c$. A slow-moving object weights the
spatial term by a factor of order $(v/c)^2$ and barely notices it, which is
exactly why $\eqref{eq:newton-deflection}$'s ordinary Newtonian mechanics,
built for slow particles, only ever captured the time half. Light has no such
suppression, and the result is exact, not an approximate correction on top of
Newton:

$$
\alpha_{\mathrm{GR}} = \frac{4GM}{c^2 b} = 2\,\alpha_{\mathrm N}.
\label{eq:gr-deflection}
$$

**Worked check: starlight grazing the Sun.** Put $M=M_\odot$ and $b=R_\odot$
— light skimming the visible edge of the Sun, the only geometry an eclipse
lets you test from the ground, since otherwise the Sun's own glare drowns out
any star close enough on the sky to show a measurable bend.
$GM_\odot/c^2\approx1476.6$ m
<!-- check: ch16.GM_over_c2_m = 1476.6 ± 0.1 --> — exactly half the Sun's own
Schwarzschild radius, $2.953$ km
<!-- check: ch16.schwarzschild_radius_sun_km = 2.953 ± 0.001 -->, a number
with no other role in this calculation beyond being a memorable way to carry
$GM_\odot/c^2$ around. Divide by $R_\odot=6.957\times10^8$ m and convert
radians to arcsec with the $206265$ factor [Ch. 9](09-units.md#angles-on-the-sky)
built:

$$
\alpha_{\mathrm N} = \frac{2\times1476.6\ \text{m}}{6.957\times10^8\ \text{m}}
\times 206265'' \approx 0.876''.
$$

<!-- check: ch16.alpha_newton_arcsec = 0.8756 ± 0.001 -->

$$
\alpha_{\mathrm{GR}} = 2\,\alpha_{\mathrm N} \approx 1.751''.
$$

<!-- check: ch16.alpha_gr_arcsec = 1.7512 ± 0.001 -->

The ratio is exactly $2$
<!-- check: ch16.gr_over_newton_ratio = 2.0 ± 1e-9 --> — nothing in
$\eqref{eq:newton-deflection}$ and $\eqref{eq:gr-deflection}$ lets it be
anything else, since $\alpha_{\mathrm{GR}}$ was *defined* as double
$\alpha_{\mathrm N}$ above. The physics is in which of the two numbers,
$0.876''$ or $1.751''$, the sky actually shows.

That question is what the Astronomer Royal Frank Dyson organized twin
expeditions to answer during the total solar eclipse of May 29, 1919 — Arthur
Eddington to Príncipe, Andrew Crommelin and Charles Davidson to Sobral — a
total eclipse being the only moment starlight near the Sun's limb is visible
at all. Three numbers were on the table going
in: $0''$ (no deflection, Newton's first-approximation intuition that a
massless particle isn't pulled by gravity at all), $0.876''$
(Cavendish/Soldner's Newtonian corpuscular estimate, matching Einstein's own
incomplete 1911 result), and $1.751''$ (the full 1915 theory). Dyson,
Eddington, and Davidson's 1920 paper reported results close to the GR
prediction and clearly separated from the Newtonian one, and the announcement
made Einstein a worldwide public figure within days.

!!! tip "You already know this"
    A well-designed experiment does not just confirm a model — it
    discriminates between models that make *distinct, numeric* predictions.
    Eddington's eclipse had three sharply separated candidates to rule
    between: $0''$, $0.876''$, and $1.751''$. That is exactly the shape of
    the test this repository runs on its own money number in
    [Ch. 25](25-money-number.md#the-money-number): not "is there a signal,"
    but "which of several numerically distinct values does the data actually
    prefer" — and, as with Eddington's data, the answer there turns out to
    be less clean than a single confirmed number.

## The thin-lens approximation { #the-thin-lens }

A real galaxy is not a point mass; it is a 3-D distribution of stars and dark
matter with genuine depth along the line of sight. Superposing
$\eqref{eq:gr-deflection}$ over every mass element in that distribution, and
justifying the step where a 3-D galaxy collapses to a flat 2-D sheet, is what
[Ch. 6](06-vector-calculus.md#poisson-for-lensing) forward-referenced as this
chapter's job.

Start from the Newtonian potential $\Phi$ sourced by the lens's 3-D density
$\rho$, which obeys $\nabla^2\Phi=4\pi G\rho$
([Ch. 6](06-vector-calculus.md#poisson-for-lensing)'s Poisson equation, one
dimension up — the same divergence-theorem argument, sourced by mass instead
of a unit point charge). Pick a line of sight at fixed transverse physical
position $\boldsymbol\xi$ and let $z$ run along it. Split the 3-D Laplacian
into a transverse piece and a $z$ piece, $\nabla^2\Phi = \nabla^2_\perp\Phi +
\partial^2\Phi/\partial z^2$, and integrate the whole equation along $z$:

$$
\int_{-\infty}^{\infty}\nabla^2_\perp\Phi\,dz
= 4\pi G\int_{-\infty}^{\infty}\rho\,dz
- \left[\frac{\partial\Phi}{\partial z}\right]_{-\infty}^{\infty}.
$$

The boundary term vanishes because $\Phi\to\text{const}$ far outside any
finite mass distribution, so its slope does too. What survives is an
**exact** identity — no thinness assumed anywhere yet:

$$
\int_{-\infty}^{\infty}\nabla_\perp^2\Phi\,dz = 4\pi G\,\Sigma(\boldsymbol\xi),
\qquad \Sigma(\boldsymbol\xi)\equiv\int_{-\infty}^{\infty}\rho\,dz.
\label{eq:projected-poisson}
$$

$\Sigma$ is the same line-of-sight projection [Ch. 3](03-integrals.md#the-abel-projection)
built for a spherical $\rho\sim r^{-\gamma}$; here it is the identical
operation — integrate the density straight down the line of sight — applied
to whatever shape the galaxy actually has.

**Where "thin" actually enters.** $\eqref{eq:projected-poisson}$ needed no
approximation. The approximation is in what comes next: treating every mass
element as sitting at one shared angular-diameter distance
$D_{\mathrm d}$, rather than at its own true distance $D_{\mathrm d}+z$
depending on where along the line of sight it happens to be. That is valid
exactly when the galaxy's own line-of-sight depth is negligible next to
$D_{\mathrm d}$ itself. A giant elliptical is a few to a few tens of
kiloparsecs across; a fiducial lens at $z_{\mathrm l}=0.5$ (the same system
[Ch. 10](10-galaxies.md#velocity-dispersion) and
[Ch. 19](19-einstein-radius.md#theta-e-from-sigma-v) use) sits at
$D_{\mathrm d}\approx1259$ Mpc
<!-- check: ch16.D_d_fiducial_mpc = 1259 ± 1 -->. A $10$-kpc depth against
that distance is a ratio of about $8\times10^{-6}$
<!-- check: ch16.depth_over_distance_ratio = 7.94e-6 ± 0.5e-6 --> — eight
parts in a million. That number, not any statement about a galaxy being
"flat," is the actual license to write $\Sigma(\boldsymbol\theta)$ as living
on a single plane at a single distance.

With the lens confined to one plane, convert to the angular coordinate
$\boldsymbol\theta=\boldsymbol\xi/D_{\mathrm d}$ and define the **lensing
potential**, absorbing the distance ratios and the GR normalization from the
previous section into one object:

$$
\psi(\boldsymbol\theta) \equiv \frac{D_{\mathrm{ds}}}{D_{\mathrm d}D_{\mathrm s}}
\cdot\frac{2}{c^2}\int_{-\infty}^{\infty}\Phi(D_{\mathrm d}\boldsymbol\theta,z)\,dz.
\label{eq:psi-projection}
$$

($D_{\mathrm{ds}}$ here is the distance *between* the lens and source
redshifts, never $D_{\mathrm s}-D_{\mathrm d}$ —
[Ch. 15](15-distances.md#distances-do-not-add) is the reminder to keep
repeating.) The $2/c^2$ is not a free normalization choice: it is
$\eqref{eq:gr-deflection}$'s factor of two, carried through. Had the previous
section's naive equivalence-principle result been the whole truth, this
prefactor would read $1/c^2$, not $2/c^2$ — and, as Exercise 16.4 works out
exactly, everything downstream would silently be off by a factor of $2$ in
$\kappa$.

Now differentiate $\eqref{eq:psi-projection}$. Since $\boldsymbol\xi=D_{\mathrm
d}\boldsymbol\theta$ is a uniform rescaling, the chain rule gives
$\nabla^2_\theta = D_{\mathrm d}^2\,\nabla^2_\perp$ — the same conjugation
rule [Exercise 6.1](06-vector-calculus.md#exercises) used for a rotation
matrix, here for a uniform scale factor instead. Applying that and
$\eqref{eq:projected-poisson}$:

$$
\nabla^2_\theta\psi = \frac{D_{\mathrm{ds}}}{D_{\mathrm d}D_{\mathrm s}}
\cdot\frac{2}{c^2}\cdot D_{\mathrm d}^2\cdot 4\pi G\,\Sigma(\boldsymbol\theta)
= \frac{8\pi G\,D_{\mathrm d}D_{\mathrm{ds}}}{c^2 D_{\mathrm s}}\,\Sigma(\boldsymbol\theta).
$$

Recognize the prefactor using [Ch. 15](15-distances.md#sigma-crit)'s
$\Sigma_{\mathrm{cr}} = c^2D_{\mathrm s}/(4\pi G D_{\mathrm d}D_{\mathrm{ds}})$,
so $1/\Sigma_{\mathrm{cr}} = 4\pi G D_{\mathrm d}D_{\mathrm{ds}}/(c^2D_{\mathrm s})$,
and the prefactor above is exactly twice that:

$$
\nabla^2_\theta\psi = \frac{2\,\Sigma(\boldsymbol\theta)}{\Sigma_{\mathrm{cr}}}
= 2\kappa(\boldsymbol\theta). \label{eq:poisson-projected}
$$

This is [Ch. 6](06-vector-calculus.md#poisson-for-lensing)'s
$\nabla^2\psi=2\kappa$ again, this time earned from an actual line-of-sight
integral over real mass rather than from an abstract trace-of-Hessian
argument — the two derivations meeting from opposite directions is the check
that both are internally consistent.

**Two unrelated 2's, reconciled, not contradicted.** Ch. 6 flagged
$\nabla^2\psi=2\kappa$'s "2" as pure convention — the trace of a $2\times2$
identity matrix, true by the *definitions* of $\kappa$ and $\Gamma$ as the
isotropic and traceless parts of the Hessian of $\psi$, for *any* physical
theory of gravity you plugged into $\psi$. That claim survives this section
intact: nothing about $\eqref{eq:poisson-projected}$'s derivation used GR
specifically — swap $\eqref{eq:psi-projection}$'s $2/c^2$ for a different
constant and the same trace algebra still forces $\nabla^2\psi=2\kappa$,
just with $\kappa$ meaning something else. The GR-specific fact lives one
level down, in *which* prefactor $\eqref{eq:psi-projection}$ is entitled to
use — the $2$ inherited from $\eqref{eq:gr-deflection}$, not the trace's $2$.
They are still two unrelated 2's, exactly as Ch. 6 said; this section only
shows precisely where each one enters the same formula.

One loose end, honestly flagged rather than swept past: for a genuine point
mass, $\Phi=-GM/r$ makes the raw integral $\int\Phi\,dz$ in
$\eqref{eq:psi-projection}$ diverge logarithmically as $z\to\pm\infty$ — an
idealized point mass has infinite reach along every line of sight. $\psi$
itself is only ever needed through its gradient
$\boldsymbol\alpha=\nabla\psi$ ([Ch. 6](06-vector-calculus.md#greens-function)'s
point-mass check used the *form* $\theta_{\mathrm E}^2\ln r$ for exactly this
reason), and that gradient is perfectly finite even where $\psi$ needs an
arbitrary additive constant to be well defined. It is also why `gigalens`'s
production EPL profile, cited in
[Ch. 6](06-vector-calculus.md#connect), hands you a closed-form
$\boldsymbol\alpha$ directly and never constructs $\psi$ at all.

## Connect to the repo { #connect }

`site/guide_src/lensing.py:9-19`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/lensing.py#L9-L19))
states the payoff of this whole chapter as a module-docstring bullet:
"Angles in arcsec throughout. The lens-modeling likelihood in this repo is
entirely angular — no cosmology enters it." That sentence is only true
because $G$, $c$, and $M$ were absorbed once, here, into $\Sigma_{\mathrm{cr}}$
and the profile normalizations — never to be typed again in a per-galaxy fit.
`site/guide_src/cosmo.py:1-19`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/cosmo.py#L1-L19))
names the three files where that absorption is undone and cosmology briefly
becomes real again; `site/guide_src/cosmo.py:45`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/cosmo.py#L45))
is this chapter's $\Sigma_{\mathrm{cr}}=c^2D_{\mathrm s}/(4\pi G D_{\mathrm
d}D_{\mathrm{ds}})$, written as code. `reproductions/sheu-2024b/04_setup_multiplane.py:128`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/sheu-2024b/04_setup_multiplane.py#L128))
is the identical formula computed inside a real campaign, for the Carousel
cluster lens — the same line `cosmo.sigma_crit` reproduces to five
significant figures in [Ch. 15](15-distances.md#the-carousel).

None of $G$, $c$, or $M$ survives past this chapter as an explicit constant
in this repository's own code: `gigalens`'s EPL and SIE deflection fields
([Ch. 20](20-profiles.md#the-epl-and-gamma)) are written entirely in
$\theta_{\mathrm E}$, $q$, $\phi$, and $\gamma$ — the coupling constant this
chapter spent its whole length pinning down never appears in them by name
again.

## Exercises { #exercises }

??? question "Exercise 16.1 — Dimensional check on $2GM/(c^2b)$"
    Confirm from units alone that $2GM/(c^2b)$ is dimensionless (an angle, in
    radians), using $[G]=\text{m}^3\,\text{kg}^{-1}\,\text{s}^{-2}$,
    $[M]=\text{kg}$, $[c]=\text{m}\,\text{s}^{-1}$, $[b]=\text{m}$.

    ??? success "Solution"

        $$
        \frac{[G][M]}{[c]^2[b]}
        = \frac{\text{m}^3\,\text{kg}^{-1}\,\text{s}^{-2}\cdot\text{kg}}
        {(\text{m}^2\,\text{s}^{-2})\cdot\text{m}} = \frac{\text{m}^3\,\text{s}^{-2}}{\text{m}^3\,\text{s}^{-2}} = 1.
        $$

        Dimensionless, as an angle in radians has to be — the factor of $2$
        is a pure number and does not affect the check. The same units
        argument applies unchanged to $\eqref{eq:gr-deflection}$'s $4$.

??? question "Exercise 16.2 — Redo the derivation for a slower projectile"
    [Newtonian deflection](#newtonian-deflection) assumed the passing object
    moves at $c$. Redo the same impulse-approximation integral for something
    moving at a general speed $v$ (a comet, say, not a photon), and show the
    deflection angle generalizes to $\alpha=2GM/(v^2b)$. What does this say
    about how strongly a slow-moving object is deflected, compared to light,
    at the same impact parameter?

    ??? success "Solution"
        Parametrize the path by $x=vt$ instead of $x=ct$. The perpendicular
        acceleration $a_\perp(x)$ is unchanged (it never depended on the
        passing object's speed), but $dt=dx/v$ now, so

        $$
        \Delta v_\perp = \frac{GMb}{v}\int_{-\infty}^\infty\frac{dx}{(b^2+x^2)^{3/2}}
        = \frac{2GM}{vb},
        $$

        and the deflection angle is $\Delta v_\perp/v = 2GM/(v^2b)$ — set
        $v=c$ to recover $\eqref{eq:newton-deflection}$. Because this scales
        as $1/v^2$, a *slower* object is deflected *more* by the same mass at
        the same impact parameter, not less — light, moving at the fastest
        possible speed, is the *least* bent per unit $GM/b$ of anything that
        could pass by. A slow-moving spacecraft skimming the Sun at, say,
        $30$ km/s would swing through a far larger angle than a photon at
        the same distance ever does; it is only because $c$ is so large that
        $\alpha_{\mathrm N}$ comes out as a fraction of an arcsecond at all.

??? question "Exercise 16.3 — Which part of 'thin lens' is exact, and which is approximate"
    [The thin-lens approximation](#the-thin-lens) derived
    $\eqref{eq:projected-poisson}$, $\int\nabla_\perp^2\Phi\,dz=4\pi
    G\Sigma(\boldsymbol\xi)$, without ever assuming the lens is thin. Where,
    precisely, did an approximation get made — and what physical quantity
    would have to be small for it to be justified?

    ??? success "Solution"
        $\eqref{eq:projected-poisson}$ itself is exact: it needs only that
        $\Phi$ (and its $z$-derivative) go to a constant far outside the mass
        distribution, which is true for any finite lens regardless of its
        depth. The approximation is the step right after — treating every
        mass element along the line of sight as sitting at the *same*
        angular-diameter distance $D_{\mathrm d}$, so that a single number
        $D_{\mathrm d}$ (and hence a single $\Sigma_{\mathrm{cr}}$) can be
        used for the whole galaxy. That is justified when the galaxy's own
        line-of-sight depth is tiny compared to $D_{\mathrm d}$ — the
        $\sim8\times10^{-6}$ ratio computed in the text — not by any property
        of the projection integral itself. A cluster-scale lens with several
        distinct mass clumps spread over hundreds of Mpc along the line of
        sight (a genuine multi-plane lens) is exactly the regime where this
        second approximation, not the first, starts to fail.

??? question "Exercise 16.4 — What would happen to $\kappa$ under the wrong theory of gravity"
    Suppose $\psi_{\mathrm{naive}}(\boldsymbol\theta)$ used
    $\eqref{eq:psi-projection}$'s formula with $1/c^2$ in place of $2/c^2$ —
    i.e., only the incomplete 1911 equivalence-principle deflection, never
    doubled by the 1915 field equations. Using the *same*
    $\Sigma_{\mathrm{cr}}=c^2D_{\mathrm s}/(4\pi GD_{\mathrm d}D_{\mathrm{ds}})$
    formula from [Ch. 15](15-distances.md#sigma-crit), what would
    $\nabla^2_\theta\psi_{\mathrm{naive}}$ come out to, in terms of the *true*
    $\kappa=\Sigma/\Sigma_{\mathrm{cr}}$?

    ??? success "Solution"
        Repeat the derivation with $1/c^2$ instead of $2/c^2$: every step is
        identical except the prefactor is now half of what it was, so

        $$
        \nabla^2_\theta\psi_{\mathrm{naive}}
        = \frac{4\pi G\,D_{\mathrm d}D_{\mathrm{ds}}}{c^2D_{\mathrm s}}\,\Sigma(\boldsymbol\theta)
        = \frac{\Sigma(\boldsymbol\theta)}{\Sigma_{\mathrm{cr}}} = \kappa(\boldsymbol\theta),
        $$

        not $2\kappa$. But Ch. 6's trace identity says the quantity anyone
        would *call* "$\kappa$" from $\psi_{\mathrm{naive}}$ — half the trace of
        its Hessian, by definition — must still satisfy
        $\nabla^2\psi_{\mathrm{naive}}=2\kappa_{\mathrm{naive}}$ regardless. Comparing
        the two expressions, $\kappa_{\mathrm{naive}}=\Sigma/(2\Sigma_{\mathrm{cr}}) =
        \tfrac12\kappa$: using the correct $\Sigma_{\mathrm{cr}}$ formula with
        the wrong (undoubled) light-bending physics would make $\kappa$ read
        at exactly half strength for the same real mass. Since the Einstein
        radius is defined by mean $\kappa=1$
        ([Ch. 19](19-einstein-radius.md#the-mean-convergence-identity)), a
        universe that bent light the Newtonian amount would need to enclose
        roughly twice the projected mass before calling a radius "Einstein" —
        the single factor of two from [The factor of two](#the-factor-of-two)
        propagating, unaltered, all the way to a number every strong-lensing
        paper quotes in its abstract.
