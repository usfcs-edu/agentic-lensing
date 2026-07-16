# 18. Magnification, critical curves, and caustics

Chapter 17 gave you the lens equation, $\boldsymbol\beta = \boldsymbol\theta - \boldsymbol\alpha(\boldsymbol\theta)$
([Ch. 17](17-lens-equation.md#the-lens-equation)), and the fact that it can have more than one
solution for a single source. This chapter answers the two questions that follow immediately: how
bright is each of those solutions, and where, exactly, does a second one come from? Both answers
are the same object, differentiated once — the lens Jacobian
$A \equiv \partial\boldsymbol\beta/\partial\boldsymbol\theta$, whose determinant Chapter 4 already
told you is a literal area ratio. Its reciprocal is the magnification. The curve where it vanishes
is a *critical curve*. The image of that curve under the lens map is a *caustic*. And crossing a
caustic is the precise mechanism by which a second, third, or fourth image is born. By the end of
this chapter you can compute all four — magnification, critical curve, caustic, and the parity of
an image — from nothing but a deflection field and the differentiation Chapter 4 already taught
you, and check every one of them against a number this repository's own figure-generation code and
production forward model actually compute.

!!! abstract "What you can skip"
    Chapter 4 already derived that $|\det J|$ is a local area-scaling factor and opened the
    Log-Det Ledger with $\mu = 1/|\det A|$ as row 1 — if that chapter is fresh, skip straight to
    [Magnification is a Jacobian](#magnification-is-a-jacobian) for the lensing-specific
    radial/tangential split, or to [Critical curves](#critical-curves) if you also recall that a
    symmetric matrix's determinant is the product of its eigenvalues (Chapter 5). What is not
    boilerplate: why the *sign* of $\det A$, not just its magnitude, carries distinct physical
    information ([Parity](#parity)), and why a curve that shrinks to a single point in the image
    plane does not shrink to a point in the source plane ([Caustics](#caustics)) — a fact this
    repository's own figure code spends a paragraph justifying because it is genuinely
    unintuitive the first time.

## Magnification is a Jacobian { #magnification-is-a-jacobian }

The lens Jacobian is the derivative of the lens equation, taken exactly the way Chapter 4 defines
a Jacobian:

$$
A \;\equiv\; \frac{\partial \boldsymbol\beta}{\partial \boldsymbol\theta}
\;=\; I - \frac{\partial \boldsymbol\alpha}{\partial \boldsymbol\theta}. \label{eq:jacA}
$$

Because $\boldsymbol\alpha = \nabla\psi$ for a scalar lensing potential $\psi$
([Ch. 6](06-vector-calculus.md#poisson-for-lensing)), $\partial\boldsymbol\alpha/\partial\boldsymbol\theta$
is a Hessian, and Chapter 4 already proved a gradient field's Jacobian is automatically symmetric
([Ch. 4](04-multivariable.md#gradient-jacobian-hessian)). So $A$ is symmetric at every image
position, before any lens-specific physics is chosen — which is exactly what lets Chapter 5's
general fact about symmetric $2\times2$ matrices apply here: $A$ diagonalizes in a real,
orthonormal eigenbasis ([Ch. 5](05-linear-algebra.md#symmetric-2x2)).

A short physical argument fixes what $1/\det A$ means. Light travel conserves photon number — no
absorption, no emission along a vacuum ray path — so *surface brightness* (flux per unit solid
angle) is the same at the source and at the image. A tiny image-plane patch of solid angle
$d\Omega_\theta$ maps, under $\eqref{eq:jacA}$'s local linearization, to a source-plane patch of
solid angle $d\Omega_\beta = |\det A|\,d\Omega_\theta$ (Chapter 4's area-ratio argument, applied to
$A$ specifically). Flip that around: the *image* occupies a solid angle $d\Omega_\beta/|\det A|$
for a fixed intrinsic source patch, so a bigger apparent patch at the *same* surface brightness
means more total flux, by a factor $1/|\det A|$. That factor is the magnification, and its signed
version is what this repository — and this guide — call $\mu$:

$$
\mu \;\equiv\; \frac{1}{\det A}. \label{eq:mu}
$$

`site/guide_src/lensing.py:139` (`magnification`) computes exactly $\eqref{eq:mu}$, and its own
docstring states this chapter's opening claim in one line: *"This is the SAME change-of-variables
factor that a normalizing flow applies as $-\log|\det J|$."* Chapter 4 opened the Log-Det Ledger
with this fact stated abstractly, before this book had a lens equation to check it against. Here
is the check.

For a *circularly symmetric* deflection field, $\boldsymbol\alpha(\boldsymbol\theta) =
\alpha(r)\,\hat{\boldsymbol\theta}$ with $r = |\boldsymbol\theta|$, symmetry does the
eigendecomposition for you: at any point, the eigenvectors of $A$ are simply the local radial and
tangential directions. Move a small step $d\theta_r$ purely radially, and the deflection changes
by $\alpha'(r)\,d\theta_r$ in the same radial direction — giving a radial eigenvalue
$A_{rr} = 1 - \alpha'(r)$. Move a small step $d\theta_t$ purely tangentially instead: this
rotates your position by an angle $d\theta_t/r$, and by circular symmetry the deflection vector
rotates by that identical angle, which displaces it tangentially by
$\alpha(r)\cdot(d\theta_t/r)$ — giving a tangential eigenvalue $A_{tt} = 1 - \alpha(r)/r$. So

$$
\det A = A_{rr}A_{tt} = \bigl(1-\alpha'(r)\bigr)\left(1-\frac{\alpha(r)}{r}\right).
$$

Apply this to the singular isothermal sphere, $\alpha(r) = \theta_{\mathrm{E}}$
(`lensing.py:54`, `sis_deflection`) — a *constant*-modulus deflection, "the flat rotation curve of
a galaxy in one line: the deflection does not care how far out you are, only which way"
(`lensing.py:57-58`). Constant $\alpha$ means $\alpha'(r) = 0$ identically, so $A_{rr} = 1$ for
every $r > 0$: an SIS has no radial critical curve at all, ever — only $A_{tt} = 1 -
\theta_{\mathrm{E}}/r$ can vanish. That gives a magnification

$$
\mu(r) = \frac{1}{1-\theta_{\mathrm{E}}/r} = \frac{r}{r-\theta_{\mathrm{E}}}.
$$

At $\theta_{\mathrm{E}} = 1''$ and an image at $r = 2''$, this predicts $\mu = 2$
<!-- check: ch18.sis_mu_analytic = 2.0 ± 1e-9 -->. `lensing.py:101` (`lens_jacobian`) never sees
this formula — it differentiates `sis_deflection` numerically, by central differences, and
`magnification` takes $1/\det A$ of the result. That purely numerical route gives $\mu =
1.999999999962$ <!-- check: ch18.sis_mu_numerical = 2.0 ± 1e-9 -->, agreeing with the closed form
to $3.8\times10^{-11}$ <!-- check: ch18.sis_mu_analytic_match = 3.81e-11 ± 1e-9 -->
— a Jacobian built from finite differences, matching an analytic eigenvalue argument to ten
decimal places. `worked_examples.py --show ch18` reproduces both numbers directly.

!!! note "Log-Det Ledger — row 1, paid off"
    Chapter 4 opened this ledger with $\mu = 1/|\det A|$ stated abstractly. This section supplies
    everything that was missing: $A$ is the derivative of a real lens equation
    ([Ch. 17](17-lens-equation.md#the-lens-equation)), $\mu = 1/\det A$ is computed by a genuine
    Jacobian differentiation (`lensing.py:101-146`), and the result is checked against a
    closed-form eigenvalue argument to machine precision. Row 1 is not a metaphor for the rest of
    this book — it is arithmetic that reproduces.

## Critical curves { #critical-curves }

A **critical curve** is the locus $\{\boldsymbol\theta : \det A(\boldsymbol\theta) = 0\}$. There,
$\mu$ formally diverges: a point source placed exactly on the corresponding caustic (below) would
be infinitely magnified. Real sources have finite extent and real telescopes have a PSF
([Ch. 11](11-observation.md#ccds-and-psf)), so nothing in an actual image is literally infinite —
but images that straddle a critical curve get *very* bright and very stretched, which is exactly
what a giant arc is.

Because $\det A = A_{rr}A_{tt}$, it vanishes when *either* eigenvalue vanishes on its own,
and those are two geometrically distinct curves: a **tangential** critical curve ($A_{tt}=0$)
and a **radial** critical curve ($A_{rr}=0$). The SIS above has only the first — $A_{rr}
\equiv 1$ never crosses zero. A **singular isothermal ellipsoid** (SIE, $q < 1$) has both, because
breaking circular symmetry lets the deflection field's radial derivative do something an SIS's
never can.

!!! tip "You already know this"
    A critical curve is where the lensing Jacobian becomes singular — exactly the condition a
    normalizing flow's bijector is engineered to *forbid* everywhere (a well-built flow keeps
    $\det J$ bounded away from zero, precisely so $\log|\det J|$ never blows up). The lens
    equation carries no such constraint: nothing stops $\det A$ from crossing zero, and when it
    does, the map stops being locally invertible in one direction — which is the entire mechanism
    behind multiple imaging. A flow is built to never do what a lens does routinely.

<figure markdown="span">
  ![Critical curves and caustics for an SIE lens](figures/ch18-sie-caustics-light.svg#only-light){ width="90%" }
  ![Critical curves and caustics for an SIE lens](figures/ch18-sie-caustics-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 18.1.** Critical curves ($\det A=0$, left) and the caustics
  they map to (right), for an SIE with $\theta_{\mathrm{E}}=1''$, $q=0.7$. Nothing here is drawn:
  every curve comes from numerically differentiating the deflection field
  (`site/guide_src/figures.py:144-209`). Left: the tangential critical curve (green) is a single
  closed oval reaching $0.9999''$ along its long axis and $0.6999''$ along its short axis
  <!-- check: ch18.sie_crit_x_extent = 0.99993 ± 0.0001 --><!-- check: ch18.sie_crit_y_extent = 0.69990 ± 0.0001 -->
  — indistinguishable, at this precision, from $\theta_{\mathrm{E}}$ and $q\,\theta_{\mathrm{E}}$
  exactly. The small dot at the center marks the radial critical curve, which for a singular
  profile is too small for this grid to resolve (the code masks everything inside $0.08''$ of
  center rather than plot the ~3-pixel artifact a naive contour would return there). Right: the
  four-cusped astroid is the tangential curve's image; the dashed curve is the radial branch's
  caustic — reaching a maximum radius of $0.8102''$
  <!-- check: ch18.sie_cut_radius_max = 0.8102 ± 0.0001 --> despite originating from a single
  point.</figcaption>
</figure>

The radial critical curve's disappearing act is a direct consequence of the profile being
*singular*: convergence $\kappa \sim 1/R$ diverges as $R\to0$
([Ch. 20](20-profiles.md#the-epl-and-gamma)), and as the regularizing core radius `lensing.py:65`
(`sie_deflection`'s `s` argument) shrinks toward zero, the small oval where $A_{rr}=0$ shrinks
with it, collapsing onto the single point $\boldsymbol\theta = \boldsymbol 0$ in the exactly
singular limit. You might expect its caustic to collapse to a point too. It does not — which is
the subject of the next section.

## Caustics { #caustics }

A **caustic** is the image of a critical curve under the lens map: push every point on
$\{\det A = 0\}$ through $\boldsymbol\beta = \boldsymbol\theta - \boldsymbol\alpha(\boldsymbol\theta)$
and plot where it lands in the source plane. The word is borrowed from ordinary optics — the same
bright, curved lines that form at the bottom of a swimming pool, or on the wall behind a wine
glass, are caustics of a completely different (but equally singular) map. Figure 18.1's right
panel is one: the tangential branch's caustic is the **astroid**, a four-cusped closed curve, small
compared to $\theta_{\mathrm{E}}$.

Here is why crossing a smooth stretch of caustic — away from a cusp — always changes the image
count by exactly two, never one or three. Near such a point, exactly one eigenvalue of $A$
vanishes; call its eigendirection $\parallel$ and the other $\perp$. Because $A_{\parallel\parallel}$
is a smooth function of position that happens to equal zero right there, the first surviving term
in a Taylor expansion of $\beta_\parallel$ along the $\parallel$ direction is *quadratic*, not
linear — precisely Chapter 2's argument for why a stationary point needs a second-order term to
resolve it at all ([Ch. 2](02-derivatives.md#why-second-order)):

$$
\beta_\parallel(\theta_\parallel) \;\approx\; \beta_{\parallel,0} + c\,\theta_\parallel^2,
\qquad c \ne 0.
$$

Solving for $\theta_\parallel$ given a source position $\beta_\parallel$ gives $\theta_\parallel =
\pm\sqrt{(\beta_\parallel-\beta_{\parallel,0})/c}$: two real solutions on one side of
$\beta_{\parallel,0}$, none on the other. A source crossing the caustic point $\beta_{\parallel,0}$
therefore gains or loses exactly one *pair* of images, born or annihilated together right at the
critical curve (where $\mu$ diverges, consistent with the two roots merging as
$\beta_\parallel\to\beta_{\parallel,0}$). This is the **fold catastrophe** — the same local algebra
as a saddle-node bifurcation in a dynamical system, a stable and unstable fixed point annihilating
each other as a parameter crosses a threshold. The astroid's four cusps are points where *two*
folds meet, and crossing through one adds a further pair; a source deep inside the astroid sees
more images than one just inside a smooth edge.

The radial branch's caustic is the more surprising object, and this repository's own comment
explains why in one line: as the regularizing core shrinks, the radial critical curve shrinks to
the single point $\boldsymbol\theta=\boldsymbol0$, but its deflection there does not vanish —
$\boldsymbol\alpha(\boldsymbol0)$ stays finite, of order $\theta_{\mathrm{E}}$, exactly the same
"flat rotation curve" fact used above, and its *direction* still depends on which way you
approached the origin. So walk a vanishingly small circle of radius $\varepsilon$ around the
center, parametrized by angle $\phi$ so that $\boldsymbol\theta = \varepsilon(\cos\phi,\sin\phi)$,
and push it through the lens equation:
$\boldsymbol\beta(\varepsilon,\phi) = \varepsilon(\cos\phi,\sin\phi) -
\boldsymbol\alpha(\varepsilon,\phi) \to -\boldsymbol\alpha(0,\phi)$ as $\varepsilon\to0$ — an
$O(\theta_{\mathrm{E}})$-sized closed curve, not a point, because the limit keeps all of
$\boldsymbol\alpha$'s angular structure even as the circle it came from shrinks away. This is
`site/guide_src/figures.py:176-195`'s construction exactly, and it is why a curve that vanishes in
the image plane still produces a full curve — this chapter's **cut** — in the source plane.
`site/guide_src/worked_examples.py`'s `ch18_magnification` reproduces the same number
(`0.8102368876963119`) by an independent bisection/angular-sampling route rather than importing
the figure code, and the two agree to machine precision
<!-- check: ch18.sie_cut_radius_max_match = 0.0 ± 1e-9 -->.

The repository's own docstring states the cut's role in image counting directly: "inside the cut a
source has two images; outside it has one" (`site/guide_src/figures.py:156-157`) — a second, genuine
(if tiny, at this core radius) fold, nested inside the astroid's. A circular lens has no such
second fold: its only critical curve is the full Einstein ring, and the *entire* ring degenerates
to the single caustic point $\boldsymbol\beta=0$, leaving nothing to nest a further transition
inside. Solving the SIS lens equation directly (as this chapter already has) shows its own image
count still changes, from one to two — but at $|\boldsymbol\beta|=\theta_{\mathrm{E}}$, not at the
caustic point itself: the second image does not meet a partner at a smooth critical curve there,
it shrinks straight into the profile's singular center ($r_2 = \theta_{\mathrm{E}}-\beta \to 0$)
and disappears — a different, singularity-driven way for an image to vanish, not a fold. An SIE's
genuine second critical curve is why real strong lenses come as **doubles** and **quads** — the
observational names for exactly the multiplicity regions Figure 18.1's right panel draws.

## Parity { #parity }

$\mu = 1/\det A$ is signed, not just an area ratio. `lensing.py:140` says it plainly:
*"Signed: negative $\mu$ means a parity-flipped image."* Chapter 4's Exercise 4.4 asked what a
negative $\det A$ could mean physically, before this book had a lens equation to answer with. Now
it does.

$\det A = A_{rr}A_{tt}$ changes sign exactly when one eigenvalue changes sign while the other
does not — which is exactly what happens at a critical curve. In the fold analysis above, the two
newly created images have local slopes $d\beta_\parallel/d\theta_\parallel = 2c\,\theta_\parallel$
of *opposite* sign (one at $+\theta_\parallel$, one at $-\theta_\parallel$): one image preserves
orientation, the other reverses it. A positive-parity ($\mu>0$) image looks like the source,
rotated and stretched; a negative-parity ($\mu<0$) image is its mirror image, not just its
magnified copy.

The SIS worked example already contains both signs. At $r=2''$ (outside $\theta_{\mathrm{E}}=1''$,
only one image exists there), $\mu=+2$: direct parity, magnified. At $r=0.5''$ — inside
$\theta_{\mathrm{E}}$, where $A_{tt} = 1-\theta_{\mathrm{E}}/r = 1-2 = -1$ while $A_{rr}$
stays $+1$ — the same numerical Jacobian gives $\mu = -1$
<!-- check: ch18.sis_mu_inside_analytic = -1.0 ± 1e-9 -->, matching the closed form
$r/(r-\theta_{\mathrm{E}}) = 0.5/(-0.5)$ to $4.0\times10^{-10}$
<!-- check: ch18.sis_mu_inside_analytic_match = 3.99e-10 ± 1e-9 -->. This is the SIS's second
image — the one an Einstein-cross or a double-lens configuration always has a counterpart of —
and $|\mu|=1$ there: it carries exactly the source's own flux, neither magnified nor demagnified,
only flipped. Magnitude and sign are independent axes of the same number; a magnification of
exactly $1$ is not "no lensing," it can be a full mirror flip.

## Connect to the repo { #connect }

- `site/guide_src/lensing.py:101-146` — `lens_jacobian`, `kappa_gamma_from_jacobian`, and
  `magnification`: the whole chapter, in 46 lines. `lens_jacobian` differentiates numerically by
  design, so the guide's magnification numbers are never quoting a special-case closed form
  without also checking it against real finite differences.
- `site/guide_src/figures.py:144-209` — Figure 18.1's generator (`sie_caustics`). Its docstring is
  worth reading in full: it explains, in the code's own words, exactly why the tangential and
  radial branches need different numerics (a grid contour for one, an angular sweep for the
  other) — the same distinction this chapter's Critical curves and Caustics sections build on.
- `reproductions/claude-giga-lens/30_recipe_e2e.py:362-387` (`crit_field`) — the production
  version of this chapter's Jacobian, run on a real MAP fit rather than a toy SIE. It takes a
  cleaner shortcut than the guide's figure code: rather than contouring $\det A = 0$ (which
  conflates both branches into one curve), it computes the tangential eigenvalue directly —
  its own variable is literally called `lam_t` — via the quadratic formula, and contours *that*,
  sidestepping the radial branch's
  resolution problem entirely. The campaign's own report shows the result — "MAP model with
  tangential critical curve" — as one panel of its end-to-end recipe figure
  ([the recipe figure, §8](../current/claude-giga-lens/index.md#sec:recipe)).
- [Ch. 4, "The Log-Det Ledger"](04-multivariable.md#the-log-det-ledger) — row 1, opened there and
  closed here.
- [Ch. 5](05-linear-algebra.md#symmetric-2x2) — the general symmetric-$2\times2$ eigendecomposition
  this chapter specializes to the radial/tangential case.
- [Ch. 20](20-profiles.md#the-epl-and-gamma) and [Ch. 21](21-degeneracies.md#the-mass-sheet-degeneracy)
  — the EPL generalizes the SIS/SIE profiles used here, and every critical curve and caustic in
  this chapter recurs, unchanged in kind, once $\gamma\ne2$ is on the table.

## Exercises { #exercises }

??? question "Exercise 18.1 — the radial/tangential split, from scratch"
    For a circularly symmetric deflection field $\alpha(r)$, this chapter asserted
    $A_{rr} = 1-\alpha'(r)$ and $A_{tt} = 1-\alpha(r)/r$ from a symmetry argument. Confirm
    it by direct differentiation: write $\alpha_1(\theta_1,\theta_2) = \alpha(\rho)\,\theta_1/\rho$
    and $\alpha_2(\theta_1,\theta_2) = \alpha(\rho)\,\theta_2/\rho$ with
    $\rho=\sqrt{\theta_1^2+\theta_2^2}$, and compute $\partial\alpha_1/\partial\theta_1$ and
    $\partial\alpha_2/\partial\theta_2$ at the point $(\theta_1,\theta_2)=(r,0)$.

    ??? success "Solution"
        $\partial\alpha_1/\partial\theta_1 = \alpha'(\rho)\left(\frac{\theta_1}{\rho}\right)^2 +
        \alpha(\rho)\,\frac{\theta_2^2}{\rho^3}$. At $(r,0)$: $\theta_2=0$, $\theta_1/\rho=1$, so
        this is $\alpha'(r)$, giving $a_{11}=1-\alpha'(r)=A_{rr}$.

        $\partial\alpha_2/\partial\theta_2 = \alpha'(\rho)\left(\frac{\theta_2}{\rho}\right)^2 +
        \alpha(\rho)\,\frac{\theta_1^2}{\rho^3}$. At $(r,0)$: $\theta_2/\rho=0$,
        $\theta_1^2/\rho^3=1/r$, so this is $\alpha(r)/r$, giving $a_{22}=1-\alpha(r)/r=A_{tt}$.

        Both off-diagonal terms, $\partial\alpha_1/\partial\theta_2$ and
        $\partial\alpha_2/\partial\theta_1$, carry a factor of $\theta_2$ that vanishes at
        $(r,0)$ — confirming that the radial/tangential axes really are the eigenbasis there, not
        just a convenient guess.

??? question "Exercise 18.2 — a parity flip you can predict without a computer"
    Using only $\mu(r) = r/(r-\theta_{\mathrm{E}})$ for the SIS, at what radius does $\mu = -1$
    exactly, for $\theta_{\mathrm{E}}=1''$? What does $|\mu|=1$ combined with a negative sign mean
    for the resulting image, physically?

    ??? success "Solution"
        Solve $r/(r-1) = -1 \Rightarrow r = -(r-1) \Rightarrow 2r = 1 \Rightarrow r = 0.5''$ —
        exactly the point this chapter checked numerically
        <!-- check: ch18.sis_mu_inside_analytic = -1.0 ± 1e-9 -->. Physically: the image carries
        precisely the source's own flux (no brightening, no dimming — $|\mu|=1$), but it is a
        mirror image, not a magnified copy, because $\det A<0$ there. This resolves Chapter 4's
        Exercise 4.4: a sign flip in $\det A$ is not a numerical curiosity, it is the difference
        between "the same shape, bigger" and "the same shape, flipped."

??? question "Exercise 18.3 — why the SIS never shows a quad"
    Using this chapter's eigenvalue argument, explain in one or two sentences why a circularly
    symmetric lens (SIS or SIE with $q=1$) can never produce four images of a single source, no
    matter where the source sits, while an SIE with $q<1$ routinely does.

    ??? success "Solution"
        A circular lens has $A_{rr}\equiv1$ for all $r>0$ (this chapter's derivation:
        $\alpha(r)=\theta_{\mathrm{E}}$ is constant, so $\alpha'(r)=0$ identically) — there is no
        radial critical curve at any nonzero radius, hence nothing to nest a second multiplicity
        transition inside the tangential one. Its own image count still changes, from one to two,
        but (as the Caustics section works out) that boundary sits at
        $|\boldsymbol\beta|=\theta_{\mathrm{E}}$, where the would-be second image shrinks into the
        singular center and disappears, rather than meeting a partner at a smooth critical curve —
        not a fold. An SIE with $q<1$ breaks the symmetry that forces $A_{rr}\equiv1$, so it
        acquires a second, genuine critical curve of its own — and with it, a real fold, a real
        second multiplicity transition, and the possibility of four images.

??? question "Exercise 18.4 — reading Figure 18.1's numbers as a consistency check"
    Figure 18.1 reports the tangential critical curve's extent as $0.9999''$ along its long axis
    and $0.6999''$ along its short axis, for $\theta_{\mathrm{E}}=1''$, $q=0.7$. State the
    conjecture these numbers support about the *exact* ($s\to0$) singular limit, and say what
    would falsify it.

    ??? success "Solution"
        The conjecture: in the exactly singular limit, the tangential critical curve touches the
        principal axes at exactly $\theta_{\mathrm{E}}$ and $q\,\theta_{\mathrm{E}}$ — here $1.0''$
        and $0.7''$ — with the $0.9999''$/$0.6999''$ reported values differing from those only
        because `lensing.py:65`'s `sie_deflection` uses a small but nonzero core, $s=10^{-4}$, to
        regularize the central singularity. It would be falsified by re-running
        `ch18_magnification`'s bisection with a *smaller* `s` and finding the extents move *away*
        from $1.0$ and $0.7$ rather than toward them — which is not what happens: shrinking $s$
        from $10^{-4}$ to $10^{-10}$ moves both numbers past nine nines of agreement with
        $\theta_{\mathrm{E}}$ and $q\,\theta_{\mathrm{E}}$ exactly, the signature of a genuine
        $s\to0$ limit rather than a coincidence at one particular core size.
