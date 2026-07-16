# The lens equation: beta = theta - alpha

Chapter 16 established that mass bends light and by how much. This chapter
turns that fact into a single equation you can actually solve, and asks the
question a bent-light fact alone doesn't answer: given a source at one
position, how many images does an observer see, and where? The answer falls
out of nothing more than asking how many roots a (generally nonlinear)
equation has — which is why a lens can show one image, two, four, or a full
ring, and why that number is not a free choice of the mass distribution but a
consequence you can derive. By the end of this chapter you will have solved
the equation this repository's own renderer never has to solve, and you will
know exactly why it doesn't have to.

!!! abstract "What you can skip"
    You already know that a nonlinear equation can have zero, one, or several
    roots, and that finding every root of $g(\theta)=c$ is a fundamentally
    different (and often harder) task than evaluating $g$ once. Skip any
    refresher on "root-finding is a search, not algebra." What is not
    boilerplate: the specific physical content packed into $\boldsymbol\alpha$
    (the distance-ratio scaling that erases cosmology from its shape, Chapter
    16's payoff, cashed in here), the singular isothermal sphere's own
    doubles-versus-singles rule (worth deriving once by hand — it is not the
    "always two images" folklore you may have half-remembered), and why
    `gigalens`'s own image renderer never actually inverts the equation this
    chapter is named after (["Connect to the repo"](#connect)).

## The lens equation { #the-lens-equation }

Chapter 16 derived the physical bend angle $\hat{\boldsymbol\alpha}$ that a
mass produces in general relativity, and the thin-lens approximation that
lets a real, extended 3-D mass be treated as a single deflection evaluated at
one plane ([Ch. 16](16-deflection.md#the-thin-lens)). Folding in the
angular-diameter distances that convert a physical bend angle into an angular
one produces a *scaled* (or "reduced") deflection,

$$
\boldsymbol\alpha(\boldsymbol\theta) \;\equiv\; \frac{D_{\mathrm{ds}}}{D_{\mathrm s}}\,
\hat{\boldsymbol\alpha}\bigl(D_{\mathrm d}\boldsymbol\theta\bigr),
$$

where $D_{\mathrm{ds}}$ is the lens-to-source angular diameter distance — *not*
$D_{\mathrm s}-D_{\mathrm d}$
([Ch. 15](15-distances.md#distances-do-not-add): the two differ by a factor of
$2.3$ at $z_l=0.5,\ z_s=2.0$, and nothing about that gotcha goes away just
because it's buried inside $\boldsymbol\alpha$ now). Every distance factor
lives inside this one definition, which is exactly why `site/guide_src/lensing.py`'s
own module docstring can say "the lens-modeling likelihood in this repo is
entirely angular — no cosmology enters it" (`site/guide_src/lensing.py:9-12`):
once $\boldsymbol\alpha$ is built, it is a pure function of $\boldsymbol\theta$,
in arcsec, with every $D$ already folded away.

With that scaling in hand, the lens equation is a statement about where light
*appears* to come from versus where it *actually* came from: the apparent,
observed position $\boldsymbol\theta$ and the true source position
$\boldsymbol\beta$ differ by exactly the deflection accumulated along the way,

$$
\boldsymbol\beta = \boldsymbol\theta - \boldsymbol\alpha(\boldsymbol\theta). \label{eq:lenseq}
$$

Every symbol in $\eqref{eq:lenseq}$ is an angle. $\boldsymbol\theta$ is where
you point a telescope; $\boldsymbol\beta$ is where the source would be if
nothing were in the way; $\boldsymbol\alpha(\boldsymbol\theta)$ is what the
mass along that line of sight cost you, in arcsec, to get from one to the
other.

Differentiate $\eqref{eq:lenseq}$ once and you get the object the rest of
Part IV spends three chapters decomposing, the **lens Jacobian**

$$
A(\boldsymbol\theta) \;\equiv\; \frac{\partial\boldsymbol\beta}{\partial\boldsymbol\theta}
\;=\; I - \frac{\partial\boldsymbol\alpha}{\partial\boldsymbol\theta}. \label{eq:lensjac}
$$

[Ch. 6](06-vector-calculus.md#poisson-for-lensing) already leaned on exactly
this definition — using $\boldsymbol\alpha=\nabla\psi$ and a trace argument —
to show $A=I-H_\psi=(1-\kappa)I-\Gamma$; [Ch. 5](05-linear-algebra.md#symmetric-2x2)
diagonalizes that split, and [Ch. 18](18-magnification.md#magnification-is-a-jacobian)
takes its determinant to get magnification. This chapter needs $A$ for
something more basic first: whether $\eqref{eq:lenseq}$ has one solution at
all, or more than one.

If $\boldsymbol\alpha$ happened to be *linear* in $\boldsymbol\theta$, the
question would have a boring answer. One of this repo's own deflection fields
genuinely is: external shear, `L.shear_deflection`
(`site/guide_src/lensing.py:93-95`),
$\boldsymbol\alpha(\boldsymbol\theta)=(\gamma_1\theta_1+\gamma_2\theta_2,\
\gamma_2\theta_1-\gamma_1\theta_2)$, is linear in $\boldsymbol\theta$ by
construction. For a linear $\boldsymbol\alpha$, $A$ in $\eqref{eq:lensjac}$ is
a *constant* matrix, $\eqref{eq:lenseq}$ is a linear equation, and (whenever
$A$ is invertible) it has exactly one solution,
$\boldsymbol\theta=A^{-1}\boldsymbol\beta$, for every $\boldsymbol\beta$.
Shear alone never multiplies an image; it only shifts and stretches the one
you already have. Every real lens model in this repo pairs shear with an EPL
or SIE mass profile precisely because those deflections are *not* linear —
and nonlinearity is the entire reason strong lensing can produce more than
one image in the first place.

!!! tip "You already know this"
    Given $\boldsymbol\theta$, computing $\boldsymbol\beta$ from
    $\eqref{eq:lenseq}$ is one function evaluation — this repo's own image
    renderer does exactly this, looping over every pixel $\boldsymbol\theta$
    in the image plane and evaluating $\boldsymbol\beta(\boldsymbol\theta)$
    directly (`.../gigalens/jax/simulator.py:53-59`, cited in full in
    ["Connect to the repo"](#connect)). Given $\boldsymbol\beta$, finding
    *every* $\boldsymbol\theta$ that produced it is a different and harder
    problem: root-finding on a map that need not be one-to-one. That is the
    same asymmetry as sampling a generative model — running the generator
    forward is cheap — versus finding every latent that could have produced a
    given observation, which may have zero, one, or many answers depending on
    whether the generator happens to be injective there.

For a circularly symmetric mass — the singular isothermal sphere (SIS) of
Chapter 16 — $\boldsymbol\alpha$ always points radially, with a magnitude that
depends only on $|\boldsymbol\theta|$. Put the source on the positive
$\theta_1$-axis at $\beta_0\ge0$; every image then lies on that same axis too,
so the full 2-D problem collapses to one real equation in one real unknown
$\theta$, where a *negative* $\theta$ is shorthand for "the opposite side of
the lens center." `L.sis_deflection` (`site/guide_src/lensing.py:54-62`),
evaluated along $\theta_2=0$, is exactly $\alpha_1(\theta)=\theta_{\mathrm
E}\,\mathrm{sign}(\theta)$: constant modulus, direction only. $\eqref{eq:lenseq}$
becomes the scalar equation

$$
\beta_0 = \theta - \theta_{\mathrm E}\,\mathrm{sign}(\theta).
$$

Plot the right-hand side against $\theta$ and read off every $\theta$ where
the curve crosses the horizontal line $\beta=\beta_0$: that is solving the
lens equation *graphically*.

<figure markdown="span">
  ![Solving beta = theta - alpha graphically for an SIS: the curve theta - alpha(theta) crosses a horizontal beta line at two points](figures/ch17-lens-equation-light.svg#only-light){ width="90%" }
  ![Solving beta = theta - alpha graphically for an SIS: the curve theta - alpha(theta) crosses a horizontal beta line at two points](figures/ch17-lens-equation-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 17.1.** The SIS lens equation
  $\beta = \theta - \alpha(\theta)$ for $\theta_{\mathrm E}=1''$, plotted as a
  curve against $\theta$. A source at $\beta_0=0.4''$ (dashed line) crosses
  that curve at the two marked points: two images, one solution each on
  either side of the lens center. Nothing here is drawn freehand — the curve
  is $\theta - \theta_{\mathrm E}\,\mathrm{sign}(\theta)$, evaluated on a grid
  and plotted.</figcaption>
</figure>

Read the same answer off algebra by splitting on the sign of $\theta$. For
$\theta>0$: $\beta_0=\theta-\theta_{\mathrm E}$, so
$\theta=\beta_0+\theta_{\mathrm E}$ — positive whenever $\beta_0\ge0$, so this
root always exists. For $\theta<0$: $\beta_0=\theta+\theta_{\mathrm E}$, so
$\theta=\beta_0-\theta_{\mathrm E}$ — negative only if $\beta_0<\theta_{\mathrm
E}$. At $\theta_{\mathrm E}=1''$, $\beta_0=0.4''$ (Figure 17.1's own numbers),
the two roots are

$$
\theta = 1.4'' \quad\text{and}\quad \theta = -0.6''
$$

<!-- check: ch17.image_1 = 1.4 ± 1e-6 --><!-- check: ch17.image_2 = -0.6 ± 1e-6 -->,
two images: a **double**. `worked_examples.py`'s `ch17_lens_equation` checks
both against a numerical solver — `scipy.optimize.brentq`, bracketing each
branch of the *same* `L.sis_deflection` used above, not a reimplementation —
and finds agreement to within $10^{-8}$ on both
<!-- check: ch17.root_diff_1 = 0.0 ± 1e-8 --><!-- check: ch17.root_diff_2 = 0.0 ± 1e-8 -->,
essentially floating-point exact. For the SIS, algebra and root-finding agree
because algebra is available at all; [Ch. 22](22-inference.md#the-forward-model)'s
production EPL+shear model has no closed-form inverse, and — per the tip above
— `gigalens` never needs one, because it never solves $\eqref{eq:lenseq}$ for
$\theta$ in the first place.

One more fact falls out of the same two roots for free. Their separation is

$$
|\theta_1-\theta_2| = (\beta_0+\theta_{\mathrm E}) - (\beta_0-\theta_{\mathrm E}) = 2\theta_{\mathrm E}
$$

<!-- check: ch17.separation = 2.0 ± 1e-6 -->, and $\beta_0$ cancels
completely. For an SIS double, the image separation measures $2\theta_{\mathrm
E}$ directly, regardless of exactly where inside the Einstein radius the
source sits. That cancellation is the algebraic fact underneath
[Chapter 19](19-einstein-radius.md#what-theta-e-is)'s Einstein-radius
measurements: an image separation is a distance you can rule with a caliper on
a real image, and it converts to $\theta_{\mathrm E}$ by nothing harder than
dividing by two.

## Multiple images { #multiple-images }

Ask when $\eqref{eq:lenseq}$ can have more than one root for a fixed
$\boldsymbol\beta$. In 1-D, the right-hand side $g(\theta)=\theta-\alpha(\theta)$
has exactly one root per value of $\beta$ whenever $g$ is monotonic — a
strictly increasing (or decreasing) function crosses any horizontal line
exactly once, by definition of "crosses a line." Its derivative is
$g'(\theta)=1-\alpha'(\theta)$, which by $\eqref{eq:lensjac}$ is nothing but
$A$ itself in one dimension. So $g$ stops being monotonic exactly where $A$
changes sign — where $\alpha'(\theta)$ crosses $1$. That is not a coincidence
offered on faith: $A=0$ is precisely
[Chapter 18](18-magnification.md#critical-curves)'s definition of a critical
curve, and this section's whole account of multiple imaging is the 1-D,
un-signed-determinant preview of that chapter's 2-D one. A deflection has to
bend "hard enough somewhere" ($\alpha'>1$) for $g$ to fold and produce more
than one root; a deflection with $\alpha'<1$ everywhere can only ever produce
a single image, whatever $\boldsymbol\beta$ is.

Apply that to the SIS split above: two roots exist only for
$|\beta_0|<\theta_{\mathrm E}$ — source *inside* the Einstein radius — and only
one for $|\beta_0|\ge\theta_{\mathrm E}$. Push the same source outward, to
$\beta_0=1.5''$ with $\theta_{\mathrm E}=1''$ still, and the major image
survives at $\theta=\beta_0+\theta_{\mathrm E}=2.5''$
<!-- check: ch17.single_image_theta = 2.5 ± 1e-6 -->, but the minor-image
branch's candidate, $\theta=\beta_0-\theta_{\mathrm E}=0.5''$
<!-- check: ch17.invalid_branch_candidate = 0.5 ± 1e-6 -->, is *not* negative —
it contradicts the assumption ($\theta<0$) it was derived under. There is no
second root, full stop; the minor image does not merely fade, it is not a
solution. `worked_examples.py --show ch17` confirms this the hard way, not
just algebraically: asking `scipy.optimize.brentq` to bracket a sign change on
that branch raises `ValueError` — there genuinely is no root to find
<!-- check: ch17.second_root_correctly_absent = 1.0 ± 0.0 -->, which is a
different and stronger statement than "a very faint one."

Push $\beta_0$ the other way, to exactly $0$, and the two 1-D roots collide at
$\pm\theta_{\mathrm E}$ — but the 1-D reduction hides what really happens at
exact alignment. With a circularly symmetric lens and the source dead-center
behind it, *every* azimuthal angle around the circle of radius
$\theta_{\mathrm E}$ is an equally valid image, because the full problem has a
continuous rotational symmetry that choosing "the $\theta_1$-axis" silently
threw away. The image is not two points; it is a full circle, the **Einstein
ring** this repository's own reproductions are named after.

Three words cover almost every strong lens actually observed: a **double**
(two images, the generic case above), a **ring** (the measure-zero
perfectly-aligned case), or a **quad** (four images). A quad needs more than
circular symmetry: [Chapter 18](18-magnification.md#caustics)'s astroid
caustic, produced by an elliptical mass ($q<1$, SIE or EPL —
[Ch. 20](20-profiles.md#ellipticity)), is where a source can sit inside a
region bounded by *two* nested folds and pick up two extra roots. A classical
result in lensing theory (Burke 1981) sharpens this further: for any smooth,
*non-singular* convergence, the total image count is always odd. The SIS
above does not violate that theorem — it is exempt from it, because the SIS
is deliberately singular ($\kappa_{\mathrm{SIS}}=\theta_{\mathrm E}/2\theta$
diverges as $\theta\to0$, [Ch. 6](06-vector-calculus.md#poisson-for-lensing)).
The theorem's "missing" odd image sits, in principle, exactly at that
singularity, arbitrarily demagnified; this repo's own `sie_deflection`
(`site/guide_src/lensing.py:65-78`) adds a tiny core radius $s$ purely to keep
that division well-defined in floating point, not because the model asserts a
physically resolved core. That is the reason surveys report doubles and quads
— even counts — rather than the odd counts a smoother mass distribution would
formally guarantee.

## The $\psi$–$\alpha$–$\kappa$ trio { #the-psi-alpha-kappa-trio }

Three objects, one potential, a loop that started in
[Ch. 6](06-vector-calculus.md#poisson-for-lensing) and closes here.

!!! note "The trio, assembled"
    | Object | Defining relation | Reads as | Chapter |
    |---|---|---|---|
    | $\psi(\boldsymbol\theta)$ | sourced by mass via the $\ln r$ Green's function | the lensing potential | Ch. 6 |
    | $\boldsymbol\alpha=\nabla\psi$ | first derivative of $\psi$ | the deflection; bends light | Ch. 6, Ch. 16 |
    | $\kappa=\tfrac12\nabla^2\psi$ | trace of the second derivative | the mass, in $\Sigma_{\mathrm{cr}}$ units | Ch. 6, Ch. 15 |
    | $\boldsymbol\beta=\boldsymbol\theta-\boldsymbol\alpha$ | $\eqref{eq:lenseq}$, this chapter | maps an image to its source | Ch. 17 |
    | $A=\partial\boldsymbol\beta/\partial\boldsymbol\theta=(1-\kappa)I-\Gamma$ | $\eqref{eq:lensjac}$, split in [Ch. 5](05-linear-algebra.md#symmetric-2x2) | governs multiplicity here, magnification in Ch. 18 | Ch. 5, Ch. 17, Ch. 18 |

    Read the table top to bottom and you have this repository's entire
    strong-lensing forward model in five lines: mass sources a potential, the
    potential's gradient bends light, the lens equation turns that bend into
    an observed position, and the *same* potential's second derivative — split
    into an isotropic part $\kappa$ and a traceless part $\Gamma$ — is exactly
    the Jacobian whose folding (established this chapter for the SIS)
    [Chapter 18](18-magnification.md#magnification-is-a-jacobian) turns into a
    signed magnification.

None of this repo's production code ever materializes $\psi$ itself. The
`gigalens` EPL and SIE profiles hand you closed-form $\boldsymbol\alpha$
directly — someone did the Green's-function convolution by hand, once, for
these specific profiles, and the code inherits the answer without ever writing
$\psi(\boldsymbol\theta)$ down. What makes that shortcut trustworthy is
exactly the curl-free identity [Ch. 6](06-vector-calculus.md#divergence-laplacian)
proved: $\nabla\times(\nabla\psi)\equiv0$ for any $\psi$, so any candidate
$\boldsymbol\alpha$ that failed the corresponding check
($\partial\alpha_{\theta_1}/\partial\theta_2=\partial\alpha_{\theta_2}/\partial\theta_1$)
could not have come from a potential at all — and would not be a legitimate
deflection to plug into $\eqref{eq:lenseq}$, however smooth it looked on
paper. The trio is not three independent facts about lensing; it is one
constraint ($\psi$ exists) with three readouts.

## Connect to the repo { #connect }

- `site/guide_src/lensing.py:54-62` (`sis_deflection`) — the SIS deflection
  this chapter roots, both algebraically and with `scipy.optimize.brentq`.
- `site/guide_src/lensing.py:93-95` (`shear_deflection`) — the one deflection
  field in this repo that is exactly linear, hence provably incapable of
  multiplying an image on its own.
- `site/guide_src/lensing.py:101-123` (`lens_jacobian`) — $\eqref{eq:lensjac}$,
  computed by central differences on the lens equation itself, exactly as
  defined here.
- `site/guide_src/figures.py:120-139` (`lens_equation_1d`) — Figure 17.1's
  generator; change `beta0` and every number this chapter derives changes with
  it, because the figure and the algebra above read the same two lines.
- [`reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/simulator.py:53-59`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/simulator.py#L53)
  (`_beta`) — the production forward direction of $\eqref{eq:lenseq}$: loop
  over every image-plane pixel $\boldsymbol\theta$, evaluate
  $\boldsymbol\beta(\boldsymbol\theta)$ directly, sum over every mass
  component. No root-finding anywhere in sight, because rendering an image
  never requires inverting the lens equation — only evaluating it.
- `site/guide_src/worked_examples.py --show ch17` reproduces every tagged
  number in this chapter, including the multiplicity check that shows the
  minor image at $\beta_0=1.5''$ has no root to find, not merely a faint one.

## Exercises { #exercises }

??? question "Exercise 17.1 — solve the lens equation by hand"
    Using Figure 17.1's own numbers, $\theta_{\mathrm E}=1''$,
    $\beta_0=0.4''$: split the SIS lens equation
    $\beta_0=\theta-\theta_{\mathrm E}\,\mathrm{sign}(\theta)$ on the sign of
    $\theta$ and solve each branch. Confirm both roots against
    `worked_examples.py --show ch17`.

    ??? success "Solution"
        For $\theta>0$: $0.4=\theta-1\Rightarrow\theta=1.4$, and $1.4>0$, so
        this root is valid. For $\theta<0$: $0.4=\theta+1\Rightarrow\theta=-0.6$,
        and $-0.6<0$, so this root is valid too. Two roots, a double:
        $\theta=1.4''$ and $\theta=-0.6''$
        <!-- check: ch17.image_1 = 1.4 ± 1e-6 --><!-- check: ch17.image_2 = -0.6 ± 1e-6 -->,
        matching `ch17_lens_equation`'s `image_1` and `image_2` exactly.

??? question "Exercise 17.2 — the separation is invariant to where the source sits"
    Redo Exercise 17.1 with $\theta_{\mathrm E}=1''$ still, but
    $\beta_0=0.9''$ instead of $0.4''$. Find the two new image positions and
    their separation. What changed, and what didn't?

    ??? success "Solution"
        For $\theta>0$: $\theta=0.9+1=1.9$. For $\theta<0$:
        $\theta=0.9-1=-0.1$, still negative, so still a valid double. Both
        image positions moved (from $1.4,-0.6$ to $1.9,-0.1$), but the
        separation is $1.9-(-0.1)=2.0$ — unchanged. That is not a coincidence
        of this particular pair of numbers: the derivation in
        ["The lens equation"](#the-lens-equation) showed
        $|\theta_1-\theta_2|=2\theta_{\mathrm E}$ with $\beta_0$ cancelling
        algebraically, so the answer *must* equal the same
        <!-- check: ch17.separation = 2.0 ± 1e-6 --> this chapter already
        computed at $\beta_0=0.4$, for any $\beta_0$ inside the Einstein
        radius. Image separation measures $\theta_{\mathrm E}$; it does not
        measure where the source is.

??? question "Exercise 17.3 — when does the minor image disappear?"
    Still with $\theta_{\mathrm E}=1''$: at what value of $\beta_0$ does the
    SIS stop producing two images? For $\beta_0=1.5''$, find the surviving
    image's position, and show algebraically why the second branch has no
    valid solution (don't just say "it's very faint").

    ??? success "Solution"
        From the split in ["Multiple images"](#multiple-images), the
        $\theta<0$ branch requires $\beta_0<\theta_{\mathrm E}=1$; at
        $\beta_0=1.5$ that condition fails, so the transition happens at
        $\beta_0=\theta_{\mathrm E}=1''$ exactly. The surviving (major) image
        is at $\theta=\beta_0+\theta_{\mathrm E}=2.5''$
        <!-- check: ch17.single_image_theta = 2.5 ± 1e-6 -->. The candidate
        for the other branch is $\theta=\beta_0-\theta_{\mathrm E}=0.5''$
        <!-- check: ch17.invalid_branch_candidate = 0.5 ± 1e-6 -->, which is
        *not* negative — it violates the $\theta<0$ assumption used to derive
        it, so it is not a solution of the original equation at all, not a
        demagnified one. `worked_examples.py`'s `ch17_lens_equation` confirms
        this by trying to bracket a root there with `scipy.optimize.brentq`
        and getting a `ValueError` (no sign change) instead of a tiny number
        <!-- check: ch17.second_root_correctly_absent = 1.0 ± 0.0 -->.

??? question "Exercise 17.4 — close the trio: from $\psi$ to images, in one pass"
    [Ch. 6, Exercise 6.3](06-vector-calculus.md#exercises) gave the SIS
    potential $\psi_{\mathrm{SIS}}(\theta)=\theta_{\mathrm E}\,\theta$ and
    found $\nabla\psi_{\mathrm{SIS}}=\theta_{\mathrm E}\,\hat\theta$ (constant
    modulus $\theta_{\mathrm E}$, radially outward). Starting *only* from that
    gradient — not from `L.sis_deflection` — rebuild the SIS lens equation
    along the $\theta_1$-axis and re-derive this chapter's two image positions
    at $\theta_{\mathrm E}=1''$, $\beta_0=0.4''$.

    ??? success "Solution"
        Restricted to the $\theta_1$-axis, "radially outward at constant
        modulus $\theta_{\mathrm E}$" means $\alpha_1(\theta)=\theta_{\mathrm
        E}$ for $\theta>0$ and $\alpha_1(\theta)=-\theta_{\mathrm E}$ for
        $\theta<0$ — i.e. $\alpha_1(\theta)=\theta_{\mathrm
        E}\,\mathrm{sign}(\theta)$, exactly the deflection this chapter used,
        now obtained purely from $\eqref{eq:lensjac}$'s partner identity
        $\boldsymbol\alpha=\nabla\psi$ rather than from `L.sis_deflection`
        directly. Substituting into $\eqref{eq:lenseq}$ reproduces the same
        scalar equation, $\beta_0=\theta-\theta_{\mathrm
        E}\,\mathrm{sign}(\theta)$, whose roots were already found in Exercise
        17.1: $\theta=1.4''$ and $\theta=-0.6''$
        <!-- check: ch17.image_1 = 1.4 ± 1e-6 --><!-- check: ch17.image_2 = -0.6 ± 1e-6 -->.
        The trio table's top row and bottom row meet: a potential written down
        in one line in Chapter 6 predicts, by nothing but differentiation and
        this chapter's algebra, exactly where the two images of a real source
        would land.
