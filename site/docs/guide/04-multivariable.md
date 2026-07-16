# 4. Gradient, Jacobian, Hessian: det J is an area scaling

This chapter takes three objects you already compute — the gradient, the
Jacobian, the Hessian — and asks what the determinant of a Jacobian actually
*means*. The answer is not abstract linear algebra. $\det J$ is a literal area
(or volume) ratio, and this repository computes that exact ratio, under three
different names, in three different files: it is the lensing magnification
$\mu = 1/|\det A|$ that Chapter 18 uses to explain why a background galaxy
looks brighter than it should; it is the $\log|\det J_T|$ term every
normalizing flow you have trained already adds to a log-density; and it is the
$-\tfrac12\log\det A$ Occam penalty buried in this campaign's marginal
likelihood. By the end of this chapter you will be able to derive that
reciprocal-determinant law from scratch, not just recognize it when you see
it.

!!! abstract "What you can skip"
    You already build and differentiate Jacobians for a living — every layer
    of a neural net is a Jacobian-vector or vector-Jacobian product, and you
    already invoke the change-of-variables theorem every time you compute
    $\log q(x) = \log p(u) - \log|\det J|$ under a bijector. Skip the review
    of *how* to compute a multivariable derivative if you want. What is not
    boilerplate here: the literal geometric reading of $\det J$ as an area
    ratio (["det J as an area scaling factor"](#det-j-as-area-scaling)), and
    the specific three places this repository's code computes it
    (["The Log-Det Ledger"](#the-log-det-ledger)).

## Gradient, Jacobian, Hessian { #gradient-jacobian-hessian }

Chapter 2 defined the derivative of a scalar function of one variable as the
best local *linear* approximation: $f(x+h) \approx f(x) + f'(x)\,h$. Nothing
about that idea depends on there being one input or one output. For a
vector-valued function of several variables, $F : \mathbb{R}^n \to
\mathbb{R}^m$, the identical statement reads

$$
F(\mathbf{x} + \mathbf{h}) \;\approx\; F(\mathbf{x}) + J_F(\mathbf{x})\,\mathbf{h},
\qquad
(J_F)_{ij} = \frac{\partial F_i}{\partial x_j}. \label{eq:jaclin}
$$

$J_F$, the **Jacobian**, is an $m\times n$ matrix of partial derivatives, and
$\eqref{eq:jaclin}$ says exactly what Chapter 2 said: near $\mathbf{x}$, the
nonlinear map $F$ is well-approximated by the linear map
$\mathbf{h} \mapsto J_F(\mathbf{x})\,\mathbf{h}$. When $m=1$ (a scalar-valued
function), $J_F$ is a single row, and its transpose is the **gradient**
$\nabla f$ you already know as "the direction of steepest ascent." The
gradient is a special case of the Jacobian, not a different object.

Differentiate the gradient once more and you get the **Hessian**, the
$n\times n$ matrix of second partials, $H_{ij} = \partial^2 f/\partial
x_i\,\partial x_j$. It extends the Taylor expansion to second order exactly as
Chapter 2 previewed:

$$
f(\mathbf{x}+\mathbf{h}) \;\approx\; f(\mathbf{x}) + \nabla f(\mathbf{x})\cdot\mathbf{h}
+ \tfrac12\,\mathbf{h}^{\!\top} H(\mathbf{x})\,\mathbf{h}. \label{eq:hess}
$$

Two facts about $H$ matter more than its definition. First, for any function
whose second partials are continuous, mixed partials commute
($\partial^2 f/\partial x_i\partial x_j = \partial^2 f/\partial x_j\partial
x_i$ — Schwarz's theorem), so **the Hessian is always symmetric**. Second, a
zero gradient tells you that you are at a *stationary* point — it says nothing
about whether that point is a minimum, a maximum, or a saddle. Only the
Hessian's eigenvalues answer that (Chapter 2's own argument for why a
second-order model is
[necessary](02-derivatives.md#why-second-order); Chapter 5 makes it precise
for exactly this repository's own [saddle](05-linear-algebra.md#definiteness-and-saddles)).

Here is why the Hessian is not a side character in this guide. The Jacobian of
a *gradient field* is automatically a Hessian: if $F = \nabla\psi$ for some
scalar potential $\psi$, then

$$
(J_F)_{ij} = \frac{\partial F_i}{\partial x_j}
= \frac{\partial}{\partial x_j}\left(\frac{\partial \psi}{\partial x_i}\right)
= \frac{\partial^2\psi}{\partial x_i\,\partial x_j} = H_{ij}(\psi).
$$

Chapter 6 derives why the deflection field of a gravitational lens is exactly
such a gradient, $\boldsymbol\alpha = \nabla\psi$, for a scalar lensing
potential $\psi$
([Ch. 6](06-vector-calculus.md#poisson-for-lensing)). That single fact is why
the Jacobian this repository differentiates at every image position — the
matrix that Chapters 5, 18 and 20 spend three chapters decomposing — is
guaranteed symmetric before you ever compute a single entry. A symmetric
matrix has real eigenvalues and orthogonal eigenvectors; that is what makes
the clean split into an isotropic part (convergence) and a shear part possible
at all, and it is not a coincidence — it follows from $\eqref{eq:jaclin}$
applied to a gradient field.

## The change-of-variables theorem { #change-of-variables }

Single-variable calculus has a substitution rule:
$\int f(g(x))\,g'(x)\,dx = \int f(u)\,du$ with $u = g(x)$. The $g'(x)$ factor
is a *local stretching factor* — it says how much a small interval $dx$ is
stretched or compressed when mapped through $g$. The multivariable
generalization keeps that reading and replaces the scalar $g'(x)$ with the
determinant of the Jacobian: for a smooth, invertible map $T: U \to V$,

$$
\int_{V} g(\mathbf{y})\, d^n y \;=\; \int_{U} g\bigl(T(\mathbf{x})\bigr)\,
\bigl|\det J_T(\mathbf{x})\bigr|\, d^n x. \label{eq:cov}
$$

$\eqref{eq:cov}$ is the **change-of-variables theorem**. The next section
derives *why* $|\det J_T|$ is the right local factor; for now, notice what
$\eqref{eq:cov}$ says about *densities*. If $\mathbf{x}$ has probability
density $p_U$ and $\mathbf{y} = T(\mathbf{x})$ has density $p_X$, conservation
of probability mass on matching infinitesimal patches
($p_U(\mathbf{x})\,d^nx = p_X(\mathbf{y})\,d^ny$) combined with
$\eqref{eq:cov}$'s local scaling gives

$$
p_U(\mathbf{x}) = p_X\bigl(T(\mathbf{x})\bigr)\,\bigl|\det J_T(\mathbf{x})\bigr|,
\qquad\text{i.e.}\qquad
\log p_U(\mathbf{x}) = \log p_X\bigl(T(\mathbf{x})\bigr) + \log\bigl|\det J_T(\mathbf{x})\bigr|.
$$

!!! tip "You already know this"
    That last line is not an analogy — it is the identical arithmetic a
    normalizing flow performs to evaluate a pulled-back density. Compare it to
    the docstring of `reproductions/claude-giga-lens/cgl/flows.py:20`, which
    writes the NeuTra pullback used by this campaign's samplers as
    `logp_u(u) = logp_target(T(u)) + log|det J_T(u)|`. You have been computing
    $\eqref{eq:cov}$ every time you called `.log_prob()` on a bijected
    distribution; you were just calling the theorem by a different name.

## det J as an area scaling factor { #det-j-as-area-scaling }

Where does the $|\det J|$ factor in $\eqref{eq:cov}$ come from? Start with a
*linear* map, $\mathbf{y} = A\mathbf{x}$, and ask what happens to the area of
a unit square under it. Two edge vectors of the square, $(1,0)$ and $(0,1)$,
map to the two columns of $A$. The area of the parallelogram spanned by two
vectors $(a_{11},a_{21})$ and $(a_{12},a_{22})$ is
$|a_{11}a_{22} - a_{12}a_{21}| = |\det A|$ — the shoelace formula applied to a
parallelogram *is* the definition of a $2\times 2$ determinant. So for a
linear map, "area scales by $|\det A|$" is not a theorem to prove; it is what
the determinant was built to compute.

A general smooth map $T$ is not linear, but $\eqref{eq:jaclin}$ says it is
*locally* linear, with $J_T(\mathbf{x})$ as the local linear map. Shrink a
patch of the domain down to an infinitesimal square at $\mathbf{x}$: its image
is, to leading order, a parallelogram with area $|\det J_T(\mathbf{x})|$ times
the original. Integrate that local scaling factor over every patch in $U$ and
you recover $\eqref{eq:cov}$ exactly. The change-of-variables theorem is
linearization ([Ch. 2](02-derivatives.md#derivative-as-linear-map)), applied
pointwise, summed over a region.

<figure markdown="span">
  ![A unit square pushed through a linear map A; the image's area is |det A| times the square's](figures/ch04-det-j-area-light.svg#only-light){ width="90%" }
  ![A unit square pushed through a linear map A; the image's area is |det A| times the square's](figures/ch04-det-j-area-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 4.1.** A unit square (left) pushed through the linear
  map $A = \begin{pmatrix}0.6 & 0.25\\ 0.1 & 0.85\end{pmatrix}$ into the parallelogram on the
  right. Nothing is drawn freehand: the right-hand shape is literally $A$ applied to the
  square's four corners. The annotated ratio is $\det A$
  <!-- check: ch04.det_A = 0.485 ± 1e-9 -->, and it is the same number whether you compute it
  from the $2\times2$ determinant formula or from the shoelace-formula area of the polygon on
  the right.</figcaption>
</figure>

Compute both routes yourself. The determinant of $A$ above is
$0.6\times 0.85 - 0.25\times 0.1 = 0.51 - 0.025 = 0.485$
<!-- check: ch04.det_A = 0.485 ± 1e-9 -->. Independently, push the square's four corners
$(0,0), (1,0), (1,1), (0,1)$ through $A$ and apply the shoelace formula to the resulting
quadrilateral: the area comes out to $0.485$
<!-- check: ch04.image_area_shoelace = 0.485 ± 1e-9 -->, matching the determinant to machine
precision, because the shoelace formula and the $2\times2$ determinant are computing the same
quantity two different ways. `worked_examples.py --show ch04` reproduces both numbers directly.

Two immediate uses of that reciprocal direction. **Magnification**: Chapter 18
defines the lensing Jacobian $A \equiv \partial\boldsymbol\beta/\partial\boldsymbol\theta$ —
how a source-plane displacement $\boldsymbol\beta$ responds to an image-plane one
$\boldsymbol\theta$ — and shows that surface brightness is conserved along a
light ray, so the flux ratio between an image and its true, unlensed source is
exactly the *inverse* area ratio,
$\mu \equiv 1/|\det A|$
([Ch. 18](18-magnification.md#magnification-is-a-jacobian)). A determinant
smaller than one in the natural ($\boldsymbol\theta\to\boldsymbol\beta$)
direction is a magnification bigger than one in the observed direction — the
same $|\det J|$ arithmetic this section just verified, applied in reverse.
**Normalizing flows**: a flow's *forward* Jacobian determinant is exactly the
$J_T$ of $\eqref{eq:cov}$, and its log-density correction is $+\log|\det J_T|$
on the base side, as derived above. Both are the identical operation; only the
sign convention and which side of the map you call "base" differ.

## The Log-Det Ledger { #the-log-det-ledger }

This repository computes $\log|\det(\,\cdot\,)|$ of a $2\times2$ (or larger)
matrix in three unrelated-looking places. This chapter opens the ledger with
the first two; Chapter 8 adds the third; Chapter 23 closes it by showing all
three are, line for line, the same computation.

!!! note "Log-Det Ledger — rows 1 and 2, opened here"
    | # | Costume | Formula | Where |
    |---|---|---|---|
    | 1 | Lensing magnification | $\mu = 1/\lvert\det A\rvert$, $A=\partial\boldsymbol\beta/\partial\boldsymbol\theta$ | `site/guide_src/lensing.py:139` |
    | 2 | Normalizing-flow pullback density | $\log p_U(\mathbf u) = \log p_X(T(\mathbf u)) + \log\lvert\det J_T(\mathbf u)\rvert$ | `reproductions/claude-giga-lens/cgl/flows.py:20` |
    | 3 | Gaussian-evidence Occam term (opens in [Ch. 8](08-probability.md#evidence-and-nats)) | $-\tfrac12\log\det A$ | `reproductions/claude-giga-lens/cgl/marg.py:48,51` |

    Rows 1 and 2 are the *same* theorem — $\eqref{eq:cov}$ — read in opposite
    directions: one turns an area ratio into a brightness ratio, the other
    turns it into a density-correction term. Row 3 is a different-looking
    object (it subtracts a log-determinant from a log-likelihood rather than
    adding one to a log-density) and Chapter 23 shows it collapses to the same
    arithmetic once you write the Gaussian evidence integral out explicitly.
    Ch. 23 [closes the ledger](23-samplers.md#closing-the-log-det-ledger).

`site/guide_src/lensing.py:139` is worth reading directly — its own docstring
already states this chapter's thesis in one line: *"This is the SAME
change-of-variables factor that a normalizing flow applies as $-\log|\det J|$
... Three log-dets, one idea."* This guide did not invent that framing; the
repository's own numerics code did.

One notation hazard, flagged now because it recurs: `reproductions/claude-giga-lens/cgl/marg.py:48`
also calls its own matrix $A$ — but that $A$ is $\tilde X^{\!\top}\tilde X + \Lambda$, the Gram
matrix of a whitened design matrix plus a ridge term, and has nothing to do with the lensing
Jacobian $A = \partial\boldsymbol\beta/\partial\boldsymbol\theta$ of row 1. The repository reuses
the letter; this guide does not, past this sentence — from here on, $A$ means the lensing
Jacobian unless a chapter says otherwise.

## Connect to the repo { #connect }

Everything in this chapter is small enough to read directly.

- `site/guide_src/lensing.py:101-146` — `lens_jacobian` builds a $2\times2$
  Jacobian numerically, by central differences, exactly as $\eqref{eq:jaclin}$
  defines it; `magnification` then takes $1/\det(\,\cdot\,)$ of it. Chapter 18
  runs this on a real deflection field and checks the numerical result against
  a closed-form SIS magnification.
- `reproductions/claude-giga-lens/cgl/flows.py:1-21` — the module docstring is
  the pullback-density derivation of this chapter's ["change-of-variables"](#change-of-variables)
  section, in the form the campaign's NeuTra and glnt samplers actually run
  (Chapter 23).
- `reproductions/claude-giga-lens/cgl/marg.py:31-55` — the third ledger
  costume, `-1/2 logdetA`, computed via a Cholesky factor
  (`logdetA = 2 * sum(log(diag(chol)))`) rather than a direct determinant,
  because a Cholesky log-determinant is numerically stable at the condition
  numbers this repository's real fits produce (Chapter 5 quantifies exactly
  how large those condition numbers get).
- `reproductions/claude-giga-lens/papers/main.tex:466-471` (`\label{eq:logl}`)
  — the typeset version of the Occam term, for when Chapter 8 sends you back
  to read it in context.
- `site/guide_src/figures.py:28-49` — Figure 4.1 above, generated (not drawn)
  by pushing a literal unit square through a literal NumPy matrix; change the
  matrix and the figure, the annotated determinant, and the caption's number
  all change together, because they are computed from the same four lines.

## Exercises { #exercises }

??? question "Exercise 4.1 — det J, two ways"
    Using $A = \begin{pmatrix}0.6 & 0.25\\ 0.1 & 0.85\end{pmatrix}$ from Figure 4.1: (a) compute
    $\det A$ directly from the $2\times2$ formula. (b) Push the unit square's four corners
    through $A$ and compute the resulting parallelogram's area with the shoelace formula,
    $\text{area} = \tfrac12\bigl|\sum_i (x_iy_{i+1} - x_{i+1}y_i)\bigr|$. Confirm the two answers
    agree, then check both against `worked_examples.py --show ch04`.

    ??? success "Solution"
        (a) $\det A = (0.6)(0.85) - (0.25)(0.1) = 0.51 - 0.025 = 0.485$.

        (b) The corners map to $(0,0)$, $(0.6, 0.1)$, $(0.85, 0.95)$, $(0.25, 0.85)$ (each corner
        $(x,y)$ maps to $(0.6x+0.25y,\ 0.1x+0.85y)$). The shoelace sum gives area $= 0.485$,
        matching (a) exactly — it must, since both computations are literally the same
        determinant, one written as a $2\times2$ formula and the other as a sum over edges.
        `worked_examples.py`'s `ch04_det_j_area` returns `det_A = 0.485` and
        `image_area_shoelace = 0.485`
        <!-- check: ch04.det_A = 0.485 ± 1e-9 --><!-- check: ch04.image_area_shoelace = 0.485 ± 1e-9 -->.

??? question "Exercise 4.2 — the reciprocal, by definition"
    Given $\det A = 0.485$ from Exercise 4.1, what is $\mu = 1/|\det A|$? If $A$ stood for an
    actual lensing Jacobian $\partial\boldsymbol\beta/\partial\boldsymbol\theta$, would the
    corresponding image be brighter or dimmer than the source that produced it?

    ??? success "Solution"
        $\mu = 1/0.485 = 2.0619$ <!-- check: ch04.mu = 2.0619 ± 1e-4 -->. Since $\mu > 1$, the
        image would be *brighter*: a source-plane patch of fixed size corresponds to an
        image-plane patch about twice as large (Chapter 18's full argument is that surface
        brightness is conserved, so a bigger apparent patch at the same brightness-per-area
        means more total flux). A determinant less than one in the $\boldsymbol\theta \to
        \boldsymbol\beta$ direction always means magnification greater than one in the observed
        direction — the two are reciprocals by construction, not by coincidence.

??? question "Exercise 4.3 — a gradient's Jacobian is a Hessian, and it is symmetric"
    Let $\psi(x_1,x_2) = x_1^2 x_2 + x_2^3$. Compute $\nabla\psi$, then compute the Jacobian of
    $\nabla\psi$ (treating it as a map $\mathbb{R}^2\to\mathbb{R}^2$) directly from its four
    partial derivatives. Confirm it equals the Hessian of $\psi$, and confirm it is symmetric.

    ??? success "Solution"
        $\nabla\psi = (2x_1x_2,\ x_1^2 + 3x_2^2)$. Differentiating this vector field:

        $$
        J_{\nabla\psi} = \begin{pmatrix}
        \partial(2x_1x_2)/\partial x_1 & \partial(2x_1x_2)/\partial x_2 \\
        \partial(x_1^2+3x_2^2)/\partial x_1 & \partial(x_1^2+3x_2^2)/\partial x_2
        \end{pmatrix}
        = \begin{pmatrix} 2x_2 & 2x_1 \\ 2x_1 & 6x_2 \end{pmatrix}.
        $$

        This is exactly $H(\psi)$: $\partial^2\psi/\partial x_1^2 = 2x_2$,
        $\partial^2\psi/\partial x_2^2 = 6x_2$, and both mixed partials equal $2x_1$. The
        off-diagonal entries agree ($2x_1 = 2x_1$) because mixed partials commute for any
        polynomial (Schwarz's theorem) — the matrix is symmetric for every $(x_1,x_2)$, not just
        by luck at one point. This is the general fact Chapter 6 invokes for the lensing
        potential: $\boldsymbol\alpha = \nabla\psi$ makes the deflection field's own Jacobian
        automatically symmetric, before any lens-specific physics enters.

??? question "Exercise 4.4 — when does $\mu$ flip sign?"
    $\mu = 1/\det A$ is *signed*, not just an absolute ratio. Under what condition on $A$'s
    entries does $\det A < 0$? Given that $A = (1-\kappa)I - \Gamma$ for lensing (Chapter 5
    derives this decomposition), can you say anything yet about what a negative $\mu$ might mean
    physically? (You are not expected to fully answer this — it is Chapter 18's job. State what
    you can derive now, and what you cannot.)

    ??? success "Solution"
        For a $2\times2$ matrix, $\det A = a_{11}a_{22} - a_{12}a_{21} < 0$ exactly when the
        product of the diagonal entries is smaller than the product of the off-diagonal
        entries — geometrically, when the map reverses orientation (a right-handed basis maps to
        a left-handed one). What you *can* derive now: since $\mu = 1/\det A$ and $\det A$ can be
        negative, $\mu$ can be negative too, and nothing in this chapter's area-ratio argument
        breaks — $|\det A|$ is still the area-scaling factor, and the sign is separate
        information. What you cannot yet derive: *why* a sign flip corresponds to a genuinely
        different physical image (a parity-flipped image, one of the tell-tale signs that you
        have crossed a critical curve). That requires the actual lens equation and its multiple
        solutions, which is exactly what Chapter 17 sets up and Chapter 18
        ([parity](18-magnification.md#parity)) resolves.
