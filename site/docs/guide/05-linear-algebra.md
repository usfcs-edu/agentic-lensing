# Eigenvalues, saddles, and conditioning

Every lens Jacobian this book differentiates turns out to be a *symmetric*
$2\times2$ matrix, and a symmetric $2\times2$ matrix is one of the few objects
in mathematics whose eigenproblem you can solve completely, by hand, in four
lines. This chapter does that once and spends it three ways: reading
convergence and shear directly out of a Jacobian's eigenvalues (which sets up
[Ch. 18](18-magnification.md#critical-curves)'s critical curves), saying
precisely what a *saddle* is and why the campaign's own MAP-finder walked into
one without noticing (which sets up [Ch. 26](26-the-saddle.md)), and putting a
number — not an adjective — on how badly conditioned this repo's hardest
real posterior actually is. [Ch. 2](02-derivatives.md#why-second-order)
already showed you the crime scene; this chapter builds the tool that reads it.

!!! abstract "What you can skip"
    If you can already state and prove the spectral theorem for real symmetric
    matrices, classify a Hessian as positive/negative (semi)definite or
    indefinite from its eigenvalues, and define a matrix's condition number as
    a ratio of singular values, skim
    [Symmetric $2\times2$s](#symmetric-2x2) for the lensing-specific reading of
    $\kappa$ and shear and read
    [Definiteness and saddles](#definiteness-and-saddles) and
    [Conditioning](#conditioning) mainly for the repo's own numbers. This
    chapter reproves the $2\times2$ spectral theorem from the characteristic
    polynomial rather than citing the general $n$-dimensional version you
    already own — it is four lines, and the specific form pays for itself
    immediately.

## Symmetric $2\times2$ matrices eigendecompose in closed form { #symmetric-2x2 }

[Ch. 4](04-multivariable.md#gradient-jacobian-hessian) built the lens Jacobian
$A = \partial\boldsymbol\beta / \partial\boldsymbol\theta$ and read its
determinant as a local area ratio. This chapter asks a sharper question: not
just how much area $A$ scales, but *in which directions*, and by how much
each.

$A$ is not an arbitrary $2\times2$ matrix. Because the deflection is a
gradient, $\boldsymbol\alpha = \nabla\psi$, the lens equation gives
$A = I - \mathrm{Hess}(\psi)$, and mixed partial derivatives of a smooth
scalar commute — $\partial^2\psi/\partial\theta_1\partial\theta_2 =
\partial^2\psi/\partial\theta_2\partial\theta_1$
([Ch. 6](06-vector-calculus.md#poisson-for-lensing) states this properly). So
$a_{12}=a_{21}$ always: the lens Jacobian is *symmetric*, every time, for
every mass profile. That single fact is why the rest of this section works.

Write a general symmetric $2\times2$ matrix as

$$
A = \begin{pmatrix} a & b \\ b & d \end{pmatrix}.
$$

Its eigenvalues solve $\det(A - \lambda I) = 0$, which expands to the
characteristic polynomial

$$
\lambda^2 - (a+d)\,\lambda + (ad - b^2) = 0.
$$

The quadratic formula gives

$$
\lambda_{1,2} = \frac{a+d}{2} \;\pm\; \sqrt{\left(\frac{a-d}{2}\right)^2 + b^2}.
$$

Two things fall out of this for free, and both are the spectral theorem for
$2\times2$ matrices, proved rather than quoted. First, the quantity under the
square root is a sum of two squares, so it is never negative — $\lambda_1$
and $\lambda_2$ are always real. (In more than two dimensions the same
statement is true but needs more machinery than the quadratic formula; here
you get it for nothing.) Second, if $b=0$ the eigenvectors are the coordinate
axes, and rotating a symmetric matrix's frame to make $b=0$ is always
possible — the eigenvectors of a symmetric matrix are always orthogonal. You
will use both facts without comment for the rest of this book.

Now specialize. Write the trace of $A$ as an isotropic part and the remainder
as a traceless part:

$$
A = (1-\kappa)\, I \;-\; \Gamma,
\qquad
\Gamma = \begin{pmatrix} \gamma_1 & \gamma_2 \\ \gamma_2 & -\gamma_1 \end{pmatrix}.
$$

$\kappa$ (the trace half) is the isotropic squeeze: it scales both directions
equally. $\Gamma$ (the traceless remainder) is the *shear*: it stretches one
direction while compressing the perpendicular one. Trace zero means these two
effects cancel *to first order* — a determinant's linear-order response to a
perturbation is its trace, so a traceless perturbation costs no area to first
order — but they do not cancel exactly at finite shear; Exercise 5.1 has you
check this on the figure's own numbers. This split is exactly what `lensing.py` computes
(`kappa_gamma_from_jacobian`, `lensing.py:126-136`): $\kappa = 1 -
\tfrac12(a_{11}+a_{22})$ from the trace, $\gamma_1 = -\tfrac12(a_{11}-a_{22})$
and $\gamma_2 = -\tfrac12(a_{12}+a_{21})$ from what is left over. Because
$\Gamma$ is already traceless, its own eigenvalues are $\pm\sqrt{\gamma_1^2 +
\gamma_2^2}$ directly from the formula above (the trace term vanishes, and
$ad - b^2$ becomes $-\gamma_1^2-\gamma_2^2$). Subtracting $\Gamma$ from
$(1-\kappa)I$ therefore gives

$$
\lambda_{\mathrm t} = (1-\kappa) - \sqrt{\gamma_1^2+\gamma_2^2},
\qquad
\lambda_{\mathrm r} = (1-\kappa) + \sqrt{\gamma_1^2+\gamma_2^2}.
$$

These are named for what they turn out to mean physically: near a
roughly-circular lens, the smaller eigenvalue $\lambda_{\mathrm t}$ governs
the **tangential** direction (perpendicular to the line to the lens centre)
and the larger $\lambda_{\mathrm r}$ the **radial** direction. That naming is
not yet justified by anything in this chapter — it becomes the content of
[Ch. 18](18-magnification.md#critical-curves), where $\lambda_{\mathrm t}=0$
turns out to be exactly the condition that draws the Einstein ring.

**Worked example.** Take $\kappa=0.3$, $\gamma_1=0.3$, $\gamma_2=0$ — a lens
with both an isotropic squeeze and a shear stretch of equal size. The shear
magnitude is $\sqrt{\gamma_1^2+\gamma_2^2} = 0.3$
<!-- check: ch05.toy_shear_mag = 0.3 ± 0.001 -->, so

$$
\lambda_{\mathrm t} = 0.7 - 0.3 = 0.4,
\qquad
\lambda_{\mathrm r} = 0.7 + 0.3 = 1.0.
$$

<!-- check: ch05.eig_tangential = 0.4 ± 1e-6 -->
<!-- check: ch05.eig_radial = 1.0 ± 1e-6 -->

A unit source circle, pushed backward through $A^{-1}$ into the image plane,
comes out stretched by $1/\lambda_{\mathrm t}=2.5$
<!-- check: ch05.stretch_tangential = 2.5 ± 1e-6 --> along one axis and by
$1/\lambda_{\mathrm r}=1.0$
<!-- check: ch05.stretch_radial = 1.0 ± 1e-6 --> (unchanged) along the other —
an ellipse, not a circle, which is the entire mechanism by which a lens turns
a round source into an arc.

<figure markdown="span">
  ![A unit circle mapped by A = (1-kappa) I - Gamma, for pure convergence, pure shear, and both together](figures/ch05-kappa-gamma-eigen-light.svg#only-light){ width="90%" }
  ![A unit circle mapped by A = (1-kappa) I - Gamma, for pure convergence, pure shear, and both together](figures/ch05-kappa-gamma-eigen-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 5.1.** A unit source circle (dashed) pushed
  through $A^{-1}$ for three cases: pure convergence ($\kappa=0.3$, no shear —
  an isotropic circle, unchanged in shape), pure shear ($\kappa=0$,
  $\gamma_1=0.3$, $\gamma_2=0$), and both together (the worked example above,
  right panel: $\lambda_{\mathrm t}=0.4$, $\lambda_{\mathrm r}=1.0$). The
  panel titles, generated by `figures.py`, write the shear magnitude as a bare
  "$\gamma$" for plot-space brevity — this guide reserves that symbol for the
  density slope, so read the figure's own label as
  $\sqrt{\gamma_1^2+\gamma_2^2}$. It is a small, live example of exactly the
  overloading hazard this guide's notation contract exists to police.</figcaption>
</figure>

## Definiteness, and why a saddle is not a mode { #definiteness-and-saddles }

A symmetric matrix $A$ is **positive definite** if every eigenvalue is
positive (equivalently $\mathbf x^{\mathsf T} A\, \mathbf x > 0$ for every
$\mathbf x \neq 0$), **negative definite** if every eigenvalue is negative,
and **indefinite** if it has eigenvalues of both signs. For a $2\times2$
symmetric matrix you do not need the eigenvalues to tell which case you are
in: since $\lambda_1\lambda_2 = \det A$ and $\lambda_1+\lambda_2 =
\operatorname{tr} A$,

$$
\det A > 0,\ \operatorname{tr} A > 0 \;\Rightarrow\; \text{PD}, \qquad
\det A > 0,\ \operatorname{tr} A < 0 \;\Rightarrow\; \text{ND}, \qquad
\det A < 0 \;\Rightarrow\; \text{indefinite}.
$$

An indefinite Hessian is what a **saddle** is, in coordinates: a point where
the gradient is zero but the function curves up in some directions and down
in others — [Ch. 2, Exercise 2.3](02-derivatives.md#exercises) works
$f(x,y)=x^2-y^2$ by hand and you should have that toy example in mind for
what follows. The determinant test above is exactly why
[Ch. 4](04-multivariable.md#det-j-as-area-scaling)'s determinant machinery was
worth building before this chapter: you can diagnose a saddle from $\det A$
alone, with no eigendecomposition at all, the moment $\det A < 0$.

!!! tip "You already know this"
    "A vanishing gradient is not a verdict" is folklore in non-convex
    optimization for exactly this reason: in high dimensions, saddle points
    vastly outnumber true minima, and a first-order method can stall on one
    indefinitely. Dauphin et al. (2014), *"Identifying and attacking the
    saddle point problem in high-dimensional non-convex optimization,"* is the
    paper most ML researchers know for this. This repo's own `foundry-i` phase
    independently implemented that paper's fix — replace the Hessian $H$ by
    its *absolute* curvature $|H| = V\,\mathrm{diag}(|\lambda_i|)\,V^{\mathsf
    T}$, so a Newton-style step *descends* along every eigendirection instead
    of climbing the negative-curvature ones — in
    `reproductions/foundry-i/32_saddlefree_newton.py`, on an earlier saddle in
    this same modelling pipeline. Hold onto that formula,
    $V\,\mathrm{diag}(1/|\lambda_i|)\,V^{\mathsf T}$: [Ch.
    26](26-the-saddle.md#the-same-matrix-twice) shows it again, used for a
    different purpose, where it stops being correct.

The campaign hit the real thing directly. Its correlated-noise fit's
MAP-finding stage drove $\nabla\log p$ to zero and reported success; a
second-order check — `jax.hessian` of the log-posterior at that point
(`reproductions/claude-giga-lens/cgl/e2.py:554`, eigendecomposed at
`cgl/e2.py:557-558`) — showed the point was a **saddle**: minimum eigenvalue
$-14.85$, five of forty-six eigenvalues negative
(`reproductions/claude-giga-lens/CAMPAIGN.md`, "P1c metric-fix attempts —
2026-07-10"; main.tex,
[§6.3](../current/claude-giga-lens/index.md#sec:samplersaga)).
<!-- check: ch05.saddle_min_eig = -14.85 ± 0.01 -->
<!-- check: ch05.saddle_n_negative = 5 ± 0 -->
<!-- check: ch05.saddle_ndim = 46 ± 0 -->
A negative eigenvalue of $H$ there means the log-posterior keeps *rising* as
you move away from that point along that eigendirection — `map_polish`
reported a local *trough* of the log-posterior along that one axis, not a
peak, so nearby points along it score higher, not lower. The campaign's own
ledger catches
this concretely, on the same fit: the saddle-consistent point $\gamma=1.27$
scored $\log p = -4757$, while $\gamma=1.10$ — a direction the saddle-seeded
optimizer never explored — scored $\log p=-4683$, a full 74 nats higher.
<!-- check: ch05.gamma_at_saddle_map = 1.27 ± 0.01 -->
<!-- check: ch05.logp_at_saddle_map = -4757 ± 1 -->
<!-- check: ch05.gamma_at_true_peak = 1.10 ± 0.01 -->
<!-- check: ch05.logp_at_true_peak = -4683 ± 1 -->
<!-- check: ch05.saddle_logp_gain = 74 ± 1 -->
(Both $\gamma$ values here are a mid-campaign snapshot, superseded by [Ch.
25](25-money-number.md#the-money-number)'s converged answer; they are quoted
only to make "a saddle is not the top" concrete on data instead of a toy.) A
`map_polish` stage that stops the instant $\nabla\log p=0$ has, by
construction, no way to see any of this — exactly [Ch. 2](02-derivatives.md#why-second-order)'s
point, now with the repo's own numbers attached. [Ch.
26](26-the-saddle.md#the-map-is-a-saddle) tells the rest of the story: why
this particular saddle resisted the standard fix, and how it was eventually
sampled around rather than through.

## Conditioning: $\mathrm{cond}\sim10^{14}$ is not an abstraction { #conditioning }

For a symmetric matrix, define the **condition number** as the ratio of the
largest to the smallest eigenvalue by absolute value,
$\mathrm{cond}(A) = |\lambda_{\max}|/|\lambda_{\min}|$. (For a general,
non-symmetric matrix the definition uses singular values instead; every
matrix this book differentiates is symmetric, so the distinction never
matters again.) It measures how much $A$ can amplify a relative error: solving
$A\mathbf x = \mathbf b$, a relative perturbation of size $\varepsilon$ in
$\mathbf b$ can produce a relative perturbation as large as
$\mathrm{cond}(A)\cdot\varepsilon$ in $\mathbf x$ — and every number stored in
float64 already carries a relative rounding error on the order of machine
epsilon, $\varepsilon_{64}\approx2.22\times10^{-16}$
<!-- check: ch05.float64_eps = 2.220446e-16 ± 1e-22 -->,
whether you asked for it or not.

Float64 gives you about $-\log_{10}\varepsilon_{64}\approx15.65$ significant
decimal digits of headroom
<!-- check: ch05.float64_digits_total = 15.6536 ± 0.001 -->.
Multiplying by a condition number of $10^{14}$ spends 14 of those digits
purely amplifying input error, leaving

$$
15.65 - \log_{10}(10^{14}) \approx 1.65
$$

<!-- check: ch05.stable_digits_remaining = 1.6536 ± 0.001 -->

trustworthy decimal digits along the worst-conditioned direction — under two.
That is the sense in which "$\mathrm{cond}\sim10^{14}$" is not a figure of
speech: it is a direct statement that a naive Hessian inversion along that
direction is barely more precise than a coin flip on the second digit.

This is not a hypothetical matrix. `foundry-i`'s real-data posterior — the
same physical system this campaign's final report calls **T2**, the
$46$-dimensional marginalized real posterior of the actual DESI system
DESI–165.4754−06.0423 — is, in its own words, "ultra-ill-conditioned (cond
$\sim10^{14}$): lens-light companion Sérsic centers at $H_{ii}\sim10^{12}$,
Sérsic indices $n_{\mathrm{sersic}}$ at $\sim10^{-2}$"
(`reproductions/foundry-i/README.md:114-117`). Take those two diagonal
Hessian entries at face value and divide:

$$
\frac{H_{ii}^{\max}}{H_{ii}^{\min}} = \frac{10^{12}}{10^{-2}} = 10^{14}.
$$

<!-- check: ch05.hessian_diag_max = 1e12 ± 0 -->
<!-- check: ch05.hessian_diag_min = 1e-2 ± 0 -->
<!-- check: ch05.cond_ill_conditioned = 1e14 ± 0 -->

(These are diagonal entries in the parameters' own coordinates, not
literally the Hessian's eigenvalues — off-diagonal correlations mix
parameters together in the true eigenbasis — but for a diagonally-dominant
Hessian they set the scale, and here the naive ratio reproduces the quoted
order of magnitude exactly.) The physical story behind the two numbers is
simple: a Sérsic light centre is pinned by the data to a tiny fraction of a
pixel, so the log-posterior curves *sharply* there; the Sérsic index is
barely constrained at all, so it curves *almost not at all* — one parameter
the data has opinions about, one it does not, in the same matrix. Marginalizing
away the source's shapelet amplitudes does not fix this: the campaign's final
report quotes the identical order of magnitude for the marginalized T2
target
([§2.3, "The posterior zoo"](../current/claude-giga-lens/index.md#sec:zoo-data)).
The ill-conditioning is not a bug a later processing step removed; it is a
fact about which physical quantities a single pixel grid can and cannot pin
down.

One more collision to flag by name, because this repo uses the letter twice
more: the $A$ above is the Laplace Hessian of the log-posterior — nothing to
do with the lens Jacobian $A$ from [the first section](#symmetric-2x2), and a
third, better-behaved $A$ appears in
[Ch. 22](22-inference.md#marginalising-linear-amplitudes): the
ridge-regularized normal matrix `cgl/marg.py:48` builds while profiling out
the source's linear shapelet amplitudes analytically,

$$
A = X_{\mathrm w}^{\mathsf T} X_{\mathrm w} + \mathrm{diag}(\boldsymbol\Lambda).
$$

That $A$ is positive definite *by construction*: adding
$\mathrm{diag}(\boldsymbol\Lambda)$ with every $\Lambda_i>0$ floors every
eigenvalue away from zero before anything can go wrong — the same move
[Ch. 8](08-probability.md#ridge-is-a-prior) will call a Gaussian prior and
[Ch. 22](22-inference.md#marginalising-linear-amplitudes) will call ridge
regression. At the campaign's own parity gate this $A$ measures
$\mathrm{cond}(A) = 1.37\times10^4$
<!-- check: ch05.cond_marg_normal_matrix = 1.37e4 ± 0.01e4 -->
(`reproductions/claude-giga-lens/CAMPAIGN.md`, P0 exit gate, 2026-07-06) —
about $7.3\times10^9$ times friendlier than the raw Hessian above
<!-- check: ch05.cond_improvement_factor = 7.3e9 ± 0.1e9 -->,
purely because someone added a $+\Lambda$ before the trouble started. The
same tension — a raw, ill-conditioned real-data Hessian versus a
regularized, well-conditioned stand-in for it — is what
[Ch. 23](23-samplers.md#hmc-and-the-metric) and
[Ch. 26](26-the-saddle.md#the-same-matrix-twice) fight out for the sampler's
own mass matrix.

## Connect to the repo { #connect }

- [`site/guide_src/lensing.py:126-136`](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/lensing.py#L126)
  (`kappa_gamma_from_jacobian`) — the trace/traceless split this chapter
  derives, in six lines of numpy.
- [`site/guide_src/figures.py:52-73`](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/figures.py#L52)
  (`kappa_gamma_eigen`) — generates Figure 5.1 by constructing $A=(1-\kappa)I-\Gamma$
  directly and mapping a circle through $A^{-1}$; nothing in it is hand-drawn.
- [`reproductions/claude-giga-lens/cgl/e2.py:530-572`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L530)
  (`laplace_evidence`) — computes `H`, eigendecomposes it (`w`, `V`), and
  counts `n_neg`; this is the function that caught the saddle.
- [`reproductions/foundry-i/32_saddlefree_newton.py`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/foundry-i/32_saddlefree_newton.py)
  — an earlier saddle in the same pipeline, and the $V\mathrm{diag}(1/|\lambda_i|)V^{\mathsf T}$
  formula [Ch. 26](26-the-saddle.md#the-same-matrix-twice) revisits.
- [`reproductions/claude-giga-lens/cgl/marg.py:48`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/marg.py#L48)
  — the ridge-regularized normal matrix, `cond(A) = 1.37e4` at the parity gate.
- `reproductions/claude-giga-lens/CAMPAIGN.md`, "P1c metric-fix attempts —
  2026-07-10" — the saddle diagnosis in the campaign's own words, retractions
  included.
- main.tex, [§2.3, "The posterior zoo"](../current/claude-giga-lens/index.md#sec:zoo-data)
  and [§6.3, "Why sampling this posterior required tempered SMC"](../current/claude-giga-lens/index.md#sec:samplersaga) —
  the published record of both numbers this chapter uses.

## Exercises { #exercises }

??? question "Exercise 5.1 — the other two panels of Figure 5.1"
    Using $\lambda_{\mathrm t}=(1-\kappa)-\sqrt{\gamma_1^2+\gamma_2^2}$ and
    $\lambda_{\mathrm r}=(1-\kappa)+\sqrt{\gamma_1^2+\gamma_2^2}$, compute the
    eigenvalues for the left panel ($\kappa=0.3$, $\gamma_1=\gamma_2=0$) and
    the middle panel ($\kappa=0$, $\gamma_1=0.3$, $\gamma_2=0$) of Figure 5.1.
    For each, say whether the image of a circle under $A^{-1}$ is a circle or
    an ellipse, and why. Then check the text's claim that pure shear costs no
    area *to first order*: compute $\det A$ for the middle panel and compare
    it to $1$.

    ??? success "Solution"
        Left panel: $\lambda_{\mathrm t}=\lambda_{\mathrm r}=0.7$. Equal
        eigenvalues mean $A^{-1}$ scales every direction by the same factor
        $1/0.7$ — the image is a *larger circle*, not an ellipse, because pure
        convergence has no preferred direction. Middle panel:
        $\lambda_{\mathrm t}=0.7$, $\lambda_{\mathrm r}=1.3$ — unequal, so the
        image is an ellipse, elongated along the $\lambda_{\mathrm t}$
        eigendirection where $A^{-1}$ stretches more. Only shear breaks the
        circle's symmetry; convergence alone never does, which is exactly why
        this chapter calls $\kappa$ "isotropic" and $\Gamma$ "traceless."
        For the area check: $\det A = 0.7\times1.3 = 0.91$, not $1$ — at
        $\gamma_1=0.3$ the second-order term $-\gamma_1^2=-0.09$ is not
        negligible. The claim was "no *linear-order* area cost," and
        $1-\gamma_1^2$ has no linear term in $\gamma_1$ by construction; it is
        not a claim that finite shear leaves area untouched, and here it
        plainly does not ($\mu=1/0.91\approx1.10$).

??? question "Exercise 5.2 — classify without eigendecomposing"
    A Hessian at a candidate fit point has $\operatorname{tr}H = -6$ and
    $\det H = 8$. Using only the trace/determinant test from
    [Definiteness and saddles](#definiteness-and-saddles), classify it as
    PD, ND, or indefinite. Then find its two eigenvalues directly and confirm.

    ??? success "Solution"
        $\det H = 8 > 0$ and $\operatorname{tr}H = -6 < 0$, so the test gives
        **negative definite** — a genuine local maximum of whatever function
        $H$ is the Hessian of, not a saddle. Solving
        $\lambda^2-(-6)\lambda+8=0$, i.e. $\lambda^2+6\lambda+8=0$, factors as
        $(\lambda+2)(\lambda+4)=0$, giving $\lambda=-2,-4$ — both negative,
        confirming ND without needing the factorization to know it in advance.

??? question "Exercise 5.3 — a local zero is not a global mean"
    $\lambda_{\mathrm t}=0$ says $\kappa+\sqrt{\gamma_1^2+\gamma_2^2}=1$ at a
    single image-plane point. [Ch. 19](19-einstein-radius.md#the-mean-convergence-identity)
    will show that $\bar\kappa(\theta_{\mathrm E})=1$, the *mean* convergence
    inside the Einstein radius. Are these the same statement? If not, what is
    the precise relationship between the curve $\lambda_{\mathrm t}=0$ and the
    circle $\theta=\theta_{\mathrm E}$ for a lens that is not perfectly
    circular?

    ??? success "Solution"
        No — one is local, one is global, and conflating them is a real trap.
        $\bar\kappa(\theta_{\mathrm E})=1$ is an *average* over a disc of
        radius $\theta_{\mathrm E}$; it holds by definition for any profile,
        circular or not, and says nothing about any single point on the
        boundary. $\lambda_{\mathrm t}=0$ is a pointwise condition on the
        Jacobian and traces out the **critical curve** — for a circular lens
        the critical curve is a circle of radius exactly $\theta_{\mathrm E}$
        and the two statements coincide, but for an elliptical or shear-bearing
        lens the critical curve is a non-circular closed curve while
        $\theta_{\mathrm E}$ (defined through the mean) still names a single
        radius, usually chosen as the effective radius of the same-area circle.
        [Ch. 18](18-magnification.md#critical-curves) draws the two apart
        explicitly.

??? question "Exercise 5.4 — how many digits does a friendlier matrix buy you"
    Using $15.65 - \log_{10}(\mathrm{cond}\,A)$, compute the stable digit
    count for the ridge-regularized marg normal matrix,
    $\mathrm{cond}(A)=1.37\times10^4$, and compare it to the $\sim1.65$ digits
    left by the raw Hessian's $\mathrm{cond}\sim10^{14}$.

    ??? success "Solution"
        $\log_{10}(1.37\times10^4)\approx4.14$, so
        $15.65-4.14\approx11.5$ trustworthy digits
        <!-- check: ch05.stable_digits_remaining_marg = 11.5168 ± 0.001 -->
        — essentially all of float64's headroom, versus under two for the raw
        Hessian. The $+\Lambda$ in `cgl/marg.py:48` is not a minor numerical
        nicety; it is the difference between a matrix you can safely invert
        and one you cannot.
