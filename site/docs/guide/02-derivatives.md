# Derivatives, Taylor, and why linearization is the whole game

Everything hard in this book — the deflection field, the magnification, the
Laplace approximation to a posterior, the mass matrix an HMC sampler needs to
move efficiently — is the same two moves applied to a different function: (1)
replace a nonlinear map by its best local *linear* model, and (2) when linear
is not enough, add the best local *quadratic* correction. This chapter builds
both moves in one dimension, where you can see every term, so that when they
reappear in $n$ dimensions (Ch. 4–5) as a Jacobian and a Hessian, the objects
are already familiar and only the bookkeeping is new.

!!! abstract "What you can skip"
    If you can state the derivative as a limit, prove the product and chain
    rules, and are comfortable with $O(h^2)$ meaning "shrinks like $h^2$ as
    $h\to0$," skim [Derivative as a linear map](#derivative-as-linear-map) and
    slow down at [Taylor to second order](#taylor) and
    [Why second order](#why-second-order) — those two sections reframe
    familiar material as the object the rest of the book calls "the metric,"
    "the Occam term," and "the saddle." This chapter does not redo limits or
    $\epsilon$-$\delta$ arguments; you already own those.

## Derivative as a linear map { #derivative-as-linear-map }

The definition you know:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}.
$$

Rearrange it and you get the statement this book actually uses, over and over,
under different names:

$$
f(x+h) = f(x) + f'(x)\, h + o(h) \qquad \text{as } h \to 0,
$$

where $o(h)$ means a remainder that shrinks *faster* than $h$ itself — divide
it by $h$ and it still goes to zero. Read that equation as a claim about
approximation, not about limits: near $x$, the graph of $f$ is indistinguishable
from the straight line through $(x, f(x))$ with slope $f'(x)$, up to an error
that vanishes faster than the distance $h$ you moved. The derivative is not a
number that happens to fall out of a limit; it is *the slope of the unique
line that makes this approximation work*. That line is the derivative's whole
job description, and it is the only property of $f'(x)$ this book ever uses.

Two facts about that linear model matter more than the mechanics of computing
it:

**It composes.** If $f(x+h) \approx f(x) + f'(x) h$ and $g(y+k) \approx
g(y) + g'(y) k$, then composing the maps composes the linear models:

$$
g(f(x+h)) \approx g(f(x)) + g'(f(x))\, f'(x)\, h,
$$

which is the chain rule, and nothing more exotic. Every gradient your neural
networks have ever used was produced by chaining exactly this identity, layer
by layer, back through a computational graph.

**It generalizes verbatim.** For $f : \mathbb{R}^n \to \mathbb{R}^m$, the same
statement holds with $f'(x)$ replaced by a matrix — the Jacobian — and $h$ a
vector:

$$
f(x+h) = f(x) + Df(x)\, h + o(\|h\|).
$$

Nothing about the *logic* changes; every entry of $Df(x)$ is still an ordinary
1-D derivative, one partial derivative at a time. Constructing that matrix,
and reading geometric meaning (area, orientation) out of its determinant, is
the business of [Ch. 4](04-multivariable.md#gradient-jacobian-hessian). Here,
file away only this: whatever a "Jacobian" turns out to be, it is asking
exactly the question this section asks — what linear map best approximates
this function near this point? — for a map with more than one input and output
coordinate.

!!! tip "You already know this"
    A neural network's backward pass computes exactly this object, one layer
    at a time: the local linear map (the Jacobian) of that layer's function,
    chained through every layer by the identity above. `jax.grad` and
    `jax.hessian`, used throughout this repo's sampler code (`cgl/fitting.py:43`,
    `cgl/e1.py:797`, `cgl/e2.py:554`), compute nothing more exotic than the two
    objects this chapter builds by hand: the first-order and second-order local
    model of a function at a point.

A single worked derivative, done from the limit definition, sets up something
you will need repeatedly: differentiating a power. For $f(x) = x^n$,

$$
f'(x) = \lim_{h\to 0} \frac{(x+h)^n - x^n}{h}.
$$

Expand $(x+h)^n$ by the binomial theorem: $(x+h)^n = x^n + n x^{n-1} h + \binom{n}{2} x^{n-2} h^2 + \cdots$. Subtract $x^n$, divide by $h$, and let $h \to 0$; every term past the first carries a positive power of $h$ and vanishes, leaving

$$
\frac{d}{dx} x^n = n\, x^{n-1}.
$$

This holds for negative and non-integer $n$ too (the proof needs a slightly
different argument, but the result is the same), which is why it will apply
directly to inverse-square-law and power-law profiles: the moment [Ch. 3](03-integrals.md#the-abel-projection)
introduces a density profile $\rho(r) \propto r^{-\gamma}$, you already have
the derivative in hand.

## Taylor to second order { #taylor }

A linear model is the *best possible* linear model, but "best possible" still
leaves an error, and sometimes that error is the thing you care about. Taylor's
theorem says how to shrink it further: add the next term.

$$
f(x_0 + h) = f(x_0) + f'(x_0)\, h + \frac{1}{2} f''(x_0)\, h^2 + O(h^3).
$$

The pattern is: each successive term uses one more derivative and one higher
power of $h$, and the *error* — the gap between $f$ and the truncated
polynomial — shrinks one power of $h$ faster than the last term you kept. Keep
only the first two terms and you recover last section's linear model, with
error $O(h^2)$. Keep three terms and the error drops to $O(h^3)$: a strictly
better approximation, at the cost of one more derivative.

**A concrete payoff: why the repo differentiates numerically by central
differences.** [Ch. 4](04-multivariable.md#gradient-jacobian-hessian) will need
the lens-equation Jacobian, and `lensing.py`'s implementation gets it not from
an analytic formula but from finite differences:

```python
bx_px, by_px = beta(x + h, y)
bx_mx, by_mx = beta(x - h, y)
...
a11 = (bx_px - bx_mx) / (2 * h)
```

(`lensing.py:101-123`, `lens_jacobian`). Why evaluate at *both* $x+h$ and
$x-h$ and average, rather than the more obvious $(f(x+h)-f(x))/h$? Taylor's
theorem answers it directly. Expand both points to third order:

$$
f(x+h) = f(x) + f'(x) h + \tfrac{1}{2} f''(x) h^2 + \tfrac{1}{6} f'''(x) h^3 + O(h^4),
$$

$$
f(x-h) = f(x) - f'(x) h + \tfrac{1}{2} f''(x) h^2 - \tfrac{1}{6} f'''(x) h^3 + O(h^4).
$$

Subtracting cancels every *even*-power term ($f(x)$, the $h^2$ term):

$$
f(x+h) - f(x-h) = 2 f'(x) h + \tfrac{1}{3} f'''(x) h^3 + O(h^5),
$$

and dividing by $2h$ gives

$$
\frac{f(x+h) - f(x-h)}{2h} = f'(x) + \frac{1}{6} f'''(x)\, h^2 + O(h^4).
$$

The one-sided difference $(f(x+h)-f(x))/h$ keeps the $h^2$ term from the
*first* expansion above and errs at $O(h)$. The centered difference cancels it
by symmetry and errs at $O(h^2)$ — two more correct digits for the same number
of function evaluations, for free, just by evaluating symmetrically around the
point instead of stepping forward from it. That is the entire reason the repo's
numerical Jacobian is centered and not forward: it is a direct, unglamorous
consequence of Taylor's theorem, not a stylistic choice.

The same expansion generalizes to $\mathbb{R}^n \to \mathbb{R}$: gradient
$\nabla f(x_0)$ replaces $f'(x_0)$, and the second-order term becomes a
**quadratic form** built from the Hessian matrix $H$ of second partial
derivatives,

$$
f(x_0 + h) \approx f(x_0) + \nabla f(x_0) \cdot h + \frac{1}{2} h^{\mathsf T} H h.
$$

Building $H$ and reading its eigenstructure is [Ch. 5](05-linear-algebra.md#symmetric-2x2)'s
job. What matters here is the *shape* of the answer: one vector (first-order
information) and one symmetric matrix (second-order information) are all you
ever get from a Taylor expansion, in any number of dimensions, and this book
never needs anything past the quadratic term.

## Why second order { #why-second-order }

Here is the payoff the whole chapter has been building toward, and it is worth
stating baldly: **every hard object in this book is a first- or second-order
local Taylor model of something**, and knowing that turns four topics that
look unrelated — a bent light ray, a magnified image, a Bayesian evidence
integral, and a Hamiltonian sampler's step size — into one idea applied four
times.

- The **deflection field** is a nonlinear map from image position to source
  position, but at any single image location $\theta_0$, an infinitesimal
  perturbation $d\theta$ produces $d\beta = A(\theta_0)\, d\theta$ — a strictly
  linear relationship, i.e. exactly this chapter's first-order model, with $A$
  the Jacobian ([Ch. 17](17-lens-equation.md#the-lens-equation),
  [Ch. 18](18-magnification.md#magnification-is-a-jacobian)).
- **Magnification** is a scalar built from that same Jacobian ($\mu = 1/\det A$)
  — a fact about a *derivative*, not a separate piece of physics
  ([Ch. 18](18-magnification.md#magnification-is-a-jacobian)).
- The **Laplace approximation** to a Bayesian posterior is literally this
  chapter's second-order Taylor expansion, applied to $\log p(\theta)$ instead
  of a lensing map: expand around the MAP $\theta^\star$ (where $\nabla \log p
  = 0$, so the first-order term vanishes identically), and the Hessian of
  $\log p$ becomes the precision matrix of a Gaussian approximation to the
  whole posterior ([Ch. 8](08-probability.md#laplace)).
- An **HMC sampler's mass matrix** is chosen to match the same Hessian, because
  a sampler that does not know the local quadratic shape of $\log p$ takes
  steps that are too small in narrow directions and too large in wide ones
  ([Ch. 23](23-samplers.md#hmc-and-the-metric)).

That list is also a warning, and this repo lived it. At a **critical point** —
anywhere the gradient is zero — the first-order term of a Taylor expansion
vanishes *identically*. That is exactly what a gradient-based optimizer such as
`jax.value_and_grad` (`cgl/fitting.py:43`, `cgl/e1.py:797,892`) is built to find:
a point where it has nothing left to correct. But a vanishing gradient tells
you a point is *flat* to first order; it says nothing about whether the
function curves up, down, or up-in-some-directions-and-down-in-others there.
That question belongs entirely to the second-order term — the Hessian — which
is precisely why [Ch. 5](05-linear-algebra.md#definiteness-and-saddles) exists.

The campaign's correlated-noise fit hit this directly. Its MAP-finding stage
(`map_polish`) drove the gradient of the log-posterior to zero and stopped,
reporting success. A second-order check — computing the Hessian of the
log-posterior at that point with `jax.hessian` (`cgl/e2.py:554`) — showed the
point was a **saddle, not a mode**: minimum eigenvalue $-14.85$, with five
negative directions (main.tex, [§6.3](../current/claude-giga-lens/index.md#sec:samplersaga);
CAMPAIGN.md, "P1c metric-fix attempts — 2026-07-10"). The sampler's step-size
metric (the exact construction, and why "mass" and "covariance" are easy to
swap by mistake, is [Ch. 23](23-samplers.md#hmc-and-the-metric)'s business) is
built by reading a length scale off each Hessian eigenvalue $\lambda_i$ — which
only means what it is supposed to mean if $\lambda_i > 0$, i.e. if the
log-posterior is bowl-shaped along that direction. Five of this Hessian's
eigenvalues are negative: the log-posterior curves *away* from the stationary
point along those directions, not toward it, so there is no local bowl for a
length scale to describe. Two repairs were tried, and both were built
specifically to *avoid* using $|\lambda_i|$ on an indefinite Hessian: one
floored the raw diagonal $|H_{ii}|$ instead of the eigenvalues, one replaced
the Hessian metric outright with an independently-fit SVI covariance. Both
were positive-definite by construction, and both still left the sampler's
$\hat{R}$ diagnostic stuck at 21–22. The gradient alone could not have told
you any of this; only the second-order term could, and only because someone
thought to compute it.
[Ch. 26](26-the-saddle.md#the-map-is-a-saddle) tells this story in full,
including how it was eventually resolved. File it
away here as the concrete argument for why "why second order" is not academic
throat-clearing: a real campaign, with real compute charged to it, spent time
chasing a fix to a problem that the second-order term would have diagnosed
immediately.

## Connect to the repo { #connect }

- `lensing.py:101-123` (`lens_jacobian`) — the numerical Jacobian is a
  **centered** finite difference, $(f(x+h)-f(x-h))/2h$, precisely because
  (as derived above) that cancellation of the even-order Taylor term buys an
  extra order of accuracy over a one-sided difference. [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/lensing.py#L101)
- `cgl/fitting.py:43` and `cgl/e1.py:797,892` — `jax.value_and_grad`, the
  first-order local linear model, used to walk both the MAP optimizer and the
  SVI (variational) fit toward a stationary point.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/fitting.py#L43)
- `cgl/e2.py:554` — `jax.hessian(lp1)(z_map)`, the second-order local
  quadratic model of the log-posterior at the MAP; its eigenvalues (`w`,
  `n_neg` at `cgl/e2.py:557-558`) are the diagnostic that caught the saddle.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L554)
- The saddle finding in full: main.tex [§6.3, "Why sampling this posterior
  required tempered SMC"](../current/claude-giga-lens/index.md#sec:samplersaga).

## Exercises { #exercises }

??? question "Exercise 2.1 — the power rule from the limit definition"
    Using the binomial theorem, show that $\frac{d}{dx} x^3 = 3x^2$ directly
    from the limit definition $f'(x) = \lim_{h\to0} \big[(x+h)^3 - x^3\big]/h$.
    Which terms survive the limit, and why?

    ??? success "Solution"
        Expand $(x+h)^3 = x^3 + 3x^2 h + 3x h^2 + h^3$. Subtract $x^3$ and
        divide by $h$:

        $$
        \frac{(x+h)^3 - x^3}{h} = 3x^2 + 3xh + h^2.
        $$

        Every surviving term after the first carries at least one positive
        power of $h$, so as $h \to 0$ they vanish, leaving $3x^2$. Nothing
        about this argument is special to the exponent 3 — the same
        cancellation, one binomial-theorem term at a time, produces
        $nx^{n-1}$ for any positive integer $n$.

??? question "Exercise 2.2 — why the centered difference is one order better"
    A colleague suggests replacing `lensing.py`'s centered difference with the
    cheaper one-sided form $(f(x+h) - f(x))/h$, computing half as many
    evaluations. Using the Taylor expansions in [Taylor to second
    order](#taylor), state the leading-order error of each form and say by
    how many powers of $h$ they differ.

    ??? success "Solution"
        From $f(x+h) = f(x) + f'(x)h + \tfrac12 f''(x) h^2 + O(h^3)$, the
        one-sided difference is

        $$
        \frac{f(x+h)-f(x)}{h} = f'(x) + \frac{1}{2} f''(x)\, h + O(h^2),
        $$

        an error of order $h$ (unless $f''(x_0)=0$). The centered difference,
        derived in the text, has error $\frac16 f'''(x) h^2 + O(h^4)$ — order
        $h^2$. The centered form is one full power of $h$ more accurate for
        the *same* step size, at the cost of one extra function evaluation
        (it needs $f(x-h)$ as well as $f(x+h)$). That is exactly the trade the
        repo makes in `lensing.py:115-122`.

??? question "Exercise 2.3 — a zero gradient is not a verdict"
    Consider $f(x, y) = x^2 - y^2$. Verify that $\nabla f = 0$ at the origin.
    Then evaluate $f$ along the path $x=t, y=0$ and along $x=0, y=t$ for small
    $t \neq 0$. What do the two paths tell you that the gradient alone did
    not, and which repo finding does this mirror?

    ??? success "Solution"
        $\nabla f = (2x, -2y)$, which is exactly $(0,0)$ at the origin — a bona
        fide critical point. Along $x=t, y=0$: $f = t^2 > 0$, curving *upward*.
        Along $x=0, y=t$: $f = -t^2 < 0$, curving *downward*. The same
        stationary point is a local max in one direction and a local min in
        another — the textbook definition of a saddle. The gradient could not
        distinguish this from a true minimum or maximum; only evaluating $f$
        off-origin (equivalently, the Hessian $\begin{pmatrix}2&0\\0&-2\end{pmatrix}$,
        with one positive and one negative eigenvalue) reveals it. This is the
        toy version of exactly what `jax.hessian` caught in the campaign's
        correlated-noise MAP (min eigenvalue $-14.85$, five negative
        directions): a `map_polish` stage that stops once $\nabla \log p = 0$
        has, by construction, no way to see this.

??? question "Exercise 2.4 — composing two linear models"
    The lens equation ([Ch. 17](17-lens-equation.md#the-lens-equation)) is a
    composition: image light depends on source position $\beta$, which depends
    on image position $\theta$ through $\beta(\theta) = \theta - \alpha(\theta)$.
    If a downstream likelihood is a function $L(\beta)$, write down the
    first-order (linear) model of $L(\beta(\theta))$ near a point $\theta_0$,
    in terms of $\nabla L$ and the Jacobian of $\beta$. Do not evaluate
    anything — just apply the chain rule from [Derivative as a linear
    map](#derivative-as-linear-map).

    ??? success "Solution"
        Let $A(\theta_0) = D\beta(\theta_0)$ be the lens Jacobian ([Ch.
        18](18-magnification.md#magnification-is-a-jacobian)). Composing the
        two linear models, exactly as in the chain-rule identity derived in
        the text,

        $$
        L(\beta(\theta_0 + h)) \approx L(\beta(\theta_0)) + \nabla L(\beta(\theta_0))^{\mathsf T} A(\theta_0)\, h.
        $$

        This is the same computation `jax.grad` performs automatically when it
        differentiates a likelihood through the forward model in
        [Ch. 22](22-inference.md#the-forward-model) — one chain-rule
        multiplication, $\nabla L$ against the Jacobian of everything upstream
        of it, with no new mathematics past what this chapter derived.
