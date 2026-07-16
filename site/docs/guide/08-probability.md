# 8. Likelihood, posterior, evidence, and the nat

This chapter turns "fit a Gaussian to pixel residuals" into a full Bayesian
machine: a likelihood, a prior, and an evidence integral whose logarithm is
measured in the same unit — the nat — as a cross-entropy loss. Two moves make
that machine useful for a real lensing posterior. The first is mostly a name
change: an ordinary least-squares fit is already a Gaussian log-likelihood, and
its familiar diagnostic, chi-squared, follows from it in three lines. The
second is a genuinely different move for a CS reader: Bayesian model
comparison replaces "lower loss wins" with a full-volume Bayes factor, and this
repo's own money number was decided by exactly one such comparison — a
191-nat swing that flipped a scientific verdict. Getting that comparison
analytically, without brute-force integrating a 46-dimensional posterior,
needs the Laplace approximation: the same second-order Taylor idea from
Chapter 2, applied to a log-posterior instead of a loss surface. By the end of
this chapter you will have derived that approximation, watched it become
*exact* on a toy problem shaped exactly like this repo's own marginalization,
and seen precisely how it breaks — not approximately, but formally undefined —
at the kind of point this repo's real posterior turns out to have.

!!! abstract "What you can skip"
    Bayes' theorem itself needs no derivation from you: $p(\theta\mid D) \propto p(D\mid\theta)\,p(\theta)$ is the one-line update rule behind every Bayesian
    classifier you have built, and the [Bayes](#bayes) section states it and
    moves on. That a Gaussian log-likelihood reduces to a sum of squared,
    weighted residuals is also not new — it is the reason mean-squared error is
    the default regression loss. What *is* new here is the astronomy-specific
    packaging of that fact (chi-squared, reduced chi-squared, "goodness of
    fit"), the convention that model evidence is always quoted in nats, and the
    Laplace approximation as a literal 42-line function in this repo's code,
    not a textbook aside.

## Bayes { #bayes }

Every posterior in this repository is built from one identity, which follows
from nothing but the definition of a conditional probability,
$p(A,B) = p(A\mid B)\,p(B) = p(B\mid A)\,p(A)$, rearranged:

$$
p(\theta \mid D) = \frac{p(D\mid\theta)\,p(\theta)}{p(D)}. \label{eq:bayes}
$$

Nothing in Equation $\eqref{eq:bayes}$ is astronomy-specific. $\theta$ is
whatever this repo happens to be fitting — for the money number it is the
46-dimensional parameter vector of an EPL mass profile, an external shear,
four Sérsic lens-light components and a marginalized source, worked out in
[Chapter 22](22-inference.md#the-forward-model). $D$ is the pixel data, one
HST cutout's worth of numbers. $p(D\mid\theta)$ is the likelihood — how
probable the data are, for a candidate $\theta$, under the noise model of
[Chapter 11](11-observation.md#the-noise-model). $p(\theta)$ is the prior:
whatever you believed about $\theta$ before this particular image, expressed
as a distribution — this campaign's own EPL-slope prior is a
`TruncatedNormal(2.0, 0.25, low=1.0, high=2.7)`
(`reproductions/claude-giga-lens/cgl/e2.py:110`), and you will lean on that
exact prior in [Chapter 25](25-money-number.md#the-money-number). And $p(D)$,
the evidence, is the number this chapter spends most of its remaining time on.

Notice what is *not* in Equation $\eqref{eq:bayes}$: a loss function, a
gradient step, an optimizer. It is a statement about probability
distributions, not an algorithm. Turning it into one — MAP estimation, HMC,
SMC — is the content of Chapters 22 through 26; this chapter builds the piece
all of them share: what the right-hand side actually *is*, term by term.

## Chi-squared { #chi-squared }

Assume the per-pixel noise is Gaussian and independent — the diagonal
likelihood of [Chapter 11](11-observation.md#the-noise-model), before this
repo's central complication (correlated noise, [Chapter 24](24-correlated-noise.md#the-covariance-model))
enters. Write the residual at pixel $i$ as $r_i = (d_i - m_i(\theta))/\sigma_i$.
Each $r_i \sim \mathcal N(0,1)$, and

$$
\log p(D\mid\theta) = -\frac12\sum_i r_i^2 - \sum_i \log\!\big(\sigma_i\sqrt{2\pi}\big)
= -\frac12\chi^2 + \text{const}, \qquad \chi^2 \equiv \sum_i r_i^2. \label{eq:chi2}
$$

!!! tip "You already know this"
    $\chi^2$ is twice a negative log-likelihood — the same object as a
    weighted mean-squared-error loss with per-example weights $1/\sigma_i^2$.
    Minimizing $\chi^2$ over $\theta$ *is* maximum-likelihood fitting; it is
    the identical arithmetic to minimizing an MSE loss, wearing a name
    inherited from 1900s statistics instead of 2010s deep learning.

For $n$ independent unit-variance residuals fit with $k$ linear parameters,
$\mathbb E[\chi^2] = n-k$ (Exercise 8.1 asks you to show this). The diagnostic
astronomers actually report is the *reduced* chi-squared,
$\chi^2_\nu \equiv \chi^2/(n-k)$, which should sit near 1 for a correctly
specified Gaussian model. Too far above 1 means the model or the noise is
wrong. Too far *below* 1 means — also that the model or the noise is wrong,
just in the other direction, which is the easier of the two to forget.

This repo forgot it, twice, in opposite directions, and both incidents are
load-bearing enough that the code now refuses to run without the fix.
Foundry-I's earliest native-scale HST fits reported a floor of
$\chi^2_\nu = 3.4$
<!-- check: ch08.chi2_nu_psf_broadened = 3.4 ± 0.05 -->
— a badly fitting model, or so it looked. The real cause was a rendering
convention bug: `lenstronomy`'s `subgrid_kernel` upsamples a PSF kernel
internally by the supersampling factor, and the PSF handed to it had *already*
been supersampled, so every native-scale fit was convolved with an effective
PSF twice as broad as the true one. Fixing only that — same data, same noise
model, same code otherwise — dropped $\chi^2_\nu$ to $1.05$
<!-- check: ch08.chi2_nu_psf_fixed = 1.05 ± 0.05 -->,
a $3.2\times$ improvement
<!-- check: ch08.psf_fix_ratio = 3.24 ± 0.05 -->
from correcting a convolution kernel's sampling convention, not the physics
(`reproductions/foundry-i/README.md:19-36`; the guard that now refuses to let
this recur is `reproductions/claude-giga-lens/cgl/guards.py:74-91`).

The opposite failure is subtler, because it looks like success. An early
sky-noise calibration reported $\chi^2_\nu = 0.451$
<!-- check: ch08.chi2_nu_sky_artifact = 0.451 ± 0.001 -->
— a suspiciously excellent fit, and "suspicious" turned out to be the correct
reaction. The noise $\sigma$ for that fit had been calibrated on *raw* image
pixels, roughly 70% of which were diffuse lens-light flux, not sky background;
an inflated $\sigma$ shrinks every $r_i$, and therefore $\chi^2$, exactly as
it should when you divide by too large a number. Recalibrating $\sigma$ on
model-subtracted residuals — pixels with the lens light actually removed —
gave the honest value, $\chi^2_\nu = 0.92$
<!-- check: ch08.chi2_nu_sky_honest = 0.92 ± 0.01 -->,
a correction of $2.04\times$
<!-- check: ch08.sky_inflation_ratio = 2.04 ± 0.01 -->
in the *other* direction (`reproductions/foundry-i/README.md:19-27`; guarded
at `reproductions/claude-giga-lens/cgl/guards.py:129-142`), which is why the
guard docstring still calls the retracted number "celebrated."

The lesson both incidents teach is the same one: $\chi^2_\nu$ is a diagnostic
*relative to the noise model you fed it*, not an absolute truth detector, and
it can mislead in either direction.

## Evidence and nats { #evidence-and-nats }

Fitting one model to data answers "what parameters?" Comparing two models — or,
as in this repo, two disconnected modes of the *same* posterior (a source can
sit behind either a steep or a shallow density slope with almost equally good
pixel fits, [Chapter 21](21-degeneracies.md#the-mass-sheet-degeneracy)) — needs
a different question: which explanation, integrated over *all* of its
parameters, accounts for more of the data's probability? That integral is the
evidence,

$$
Z(D) \equiv p(D) = \int p(D\mid\theta)\,p(\theta)\,d\theta, \label{eq:evidence}
$$

and comparing two candidates $H_1, H_2$ with a Bayes factor $K = Z_1/Z_2$ is
the Bayesian generalization of "lower loss wins." Because $Z$ is a
probability, this repo always reports $\log Z$, and always in **nats** —
natural-log units, never $\log_{10}$ (dex) and never $\log_2$ (bits). That
convention is not a stylistic accident: it is the same logarithm, and the same
unit, as a cross-entropy loss.

!!! tip "You already know this"
    A cross-entropy loss summed over a dataset, $-\sum_i \log q(x_i)$, is
    already measured in nats whenever your framework's `log` is the natural
    log — PyTorch's and JAX's both are. $\log Z$ is the identical object
    applied to a whole model instead of one example, and the two are additive
    for the identical reason: $\log \prod_i p(x_i) = \sum_i \log p(x_i)$.

Because everything is additive in log-space, a *difference* in evidence is a
sum of nats, and that sum is exactly this repo's headline result. Per-basin
tempered-SMC evidence on the money-number posterior gives
$\Delta\log Z = \log Z_{\mathrm{steep}} - \log Z_{\mathrm{low}} = +162.2$ nats
<!-- check: ch25.dlogz_diagonal = 162.2 ± 0.1 -->
under the diagonal likelihood (favouring the steep basin overwhelmingly), and
$-28.9$ nats
<!-- check: ch25.dlogz_correlated = -28.9 ± 0.1 -->
under the correlated likelihood of [Chapter 24](24-correlated-noise.md#the-covariance-model)
(favouring the low basin instead) — a swing of $191.1$ nats
<!-- check: ch25.evidence_swing_nats = 191.1 ± 0.05 -->.
Exponentiated, that swing is a probability-ratio factor of roughly
$10^{83}$
<!-- check: ch25.bayes_factor_log10 = 83.0 ± 0.1 -->:
not a nudge, an inversion of which basin the data prefer. What "steep," "low,"
and "basin" mean, and why this flip is the campaign's central finding rather
than a bug, is [Chapter 25](25-money-number.md#the-evidence-flip) in full.
The only point here is what a few hundred nats of anything *mean* as a
number — and that it is the same currency you already spend every time you
report a training loss.

## Laplace { #laplace }

When the evidence integral in Equation $\eqref{eq:evidence}$ cannot be done in
closed form over every dimension — which is every real case in this repo
except the linear amplitudes of the next section — one option is Monte Carlo
([Chapter 23](23-samplers.md#tempering-and-smc)). A cheaper alternative, when
it applies, borrows [Chapter 2](02-derivatives.md#taylor)'s Taylor expansion:
fit the *shape* of the log-posterior near its peak, and integrate over that
shape instead of the true one.

Let $\hat\theta$ be the mode (the MAP point), and expand $-\log p(\theta\mid D)$
to second order about it, exactly as in Chapter 2:

$$
-\log p(\theta\mid D) \approx -\log p(\hat\theta\mid D)
+ \frac12(\theta-\hat\theta)^\top H (\theta-\hat\theta),
\qquad H \equiv -\nabla\nabla \log p(\theta\mid D)\Big|_{\hat\theta}.
$$

$H$ is a Hessian — the [Chapter 5](05-linear-algebra.md#definiteness-and-saddles)
object again, and its eigendecomposition decides everything that follows. $H$
is symmetric, so its eigenvalues are real, and if — and only if — every one is
positive, $H$ is positive-definite and $\hat\theta$ is a genuine local minimum
of $-\log p$ (a mode, not a saddle). Approximating the un-normalized posterior
near $\hat\theta$ as a multivariate Gaussian with covariance $H^{-1}$, and
integrating that Gaussian exactly, gives the **Laplace approximation** to the
evidence:

$$
\log Z \approx \underbrace{\log p(D\mid\hat\theta) + \log p(\hat\theta)}_{\text{best fit}}
\;+\; \underbrace{\frac{d}{2}\log(2\pi) - \frac12\log\det H}_{\text{Occam factor}}.
\label{eq:laplace}
$$

The first pair of terms is simply how good the best fit is. The second pair is
new: it is a *volume*, not a fit quality — how much room the posterior has
around $\hat\theta$, measured through a determinant, again. The Log-Det Ledger's
[row 1](04-multivariable.md#the-log-det-ledger) was an area-scaling factor;
row 2 was a normalizing flow's density correction; this is row 3, and
[Chapter 23](23-samplers.md#closing-the-log-det-ledger) closes the ledger with
all three side by side. A *larger* $\det H$ — sharper curvature, a model that
must have its parameters tuned to a fine tolerance to fit at all — means a
*smaller* Occam factor: a fussy model is penalized relative to a forgiving one
that fits about as well without needing to be finely tuned. That is Occam's
razor as arithmetic, not metaphor.

This repo runs Equation $\eqref{eq:laplace}$ verbatim.
`reproductions/claude-giga-lens/cgl/e2.py:564`, inside `laplace_evidence()`,
computes `log_ev = logp_map + 0.5 * ndim * np.log(2 * np.pi) - 0.5 * logdet_H`
from the actual Hessian of the correlated log-posterior at its polished MAP.
The same function's docstring states, in one sentence, exactly the failure
mode Equation $\eqref{eq:laplace}$ predicts: `log_det H` needs every
eigenvalue of $H$ positive, and when $H$ is "INDEFINITE... the log-evidence
uses the floored-positive spectrum (a saddle evidence is only a rough
basin-mass proxy; caveated)"
(`reproductions/claude-giga-lens/cgl/e2.py:538-539`). A saddle has at least one
*negative* eigenvalue — $\hat\theta$ is not a minimum of $-\log p$ in every
direction, so the local Gaussian the whole derivation rests on does not exist
there. The formula does not merely become inaccurate; it becomes undefined,
since $\log$ of a non-positive number has no real value. The code's response
is to floor every eigenvalue at a small positive constant before taking its
log, trading a crash for a number honestly labeled a "rough proxy," not an
evidence. Whether this repo's own money-number MAP actually *is* such a
saddle — and what breaks, and what still works, when it is — is
[Chapter 26](26-the-saddle.md#the-map-is-a-saddle) in full.

You do not have to take the exactness of Equation $\eqref{eq:laplace}$ on
faith. Whenever the log-posterior *is* exactly quadratic — a linear model with
a Gaussian likelihood and a Gaussian prior, which is precisely what this
repo's marginalized shapelet source amplitudes are
([Chapter 22](22-inference.md#marginalising-linear-amplitudes)) — the Laplace
approximation is not an approximation at all; it is the exact integral. Take
the smallest possible instance of
`reproductions/claude-giga-lens/cgl/marg.py`'s algebra: five points
$(x_i, y_i)$, one linear amplitude $a$ with $y = ax$, ridge prior
$a \sim \mathcal N(0, 1)$. `marg.py`'s formula gives — constants dropped,
exactly as its own docstring says — a closed-form $\log L$ built only from
$b = x^\top y$, $A = x^\top x + 1$, and $\log\det A = \log A = 4.025$
<!-- check: ch08.toy_logdetA = 4.025 ± 0.01 -->.
Restore the dropped normalization ($\tfrac12\log 2\pi$ — the same additive
constant the campaign's own report tracks explicitly as "$+\ \mathrm{const}$",
`reproductions/claude-giga-lens/papers/main.tex:471`) and you get the full
evidence, $\log Z = -3.183$
<!-- check: ch08.toy_logZ_closed = -3.183 ± 0.001 -->.
Now do it the hard way: integrate the un-normalized posterior over $a$
*numerically*, on a fine grid, with no closed form assumed anywhere. The
brute-force integral gives $\log Z = -3.183$
<!-- check: ch08.toy_logZ_numeric = -3.183 ± 0.001 -->,
and the two agree not merely to three decimal places but to
$6\times10^{-15}$
<!-- check: ch08.toy_laplace_exact_diff = 0.0 ± 1e-9 -->
— floating-point noise, nothing else. `marg.py` is not approximating an
integral that would be too expensive to compute directly; for these
particular (linear) parameters, it *is* the integral.

## Ridge is a prior { #ridge-is-a-prior }

Run the same Gaussian-times-Gaussian argument forward, instead of integrating
it, and you get a familiar training-time identity. For a linear model
$y = X\theta + \text{noise}$ (noise already whitened to unit variance) with an
independent Gaussian prior $\theta \sim \mathcal N(0, \Lambda^{-1})$ per
parameter, the log-posterior is

$$
\log p(\theta\mid D) = -\frac12\|y - X\theta\|^2 - \frac12\theta^\top\Lambda\theta
+ \text{const},
$$

and setting its gradient to zero gives

$$
(X^\top X + \Lambda)\,\hat\theta = X^\top y. \label{eq:ridge}
$$

!!! tip "You already know this"
    Equation $\eqref{eq:ridge}$ is ridge regression's normal equation, and
    $\Lambda$ is the L2 penalty your optimizer calls `weight_decay`. A
    Gaussian prior on a parameter with precision $\lambda = 1/\sigma_{\mathrm{prior}}^2$,
    and an L2 regularizer with coefficient $\lambda$, are not merely
    analogous — the MAP estimate under the first is the exact minimizer of
    the second.

This is the entire content of `reproductions/claude-giga-lens/cgl/marg.py`'s 55
lines ([Chapter 22](22-inference.md#the-occam-term) runs it at full scale,
marginalizing dozens of shapelet source amplitudes analytically instead of
sampling them): $b = X_w^\top R_w$, $A = X_w^\top X_w + \Lambda$,
$\hat a = A^{-1}b$ by a Cholesky solve — never a matrix inverse, recalling
[Chapter 5](05-linear-algebra.md#conditioning)'s conditioning warning: an
ill-conditioned $A$ does not forgive `np.linalg.inv`. And $\log\det A$ is the
*same* Occam term as Equation $\eqref{eq:laplace}$'s $\log\det H$, because for
these particular (linear) parameters the log-posterior genuinely is the exact
quadratic Chapter 2 promised, and $A$ *is* that quadratic's Hessian. This
repo's own prior precision, `Lambda_ii = (i+1)/25`
(`reproductions/claude-giga-lens/cgl/marg.py:38-39`), grows with shapelet
order $i$ — a smoothness prior, penalizing higher (wigglier, noisier) basis
functions harder, the direct analogue of preferring a smaller network over a
larger one at equal training loss.

The Occam term is audited, not merely derived. This repo's parity harness
checks $-\tfrac12\log\det A$ against `numpy.linalg.slogdet` at a stored
validation point (`map_marg_pd.npz`) and finds agreement to $10^{-10}$, with
$\log\det A = 323.229$
<!-- check: ch08.occam_logdetA_parity = 323.229 ± 0.001 -->
and $\mathrm{cond}(A) \approx 1.4\times10^4$
<!-- check: ch08.occam_condA_parity = 14000 ± 1 -->
(`reproductions/claude-giga-lens/papers/main.tex`, Table `tab:parity`, Gate E).
`gigalens`'s own linear-amplitude solver omits this term entirely — it runs a
plain least-squares solve and reports the best-fit likelihood with no Occam
correction at all, which is the reason this campaign's evidence numbers
(the previous section) exist as a contribution rather than a rerun.

!!! note "Log-Det Ledger — row 3"
    A determinant appears a third time, in a third costume. Row 1
    ([Ch. 4](04-multivariable.md#the-log-det-ledger)) was the area-scaling
    factor of a change of variables; row 2 was a normalizing flow's
    $\log|\det J|$; row 3 is $-\tfrac12\log\det A$, the Occam factor of a
    Gaussian evidence — the same object, up to a factor of $-\tfrac12$, as the
    Laplace Hessian's $\log\det H$ above. [Chapter 23](23-samplers.md#closing-the-log-det-ledger)
    closes the ledger and puts all three side by side; [Chapter 22](22-inference.md#the-occam-term)
    runs this exact term in production, on the real correlated-likelihood
    posterior.

## Connect to the repo { #connect }

- `reproductions/claude-giga-lens/cgl/marg.py` (55 lines total) — the whole
  chapter's ridge/Occam algebra, compressed. Every line of `marg_loglik`
  (lines 31-55) is one term in Equations $\eqref{eq:ridge}$ and
  $\eqref{eq:laplace}$; its module docstring is quoted almost verbatim above.
- `reproductions/claude-giga-lens/cgl/e2.py:531-572` — `laplace_evidence()`,
  the Laplace approximation as running code, floored-eigenvalue caveat
  included.
- `reproductions/claude-giga-lens/cgl/guards.py:74-142` — the two
  chi-squared retractions, encoded as literal, permanent guard functions:
  this repo's institutional memory refuses to let either bug recur silently.
- `reproductions/claude-giga-lens/papers/main.tex` — Table `tab:parity`
  (Gate E) carries the audited Occam-term number this chapter uses.
- `reproductions/foundry-i/README.md:19-42` — the PSF-convention fix and the
  sky-calibration retraction, in the campaign's own words.

## Exercises { #exercises }

??? question "Exercise 8.1 — Why $\mathbb E[\chi^2] = n-k$"
    For $n$ independent, unit-variance normal residuals fit by ordinary least
    squares against $k$ linear parameters, show that $\mathbb E[\chi^2] = n-k$
    rather than $n$.

    ??? success "Solution"
        Each individual $r_i \sim \mathcal N(0,1)$ has $\mathbb E[r_i^2] = 1$,
        so a *raw* residual vector (no fitting) would give
        $\mathbb E[\chi^2] = n$. But least-squares fitting chooses the $k$
        parameters that make the residual vector orthogonal to the $k$-dimensional
        column space of the design matrix $X$ — that orthogonality is the
        normal equation's geometric content. The residual is therefore
        confined to an $(n-k)$-dimensional subspace, and the sum of squares of
        a standard normal vector restricted to a $(n-k)$-dimensional subspace
        has expectation $n-k$, not $n$. Every fitted parameter "uses up" one
        degree of freedom's worth of expected $\chi^2$.

??? question "Exercise 8.2 — Ridge from the posterior"
    Starting from the log-posterior $\log p(\theta\mid D) = -\tfrac12\|y-X\theta\|^2 -\tfrac12\theta^\top\Lambda\theta + \text{const}$, differentiate with respect to $\theta$ and set the gradient to zero to derive Equation $\eqref{eq:ridge}$.

    ??? success "Solution"
        $\nabla_\theta\!\left[-\tfrac12\|y-X\theta\|^2\right] = X^\top(y - X\theta)$, and $\nabla_\theta\!\left[-\tfrac12\theta^\top\Lambda\theta\right] = -\Lambda\theta$ (using that $\Lambda$ is symmetric — it is diagonal here). Summing and setting to zero: $X^\top y - X^\top X\theta - \Lambda\theta = 0$, i.e. $(X^\top X + \Lambda)\theta = X^\top y$, exactly Equation $\eqref{eq:ridge}$. Set $\Lambda = 0$ and you recover the ordinary least-squares normal equation; a Gaussian prior is what turns $X^\top X$, which can be singular or ill-conditioned, into $X^\top X + \Lambda$, which — for any $\Lambda \succ 0$ — is provably invertible. Regularization as numerical hygiene and regularization as a Bayesian belief are the same $\Lambda$.

??? question "Exercise 8.3 — What omitting the Occam term costs you"
    Using this chapter's toy numbers (five points, one ridge-regularized
    amplitude $a$), compute what `marg.py`'s $\log L$ would have been *without*
    the Occam term — i.e. the plain best-fit likelihood
    $-\tfrac12\|y-X\hat a\|^2$ that `gigalens`'s own least-squares path
    reports. How many nats does dropping the Occam term buy, and does that
    number depend on how tightly the data constrain $a$?

    ??? success "Solution"
        Dropping $-\tfrac12\log\det A$ from `marg.py`'s formula gives
        $\log L_{\mathrm{no\ Occam}} = -2.089$
        <!-- check: ch08.toy_logL_no_occam = -2.089 ± 0.001 -->,
        compared with the correct $\log L_{\mathrm{marg}} = -4.102$
        <!-- check: ch08.toy_logL_marg_style = -4.102 ± 0.001 -->
        — a gap of exactly $\tfrac12\log\det A = 2.013$ nats
        <!-- check: ch08.toy_occam_correction_nats = 2.0127 ± 0.001 -->,
        precisely half the Occam term this chapter has been tracking. Omitting
        the term always makes the reported likelihood *larger* (less
        negative), by an amount that grows with $\log\det A$ — which grows
        both with how many parameters are marginalized and with how tightly
        the data constrain them. A model with more free amplitudes, or
        amplitudes pinned down more tightly, collects a *bigger* uncorrected
        bonus for exactly the flexibility Occam's razor exists to penalize:
        the mechanism by which an evidence comparison without this term
        systematically favors more flexible models.

??? question "Exercise 8.4 — Why flooring an eigenvalue is not a fix"
    `cgl/e2.py`'s `laplace_evidence()` floors any Hessian eigenvalue below
    `eig_floor` before computing $\log\det H$. Explain in one sentence why
    this makes the function *not crash*, and in a second sentence why the
    number it then returns is not a real Bayesian evidence.

    ??? success "Solution"
        Flooring replaces every non-positive eigenvalue with a small positive
        constant, so every factor inside the log is positive and $\log\det H$
        is a well-defined real number — the crash (a $\log$ of zero or a
        negative number) is avoided by construction. But the Laplace
        derivation assumed a genuine local Gaussian approximation around a
        *minimum* of $-\log p$; at a saddle, the true posterior surface
        curves *away* from $\hat\theta$ in the negative-eigenvalue directions
        rather than toward it, so no Gaussian with a positive-definite
        covariance actually approximates the posterior there. The floored
        number is a well-defined arithmetic quantity, and a useful rough
        proxy for how much probability mass sits in that basin — but it is
        not the integral Equation $\eqref{eq:laplace}$ derived, because the
        assumption the derivation needed (an everywhere-positive-definite $H$)
        is false at the point being evaluated.
