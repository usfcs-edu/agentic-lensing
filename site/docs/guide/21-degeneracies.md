# 21. Degeneracies: the directions where the data says nothing

Every inverse problem has directions in parameter space where the data cannot
tell you anything: change the parameter, and every quantity you can measure
stays exactly the same. Strong lensing has an unusually clean example of this
— you can prove it in four lines, from the lens equation alone — and this
repository has run into a member of its family twice: once under the wrong
name, once under no name at all. This chapter derives the exact case (the mass
sheet), shows what it costs a $H_0$ measurement, generalizes it to the messier
case this campaign's flexible source model actually exhibits, and gives you
the one-line test that tells a real degeneracy from a merely stubborn
posterior — a test [Ch. 5](05-linear-algebra.md#conditioning) and
[Ch. 26](26-the-saddle.md) both need.

!!! abstract "What you can skip"
    The linear algebra here is old news, restated: $A_{\lambda_{\mathrm{MST}}}
    = \lambda_{\mathrm{MST}} A$ is a scalar multiple of a matrix, and
    $\det(cA) = c^2\det A$ for a $2\times2$ is one line from
    [Ch. 5](05-linear-algebra.md#symmetric-2x2). If you already think fluently
    in terms of non-identifiable parameters, flat loss directions, and gauge
    symmetries — $GL(r)$ invariance in matrix factorization, the scaling and
    permutation symmetries of a ReLU net, rotational ambiguity in ICA — skim
    ["A degeneracy is a gauge symmetry"](#degeneracy-is-gauge-symmetry) for the
    lensing vocabulary only. What is not boilerplate: the actual mass-sheet
    family, why it makes $H_0$ from time delays a hard measurement in
    principle (not just in practice), and the two times this repository ran
    into a member of this family — handled correctly once, incorrectly once —
    for reasons worth knowing before you trust either verdict.

## The mass-sheet degeneracy { #the-mass-sheet-degeneracy }

No lens galaxy sits alone. A foreground group, a cluster halo, or just the
large-scale structure along the line of sight always contributes some extra
mass, and over the tiny patch of sky an Einstein ring occupies (order one
arcsecond) that extra convergence is, to good approximation, *constant* — a
uniform sheet laid across the field. The question this section answers: can
you tell, from the images alone, how much of the lensing you are looking at
is the galaxy and how much is the invisible sheet in front of and behind it?

This guide reserves the bare symbol $\lambda$ for the SMC tempering parameter
of [Ch. 23](23-samplers.md#tempering-and-smc); to keep the two apart on sight,
every mass-sheet parameter in this chapter carries a subscript,
$\lambda_{\mathrm{MST}}$. Define a one-parameter family of potentials built
from any lens potential $\psi(\boldsymbol\theta)$:

$$
\psi_{\lambda_{\mathrm{MST}}}(\boldsymbol\theta) \;=\;
\lambda_{\mathrm{MST}}\,\psi(\boldsymbol\theta) \;+\;
(1-\lambda_{\mathrm{MST}})\,\tfrac12|\boldsymbol\theta|^2.
\label{eq:mst-psi}
$$

Apply $\kappa = \tfrac12\nabla^2\psi$
([Ch. 17](17-lens-equation.md#the-psi-alpha-kappa-trio)) and the one-line fact
that $\nabla^2\big(\tfrac12|\boldsymbol\theta|^2\big) = 2$ in two dimensions:

$$
\kappa_{\lambda_{\mathrm{MST}}}(\boldsymbol\theta) =
\lambda_{\mathrm{MST}}\,\kappa(\boldsymbol\theta) + (1-\lambda_{\mathrm{MST}}).
\label{eq:mst-kappa}
$$

That *is* the sheet: a uniform layer of convergence $(1-\lambda_{\mathrm{MST}})$
— the same number everywhere — sitting on top of the original galaxy rescaled
by $\lambda_{\mathrm{MST}}$. Differentiating $\eqref{eq:mst-psi}$ once gives
the deflection, $\boldsymbol\alpha_{\lambda_{\mathrm{MST}}}(\boldsymbol\theta) =
\nabla\psi_{\lambda_{\mathrm{MST}}} = \lambda_{\mathrm{MST}}\boldsymbol\alpha(\boldsymbol\theta)
+ (1-\lambda_{\mathrm{MST}})\boldsymbol\theta$, and substituting into the lens
equation ([Ch. 17](17-lens-equation.md#the-lens-equation)),
$\boldsymbol\beta = \boldsymbol\theta - \boldsymbol\alpha(\boldsymbol\theta)$,
collapses everything:

$$
\boldsymbol\beta_{\lambda_{\mathrm{MST}}}(\boldsymbol\theta) =
\boldsymbol\theta - \boldsymbol\alpha_{\lambda_{\mathrm{MST}}}(\boldsymbol\theta)
= \lambda_{\mathrm{MST}}\big[\boldsymbol\theta - \boldsymbol\alpha(\boldsymbol\theta)\big]
= \lambda_{\mathrm{MST}}\,\boldsymbol\beta(\boldsymbol\theta).
\label{eq:mst-beta}
$$

Four lines: $\eqref{eq:mst-psi}$ defines the family, one Laplacian gives
$\eqref{eq:mst-kappa}$, one gradient gives the deflection, one substitution
gives $\eqref{eq:mst-beta}$. Now read $\eqref{eq:mst-beta}$ as a statement
about *images*. Suppose the true lens ($\lambda_{\mathrm{MST}}=1$) produces
two images $\theta_1,\theta_2$ of a single source at $\beta_0$ — take the SIS
double [Ch. 17](17-lens-equation.md#the-lens-equation) solves,
$\theta_{\mathrm E}=1$, $\beta_0=0.4$
<!-- check: ch21.beta0 = 0.4 ± 0.001 -->, giving $\theta_1=1.4$
<!-- check: ch21.theta1 = 1.4 ± 0.001 --> and $\theta_2=-0.6$
<!-- check: ch21.theta2 = -0.6 ± 0.001 -->. Equation $\eqref{eq:mst-beta}$
says that for *any* $\lambda_{\mathrm{MST}}$, plugging those same two image
positions into the transformed lens returns
$\beta_{\lambda_{\mathrm{MST}}}(\theta_1) = \beta_{\lambda_{\mathrm{MST}}}(\theta_2)
= \lambda_{\mathrm{MST}}\beta_0$ — still one self-consistent source position,
just the wrong one, off by exactly $\lambda_{\mathrm{MST}}$. Checked directly
at $\lambda_{\mathrm{MST}}=0.8$: both images return
$0.320$ <!-- check: ch21.beta_lam_1 = 0.32 ± 0.001 --> (their disagreement is
$2\times10^{-16}$ <!-- check: ch21.beta_consistency_gap = 0.0 ± 1e-9 -->, pure
floating-point noise), exactly $0.8\times0.4$
<!-- check: ch21.beta_lambda_ratio = 0.8 ± 1e-6 -->. Image astrometry — and,
by the identical argument applied to two *different* sources, image
separations and image-count statistics — cannot see $\lambda_{\mathrm{MST}}$
at all.

Magnification is the same story one derivative further. Since $A = I -
\mathrm{Hess}(\psi)$ ([Ch. 5](05-linear-algebra.md#symmetric-2x2)) and Hess of
the added quadratic term is the identity matrix,
$A_{\lambda_{\mathrm{MST}}} = \lambda_{\mathrm{MST}} A$ — literally a scalar
multiple, so $\det A_{\lambda_{\mathrm{MST}}} = \lambda_{\mathrm{MST}}^2\det A$
and $\mu_{\lambda_{\mathrm{MST}}} = \mu/\lambda_{\mathrm{MST}}^2$ at *every*
image, uniformly. The overall image brightness moves — but it is degenerate
with the source's own unknown intrinsic luminosity anyway, so nothing is lost.
The *ratio* of magnifications between two images of one source — the flux
ratio, which imaging genuinely does measure without knowing the source's true
brightness — cancels the $\lambda_{\mathrm{MST}}^2$ exactly and is invariant.
Even the arc shapes survive: a scalar multiple of $A$ keeps every eigenvector
(orientation) and every eigenvalue *ratio* (axis ratio) fixed, only the scale
moves — and that scale is absorbed by the same source-size ambiguity as the
flux. Every observable you can build from where the images are, how their
brightnesses compare, or what shape they trace is provably blind to
$\lambda_{\mathrm{MST}}$. That is what "the data says nothing" means in this
chapter's title — literally, not rhetorically. One observable is not blind to
it.

## Time delays and $H_0$ { #time-delays-and-h0 }

Multiple images of one source do not arrive at the same time: each ray
travels a different path through a different depth of the potential well, and
general relativity says the light-travel time along image $\theta$ depends on
the **Fermat potential**

$$
\tau(\boldsymbol\theta;\boldsymbol\beta) = \tfrac12|\boldsymbol\theta-\boldsymbol\beta|^2
- \psi(\boldsymbol\theta),
$$

with the observed delay between two images $\Delta t_{ij} = (D_{\Delta t}/c)
\big[\tau(\theta_i;\beta) - \tau(\theta_j;\beta)\big]$. The distance factor
is the *time-delay distance* $D_{\Delta t} \equiv (1+z_{\mathrm d})D_{\mathrm
d}D_{\mathrm s}/D_{\mathrm{ds}}$
([Ch. 15](15-distances.md#three-distances) built $D_{\mathrm d}$, $D_{\mathrm
s}$, $D_{\mathrm{ds}}$, and reminded you they do not add). Every
angular-diameter distance is $H_0$ times a dimensionless integral over
$\Omega_{\mathrm m}, \Omega_\Lambda$ and redshift alone
([Ch. 14](14-frw.md#friedmann)), so one net factor of $H_0$ survives the
product-over-quotient: $D_{\Delta t}\propto 1/H_0$ at fixed cosmology. This is
the entire basis of time-delay cosmography — measure $\Delta t$, model $\Delta
\tau$, read off $D_{\Delta t}$, read off $H_0$.

Now transform. Expand $\tau_{\lambda_{\mathrm{MST}}}\big(\boldsymbol\theta;
\lambda_{\mathrm{MST}}\boldsymbol\beta\big) -
\lambda_{\mathrm{MST}}\,\tau(\boldsymbol\theta;\boldsymbol\beta)$ using
$\eqref{eq:mst-psi}$: every term carrying $\boldsymbol\theta\cdot\boldsymbol\beta$
or $|\boldsymbol\theta|^2$ cancels, leaving a remainder that does not depend on
$\boldsymbol\theta$ at all:

$$
\tau_{\lambda_{\mathrm{MST}}}\big(\boldsymbol\theta;\lambda_{\mathrm{MST}}\boldsymbol\beta\big)
= \lambda_{\mathrm{MST}}\,\tau(\boldsymbol\theta;\boldsymbol\beta)
- \tfrac12\lambda_{\mathrm{MST}}(1-\lambda_{\mathrm{MST}})|\boldsymbol\beta|^2.
$$

The offset is the same constant at every image of one source, so it cancels
in any *difference*, and differencing is all a time delay is:

$$
\Delta\tau_{\lambda_{\mathrm{MST}}} = \lambda_{\mathrm{MST}}\,\Delta\tau.
\label{eq:mst-tau}
$$

On the SIS toy, $\Delta\tau = -0.8$ <!-- check: ch21.dtau_true = -0.8 ± 0.001 -->
at $\lambda_{\mathrm{MST}}=1$ and $-0.64$
<!-- check: ch21.dtau_lam = -0.64 ± 0.001 --> at $\lambda_{\mathrm{MST}}=0.8$
— a ratio of $0.8$ <!-- check: ch21.fermat_diff_ratio = 0.8 ± 1e-6 -->,
exactly $\lambda_{\mathrm{MST}}$, confirming $\eqref{eq:mst-tau}$ to machine
precision.

Here is the cost. $\Delta t_{\mathrm{obs}}$ is fixed data, independent of which
member of the family you fit. Standard practice assumes no sheet —
$\lambda_{\mathrm{MST}}=1$ — and solves $D_{\Delta t}^{\mathrm{assumed}} =
c\,\Delta t_{\mathrm{obs}}/\Delta\tau^{\mathrm{assumed}}$. If the true system
actually sits at some $\lambda^\star<1$ (a real, unmodeled sheet), then by
$\eqref{eq:mst-tau}$ the assumed model's own Fermat difference is
$\Delta\tau^{\mathrm{assumed}} = \Delta\tau^{\mathrm{true}}/\lambda^\star$ —
*larger* in magnitude — so it needs a *smaller* $D_{\Delta t}$ to reproduce the
same $\Delta t_{\mathrm{obs}}$. Since $D_{\Delta t}\propto1/H_0$, a smaller
inferred distance means a larger inferred $H_0$:

$$
H_0^{\mathrm{assumed}} = \frac{H_0^{\mathrm{true}}}{\lambda^\star}.
$$

At $\lambda^\star=0.8$ that is $H_0$ high by a factor of
$1.25$ <!-- check: ch21.h0_bias_factor = 1.25 ± 1e-6 --> — a $20\%$ unmodeled
sheet produces a $25\%$ overestimate of the Hubble constant, and imaging alone
can never catch it, because [the last section](#the-mass-sheet-degeneracy)
just showed imaging cannot see $\lambda_{\mathrm{MST}}$ at any value. This is
the reason time-delay $H_0$ measurements quote a systematic budget dominated
by "mass-sheet" or "external convergence," not by astrometry.

What *can* see it: a measurement that depends on mass, not on light-bending
geometry. The lensing galaxy's stellar velocity dispersion $\sigma_v$
([Ch. 10](10-galaxies.md#velocity-dispersion),
[Ch. 19](19-einstein-radius.md#theta-e-from-sigma-v)) is exactly that — DESI's
own Foundry survey notes it in as many words: *"the velocity dispersion of the
lensing galaxies can be used to break the mass-sheet degeneracy in lens
modeling"* (`reproductions/foundry-ii/data/paper_text.txt:1529`). A dynamical
mass and a lensing mass agreeing is not a formality; it is the only kind of
evidence this particular degeneracy cannot fake.

## Source versus mass { #source-vs-mass }

The mass-sheet transformation is the clean special case of a more general
fact: *any* smooth change to the deflection field can be exactly compensated
by a corresponding change to the (unknown) reconstructed source, provided the
source model is flexible enough to absorb it. MST is the special case where
the change is a uniform sheet and the compensation is a pure rescale of the
source position; nothing in the argument requires the compensation to be that
simple. This matters directly here because this campaign's forward model does
not fit a fixed source — it fits a Sérsic core plus $28$ linear shapelet
amplitudes, marginalized analytically
(`reproductions/claude-giga-lens/cgl/marg.py:31`,
[Ch. 22](22-inference.md#marginalising-linear-amplitudes)), a genuinely
flexible basis with room to reshape itself around a change in the mass model.

This repository has met a member of this family twice, and got it right only
once. The Foundry-I reproduction first described soft, ill-conditioned
posterior directions coupling $\gamma$, $e_1/e_2$, and
$\gamma_{\mathrm{ext}}$ as "mass-sheet / slope-ellipticity degeneracies"
(`reproductions/foundry-i/papers/main.tex:503`) — and then retracted the
label:

> *"An earlier draft labeled this 'mass–sheet, slope–ellipticity' degeneracies;
> we retract those labels — nothing in this single-band, fixed-source analysis
> invokes the mass-sheet transformation, and the slope–ellipticity coupling
> was an empirical observation of this posterior, not a citation-backed
> degeneracy. The valley largely closed once the likelihood was corrected."*
> (`reproductions/foundry-i/papers/evolution.tex:713`)

The retraction is correct for a reason worth internalizing: that campaign's
source was *fixed*, not flexible, and its parametrization contains no explicit
uniform-$\kappa$ sheet term at all — so $\lambda_{\mathrm{MST}}$ was never
actually a free direction of that model, whatever the posterior looked like.
An ill-conditioned valley is evidence something is *underconstrained*; it is
not, by itself, evidence of *which* named degeneracy is responsible, and here
the valley turned out to be a symptom of a misspecified likelihood, not a
structural symmetry — it shrank once the likelihood was fixed rather than once
more data arrived, which is exactly the signature of the former, not the
latter.

The claude-giga-lens campaign's own forward model, by contrast, *does* have a
flexible source, and it ran into a genuine source-versus-mass ambiguity there:
a Sérsic-versus-shapelet source-*centre* degeneracy, two clusters at
$-0.15$ <!-- check: ch21.source_centre_cluster_neg = -0.15 ± 0.01 --> and
$+0.09$ <!-- check: ch21.source_centre_cluster_pos = 0.09 ± 0.01 -->, with
$\hat R = 22.3/15.1$
<!-- check: ch21.rhat_source_centre_hi = 22.3 ± 0.05 -->
<!-- check: ch21.rhat_source_centre_lo = 15.1 ± 0.05 -->
(`reproductions/claude-giga-lens/CAMPAIGN.md:179`,
`reproductions/claude-giga-lens/papers/main.tex:914`). This one is real —
the source model is flexible enough for the mechanism to operate — but the
campaign explicitly checked whether it threatens $\gamma$, rather than
assuming either way, and found it decoupled. [Ch. 26](26-the-saddle.md#the-nuisance)
is where you verify that claim yourself rather than take it on faith; this
chapter's job is only to hand you the concept and the standard of proof it was
held to. "Degeneracy" is not an adjective you attach to any soft direction you
find. It is a specific, checkable claim: exhibit the transformation, in the
model you actually fit, that leaves the likelihood exactly unchanged. If you
cannot exhibit it, you have found ill-conditioning, which is a different
disease with a different cure.

## A degeneracy is a gauge symmetry { #degeneracy-is-gauge-symmetry }

Put the last three sections in one sentence: a degeneracy is a direction $v$
in parameter space along which the log-likelihood does not change at all,
$\mathcal L(x_0 + \epsilon v) = \mathcal L(x_0)$ for every $\epsilon$, not just
to first order. Differentiate that statement twice and $v$ is a null vector of
the Hessian — $v^\top H v = 0$ — which is exactly a zero eigenvalue
([Ch. 5](05-linear-algebra.md#definiteness-and-saddles)). A family of such
directions, indexed continuously the way $\lambda_{\mathrm{MST}}$ indexes one,
is precisely what a physicist calls a **gauge symmetry**: a continuous group
of transformations of the parameters that leaves every observable prediction
untouched. $\lambda_{\mathrm{MST}}\in\mathbb R$ is a one-parameter Lie group
acting on (mass model, source position) jointly, and $\eqref{eq:mst-beta}$ is
the statement that the likelihood built from image positions is invariant
under its action — a textbook example, not a metaphor.

!!! tip "You already know this"
    A rank-$r$ factorization $W=UV^\top$, the kind behind matrix completion,
    linear autoencoders, and embedding tables, has the identical structure:
    for any invertible $T\in GL(r)$, $(UT)(T^{-1}V^\top) = UV^\top$ reproduces
    the *same* $W$ and therefore the same loss, for every $T$. The minimum is
    not a point; it is an $r^2$-dimensional manifold of equally good
    solutions, and the loss Hessian evaluated anywhere on that manifold has
    exactly $r^2$ zero eigenvalues — one per direction you can move along the
    manifold for free. $\lambda_{\mathrm{MST}}$ is that same $T$, specialized
    to a one-parameter subgroup acting on a lens model instead of a
    factorization.

Three failure modes get conflated in practice and are worth pulling apart,
because this book meets all three:

- **An exact degeneracy** (zero eigenvalue, a real gauge symmetry) is
  *permanent*. No amount of more imaging data shrinks it, because the
  transformation leaves every imaging observable exactly fixed by
  construction — you need a qualitatively different observable ($\sigma_v$,
  not another image) to touch it at all. The mass-sheet family is one.
- **Ill-conditioning** (a small but nonzero eigenvalue, $\mathrm{cond}\to$
  large but not infinite) is a statement about how much *this* dataset, at
  *this* precision, constrains a direction — and it can shrink with a
  corrected likelihood or better data, as Foundry-I's retracted valley did.
  It is not protected by any symmetry and should never borrow one's name
  without exhibiting the transformation.
- **A saddle** (a *negative* eigenvalue) is neither of the above: it says the
  log-posterior *decreases* moving away from that point in both directions
  along $v$, which is a statement about a bad reference point, not about an
  invariance of the model. [Ch. 26](26-the-saddle.md#the-map-is-a-saddle)
  meets this one directly — the same $V\,\mathrm{diag}(1/|\lambda_i|)\,V^\top$
  formula that correctly builds a descent direction at a saddle is invalid as
  a covariance for exactly this reason.

$\mathrm{cond}\to\infty$ is the limit where the second failure mode becomes
indistinguishable, in finite precision, from the first — which is why
[Ch. 5](05-linear-algebra.md#conditioning)'s $\mathrm{cond}\sim10^{14}$
number and this chapter's $\lambda_{\mathrm{MST}}$ are two ends of the same
idea, not two different ones.

!!! note "γ Ledger"
    **What this chapter rules in or out about $\gamma = 1.103$
    <!-- check: ch25.gamma_money = 1.103 ± 0.008 -->:** two acquittals, no
    verdict. First, the broad, ill-conditioned valley coupling $\gamma$,
    $e_1/e_2$, and $\gamma_{\mathrm{ext}}$ that Foundry-I once called a
    mass-sheet degeneracy was retracted for good reason — that model contains
    no uniform-$\kappa$ sheet term, so $\lambda_{\mathrm{MST}}$ was never
    actually loose there, and the valley closed with the likelihood, not with
    more data. That result does not transfer automatically to
    claude-giga-lens, but it sets the correct standard of proof for the next
    item. Second, the one real source-versus-mass ambiguity this campaign's
    flexible shapelet source does exhibit — the source-centre $\hat
    R=22.3$ — is explicitly checked and found decoupled from $\gamma$
    (`reproductions/claude-giga-lens/CAMPAIGN.md:179`); Ch. 26 is where you
    confirm the mechanism rather than the claim. Neither incident touches the
    residual scale-dependence — $1.103$ against a $1.433$
    <!-- check: ch25.gamma_anchor = 1.433 ± 0.034 --> anchor — that
    [Ch. 25](25-money-number.md#the-sigma-arithmetic) attributes to
    source-model and PSF systematics. This chapter clears two suspects, not
    the crime.

## Connect to the repo { #connect }

- `reproductions/foundry-i/papers/evolution.tex:713`
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/foundry-i/papers/evolution.tex#L713)
  and `reproductions/foundry-i/papers/main.tex:503`
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/foundry-i/papers/main.tex#L503)
  — the retracted "mass-sheet / slope-ellipticity" label, and why the
  retraction is correct.
- `reproductions/claude-giga-lens/CAMPAIGN.md:179`
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/CAMPAIGN.md#L179)
  and `reproductions/claude-giga-lens/papers/main.tex:914`
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/papers/main.tex#L914)
  — the real source-centre $\hat R=22.3$, decoupled from $\gamma$;
  [Ch. 26](26-the-saddle.md#the-nuisance) verifies it.
- `reproductions/claude-giga-lens/cgl/marg.py:31`
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/marg.py#L31)
  — `marg_loglik`, the 28-amplitude analytic marginalization that makes this
  campaign's source flexible enough for a source-versus-mass ambiguity to
  bite at all.
- `site/guide_src/lensing.py:54`
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/lensing.py#L54)
  — `sis_deflection`, the function the worked example's algebra mirrors by
  hand.
- `site/guide_src/worked_examples.py` (`ch21_mass_sheet_degeneracy`) — the
  numeric checks behind every tagged number in this chapter.

## Exercises { #exercises }

??? question "Exercise 21.1 — Verify the invariance yourself"
    Using $\theta_{\mathrm E}=1$, $\beta_0=0.4$, images $\theta_1=1.4$,
    $\theta_2=-0.6$, and $\lambda_{\mathrm{MST}}=0.8$: compute
    $\alpha_{\lambda_{\mathrm{MST}}}(\theta) = \lambda_{\mathrm{MST}}\,\theta_{\mathrm E}
    \,\mathrm{sign}(\theta) + (1-\lambda_{\mathrm{MST}})\theta$ at both images,
    then $\beta_{\lambda_{\mathrm{MST}}}(\theta) = \theta -
    \alpha_{\lambda_{\mathrm{MST}}}(\theta)$. Do the two images agree on a
    single source position, and is it $\lambda_{\mathrm{MST}}\beta_0$?

    ??? success "Solution"
        $\alpha_{\lambda_{\mathrm{MST}}}(1.4) = 0.8(1) + 0.2(1.4) = 1.08$, so
        $\beta_{\lambda_{\mathrm{MST}}}(1.4) = 1.4-1.08=0.32$.
        $\alpha_{\lambda_{\mathrm{MST}}}(-0.6) = 0.8(-1)+0.2(-0.6)=-0.92$, so
        $\beta_{\lambda_{\mathrm{MST}}}(-0.6)=-0.6-(-0.92)=0.32$. Both give
        $0.32$ <!-- check: ch21.beta_lam_1 = 0.32 ± 0.001 -->
        $=0.8\times0.4$ exactly. The transformed lens is just as consistent
        with "one source" as the true one — imaging cannot distinguish them.

??? question "Exercise 21.2 — Derive the $H_0$ bias formula symbolically"
    Starting from $\Delta\tau_{\lambda_{\mathrm{MST}}} = \lambda_{\mathrm{MST}}
    \Delta\tau$ and $\Delta t_{\mathrm{obs}} = (D_{\Delta t}/c)\Delta\tau$,
    with $\Delta t_{\mathrm{obs}}$ fixed, derive
    $H_0^{\mathrm{assumed}}/H_0^{\mathrm{true}}$ as a function of
    $\lambda^\star$ alone (do not plug in $0.8$).

    ??? success "Solution"
        Both descriptions reproduce the same data:
        $D_{\Delta t}^{\mathrm{assumed}}\Delta\tau^{\mathrm{assumed}} =
        D_{\Delta t}^{\mathrm{true}}\Delta\tau^{\mathrm{true}}$
        $(=c\,\Delta t_{\mathrm{obs}})$. Substitute $\Delta\tau^{\mathrm{true}}
        = \lambda^\star\Delta\tau^{\mathrm{assumed}}$ (the assumed model sits
        at $\lambda_{\mathrm{MST}}=1$ relative to the true one at
        $\lambda^\star$): $D_{\Delta t}^{\mathrm{assumed}} =
        \lambda^\star D_{\Delta t}^{\mathrm{true}}$. Since
        $D_{\Delta t}\propto 1/H_0$, inverting gives
        $H_0^{\mathrm{assumed}} = H_0^{\mathrm{true}}/\lambda^\star$ — no
        approximation, exact at any $\lambda^\star$, and it reproduces
        $1.25$ <!-- check: ch21.h0_bias_factor = 1.25 ± 1e-6 --> at
        $\lambda^\star=0.8$.

??? question "Exercise 21.3 — What would make the retraction wrong?"
    Foundry-I retracted its "mass-sheet degeneracy" label because the fitted
    model has no explicit uniform-$\kappa$ term and a fixed source. State, in
    one sentence, what you would need to see in a model's parametrization
    (not its posterior) to justify the label, and why a wide, hard-to-sample
    valley is not by itself sufficient evidence.

    ??? success "Solution"
        You need the family itself to be exhibitable: some parameter or
        parameter combination whose variation, holding the fit's *other*
        free choices free to compensate (in this case, a source that can
        rescale), leaves the predicted images and flux ratios provably
        unchanged — i.e. you need to write down $\eqref{eq:mst-beta}$ or its
        analogue for your specific model and confirm it holds identically,
        not approximately. A wide valley only tells you the posterior is
        weakly informative in some combination of directions; it says
        nothing about *why*, and — as here — the reason is sometimes a fixable
        likelihood defect rather than a structural symmetry. Ill-conditioning
        is a symptom common to both; only the exhibited transformation
        distinguishes the diagnosis.

??? question "Exercise 21.4 — Saddle, degeneracy, or neither?"
    Chapter 5 records a Laplace Hessian with minimum eigenvalue $-14.85$
    <!-- check: ch05.saddle_min_eig = -14.85 ± 0.01 --> and five negative
    directions
    <!-- check: ch05.saddle_n_negative = 5 ± 0 --> at the campaign's polished
    MAP. Is that a mass-sheet-style degeneracy? If not, what single number
    from that same Hessian *would* signal one, and what value would it need?

    ??? success "Solution"
        No. A negative eigenvalue means the log-posterior *decreases* moving
        away from that point along that direction on both sides — a local
        maximum along every other direction but a minimum along this one,
        i.e. a saddle, which is a statement about a bad reference point. A
        genuine degeneracy needs a eigenvalue of exactly $0$ (flat, not
        curved either way) at the *true* mode, not at a misplaced one. The
        campaign's own diagnosis, in fact, finds the high-density point
        elsewhere entirely
        (<!-- check: ch05.gamma_at_true_peak = 1.10 ± 0.01 -->, higher
        posterior density than the saddle-consistent MAP) — which is exactly
        why a Hessian built at the wrong point cannot be trusted as a
        covariance, degenerate or not, and why [Ch. 26](26-the-saddle.md#the-same-matrix-twice)
        exists.
