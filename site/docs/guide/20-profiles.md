# SIS, SIE, EPL: what gamma actually means

[Ch. 19](19-einstein-radius.md) built the Einstein radius on top of one
special mass profile — a circular, isothermal sphere — because that is the
minimum object you need to make the definition concrete. This chapter takes
the training wheels off, twice. First it lets the profile be elliptical
instead of circular (SIS $\to$ SIE), because no real galaxy is circular on
the sky. Then it lets the density slope itself vary instead of being fixed at
"isothermal" (SIE $\to$ EPL), because that slope, called $\gamma$, is the
single number this whole repository's final report exists to pin down. By
the end of this chapter you can write down exactly
what the claim "$\gamma = 1.103$" asserts about a galaxy's mass, derive the
one formula that normalizes it, and — because this repo uses the letter
$\gamma$ for three unrelated things — you will know, in every later chapter,
which one is meant. This chapter also opens the **$\gamma$ Ledger** formally:
every later chapter that touches the money number adds a row.

!!! abstract "What you can skip"
    You already own power laws, $2\times2$ linear maps, and Taylor expansion
    to second order ([Ch. 2](02-derivatives.md)). If you are comfortable with
    a doubled-angle (spin-2) representation of orientation — the same idea as
    an unsigned gradient direction or a structure-tensor eigenvector in
    computer vision — skim [Ellipticity](#ellipticity) for the repo-specific
    numbers only. If you already trust [Ch. 5](05-linear-algebra.md)'s
    Jacobian decomposition $A = (1-\kappa)I - \Gamma$, skim
    [External shear](#external-shear) for the same reason.

## SIS to SIE { #sis-to-sie }

The singular isothermal sphere (SIS) from [Ch. 10](10-galaxies.md#the-isothermal-conspiracy)
has convergence $\kappa_{\mathrm{SIS}}(\theta) = \theta_{\mathrm{E}}/(2\theta)$
and — [Ch. 6](06-vector-calculus.md#poisson-for-lensing) hands you this fact
without proving it — a deflection of *constant modulus*,
$\boldsymbol\alpha = \theta_{\mathrm{E}}\hat{\boldsymbol\theta}$
(`site/guide_src/lensing.py:54`). Here is the proof, and it is short because
it reuses machinery you already have. For any circularly symmetric lens,
[Ch. 6](06-vector-calculus.md#poisson-for-lensing)'s $\nabla^2\psi=2\kappa$,
written out as the radial Laplacian of a function that depends only on
$\theta$, together with $\alpha = d\psi/d\theta$, is
$\frac{1}{\theta}\frac{d}{d\theta}\!\big(\theta\,\alpha(\theta)\big) = 2\kappa(\theta)$.
Integrate both sides from $0$ to $\theta$:

$$
\theta\,\alpha(\theta) \;=\; \int_0^\theta 2\kappa(t)\,t\,dt \;=\; \theta^2\,\bar\kappa(\theta)
\quad\Longrightarrow\quad
\alpha(\theta) \;=\; \theta\,\bar\kappa(\theta),
$$

using [Ch. 19](19-einstein-radius.md#the-mean-convergence-identity)'s own
definition of $\bar\kappa$ in the last step. This is a 2-D Newton's shell
theorem: the deflection at radius $\theta$ depends only on the *mean*
convergence enclosed, never on how that mass is arranged inside. For the SIS,
$\bar\kappa_{\mathrm{SIS}}(\theta) = \theta_{\mathrm{E}}/\theta$ (the same
power law as $\kappa$ itself, one section below), so
$\alpha(\theta) = \theta\cdot(\theta_{\mathrm{E}}/\theta) = \theta_{\mathrm{E}}$
— every $\theta$ cancels, leaving a deflection that does not know how far out
you are. That is a special, $\gamma=2$-specific coincidence, not a generic
property of power-law lenses; the general case returns in the next section.

Real galaxies, including every lens this repository models, are not
circular — light and mass both trace out ellipses on the sky. The natural
next model keeps the isothermal ($\gamma=2$) density law but flattens its
*iso-density contours* into ellipses of axis ratio $q$: the singular
isothermal ellipsoid (SIE). Its deflection has a standard closed form from
the lensing literature, implemented at `site/guide_src/lensing.py:65`:

$$
\alpha_x = \frac{\theta_{\mathrm E}\,q}{\sqrt{1-q^2}}\arctan\!\left(\frac{\sqrt{1-q^2}\,x}{\psi+s}\right),
\qquad
\alpha_y = \frac{\theta_{\mathrm E}\,q}{\sqrt{1-q^2}}\operatorname{artanh}\!\left(\frac{\sqrt{1-q^2}\,y}{\psi+q^2 s}\right),
$$

with $\psi = \sqrt{q^2(x^2+s^2)+y^2}$, $(x,y)$ already rotated into the frame
aligned with the ellipse's own axes (`lensing.py`'s `_rotate` handles turning
the position angle $\phi$ in and back out around this formula), and $s$ a
small core radius that regularizes the point-mass singularity at the center.
Gigalens' own `sie.py` defines the identical constant, `s_scale = 1e-4`
([`vendor/gigalens-sean/.../mass/sie.py:11`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/profiles/mass/sie.py#L11))
— but a same-named local variable shadows it to $0$ one line into
`_param_conv`
([`:16`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/profiles/mass/sie.py#L16)),
so gigalens' own SIE actually runs with zero core radius; only the unused
class attribute matches `lensing.py`'s choice, not gigalens' executed
behavior.
Deriving that closed form is a genuine complex-analysis exercise (integrating
an elliptical isothermal potential); this guide takes it as a fact and checks
the one property that matters: at $q\to1$, the SIE must reduce to the SIS.
`lensing.sie_deflection` and `lensing.sis_deflection`, evaluated at the same
point $(\theta_x,\theta_y)=(0.6,0.8)''$ (so $r=\theta_{\mathrm{E}}=1''$),
agree to within $0.057''$ at $q=0.9$
<!-- check: ch20.sie_sis_diff_q0p9 = 0.0569 ± 0.0001 -->
and to within $0.00064''$ at $q=0.999$
<!-- check: ch20.sie_sis_diff_q0p999 = 0.000639 ± 0.00001 -->
— shrinking roughly in proportion to $1-q$, as the closed form's own
$\sqrt{1-q^2}$ factors predict. It does not shrink to machine zero, because
`sie_deflection` clips $q$ at $0.99999$ internally and carries a nonzero core
$s=10^{-4}$ by default (`lensing.py:65,73`) — a real numerical seam, not a bug:
the exact $q=1$ limit is a removable singularity in the formula, and the code
sidesteps computing it by staying a hair away from it.

## The EPL and gamma { #the-epl-and-gamma }

The SIE fixed the density slope at $\gamma=2$. [Ch. 10](10-galaxies.md#the-isothermal-conspiracy)
already flagged that real ellipticals only *cluster near* isothermal, not
sit exactly on it — so the campaign's actual mass model promotes $\gamma$
from a constant to a free parameter, giving the elliptical power law (EPL):
the same isothermal ellipsoid, generalized to $\rho(r)\sim r^{-\gamma}$ for
any $\gamma$. [Ch. 3](03-integrals.md#the-abel-projection) already derived
the *exponent* this profile projects to: $\Sigma(R)\sim R^{1-\gamma}$, hence
$\kappa(R)\sim R^{1-\gamma}$, for any $\gamma$ — that chapter also showed
that the exponent's own literal amplitude, $2A\,I(\gamma)$ in
[Ch. 3, Eq. 3.1](03-integrals.md#the-abel-projection), blows up for an
idealized, untruncated halo as $\gamma\to1^+$ (the $p$-test threshold on
$I(\gamma)$), and explicitly left this chapter the job of saying where the
campaign's *actual* prefactor, $(3-\gamma)/2$, comes from.

It comes from Ch. 19's identity, not from that divergent integral. Posit
$\kappa(R) = A(\gamma)\,(\theta_{\mathrm E}/R)^{\gamma-1}$ — the right
exponent, an undetermined amplitude — and *demand* that
$\bar\kappa(\theta_{\mathrm E})=1$, the defining property of the Einstein
radius for any profile. Integrating exactly as in the SIS derivation above,

$$
\bar\kappa(\theta) = \frac{2}{\theta^2}\int_0^\theta A\,\theta_{\mathrm E}^{\gamma-1}t^{1-\gamma}\,t\,dt
= \frac{2A}{3-\gamma}\left(\frac{\theta_{\mathrm E}}{\theta}\right)^{\gamma-1},
$$

so $\bar\kappa(\theta_{\mathrm E}) = 2A/(3-\gamma)$; setting this to $1$ gives
$A = (3-\gamma)/2$ — exactly the constant `epl_kappa` uses
(`site/guide_src/lensing.py:81`):

$$
\kappa(R) \;=\; \frac{3-\gamma}{2}\left(\frac{\theta_{\mathrm E}}{R}\right)^{\gamma-1}.
$$

This normalization sidesteps Ch. 3's divergence entirely: it is fixed by a
convention (mean interior $\kappa=1$ at one radius), not by integrating an
infinite halo, so it stays perfectly finite for every $\gamma$ the prior
allows — including exactly at $\gamma=1$. Gigalens' own EPL implementation
uses the identical exponent under the lenstronomy convention
$t=\gamma-1$ ([`vendor/gigalens-sean/.../mass/epl.py:24`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/profiles/mass/epl.py#L24));
`site/guide_src/lensing.py` is deliberately not a wrapper around it, but the
convergence law is the same law.

Two facts fall out of this derivation for free. First, setting $R=\theta_{\mathrm E}$
in the boxed formula gives the *local* convergence right at the Einstein
ring: $\kappa(\theta_{\mathrm E}) = (3-\gamma)/2$ — a different number from
the *mean* interior convergence, which is exactly $1$ for every $\gamma$ by
construction. At $\gamma=2$ (isothermal) these give $0.5$
<!-- check: ch20.kappa_theta_e_isothermal = 0.5 ± 1e-9 -->:
the density right at an isothermal Einstein ring is exactly half the mean
density inside it. Second, at $\gamma=1$ exactly, the exponent $\gamma-1$
vanishes and $\kappa(R) = 1$ for *every* $R$, not just at $\theta_{\mathrm E}$
<!-- check: ch20.kappa_theta_e_wall = 1.0 ± 1e-9 -->
— a perfectly finite, spatially uniform sheet. That is the critical mass
sheet: [Ch. 21](21-degeneracies.md#the-mass-sheet-degeneracy) shows a uniform
$\kappa=1$ sheet is invisible to every imaging observable, the single most
degenerate mass distribution in all of lensing. So $\gamma=1$ is a wall in
*two* related but distinct senses — Ch. 3's sense (an idealized untruncated
halo's total surface density literally diverges there) and this chapter's
sense (even the finite, correctly normalized model becomes totally
unidentifiable there) — and the campaign's own prior,
`gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 2.7)`
(`reproductions/claude-giga-lens/cgl/e2.py:110`), places its hard floor at
exactly that boundary.

Hold that against the money number. $\gamma_{\mathrm{binned}}(\mathrm{corr,low})=1.103$
<!-- check: ch25.gamma_money = 1.103 ± 0.008 --> gives
$\kappa(\theta_{\mathrm E}) = 0.9485$
<!-- check: ch20.kappa_theta_e_money = 0.9485 ± 1e-9 -->
— only $0.0515$ short of the fully degenerate sheet's $\kappa(\theta_{\mathrm E})=1$
<!-- check: ch20.kappa_theta_e_money_gap_to_wall = 0.0515 ± 1e-9 -->,
and well past isothermal's own $0.5$ on the way there.
[Ch. 10](10-galaxies.md#the-isothermal-conspiracy) already flagged this value
as $3.588$ prior-sigma from isothermal
<!-- check: ch25.prior_sigmas_below_prior_mean = 3.588 ± 0.001 -->;
this is a sharper, structural way to be worried about the same number,
because it says something about *identifiability*, not just prior surprise.
The anchor,
$\gamma=1.433$ <!-- check: ch25.gamma_anchor = 1.433 ± 0.034 -->, sits at
$\kappa(\theta_{\mathrm E})=0.7835$
<!-- check: ch20.kappa_theta_e_anchor = 0.7835 ± 1e-9 -->
— a real departure from isothermal, but nowhere near the degenerate wall. The
fine-product artifact, $\gamma=2.585$
<!-- check: ch25.gamma_fine_diag_artifact = 2.585 ± 0.001 -->, sits at $0.2075$
<!-- check: ch20.kappa_theta_e_artifact = 0.2075 ± 1e-9 -->,
on the opposite, steep side. All four values share the same mean interior
convergence, exactly $1$ at $\theta_{\mathrm E}$
<!-- check: ch20.mean_kappa_money = 1.0 ± 1e-6 -->
— that identity, checked here for the money slope by direct quadrature, does
not care what $\gamma$ is; only the *local* number does.

<figure markdown="span">
  ![kappa(R) for the four EPL slopes this campaign reported for one galaxy](figures/ch20-epl-slope-light.svg#only-light){ width="90%" }
  ![kappa(R) for the four EPL slopes this campaign reported for one galaxy](figures/ch20-epl-slope-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 20.1.** $\kappa(R)$ for the money number
  ($\gamma=1.103$), the anchor ($\gamma=1.433$), the isothermal reference
  ($\gamma=2.000$), and the fine-product artifact ($\gamma=2.585$) — the same
  four values [Ch. 25](25-money-number.md) adjudicates. Every curve crosses
  $(3-\gamma)/2$ at $R=\theta_{\mathrm E}$ (the dotted line) by construction,
  and every curve has the identical mean interior convergence there. The
  money curve is visibly the flattest: a shallow slope spreads mass out
  instead of concentrating it, which is exactly why its $\kappa(\theta_{\mathrm E})$
  sits closest to the fully degenerate, perfectly flat sheet at $\gamma=1$.</figcaption>
</figure>

!!! note "The gamma overload, paid in full"
    This repository uses the single glyph $\gamma$ for three unrelated
    objects, and the report's own authors clearly knew it:
    `reproductions/claude-giga-lens/papers/main.tex` defines separate macros
    for each
    ([lines 24](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/papers/main.tex#L24)–`27`:
    `\gammaEPL`, `\gext`, `\gextone`, `\gexttwo`). This guide mirrors that
    discipline exactly:

    1. **Bare $\gamma$** — the EPL's 3-D density slope, $\rho\sim r^{-\gamma}$,
       this section's subject and the campaign's money parameter. This is the
       *only* meaning bare $\gamma$ ever carries in this guide.
    2. **$\gamma_{\mathrm{ext}}$** — the external shear's *magnitude*, the
       next section's subject. Never bare $\gamma$, never $\gamma_{sh}$.
    3. **$\gamma_1,\gamma_2$** — shear *components*. These name two different
       but related quantities depending on context: [Ch. 5](05-linear-algebra.md#symmetric-2x2)'s
       traceless part of a Jacobian ($A=(1-\kappa)I-\Gamma$, computed from
       whatever deflection field you differentiate), or the external-shear
       model's own two parameters (next section). The next section shows
       these literally coincide when the external field is isolated — but
       not in general, since a full EPL+shear Jacobian sums a spatially
       varying piece (from the EPL) and a constant piece (from the shear).
       Wherever both could appear in the same breath, this guide subscripts
       the external one: $\gamma_{\mathrm{ext},1},\gamma_{\mathrm{ext},2}$.

## Ellipticity { #ellipticity }

Every ellipse in this repository's model — the EPL mass, the SIE, and every
one of the four lens-light Sersic components plus the source — is
parameterized the same way: not by $(q,\phi)$ directly, but by a pair
$(e_1,e_2)$ that gigalens' `epl.py` and `sie.py` both convert identically,
implemented once at `site/guide_src/lensing.py:32`:

$$
\phi = \tfrac{1}{2}\operatorname{atan2}(e_2,e_1), \qquad
c = |e| = \sqrt{e_1^2+e_2^2}, \qquad
q = \frac{1-c}{1+c}.
$$

The half-angle is not a stylistic choice. Write the complex ellipticity
$\epsilon = e_1+ie_2$. If $c=|\epsilon|$ is fixed, the map $\phi\mapsto\epsilon=c\,e^{2i\phi}$
sends a *full* $180^\circ$ rotation of the position angle, $\phi\to\phi+\pi$,
to $\epsilon\,e^{2\pi i}=\epsilon$ — the *identical* complex number. That has
to be true: an ellipse has no arrowhead, so rotating it by $180^\circ$
produces the same ellipse, and the parameterization had better not invent a
difference where there is none. A *quarter* turn, $\phi\to\phi+\pi/2$, is a
different story — it swaps which axis is which, and $\epsilon$ must change:
$\epsilon\,e^{i\pi}=-\epsilon$.

Take $(e_1,e_2)=(0.3,0.4)$ — a 3-4-5 triangle, chosen for clean numbers, not
a typical draw from the mass ellipticity's own prior
(`e1,e2 ~ Normal(0, 0.1)`, `cgl/e2.py:111`, a few times smaller).
`ellip_to_q_phi` gives $q = 1/3$
<!-- check: ch20.ellip_q = 0.333333 ± 1e-6 -->
and $\phi = 26.565^\circ$
<!-- check: ch20.ellip_phi_deg = 26.565051 ± 1e-4 -->.
Reconstructing $\epsilon$ at $\phi+\pi$ with the same $c=(1-q)/(1+q)=0.5$
reproduces $e_1=0.3$ exactly
<!-- check: ch20.spin2_e1_at_phi_plus_pi = 0.3 ± 1e-9 -->
— the invariance holds to floating-point precision, as the algebra above
promised — while reconstructing at $\phi+\pi/2$ flips the sign,
$e_1=-0.3$ <!-- check: ch20.spin2_e1_at_phi_plus_halfpi = -0.3 ± 1e-9 -->.

!!! tip "You already know this"
    This is exactly the doubled-angle trick used everywhere an "orientation"
    has no arrowhead — a line, an edge, a nematic director. A structure
    tensor's dominant eigenvector in image processing is conventionally
    reported mod $\pi$, not mod $2\pi$, for the identical reason: an unsigned
    gradient direction and its $180^\circ$-rotated twin describe the same
    edge. Spin-2 is the physics name for a doubled-angle representation you
    have almost certainly already coded.

## External shear { #external-shear }

Mass sitting outside the modeled lens galaxy — a companion, the group or
cluster environment, structure along the line of sight — still bends light,
but its potential $\psi_{\mathrm{ext}}(x,y)$ near the lens center is smooth
and slowly varying compared to the galaxy's own steep potential. Taylor-expand
it to second order ([Ch. 2](02-derivatives.md#taylor)): the constant term is
an unobservable overall phase, the linear term is a uniform image shift
indistinguishable from moving the source (an unobservable degeneracy), so the
first term with any observable content is the *quadratic* one — a constant
Hessian. If none of that external mass is itself sitting at the lens plane
within the modeled region, [Ch. 6](06-vector-calculus.md#poisson-for-lensing)'s
$\nabla^2\psi=2\kappa$ forces the trace of that constant Hessian to vanish
locally, leaving only its traceless part: a spatially *uniform* shear,
$(\gamma_{\mathrm{ext},1},\gamma_{\mathrm{ext},2})$, and nothing else. That
is the external-shear model, `shear_deflection`
(`site/guide_src/lensing.py:93`):

$$
\alpha_x = \gamma_{\mathrm{ext},1}\,x+\gamma_{\mathrm{ext},2}\,y, \qquad
\alpha_y = \gamma_{\mathrm{ext},2}\,x-\gamma_{\mathrm{ext},1}\,y.
$$

The campaign's own prior draws each component independently,
$\gamma_1,\gamma_2\sim\mathrm{Normal}(0,0.05)$
(`reproductions/claude-giga-lens/cgl/e2.py:113`–`114`) — note the code's bare
parameter names, exactly the collision this chapter's ledger warned about;
in this guide's notation those are $\gamma_{\mathrm{ext},1},\gamma_{\mathrm{ext},2}$.
If you want the shear quoted as a single magnitude-and-angle pair the way
$\gamma_{\mathrm{ext}}$ is defined in the notation contract, it is
$\gamma_{\mathrm{ext}}=\sqrt{\gamma_{\mathrm{ext},1}^2+\gamma_{\mathrm{ext},2}^2}$
— the shear's own analogue of ellipticity's $c=|e|$ above.

Differentiate `shear_deflection` alone with `lens_jacobian`
([Ch. 5](05-linear-algebra.md#symmetric-2x2)'s numerical Jacobian, applied
here to a deflection field that has nothing else in it) at one plausible
draw, $\gamma_{\mathrm{ext},1}=0.03,\ \gamma_{\mathrm{ext},2}=-0.02$ (so
$\gamma_{\mathrm{ext}}=0.036056$
<!-- check: ch20.ext_shear_magnitude = 0.036056 ± 1e-5 -->),
evaluated at an arbitrary point $(0.4,0.7)''$: `kappa_gamma_from_jacobian`
recovers $\kappa=0$
<!-- check: ch20.ext_shear_kappa_recovered = 0.0 ± 1e-9 -->,
$\gamma_1=0.03$
<!-- check: ch20.ext_shear_gamma1_recovered = 0.03 ± 1e-6 -->, and
$\gamma_2=-0.02$
<!-- check: ch20.ext_shear_gamma2_recovered = -0.02 ± 1e-6 -->
— exactly the input, to machine precision, at *every* point you would care
to check, because $\alpha_x,\alpha_y$ are linear in $(x,y)$ and a linear
map's own derivative has no remainder to miss ([Ch. 2](02-derivatives.md#taylor)'s
point about Taylor expansion, cashed out as a finite-difference check with
zero truncation error). This is the resolution the overload warning
promised: external shear is not a *new* kind of object, it is Ch. 5's
$(\gamma_1,\gamma_2)$ decomposition of the total Jacobian, restricted to just
the one additive, spatially constant term that an external tidal field
contributes. In a full EPL+shear fit, the total Jacobian sums this constant
piece with the EPL's own spatially varying convergence and shear — the two
notions of $\gamma_1,\gamma_2$ genuinely coincide only when you isolate the
external term this way, which is exactly why the notation contract insists
on the subscript everywhere else. One further honest note: an external
shear's stretch and the lens's own ellipticity both distort images in
visually similar ways, and disentangling the two from imaging data alone is
a real, separate degeneracy — [Ch. 21](21-degeneracies.md) is where
degeneracies of this kind get a full treatment; this chapter only flags it.

## Source models { #source-models }

Everything so far describes the *lens*: the foreground galaxy's mass (EPL +
shear) and light (four Sersic components, [Ch. 10](10-galaxies.md#the-sersic-profile)).
The *source* — the background galaxy actually being lensed, whose light,
run through the lens equation, produces the arcs and rings
[Ch. 22](22-inference.md#the-forward-model) fits pixel by pixel — needs its
own light model, and a single Sersic profile is not flexible enough: real
source galaxies have clumps, spiral structure, and asymmetries no smooth
elliptical profile captures. The campaign's source model is a Sersic
component (its own independent $e_1,e_2\sim\mathrm{TruncatedNormal}(0,0.15,-0.5,0.5)$,
a *third* distinct ellipticity prior alongside the mass's $\mathrm{Normal}(0,0.1)$
and the lens light's $\mathrm{TruncatedNormal}(0,0.15,-0.4,0.4)$;
`cgl/e2.py:120`–`125`) *plus* a shapelet expansion layered on top.

Shapelets are 2-D Gauss–Hermite basis functions — a Gaussian envelope times a
Hermite polynomial in $x$ and $y$, truncated at total polynomial order
$n_{\max}$
([`vendor/gigalens-sean/.../light/shapelets.py:19`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/profiles/light/shapelets.py#L19)).
At $n_{\max}=6$ — the campaign's choice — the count of independent modes is
the triangular number $(n_{\max}+1)(n_{\max}+2)/2 = 28$
<!-- check: ch20.shapelet_depth_nmax6 = 28 ± 1e-9 -->,
matching the model's own docstring, "Sersic + Shapelets($n_{\max}=6$) source
with 28 EXPLICIT amps"
(`reproductions/claude-giga-lens/cgl/e2.py:13`), each with an independent
amplitude parameter and its own Gaussian prior
(`cgl/e2.py:126`–`131`).

!!! tip "You already know this"
    A truncated orthogonal basis layered on top of a smooth parametric fit
    to catch residual structure is a move you have made before, whatever you
    called it — a low-order DCT or wavelet expansion of a residual image, a
    handful of extra principal components kept after the dominant ones. The
    physics content here is only which basis (2-D Hermite functions under a
    Gaussian weight, not sines or wavelets) and how many terms.

Because each shapelet amplitude multiplies a *fixed* rendered image and the
28 contributions simply sum, these amplitudes enter the forward model
*linearly* — unlike $\gamma$, $q$, $\phi$, or any of the profile shape
parameters above, all of which enter nonlinearly. That linearity is exactly
what [Ch. 22](22-inference.md#marginalising-linear-amplitudes) exploits: the
28 amplitudes get integrated out analytically (a ridge-regularized normal
equation, `reproductions/claude-giga-lens/cgl/marg.py`) rather than sampled,
which is why the campaign's sampled parameter count drops from 74 dimensions
to 46 (`cgl/e2.py:13`). This chapter only names the 28 parameters being
marginalized; deriving the marginalization itself is Ch. 22's job.

!!! note "γ Ledger"
    **What this chapter rules in or out about $\gamma = 1.103$:** nothing
    empirically — no data enters this chapter, only definitions. What it
    does establish is the *structural* stakes. $\gamma=1$ is not an
    arbitrary round number for the prior's hard wall: it is exactly where
    the EPL's local convergence at $\theta_{\mathrm E}$ becomes a spatially
    uniform sheet, the most degenerate mass distribution lensing can produce
    ([Ch. 21](21-degeneracies.md#the-mass-sheet-degeneracy)). The money
    number's own $\kappa(\theta_{\mathrm E})=0.9485$
    <!-- check: ch20.kappa_theta_e_money = 0.9485 ± 1e-9 --> sits only
    $0.0515$ short of it
    <!-- check: ch20.kappa_theta_e_money_gap_to_wall = 0.0515 ± 1e-9 -->
    — a sharper, more structural way to read its closeness to the wall than
    [Ch. 10](10-galaxies.md#the-isothermal-conspiracy)'s $3.588$-prior-sigma
    framing
    <!-- check: ch25.prior_sigmas_below_prior_mean = 3.588 ± 0.001 -->,
    though both point at the same number.
    Whether the *data* genuinely demand a profile sitting that close to
    degenerate, or whether some part of the pipeline pushed it there
    artificially, is not this chapter's question — it belongs to
    [Ch. 21](21-degeneracies.md), [Ch. 25](25-money-number.md#the-sigma-arithmetic),
    and [Ch. 26](26-the-saddle.md).

## Connect to the repo { #connect }

- `site/guide_src/lensing.py:32` (`ellip_to_q_phi`), `:65` (`sie_deflection`),
  `:81` (`epl_kappa`), `:93` (`shear_deflection`), `:179` (`mean_kappa_within`) —
  every function this chapter computes with.
- [`reproductions/claude-giga-lens/cgl/e2.py:109`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L109)–`114` —
  the lens-mass prior block: `theta_E`, `gamma` (line 110), `e1`, `e2`
  (line 111), `gamma1`, `gamma2` (lines 113–114).
- [`reproductions/claude-giga-lens/cgl/e2.py:120`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L120)–`131` —
  the source Sersic and shapelet priors, and the 28 amplitude names.
- [`vendor/gigalens-sean/.../mass/epl.py:9`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/profiles/mass/epl.py#L9)
  and `:24` (`t = gamma - 1`) — the production EPL, the lenstronomy convention.
- [`vendor/gigalens-sean/.../mass/sie.py:17`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/profiles/mass/sie.py#L17)–`19` —
  the SIE's own $(e_1,e_2)\to(q,\phi)$ conversion, identical to `epl.py`'s.
- [`vendor/gigalens-sean/.../light/shapelets.py:19`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/profiles/light/shapelets.py#L19) —
  the shapelet depth formula, `(n_max+1)*(n_max+2)/2`.
- [`reproductions/claude-giga-lens/papers/main.tex:24`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/papers/main.tex#L24)–`27` —
  the report's own `\gammaEPL`/`\gext`/`\gextone`/`\gexttwo` macros, the same
  disambiguation this chapter makes in prose.
- `reproductions/claude-giga-lens/CAMPAIGN.md:134` — the money number,
  `gamma_binned(corr, low) = 1.103 ± 0.008`, for one real system,
  DESI-165.4754−06.0423.

## Exercises { #exercises }

??? question "Exercise 20.1 — deriving kappa(theta_E) = (3-gamma)/2 by hand"
    Starting from $\kappa(R) = A(\theta_{\mathrm E}/R)^{\gamma-1}$ and the
    requirement $\bar\kappa(\theta_{\mathrm E})=1$, redo the integral in
    [The EPL and gamma](#the-epl-and-gamma) to solve for $A$, then evaluate
    $\kappa(\theta_{\mathrm E})$ at $\gamma=2$ and at $\gamma=1$. Explain in
    one sentence why the $\gamma=1$ answer is special.

    ??? success "Solution"
        $\bar\kappa(\theta)=\dfrac{2A}{3-\gamma}\Big(\dfrac{\theta_{\mathrm E}}{\theta}\Big)^{\gamma-1}$,
        so $\bar\kappa(\theta_{\mathrm E})=\dfrac{2A}{3-\gamma}=1 \Rightarrow A=\dfrac{3-\gamma}{2}$.
        Then $\kappa(\theta_{\mathrm E})=A\cdot1^{\gamma-1}=A=(3-\gamma)/2$: at
        $\gamma=2$, $\kappa(\theta_{\mathrm E})=0.5$
        <!-- check: ch20.kappa_theta_e_isothermal = 0.5 ± 1e-9 -->; at
        $\gamma=1$, $\kappa(\theta_{\mathrm E})=1.0$
        <!-- check: ch20.kappa_theta_e_wall = 1.0 ± 1e-9 -->. The $\gamma=1$
        case is special because the *exponent* $\gamma-1$ is also zero there,
        so $\kappa(R)=1$ everywhere, not just at $\theta_{\mathrm E}$ — the
        local and mean statements coincide only at this one value of $\gamma$.

??? question "Exercise 20.2 — the SIE-to-SIS floor"
    Using `lensing.sie_deflection` and `lensing.sis_deflection` at
    $(\theta_x,\theta_y)=(0.6,0.8)''$, $\theta_{\mathrm E}=1''$, evaluate the
    gap at $q=0.999999$ (six nines) and compare it to the $q=0.999$ gap
    quoted in the main text. Why does pushing $q$ three orders of magnitude
    closer to $1$ not shrink the gap by a comparable factor?

    ??? success "Solution"
        ```python
        import lensing as L
        import numpy as np
        x, y = np.array([0.6]), np.array([0.8])
        sis = L.sis_deflection(x, y, 1.0)
        sie = L.sie_deflection(x, y, 1.0, 0.999999, 0.0)
        ```
        The gap at $q=0.999999$ is $1.046\times10^{-4}$
        <!-- check: ch20.sie_sis_diff_q0p999999 = 0.0001046 ± 0.000001 -->
        — pushing $q$ a thousand times closer to $1$ than $q=0.999$ only
        shrank the gap from $6.39\times10^{-4}$ by a factor of six, not by
        another thousand. It has hit a floor. The reason is in the code, not
        the algebra: `sie_deflection` clips its own $q$ argument to a maximum
        of $0.99999$ before doing anything else (`site/guide_src/lensing.py:73`),
        so passing $q=0.999999$ silently computes with $q=0.99999$ instead.
        Combined with the nonzero core radius $s=10^{-4}$ that never
        vanishes, the *implemented* function cannot get arbitrarily close to
        the SIS — only the *mathematical* limit can. A derivation checked
        against code always inherits the code's own approximations.

??? question "Exercise 20.3 — the quarter-turn, algebraically"
    Using $\epsilon = c\,e^{2i\phi}$, show algebraically why
    $\phi\to\phi+\pi/2$ flips the sign of $\epsilon$ while
    $\phi\to\phi+\pi$ leaves it unchanged, and confirm your answer against
    the numbers in [Ellipticity](#ellipticity).

    ??? success "Solution"
        $\epsilon(\phi+\pi/2) = c\,e^{2i(\phi+\pi/2)} = c\,e^{2i\phi}e^{i\pi} = -c\,e^{2i\phi} = -\epsilon(\phi)$,
        while $\epsilon(\phi+\pi) = c\,e^{2i\phi}e^{2i\pi} = c\,e^{2i\phi} = \epsilon(\phi)$,
        since $e^{2\pi i}=1$ but $e^{i\pi}=-1$. The factor of $2$ in the
        exponent is exactly what turns a $180^\circ$ geometric rotation into
        a *full* $360^\circ$ turn of the complex phase (a no-op) while a
        $90^\circ$ geometric rotation becomes only a $180^\circ$ phase turn
        (a sign flip) — matching
        $e_1(\phi+\pi)=0.3$ <!-- check: ch20.spin2_e1_at_phi_plus_pi = 0.3 ± 1e-9 -->
        and
        $e_1(\phi+\pi/2)=-0.3$ <!-- check: ch20.spin2_e1_at_phi_plus_halfpi = -0.3 ± 1e-9 -->
        exactly.

??? question "Exercise 20.4 — where the mean-kappa identity gets numerically hard"
    Using `mean_kappa_within`, compute $\bar\kappa(\theta_{\mathrm E})$ for
    $\gamma=1.103$ and for $\gamma=2.585$. Both should equal $1$ exactly by
    the derivation in this chapter. One of them does, to nine decimal
    places; the other misses by half a percent. Which, and why?

    ??? success "Solution"
        ```python
        import lensing as L
        import numpy as np
        f = lambda t, g: L.epl_kappa(t, np.zeros_like(t), 1.0, g, 1.0)
        L.mean_kappa_within(lambda t: f(t, 1.103), 1.0)   # 0.999999999903
        L.mean_kappa_within(lambda t: f(t, 2.585), 1.0)   # 0.993810340006
        ```
        At $\gamma=1.103$, $\bar\kappa(\theta_{\mathrm E})=0.999999999903$
        <!-- check: ch20.mean_kappa_money = 1.0 ± 1e-6 -->
        — indistinguishable from the exact $1$. At $\gamma=2.585$, it comes
        out to $0.993810$
        <!-- check: ch20.mean_kappa_artifact = 0.9938 ± 0.001 -->,
        off by half a percent. `mean_kappa_within` integrates $\kappa(t)\,t$
        on a *linear* grid starting a small distance from $t=0$. The
        integrand behaves like $t^{2-\gamma}$ near the origin: for
        $\gamma=1.103$, the exponent $2-\gamma=0.897$ is positive, so the
        integrand vanishes smoothly at the center and a linear grid resolves
        it fine. For $\gamma=2.585$, $2-\gamma=-0.585$ is negative — the
        integrand *diverges* as $t\to0$ (integrably, but steeply), and a
        linear grid starting at a finite $t_{\min}=\theta_{\mathrm E}/n$
        undersamples that cusp. This is a numerical-integration artifact of
        this exercise's grid, not a claim about the physical profile — but
        it is a fair warning that steep EPL slopes ($\gamma>2$, like the
        fine-product $\gamma=2.585$ artifact this campaign actually
        encountered) are cuspier at the center and genuinely harder to
        render and integrate accurately than shallow ones, independent of
        any noise or likelihood issue.
